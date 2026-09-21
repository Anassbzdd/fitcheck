"""Mode B: the interactive dispatch loop.

Each test drives `_dispatch` against a session built by hand, so a command is
exercised without a terminal. The loop itself is covered once, at the bottom, by
feeding a script to `run_repl` through a fake stdin.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from typing import Any

import pytest
from fitcheck import repl
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.estimator import ServingConfig, TrainingConfig
from fitcheck.gpu_db import GpuSpec, get_gpu
from rich.console import Console

_MODEL = "meta-llama/Llama-3.1-8B"


@pytest.fixture
def llama_on_hub(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)


@pytest.fixture
def llama_model(llama_on_hub: None) -> ModelConfig:
    return fetch_model_config(_MODEL)


class _Repl:
    """A session wired to a string buffer, with one report per `run`."""

    def __init__(
        self,
        model: ModelConfig | None = None,
        gpu: GpuSpec | None = None,
        training: TrainingConfig | None = None,
    ) -> None:
        self.buffer = io.StringIO()
        console = Console(file=self.buffer, width=200, no_color=True, legacy_windows=False)
        self.session = repl._Session(
            console=console,
            glyphs=repl._ASCII_GLYPHS,
            ascii_only=True,
            model=model,
            model_id=_MODEL if model is not None else None,
            gpu=gpu,
            training=training if training is not None else TrainingConfig(),
        )

    def run(self, line: str) -> str:
        before = self.buffer.tell()
        repl._dispatch(self.session, line)
        self.buffer.seek(before)
        return " ".join(self.buffer.read().split())

    def fails(self, line: str) -> str:
        with pytest.raises(repl._ReplError) as caught:
            self.run(line)
        return str(caught.value)


@pytest.fixture
def loaded(llama_model: ModelConfig) -> _Repl:
    """A session that already has a model and a card: the common starting point."""
    return _Repl(
        model=llama_model,
        gpu=get_gpu("4090"),
        training=TrainingConfig(
            precision="bf16",
            quantization="nf4",
            optimizer="adamw",
            optimizer_dtype="fp32",
            batch_size=1,
            seq_len=2048,
            lora_rank=64,
            lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
            grad_checkpoint=True,
            flash_attn=True,
        ),
    )


# ----------------------------------------------------------------------- model / gpu


def test_model_loads_a_config_and_reports_the_geometry(llama_on_hub: None) -> None:
    session = _Repl()

    output = session.run(f"model {_MODEL}")

    assert "Loaded Llama-3.1-8B" in output
    assert "32 heads, GQA 8 KV heads" in output
    assert session.session.model_id == _MODEL


def test_model_needs_exactly_one_argument() -> None:
    assert "Usage: model <huggingface_id>" in _Repl().fails("model")


def test_model_reports_a_config_it_cannot_read(
    fake_config_download: Callable[..., None],
) -> None:
    """A config.json missing half its keys is a message, not a traceback."""
    fake_config_download({"hidden_size": 4096})

    message = _Repl().fails("model some/model")

    assert "some/model" in message


def test_gpu_sets_the_card_and_invalidates_the_last_report(loaded: _Repl) -> None:
    loaded.run("memory")
    assert loaded.session.last is not None

    output = loaded.run("gpu a100-80")

    assert "Target GPU set to" in output
    assert "A100" in output
    assert loaded.session.last is None


def test_gpu_accepts_a_vram_override(loaded: _Repl) -> None:
    output = loaded.run("gpu mycard --vram-mib 24000")

    assert "mycard" in output
    assert loaded.session.gpu is not None
    assert loaded.session.gpu.vram_mib == 24_000


def test_gpu_rejects_a_vram_override_without_a_number(loaded: _Repl) -> None:
    assert "--vram-mib needs a positive integer" in loaded.fails("gpu mycard --vram-mib")


def test_gpu_rejects_too_many_names(loaded: _Repl) -> None:
    assert "Usage: gpu <name>" in loaded.fails("gpu 4090 t4")


def test_gpu_rejects_no_arguments_at_all(loaded: _Repl) -> None:
    assert "Usage: gpu <name>" in loaded.fails("gpu")


def test_an_unknown_card_suggests_the_vram_override(loaded: _Repl) -> None:
    message = loaded.fails("gpu rtx-9090")

    assert "Unknown GPU 'rtx-9090'" in message
    assert "--vram-mib 24000" in message


# ----------------------------------------------------------------------- memory


def test_memory_estimates_and_remembers_the_flags(loaded: _Repl) -> None:
    output = loaded.run("memory --batch-size 4")

    assert "TOTAL (predicted peak)" in output
    assert "30,607" in output
    assert "[X] DOES NOT FIT" in output
    assert loaded.session.training.batch_size == 4


def test_memory_flags_stick_across_lines(loaded: _Repl) -> None:
    loaded.run("memory --batch-size 4 --seq-len 1024")
    loaded.run("memory --batch-size 2")

    assert loaded.session.training.seq_len == 1024
    assert loaded.session.training.batch_size == 2


def test_memory_json_prints_the_machine_payload(loaded: _Repl) -> None:
    before = loaded.buffer.tell()
    repl._dispatch(loaded.session, "memory --json")
    loaded.buffer.seek(before)
    payload = json.loads(loaded.buffer.read())

    assert payload["verdict"]["fits"] is True
    assert payload["training"]["quantization"] == "nf4"


def test_memory_verbose_and_explain_add_their_panels(loaded: _Repl) -> None:
    output = loaded.run("memory --verbose --explain")

    assert "hidden_size" in output
    assert "A_act charged" in output
    assert "Largest component:" in output


def test_a_sticky_flag_is_cleared_by_its_off_switch(loaded: _Repl) -> None:
    loaded.run("memory --no-flash-attn")

    assert loaded.session.training.flash_attn is False


def test_a_flag_and_its_off_switch_on_one_line_contradict(loaded: _Repl) -> None:
    message = loaded.fails("memory --flash-attn --no-flash-attn")

    assert "contradict each other" in message


def test_the_qlora_shorthand_sets_three_dials_at_once(llama_model: ModelConfig) -> None:
    session = _Repl(model=llama_model, gpu=get_gpu("4090"))

    session.run("memory --qlora")

    assert session.session.training.quantization == "nf4"
    assert session.session.training.precision == "bf16"
    assert session.session.training.grad_checkpoint is True


def test_no_lora_switches_to_full_fine_tuning(llama_model: ModelConfig) -> None:
    session = _Repl(model=llama_model, gpu=get_gpu("a100-80"))

    output = session.run("memory --no-lora")

    assert session.session.training.lora_rank is None
    assert "full fine-tune" in output


def test_memory_without_a_model_says_what_to_run_first() -> None:
    message = _Repl(gpu=get_gpu("4090")).fails("memory")

    assert "No model loaded" in message


def test_memory_without_a_model_or_a_gpu_names_both() -> None:
    message = _Repl().fails("memory")

    assert "`model <id>` and `gpu <name>` first" in message


def test_memory_without_a_gpu_says_so(llama_model: ModelConfig) -> None:
    message = _Repl(model=llama_model).fails("memory")

    assert "No GPU set" in message


def test_memory_takes_a_one_shot_gpu_override(loaded: _Repl) -> None:
    loaded.run("memory --gpu t4")

    assert loaded.session.gpu == get_gpu("4090")
    assert loaded.session.last is not None
    assert loaded.session.last.gpu == get_gpu("t4")


def test_memory_refuses_a_quantized_full_fine_tune(loaded: _Repl) -> None:
    with pytest.raises(Exception, match="not modelled by fitcheck"):
        loaded.run("memory --no-lora --quant nf4")


def test_double_quant_is_dropped_when_the_base_stops_being_nf4(loaded: _Repl) -> None:
    loaded.run("memory --double-quant")
    assert loaded.session.training.double_quant is True

    loaded.run("memory --quant none")
    assert loaded.session.training.double_quant is False


# ----------------------------------------------------------------------- infer


def test_infer_prices_weights_plus_the_kv_cache(loaded: _Repl) -> None:
    output = loaded.run("infer --quant nf4 --concurrent 4")

    assert "KV cache (4 x 2,048 tokens, fp16)" in output
    assert "TOTAL (resident)" in output
    assert loaded.session.serving.num_concurrent == 4


def test_infer_keeps_its_own_sticky_set(loaded: _Repl) -> None:
    loaded.run("memory --seq-len 4096")
    loaded.run("infer --seq-len 8192")

    assert loaded.session.training.seq_len == 4096
    assert loaded.session.serving.seq_len == 8192
    assert loaded.session.serving.precision == "fp16"


def test_infer_json_prints_the_machine_payload(loaded: _Repl) -> None:
    before = loaded.buffer.tell()
    repl._dispatch(loaded.session, "infer --json")
    loaded.buffer.seek(before)
    payload = json.loads(loaded.buffer.read())

    assert payload["serving"]["num_concurrent"] == 1
    assert payload["kv_cache_mib_per_token"] > 0


def test_infer_clears_double_quant_with_its_off_switch(loaded: _Repl) -> None:
    loaded.run("infer --quant nf4 --double-quant")
    assert loaded.session.serving.double_quant is True

    loaded.run("infer --no-double-quant")
    assert loaded.session.serving.double_quant is False


def test_infer_refuses_a_contradictory_double_quant_pair(loaded: _Repl) -> None:
    message = loaded.fails("infer --double-quant --no-double-quant")

    assert "contradict each other" in message


def test_infer_refuses_double_quant_outside_nf4(loaded: _Repl) -> None:
    with pytest.raises(Exception, match="--double-quant"):
        loaded.run("infer --quant int8 --double-quant")


# ----------------------------------------------------------------------- advise


def test_advise_sweeps_and_remembers_the_bounds(loaded: _Repl) -> None:
    output = loaded.run("advise --seq-lens 1024 --batch-sizes 1 --lora-ranks 8")

    assert "THE WALL" in output
    assert "PRICE OF EACH AXIS" in output
    assert loaded.session.sweep is not None
    assert loaded.session.sweep.seq_lens == (1024,)


def test_a_bare_advise_reuses_the_stored_sweep(loaded: _Repl) -> None:
    loaded.run("advise --seq-lens 1024 --batch-sizes 1 --lora-ranks 8")

    output = loaded.run("advise")

    assert "THE WALL" in output


def test_advise_refuses_to_guess_a_context_length(loaded: _Repl) -> None:
    message = loaded.fails("advise")

    assert "there is no honest default" in message


def test_advise_rejects_a_sweep_past_the_declared_maximum(loaded: _Repl) -> None:
    with pytest.raises(Exception, match="exceed --max-seq-len"):
        loaded.run("advise --seq-lens 8192 --max-seq-len 4096")


def test_advise_has_nothing_to_sweep_under_a_full_fine_tune(loaded: _Repl) -> None:
    loaded.run("memory --no-lora --quant none")

    message = loaded.fails("advise --seq-lens 1024")

    assert "leaves it nothing to sweep" in message


def test_advise_json_prints_the_machine_payload(loaded: _Repl) -> None:
    before = loaded.buffer.tell()
    repl._dispatch(
        loaded.session, "advise --seq-lens 1024 --batch-sizes 1 --lora-ranks 8 --json"
    )
    loaded.buffer.seek(before)
    payload = json.loads(loaded.buffer.read())

    assert payload["sweep"]["seq_lens"] == [1024]
    assert payload["grid_size"] == 1


def test_advise_shares_the_training_flags_with_memory(loaded: _Repl) -> None:
    loaded.run("advise --seq-lens 1024 --batch-sizes 1 --lora-ranks 8 --no-flash-attn")

    assert loaded.session.training.flash_attn is False
    assert loaded.session.last is None


# ----------------------------------------------------------------------- explain


def test_explain_reuses_the_last_report(loaded: _Repl) -> None:
    loaded.run("memory --batch-size 4")

    output = loaded.run("explain")

    assert "Largest component: activations" in output


def test_explain_computes_one_when_nothing_was_estimated_yet(loaded: _Repl) -> None:
    output = loaded.run("explain")

    assert "Largest component:" in output


def test_explain_takes_no_arguments(loaded: _Repl) -> None:
    assert "explain takes no arguments" in loaded.fails("explain --batch-size 4")


# ----------------------------------------------------------------------- compare


def test_compare_prices_the_config_on_every_named_card(loaded: _Repl) -> None:
    output = loaded.run("compare t4 a100-80")

    assert "Max bs @ 2,048" in output
    assert "Tesla T4" in output
    assert "Peak moves with the card" in output


def test_compare_ignores_a_repeated_card(loaded: _Repl) -> None:
    output = loaded.run("compare 4090 4090")

    assert output.count("RTX 4090") == 1


def test_compare_switches_to_serving_with_infer(loaded: _Repl) -> None:
    output = loaded.run("compare a100-80 --infer")

    assert "compare infer" in output
    assert "Max concurrent @ 2,048" in output


def test_compare_needs_at_least_one_card(loaded: _Repl) -> None:
    assert "Usage: compare <gpu>" in loaded.fails("compare --infer")


# ----------------------------------------------------------------------- show / state


def test_show_reports_an_empty_session() -> None:
    output = _Repl().run("show")

    assert "not loaded. Run `model <id>`" in output
    assert "not set. Run `gpu <name>`" in output
    assert "not set. Run `advise --seq-lens 1024,2048`" in output


def test_show_reports_the_loaded_state_and_the_last_estimates(loaded: _Repl) -> None:
    loaded.run("memory")
    loaded.run("infer")
    loaded.run("advise --seq-lens 1024 --batch-sizes 1 --lora-ranks 8 --max-seq-len 4096")

    output = loaded.run("show")

    assert "Llama-3.1-8B" in output
    assert "QLoRA r=64 [q,k,v,o]" in output
    assert "Last memory" in output
    assert "Last infer" in output
    assert "max seq 4,096" in output


def test_show_takes_no_arguments(loaded: _Repl) -> None:
    assert "show takes no arguments" in loaded.fails("show now")


def test_the_state_alias_reaches_show(loaded: _Repl) -> None:
    assert "session" in loaded.run("state")


# ----------------------------------------------------------------------- gpus / reset


def test_gpus_prints_the_database(loaded: _Repl) -> None:
    output = loaded.run("gpus")

    assert "fitcheck GPU database" in output
    assert "Usable %" in output


def test_gpus_takes_no_arguments(loaded: _Repl) -> None:
    assert "gpus takes no arguments" in loaded.fails("gpus all")


def test_reset_restores_the_defaults_but_keeps_model_and_gpu(loaded: _Repl) -> None:
    loaded.run("memory --batch-size 8")
    loaded.run("advise --seq-lens 1024 --batch-sizes 1 --lora-ranks 8")

    output = loaded.run("reset")

    assert "back to defaults" in output
    assert loaded.session.training == TrainingConfig()
    assert loaded.session.serving == ServingConfig()
    assert loaded.session.sweep is None
    assert loaded.session.model is not None
    assert loaded.session.gpu == get_gpu("4090")


def test_reset_takes_no_arguments(loaded: _Repl) -> None:
    assert "reset takes no arguments" in loaded.fails("reset all")


# ----------------------------------------------------------------------- help / dispatch


def test_help_lists_every_command(loaded: _Repl) -> None:
    output = loaded.run("help")

    assert "Flags persist." in output
    for command in ("model <id>", "optimize", "compare <gpu> ... [--infer]", "exit / quit"):
        assert command in output


def test_the_question_mark_alias_reaches_help(loaded: _Repl) -> None:
    assert "Flags persist." in loaded.run("?")


def test_an_empty_line_does_nothing(loaded: _Repl) -> None:
    assert loaded.run("   ") == ""


def test_an_unquoted_quote_is_reported_not_raised(loaded: _Repl) -> None:
    assert "Could not parse that line" in loaded.fails("model 'unterminated")


def test_an_unknown_command_suggests_the_closest_one(loaded: _Repl) -> None:
    message = loaded.fails("memry")

    assert "Unknown command 'memry'" in message
    assert "Did you mean `memory`?" in message


def test_an_unknown_command_with_no_near_match_just_points_at_help(
    loaded: _Repl,
) -> None:
    message = loaded.fails("zzzzzzzz")

    assert "Did you mean" not in message
    assert "Type `help` for the list" in message


def test_a_bare_model_id_is_read_as_a_forgotten_model_command(loaded: _Repl) -> None:
    message = loaded.fails(_MODEL)

    assert f"Did you mean `model {_MODEL}`?" in message


def test_commands_are_matched_case_insensitively(loaded: _Repl) -> None:
    assert "fitcheck GPU database" in loaded.run("GPUS")


def test_exit_raises_the_sentinel(loaded: _Repl) -> None:
    with pytest.raises(repl._ExitRepl):
        loaded.run("exit")


# ----------------------------------------------------------------------- the loop


def _drive(monkeypatch: pytest.MonkeyPatch, script: str, **kwargs: Any) -> tuple[int, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(script))
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, no_color=True, legacy_windows=False)

    code = repl.run_repl(console, **kwargs)
    return code, " ".join(buffer.getvalue().split())


def test_the_repl_greets_runs_a_script_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, llama_on_hub: None
) -> None:
    code, output = _drive(
        monkeypatch,
        f"gpu 4090\nmodel {_MODEL}\nmemory --qlora --lora-r 64 --flash-attn\nexit\n",
    )

    assert code == 0
    assert "fitcheck interactive." in output
    assert "Loaded Llama-3.1-8B" in output
    assert "TOTAL (predicted peak)" in output
    assert "Goodbye!" in output


def test_the_repl_echoes_the_flags_it_was_seeded_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = TrainingConfig(batch_size=4, seq_len=4096)

    code, output = _drive(monkeypatch, "exit\n", training=seeded, gpu=get_gpu("t4"))

    assert code == 0
    assert "Target GPU set to Tesla T4" in output
    assert "Flags carried in from the command line" in output
    assert "bs 4" in output


def test_the_repl_prints_an_error_and_keeps_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, output = _drive(monkeypatch, "memry\ngpus\nexit\n")

    assert code == 0
    assert "error" in output
    assert "Did you mean `memory`?" in output
    assert "fitcheck GPU database" in output


def test_the_repl_reports_a_bad_flag_without_leaving(
    monkeypatch: pytest.MonkeyPatch, llama_on_hub: None
) -> None:
    code, output = _drive(monkeypatch, f"gpu 4090\nmodel {_MODEL}\nmemory --nope\nexit\n")

    assert code == 0
    assert "error" in output
    assert "Goodbye!" in output


def test_the_repl_treats_end_of_input_as_a_goodbye(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, output = _drive(monkeypatch, "gpus\n")

    assert code == 0
    assert "Goodbye!" in output


def test_memory_help_is_printed_without_ending_the_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """click prints --help itself, and swallows the Exit it raises afterwards."""
    code, output = _drive(monkeypatch, "memory --help\nexit\n")

    assert code == 0
    assert "Flags are the CLI's and they persist" in capsys.readouterr().out
    assert "Goodbye!" in output


def test_an_unexpected_error_is_caught_and_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(session: repl._Session, args: list[str]) -> None:
        raise RuntimeError("the estimator fell over")

    monkeypatch.setitem(repl._COMMANDS, "gpus", _boom)

    code, output = _drive(monkeypatch, "gpus\nexit\n")

    assert code == 0
    assert "RuntimeError: the estimator fell over" in output
