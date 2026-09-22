"""Mode A: the click surface.

The estimator is covered elsewhere. What is checked here is the shell contract --
which exit code a verdict produces, what each output flag emits, and which flag
combinations are refused before any config.json is fetched.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from click.testing import CliRunner, Result
from fitcheck import cli
from fitcheck.cli import main

_MODEL = "meta-llama/Llama-3.1-8B"

_ENV = {"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"}

_FITTING = (
    "--qlora",
    "--lora-r", "64",
    "--lora-targets", "q,k,v,o",
    "--batch-size", "1",
    "--seq-len", "2048",
    "--flash-attn",
    "--gpu", "4090",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env=_ENV)


@pytest.fixture
def llama_on_hub(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)


@pytest.fixture
def no_repl(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Swap the REPL for a recorder: Mode B has its own test file."""
    calls: list[dict[str, Any]] = []

    def _fake_run_repl(console: object, training: object = None, gpu: object = None) -> int:
        calls.append({"training": training, "gpu": gpu})
        return 0

    monkeypatch.setattr(cli, "run_repl", _fake_run_repl)
    yield calls


def _flat(result: Result) -> str:
    return " ".join(result.output.split())




def test_a_fitting_config_exits_zero(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING])

    assert result.exit_code == 0
    assert "FITS" in _flat(result)


def test_a_config_that_does_not_fit_exits_one(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING, "--batch-size", "4"])

    assert result.exit_code == 1
    assert "DOES NOT FIT" in _flat(result)


def test_an_unknown_gpu_exits_two(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(main, [_MODEL, "--gpu", "rtx-9090"])

    assert result.exit_code == 2
    assert "Unknown GPU 'rtx-9090'" in _flat(result)


def test_an_unreadable_config_exits_two(
    runner: CliRunner, fake_config_download: Callable[..., None]
) -> None:
    fake_config_download({"hidden_size": 4096})

    result = runner.invoke(main, ["some/model"])

    assert result.exit_code == 2




def test_list_gpus_prints_the_database_and_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--list-gpus"])

    assert result.exit_code == 0
    output = _flat(result)
    assert "fitcheck GPU database" in output
    assert "4090" in output
    assert "Usable (MiB)" in output


def test_json_output_carries_every_component_and_the_verdict(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING, "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload["memory_mib"]) == {
        "weights", "lora", "optimizer", "gradients", "activations", "overhead", "total",
    }
    assert payload["verdict"]["fits"] is True
    assert payload["model"]["num_params"] == 8_030_261_248
    assert payload["gpu"]["usable_mib"] == 23_500
    assert payload["training"]["quantization"] == "nf4"
    assert payload["trainable_params"] == 54_525_952
    assert payload["activations_per_layer_mib"] > 0
    assert (
        payload["verdict"]["estimated_max_batch_size"]
        == payload["verdict"]["max_batch_size"]
    )
    assert payload["verdict"]["recommended_batch_size"] == 1
    assert payload["verdict"]["recommendation_basis"] == "point_estimate_margin"


def test_json_output_still_exits_one_when_it_does_not_fit(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING, "--batch-size", "4", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["verdict"]["fits"] is False


def test_verbose_adds_the_per_layer_detail_panel(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING, "--verbose"])

    output = _flat(result)
    assert "hidden_size" in output
    assert "A_layer, all saved tensors for one layer" in output
    assert "A_act charged" in output


def test_explain_names_the_largest_component(runner: CliRunner, llama_on_hub: None) -> None:
    """At bs=1 the NF4 base outweighs everything else."""
    result = runner.invoke(main, [_MODEL, *_FITTING, "--explain"])

    output = _flat(result)
    assert "Largest component: base model weights" in output
    assert "quantization scales" in output


def test_explain_blames_the_logits_once_the_batch_grows(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING, "--batch-size", "4", "--explain"])

    output = _flat(result)
    assert "Largest component: activations" in output
    assert "A_logits" in output


def test_no_color_still_renders_the_report(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(main, [_MODEL, *_FITTING, "--no-color"])

    assert result.exit_code == 0
    assert "TOTAL (predicted peak)" in _flat(result)


def test_version_flag_prints_the_package_name(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "fitcheck" in result.output


def test_package_version_falls_back_when_the_dist_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(name: str) -> str:
        raise cli.PackageNotFoundError(name)

    monkeypatch.setattr(cli, "version", _missing)

    assert cli._package_version() == "unknown"




def test_infer_prices_weights_plus_kv_cache(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(
        main, ["infer", _MODEL, "--quant", "nf4", "--seq-len", "2048", "--gpu", "4090"]
    )

    assert result.exit_code == 0
    output = _flat(result)
    assert "fitcheck infer" in output
    assert "KV cache" in output
    assert "TOTAL (resident)" in output


def test_infer_json_reports_the_cache_unit_costs(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main, ["infer", _MODEL, "--quant", "nf4", "--concurrent", "4", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["serving"]["num_concurrent"] == 4
    assert payload["kv_cache_mib_per_request"] > 0
    assert payload["kv_cache_mib_per_token"] > 0
    assert payload["verdict"]["max_concurrent"] >= 1


def test_infer_exits_one_when_the_cache_does_not_fit(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main, ["infer", _MODEL, "--seq-len", "16384", "--concurrent", "64", "--gpu", "t4"]
    )

    assert result.exit_code == 1


def test_infer_refuses_double_quant_outside_nf4(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, ["infer", _MODEL, "--quant", "int8", "--double-quant"])

    assert result.exit_code == 2
    assert "--double-quant" in _flat(result)




def test_advise_sweeps_and_exits_zero_when_something_fits(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main,
        [
            "advise", _MODEL, "--seq-lens", "1024", "--batch-sizes", "1",
            "--lora-ranks", "8", "--qlora", "--flash-attn", "--gpu", "a100-80",
        ],
    )

    assert result.exit_code == 0
    output = _flat(result)
    assert "fitcheck advise" in output
    assert "THE WALL" in output
    assert "PRICE OF EACH AXIS" in output


def test_advise_exits_one_when_nothing_fits(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(
        main,
        [
            "advise", _MODEL, "--seq-lens", "8192", "--batch-sizes", "8",
            "--lora-ranks", "64", "--gpu", "t4",
        ],
    )

    assert result.exit_code == 1
    assert "nothing here fits" in _flat(result)


def test_advise_json_carries_the_sweep_and_the_ceilings(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main,
        [
            "advise", _MODEL, "--seq-lens", "1024", "--batch-sizes", "1,2",
            "--lora-ranks", "8", "--qlora", "--flash-attn", "--gpu", "a100-80", "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sweep"] == {
        "batch_sizes": [1, 2], "seq_lens": [1024], "lora_ranks": [8]
    }
    assert payload["grid_size"] == 2
    assert payload["ceilings"]
    assert payload["prices"]
    assert payload["recommended"]["command"].startswith("fitcheck")


def test_advise_rejects_a_seq_len_past_the_declared_maximum(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main, ["advise", _MODEL, "--seq-lens", "1024,8192", "--max-seq-len", "4096"]
    )

    assert result.exit_code == 2
    assert "exceed --max-seq-len 4,096" in _flat(result)


def test_advise_rejects_a_non_numeric_sweep_value(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, ["advise", _MODEL, "--seq-lens", "1024,lots"])

    assert result.exit_code == 2
    assert "'lots' is not a whole number" in _flat(result)


def test_advise_rejects_a_non_positive_sweep_value(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(main, ["advise", _MODEL, "--seq-lens", "0"])

    assert result.exit_code == 2
    assert "not a positive value" in _flat(result)


def test_advise_rejects_an_empty_sweep_axis(runner: CliRunner, llama_on_hub: None) -> None:
    result = runner.invoke(main, ["advise", _MODEL, "--seq-lens", " , "])

    assert result.exit_code == 2
    assert "at least one value is required" in _flat(result)




def test_lora_targets_accepts_a_preset_and_a_bare_module_list() -> None:
    assert cli._parse_lora_targets("MINIMAL") == ["q_proj", "v_proj"]
    assert cli._parse_lora_targets("q, v_proj ,") == ["q_proj", "v_proj"]


def test_lora_targets_rejects_an_unknown_module(runner: CliRunner) -> None:
    result = runner.invoke(main, [_MODEL, "--lora-targets", "z"])

    assert result.exit_code == 2
    assert "unknown LoRA target 'z'" in _flat(result)


def test_lora_targets_rejects_a_duplicate(runner: CliRunner) -> None:
    result = runner.invoke(main, [_MODEL, "--lora-targets", "q,q_proj"])

    assert result.exit_code == 2
    assert "duplicate LoRA target 'q_proj'" in _flat(result)


def test_lora_targets_rejects_an_empty_list(runner: CliRunner) -> None:
    result = runner.invoke(main, [_MODEL, "--lora-targets", " , "])

    assert result.exit_code == 2
    assert "at least one target module is required" in _flat(result)


def test_no_lora_with_a_quantized_base_is_refused(runner: CliRunner) -> None:
    result = runner.invoke(main, [_MODEL, "--no-lora", "--quant", "nf4"])

    assert result.exit_code == 2
    assert "is not modelled by fitcheck" in _flat(result)


def test_no_lora_conflicts_with_an_explicit_adapter_flag(runner: CliRunner) -> None:
    result = runner.invoke(main, [_MODEL, "--no-lora", "--lora-r", "32"])

    assert result.exit_code == 2
    assert "--no-lora conflicts with --lora-r" in _flat(result)


def test_double_quant_under_quant_none_is_refused(runner: CliRunner) -> None:
    result = runner.invoke(main, [_MODEL, "--double-quant"])

    assert result.exit_code == 2
    assert "nothing to quantize under --quant none" in _flat(result)


def test_optimizer_dtype_only_applies_to_adamw(runner: CliRunner) -> None:
    result = runner.invoke(
        main, [_MODEL, "--optimizer", "sgd", "--optimizer-dtype", "bf16"]
    )

    assert result.exit_code == 2
    assert "applies only to --optimizer adamw" in _flat(result)


def test_an_explicit_flag_beats_the_qlora_shorthand(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main, [_MODEL, *_FITTING, "--quant", "none", "--precision", "fp16", "--json"]
    )

    payload = json.loads(result.stdout)
    assert payload["training"]["quantization"] == "none"
    assert payload["training"]["precision"] == "fp16"
    assert payload["training"]["grad_checkpoint"] is True


def test_a_vram_override_prices_a_card_outside_the_database(
    runner: CliRunner, llama_on_hub: None
) -> None:
    result = runner.invoke(
        main, [_MODEL, *_FITTING[:-2], "--gpu", "mycard", "--vram-mib", "40000", "--json"]
    )

    payload = json.loads(result.stdout)
    assert payload["gpu"]["name"] == "mycard"
    assert payload["gpu"]["vram_mib"] == 40_000




def test_no_model_id_enters_the_repl_and_exits_zero(
    runner: CliRunner, no_repl: list[dict[str, Any]]
) -> None:
    result = runner.invoke(main, ["--batch-size", "4", "--gpu", "t4"])

    assert result.exit_code == 0
    assert no_repl[0]["training"].batch_size == 4
    assert no_repl[0]["gpu"].name == "Tesla T4"


def test_the_repl_is_seeded_with_no_gpu_when_none_was_named(
    runner: CliRunner, no_repl: list[dict[str, Any]]
) -> None:
    result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert no_repl[0]["gpu"] is None


@pytest.mark.parametrize("flag", ["--json", "--verbose", "--explain"])
def test_a_formatting_flag_without_a_model_is_refused(
    runner: CliRunner, no_repl: list[dict[str, Any]], flag: str
) -> None:
    result = runner.invoke(main, [flag])

    assert result.exit_code == 2
    assert "there is nothing to format" in _flat(result)
    assert not no_repl
