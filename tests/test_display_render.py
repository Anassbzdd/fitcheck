"""What the rich renderables actually say.

Every panel is rendered through a console of fixed width, so an assertion is about
the words on the screen and not about this terminal's size. The estimator is not
under test here -- several reports are hand-built so that a branch can be reached
without hunting for a config that happens to trigger it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from fitcheck.advisor import SweepSpec, advise
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.display import (
    _activation_label,
    _adapter_label,
    _attention_label,
    _batch_label,
    _hint_savings_mib,
    _lora_label,
    _optimizer_label,
    _params_label,
    _weights_label,
    render_advisor_report,
    render_explanation,
    render_gpu_table,
    render_inference_report,
    render_report,
    render_verbose_detail,
    use_ascii_glyphs,
)
from fitcheck.estimator import (
    InferenceReport,
    MemoryReport,
    ServingConfig,
    TrainingConfig,
    estimate,
    estimate_inference,
)
from fitcheck.gpu_db import GpuSpec, get_gpu
from rich.console import Console


def _text(renderable: object, width: int = 200) -> str:
    console = Console(width=width, color_system=None, legacy_windows=False)
    with console.capture() as capture:
        console.print(renderable)
    return " ".join(capture.get().split())


@pytest.fixture
def llama_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(llama_31_8b_config)
    return fetch_model_config("meta-llama/Llama-3.1-8B")


@pytest.fixture
def small_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    mha_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(mha_config)
    return fetch_model_config("openai-community/gpt2")


@pytest.fixture
def mqa_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    mqa_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(mqa_config)
    return fetch_model_config("tiny/mqa")


@pytest.fixture
def qlora_training() -> TrainingConfig:
    return TrainingConfig(
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
    )


def _report(**overrides: Any) -> MemoryReport:
    base: dict[str, Any] = {
        "weight_mib": 1_000.0,
        "lora_mib": 10.0,
        "optimizer_mib": 20.0,
        "gradient_mib": 10.0,
        "activation_mib": 400.0,
        "overhead_mib": 560.0,
        "total_mib": 2_000.0,
        "gpu_capacity_mib": 4_000.0,
        "headroom_mib": 2_000.0,
        "fits": True,
        "max_batch_size": 4,
        "effective_batch_size": 1,
        "savings_hints": [],
        "warnings": (),
    }
    base.update(overrides)
    return MemoryReport(**base)


# ------------------------------------------------------------------- header labels


def test_attention_label_names_each_family(
    llama_model: ModelConfig, small_model: ModelConfig, mqa_model: ModelConfig
) -> None:
    assert _attention_label(small_model) == "12 heads (MHA)"
    assert _attention_label(mqa_model) == "16 heads (MQA)"
    assert _attention_label(llama_model) == "32 heads, GQA 8 KV heads"


def test_params_label_switches_unit_below_a_billion(
    llama_model: ModelConfig, small_model: ModelConfig
) -> None:
    assert _params_label(llama_model.num_params) == "8.03B params"
    assert _params_label(small_model.num_params).endswith("M params")


def test_adapter_and_component_labels_follow_the_config(
    qlora_training: TrainingConfig,
) -> None:
    lora = replace(qlora_training, quantization="none")
    full = replace(qlora_training, lora_rank=None)

    assert _adapter_label(qlora_training).startswith("QLoRA r=64 [q,k,v,o]")
    assert _adapter_label(lora).startswith("LoRA r=64")
    assert _adapter_label(full) == "full fine-tune"
    assert _lora_label(full) == "LoRA adapter (none, full fine-tune)"
    assert _weights_label(lora) == "Base model weights (bf16)"
    assert _weights_label(replace(qlora_training, double_quant=True)) == (
        "Base model weights (NF4 + double quant)"
    )


def test_labels_degrade_gracefully_without_a_training_config() -> None:
    assert _weights_label(None) == "Base model weights"
    assert _lora_label(None) == "LoRA adapter (trainable)"
    assert _activation_label(None) == "Activations"


def test_activation_and_batch_and_optimizer_labels(
    qlora_training: TrainingConfig,
) -> None:
    plain = replace(qlora_training, grad_checkpoint=False, flash_attn=False)

    assert _activation_label(plain) == "Activations (no ckpt, no flash)"
    assert _batch_label(replace(qlora_training, batch_size=2, grad_accum_steps=8)) == (
        "bs 2 x 8 accum = 16 effective"
    )
    assert _optimizer_label(qlora_training) == "adamw (fp32 states)"
    assert _optimizer_label(replace(qlora_training, optimizer="sgd")) == "sgd"


# ------------------------------------------------------------------- render_report


def test_report_renders_without_a_training_config(
    llama_model: ModelConfig,
) -> None:
    output = _text(render_report(_report(), llama_model, get_gpu("4090")))

    assert "Base model weights" in output
    assert "Max micro-batch size at this sequence length: 4." in output
    # No Config row when nothing was passed.
    assert "Config" not in output


def test_report_flags_a_tight_fit(llama_model: ModelConfig) -> None:
    tight = _report(total_mib=3_800.0, headroom_mib=200.0)
    output = _text(render_report(tight, llama_model, get_gpu("4090"), ascii_only=True))

    assert "FITS, BARELY" in output
    assert "[!]" in output


def test_report_suggests_dropping_the_batch_when_it_does_not_fit(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, batch_size=8, grad_accum_steps=2)
    over = _report(
        total_mib=9_000.0, headroom_mib=-5_000.0, fits=False,
        max_batch_size=2, effective_batch_size=16,
    )
    output = _text(render_report(over, llama_model, get_gpu("4090"), training))

    assert "DOES NOT FIT" in output
    assert "Drop batch_size to 2 to fit" in output
    assert "keep the effective batch at 16" in output


def test_report_says_when_the_batch_is_already_at_the_ceiling(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, batch_size=4)
    output = _text(
        render_report(_report(max_batch_size=4), llama_model, get_gpu("4090"), training)
    )

    assert "batch_size 4 is already the maximum" in output


def test_report_rescues_a_config_that_does_not_fit_at_all(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    over = _report(fits=False, headroom_mib=-1.0, max_batch_size=0)
    output = _text(render_report(over, llama_model, get_gpu("4090"), qlora_training))

    assert "Even batch_size=1 does not fit" in output


def test_report_prints_warnings_and_the_best_savings_hint(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    report = _report(
        savings_hints=[
            "--flash-attn: saves 12 MiB",
            "--quant nf4 -> saves 2,400 MiB",
        ],
        warnings=("the MLP shape is assumed",),
    )
    output = _text(
        render_report(report, llama_model, get_gpu("4090"), qlora_training, ascii_only=True)
    )

    assert "saves 2,400 MiB" in output
    assert "the MLP shape is assumed" in output


def test_a_hint_with_no_parsable_saving_is_not_promoted(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    """_SAVINGS_PATTERN misses: the hint grid must fall back, not crash."""
    report = _report(savings_hints=["--flash-attn: saves nothing at this shape"])
    output = _text(render_report(report, llama_model, get_gpu("4090"), qlora_training))

    assert "saves nothing at this shape" not in output
    assert _hint_savings_mib("--flash-attn: costs 5 MiB") == 0.0
    assert _hint_savings_mib("saves 1,024 MiB") == 1_024.0


def test_a_zero_capacity_card_does_not_divide_by_zero(llama_model: ModelConfig) -> None:
    empty = GpuSpec("Empty", 0, 0)
    report = _report(gpu_capacity_mib=0.0, headroom_mib=0.0, fits=False)

    assert "DOES NOT FIT" in _text(render_report(report, llama_model, empty))


# ------------------------------------------------------------------- explanation


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"weight_mib": 9_000.0}, "the frozen base model at NF4"),
        ({"lora_mib": 9_000.0}, "rank 64 across 4 target modules"),
        ({"optimizer_mib": 9_000.0}, "AdamW keeps momentum and variance in fp32"),
        ({"gradient_mib": 9_000.0}, "one .grad tensor per trainable parameter"),
        ({"overhead_mib": 9_000.0}, "CUDA context"),
    ],
)
def test_explanation_explains_whichever_component_wins(
    llama_model: ModelConfig,
    qlora_training: TrainingConfig,
    overrides: dict[str, Any],
    expected: str,
) -> None:
    report = _report(**overrides)
    output = _text(
        render_explanation(report, llama_model, qlora_training, get_gpu("4090"))
    )

    assert expected in output


def test_explanation_offers_nf4_when_the_base_is_unquantized(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, quantization="none")
    output = _text(
        render_explanation(_report(weight_mib=9_000.0), llama_model, training)
    )

    assert "Quantizing it with --quant nf4 cuts this to roughly a quarter" in output


def test_explanation_blames_the_checkpoint_store_when_the_layer_wins(
    small_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    """A 50k vocab and a long sequence put A_layer above A_logits."""
    training = replace(
        qlora_training, quantization="none", seq_len=8192, flash_attn=False
    )
    output = _text(
        render_explanation(_report(activation_mib=9_000.0), small_model, training)
    )

    assert "checkpoint store plus one recomputed layer" in output
    assert "Lowering --batch-size" in output


def test_explanation_points_at_grad_checkpointing_when_it_is_off(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, grad_checkpoint=False)
    output = _text(
        render_explanation(_report(activation_mib=9_000.0), llama_model, training)
    )

    assert "all 32 layers keep their full set of saved tensors" in output
    assert "--grad-checkpoint trades compute" in output


def test_explanation_prices_a_non_adamw_optimizer(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, optimizer="adam8bit")
    output = _text(
        render_explanation(_report(optimizer_mib=9_000.0), llama_model, training)
    )

    assert "adam8bit states over the trainable parameters" in output


def test_explanation_falls_back_to_the_default_overhead_profile(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    """A hump-form profile has no meaning without the checkpointed max()."""
    training = replace(qlora_training, grad_checkpoint=False)
    output = _text(
        render_explanation(_report(overhead_mib=9_000.0), llama_model, training, get_gpu("t4"))
    )

    assert "CUDA context floor" in output
    assert "of weights and activations" in output


def test_explanation_carries_the_warnings_through(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    report = _report(warnings=("int8 activations are modelled on one measured row",))
    output = _text(render_explanation(report, llama_model, qlora_training))

    assert "int8 activations are modelled on one measured row" in output


# ------------------------------------------------------------------- verbose detail


def test_verbose_detail_shows_the_checkpointed_arithmetic(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    output = _text(render_verbose_detail(llama_model, qlora_training))

    assert "Checkpoint store" in output
    assert "avoided by Flash Attention" in output
    assert "never overlap" in output
    assert "A_act charged" in output


def test_verbose_detail_shows_the_retained_form_without_checkpointing(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, grad_checkpoint=False, flash_attn=False)
    output = _text(render_verbose_detail(llama_model, training))

    assert "~3 copies retained per layer, no Flash Attention" in output
    assert "x 32 layers, every layer's tensors kept" in output
    assert "+ A_logits" in output


def test_verbose_detail_in_ascii_spells_the_symbols_out(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    unquantized = replace(qlora_training, quantization="none", flash_attn=False)
    output = _text(render_verbose_detail(llama_model, unquantized, ascii_only=True))

    assert "Checkpoint store (L x gamma*b*s*h)" in output
    assert "included, no Flash Attention" in output
    assert "A_logits, 3.5 fp32 copies of (b, s, V)" in output
    assert "activation dtype" in output


# ------------------------------------------------------------------- inference


def test_inference_report_names_the_cache_and_the_weights(
    llama_model: ModelConfig,
) -> None:
    serving = ServingConfig(quantization="nf4", seq_len=2048, num_concurrent=2)
    report = estimate_inference(llama_model, serving, get_gpu("a100-80"))
    output = _text(render_inference_report(report, llama_model, get_gpu("a100-80"), serving))

    assert "2 concurrent requests" in output
    assert "NF4 weights" in output
    assert "KV cache (2 x 2,048 tokens, fp16)" in output
    assert "MiB per token" in output


def test_inference_report_labels_an_unquantized_base(llama_model: ModelConfig) -> None:
    serving = ServingConfig(num_concurrent=1)
    report = estimate_inference(llama_model, serving, get_gpu("a100-80"))
    output = _text(render_inference_report(report, llama_model, get_gpu("a100-80"), serving))

    assert "1 request" in output
    assert "unquantized weights" in output
    assert "Base model weights (fp16)" in output


def _inference(**overrides: Any) -> InferenceReport:
    base: dict[str, Any] = {
        "weight_mib": 1_000.0,
        "kv_cache_mib": 500.0,
        "overhead_mib": 500.0,
        "total_mib": 2_000.0,
        "kv_mib_per_request": 500.0,
        "kv_mib_per_token": 0.25,
        "gpu_capacity_mib": 4_000.0,
        "headroom_mib": 2_000.0,
        "fits": True,
        "max_concurrent": 4,
        "warnings": (),
    }
    base.update(overrides)
    return InferenceReport(**base)


@pytest.mark.parametrize(
    ("max_concurrent", "asked", "expected"),
    [
        (0, 1, "Not even one request fits"),
        (4, 1, "Room for 4 concurrent requests"),
        (2, 8, "Drop to 2 concurrent requests"),
        (4, 4, "4 concurrent requests is already the maximum"),
    ],
)
def test_inference_concurrency_suggestions(
    llama_model: ModelConfig, max_concurrent: int, asked: int, expected: str
) -> None:
    serving = ServingConfig(num_concurrent=asked)
    report = _inference(max_concurrent=max_concurrent, fits=max_concurrent > 0)
    output = _text(
        render_inference_report(report, llama_model, get_gpu("4090"), serving)
    )

    assert expected in output


def test_inference_report_prints_its_warnings(llama_model: ModelConfig) -> None:
    report = _inference(warnings=("a paged engine allocates less",))
    output = _text(
        render_inference_report(
            report, llama_model, get_gpu("4090"), ServingConfig(), ascii_only=True
        )
    )

    assert "a paged engine allocates less" in output


# ------------------------------------------------------------------- gpu table


def test_gpu_table_lists_every_card_with_its_usable_share() -> None:
    output = _text(render_gpu_table())

    assert "fitcheck GPU database" in output
    assert "4090" in output
    assert "t4" in output
    assert "Usable %" in output


# ------------------------------------------------------------------- advisor


@pytest.fixture
def advisor_output(llama_model: ModelConfig, qlora_training: TrainingConfig) -> str:
    sweep = SweepSpec(batch_sizes=(1, 2), seq_lens=(1024, 2048), lora_ranks=(8, 64))
    report = advise(
        llama_model,
        qlora_training,
        get_gpu("a100-80"),
        sweep,
        model_id="meta-llama/Llama-3.1-8B",
    )
    return _text(render_advisor_report(report, llama_model))


def test_advisor_report_draws_frontier_wall_and_prices(advisor_output: str) -> None:
    assert "FRONTIER -- the edge of what fits" in advisor_output
    assert "batch x seq splits at the same cost" in advisor_output
    assert "Run it:" in advisor_output
    assert "THE WALL -- exact ceilings, by bisection" in advisor_output
    assert "Nearest 2^n" in advisor_output
    assert "PRICE OF EACH AXIS -- from the same anchor" in advisor_output


def test_advisor_report_echoes_the_axes_it_held_and_swept(advisor_output: str) -> None:
    assert "NF4 base" in advisor_output
    assert "grad ckpt" in advisor_output
    assert "flash attn" in advisor_output
    assert "batch 1, 2" in advisor_output
    assert "rank 8, 64" in advisor_output


def test_advisor_report_warns_that_tokens_per_step_is_not_a_speed(
    advisor_output: str,
) -> None:
    assert "not a speed" in advisor_output
    assert "fitcheck has no throughput model" in advisor_output


def test_advisor_report_says_so_when_nothing_fits(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    sweep = SweepSpec(batch_sizes=(16,), seq_lens=(8192,), lora_ranks=(64,))
    report = advise(llama_model, qlora_training, get_gpu("t4"), sweep)
    output = _text(render_advisor_report(report, llama_model, ascii_only=True))

    assert "nothing here fits" in output
    assert "Nothing fits: add --grad-checkpoint" in output
    assert "FRONTIER" not in output


def test_advisor_report_surfaces_the_sweeps_own_warnings(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, quantization="int8")
    sweep = SweepSpec(batch_sizes=(1,), seq_lens=(1024,), lora_ranks=(8,))
    report = advise(llama_model, training, get_gpu("a100-80"), sweep)
    output = _text(render_advisor_report(report, llama_model))

    assert "int8" in output
    assert "unquantized base" not in output


# ------------------------------------------------------------------- console


def test_ascii_glyphs_are_chosen_by_the_console_encoding() -> None:
    class _Latin(Console):
        @property
        def encoding(self) -> str:
            return "cp1252"

    assert use_ascii_glyphs(_Latin()) is True
    assert use_ascii_glyphs(Console()) is False


def test_report_and_estimate_agree_on_the_golden_total(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, batch_size=4)
    report = estimate(llama_model, training, get_gpu("4090"))
    output = _text(render_report(report, llama_model, get_gpu("4090"), training))

    assert "30,607" in output
    assert "DOES NOT FIT" in output
