from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import pytest

from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.estimator import MemoryReport, TrainingConfig, estimate
from fitcheck.gpu_db import get_gpu

_GOLDEN_PARAMS = 8_030_261_248
_GOLDEN_W_BASE = 7_753.02
_GOLDEN_W_LORA = 208.0
_GOLDEN_S_OPTIM = 416.0
_GOLDEN_G_GRAD = 208.0
_GOLDEN_A_ACT = 20_128.0
_GOLDEN_C_OVERHEAD = 1_894.05
_GOLDEN_TOTAL = 30_607.07
_GOLDEN_MAX_BATCH = 2

_RTX_4090_USABLE = 23_500


@pytest.fixture
def qlora_training() -> TrainingConfig:
    """The golden run: QLoRA r=64 [q,k,v,o], bs=4, seq=2048, bf16, AdamW fp32, ckpt+flash."""
    return TrainingConfig(
        precision="bf16",
        quantization="nf4",
        double_quant=False,
        optimizer="adamw",
        optimizer_dtype="fp32",
        batch_size=4,
        seq_len=2048,
        lora_rank=64,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        grad_checkpoint=True,
        flash_attn=True,
        grad_accum_steps=1,
    )


@pytest.fixture
def llama_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(llama_31_8b_config)
    return fetch_model_config("meta-llama/Llama-3.1-8B")


@pytest.fixture
def golden_report(llama_model: ModelConfig, qlora_training: TrainingConfig) -> MemoryReport:
    return estimate(llama_model, qlora_training, get_gpu("4090"))


def test_pipeline_derives_the_golden_parameter_count(llama_model: ModelConfig) -> None:
    assert llama_model.num_params == _GOLDEN_PARAMS
    assert llama_model.num_kv_heads == 8


def test_golden_component_breakdown(golden_report: MemoryReport) -> None:
    assert golden_report.weight_mib == pytest.approx(_GOLDEN_W_BASE, abs=0.01)
    assert golden_report.lora_mib == pytest.approx(_GOLDEN_W_LORA, rel=1e-9)
    assert golden_report.optimizer_mib == pytest.approx(_GOLDEN_S_OPTIM, rel=1e-9)
    assert golden_report.gradient_mib == pytest.approx(_GOLDEN_G_GRAD, rel=1e-9)
    assert golden_report.activation_mib == pytest.approx(_GOLDEN_A_ACT, rel=1e-9)
    assert golden_report.overhead_mib == pytest.approx(_GOLDEN_C_OVERHEAD, abs=0.01)


def test_golden_total_and_verdict(golden_report: MemoryReport) -> None:
    assert golden_report.total_mib == pytest.approx(_GOLDEN_TOTAL, abs=0.01)
    assert round(golden_report.total_mib) == 30_607

    assert golden_report.fits is False
    assert golden_report.gpu_capacity_mib == _RTX_4090_USABLE
    assert golden_report.headroom_mib == pytest.approx(-7_107.07, abs=0.01)
    assert golden_report.headroom_mib / golden_report.gpu_capacity_mib == pytest.approx(
        -0.30, abs=0.005
    )


def test_total_is_exactly_the_sum_of_the_six_components(golden_report: MemoryReport) -> None:
    components = (
        golden_report.weight_mib,
        golden_report.lora_mib,
        golden_report.optimizer_mib,
        golden_report.gradient_mib,
        golden_report.activation_mib,
        golden_report.overhead_mib,
    )

    assert golden_report.total_mib == pytest.approx(sum(components), rel=1e-12)


def test_overhead_tracks_base_weights_and_activations_only(
    golden_report: MemoryReport,
) -> None:
    expected = 500 + 0.05 * (golden_report.weight_mib + golden_report.activation_mib)

    assert golden_report.overhead_mib == pytest.approx(expected, rel=1e-12)


def test_max_batch_size_floors_at_two(golden_report: MemoryReport) -> None:
    assert golden_report.max_batch_size == _GOLDEN_MAX_BATCH


def test_max_batch_size_boundary_is_real_not_rounded(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    gpu = get_gpu("4090")

    at_max = estimate(llama_model, replace(qlora_training, batch_size=_GOLDEN_MAX_BATCH), gpu)
    over = estimate(
        llama_model, replace(qlora_training, batch_size=_GOLDEN_MAX_BATCH + 1), gpu
    )

    assert at_max.fits is True
    assert over.fits is False
    assert over.total_mib > _RTX_4090_USABLE


def test_max_batch_size_is_not_linear_extrapolation(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    gpu = get_gpu("4090")

    at_4 = estimate(llama_model, replace(qlora_training, batch_size=4), gpu).total_mib
    at_5 = estimate(llama_model, replace(qlora_training, batch_size=5), gpu).total_mib

    assert at_5 - at_4 == pytest.approx(5_283.60, abs=0.01)


def test_max_batch_size_adapts_to_a_smaller_gpu(golden_report: MemoryReport, llama_model: ModelConfig, qlora_training: TrainingConfig) -> None:
    on_3060 = estimate(llama_model, qlora_training, get_gpu("3060-12"))

    assert on_3060.max_batch_size == 0
    assert on_3060.max_batch_size < golden_report.max_batch_size


def test_gradient_accumulation_costs_zero_memory(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    gpu = get_gpu("4090")

    without = estimate(llama_model, qlora_training, gpu)
    with_accum = estimate(llama_model, replace(qlora_training, grad_accum_steps=8), gpu)

    assert with_accum.total_mib == pytest.approx(without.total_mib, rel=1e-12)
    assert with_accum.max_batch_size == without.max_batch_size
    assert with_accum.effective_batch_size == 32
    assert without.effective_batch_size == 4


def test_activations_follow_micro_batch_not_effective_batch(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    gpu = get_gpu("4090")

    micro_4 = estimate(llama_model, replace(qlora_training, batch_size=4), gpu)
    micro_1_accum_4 = estimate(
        llama_model, replace(qlora_training, batch_size=1, grad_accum_steps=4), gpu
    )

    assert micro_1_accum_4.effective_batch_size == micro_4.effective_batch_size
    assert micro_1_accum_4.activation_mib == pytest.approx(_GOLDEN_A_ACT / 4, rel=1e-9)
    assert micro_1_accum_4.total_mib < micro_4.total_mib


def test_flash_attention_is_free_here_because_the_lm_head_hump_dominates(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    """Llama-3.1-8B has a 128k vocabulary, so under checkpointing the LM-head hump
    (16,032 MiB) is larger than one layer's recompute even with the score matrix in
    it. The peak takes the max of the two, so turning Flash Attention off costs
    nothing at this shape. Measured: SmolLM2 eager vs SDPA differed by 16 MiB."""
    gpu = get_gpu("4090")

    on = estimate(llama_model, qlora_training, gpu)
    off = estimate(llama_model, replace(qlora_training, flash_attn=False), gpu)

    assert off.activation_mib == pytest.approx(_GOLDEN_A_ACT, rel=1e-9)
    assert off.total_mib - on.total_mib == pytest.approx(0.0, abs=0.01)


def test_flash_attention_does_pay_once_the_layer_hump_wins(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    """Push the sequence out and the s^2 layer hump overtakes the LM head, at which
    point removing the score matrix is worth a great deal."""
    gpu = get_gpu("4090")
    long_seq = replace(qlora_training, seq_len=16_384, batch_size=1)

    on = estimate(llama_model, long_seq, gpu)
    off = estimate(llama_model, replace(long_seq, flash_attn=False), gpu)

    assert off.activation_mib > on.activation_mib
    assert off.total_mib - on.total_mib > 10_000


def test_full_finetune_path_replaces_lora_with_every_parameter(
    llama_model: ModelConfig,
) -> None:
    full_ft = TrainingConfig(
        precision="bf16",
        quantization="none",
        lora_rank=None,
        batch_size=1,
        seq_len=2048,
        grad_checkpoint=True,
        flash_attn=True,
    )

    report = estimate(llama_model, full_ft, get_gpu("4090"))

    assert report.lora_mib == 0.0
    assert report.weight_mib == pytest.approx(_GOLDEN_PARAMS * 2 / 1024**2, rel=1e-9)
    assert report.optimizer_mib == pytest.approx(_GOLDEN_PARAMS * 12 / 1024**2, rel=1e-9)
    assert report.fits is False
    assert report.headroom_mib < 0
    assert report.max_batch_size == 0


def test_full_finetuning_a_quantized_base_is_out_of_scope(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    with pytest.raises(ValueError, match="does not model full fine-tuning of a nf4 base model"):
        estimate(llama_model, replace(qlora_training, lora_rank=None), get_gpu("4090"))


def test_quantized_full_finetune_error_is_scoped_not_universal(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    with pytest.raises(ValueError) as raised:
        estimate(llama_model, replace(qlora_training, lora_rank=None), get_gpu("4090"))

    message = str(raised.value)
    assert "fitcheck scope limitation" in message
    assert "not a claim that quantized models cannot be trained" in message


def test_savings_hints_price_each_toggle_as_a_total_delta(
    golden_report: MemoryReport,
) -> None:
    hints = golden_report.savings_hints

    assert "adamw -> adam8bit: saves 312 MiB" in hints
    assert "--flash-attn OFF: costs 0 MiB (currently ON)" in hints
    assert "--grad-checkpoint OFF: costs +32,256 MiB (currently ON)" in hints
    assert "--grad-accum 8: costs 0 MiB (accumulation is free)" in hints


def test_report_is_json_serializable_for_ci(golden_report: MemoryReport) -> None:
    import dataclasses
    import json

    payload = json.loads(json.dumps(dataclasses.asdict(golden_report)))

    assert payload["total_mib"] == pytest.approx(_GOLDEN_TOTAL, abs=0.01)
    assert payload["fits"] is False
    assert payload["max_batch_size"] == _GOLDEN_MAX_BATCH
