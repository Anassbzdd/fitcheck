from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.display import (
    _why_largest,
    render_explanation,
    render_verbose_detail,
)
from fitcheck.estimator import (
    MemoryReport,
    TrainingConfig,
    estimate,
    trainable_params,
)
from fitcheck.gpu_db import GpuSpec, get_gpu
from fitcheck.memory.activations import _validate_quantization as _profile_for
from fitcheck.overhead_db import get_overhead_profile
from fitcheck.utils import precision_to_bytes
from rich.console import Console

_MIB = 1024.0**2


@pytest.fixture
def llama_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(llama_31_8b_config)
    return fetch_model_config("meta-llama/Llama-3.1-8B")


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


def _text(renderable: object) -> str:
    console = Console(width=120, color_system=None)
    with console.capture() as capture:
        console.print(renderable)
    return " ".join(capture.get().split())


def _explain(
    model: ModelConfig, training: TrainingConfig, gpu: GpuSpec
) -> tuple[str, MemoryReport]:
    report = estimate(model, training, gpu)
    rendered = _text(
        render_explanation(report, model, training, gpu, ascii_only=True)
    )
    return rendered, report


@pytest.mark.parametrize("quantization", ["none", "nf4", "int8"])
def test_logits_copies_come_from_the_selected_profile(
    llama_model: ModelConfig, qlora_training: TrainingConfig, quantization: str
) -> None:
    training = replace(qlora_training, quantization=quantization)
    copies = _profile_for(quantization).logits_copies

    rendered = _text(render_verbose_detail(llama_model, training, ascii_only=True))
    assert f"A_logits, {copies:g} fp32 copies of (b, s, V)" in rendered


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [("none", "L x gamma*b*s*h"), ("nf4", "L x 4*b*s*h"), ("int8", "2L x gamma*b*s*h")],
)
def test_checkpoint_store_formula_comes_from_the_selected_profile(
    llama_model: ModelConfig,
    qlora_training: TrainingConfig,
    quantization: str,
    expected: str,
) -> None:
    training = replace(qlora_training, quantization=quantization)
    rendered = _text(render_verbose_detail(llama_model, training, ascii_only=True))
    assert f"Checkpoint store ({expected})" in rendered


def test_the_logits_headline_also_quotes_the_profile_copies(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, quantization="none", batch_size=4)
    rendered, report = _explain(llama_model, training, get_gpu("a100-80"))

    assert "Largest component: activations" in rendered
    assert "A_logits: 3.5 fp32 copies" in rendered
    assert report.activation_mib > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # AdamW fp32 states over LoRA params: 8 bytes
        {"optimizer_dtype": "bf16"},  # 4 bytes, not 8
        {"optimizer": "adam8bit"},  # 2 bytes, non-AdamW wording
        {"optimizer": "sgd-momentum"},
        # Full fine-tune in bf16: 8 optimizer bytes plus fp32 master weights.
        {"quantization": "none", "lora_rank": None},
        {"quantization": "none", "lora_rank": None, "precision": "fp32"},
    ],
)
def test_optimizer_prose_quotes_the_bytes_per_param_that_was_billed(
    llama_model: ModelConfig, qlora_training: TrainingConfig, overrides: dict[str, Any]
) -> None:
    training = replace(qlora_training, **overrides)
    report = estimate(llama_model, training, get_gpu("a100-80"))

    prose = _why_largest("optimizer states", llama_model, training, get_gpu("a100-80"))
    quoted = re.search(r"([\d.]+) bytes (?:per trainable param|each)", prose)
    assert quoted is not None, prose

    billed = report.optimizer_mib * _MIB / trainable_params(llama_model, training)
    assert float(quoted.group(1)) == pytest.approx(billed, rel=1e-9)


def test_optimizer_prose_drops_the_contrast_when_the_dtypes_agree(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, optimizer_dtype="bf16")
    prose = _why_largest("optimizer states", llama_model, training, get_gpu("4090"))
    assert "4 bytes per trainable param" in prose
    assert "even though you train in" not in prose


@pytest.mark.parametrize(
    ("overrides", "expected_dtype"),
    [
        ({}, "fp32"),  # LoRA adapters are always fp32, so their grads are too
        ({"precision": "fp16"}, "fp32"),
        ({"precision": "fp16", "quantization": "none"}, "fp32"),
        ({"quantization": "none", "lora_rank": None}, "bf16"),  # full fine-tune
        ({"quantization": "none", "lora_rank": None, "precision": "fp32"}, "fp32"),
    ],
)
def test_gradient_prose_quotes_the_dtype_that_was_billed(
    llama_model: ModelConfig,
    qlora_training: TrainingConfig,
    overrides: dict[str, Any],
    expected_dtype: str,
) -> None:
    training = replace(qlora_training, **overrides)
    report = estimate(llama_model, training, get_gpu("a100-80"))

    prose = _why_largest("gradients", llama_model, training, get_gpu("a100-80"))
    quoted = re.search(r"in (\w+)\.$", prose)
    assert quoted is not None, prose
    assert quoted.group(1) == expected_dtype

    billed = report.gradient_mib * _MIB / trainable_params(llama_model, training)
    assert precision_to_bytes(quoted.group(1)) == pytest.approx(billed, rel=1e-9)


def test_calibrated_gpu_overhead_prose_quotes_the_measured_profile(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    training = replace(qlora_training, lora_rank=1, lora_targets=["q_proj"], seq_len=512)
    report = estimate(llama_model, training, get_gpu("t4"))
    prose = _why_largest("CUDA overhead", llama_model, training, get_gpu("t4"))

    profile = get_overhead_profile("t4", training.flash_attn, training.quantization)
    assert profile.measured
    assert f"the measured {profile.base_context_mib:,.3f} MiB CUDA context" in prose
    assert "500 MiB CUDA context floor" not in prose

    # The quoted coefficient and hump must reproduce the charged C_overhead.
    quoted = re.search(r"plus ([\d.]+)% of A_\w+ \(([\d,]+) MiB\)", prose)
    assert quoted is not None, prose
    fragmentation = float(quoted.group(1)) / 100.0
    hump = float(quoted.group(2).replace(",", ""))
    assert profile.base_context_mib + fragmentation * hump == pytest.approx(
        report.overhead_mib, rel=1e-3
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # an uncalibrated card keeps the conservative default
        {"quantization": "int8"},  # no int8 profile ships for any card
    ],
)
def test_uncalibrated_overhead_prose_quotes_the_default_profile(
    llama_model: ModelConfig, qlora_training: TrainingConfig, overrides: dict[str, Any]
) -> None:
    training = replace(qlora_training, seq_len=512, **overrides)
    gpu = get_gpu("4090") if not overrides else get_gpu("t4")
    report = estimate(llama_model, training, gpu)
    prose = _why_largest("CUDA overhead", llama_model, training, gpu)

    assert "the 500 MiB CUDA context floor plus 5% of weights and activations" in prose
    expected = 500.0 + 0.05 * (report.weight_mib + report.activation_mib)
    assert expected == pytest.approx(report.overhead_mib, rel=1e-9)


def test_a_calibrated_gpu_without_checkpointing_falls_back_in_prose_too(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    """estimate_overhead drops to the default profile with no checkpointed max()."""
    training = replace(qlora_training, seq_len=256, grad_checkpoint=False)
    report = estimate(llama_model, training, get_gpu("t4"))
    prose = _why_largest("CUDA overhead", llama_model, training, get_gpu("t4"))

    assert "the 500 MiB CUDA context floor plus 5% of weights and activations" in prose
    expected = 500.0 + 0.05 * (report.weight_mib + report.activation_mib)
    assert expected == pytest.approx(report.overhead_mib, rel=1e-9)


def test_render_explanation_without_a_gpu_still_renders(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    report = estimate(llama_model, qlora_training, get_gpu("t4"))
    assert _text(
        render_explanation(report, llama_model, qlora_training, ascii_only=True)
    )
