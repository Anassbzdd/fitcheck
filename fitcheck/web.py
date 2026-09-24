"""Typed, Gradio-independent adapter for the FitCheck web app."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from fitcheck.advisor import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_LORA_RANKS,
    AdvisorReport,
    SweepSpec,
    advise,
)
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.estimator import (
    InferenceReport,
    MemoryReport,
    ServingConfig,
    TrainingConfig,
    activation_breakdown,
    estimate,
    estimate_inference,
)
from fitcheck.gpu_db import GpuSpec, get_gpu
from fitcheck.memory.lora import (
    LORA_TARGETS_FULL,
    LORA_TARGETS_MINIMAL,
    LORA_TARGETS_STANDARD,
)
from fitcheck.validation import validate_flag

_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "minimal": LORA_TARGETS_MINIMAL,
    "standard": LORA_TARGETS_STANDARD,
    "full": LORA_TARGETS_FULL,
}


@dataclass(frozen=True)
class TrainingResult:
    """Inputs and existing estimator output for one training request."""

    model: ModelConfig
    training: TrainingConfig
    gpu: GpuSpec
    report: MemoryReport
    activation_details: dict[str, float]


@dataclass(frozen=True)
class ServingResult:
    """Inputs and existing estimator output for one serving request."""

    model: ModelConfig
    serving: ServingConfig
    gpu: GpuSpec
    report: InferenceReport


@dataclass(frozen=True)
class AdvisorResult:
    """Inputs and existing advisor output for one sweep request."""

    model: ModelConfig
    report: AdvisorReport


def _positive_whole_number(value: object, name: str) -> int:
    """Accept integer-like Gradio numbers while rejecting fractions and booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive whole number")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{name} must be a positive whole number")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive whole number")
    return integer


def _positive_whole_numbers(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain at least one positive whole number")
    parsed = tuple(_positive_whole_number(value, name) for value in values)
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive whole number")
    return parsed


def _gpu(gpu_name: str, custom_vram_mib: object | None) -> GpuSpec:
    custom_vram = (
        None
        if custom_vram_mib is None
        else _positive_whole_number(custom_vram_mib, "custom total VRAM in MiB")
    )
    return get_gpu(
        None if custom_vram is not None else gpu_name,
        vram_mib=custom_vram,
    )


def _targets(target_preset: str) -> list[str]:
    if not isinstance(target_preset, str):
        raise ValueError("LoRA target preset must be minimal, standard, or full")
    try:
        return list(_TARGET_PRESETS[target_preset.strip().casefold()])
    except KeyError as error:
        raise ValueError(
            "LoRA target preset must be minimal, standard, or full"
        ) from error


def _training_config(
    *,
    batch_size: object,
    seq_len: object,
    quantization: str,
    double_quant: bool,
    precision: str,
    lora_rank: object,
    target_preset: str,
    full_finetuning: bool,
    optimizer: str,
    optimizer_dtype: str,
    grad_checkpoint: bool,
    flash_attn: bool,
    grad_accum_steps: object,
) -> TrainingConfig:
    full_finetuning = validate_flag(full_finetuning, "full_finetuning")
    grad_checkpoint = validate_flag(grad_checkpoint, "grad_checkpoint")
    flash_attn = validate_flag(flash_attn, "flash_attn")
    validated_rank = _positive_whole_number(lora_rank, "LoRA rank")
    rank = None if full_finetuning else validated_rank
    return TrainingConfig(
        precision=precision,
        quantization=quantization,
        double_quant=double_quant,
        optimizer=optimizer,
        optimizer_dtype=optimizer_dtype,
        batch_size=_positive_whole_number(batch_size, "micro-batch size"),
        seq_len=_positive_whole_number(seq_len, "sequence length"),
        lora_rank=rank,
        lora_targets=_targets(target_preset),
        grad_checkpoint=grad_checkpoint,
        flash_attn=flash_attn,
        grad_accum_steps=_positive_whole_number(
            grad_accum_steps, "gradient accumulation steps"
        ),
    )


def estimate_training_request(
    model_id: str,
    gpu_name: str,
    *,
    batch_size: object = 1,
    seq_len: object = 2048,
    quantization: str = "nf4",
    double_quant: bool = False,
    precision: str = "bf16",
    lora_rank: object = 16,
    target_preset: str = "standard",
    full_finetuning: bool = False,
    optimizer: str = "adamw",
    optimizer_dtype: str = "fp32",
    grad_checkpoint: bool = True,
    flash_attn: bool = False,
    grad_accum_steps: object = 1,
    custom_vram_mib: object | None = None,
) -> TrainingResult:
    """Build a training request and call FitCheck's existing estimator."""
    training = _training_config(
        batch_size=batch_size,
        seq_len=seq_len,
        quantization=quantization,
        double_quant=double_quant,
        precision=precision,
        lora_rank=lora_rank,
        target_preset=target_preset,
        full_finetuning=full_finetuning,
        optimizer=optimizer,
        optimizer_dtype=optimizer_dtype,
        grad_checkpoint=grad_checkpoint,
        flash_attn=flash_attn,
        grad_accum_steps=grad_accum_steps,
    )
    gpu = _gpu(gpu_name, custom_vram_mib)
    model = fetch_model_config(model_id)
    report = estimate(model, training, gpu)
    return TrainingResult(
        model=model,
        training=training,
        gpu=gpu,
        report=report,
        activation_details=activation_breakdown(model, training),
    )


def estimate_serving_request(
    model_id: str,
    gpu_name: str,
    *,
    seq_len: object = 2048,
    num_concurrent: object = 1,
    quantization: str = "none",
    double_quant: bool = False,
    precision: str = "fp16",
    custom_vram_mib: object | None = None,
) -> ServingResult:
    """Build a serving request and call FitCheck's existing inference estimator."""
    serving = ServingConfig(
        precision=precision,
        quantization=quantization,
        double_quant=double_quant,
        seq_len=_positive_whole_number(seq_len, "sequence length"),
        num_concurrent=_positive_whole_number(
            num_concurrent, "concurrent requests"
        ),
    )
    gpu = _gpu(gpu_name, custom_vram_mib)
    model = fetch_model_config(model_id)
    return ServingResult(
        model=model,
        serving=serving,
        gpu=gpu,
        report=estimate_inference(model, serving, gpu),
    )


def advise_training_request(
    model_id: str,
    gpu_name: str,
    *,
    batch_sizes: object = DEFAULT_BATCH_SIZES,
    seq_lens: object = (1024, 2048),
    lora_ranks: object = DEFAULT_LORA_RANKS,
    max_context_length: object | None = None,
    quantization: str = "nf4",
    double_quant: bool = False,
    precision: str = "bf16",
    target_preset: str = "standard",
    optimizer: str = "adamw",
    optimizer_dtype: str = "fp32",
    grad_checkpoint: bool = True,
    flash_attn: bool = False,
    custom_vram_mib: object | None = None,
) -> AdvisorResult:
    """Build a validated sweep and call FitCheck's existing advisor."""
    batches = _positive_whole_numbers(batch_sizes, "batch sizes")
    sequences = _positive_whole_numbers(seq_lens, "sequence lengths")
    ranks = _positive_whole_numbers(lora_ranks, "LoRA ranks")
    if max_context_length is not None:
        maximum = _positive_whole_number(max_context_length, "known maximum context length")
        too_long = sorted({value for value in sequences if value > maximum})
        if too_long:
            values = ", ".join(f"{value:,}" for value in too_long)
            raise ValueError(
                f"Selected sequence lengths {values} exceed the known maximum context "
                f"length {maximum:,}. Pricing a context the model cannot serve would "
                "recommend a run that cannot be trained."
            )

    base = _training_config(
        batch_size=min(batches),
        seq_len=min(sequences),
        quantization=quantization,
        double_quant=double_quant,
        precision=precision,
        lora_rank=min(ranks),
        target_preset=target_preset,
        full_finetuning=False,
        optimizer=optimizer,
        optimizer_dtype=optimizer_dtype,
        grad_checkpoint=grad_checkpoint,
        flash_attn=flash_attn,
        grad_accum_steps=1,
    )
    gpu = _gpu(gpu_name, custom_vram_mib)
    model = fetch_model_config(model_id)
    report = advise(
        model,
        base,
        gpu,
        SweepSpec(batch_sizes=batches, seq_lens=sequences, lora_ranks=ranks),
        model_id=model_id,
    )
    return AdvisorResult(model=model, report=report)
