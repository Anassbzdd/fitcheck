# Orchestrator: calls all 6 components, returns MemoryReport
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from fitcheck.config_parser import ModelConfig
from fitcheck.gpu_db import GpuSpec
from fitcheck.memory.activations import estimate_activation_memory
from fitcheck.memory.gradients import estimate_gradient_memory
from fitcheck.memory.lora import (
    LORA_TARGETS_STANDARD,
    _target_dims,
    estimate_lora_memory,
)
from fitcheck.memory.optimizer import estimate_optimizer_memory
from fitcheck.memory.overhead import estimate_overhead
from fitcheck.memory.weights import QuantizationConfig, estimate_weight_memory

_QUANTIZATIONS = ("none", "nf4", "int8")
_MAX_BATCH_SEARCH_CEILING = 1 << 20
_HINT_GRAD_ACCUM_STEPS = 8


@dataclass(frozen=True)
class TrainingConfig:
    precision: str = "bf16"
    quantization: str = "none"
    double_quant: bool = False
    optimizer: str = "adamw"
    optimizer_dtype: str = "fp32"
    batch_size: int = 1
    seq_len: int = 2048
    lora_rank: int | None = 16
    lora_targets: list[str] = field(
        default_factory=lambda: list(LORA_TARGETS_STANDARD)
    )
    grad_checkpoint: bool = False
    flash_attn: bool = False
    grad_accum_steps: int = 1


@dataclass(frozen=True)
class MemoryReport:
    weight_mib: float
    lora_mib: float
    optimizer_mib: float
    gradient_mib: float
    activation_mib: float
    overhead_mib: float
    total_mib: float
    gpu_capacity_mib: float
    headroom_mib: float
    fits: bool
    max_batch_size: int
    effective_batch_size: int
    savings_hints: list[str]


@dataclass(frozen=True)
class _Components:
    weight_mib: float
    lora_mib: float
    optimizer_mib: float
    gradient_mib: float
    activation_mib: float
    overhead_mib: float

    @property
    def total_mib(self) -> float:
        return (
            self.weight_mib
            + self.lora_mib
            + self.optimizer_mib
            + self.gradient_mib
            + self.activation_mib
            + self.overhead_mib
        )


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_flag(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_quantization(quantization: str) -> str:
    if not isinstance(quantization, str):
        raise ValueError("quantization must be a string")

    normalized_quantization = quantization.strip().casefold()
    if normalized_quantization not in _QUANTIZATIONS:
        supported = ", ".join(_QUANTIZATIONS)
        raise ValueError(
            f"Unsupported quantization '{quantization}'. Supported: {supported}."
        )
    return normalized_quantization


def _count_lora_params(config: ModelConfig, rank: int, targets: Iterable[str]) -> int:
    target_list = list(targets)
    if not target_list:
        raise ValueError("lora_targets must contain at least one LoRA target module")

    seen: set[str] = set()
    dims_sum = 0
    for target in target_list:
        if target in seen:
            raise ValueError(f"Duplicate LoRA target '{target}'")
        seen.add(target)
        d_in, d_out = _target_dims(config, target)
        dims_sum += d_in + d_out

    return config.num_layers * rank * dims_sum


def _base_weight_memory(config: ModelConfig, training: TrainingConfig) -> float:
    quantization = _validate_quantization(training.quantization)
    _validate_flag(training.double_quant, "double_quant")

    if quantization == "none":
        return estimate_weight_memory(config.num_params, training.precision)

    return estimate_weight_memory(
        config.num_params,
        quantization,
        QuantizationConfig(enabled=True, double_quant=training.double_quant),
    )


def _compute_components(config: ModelConfig, training: TrainingConfig) -> _Components:
    if not isinstance(config, ModelConfig):
        raise ValueError("model_config must be a ModelConfig")

    quantization = _validate_quantization(training.quantization)
    is_lora = training.lora_rank is not None
    if quantization != "none" and not is_lora:
        raise ValueError(
            f"fitcheck does not model full fine-tuning of a {quantization} base model: "
            "its memory model assumes a quantized base stays frozen while only adapters "
            "train. This is a fitcheck scope limitation, not a claim that quantized "
            "models cannot be trained. Set lora_rank, or use quantization='none'."
        )

    weight_mib = _base_weight_memory(config, training)

    if is_lora:
        rank = _validate_positive_int(training.lora_rank, "lora_rank")
        targets = list(training.lora_targets)
        lora_mib = estimate_lora_memory(config, rank, targets, training.precision)
        trainable_params = _count_lora_params(config, rank, targets)
    else:
        lora_mib = 0.0
        trainable_params = config.num_params

    optimizer_mib = estimate_optimizer_memory(
        trainable_params,
        training.optimizer,
        is_lora,
        training.optimizer_dtype,
        training.precision,
    )
    gradient_mib = estimate_gradient_memory(trainable_params, training.precision)
    activation_mib = estimate_activation_memory(
        config,
        training.batch_size,
        training.seq_len,
        training.grad_checkpoint,
        training.flash_attn,
        training.precision,
    )

    return _Components(
        weight_mib=weight_mib,
        lora_mib=lora_mib,
        optimizer_mib=optimizer_mib,
        gradient_mib=gradient_mib,
        activation_mib=activation_mib,
        overhead_mib=estimate_overhead(weight_mib, activation_mib),
    )


def _total_at_batch(
    config: ModelConfig, training: TrainingConfig, batch_size: int
) -> float:
    return _compute_components(config, replace(training, batch_size=batch_size)).total_mib


def _max_batch_size(
    config: ModelConfig, training: TrainingConfig, usable_mib: float
) -> int:
    if _total_at_batch(config, training, 1) > usable_mib:
        return 0

    low, high = 1, 2
    while high <= _MAX_BATCH_SEARCH_CEILING and _total_at_batch(config, training, high) <= usable_mib:
        low, high = high, high * 2

    if high > _MAX_BATCH_SEARCH_CEILING:
        return low

    while high - low > 1:
        mid = (low + high) // 2
        if _total_at_batch(config, training, mid) <= usable_mib:
            low = mid
        else:
            high = mid

    return low


def _format_delta(delta_mib: float) -> str:
    rounded_delta = round(delta_mib)
    if rounded_delta < 0:
        return f"saves {abs(rounded_delta):,} MiB"
    if rounded_delta > 0:
        return f"costs +{rounded_delta:,} MiB"
    return "costs 0 MiB"


def _savings_hints(
    config: ModelConfig, training: TrainingConfig, baseline_mib: float
) -> list[str]:

    def delta(**overrides: object) -> str:
        variant = replace(training, **overrides)
        return _format_delta(_compute_components(config, variant).total_mib - baseline_mib)

    hints: list[str] = []

    if training.optimizer.strip().casefold() != "adam8bit":
        hints.append(f"{training.optimizer} -> adam8bit: {delta(optimizer='adam8bit')}")

    for attribute, flag in (("flash_attn", "--flash-attn"), ("grad_checkpoint", "--grad-checkpoint")):
        is_on = getattr(training, attribute)
        hints.append(
            f"{flag} {'OFF' if is_on else 'ON'}: {delta(**{attribute: not is_on})} "
            f"(currently {'ON' if is_on else 'OFF'})"
        )

    hints.append(
        f"--grad-accum {_HINT_GRAD_ACCUM_STEPS}: "
        f"{delta(grad_accum_steps=_HINT_GRAD_ACCUM_STEPS)} (accumulation is free)"
    )

    return hints


def estimate(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    gpu_spec: GpuSpec,
) -> MemoryReport:
    if not isinstance(gpu_spec, GpuSpec):
        raise ValueError("gpu_spec must be a GpuSpec")

    grad_accum_steps = _validate_positive_int(
        training_config.grad_accum_steps, "grad_accum_steps"
    )

    components = _compute_components(model_config, training_config)
    total_mib = components.total_mib
    capacity_mib = float(gpu_spec.usable_mib)

    return MemoryReport(
        weight_mib=components.weight_mib,
        lora_mib=components.lora_mib,
        optimizer_mib=components.optimizer_mib,
        gradient_mib=components.gradient_mib,
        activation_mib=components.activation_mib,
        overhead_mib=components.overhead_mib,
        total_mib=total_mib,
        gpu_capacity_mib=capacity_mib,
        headroom_mib=capacity_mib - total_mib,
        fits=total_mib <= capacity_mib,
        max_batch_size=_max_batch_size(model_config, training_config, capacity_mib),
        # Accumulation is a display-only multiplier — it costs exactly zero memory.
        effective_batch_size=training_config.batch_size * grad_accum_steps,
        savings_hints=_savings_hints(model_config, training_config, total_mib),
    )
