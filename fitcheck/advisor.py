from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import product
from typing import Iterable, Sequence

from fitcheck.config_parser import ModelConfig
from fitcheck.estimator import (
    TrainingConfig,
    _compute_components,
    _largest_fitting,
)
from fitcheck.gpu_db import GPU_DB, GpuSpec
from fitcheck.memory.lora import LORA_TARGETS_STANDARD

DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_LORA_RANKS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)

_TIE_REL_TOL = 1e-9
_AXIS_TOKENS = "tokens_per_step"
_AXIS_RANK = "lora_rank"
_AXIS_BATCH = "batch_size"


@dataclass(frozen=True)
class SweepSpec:
    batch_sizes: tuple[int, ...]
    seq_lens: tuple[int, ...]
    lora_ranks: tuple[int, ...]


@dataclass(frozen=True)
class FrontierPoint:
    batch_size: int
    seq_len: int
    lora_rank: int
    tokens_per_step: int
    total_mib: float
    fits: bool
    command: str
    equivalent_splits: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AxisPrice:
    axis: str
    from_value: int
    to_value: int
    delta_mib: float


@dataclass(frozen=True)
class AxisCeiling:
    axis: str
    max_value: int
    total_mib_at_max: float


@dataclass(frozen=True)
class AdvisorReport:
    model_name: str
    gpu: GpuSpec
    base: TrainingConfig
    sweep: SweepSpec
    grid_size: int
    fitting_count: int
    frontier: list[FrontierPoint]
    ceilings: list[AxisCeiling]
    prices: list[AxisPrice]
    anchor: TrainingConfig
    recommended: FrontierPoint | None


def _validate_axis(values: Iterable[int], name: str) -> tuple[int, ...]:
    value_list = list(values)
    if not value_list:
        raise ValueError(f"{name} must contain at least one value")

    for value in value_list:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must contain positive integers, got {value!r}")

    return tuple(sorted(set(value_list)))


def _validated_sweep(sweep: SweepSpec) -> SweepSpec:
    if not isinstance(sweep, SweepSpec):
        raise ValueError("sweep must be a SweepSpec")

    return SweepSpec(
        batch_sizes=_validate_axis(sweep.batch_sizes, "batch_sizes"),
        seq_lens=_validate_axis(sweep.seq_lens, "seq_lens"),
        lora_ranks=_validate_axis(sweep.lora_ranks, "lora_ranks"),
    )


def _gpu_flag(gpu: GpuSpec) -> str:
    for key, spec in GPU_DB.items():
        if spec == gpu:
            return f"--gpu {key}"
    return f"--vram-mib {gpu.vram_mib}"


def _command(model_name: str, gpu: GpuSpec, training: TrainingConfig) -> str:
    parts = [
        "fitcheck",
        model_name,
        _gpu_flag(gpu),
        f"--batch-size {training.batch_size}",
        f"--seq-len {training.seq_len}",
        f"--lora-r {training.lora_rank}",
        f"--precision {training.precision}",
    ]

    if training.quantization != "none":
        parts.append(f"--quant {training.quantization}")
    if training.double_quant:
        parts.append("--double-quant")
    if training.optimizer != "adamw":
        parts.append(f"--optimizer {training.optimizer}")
    elif training.optimizer_dtype != "fp32":
        parts.append(f"--optimizer-dtype {training.optimizer_dtype}")
    if tuple(training.lora_targets) != LORA_TARGETS_STANDARD:
        parts.append(f"--lora-targets {','.join(training.lora_targets)}")
    if training.grad_checkpoint:
        parts.append("--grad-checkpoint")
    if training.flash_attn:
        parts.append("--flash-attn")

    return " ".join(parts)


def _dominates(left: FrontierPoint, right: FrontierPoint) -> bool:
    at_least_as_good = (
        left.tokens_per_step >= right.tokens_per_step
        and left.lora_rank >= right.lora_rank
    )
    strictly_better = (
        left.tokens_per_step > right.tokens_per_step
        or left.lora_rank > right.lora_rank
    )
    return at_least_as_good and strictly_better


def _grouped_points(
    config: ModelConfig,
    base: TrainingConfig,
    gpu: GpuSpec,
    sweep: SweepSpec,
) -> tuple[list[FrontierPoint], int, int]:
    usable_mib = float(gpu.usable_mib)
    groups: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    grid_size = 0
    fitting_count = 0

    for batch_size, seq_len, lora_rank in product(
        sweep.batch_sizes, sweep.seq_lens, sweep.lora_ranks
    ):
        grid_size += 1
        total_mib = _compute_components(
            config,
            replace(
                base, batch_size=batch_size, seq_len=seq_len, lora_rank=lora_rank
            ),
        ).total_mib
        if total_mib > usable_mib:
            continue

        fitting_count += 1
        key = (batch_size * seq_len, lora_rank)
        groups.setdefault(key, []).append((batch_size, seq_len, total_mib))

    points: list[FrontierPoint] = []
    for (tokens_per_step, lora_rank), splits in groups.items():
        cheapest_mib = min(total_mib for _, _, total_mib in splits)
        tied = [
            (batch_size, seq_len)
            for batch_size, seq_len, total_mib in splits
            if math.isclose(total_mib, cheapest_mib, rel_tol=_TIE_REL_TOL)
        ]
        tied.sort(key=lambda split: -split[0])
        batch_size, seq_len = tied[0]

        points.append(
            FrontierPoint(
                batch_size=batch_size,
                seq_len=seq_len,
                lora_rank=lora_rank,
                tokens_per_step=tokens_per_step,
                total_mib=cheapest_mib,
                fits=True,
                command=_command(
                    config.name,
                    gpu,
                    replace(
                        base,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        lora_rank=lora_rank,
                    ),
                ),
                equivalent_splits=tuple(tied),
            )
        )

    return points, grid_size, fitting_count


def _frontier(points: Sequence[FrontierPoint]) -> list[FrontierPoint]:
    surviving = [
        point
        for point in points
        if not any(_dominates(other, point) for other in points if other is not point)
    ]
    surviving.sort(key=lambda point: (-point.tokens_per_step, -point.lora_rank))
    return surviving


def _total_at(
    config: ModelConfig,
    base: TrainingConfig,
    batch_size: int,
    seq_len: int,
    lora_rank: int,
) -> float:
    return _compute_components(
        config,
        replace(base, batch_size=batch_size, seq_len=seq_len, lora_rank=lora_rank),
    ).total_mib


def _ceilings(
    config: ModelConfig, anchor: TrainingConfig, gpu: GpuSpec
) -> list[AxisCeiling]:
    usable_mib = float(gpu.usable_mib)

    max_batch = _largest_fitting(
        lambda batch_size: _total_at(
            config, anchor, batch_size, anchor.seq_len, anchor.lora_rank
        ),
        usable_mib,
    )
    max_rank = _largest_fitting(
        lambda lora_rank: _total_at(
            config, anchor, anchor.batch_size, anchor.seq_len, lora_rank
        ),
        usable_mib,
    )

    def total_at_ceiling(batch_size: int, lora_rank: int) -> float:
        return _total_at(
            config,
            anchor,
            max(batch_size, 1),
            anchor.seq_len,
            max(lora_rank, 1),
        )

    return [
        AxisCeiling(
            axis=_AXIS_BATCH,
            max_value=max_batch,
            total_mib_at_max=total_at_ceiling(max_batch, anchor.lora_rank),
        ),
        AxisCeiling(
            axis=_AXIS_TOKENS,
            max_value=max_batch * anchor.seq_len,
            total_mib_at_max=total_at_ceiling(max_batch, anchor.lora_rank),
        ),
        AxisCeiling(
            axis=_AXIS_RANK,
            max_value=max_rank,
            total_mib_at_max=total_at_ceiling(anchor.batch_size, max_rank),
        ),
    ]


def _prices(config: ModelConfig, anchor: TrainingConfig) -> list[AxisPrice]:
    baseline_mib = _total_at(
        config, anchor, anchor.batch_size, anchor.seq_len, anchor.lora_rank
    )
    tokens = anchor.batch_size * anchor.seq_len
    prices: list[AxisPrice] = []

    def add(axis: str, from_value: int, to_value: int, total_mib: float) -> None:
        prices.append(
            AxisPrice(
                axis=axis,
                from_value=from_value,
                to_value=to_value,
                delta_mib=total_mib - baseline_mib,
            )
        )

    if anchor.lora_rank > 1:
        halved_rank = anchor.lora_rank // 2
        add(
            _AXIS_RANK,
            anchor.lora_rank,
            halved_rank,
            _total_at(config, anchor, anchor.batch_size, anchor.seq_len, halved_rank),
        )
    add(
        _AXIS_RANK,
        anchor.lora_rank,
        anchor.lora_rank * 2,
        _total_at(
            config, anchor, anchor.batch_size, anchor.seq_len, anchor.lora_rank * 2
        ),
    )

    if anchor.batch_size > 1:
        halved_batch = anchor.batch_size // 2
        add(
            _AXIS_TOKENS,
            tokens,
            halved_batch * anchor.seq_len,
            _total_at(
                config, anchor, halved_batch, anchor.seq_len, anchor.lora_rank
            ),
        )
    add(
        _AXIS_TOKENS,
        tokens,
        anchor.batch_size * 2 * anchor.seq_len,
        _total_at(
            config, anchor, anchor.batch_size * 2, anchor.seq_len, anchor.lora_rank
        ),
    )

    return prices


def _anchor_config(
    base: TrainingConfig,
    sweep: SweepSpec,
    recommended: FrontierPoint | None,
) -> TrainingConfig:
    if recommended is not None:
        return replace(
            base,
            batch_size=recommended.batch_size,
            seq_len=recommended.seq_len,
            lora_rank=recommended.lora_rank,
        )

    return replace(
        base,
        batch_size=min(sweep.batch_sizes),
        seq_len=min(sweep.seq_lens),
        lora_rank=min(sweep.lora_ranks),
    )


def advise(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    gpu_spec: GpuSpec,
    sweep: SweepSpec,
) -> AdvisorReport:
    if not isinstance(model_config, ModelConfig):
        raise ValueError("model_config must be a ModelConfig")
    if not isinstance(gpu_spec, GpuSpec):
        raise ValueError("gpu_spec must be a GpuSpec")
    if training_config.lora_rank is None:
        raise ValueError(
            "advise sweeps the LoRA rank, so it cannot start from a full fine-tuning "
            "config. Set lora_rank on the base config, or use estimate() for full FT."
        )

    validated_sweep = _validated_sweep(sweep)

    _compute_components(
        model_config,
        replace(
            training_config,
            batch_size=validated_sweep.batch_sizes[0],
            seq_len=validated_sweep.seq_lens[0],
            lora_rank=validated_sweep.lora_ranks[0],
        ),
    )

    points, grid_size, fitting_count = _grouped_points(
        model_config, training_config, gpu_spec, validated_sweep
    )
    frontier = _frontier(points)
    recommended = frontier[0] if frontier else None
    anchor = _anchor_config(training_config, validated_sweep, recommended)

    return AdvisorReport(
        model_name=model_config.name,
        gpu=gpu_spec,
        base=training_config,
        sweep=validated_sweep,
        grid_size=grid_size,
        fitting_count=fitting_count,
        frontier=frontier,
        ceilings=_ceilings(model_config, anchor, gpu_spec),
        prices=_prices(model_config, anchor),
        anchor=anchor,
        recommended=recommended,
    )
