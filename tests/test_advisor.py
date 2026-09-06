from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import pytest

from fitcheck.advisor import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_LORA_RANKS,
    AxisCeiling,
    FrontierPoint,
    SweepSpec,
    _dominates,
    advise,
)
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.estimator import TrainingConfig
from fitcheck.gpu_db import GpuSpec, get_gpu

_RTX_4090_USABLE = 23_500

# From docs/ADVISOR.md §5, all produced by the shipped estimator on the golden shape.
_ANCHOR_TOTAL_MIB = 20_039.87
_RANK_DOUBLE_COST_MIB = 832.0
_RANK_HALVE_SAVING_MIB = -416.0
_TOKENS_DOUBLE_COST_MIB = 10_567.20
_MAX_RANK_AT_4096_TOKENS = 330
_MAX_RANK_AT_2048_TOKENS = 736


@pytest.fixture
def llama_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(llama_31_8b_config)
    return fetch_model_config("meta-llama/Llama-3.1-8B")


@pytest.fixture
def qlora_base() -> TrainingConfig:
    """The fixed axes of the golden run. batch/seq/rank are what the sweep varies."""
    return TrainingConfig(
        precision="bf16",
        quantization="nf4",
        double_quant=False,
        optimizer="adamw",
        optimizer_dtype="fp32",
        batch_size=1,
        seq_len=2048,
        lora_rank=64,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        grad_checkpoint=True,
        flash_attn=True,
    )


@pytest.fixture
def sweep() -> SweepSpec:
    return SweepSpec(
        batch_sizes=(1, 2, 4, 8, 16),
        seq_lens=(512, 1024, 2048, 4096, 8192),
        lora_ranks=DEFAULT_LORA_RANKS,
    )


def _point(tokens_per_step: int, lora_rank: int, total_mib: float = 0.0) -> FrontierPoint:
    return FrontierPoint(
        batch_size=1,
        seq_len=tokens_per_step,
        lora_rank=lora_rank,
        tokens_per_step=tokens_per_step,
        total_mib=total_mib,
        fits=True,
        command="",
        equivalent_splits=((1, tokens_per_step),),
    )


def _ceiling(ceilings: list[AxisCeiling], axis: str) -> AxisCeiling:
    return next(ceiling for ceiling in ceilings if ceiling.axis == axis)


# --- the dominance comparator ------------------------------------------------------


def test_dominates_when_both_objectives_are_better() -> None:
    assert _dominates(_point(4096, 64), _point(2048, 32))


def test_dominates_when_one_objective_ties_and_the_other_wins() -> None:
    assert _dominates(_point(4096, 64), _point(4096, 32))
    assert _dominates(_point(4096, 64), _point(2048, 64))


def test_does_not_dominate_an_identical_point() -> None:
    assert not _dominates(_point(4096, 64), _point(4096, 64))


def test_does_not_dominate_when_the_two_objectives_disagree() -> None:
    """More tokens but less rank is a genuine trade — neither point may be dropped."""
    assert not _dominates(_point(4096, 32), _point(2048, 64))
    assert not _dominates(_point(2048, 64), _point(4096, 32))


def test_memory_is_not_an_objective() -> None:
    """A cheaper point with equal objectives must not survive on price alone.

    This is the regression guard for docs/ADVISOR.md §4: adding `minimise MiB` as a third
    objective made the comparator vacuous, so every fitting point survived.
    """
    expensive = _point(4096, 64, total_mib=20_000.0)
    cheap = _point(4096, 64, total_mib=10_000.0)
    assert not _dominates(cheap, expensive)
    assert not _dominates(expensive, cheap)


# --- the frontier ------------------------------------------------------------------


def test_frontier_is_far_smaller_than_the_fitting_set(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)

    assert report.grid_size == 150
    assert 0 < len(report.frontier) < report.fitting_count


def test_frontier_points_all_fit_the_card(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)

    for point in report.frontier:
        assert point.fits
        assert point.total_mib <= _RTX_4090_USABLE


def test_no_frontier_point_dominates_another(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)

    for left in report.frontier:
        for right in report.frontier:
            if left is not right:
                assert not _dominates(left, right)


def test_frontier_is_ordered_by_tokens_then_rank(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)
    keys = [(-point.tokens_per_step, -point.lora_rank) for point in report.frontier]

    assert keys == sorted(keys)


def test_recommended_is_the_head_of_the_frontier(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)

    assert report.recommended is report.frontier[0]
    assert report.recommended.tokens_per_step == max(
        point.tokens_per_step for point in report.frontier
    )


def test_recommended_is_none_when_nothing_fits(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    tiny_sweep = SweepSpec(batch_sizes=(8,), seq_lens=(8192,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("t4"), tiny_sweep)

    assert report.fitting_count == 0
    assert report.frontier == []
    assert report.recommended is None


# --- ties --------------------------------------------------------------------------


def test_equal_cost_splits_are_grouped_into_one_row(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    """With ckpt + flash, A_act depends only on b*s, so every split ties exactly."""
    tie_sweep = SweepSpec(
        batch_sizes=(1, 2, 4, 8), seq_lens=(512, 1024, 2048, 4096), lora_ranks=(64,)
    )

    report = advise(llama_model, qlora_base, get_gpu("4090"), tie_sweep)
    at_4096 = next(
        point for point in report.frontier if point.tokens_per_step == 4096
    )

    assert set(at_4096.equivalent_splits) == {(8, 512), (4, 1024), (2, 2048), (1, 4096)}
    assert at_4096.batch_size == 8


def test_eager_attention_breaks_the_tie(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    """Without Flash Attention the 9y*b*n_h*s^2 term is quadratic in s, so long
    sequences cost more than the same tokens spread over a larger batch."""
    eager_base = replace(qlora_base, flash_attn=False)
    tie_sweep = SweepSpec(
        batch_sizes=(1, 2, 4, 8), seq_lens=(512, 1024, 2048, 4096), lora_ranks=(64,)
    )

    report = advise(llama_model, eager_base, get_gpu("4090"), tie_sweep)
    at_4096 = next(
        point for point in report.frontier if point.tokens_per_step == 4096
    )

    assert (1, 4096) not in at_4096.equivalent_splits
    assert (8, 512) in at_4096.equivalent_splits


# --- prices ------------------------------------------------------------------------


def test_axis_prices_match_the_documented_deltas(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    """docs/ADVISOR.md §5a, at the batch 2 x seq 2048 rank 64 anchor."""
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)
    prices = {(price.axis, price.to_value): price.delta_mib for price in report.prices}

    assert report.anchor.batch_size == 2
    assert report.anchor.lora_rank == 64
    assert prices[("lora_rank", 128)] == pytest.approx(_RANK_DOUBLE_COST_MIB)
    assert prices[("lora_rank", 32)] == pytest.approx(_RANK_HALVE_SAVING_MIB)
    assert prices[("tokens_per_step", 8192)] == pytest.approx(_TOKENS_DOUBLE_COST_MIB)


def test_tokens_cost_far_more_than_rank(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    """The headline of the whole command: the two axes are not in the same league."""
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)
    prices = {(price.axis, price.to_value): price.delta_mib for price in report.prices}

    assert prices[("tokens_per_step", 8192)] > 12 * prices[("lora_rank", 128)]


def test_prices_omit_a_halving_that_does_not_exist(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    anchored = SweepSpec(batch_sizes=(1,), seq_lens=(2048,), lora_ranks=(1,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)
    moves = {(price.axis, price.from_value, price.to_value) for price in report.prices}

    assert ("lora_rank", 1, 0) not in moves
    assert not any(price.to_value < price.from_value for price in report.prices)


# --- ceilings ----------------------------------------------------------------------


def test_rank_ceiling_is_bisected_past_the_grid(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    """docs/ADVISOR.md §5c: the grid stops at 256, the real wall is 330."""
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)
    rank_ceiling = _ceiling(report.ceilings, "lora_rank")

    assert rank_ceiling.max_value == _MAX_RANK_AT_4096_TOKENS
    assert rank_ceiling.total_mib_at_max <= _RTX_4090_USABLE


def test_rank_ceiling_rises_when_tokens_fall(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    anchored = SweepSpec(batch_sizes=(1,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)

    assert _ceiling(report.ceilings, "lora_rank").max_value == _MAX_RANK_AT_2048_TOKENS


def test_tokens_ceiling_is_the_batch_ceiling_times_seq_len(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)
    batch_ceiling = _ceiling(report.ceilings, "batch_size")
    tokens_ceiling = _ceiling(report.ceilings, "tokens_per_step")

    assert batch_ceiling.max_value == 2
    assert tokens_ceiling.max_value == 4096
    assert tokens_ceiling.max_value == batch_ceiling.max_value * report.anchor.seq_len


def test_cutting_the_rank_does_not_move_the_tokens_wall(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    """The finding worth printing: rank is not the knob that is stopping you."""
    at_rank_64 = advise(
        llama_model,
        qlora_base,
        get_gpu("4090"),
        SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,)),
    )
    at_rank_8 = advise(
        llama_model,
        qlora_base,
        get_gpu("4090"),
        SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(8,)),
    )

    assert (
        _ceiling(at_rank_8.ceilings, "tokens_per_step").max_value
        == _ceiling(at_rank_64.ceilings, "tokens_per_step").max_value
        == 4096
    )


def test_ceilings_are_zero_when_nothing_fits(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    sweep = SweepSpec(batch_sizes=(1,), seq_lens=(8192,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("t4"), sweep)

    assert _ceiling(report.ceilings, "batch_size").max_value == 0
    assert _ceiling(report.ceilings, "lora_rank").max_value == 0
    assert _ceiling(report.ceilings, "batch_size").total_mib_at_max > 14_000


# --- the anchor and the report -----------------------------------------------------


def test_anchor_follows_the_recommendation(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)

    assert report.recommended is not None
    assert report.anchor.batch_size == report.recommended.batch_size
    assert report.anchor.seq_len == report.recommended.seq_len
    assert report.anchor.lora_rank == report.recommended.lora_rank


def test_anchor_falls_back_to_the_smallest_grid_point(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    sweep = SweepSpec(batch_sizes=(4, 8), seq_lens=(8192,), lora_ranks=(64, 128))

    report = advise(llama_model, qlora_base, get_gpu("t4"), sweep)

    assert report.recommended is None
    assert (report.anchor.batch_size, report.anchor.lora_rank) == (4, 64)


def test_anchor_total_matches_the_documented_number(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)

    assert report.recommended is not None
    assert report.recommended.total_mib == pytest.approx(_ANCHOR_TOTAL_MIB, abs=0.01)


def test_the_report_carries_the_fixed_axes_unchanged(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    report = advise(llama_model, qlora_base, get_gpu("4090"), sweep)

    assert report.base is qlora_base
    assert report.model_name == "Llama-3.1-8B"
    assert report.gpu == get_gpu("4090")


# --- the pasteable command ----------------------------------------------------------


def test_command_is_runnable_and_names_the_swept_axes(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, get_gpu("4090"), anchored)

    assert report.recommended is not None
    assert report.recommended.command == (
        "fitcheck Llama-3.1-8B --gpu 4090 --batch-size 2 --seq-len 2048 "
        "--lora-r 64 --precision bf16 --quant nf4 --grad-checkpoint --flash-attn"
    )


def test_command_spells_out_every_non_default_fixed_axis(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    unusual = replace(
        qlora_base,
        double_quant=True,
        optimizer="adam8bit",
        lora_targets=["q_proj", "v_proj"],
    )
    anchored = SweepSpec(batch_sizes=(1,), seq_lens=(1024,), lora_ranks=(16,))

    report = advise(llama_model, unusual, get_gpu("4090"), anchored)

    assert report.recommended is not None
    assert report.recommended.command == (
        "fitcheck Llama-3.1-8B --gpu 4090 --batch-size 1 --seq-len 1024 "
        "--lora-r 16 --precision bf16 --quant nf4 --double-quant "
        "--optimizer adam8bit --lora-targets q_proj,v_proj "
        "--grad-checkpoint --flash-attn"
    )


def test_command_names_a_non_default_optimizer_dtype(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    anchored = SweepSpec(batch_sizes=(1,), seq_lens=(1024,), lora_ranks=(16,))

    report = advise(
        llama_model, replace(qlora_base, optimizer_dtype="bf16"), get_gpu("4090"), anchored
    )

    assert report.recommended is not None
    assert "--optimizer-dtype bf16" in report.recommended.command


def test_command_falls_back_to_vram_mib_for_an_unlisted_card(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    custom_gpu = GpuSpec("Custom 32GB", 32_768, 30_000)
    anchored = SweepSpec(batch_sizes=(2,), seq_lens=(2048,), lora_ranks=(64,))

    report = advise(llama_model, qlora_base, custom_gpu, anchored)

    assert report.recommended is not None
    assert "--vram-mib 32768" in report.recommended.command
    assert "--gpu" not in report.recommended.command


# --- validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_sweep",
    [
        SweepSpec(batch_sizes=(), seq_lens=(2048,), lora_ranks=(64,)),
        SweepSpec(batch_sizes=(1,), seq_lens=(), lora_ranks=(64,)),
        SweepSpec(batch_sizes=(1,), seq_lens=(2048,), lora_ranks=()),
    ],
)
def test_advise_rejects_an_empty_axis(
    llama_model: ModelConfig, qlora_base: TrainingConfig, bad_sweep: SweepSpec
) -> None:
    with pytest.raises(ValueError, match="at least one value"):
        advise(llama_model, qlora_base, get_gpu("4090"), bad_sweep)


@pytest.mark.parametrize(
    "bad_sweep",
    [
        SweepSpec(batch_sizes=(0,), seq_lens=(2048,), lora_ranks=(64,)),
        SweepSpec(batch_sizes=(1,), seq_lens=(-2048,), lora_ranks=(64,)),
        SweepSpec(batch_sizes=(1,), seq_lens=(2048,), lora_ranks=(True,)),
    ],
)
def test_advise_rejects_a_non_positive_axis_value(
    llama_model: ModelConfig, qlora_base: TrainingConfig, bad_sweep: SweepSpec
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        advise(llama_model, qlora_base, get_gpu("4090"), bad_sweep)


def test_advise_rejects_a_full_fine_tuning_base(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    with pytest.raises(ValueError, match="sweeps the LoRA rank"):
        advise(
            llama_model, replace(qlora_base, lora_rank=None), get_gpu("4090"), sweep
        )


def test_advise_rejects_a_bad_fixed_axis_before_the_sweep(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    """Rule 4: invalid combinations fail up front, not part-way through the grid."""
    with pytest.raises(ValueError, match="Unsupported quantization"):
        advise(
            llama_model, replace(qlora_base, quantization="fp4"), get_gpu("4090"), sweep
        )


def test_advise_rejects_a_bad_gpu_and_model(
    llama_model: ModelConfig, qlora_base: TrainingConfig, sweep: SweepSpec
) -> None:
    with pytest.raises(ValueError, match="gpu_spec must be a GpuSpec"):
        advise(llama_model, qlora_base, "4090", sweep)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_config must be a ModelConfig"):
        advise("llama", qlora_base, get_gpu("4090"), sweep)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sweep must be a SweepSpec"):
        advise(llama_model, qlora_base, get_gpu("4090"), (1, 2))  # type: ignore[arg-type]


def test_axes_are_deduplicated_and_sorted(
    llama_model: ModelConfig, qlora_base: TrainingConfig
) -> None:
    messy = SweepSpec(batch_sizes=(2, 1, 2), seq_lens=(2048,), lora_ranks=(64, 8, 64))

    report = advise(llama_model, qlora_base, get_gpu("4090"), messy)

    assert report.sweep.batch_sizes == (1, 2)
    assert report.sweep.lora_ranks == (8, 64)
    assert report.grid_size == 4


def test_default_axes_are_positive_and_ordered() -> None:
    assert DEFAULT_BATCH_SIZES == tuple(sorted(DEFAULT_BATCH_SIZES))
    assert DEFAULT_LORA_RANKS == tuple(sorted(DEFAULT_LORA_RANKS))
    assert all(value > 0 for value in DEFAULT_BATCH_SIZES + DEFAULT_LORA_RANKS)
