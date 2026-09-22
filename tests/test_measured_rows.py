"""The corrected activation profiles, held against the rows that produced them.

`tests/test_activations.py` checks the formula against hand-derived numbers. This file
checks it against `data/measurements/`, which is the only evidence fitcheck has that the
constants describe a real allocator rather than a tidy derivation.

Offline on purpose. Every row in the 2026-09-15 sweep carries the `model_config` it was
measured with, so nothing here touches the Hub: a guard that only runs when the network
is up is a guard that reports green on the days it matters least.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.estimator import TrainingConfig, estimate
from fitcheck.gpu_db import get_gpu

_ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "measurements"
_SWEEP = _ARCHIVE / "t4-sweep-2026-09-15.json"

# Leave headroom for future rows without allowing under-prediction to pass unnoticed.
_MAX_ABS_ACTIVATION_ERROR_PCT = 8.0
_MAX_MEAN_ACTIVATION_ERROR_PCT = 2.0


def _rows() -> list[dict]:
    return json.loads(_SWEEP.read_text(encoding="utf-8"))["runs"]


def _training(run: dict) -> TrainingConfig:
    return TrainingConfig(
        precision=run["precision"],
        quantization=run["quantization"],
        double_quant=run["double_quant"],
        optimizer=run["optimizer"],
        lora_rank=run["lora_rank"],
        lora_targets=list(run["lora_targets"]),
        batch_size=run["batch_size"],
        seq_len=run["seq_len"],
        grad_checkpoint=run["grad_checkpoint"],
        flash_attn=run["kernel"] == "flash",
    )


def test_sweep_archive_is_one_session() -> None:
    """One CUDA context across every row is what makes the C_overhead fit meaningful."""
    rows = _rows()
    assert len(rows) == 40
    assert {r["measured"]["cuda_context_mib"] for r in rows} == {140.875}
    assert {r["run"]["gpu_key"] for r in rows} == {"t4"}
    assert {r["run"]["quantization"] for r in rows} == {"none", "nf4"}


def test_activation_formula_reproduces_every_measured_row() -> None:
    errors: list[tuple[str, float]] = []

    for row in _rows():
        run = row["run"]
        predicted = estimate(
            ModelConfig(**row["model_config"]), _training(run), get_gpu(run["gpu_key"])
        ).activation_mib
        measured = row["measured"]["activation_mib"]
        label = (
            f"{row['model_id']} {run['quantization']}/{run['kernel']} "
            f"bs{run['batch_size']} seq{run['seq_len']}"
        )
        errors.append((label, 100.0 * (predicted - measured) / measured))

    worst = max(errors, key=lambda pair: abs(pair[1]))
    mean = sum(abs(error) for _, error in errors) / len(errors)

    assert abs(worst[1]) < _MAX_ABS_ACTIVATION_ERROR_PCT, (
        f"worst A_act error {worst[1]:+.1f}% on {worst[0]}"
    )
    assert mean < _MAX_MEAN_ACTIVATION_ERROR_PCT, f"mean A_act error {mean:.1f}%"


def test_quantized_rows_did_not_move_when_the_profiles_landed() -> None:
    """nf4 kept its constants, so every archived prediction must reproduce exactly."""
    for row in _rows():
        run = row["run"]
        if run["quantization"] != "nf4":
            continue
        predicted = estimate(
            ModelConfig(**row["model_config"]), _training(run), get_gpu(run["gpu_key"])
        )
        assert predicted.activation_mib == pytest.approx(
            row["predicted"]["activation_mib"], rel=1e-9
        )


def test_unquantized_rows_are_the_half_that_moved() -> None:
    """The unquantized profile must produce lower activation estimates than nf4."""
    from dataclasses import replace

    for row in _rows():
        run = row["run"]
        if run["quantization"] != "none":
            continue
        config = ModelConfig(**row["model_config"])
        gpu = get_gpu(run["gpu_key"])
        training = _training(run)
        as_quantized = estimate(config, replace(training, quantization="nf4"), gpu)
        assert as_quantized.activation_mib > estimate(
            config, training, gpu
        ).activation_mib


_PHASE3 = _ARCHIVE / "t4-phase3-2026-09-16.json"

# Under-prediction is the tighter budget because it risks a real OOM.
_MAX_UNDER_PREDICTION_PCT = 8.0
_MAX_OVER_PREDICTION_PCT = 14.0
_MAX_MEAN_PROCESS_ERROR_PCT = 5.0


def _all_overhead_rows() -> list[dict]:
    """Every archived row that can score the C_overhead profiles.

    Needs `model_config` (to rebuild the estimate offline) and gradient checkpointing
    (the hump form only exists under a `max()`, which only checkpointing creates).
    """
    rows: list[dict] = []
    for path in (_SWEEP, _PHASE3):
        for row in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if "model_config" in row and row["run"].get("grad_checkpoint"):
                rows.append(row)
    return rows


def _process_error_pct(row: dict) -> float:
    run = row["run"]
    predicted = estimate(
        ModelConfig(**row["model_config"]), _training(run), get_gpu(run["gpu_key"])
    )
    measured = row["measured"]["peak_reserved_mib"] + row["measured"]["cuda_context_mib"]
    return 100.0 * (predicted.total_mib - measured) / measured


def test_the_shipped_profiles_never_under_predict_a_measured_row_by_more_than_8pct() -> None:
    """The OOM direction. This is the gate that actually protects a user.

    An over-prediction tells someone their config will not fit when it would have.
    An under-prediction tells them it will fit, and then training dies part way in.
    """
    worst = min(_process_error_pct(row) for row in _all_overhead_rows())

    assert worst > -_MAX_UNDER_PREDICTION_PCT, (
        f"a measured row is under-predicted by {worst:.1f}%, past the "
        f"{_MAX_UNDER_PREDICTION_PCT}% budget -- some F is too low"
    )


def test_the_shipped_profiles_reproduce_every_measured_row() -> None:
    errors = [(_process_error_pct(row), row) for row in _all_overhead_rows()]
    assert len(errors) >= 49

    worst_over, over_row = max(errors, key=lambda pair: pair[0])
    mean = sum(abs(error) for error, _ in errors) / len(errors)

    assert worst_over < _MAX_OVER_PREDICTION_PCT, (
        f"{over_row['model_id']} {over_row['run']['quantization']}/"
        f"{over_row['run']['kernel']} over-predicted by {worst_over:.1f}%"
    )
    assert mean < _MAX_MEAN_PROCESS_ERROR_PCT


def test_the_t4_profiles_beat_the_uncalibrated_default() -> None:
    """Shipping constants has to be better than not shipping them, row by row.

    The default bills 500 MiB of context (the real figure is 140.875) plus a flat 5%
    of W_base + A_act, which has nothing to do with how the allocator actually behaves.
    """
    from fitcheck.memory.overhead import estimate_overhead
    from fitcheck.overhead_db import DEFAULT_OVERHEAD_PROFILE

    rows = _all_overhead_rows()
    calibrated = [abs(_process_error_pct(row)) for row in rows]

    uncalibrated = []
    for row in rows:
        run = row["run"]
        predicted = estimate(
            ModelConfig(**row["model_config"]), _training(run), get_gpu(run["gpu_key"])
        )
        tensors = predicted.total_mib - predicted.overhead_mib
        default_total = tensors + estimate_overhead(
            predicted.weight_mib,
            predicted.activation_mib,
            DEFAULT_OVERHEAD_PROFILE,
            run["seq_len"],
        )
        measured = (
            row["measured"]["peak_reserved_mib"] + row["measured"]["cuda_context_mib"]
        )
        uncalibrated.append(abs(100.0 * (default_total - measured) / measured))

    assert max(calibrated) < max(uncalibrated)
    assert sum(calibrated) / len(calibrated) < sum(uncalibrated) / len(uncalibrated)
