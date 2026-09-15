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

# The worst single row of the 2026-09-15 sweep is SmolLM2-360M nf4 eager seq 4096 at
# -4.8%. 8% leaves room for a row that is added later without loosening the guard to
# the point where a real regression slips through -- the pre-correction worst was +32%.
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
    """Guards the direction of the correction, not just its size.

    Billing the `none` rows with the nf4 profile is what fitcheck used to do. Every one
    of them must come out HIGHER that way -- if this ever passes with the two profiles
    swapped, the table in _PROFILES has been transposed.
    """
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
