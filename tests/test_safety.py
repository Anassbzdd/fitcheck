from __future__ import annotations

import json
from pathlib import Path

from fitcheck.config_parser import ModelConfig
from fitcheck.estimator import TrainingConfig, estimate
from fitcheck.gpu_db import get_gpu
from fitcheck.safety import (
    conservative_reserve_mib,
    final_validation_scope_matches,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_final_validation_reserves_are_scoped_to_the_measured_profiles() -> None:
    assert conservative_reserve_mib("t4", False, "nf4") == 701.0
    assert conservative_reserve_mib("t4", True, "nf4") == 2_264.0
    assert conservative_reserve_mib("4090", True, "nf4") is None
    assert conservative_reserve_mib("t4", True, "none") is None
    assert final_validation_scope_matches(
        precision="fp16",
        quantization="nf4",
        double_quant=False,
        optimizer="adamw",
        optimizer_dtype="fp32",
        lora_rank=16,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        grad_checkpoint=True,
        seq_len=1024,
    )
    assert not final_validation_scope_matches(
        precision="bf16",
        quantization="nf4",
        double_quant=False,
        optimizer="adamw",
        optimizer_dtype="fp32",
        lora_rank=16,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        grad_checkpoint=True,
        seq_len=1024,
    )


def test_a_near_capacity_t4_estimate_is_uncertain_when_the_reserve_does_not_fit() -> None:
    model = ModelConfig(
        name="Llama-3.2-3B-Instruct",
        num_params=3_212_749_824,
        hidden_size=3072,
        num_layers=28,
        num_attention_heads=24,
        num_kv_heads=8,
        intermediate_size=8192,
        vocab_size=128256,
        head_dim=128,
        tie_word_embeddings=True,
    )
    training = TrainingConfig(
        precision="fp16",
        quantization="nf4",
        batch_size=4,
        seq_len=1024,
        lora_rank=16,
        grad_checkpoint=True,
        flash_attn=True,
    )

    report = estimate(model, training, get_gpu("t4"))

    assert report.fits is True
    assert report.safe_fits is False
    assert report.uncertain is True
    assert report.verdict == "uncertain"
    assert report.recommendation_basis == "validated_reserve"
    assert report.safe_total_mib is not None
    assert report.safe_total_mib > report.gpu_capacity_mib
    assert report.recommended_batch_size == 2
    assert report.recommended_batch_size < report.max_batch_size


def test_boundary_validation_preserves_six_safe_and_six_unsafe_rows() -> None:
    path = _ROOT / "data" / "measurements" / "validation" / "final-t4-20260921-verdicts.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"]

    assert payload["provenance"]["verdict_basis"] == (
        "point estimate only; the product's conservative reserve may mark a "
        "point-safe case uncertain"
    )
    assert len(rows) == 12
    assert sum(row["identity"]["role"] == "ceiling-fit" for row in rows) == 6
    assert sum(row["identity"]["role"] == "ceiling-oom" for row in rows) == 6
    assert all(
        row["predicted_fits"] is True
        for row in rows
        if row["identity"]["role"] == "ceiling-fit"
    )
    assert all(
        row["predicted_fits"] is False
        for row in rows
        if row["identity"]["role"] == "ceiling-oom"
    )
    assert all(
        row["within_safety_budget"] is True
        for row in rows
        if row["identity"]["role"] == "ceiling-fit"
    )
    assert all(
        row["outcome"] == "oom" or row["within_safety_budget"] is False
        for row in rows
        if row["identity"]["role"] == "ceiling-oom"
    )
