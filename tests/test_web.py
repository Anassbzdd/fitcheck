from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fitcheck import web
from fitcheck.advisor import SweepSpec, advise
from fitcheck.config_parser import fetch_model_config
from fitcheck.estimator import ServingConfig, estimate_inference
from fitcheck.gpu_db import get_gpu

_MODEL_ID = "meta-llama/Llama-3.1-8B"


def _install_model(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)


def test_training_request_preserves_golden_estimator_values(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    _install_model(fake_config_download, llama_31_8b_config)

    result = web.estimate_training_request(
        _MODEL_ID,
        "4090",
        batch_size=4,
        seq_len=2048,
        quantization="nf4",
        precision="bf16",
        lora_rank=64,
        target_preset="standard",
        grad_checkpoint=True,
        flash_attn=True,
    )

    assert result.report.weight_mib == pytest.approx(7_753.02, abs=0.01)
    assert result.report.lora_mib == pytest.approx(208.0)
    assert result.report.optimizer_mib == pytest.approx(416.0)
    assert result.report.gradient_mib == pytest.approx(208.0)
    assert result.report.activation_mib == pytest.approx(20_128.0)
    assert result.report.overhead_mib == pytest.approx(1_894.05, abs=0.01)
    assert result.report.total_mib == pytest.approx(30_607.07, abs=0.01)
    assert result.report.verdict == "does_not_fit"
    assert result.report.max_batch_size == 2
    assert result.activation_details["logits_mib"] == pytest.approx(16_032.0)


def test_custom_vram_overrides_preset_at_exactly_95_percent(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    _install_model(fake_config_download, llama_31_8b_config)

    result = web.estimate_training_request(
        _MODEL_ID, "4090", custom_vram_mib=12_345
    )

    assert result.gpu.name == "Custom GPU"
    assert result.gpu.vram_mib == 12_345
    assert result.report.gpu_capacity_mib == 12_345 * 95 // 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 1.5),
        ("seq_len", 2048.25),
        ("lora_rank", 16.5),
        ("grad_accum_steps", 2.1),
        ("custom_vram_mib", 24_576.5),
    ],
)
def test_training_rejects_fractional_web_numbers(
    monkeypatch: pytest.MonkeyPatch, field: str, value: float
) -> None:
    monkeypatch.setattr(
        web,
        "fetch_model_config",
        lambda _: pytest.fail("numeric validation must run before model fetch"),
    )

    with pytest.raises(ValueError, match="positive whole number"):
        web.estimate_training_request(_MODEL_ID, "4090", **{field: value})


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "2"])
def test_training_rejects_invalid_web_numbers(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setattr(
        web,
        "fetch_model_config",
        lambda _: pytest.fail("numeric validation must run before model fetch"),
    )

    with pytest.raises(ValueError, match="positive whole number"):
        web.estimate_training_request(_MODEL_ID, "4090", batch_size=value)


def test_serving_request_preserves_existing_report_values(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    _install_model(fake_config_download, llama_31_8b_config)
    model = fetch_model_config(_MODEL_ID)
    serving = ServingConfig(
        precision="fp16",
        quantization="nf4",
        double_quant=False,
        seq_len=2048,
        num_concurrent=4,
    )
    expected = estimate_inference(model, serving, get_gpu("4090"))

    result = web.estimate_serving_request(
        _MODEL_ID,
        "4090",
        seq_len=2048,
        num_concurrent=4,
        quantization="nf4",
    )

    assert result.report == expected
    assert result.report.kv_mib_per_request > 0
    assert result.report.kv_mib_per_token > 0


def test_serving_rejects_fractional_concurrency() -> None:
    with pytest.raises(ValueError, match="positive whole number"):
        web.estimate_serving_request(_MODEL_ID, "4090", num_concurrent=1.25)


def test_advisor_request_calls_existing_advisor_and_returns_its_report(
    monkeypatch: pytest.MonkeyPatch,
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    _install_model(fake_config_download, llama_31_8b_config)
    original_advise = advise
    calls: list[tuple[SweepSpec, str | None]] = []

    def observed_advise(*args: Any, **kwargs: Any) -> Any:
        calls.append((args[3], kwargs.get("model_id")))
        return original_advise(*args, **kwargs)

    monkeypatch.setattr(web, "advise", observed_advise)
    result = web.advise_training_request(
        _MODEL_ID,
        "4090",
        batch_sizes=[1, 2],
        seq_lens=[512, 1024],
        lora_ranks=[8, 16],
        max_context_length=2048,
    )

    assert calls == [(SweepSpec((1, 2), (512, 1024), (8, 16)), _MODEL_ID)]
    assert result.report.grid_size == 8
    assert result.report.fitting_count > 0
    assert result.report.frontier


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantization": "none", "double_quant": True}, "double_quant"),
        (
            {"quantization": "nf4", "full_finetuning": True},
            "does not model full fine-tuning",
        ),
    ],
)
def test_training_rejects_unsupported_combinations(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_model(fake_config_download, llama_31_8b_config)

    with pytest.raises(ValueError, match=message):
        web.estimate_training_request(_MODEL_ID, "4090", **kwargs)


def test_advisor_rejects_context_lengths_above_known_limit() -> None:
    with pytest.raises(ValueError, match="cannot be trained"):
        web.advise_training_request(
            _MODEL_ID,
            "4090",
            batch_sizes=[1],
            seq_lens=[1024, 4096],
            lora_ranks=[8],
            max_context_length=2048,
        )


def test_advisor_rejects_fractional_sweep_values() -> None:
    with pytest.raises(ValueError, match="positive whole number"):
        web.advise_training_request(
            _MODEL_ID,
            "4090",
            batch_sizes=[1, 2.5],
            seq_lens=[1024],
            lora_ranks=[8],
        )
