from __future__ import annotations
import inspect
import pytest
from fitcheck.memory.gradients import estimate_gradient_memory
from fitcheck.utils import bytes_to_mib

_MIB = 1024**2

_GOLDEN_LORA_PARAMS = 54_525_952
_GOLDEN_G_GRAD_MIB = 104.0

_LLAMA_31_8B_PARAMS = 8_030_261_248


def test_estimate_gradient_memory_golden_qlora_bf16() -> None:
    result = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, "bf16")

    assert result == pytest.approx(_GOLDEN_G_GRAD_MIB, rel=1e-9)
    assert result == pytest.approx(_GOLDEN_LORA_PARAMS * 2 / _MIB, rel=1e-9)


@pytest.mark.parametrize(
    ("precision", "bytes_per_param"),
    [("fp32", 4), ("fp16", 2), ("bf16", 2)],
)
def test_estimate_gradient_memory_precision_table(
    precision: str, bytes_per_param: int
) -> None:
    result = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, precision)

    expected = bytes_to_mib(_GOLDEN_LORA_PARAMS * bytes_per_param)
    assert result == pytest.approx(expected, rel=1e-9)


def test_estimate_gradient_memory_fp32_is_exactly_double_bf16() -> None:
    bf16 = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, "bf16")
    fp32 = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, "fp32")

    assert fp32 == pytest.approx(bf16 * 2, rel=1e-9)
    assert fp32 == pytest.approx(208.0, rel=1e-9)


def test_estimate_gradient_memory_is_not_hardcoded_to_two_bytes() -> None:
    fp32 = estimate_gradient_memory(1_048_576, "fp32")

    assert fp32 == pytest.approx(4.0, rel=1e-9)
    assert fp32 != pytest.approx(2.0, rel=1e-9)


def test_estimate_gradient_memory_full_ft_llama_8b_bf16() -> None:
    result = estimate_gradient_memory(_LLAMA_31_8B_PARAMS, "bf16")

    expected = bytes_to_mib(_LLAMA_31_8B_PARAMS * 2)
    assert result == pytest.approx(expected, rel=1e-9)
    assert result == pytest.approx(15_316.94, rel=1e-4)


def test_estimate_gradient_memory_takes_no_accumulation_argument() -> None:
    parameters = inspect.signature(estimate_gradient_memory).parameters

    assert set(parameters) == {"trainable_params", "precision"}
    assert not any("accum" in name for name in parameters)


def test_estimate_gradient_memory_only_counts_trainable_params() -> None:
    adapters = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, "bf16")
    whole_model = estimate_gradient_memory(_LLAMA_31_8B_PARAMS, "bf16")

    assert adapters < whole_model / 100


def test_estimate_gradient_memory_mirrors_lora_weight_memory_at_same_precision() -> None:
    gradients = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, "bf16")

    assert gradients == pytest.approx(_GOLDEN_G_GRAD_MIB, rel=1e-9)


def test_estimate_gradient_memory_scales_linearly_with_trainable_params() -> None:
    small = estimate_gradient_memory(1_000_000, "bf16")
    large = estimate_gradient_memory(4_000_000, "bf16")

    assert large == pytest.approx(small * 4, rel=1e-9)


@pytest.mark.parametrize("precision", ["BF16", "  bf16  ", "Bf16"])
def test_estimate_gradient_memory_precision_is_normalized(precision: str) -> None:
    result = estimate_gradient_memory(_GOLDEN_LORA_PARAMS, precision)

    assert result == pytest.approx(_GOLDEN_G_GRAD_MIB, rel=1e-9)


def test_estimate_gradient_memory_returns_mib_not_mb() -> None:
    result = estimate_gradient_memory(1_048_576, "bf16")

    assert result == pytest.approx(2.0, rel=1e-9)
    assert result != pytest.approx(1_048_576 * 2 / 1_000_000, rel=1e-3)


@pytest.mark.parametrize("bad_params", [0, -1, True, 12.5, "1000", None])
def test_estimate_gradient_memory_rejects_invalid_trainable_params(
    bad_params: object,
) -> None:
    with pytest.raises(ValueError, match="trainable_params must be a positive integer"):
        estimate_gradient_memory(bad_params, "bf16")


def test_estimate_gradient_memory_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        estimate_gradient_memory(1_000_000, "fp4")


def test_estimate_gradient_memory_rejects_non_string_precision() -> None:
    with pytest.raises(ValueError, match="precision must be a string"):
        estimate_gradient_memory(1_000_000, 2)
