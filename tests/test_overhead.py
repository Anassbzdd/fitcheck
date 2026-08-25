from __future__ import annotations
import inspect
import pytest
from fitcheck.memory.overhead import estimate_overhead

_GOLDEN_W_BASE_MIB = 4_068.45
_GOLDEN_A_ACT_MIB = 3_136.0
_GOLDEN_C_OVERHEAD_MIB = 860.2225


def test_estimate_overhead_golden_qlora_llama_31_8b() -> None:
    result = estimate_overhead(_GOLDEN_W_BASE_MIB, _GOLDEN_A_ACT_MIB)

    assert result == pytest.approx(_GOLDEN_C_OVERHEAD_MIB, rel=1e-9)
    assert round(result, 2) == pytest.approx(860.22, rel=1e-9)


def test_estimate_overhead_is_base_plus_five_percent() -> None:
    result = estimate_overhead(1_000.0, 3_000.0)

    assert result == pytest.approx(500.0 + 0.05 * 4_000.0, rel=1e-9)


def test_estimate_overhead_floor_is_the_cuda_context() -> None:
    assert estimate_overhead(0.0, 0.0) == pytest.approx(500.0, rel=1e-9)


def test_estimate_overhead_depends_on_the_sum_not_the_split() -> None:
    weights_heavy = estimate_overhead(4_000.0, 100.0)
    activations_heavy = estimate_overhead(100.0, 4_000.0)

    assert weights_heavy == pytest.approx(activations_heavy, rel=1e-9)


def test_estimate_overhead_grows_with_each_argument() -> None:
    baseline = estimate_overhead(1_000.0, 1_000.0)

    assert estimate_overhead(2_000.0, 1_000.0) == pytest.approx(baseline + 50.0, rel=1e-9)
    assert estimate_overhead(1_000.0, 2_000.0) == pytest.approx(baseline + 50.0, rel=1e-9)


def test_estimate_overhead_takes_only_base_weights_and_activations() -> None:
    parameters = inspect.signature(estimate_overhead).parameters

    assert list(parameters) == ["weight_memory", "activation_memory"]


@pytest.mark.parametrize("bad_value", [-1.0, True, "500", None])
def test_estimate_overhead_rejects_invalid_weight_memory(bad_value: object) -> None:
    with pytest.raises(ValueError, match="weight_memory must be a non-negative number"):
        estimate_overhead(bad_value, _GOLDEN_A_ACT_MIB)


@pytest.mark.parametrize("bad_value", [-1.0, True, "500", None])
def test_estimate_overhead_rejects_invalid_activation_memory(bad_value: object) -> None:
    with pytest.raises(
        ValueError, match="activation_memory must be a non-negative number"
    ):
        estimate_overhead(_GOLDEN_W_BASE_MIB, bad_value)
