from __future__ import annotations
import inspect
import pytest
from fitcheck.memory.overhead import estimate_overhead
from fitcheck.overhead_db import (
    DEFAULT_OVERHEAD_PROFILE,
    REFERENCE_SEQ_LEN,
    OverheadProfile,
)

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

    assert estimate_overhead(2_000.0, 1_000.0) == estimate_overhead(1_000.0, 2_000.0)
    assert estimate_overhead(2_000.0, 1_000.0) > baseline


def test_estimate_overhead_takes_the_two_sizes_then_the_calibration() -> None:
    parameters = inspect.signature(estimate_overhead).parameters

    assert list(parameters) == [
        "weight_memory",
        "activation_memory",
        "profile",
        "seq_len",
    ]


def test_estimate_overhead_without_a_profile_is_the_default_profile() -> None:
    explicit = estimate_overhead(
        1_000.0, 3_000.0, DEFAULT_OVERHEAD_PROFILE, seq_len=2048
    )

    assert explicit == pytest.approx(estimate_overhead(1_000.0, 3_000.0), rel=1e-12)


def test_seq_len_does_not_move_an_uncalibrated_card() -> None:
    """The default profile has no sequence slope, so 9.3 changed nothing for it."""
    flat = estimate_overhead(1_000.0, 3_000.0)

    for seq_len in (128, 512, 2048, 8192, 131_072):
        assert estimate_overhead(1_000.0, 3_000.0, seq_len=seq_len) == pytest.approx(
            flat, rel=1e-12
        )


# ---------------------------------------------------------------------------------
# Fitted profiles
# ---------------------------------------------------------------------------------


def _profile(**overrides: object) -> OverheadProfile:
    fields = dict(
        gpu="Test GPU",
        kernel="eager",
        base_context_mib=140.0,
        fragmentation=0.20,
        fragmentation_per_octave=0.0,
        seq_len_min=512,
        seq_len_max=4096,
        runs=8,
    )
    fields.update(overrides)
    return OverheadProfile(**fields)  # type: ignore[arg-type]


def test_a_fitted_profile_replaces_both_constants() -> None:
    result = estimate_overhead(1_000.0, 3_000.0, _profile())

    assert result == pytest.approx(140.0 + 0.20 * 4_000.0, rel=1e-9)


def test_fragmentation_slope_is_per_octave_of_sequence() -> None:
    profile = _profile(fragmentation_per_octave=0.04)

    at_reference = estimate_overhead(1_000.0, 3_000.0, profile, REFERENCE_SEQ_LEN)
    at_double = estimate_overhead(1_000.0, 3_000.0, profile, 2 * REFERENCE_SEQ_LEN)
    at_half = estimate_overhead(1_000.0, 3_000.0, profile, REFERENCE_SEQ_LEN // 2)

    assert at_reference == pytest.approx(140.0 + 0.20 * 4_000.0, rel=1e-9)
    assert at_double == pytest.approx(140.0 + 0.24 * 4_000.0, rel=1e-9)
    assert at_half == pytest.approx(140.0 + 0.16 * 4_000.0, rel=1e-9)


def test_the_slope_is_not_extrapolated_past_the_calibrated_range() -> None:
    """A trend measured over 512-4096 tokens is evidence about 512-4096 tokens."""
    profile = _profile(fragmentation_per_octave=0.04, seq_len_min=512, seq_len_max=4096)

    at_top = estimate_overhead(1_000.0, 3_000.0, profile, 4_096)
    at_bottom = estimate_overhead(1_000.0, 3_000.0, profile, 512)

    assert estimate_overhead(1_000.0, 3_000.0, profile, 131_072) == pytest.approx(
        at_top, rel=1e-12
    )
    assert estimate_overhead(1_000.0, 3_000.0, profile, 64) == pytest.approx(
        at_bottom, rel=1e-12
    )


def test_fragmentation_never_goes_negative() -> None:
    profile = _profile(
        fragmentation=0.02, fragmentation_per_octave=0.5, seq_len_min=64, seq_len_max=4096
    )

    assert estimate_overhead(1_000.0, 3_000.0, profile, 64) == pytest.approx(
        profile.base_context_mib, rel=1e-12
    )


def test_the_slope_is_ignored_when_no_seq_len_is_given() -> None:
    profile = _profile(fragmentation_per_octave=0.04)

    assert estimate_overhead(1_000.0, 3_000.0, profile) == pytest.approx(
        estimate_overhead(1_000.0, 3_000.0, profile, REFERENCE_SEQ_LEN), rel=1e-12
    )


# ---------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------


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


@pytest.mark.parametrize("bad_value", [0, -1, True, "2048", 2048.0])
def test_estimate_overhead_rejects_invalid_seq_len(bad_value: object) -> None:
    with pytest.raises(ValueError, match="seq_len must be a positive integer or None"):
        estimate_overhead(1_000.0, 1_000.0, _profile(), bad_value)


@pytest.mark.parametrize("bad_value", ["t4", 500.0, object()])
def test_estimate_overhead_rejects_a_non_profile(bad_value: object) -> None:
    with pytest.raises(ValueError, match="profile must be an OverheadProfile or None"):
        estimate_overhead(1_000.0, 1_000.0, bad_value)
