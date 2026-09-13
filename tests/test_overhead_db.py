from __future__ import annotations

import pytest

from fitcheck.gpu_db import GPU_DB, GpuSpec, get_gpu, gpu_key_for
from fitcheck.overhead_db import (
    DEFAULT_OVERHEAD_PROFILE,
    KERNEL_EAGER,
    KERNEL_FLASH,
    OVERHEAD_DB,
    OverheadProfile,
    get_overhead_profile,
    kernel_name,
    list_overhead_profiles,
)


def test_the_default_profile_is_the_pre_calibration_constant_pair() -> None:
    """Changing these silently would move every uncalibrated card's estimate."""
    assert DEFAULT_OVERHEAD_PROFILE.base_context_mib == 500.0
    assert DEFAULT_OVERHEAD_PROFILE.fragmentation == 0.05
    assert DEFAULT_OVERHEAD_PROFILE.fragmentation_per_octave == 0.0
    assert not DEFAULT_OVERHEAD_PROFILE.measured


def test_kernel_name_splits_the_way_the_activation_formula_does() -> None:
    assert kernel_name(False) == KERNEL_EAGER
    assert kernel_name(True) == KERNEL_FLASH


@pytest.mark.parametrize("bad_value", [None, "flash", 1, 0])
def test_kernel_name_rejects_a_non_boolean(bad_value: object) -> None:
    with pytest.raises(ValueError, match="flash_attn must be a boolean"):
        kernel_name(bad_value)


def test_an_unmeasured_card_falls_back_to_the_default() -> None:
    assert get_overhead_profile("4090", False) is DEFAULT_OVERHEAD_PROFILE
    assert get_overhead_profile("4090", True) is DEFAULT_OVERHEAD_PROFILE


def test_a_custom_card_has_no_key_and_so_no_profile() -> None:
    assert get_overhead_profile(None) is DEFAULT_OVERHEAD_PROFILE


def test_an_unknown_key_is_a_missing_fit_not_an_error() -> None:
    """A card fitcheck has never heard of still gets an estimate, just not a fitted one."""
    assert get_overhead_profile("no-such-gpu", False) is DEFAULT_OVERHEAD_PROFILE


def test_lookup_is_case_and_whitespace_insensitive_like_get_gpu() -> None:
    assert get_overhead_profile("  T4  ", False) == get_overhead_profile("t4", False)


@pytest.mark.parametrize("bad_value", [1, 2.0, object()])
def test_get_overhead_profile_rejects_a_non_string_key(bad_value: object) -> None:
    with pytest.raises(ValueError, match="gpu_key must be a string or None"):
        get_overhead_profile(bad_value)


def test_every_shipped_profile_is_keyed_by_a_real_gpu_and_kernel() -> None:
    for (gpu_key, kernel), profile in list_overhead_profiles():
        assert gpu_key in GPU_DB, f"{gpu_key} is not a GPU_DB key"
        assert kernel in (KERNEL_EAGER, KERNEL_FLASH)
        assert isinstance(profile, OverheadProfile)


def test_every_shipped_profile_is_physical_and_says_what_it_came_from() -> None:
    for _, profile in list_overhead_profiles():
        assert profile.base_context_mib >= 0.0
        assert profile.fragmentation >= 0.0
        assert profile.runs > 0, "a profile with no runs behind it is not a fit"
        assert profile.source.strip(), "a fitted profile must record its provenance"
        assert profile.seq_len_min <= profile.seq_len_max


def test_the_db_is_empty_until_the_sweep_lands() -> None:
    """Task 9.3's remaining half. Delete this test when profiles ship.

    It is here so that populating OVERHEAD_DB is a deliberate act with a test to
    update, rather than something that slips in half-finished.
    """
    assert OVERHEAD_DB == {}


# ---------------------------------------------------------------------------------
# gpu_key_for: the reverse lookup the estimator needs
# ---------------------------------------------------------------------------------


def test_gpu_key_for_finds_the_key_of_every_database_card() -> None:
    for key, spec in GPU_DB.items():
        found = gpu_key_for(spec)
        assert found is not None
        # `h100` and `h100-80` are the same card under two names, so the reverse
        # lookup can only return one of them -- and either is correct.
        assert GPU_DB[found] == spec


def test_gpu_key_for_round_trips_through_get_gpu() -> None:
    assert gpu_key_for(get_gpu("t4")) == "t4"
    assert gpu_key_for(get_gpu("4090")) == "4090"


def test_gpu_key_for_a_custom_card_is_none() -> None:
    assert gpu_key_for(get_gpu("My Card", vram_mib=20_000)) is None


def test_gpu_key_for_rejects_a_non_spec() -> None:
    with pytest.raises(ValueError, match="spec must be a GpuSpec"):
        gpu_key_for("t4")


def test_a_lookalike_spec_still_resolves_by_value() -> None:
    clone = GpuSpec("Tesla T4", 14_912, 14_000)

    assert gpu_key_for(clone) == "t4"
