from __future__ import annotations

import pytest
from fitcheck.gpu_db import GPU_DB, GpuSpec, get_gpu, gpu_key_for
from fitcheck.overhead_db import (
    DEFAULT_OVERHEAD_PROFILE,
    KERNEL_EAGER,
    KERNEL_FLASH,
    OVERHEAD_DB,
    QUANTIZATIONS,
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


def test_every_shipped_profile_is_keyed_by_a_real_gpu_kernel_and_quant() -> None:
    for (gpu_key, kernel, quantization), profile in list_overhead_profiles():
        assert gpu_key in GPU_DB, f"{gpu_key} is not a GPU_DB key"
        assert kernel in (KERNEL_EAGER, KERNEL_FLASH)
        assert quantization in QUANTIZATIONS
        assert isinstance(profile, OverheadProfile)
        assert profile.kernel == kernel
        assert profile.quantization == quantization


def test_every_shipped_profile_is_physical_and_says_what_it_came_from() -> None:
    for _, profile in list_overhead_profiles():
        assert profile.base_context_mib >= 0.0
        assert profile.fragmentation >= 0.0
        assert profile.runs > 0, "a profile with no runs behind it is not a fit"
        assert profile.source.strip(), "a fitted profile must record its provenance"
        assert profile.seq_len_min <= profile.seq_len_max


def test_the_shipped_profiles_are_the_t4_calibration_of_task_9_3() -> None:
    """Quantization is part of the profile key because fragmentation differs by storage format."""
    assert set(OVERHEAD_DB) == {
        ("t4", "flash", "none"),
        ("t4", "flash", "nf4"),
        ("t4", "eager", "none"),
        ("t4", "eager", "nf4"),
    }
    for profile in OVERHEAD_DB.values():
        assert profile.uses_hump_form
        assert profile.base_context_mib == 140.875
        assert profile.fragmentation_per_octave == 0.0


def test_the_flash_profiles_have_no_layer_win_coefficient() -> None:
    """`A_layer` loses the max() in every flash row ever measured.

    A layer-win under Flash Attention needs roughly `vocab < 3.75 * hidden_size`.
    No measured model is that shape, so the coefficient is honestly absent rather
    than guessed, and `fragmentation_for` falls back to the logits-win value.
    """
    for kernel, expected in (("flash", None), ("eager", float)):
        for quantization in ("none", "nf4"):
            profile = OVERHEAD_DB[("t4", kernel, quantization)]
            if expected is None:
                assert profile.hump_fragmentation_layer_win is None
            else:
                assert isinstance(profile.hump_fragmentation_layer_win, float)


def test_quantization_is_part_of_the_lookup() -> None:
    """The whole point of the 3-part key: same card, same kernel, different F."""
    none = get_overhead_profile("t4", True, "none")
    nf4 = get_overhead_profile("t4", True, "nf4")

    assert none is not nf4
    assert none.hump_fragmentation_logits_win != nf4.hump_fragmentation_logits_win


def test_an_unmeasured_quantization_falls_back_to_the_default() -> None:
    """`int8` has no measured row at all, so it must not borrow nf4's constants."""
    assert get_overhead_profile("t4", True, "int8") is DEFAULT_OVERHEAD_PROFILE
    assert get_overhead_profile("t4", False, "int8") is DEFAULT_OVERHEAD_PROFILE


def test_gpu_key_for_finds_the_key_of_every_database_card() -> None:
    for spec in GPU_DB.values():
        found = gpu_key_for(spec)
        assert found is not None
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
