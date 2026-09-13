# Component 6
from __future__ import annotations

from math import log2

from fitcheck.overhead_db import (
    DEFAULT_OVERHEAD_PROFILE,
    REFERENCE_SEQ_LEN,
    OverheadProfile,
)

_BASE_CONTEXT_MIB = DEFAULT_OVERHEAD_PROFILE.base_context_mib
_FRAGMENTATION_FRACTION = DEFAULT_OVERHEAD_PROFILE.fragmentation


def _validate_memory(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _validate_seq_len(seq_len: int | None) -> int | None:
    if seq_len is None:
        return None
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer or None")
    return seq_len


def _validate_profile(profile: OverheadProfile | None) -> OverheadProfile:
    if profile is None:
        return DEFAULT_OVERHEAD_PROFILE
    if not isinstance(profile, OverheadProfile):
        raise ValueError("profile must be an OverheadProfile or None")
    return profile


def _fragmentation_at(profile: OverheadProfile, seq_len: int | None) -> float:
    fragmentation = profile.fragmentation
    if seq_len is not None and profile.fragmentation_per_octave:
        in_range = min(max(seq_len, profile.seq_len_min), profile.seq_len_max)
        fragmentation += profile.fragmentation_per_octave * log2(
            in_range / REFERENCE_SEQ_LEN
        )
    return max(fragmentation, 0.0)


def estimate_overhead(
    weight_memory: float,
    activation_memory: float,
    profile: OverheadProfile | None = None,
    seq_len: int | None = None,
) -> float:
    base_weights = _validate_memory(weight_memory, "weight_memory")
    activations = _validate_memory(activation_memory, "activation_memory")
    resolved_profile = _validate_profile(profile)
    resolved_seq_len = _validate_seq_len(seq_len)

    fragmentation = _fragmentation_at(resolved_profile, resolved_seq_len)

    return resolved_profile.base_context_mib + fragmentation * (
        base_weights + activations
    )
