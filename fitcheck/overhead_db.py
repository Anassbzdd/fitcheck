from __future__ import annotations

from dataclasses import dataclass

REFERENCE_SEQ_LEN = 2048

KERNEL_EAGER = "eager"
KERNEL_FLASH = "flash"


@dataclass(frozen=True)
class OverheadProfile:
    gpu: str
    kernel: str
    base_context_mib: float
    fragmentation: float
    fragmentation_per_octave: float = 0.0
    seq_len_min: int = REFERENCE_SEQ_LEN
    seq_len_max: int = REFERENCE_SEQ_LEN
    runs: int = 0
    worst_over_pct: float = 0.0
    worst_under_pct: float = 0.0
    source: str = "unmeasured default"

    @property
    def measured(self) -> bool:
        return self.runs > 0

DEFAULT_OVERHEAD_PROFILE = OverheadProfile(
    gpu="any",
    kernel="any",
    base_context_mib=500.0,
    fragmentation=0.05,
    fragmentation_per_octave=0.0,
    runs=0,
    source="unmeasured conservative default (pre-9.3 constants)",
)

OVERHEAD_DB: dict[tuple[str, str], OverheadProfile] = {}


def kernel_name(flash_attn: bool) -> str:
    if not isinstance(flash_attn, bool):
        raise ValueError("flash_attn must be a boolean")
    return KERNEL_FLASH if flash_attn else KERNEL_EAGER


def get_overhead_profile(
    gpu_key: str | None, flash_attn: bool = False
) -> OverheadProfile:
    kernel = kernel_name(flash_attn)
    if gpu_key is None:
        return DEFAULT_OVERHEAD_PROFILE
    if not isinstance(gpu_key, str):
        raise ValueError("gpu_key must be a string or None")

    return OVERHEAD_DB.get(
        (gpu_key.strip().casefold(), kernel), DEFAULT_OVERHEAD_PROFILE
    )


def list_overhead_profiles() -> list[tuple[tuple[str, str], OverheadProfile]]:
    return list(OVERHEAD_DB.items())
