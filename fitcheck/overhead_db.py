from __future__ import annotations

from dataclasses import dataclass

REFERENCE_SEQ_LEN = 2048

KERNEL_EAGER = "eager"
KERNEL_FLASH = "flash"

QUANTIZATIONS = ("none", "nf4", "int8")


@dataclass(frozen=True)
class OverheadProfile:
    gpu: str
    kernel: str
    base_context_mib: float
    fragmentation: float = 0.0
    fragmentation_per_octave: float = 0.0
    quantization: str = "any"
    hump_fragmentation_logits_win: float | None = None
    hump_fragmentation_layer_win: float | None = None
    seq_len_min: int = REFERENCE_SEQ_LEN
    seq_len_max: int = REFERENCE_SEQ_LEN
    runs: int = 0
    worst_over_pct: float = 0.0
    worst_under_pct: float = 0.0
    source: str = "unmeasured default"

    @property
    def measured(self) -> bool:
        return self.runs > 0

    @property
    def uses_hump_form(self) -> bool:
        return self.hump_fragmentation_logits_win is not None

    def fragmentation_for(self, logits_mib: float, layer_mib: float) -> float:
        if self.hump_fragmentation_logits_win is None:
            raise ValueError("profile does not use the hump form")
        if layer_mib > logits_mib and self.hump_fragmentation_layer_win is not None:
            return self.hump_fragmentation_layer_win
        return self.hump_fragmentation_logits_win


DEFAULT_OVERHEAD_PROFILE = OverheadProfile(
    gpu="any",
    kernel="any",
    base_context_mib=500.0,
    fragmentation=0.05,
    fragmentation_per_octave=0.0,
    runs=0,
    source="unmeasured conservative default (pre-9.3 constants)",
)


OVERHEAD_DB: dict[tuple[str, str, str], OverheadProfile] = {
    ("t4", "flash", "none"): OverheadProfile(
        gpu="Tesla T4",
        kernel="flash",
        quantization="none",
        base_context_mib=140.875,
        hump_fragmentation_logits_win=1.2744,
        hump_fragmentation_layer_win=None,
        seq_len_min=512,
        seq_len_max=4096,
        runs=12,
        worst_over_pct=2.4,
        worst_under_pct=-2.2,
        source="12 T4 runs, 5 models, seq 512-4096, task 9.3 (2026-09-16)",
    ),
    ("t4", "flash", "nf4"): OverheadProfile(
        gpu="Tesla T4",
        kernel="flash",
        quantization="nf4",
        base_context_mib=140.875,
        hump_fragmentation_logits_win=1.9224,
        hump_fragmentation_layer_win=None,
        seq_len_min=512,
        seq_len_max=4096,
        runs=17,
        worst_over_pct=3.0,
        worst_under_pct=-6.5,
        source="17 T4 runs, 6 models, seq 512-4096, task 9.3 (2026-09-16)",
    ),
    ("t4", "eager", "none"): OverheadProfile(
        gpu="Tesla T4",
        kernel="eager",
        quantization="none",
        base_context_mib=140.875,
        hump_fragmentation_logits_win=0.5987,
        hump_fragmentation_layer_win=0.9257,
        seq_len_min=512,
        seq_len_max=4096,
        runs=17,
        worst_over_pct=9.1,
        worst_under_pct=-4.9,
        source="17 T4 runs, 6 models, seq 512-4096, task 9.3 (2026-09-16)",
    ),
    ("t4", "eager", "nf4"): OverheadProfile(
        gpu="Tesla T4",
        kernel="eager",
        quantization="nf4",
        base_context_mib=140.875,
        hump_fragmentation_logits_win=0.6095,
        hump_fragmentation_layer_win=0.6913,
        seq_len_min=512,
        seq_len_max=4096,
        runs=20,
        worst_over_pct=12.7,
        worst_under_pct=-7.4,
        source="20 T4 runs, 6 models, seq 512-4096, task 9.3 (2026-09-16)",
    ),
}


def kernel_name(flash_attn: bool) -> str:
    if not isinstance(flash_attn, bool):
        raise ValueError("flash_attn must be a boolean")
    return KERNEL_FLASH if flash_attn else KERNEL_EAGER


def get_overhead_profile(
    gpu_key: str | None,
    flash_attn: bool = False,
    quantization: str = "none",
) -> OverheadProfile:
    kernel = kernel_name(flash_attn)
    if not isinstance(quantization, str):
        raise ValueError("quantization must be a string")
    if gpu_key is None:
        return DEFAULT_OVERHEAD_PROFILE
    if not isinstance(gpu_key, str):
        raise ValueError("gpu_key must be a string or None")

    return OVERHEAD_DB.get(
        (gpu_key.strip().casefold(), kernel, quantization.strip().casefold()),
        DEFAULT_OVERHEAD_PROFILE,
    )


def list_overhead_profiles() -> list[tuple[tuple[str, str, str], OverheadProfile]]:
    return list(OVERHEAD_DB.items())
