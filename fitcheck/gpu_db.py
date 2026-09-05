from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_mib: int
    usable_mib: int

# `t4` is the only entry with a measured total (14,912 MiB reported by
# torch.cuda.get_device_properties across all 20 validation runs, ECC on). Every other row is
# an estimate. A measured row for any of them is a genuinely useful contribution -- see
# .github/ISSUE_TEMPLATE/measurement.yml.

GPU_DB: dict[str, GpuSpec] = {
    # Consumer GPUs
    "3060-12": GpuSpec("RTX 3060 12GB", 12_288, 11_500),
    "4070ti": GpuSpec("RTX 4070 Ti", 12_288, 11_500),
    "5070": GpuSpec("RTX 5070", 12_288, 11_500),
    "5070ti": GpuSpec("RTX 5070 Ti", 16_384, 15_500),
    "5080": GpuSpec("RTX 5080", 16_384, 15_500),
    "3090": GpuSpec("RTX 3090", 24_576, 23_500),
    "4090": GpuSpec("RTX 4090", 24_576, 23_500),
    "5090": GpuSpec("RTX 5090", 32_768, 31_000),

    # Older / cloud GPUs
    # Measured: 14,912 MiB total with ECC on, not the 16,384 a "16 GB" label suggests.
    "t4": GpuSpec("Tesla T4", 14_912, 14_000),
    "v100-16": GpuSpec("Tesla V100 16GB", 16_384, 15_000),
    "l4": GpuSpec("L4 24GB", 24_576, 23_000),
    "a10": GpuSpec("A10 24GB", 24_576, 23_000),

    # Workstation / pro GPUs
    "a6000": GpuSpec("RTX A6000 48GB", 49_152, 46_000),
    "rtx6000-ada": GpuSpec("RTX 6000 Ada 48GB", 49_152, 46_000),
    "l40": GpuSpec("L40 48GB", 49_152, 46_000),
    "l40s": GpuSpec("L40S 48GB", 49_152, 46_000),

    # Datacenter GPUs
    "a100-40": GpuSpec("A100 40GB", 40_960, 39_500),
    "a100-80": GpuSpec("A100 80GB", 81_920, 79_000),
    "h100": GpuSpec("H100 80GB", 81_920, 79_000),
    "h100-80": GpuSpec("H100 80GB", 81_920, 79_000),
    "h200": GpuSpec("H200 141GB", 144_384, 134_000),
    "b200": GpuSpec("B200 192GB", 196_608, 176_000),
}


def list_gpus() -> list[tuple[str, GpuSpec]]:
    return list(GPU_DB.items())


def get_gpu(name: str | None = None, vram_mib: int | None = None) -> GpuSpec:
    if vram_mib is not None:
        if isinstance(vram_mib, bool) or not isinstance(vram_mib, int) or vram_mib <= 0:
            raise ValueError("vram_mib must be a positive integer")

        custom_name = name.strip() if isinstance(name, str) and name.strip() else "Custom GPU"
        return GpuSpec(custom_name, vram_mib, vram_mib * 95 // 100)

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Provide a GPU name or a positive vram_mib override")

    normalized_name = name.strip().casefold()
    try:
        return GPU_DB[normalized_name]
    except KeyError as error:
        available_gpus = ", ".join(GPU_DB)
        raise ValueError(
            f"Unknown GPU '{name}'. Available GPUs: {available_gpus}."
        ) from error
