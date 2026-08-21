from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_mib: int
    usable_mib: int


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
    "t4": GpuSpec("Tesla T4", 16_384, 15_360),
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
    "h100-80": GpuSpec("H100 80GB", 81_920, 79_000),
    "h200": GpuSpec("H200 141GB", 144_384, 140_000),
    "b200": GpuSpec("B200 192GB", 196_608, 190_000),
}


def get_gpu(name: str) -> GpuSpec:
    if not isinstance(name, str):
        raise ValueError("GPU name must be a string")

    normalized_name = name.strip().casefold()
    try:
        return GPU_DB[normalized_name]
    except KeyError as error:
        available_gpus = ", ".join(GPU_DB)
        raise ValueError(
            f"Unknown GPU '{name}'. Available GPUs: {available_gpus}."
        ) from error
