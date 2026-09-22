from __future__ import annotations

from collections.abc import Iterable

from fitcheck.overhead_db import KERNEL_EAGER, KERNEL_FLASH

# Conservative buffers are based only on the final T4 NF4 validation archive.
_FINAL_VALIDATION_RESERVE_MIB: dict[tuple[str, str, str], float] = {
    ("t4", KERNEL_EAGER, "nf4"): 701.0,
    ("t4", KERNEL_FLASH, "nf4"): 2_264.0,
}

_FINAL_VALIDATION_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})


def final_validation_scope_matches(
    *,
    precision: str,
    quantization: str,
    double_quant: bool,
    optimizer: str,
    optimizer_dtype: str,
    lora_rank: int | None,
    lora_targets: Iterable[str],
    grad_checkpoint: bool,
    seq_len: int,
) -> bool:
    targets = tuple(target.strip().casefold() for target in lora_targets)
    return (
        precision.strip().casefold() == "fp16"
        and quantization.strip().casefold() == "nf4"
        and double_quant is False
        and optimizer.strip().casefold() == "adamw"
        and optimizer_dtype.strip().casefold() == "fp32"
        and lora_rank == 16
        and len(targets) == len(_FINAL_VALIDATION_TARGETS)
        and frozenset(targets) == _FINAL_VALIDATION_TARGETS
        and grad_checkpoint is True
        and seq_len == 1024
    )


def conservative_reserve_mib(
    gpu_key: str | None,
    flash_attn: bool,
    quantization: str,
) -> float | None:
    if gpu_key is None:
        return None
    kernel = KERNEL_FLASH if flash_attn else KERNEL_EAGER
    return _FINAL_VALIDATION_RESERVE_MIB.get(
        (gpu_key.strip().casefold(), kernel, quantization.strip().casefold())
    )
