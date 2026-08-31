# Component 2
from __future__ import annotations
from typing import Iterable
from fitcheck.config_parser import ModelConfig
from fitcheck.utils import bytes_to_mib, precision_to_bytes

_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)

LORA_TARGETS_MINIMAL: tuple[str, ...] = ("q_proj", "v_proj")
LORA_TARGETS_STANDARD: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_TARGETS_FULL: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

def _validate_rank(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    return rank

def _target_dims(config: ModelConfig, target: str) -> tuple[int, int]:
    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_kv_heads * config.head_dim
    attn_out = q_out

    if target == "q_proj":
        return config.hidden_size, q_out
    if target == "k_proj":
        return config.hidden_size, kv_out
    if target == "v_proj":
        return config.hidden_size, kv_out
    if target == "o_proj":
        return attn_out, config.hidden_size
    if target == "gate_proj":
        return config.hidden_size, config.intermediate_size
    if target == "up_proj":
        return config.hidden_size, config.intermediate_size
    if target == "down_proj":
        return config.intermediate_size, config.hidden_size

    supported = ", ".join(sorted(_TARGET_MODULES))
    raise ValueError(f"Unsupported LoRA target '{target}'. Supported targets: {supported}.")


def estimate_lora_memory(
    config: ModelConfig,
    rank: int,
    targets: Iterable[str],
    precision: str,
    adapter_precision: str | None = None,
) -> float:
    validated_rank = _validate_rank(rank)
    bytes_per_param = precision_to_bytes(adapter_precision or precision)

    target_list = list(targets)
    if not target_list:
        raise ValueError("targets must contain at least one LoRA target module")

    seen: set[str] = set()
    dims_sum = 0
    for target in target_list:
        if target in seen:
            raise ValueError(f"Duplicate LoRA target '{target}'")
        seen.add(target)
        d_in, d_out = _target_dims(config, target)
        dims_sum += d_in + d_out

    total_params = config.num_layers * validated_rank * dims_sum
    total_bytes = total_params * bytes_per_param

    return bytes_to_mib(round(total_bytes))
