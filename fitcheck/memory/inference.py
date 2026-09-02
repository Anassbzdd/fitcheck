# Component 7 -- inference serving
from __future__ import annotations

from fitcheck.config_parser import ModelConfig
from fitcheck.memory.weights import estimate_weight_memory
from fitcheck.utils import bytes_to_mib, precision_to_bytes


_KV_TENSORS_PER_LAYER = 2


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _kv_cache_bytes(
    config: ModelConfig,
    seq_len: int,
    num_concurrent: int,
    bytes_per_element: float,
) -> float:
    kv_width = config.num_kv_heads * config.head_dim
    return (
        _KV_TENSORS_PER_LAYER
        * config.num_layers
        * kv_width
        * seq_len
        * num_concurrent
        * bytes_per_element
    )


def estimate_inference_memory(
    config: ModelConfig,
    precision: str,
    seq_len: int,
    num_concurrent: int = 1,
) -> float:
    if not isinstance(config, ModelConfig):
        raise ValueError("config must be a ModelConfig")

    sequence_length = _validate_positive_int(seq_len, "seq_len")
    concurrent = _validate_positive_int(num_concurrent, "num_concurrent")
    bytes_per_element = precision_to_bytes(precision)

    weight_mib = estimate_weight_memory(config.num_params, precision)
    kv_cache_bytes = _kv_cache_bytes(
        config, sequence_length, concurrent, bytes_per_element
    )

    return weight_mib + bytes_to_mib(round(kv_cache_bytes))
