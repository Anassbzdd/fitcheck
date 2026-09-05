# Component 5
from __future__ import annotations
from dataclasses import dataclass
from fitcheck.config_parser import ModelConfig
from fitcheck.utils import bytes_to_mib, precision_to_bytes

_LOGITS_COPIES = 4
_LOGITS_BYTES = 4.0

_CHECKPOINT_TENSORS_PER_LAYER = 2
_EAGER_ATTENTION_COPIES = 9


@dataclass(frozen=True)
class _ActivationParts:
    layer_bytes: float
    logits_bytes: float
    attention_matrix_bytes: float
    checkpoint_store_bytes: float


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_flag(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _activation_parts(
    config: ModelConfig,
    micro_batch: int,
    sequence_length: int,
    flash_attn: bool,
    bytes_per_element: float,
) -> _ActivationParts:
    hidden_size = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_kv_heads * config.head_dim
    bracket = (
        4 * hidden_size + 2 * q_width + 2 * kv_width + 3 * config.intermediate_size
    )
    tokens = micro_batch * sequence_length

    attention_matrix_bytes = (
        _EAGER_ATTENTION_COPIES
        * bytes_per_element
        * micro_batch
        * config.num_attention_heads
        * sequence_length**2
    )

    layer_bytes = bytes_per_element * tokens * bracket
    if not flash_attn:
        layer_bytes += attention_matrix_bytes

    return _ActivationParts(
        layer_bytes=layer_bytes,
        logits_bytes=_LOGITS_COPIES * _LOGITS_BYTES * tokens * config.vocab_size,
        checkpoint_store_bytes=(
            _CHECKPOINT_TENSORS_PER_LAYER
            * bytes_per_element
            * config.num_layers
            * tokens
            * hidden_size
        ),
        attention_matrix_bytes=attention_matrix_bytes,
    )


def estimate_activation_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    grad_checkpoint: bool,
    flash_attn: bool,
    precision: str,
) -> float:
    micro_batch = _validate_positive_int(batch_size, "batch_size")
    sequence_length = _validate_positive_int(seq_len, "seq_len")
    _validate_flag(grad_checkpoint, "grad_checkpoint")
    _validate_flag(flash_attn, "flash_attn")
    bytes_per_element = precision_to_bytes(precision)

    parts = _activation_parts(
        config, micro_batch, sequence_length, flash_attn, bytes_per_element
    )

    if grad_checkpoint:
        total_bytes = parts.checkpoint_store_bytes + max(
            parts.logits_bytes, parts.layer_bytes
        )
    else:
        total_bytes = config.num_layers * parts.layer_bytes + parts.logits_bytes

    return bytes_to_mib(round(total_bytes))
