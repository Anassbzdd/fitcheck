# Component 5
from __future__ import annotations
from dataclasses import dataclass
from fitcheck.config_parser import ModelConfig
from fitcheck.utils import bytes_to_mib, precision_to_bytes

_LOGITS_BYTES = 4.0

_HIDDEN_TENSORS_PER_LAYER = 12
_Q_TENSORS_PER_LAYER = 3
_KV_TENSORS_PER_LAYER = 1
_FF_TENSORS_PER_LAYER = 3

_EAGER_RETAINED_COPIES = 2.9

_FP32_BYTES = 4.0


@dataclass(frozen=True)
class _ActivationProfile:
    checkpoint_tensors_per_layer: float
    logits_copies: float
    eager_attention_copies: float
    checkpoint_bytes_per_element: float | None = None


_PROFILES: dict[str, _ActivationProfile] = {
    "none": _ActivationProfile(
        checkpoint_tensors_per_layer=1,
        logits_copies=3.5,
        eager_attention_copies=7.4,
    ),
    "nf4": _ActivationProfile(
        checkpoint_tensors_per_layer=1,
        logits_copies=4,
        eager_attention_copies=9,
        checkpoint_bytes_per_element=_FP32_BYTES,
    ),
    "int8": _ActivationProfile(
        checkpoint_tensors_per_layer=2,
        logits_copies=4,
        eager_attention_copies=9,
    ),
}


def _validate_quantization(quantization: str) -> _ActivationProfile:
    if not isinstance(quantization, str):
        raise ValueError("quantization must be a string")
    normalized = quantization.strip().casefold()
    if normalized not in _PROFILES:
        supported = ", ".join(_PROFILES)
        raise ValueError(
            f"Unsupported quantization '{quantization}'. Supported: {supported}."
        )
    return _PROFILES[normalized]


@dataclass(frozen=True)
class _ActivationParts:
    layer_bytes: float
    retained_layer_bytes: float
    logits_bytes: float
    attention_matrix_bytes: float
    retained_attention_matrix_bytes: float
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
    profile: _ActivationProfile = _PROFILES["none"],
) -> _ActivationParts:
    hidden_size = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_kv_heads * config.head_dim
    bracket = (
        _HIDDEN_TENSORS_PER_LAYER * hidden_size
        + _Q_TENSORS_PER_LAYER * q_width
        + _KV_TENSORS_PER_LAYER * kv_width
        + _FF_TENSORS_PER_LAYER * config.intermediate_size
    )
    tokens = micro_batch * sequence_length

    score_matrix_bytes = (
        bytes_per_element
        * micro_batch
        * config.num_attention_heads
        * sequence_length**2
    )
    attention_matrix_bytes = profile.eager_attention_copies * score_matrix_bytes
    retained_attention_matrix_bytes = _EAGER_RETAINED_COPIES * score_matrix_bytes

    layer_bytes = bytes_per_element * tokens * bracket
    retained_layer_bytes = layer_bytes
    if not flash_attn:
        layer_bytes += attention_matrix_bytes
        retained_layer_bytes += retained_attention_matrix_bytes

    return _ActivationParts(
        layer_bytes=layer_bytes,
        retained_layer_bytes=retained_layer_bytes,
        logits_bytes=profile.logits_copies * _LOGITS_BYTES * tokens * config.vocab_size,
        checkpoint_store_bytes=(
            profile.checkpoint_tensors_per_layer
            * (
                bytes_per_element
                if profile.checkpoint_bytes_per_element is None
                else profile.checkpoint_bytes_per_element
            )
            * config.num_layers
            * tokens
            * hidden_size
        ),
        attention_matrix_bytes=attention_matrix_bytes,
        retained_attention_matrix_bytes=retained_attention_matrix_bytes,
    )


def estimate_activation_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    grad_checkpoint: bool,
    flash_attn: bool,
    precision: str,
    quantization: str = "none",
) -> float:
    micro_batch = _validate_positive_int(batch_size, "batch_size")
    sequence_length = _validate_positive_int(seq_len, "seq_len")
    _validate_flag(grad_checkpoint, "grad_checkpoint")
    _validate_flag(flash_attn, "flash_attn")
    profile = _validate_quantization(quantization)
    bytes_per_element = precision_to_bytes(precision)

    parts = _activation_parts(
        config, micro_batch, sequence_length, flash_attn, bytes_per_element, profile
    )

    if grad_checkpoint:
        total_bytes = parts.checkpoint_store_bytes + max(
            parts.logits_bytes, parts.layer_bytes
        )
    else:
        total_bytes = (
            config.num_layers * parts.retained_layer_bytes + parts.logits_bytes
        )

    return bytes_to_mib(round(total_bytes))
