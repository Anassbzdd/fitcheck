# Component 5
from __future__ import annotations
from fitcheck.config_parser import ModelConfig
from fitcheck.utils import bytes_to_mib, precision_to_bytes


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_flag(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def estimate_activation_memory(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    grad_checkpoint: bool,
    flash_attn: bool,
    precision: str,
) -> float:
    """Estimate saved-activation memory for one training step, in MiB.

    A_layer = g*b*s*[6h + 2*n_kv*d_k + 3*d_ff] + g*b*n_h*s^2 * 1[no Flash Attention]
    A_act   = L*g*b*s*h + A_layer   (gradient checkpointing, every layer)
              L * A_layer           (no checkpointing)

    where g = precision_to_bytes(precision), the COMPUTE dtype — every term scales
    with it, so FP32 doubles the result relative to BF16/FP16.

    `batch_size` is the MICRO-batch size; gradient accumulation costs no memory.
    """
    micro_batch = _validate_positive_int(batch_size, "batch_size")
    sequence_length = _validate_positive_int(seq_len, "seq_len")
    _validate_flag(grad_checkpoint, "grad_checkpoint")
    _validate_flag(flash_attn, "flash_attn")
    bytes_per_element = precision_to_bytes(precision)

    hidden_size = config.hidden_size
    # h * (n_kv / n_h) exactly, since head_dim = hidden_size // num_attention_heads.
    # This is the GQA-reduced K and V width — never assume it equals hidden_size.
    kv_width = config.num_kv_heads * config.head_dim

    bracket = 6 * hidden_size + 2 * kv_width + 3 * config.intermediate_size
    tokens = micro_batch * sequence_length

    layer_bytes = bytes_per_element * tokens * bracket
    if not flash_attn:
        # Softmax matrix (b, n_h, s, s) — Flash Attention never materializes it.
        layer_bytes += (
            bytes_per_element
            * micro_batch
            * config.num_attention_heads
            * sequence_length**2
        )

    if grad_checkpoint:
        # One saved input per layer, plus one layer recomputed in full.
        total_bytes = bytes_per_element * config.num_layers * tokens * hidden_size
        total_bytes += layer_bytes
    else:
        total_bytes = config.num_layers * layer_bytes

    return bytes_to_mib(round(total_bytes))
