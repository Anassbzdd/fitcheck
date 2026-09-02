# Component 5
from __future__ import annotations
from fitcheck.config_parser import ModelConfig
from fitcheck.utils import bytes_to_mib, precision_to_bytes

_LOGITS_COPIES = 4
_LOGITS_BYTES = 4.0

# Two (b, s, h) tensors survive per checkpoint boundary, not one: non-reentrant
# checkpointing keeps the layer input AND the recomputed output. Measured on a T4
# across 20 runs -- a multiplier of 1 gives 10.4% worst-case error and 3 gives 9.8%,
# against 4.8% for 2. See docs/SPEC.md, Component 5.
_CHECKPOINT_TENSORS_PER_LAYER = 2

# Eager attention materializes the (b, n_h, s, s) score matrix about nine times at
# the compute dtype: forward is scores, masked scores, the FP32 softmax (2 gamma)
# and the cast back = 5; backward is grad-out, the FP32 softmax backward (2 gamma)
# and grad-scores = 4. Flash Attention removes all of it.
_EAGER_ATTENTION_COPIES = 9


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
    micro_batch = _validate_positive_int(batch_size, "batch_size")
    sequence_length = _validate_positive_int(seq_len, "seq_len")
    _validate_flag(grad_checkpoint, "grad_checkpoint")
    _validate_flag(flash_attn, "flash_attn")
    bytes_per_element = precision_to_bytes(precision)

    hidden_size = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_kv_heads * config.head_dim
    bracket = (
        4 * hidden_size + 2 * q_width + 2 * kv_width + 3 * config.intermediate_size
    )
    tokens = micro_batch * sequence_length

    layer_bytes = bytes_per_element * tokens * bracket
    if not flash_attn:
        layer_bytes += (
            _EAGER_ATTENTION_COPIES
            * bytes_per_element
            * micro_batch
            * config.num_attention_heads
            * sequence_length**2
        )

    logits_bytes = _LOGITS_COPIES * _LOGITS_BYTES * tokens * config.vocab_size

    if grad_checkpoint:
        # Only the checkpoints are resident for the whole backward. The LM-head
        # hump and one layer's recompute are both transient and never overlap, so
        # the peak takes whichever is larger -- summing them over-counts, and it is
        # the max that flips between the two as seq_len grows.
        stored_bytes = (
            _CHECKPOINT_TENSORS_PER_LAYER
            * bytes_per_element
            * config.num_layers
            * tokens
            * hidden_size
        )
        total_bytes = stored_bytes + max(logits_bytes, layer_bytes)
    else:
        # Without checkpointing every layer's activations stay resident, so they do
        # coexist with the logits and this really is a sum. Unmeasured so far -- no
        # ground-truth run has exercised this branch.
        total_bytes = config.num_layers * layer_bytes + logits_bytes

    return bytes_to_mib(round(total_bytes))
