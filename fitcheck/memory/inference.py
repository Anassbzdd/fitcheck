# Component 7 -- inference serving (v0.2). Training components 1-6 do not apply.
from __future__ import annotations

from fitcheck.config_parser import ModelConfig
from fitcheck.memory.weights import estimate_weight_memory
from fitcheck.utils import bytes_to_mib, precision_to_bytes

# K and V, once each. Not "2 per tensor" -- the pair is the 2.
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
    # GQA/MQA: the cache is sized by num_kv_heads * head_dim, never by hidden_size.
    # A Llama-3.1-8B (8 KV heads vs 32 query heads) cache is a quarter of what the
    # hidden_size reading would give.
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
    """Peak VRAM in MiB for serving `config`, as weights + KV cache.

    Inference keeps nothing for a backward pass: no optimizer states, no gradients,
    and no saved activations. What is left is the resident weights plus one KV cache
    entry per layer per concurrent request:

        KV = 2 * L * n_kv * d_k * s * n_concurrent * gamma

    `num_concurrent` is the batch dimension of that cache -- `seq_len` is the context
    each request holds, and `num_concurrent` is how many of them are in flight at once.

    `precision` is the serving dtype and is applied to the weights and to the cache
    alike, so a quantized-weights / fp16-cache deployment (bitsandbytes, AWQ, GPTQ) is
    not modelled here -- that needs a separate compute-dtype axis. Nor is the CUDA
    context and fragmentation overhead: this is the model-side number, and the caller
    adds `estimate_overhead` before rendering a fits/doesn't-fit verdict.

    Args:
        config: Parsed Hugging Face model config.
        precision: Serving dtype, e.g. "fp16", "bf16", "int8".
        seq_len: Context length per request, in tokens.
        num_concurrent: Requests served simultaneously. Defaults to 1.

    Returns:
        Weights + KV cache, in MiB.

    Raises:
        ValueError: If `config` is not a ModelConfig, `precision` is unsupported, or
            `seq_len` / `num_concurrent` is not a positive integer.
    """
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
