# Component 7 -- inference serving
from __future__ import annotations

from dataclasses import dataclass

from fitcheck.config_parser import ModelConfig
from fitcheck.memory.weights import QuantizationConfig, estimate_weight_memory
from fitcheck.utils import bytes_to_mib, precision_to_bytes

_KV_TENSORS_PER_LAYER = 2
_COMPUTE_PRECISIONS = ("fp32", "fp16", "bf16")
_QUANTIZATIONS = ("none", "nf4", "int8")


@dataclass(frozen=True)
class InferenceMemory:
    weight_mib: float
    kv_cache_mib: float

    @property
    def total_mib(self) -> float:
        return self.weight_mib + self.kv_cache_mib


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_precision(precision: str) -> str:
    if not isinstance(precision, str):
        raise ValueError("precision must be a string")

    normalized_precision = precision.strip().casefold()
    if normalized_precision not in _COMPUTE_PRECISIONS:
        supported = ", ".join(_COMPUTE_PRECISIONS)
        raise ValueError(
            f"Unsupported precision '{precision}'. precision is the compute dtype "
            f"(one of: {supported}); quantize the base model with quantization instead."
        )
    return normalized_precision


def _validate_quantization(quantization: str) -> str:
    if not isinstance(quantization, str):
        raise ValueError("quantization must be a string")

    normalized_quantization = quantization.strip().casefold()
    if normalized_quantization not in _QUANTIZATIONS:
        supported = ", ".join(_QUANTIZATIONS)
        raise ValueError(
            f"Unsupported quantization '{quantization}'. Supported: {supported}."
        )
    return normalized_quantization


def _validate_flag(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _serving_weight_mib(
    config: ModelConfig, precision: str, quantization: str, double_quant: bool
) -> float:
    if quantization == "none":
        return estimate_weight_memory(config.num_params, precision)

    float_params = min(config.num_unquantized_params, config.num_params)
    quantized_params = config.num_params - float_params
    if quantized_params <= 0:
        raise ValueError(
            f"'{config.name}' has no quantizable weights: every parameter is an "
            "embedding, LM head or norm. Use quantization='none'."
        )

    quantized_mib = estimate_weight_memory(
        quantized_params,
        quantization,
        QuantizationConfig(enabled=True, double_quant=double_quant),
    )
    return quantized_mib + estimate_weight_memory(float_params, precision)


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
    quantization: str = "none",
    double_quant: bool = False,
) -> InferenceMemory:
    if not isinstance(config, ModelConfig):
        raise ValueError("config must be a ModelConfig")

    compute_precision = _validate_precision(precision)
    base_quantization = _validate_quantization(quantization)
    _validate_flag(double_quant, "double_quant")
    sequence_length = _validate_positive_int(seq_len, "seq_len")
    concurrent = _validate_positive_int(num_concurrent, "num_concurrent")

    weight_mib = _serving_weight_mib(
        config, compute_precision, base_quantization, double_quant
    )
    kv_cache_bytes = _kv_cache_bytes(
        config,
        sequence_length,
        concurrent,
        precision_to_bytes(compute_precision),
    )

    return InferenceMemory(
        weight_mib=weight_mib,
        kv_cache_mib=bytes_to_mib(round(kv_cache_bytes)),
    )
