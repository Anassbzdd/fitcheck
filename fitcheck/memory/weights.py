# Component 1
from __future__ import annotations
from dataclasses import dataclass
from fitcheck.utils import bytes_to_mib, precision_to_bytes

_DEFAULT_QUANT_BLOCK_SIZE = 64
_DOUBLE_QUANT_OVERHEAD_FACTOR = 0.5
_QUANTIZABLE_PRECISIONS = frozenset({"int4", "nf4", "int8", "fp8"})

_SCALE_BYTES = 4.0
_UNQUANTIZED_BYTES = 4.0


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = False
    block_size: int = _DEFAULT_QUANT_BLOCK_SIZE
    double_quant: bool = False


def _validate_num_params(num_params: int) -> int:
    if isinstance(num_params, bool) or not isinstance(num_params, int) or num_params <= 0:
        raise ValueError("num_params must be a positive integer")
    return num_params


def _validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("quantization_config.block_size must be a positive integer")
    return block_size


def _validate_unquantized_params(unquantized_params: int, num_params: int) -> int:
    if (
        isinstance(unquantized_params, bool)
        or not isinstance(unquantized_params, int)
        or unquantized_params < 0
    ):
        raise ValueError("unquantized_params must be a non-negative integer")
    if unquantized_params > num_params:
        raise ValueError("unquantized_params cannot exceed num_params")
    return unquantized_params


def _quantization_overhead_bytes(
    num_params: int,
    precision: str,
    quantization_config: QuantizationConfig | None,
) -> float:
    if quantization_config is None or not quantization_config.enabled:
        return 0.0

    normalized_precision = precision.strip().casefold()
    if normalized_precision not in _QUANTIZABLE_PRECISIONS:
        supported = ", ".join(sorted(_QUANTIZABLE_PRECISIONS))
        raise ValueError(
            "quantization_config.enabled requires a quantized precision "
            f"(one of: {supported}), got '{precision}'"
        )

    block_size = _validate_block_size(quantization_config.block_size)
    overhead_bytes_per_param = _SCALE_BYTES / block_size

    if quantization_config.double_quant:
        overhead_bytes_per_param *= _DOUBLE_QUANT_OVERHEAD_FACTOR

    return num_params * overhead_bytes_per_param


def estimate_weight_memory(
    num_params: int,
    precision: str,
    quantization_config: QuantizationConfig | None = None,
    unquantized_params: int = 0,
) -> float:
    validated_num_params = _validate_num_params(num_params)
    bytes_per_param = precision_to_bytes(precision)

    quantizing = quantization_config is not None and quantization_config.enabled
    skipped = (
        _validate_unquantized_params(unquantized_params, validated_num_params)
        if quantizing
        else 0
    )
    quantized_params = validated_num_params - skipped

    base_weight_bytes = quantized_params * bytes_per_param
    base_weight_bytes += skipped * _UNQUANTIZED_BYTES
    quantization_overhead_bytes = _quantization_overhead_bytes(
        quantized_params, precision, quantization_config
    )

    return bytes_to_mib(round(base_weight_bytes + quantization_overhead_bytes))
