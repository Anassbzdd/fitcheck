# Component 3
from __future__ import annotations
from fitcheck.utils import bytes_to_mib, optimizer_bytes_per_param, precision_to_bytes

_MASTER_WEIGHT_BYTES_PER_PARAM = 4
_FP32_BYTES = 4.0


def _validate_trainable_params(trainable_params: int) -> int:
    if (
        isinstance(trainable_params, bool)
        or not isinstance(trainable_params, int)
        or trainable_params <= 0
    ):
        raise ValueError("trainable_params must be a positive integer")
    return trainable_params


def _validate_is_lora(is_lora: bool) -> bool:
    if not isinstance(is_lora, bool):
        raise ValueError("is_lora must be a boolean")
    return is_lora


def _keeps_master_weights(is_lora: bool, precision: str) -> bool:
    params_are_fp32 = precision_to_bytes(precision) == _FP32_BYTES
    return not is_lora and not params_are_fp32


def estimate_optimizer_memory(
    trainable_params: int,
    optimizer: str,
    is_lora: bool = True,
    optimizer_dtype: str = "fp32",
    precision: str = "bf16",
) -> float:
    validated_trainable_params = _validate_trainable_params(trainable_params)
    validated_is_lora = _validate_is_lora(is_lora)

    bytes_per_param = optimizer_bytes_per_param(optimizer, optimizer_dtype)
    if _keeps_master_weights(validated_is_lora, precision):
        bytes_per_param += _MASTER_WEIGHT_BYTES_PER_PARAM

    return bytes_to_mib(validated_trainable_params * bytes_per_param)
