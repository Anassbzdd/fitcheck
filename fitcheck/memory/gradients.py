# Component 4
from __future__ import annotations
from fitcheck.utils import bytes_to_mib, precision_to_bytes


def _validate_trainable_params(trainable_params: int) -> int:
    if (
        isinstance(trainable_params, bool)
        or not isinstance(trainable_params, int)
        or trainable_params <= 0
    ):
        raise ValueError("trainable_params must be a positive integer")
    return trainable_params


def estimate_gradient_memory(trainable_params: int, precision: str) -> float:
    validated_trainable_params = _validate_trainable_params(trainable_params)
    bytes_per_param = precision_to_bytes(precision)

    return bytes_to_mib(validated_trainable_params * bytes_per_param)
