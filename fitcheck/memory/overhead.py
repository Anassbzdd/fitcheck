# Component 6
from __future__ import annotations

_BASE_CONTEXT_MIB = 500.0
_FRAGMENTATION_FRACTION = 0.05


def _validate_memory(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def estimate_overhead(weight_memory: float, activation_memory: float) -> float:
    base_weights = _validate_memory(weight_memory, "weight_memory")
    activations = _validate_memory(activation_memory, "activation_memory")

    return _BASE_CONTEXT_MIB + _FRAGMENTATION_FRACTION * (base_weights + activations)
