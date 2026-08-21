from __future__ import annotations


_MIB_IN_BYTES = 1024**2
_PRECISION_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}
_OPTIMIZER_BYTES_PER_PARAM: dict[str, int] = {
    "adamw-fp32": 8,
    "adamw-bf16": 4,
    "adam8bit": 2,
    "sgd": 0,
    "sgd-momentum": 4,
}

def bytes_to_mib(b: int) -> float:
    if isinstance(b, bool) or not isinstance(b, int) or b < 0:
        raise ValueError("byte count must be a non-negative integer")
    return b / _MIB_IN_BYTES


def precision_to_bytes(precision: str) -> float:
    if not isinstance(precision, str):
        raise ValueError("precision must be a string")

    normalized_precision = precision.strip().casefold()
    try:
        return _PRECISION_BYTES[normalized_precision]
    except KeyError as error:
        supported = ", ".join(_PRECISION_BYTES)
        raise ValueError(
            f"Unsupported precision '{precision}'. Supported precisions: {supported}."
        ) from error


def optimizer_bytes_per_param(optimizer: str) -> int:
    if not isinstance(optimizer, str):
        raise ValueError("optimizer must be a string")

    normalized_optimizer = optimizer.strip().casefold()
    try:
        return _OPTIMIZER_BYTES_PER_PARAM[normalized_optimizer]
    except KeyError as error:
        supported = ", ".join(_OPTIMIZER_BYTES_PER_PARAM)
        raise ValueError(
            f"Unsupported optimizer '{optimizer}'. Supported optimizers: {supported}."
        ) from error
