from __future__ import annotations

COMPUTE_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16")

# Reject impossible shapes before byte arithmetic can overflow.
MAX_SEQ_LEN = 1 << 24  # 16,777,216 tokens -- past every published context window
MAX_SEQUENCES = 1 << 20  # 1,048,576 -- the estimator's own batch-search ceiling

QUANTIZATIONS: tuple[str, ...] = ("none", "nf4", "int8")
DOUBLE_QUANT_QUANTIZATION = "nf4"

_DOUBLE_QUANT_UNDER_NONE = (
    "--double-quant (double_quant) has nothing to quantize under --quant none. It cuts "
    "the NF4 scale overhead by ~75%, so pair it with --quant nf4."
)
_DOUBLE_QUANT_OUTSIDE_NF4 = (
    "--double-quant (double_quant) applies only to --quant nf4, got '{quantization}'. "
    "It is bitsandbytes' bnb_4bit_use_double_quant, a second level of quantization for "
    "the NF4 absmax scales, and {quantization} has no such scales -- billing its ~75% "
    "saving here would report a saving the run never gets. scripts/measure.py refuses "
    "the same pair."
)


def validate_precision(precision: str) -> str:
    if not isinstance(precision, str):
        raise ValueError("precision must be a string")

    normalized_precision = precision.strip().casefold()
    if normalized_precision not in COMPUTE_PRECISIONS:
        supported = ", ".join(COMPUTE_PRECISIONS)
        raise ValueError(
            f"Unsupported precision '{precision}'. precision is the compute dtype "
            f"(one of: {supported}); quantize the base model with quantization instead."
        )
    return normalized_precision


def validate_quantization(quantization: str) -> str:
    if not isinstance(quantization, str):
        raise ValueError("quantization must be a string")

    normalized_quantization = quantization.strip().casefold()
    if normalized_quantization not in QUANTIZATIONS:
        supported = ", ".join(QUANTIZATIONS)
        raise ValueError(
            f"Unsupported quantization '{quantization}'. Supported: {supported}."
        )
    return normalized_quantization


def validate_flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def double_quant_conflict(quantization: str) -> str | None:
    normalized_quantization = validate_quantization(quantization)
    if normalized_quantization == DOUBLE_QUANT_QUANTIZATION:
        return None
    if normalized_quantization == "none":
        return _DOUBLE_QUANT_UNDER_NONE
    return _DOUBLE_QUANT_OUTSIDE_NF4.format(quantization=normalized_quantization)


def validate_double_quant(double_quant: object, quantization: str) -> bool:
    enabled = validate_flag(double_quant, "double_quant")
    if enabled:
        conflict = double_quant_conflict(quantization)
        if conflict is not None:
            raise ValueError(conflict)
    return enabled
