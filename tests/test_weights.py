from __future__ import annotations
import pytest
from fitcheck.memory.weights import QuantizationConfig, estimate_weight_memory
from fitcheck.utils import bytes_to_mib

_MIB = 1024**2


@pytest.mark.parametrize(
    ("precision", "bytes_per_param"),
    [
        ("fp32", 4.0),
        ("fp16", 2.0),
        ("bf16", 2.0),
        ("int8", 1.0),
        ("fp8", 1.0),
        ("int4", 0.5),
        ("nf4", 0.5),
    ],
)
def test_estimate_weight_memory_no_quantization(precision: str, bytes_per_param: float) -> None:
    num_params = 1_000_000_000

    result = estimate_weight_memory(num_params, precision)

    expected_mib = round(num_params * bytes_per_param) / _MIB
    assert result == pytest.approx(expected_mib, rel=1e-9)


@pytest.mark.parametrize("precision", ["FP32", " bf16 ", "Int8"])
def test_estimate_weight_memory_precision_lookup_is_case_and_whitespace_insensitive(
    precision: str,
) -> None:
    result = estimate_weight_memory(1_000_000, precision)
    assert result > 0


def test_estimate_weight_memory_llama_31_8b_qlora_matches_worked_example() -> None:
    num_params = 8_030_261_248

    result = estimate_weight_memory(num_params, "int4", QuantizationConfig(enabled=True))

    base_bytes = num_params * 0.5
    overhead_bytes = num_params * (2.0 / 64)
    expected_mib = round(base_bytes + overhead_bytes) / _MIB
    assert result == pytest.approx(expected_mib, rel=1e-9)
    assert result == pytest.approx(4_068, rel=0.05)


_QUANTIZABLE_BYTES_PER_PARAM = [
    ("int4", 0.5),
    ("nf4", 0.5),
    ("int8", 1.0),
    ("fp8", 1.0),
]


@pytest.mark.parametrize(("precision", "bytes_per_param"), _QUANTIZABLE_BYTES_PER_PARAM)
def test_estimate_weight_memory_qlora_overhead_default_block_size(
    precision: str, bytes_per_param: float
) -> None:
    num_params = 1_000_000

    result = estimate_weight_memory(num_params, precision, QuantizationConfig(enabled=True))

    base_bytes = num_params * bytes_per_param
    overhead_bytes = num_params * (2.0 / 64)
    expected_mib = round(base_bytes + overhead_bytes) / _MIB
    assert result == pytest.approx(expected_mib, rel=1e-9)


@pytest.mark.parametrize(("precision", "bytes_per_param"), _QUANTIZABLE_BYTES_PER_PARAM)
def test_estimate_weight_memory_custom_block_size(precision: str, bytes_per_param: float) -> None:
    num_params = 1_000_000

    result = estimate_weight_memory(
        num_params, precision, QuantizationConfig(enabled=True, block_size=32)
    )

    base_bytes = num_params * bytes_per_param
    overhead_bytes = num_params * (2.0 / 32)
    expected_mib = round(base_bytes + overhead_bytes) / _MIB
    assert result == pytest.approx(expected_mib, rel=1e-9)


@pytest.mark.parametrize(("precision", "bytes_per_param"), _QUANTIZABLE_BYTES_PER_PARAM)
def test_estimate_weight_memory_double_quant_roughly_halves_overhead(
    precision: str, bytes_per_param: float
) -> None:
    num_params = 1_000_000

    single = estimate_weight_memory(num_params, precision, QuantizationConfig(enabled=True))
    double = estimate_weight_memory(
        num_params, precision, QuantizationConfig(enabled=True, double_quant=True)
    )

    base_mib = bytes_to_mib(round(num_params * bytes_per_param))
    single_overhead_mib = single - base_mib
    double_overhead_mib = double - base_mib

    assert double_overhead_mib == pytest.approx(single_overhead_mib * 0.5, rel=1e-9)
    assert double < single


def test_estimate_weight_memory_no_quantization_config_has_no_overhead() -> None:
    num_params = 1_000_000

    without_config = estimate_weight_memory(num_params, "fp16")
    with_disabled_config = estimate_weight_memory(
        num_params, "fp16", QuantizationConfig(enabled=False)
    )
    expected_mib = bytes_to_mib(round(num_params * 2))
    assert without_config == with_disabled_config


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_estimate_weight_memory_quantization_requires_quantized_precision(precision: str) -> None:
    with pytest.raises(ValueError, match="requires a quantized precision"):
        estimate_weight_memory(1_000_000, precision, QuantizationConfig(enabled=True))


@pytest.mark.parametrize("bad_num_params", [0, -1, True, 12.5, "1000"])
def test_estimate_weight_memory_rejects_invalid_num_params(bad_num_params: object) -> None:
    with pytest.raises(ValueError, match="num_params must be a positive integer"):
        estimate_weight_memory(bad_num_params, "fp16")


def test_estimate_weight_memory_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        estimate_weight_memory(1_000_000, "fp4")


@pytest.mark.parametrize("bad_block_size", [0, -1, True, 12.5])
def test_estimate_weight_memory_rejects_invalid_block_size(bad_block_size: object) -> None:
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        estimate_weight_memory(
            1_000_000, "int4", QuantizationConfig(enabled=True, block_size=bad_block_size)
        )
