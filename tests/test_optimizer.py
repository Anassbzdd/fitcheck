from __future__ import annotations
import pytest
from fitcheck.memory.optimizer import estimate_optimizer_memory
from fitcheck.utils import bytes_to_mib

_MIB = 1024**2

_GOLDEN_LORA_PARAMS = 54_525_952
_GOLDEN_S_OPTIM_MIB = 416.0

_LLAMA_31_8B_PARAMS = 8_030_261_248


def test_estimate_optimizer_memory_golden_qlora_adamw_fp32() -> None:
    result = estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "adamw", is_lora=True)

    assert result == pytest.approx(_GOLDEN_S_OPTIM_MIB, rel=1e-9)
    assert result == pytest.approx(_GOLDEN_LORA_PARAMS * 8 / _MIB, rel=1e-9)


def test_estimate_optimizer_memory_golden_is_four_times_gradients() -> None:
    s_optim = estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "adamw", is_lora=True)
    g_grad_mib = bytes_to_mib(_GOLDEN_LORA_PARAMS * 2)

    assert g_grad_mib == pytest.approx(104.0, rel=1e-9)
    assert s_optim == pytest.approx(g_grad_mib * 4, rel=1e-9)


@pytest.mark.parametrize(
    ("optimizer", "bytes_per_param"),
    [
        ("adamw", 8),
        ("adam8bit", 2),
        ("adamw8bit", 2),
        ("adamw-8bit", 2),
        ("adam-8bit", 2),
        ("sgd-momentum", 4),
        ("sgd", 0),
        ("sgd-nomomentum", 0),
    ],
)
def test_estimate_optimizer_memory_lora_bytes_per_param_table(
    optimizer: str, bytes_per_param: int
) -> None:
    result = estimate_optimizer_memory(1_000_000, optimizer, is_lora=True)

    assert result == pytest.approx(bytes_to_mib(1_000_000 * bytes_per_param), rel=1e-9)


@pytest.mark.parametrize(
    ("optimizer_dtype", "bytes_per_param"),
    [("fp32", 8), ("bf16", 4)],
)
def test_estimate_optimizer_memory_adamw_state_dtype(
    optimizer_dtype: str, bytes_per_param: int
) -> None:
    result = estimate_optimizer_memory(
        _GOLDEN_LORA_PARAMS, "adamw", is_lora=True, optimizer_dtype=optimizer_dtype
    )

    expected = bytes_to_mib(_GOLDEN_LORA_PARAMS * bytes_per_param)
    assert result == pytest.approx(expected, rel=1e-9)


def test_estimate_optimizer_memory_adamw_defaults_to_fp32_states() -> None:
    default = estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "adamw", is_lora=True)
    explicit_fp32 = estimate_optimizer_memory(
        _GOLDEN_LORA_PARAMS, "adamw", is_lora=True, optimizer_dtype="fp32"
    )

    assert default == explicit_fp32
    assert default == pytest.approx(_GOLDEN_S_OPTIM_MIB, rel=1e-9)


def test_estimate_optimizer_memory_defaults_to_lora() -> None:
    assert estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "adamw") == pytest.approx(
        estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "adamw", is_lora=True), rel=1e-9
    )


@pytest.mark.parametrize(
    ("optimizer", "bytes_per_param"),
    [
        ("adamw", 8 + 4),
        ("adam8bit", 2 + 4),
        ("sgd-momentum", 4 + 4),
        ("sgd", 0 + 4),
    ],
)
def test_estimate_optimizer_memory_full_ft_adds_master_weight_copy(
    optimizer: str, bytes_per_param: int
) -> None:
    result = estimate_optimizer_memory(
        1_000_000, optimizer, is_lora=False, precision="bf16"
    )

    assert result == pytest.approx(bytes_to_mib(1_000_000 * bytes_per_param), rel=1e-9)


def test_estimate_optimizer_memory_full_ft_llama_8b_adamw_is_enormous() -> None:
    result = estimate_optimizer_memory(
        _LLAMA_31_8B_PARAMS, "adamw", is_lora=False, precision="bf16"
    )

    expected = bytes_to_mib(_LLAMA_31_8B_PARAMS * 12)
    assert result == pytest.approx(expected, rel=1e-9)
    assert result == pytest.approx(91_875.0, rel=1e-3)


def test_estimate_optimizer_memory_full_ft_exceeds_lora_by_master_copy() -> None:
    lora = estimate_optimizer_memory(
        _GOLDEN_LORA_PARAMS, "adamw", is_lora=True, precision="bf16"
    )
    full = estimate_optimizer_memory(
        _GOLDEN_LORA_PARAMS, "adamw", is_lora=False, precision="bf16"
    )

    master_copy_mib = bytes_to_mib(_GOLDEN_LORA_PARAMS * 4)
    assert full - lora == pytest.approx(master_copy_mib, rel=1e-9)


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_estimate_optimizer_memory_full_ft_mixed_precision_keeps_master_copy(
    precision: str,
) -> None:
    result = estimate_optimizer_memory(
        1_000_000, "adamw", is_lora=False, precision=precision
    )

    assert result == pytest.approx(bytes_to_mib(1_000_000 * (8 + 4)), rel=1e-9)


def test_estimate_optimizer_memory_full_ft_fp32_has_no_master_copy() -> None:
    result = estimate_optimizer_memory(
        1_000_000, "adamw", is_lora=False, precision="fp32"
    )

    assert result == pytest.approx(bytes_to_mib(1_000_000 * 8), rel=1e-9)


def test_estimate_optimizer_memory_full_ft_fp32_does_not_double_count_llama_8b() -> None:
    result = estimate_optimizer_memory(
        _LLAMA_31_8B_PARAMS, "adamw", is_lora=False, precision="fp32"
    )

    double_counted = bytes_to_mib(_LLAMA_31_8B_PARAMS * 12)
    assert result == pytest.approx(bytes_to_mib(_LLAMA_31_8B_PARAMS * 8), rel=1e-9)
    assert double_counted - result == pytest.approx(30_633.0, rel=1e-3)


@pytest.mark.parametrize(
    ("precision", "w_base_bytes", "g_grad_bytes"),
    [("bf16", 2, 2), ("fp16", 2, 2), ("fp32", 4, 4)],
)
def test_full_ft_adamw_totals_sixteen_bytes_per_param_in_every_precision(
    precision: str, w_base_bytes: int, g_grad_bytes: int
) -> None:
    params = 1_000_000
    s_optim = estimate_optimizer_memory(
        params, "adamw", is_lora=False, precision=precision
    )
    w_base = bytes_to_mib(params * w_base_bytes)
    g_grad = bytes_to_mib(params * g_grad_bytes)

    assert w_base + g_grad + s_optim == pytest.approx(
        bytes_to_mib(params * 16), rel=1e-9
    )


def test_estimate_optimizer_memory_lora_ignores_precision() -> None:
    results = {
        precision: estimate_optimizer_memory(
            _GOLDEN_LORA_PARAMS, "adamw", is_lora=True, precision=precision
        )
        for precision in ("fp32", "fp16", "bf16")
    }

    assert set(results.values()) == {_GOLDEN_S_OPTIM_MIB}


def test_estimate_optimizer_memory_master_copy_keyed_on_precision_not_optimizer() -> None:
    sgd_mixed = estimate_optimizer_memory(
        1_000_000, "sgd", is_lora=False, precision="bf16"
    )
    sgd_fp32 = estimate_optimizer_memory(
        1_000_000, "sgd", is_lora=False, precision="fp32"
    )

    assert sgd_mixed == pytest.approx(bytes_to_mib(1_000_000 * 4), rel=1e-9)
    assert sgd_fp32 == 0.0


def test_estimate_optimizer_memory_sgd_lora_is_free() -> None:
    assert estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "sgd", is_lora=True) == 0.0


def test_estimate_optimizer_memory_scales_linearly_with_trainable_params() -> None:
    small = estimate_optimizer_memory(1_000_000, "adamw", is_lora=True)
    large = estimate_optimizer_memory(4_000_000, "adamw", is_lora=True)

    assert large == pytest.approx(small * 4, rel=1e-9)


def test_estimate_optimizer_memory_only_counts_trainable_params() -> None:
    adapters = estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, "adamw", is_lora=True)
    whole_model = estimate_optimizer_memory(_LLAMA_31_8B_PARAMS, "adamw", is_lora=True)

    assert adapters < whole_model / 100


@pytest.mark.parametrize("optimizer", ["ADAMW", "  adamw  ", "AdamW"])
def test_estimate_optimizer_memory_optimizer_name_is_normalized(optimizer: str) -> None:
    result = estimate_optimizer_memory(_GOLDEN_LORA_PARAMS, optimizer, is_lora=True)

    assert result == pytest.approx(_GOLDEN_S_OPTIM_MIB, rel=1e-9)


def test_estimate_optimizer_memory_returns_mib_not_mb() -> None:
    result = estimate_optimizer_memory(1_048_576, "adamw", is_lora=True)

    assert result == pytest.approx(8.0, rel=1e-9)
    assert result != pytest.approx(1_048_576 * 8 / 1_000_000, rel=1e-3)


@pytest.mark.parametrize("bad_params", [0, -1, True, 12.5, "1000", None])
def test_estimate_optimizer_memory_rejects_invalid_trainable_params(
    bad_params: object,
) -> None:
    with pytest.raises(ValueError, match="trainable_params must be a positive integer"):
        estimate_optimizer_memory(bad_params, "adamw", is_lora=True)


@pytest.mark.parametrize("bad_is_lora", [1, 0, "true", None])
def test_estimate_optimizer_memory_rejects_non_boolean_is_lora(bad_is_lora: object) -> None:
    with pytest.raises(ValueError, match="is_lora must be a boolean"):
        estimate_optimizer_memory(1_000_000, "adamw", is_lora=bad_is_lora)


def test_estimate_optimizer_memory_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        estimate_optimizer_memory(1_000_000, "lion", is_lora=True)


def test_estimate_optimizer_memory_rejects_non_string_optimizer() -> None:
    with pytest.raises(ValueError, match="optimizer must be a string"):
        estimate_optimizer_memory(1_000_000, 8, is_lora=True)


def test_estimate_optimizer_memory_rejects_unsupported_optimizer_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported optimizer_dtype"):
        estimate_optimizer_memory(
            1_000_000, "adamw", is_lora=True, optimizer_dtype="int8"
        )


@pytest.mark.parametrize("is_lora", [True, False])
def test_estimate_optimizer_memory_rejects_unsupported_precision(is_lora: bool) -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        estimate_optimizer_memory(1_000_000, "adamw", is_lora=is_lora, precision="fp4")


def test_estimate_optimizer_memory_defaults_to_mixed_precision() -> None:
    default = estimate_optimizer_memory(1_000_000, "adamw", is_lora=False)

    assert default == pytest.approx(bytes_to_mib(1_000_000 * 12), rel=1e-9)
