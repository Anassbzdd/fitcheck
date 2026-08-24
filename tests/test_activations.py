from __future__ import annotations
import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.memory.activations import estimate_activation_memory

# Golden set: Llama-3.1-8B, bs=4, seq=2048, bf16 (SPEC Appendix).
_GOLDEN_BATCH = 4
_GOLDEN_SEQ = 2048
_A_ACT_CKPT_FLASH = 3_136.0  # L*g*b*s*h (2,048) + A_layer (1,088)
_A_ACT_CKPT_NO_FLASH = 4_160.0  # A_layer gains 1,024 MiB of softmax matrix
_A_ACT_NO_CKPT_FLASH = 34_816.0  # 32 * 1,088
_A_ACT_NO_CKPT_NO_FLASH = 67_584.0  # 32 * 2,112


def _model_config(
    *,
    hidden_size: int = 4096,
    num_layers: int = 32,
    num_attention_heads: int = 32,
    num_kv_heads: int = 8,
    intermediate_size: int = 14336,
) -> ModelConfig:
    return ModelConfig(
        name="test-model",
        num_params=0,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=128256,
        head_dim=hidden_size // num_attention_heads,
        tie_word_embeddings=False,
    )


@pytest.fixture
def llama() -> ModelConfig:
    return _model_config()


def _estimate(
    config: ModelConfig,
    *,
    batch_size: int = _GOLDEN_BATCH,
    seq_len: int = _GOLDEN_SEQ,
    grad_checkpoint: bool = True,
    flash_attn: bool = True,
    precision: str = "bf16",
) -> float:
    return estimate_activation_memory(
        config, batch_size, seq_len, grad_checkpoint, flash_attn, precision
    )


def test_golden_llama_qlora_checkpoint_and_flash(llama: ModelConfig) -> None:
    assert _estimate(llama) == pytest.approx(_A_ACT_CKPT_FLASH, rel=1e-9)


@pytest.mark.parametrize(
    ("grad_checkpoint", "flash_attn", "expected"),
    [
        (True, True, _A_ACT_CKPT_FLASH),
        (True, False, _A_ACT_CKPT_NO_FLASH),
        (False, True, _A_ACT_NO_CKPT_FLASH),
        (False, False, _A_ACT_NO_CKPT_NO_FLASH),
    ],
)
def test_four_paths(
    llama: ModelConfig, grad_checkpoint: bool, flash_attn: bool, expected: float
) -> None:
    result = _estimate(llama, grad_checkpoint=grad_checkpoint, flash_attn=flash_attn)

    assert result == pytest.approx(expected, rel=1e-9)


def test_flash_attn_off_adds_exactly_the_softmax_term(llama: ModelConfig) -> None:
    # g*b*n_h*s^2 per layer = 1,024 MiB; with checkpointing only one layer is live.
    delta = _estimate(llama, flash_attn=False) - _estimate(llama)

    assert delta == pytest.approx(1_024.0, rel=1e-9)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("bf16", _A_ACT_CKPT_FLASH), ("fp16", _A_ACT_CKPT_FLASH), ("fp32", 6_272.0)],
)
def test_scales_with_compute_dtype_never_hardcoded_two(
    llama: ModelConfig, precision: str, expected: float
) -> None:
    assert _estimate(llama, precision=precision) == pytest.approx(expected, rel=1e-9)


def test_reads_intermediate_size_from_config_instead_of_assuming_4h(
    llama: ModelConfig,
) -> None:
    four_h = _model_config(intermediate_size=4 * 4096)

    assert _estimate(four_h) == pytest.approx(3_232.0, rel=1e-9)
    assert _estimate(llama) != pytest.approx(_estimate(four_h), rel=1e-6)


def test_gqa_kv_width_is_smaller_than_mha(llama: ModelConfig) -> None:
    mha = _model_config(num_kv_heads=32)

    assert _estimate(llama) < _estimate(mha)


def test_micro_batch_scales_linearly(llama: ModelConfig) -> None:
    assert _estimate(llama, batch_size=8) == pytest.approx(
        2 * _A_ACT_CKPT_FLASH, rel=1e-9
    )


def test_seq_len_is_linear_with_flash_and_superlinear_without(
    llama: ModelConfig,
) -> None:
    short = _estimate(llama, seq_len=1024, grad_checkpoint=False)
    long = _estimate(llama, seq_len=2048, grad_checkpoint=False)
    short_no_flash = _estimate(llama, seq_len=1024, grad_checkpoint=False, flash_attn=False)
    long_no_flash = _estimate(llama, seq_len=2048, grad_checkpoint=False, flash_attn=False)

    assert long == pytest.approx(2 * short, rel=1e-9)
    assert long_no_flash > 2 * short_no_flash


@pytest.mark.parametrize("bad_value", [0, -1, True, 2.5, "4", None])
def test_rejects_invalid_batch_size_and_seq_len(
    llama: ModelConfig, bad_value: object
) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        _estimate(llama, batch_size=bad_value)
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        _estimate(llama, seq_len=bad_value)


def test_rejects_non_boolean_flags_and_unsupported_precision(llama: ModelConfig) -> None:
    with pytest.raises(ValueError, match="grad_checkpoint must be a boolean"):
        _estimate(llama, grad_checkpoint="yes")
    with pytest.raises(ValueError, match="flash_attn must be a boolean"):
        _estimate(llama, flash_attn=1)
    with pytest.raises(ValueError, match="Unsupported precision"):
        _estimate(llama, precision="fp4")
