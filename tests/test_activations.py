from __future__ import annotations
import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.memory.activations import estimate_activation_memory

# Golden set: Llama-3.1-8B, bs=4, seq=2048, bf16 .
_GOLDEN_BATCH = 4
_GOLDEN_SEQ = 2048
_GOLDEN_VOCAB = 128_256


_LOGITS = 4 * 4.0 * _GOLDEN_BATCH * _GOLDEN_SEQ * _GOLDEN_VOCAB / 1024**2  # 16,032

# Under checkpointing the peak is 2 * L * gamma * b * s * h (resident) plus whichever
# of the LM-head hump or one layer's recompute is larger -- they never coexist.
_CKPT_STORE = 2 * 2 * 32 * _GOLDEN_BATCH * _GOLDEN_SEQ * 4096 / 1024**2  # 4,096
_A_LAYER_FLASH = 1_088.0
_A_LAYER_NO_FLASH = _A_LAYER_FLASH + 9 * 1_024.0  # 9 gamma copies of b*n_h*s^2

# Llama-3.1-8B has a 128k vocabulary, so the LM-head hump wins both ways here and
# Flash Attention does not move A_act at all at this shape.
_A_ACT_CKPT_FLASH = _CKPT_STORE + max(_LOGITS, _A_LAYER_FLASH)          # 20,128
_A_ACT_CKPT_NO_FLASH = _CKPT_STORE + max(_LOGITS, _A_LAYER_NO_FLASH)    # 20,128
_A_ACT_NO_CKPT_FLASH = 32 * _A_LAYER_FLASH + _LOGITS
_A_ACT_NO_CKPT_NO_FLASH = 32 * _A_LAYER_NO_FLASH + _LOGITS


def _model_config(
    *,
    hidden_size: int = 4096,
    num_layers: int = 32,
    num_attention_heads: int = 32,
    num_kv_heads: int = 8,
    intermediate_size: int = 14336,
    head_dim: int | None = None,
    vocab_size: int = 128256,
) -> ModelConfig:
    return ModelConfig(
        name="test-model",
        num_params=0,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=hidden_size // num_attention_heads if head_dim is None else head_dim,
        tie_word_embeddings=False,
    )


@pytest.fixture
def llama() -> ModelConfig:
    return _model_config()


@pytest.fixture
def gemma2_9b() -> ModelConfig:
    """Gemma-2-9B: n_h*d_k = 16*256 = 4096, wider than hidden_size = 3584."""
    return _model_config(
        hidden_size=3584,
        num_layers=42,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=14336,
        head_dim=256,
    )


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


def test_flash_attn_off_adds_nine_gamma_copies_of_the_score_matrix(
    llama: ModelConfig,
) -> None:
    # No checkpointing, so the layer hump is not hidden behind the max().
    delta = _estimate(llama, grad_checkpoint=False, flash_attn=False) - _estimate(
        llama, grad_checkpoint=False
    )

    assert delta == pytest.approx(32 * 9 * 1_024.0, rel=1e-9)


def test_flash_attn_does_not_help_when_the_lm_head_hump_dominates(
    llama: ModelConfig,
) -> None:
    """128k vocab: the logits hump beats the layer hump, so the max() picks it either
    way and Flash Attention buys nothing at this shape. Measured, not assumed --
    SmolLM2 eager vs SDPA differed by 16 MiB out of 5,297."""
    assert _estimate(llama, flash_attn=False) == pytest.approx(
        _estimate(llama), rel=1e-9
    )


def test_flash_attn_does_help_once_the_layer_hump_wins(llama: ModelConfig) -> None:
    """Small vocab and a long sequence put the layer hump on top, and then removing
    the score matrix is worth exactly nine gamma copies of it."""
    small_vocab = _model_config(vocab_size=32_000)

    delta = _estimate(small_vocab, seq_len=4096, flash_attn=False) - _estimate(
        small_vocab, seq_len=4096
    )

    assert delta > 0


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("bf16", _A_ACT_CKPT_FLASH),
        ("fp16", _A_ACT_CKPT_FLASH),
        ("fp32", 2 * _CKPT_STORE + max(_LOGITS, 2 * _A_LAYER_FLASH)),
    ],
)
def test_scales_with_compute_dtype_never_hardcoded_two(
    llama: ModelConfig, precision: str, expected: float
) -> None:
    assert _estimate(llama, precision=precision) == pytest.approx(expected, rel=1e-9)


def test_reads_intermediate_size_from_config_instead_of_assuming_4h(
    llama: ModelConfig,
) -> None:
    # Checkpointing off, so the layer hump is visible instead of hidden by the max().
    four_h = _model_config(intermediate_size=4 * 4096)
    bracket = 4 * 4096 + 2 * 4096 + 2 * 1024 + 3 * (4 * 4096)
    a_layer = 2 * _GOLDEN_BATCH * _GOLDEN_SEQ * bracket / 1024**2

    assert _estimate(four_h, grad_checkpoint=False) == pytest.approx(
        32 * a_layer + _LOGITS, rel=1e-9
    )
    assert _estimate(four_h, grad_checkpoint=False) != pytest.approx(
        _estimate(llama, grad_checkpoint=False), rel=1e-6
    )


def test_gqa_kv_width_is_smaller_than_mha(llama: ModelConfig) -> None:
    mha = _model_config(num_kv_heads=32)

    assert _estimate(llama, grad_checkpoint=False) < _estimate(
        mha, grad_checkpoint=False
    )


def test_gemma2_uses_head_dim_not_hidden_size_for_q_and_attn_output(
    gemma2_9b: ModelConfig,
) -> None:
    # bracket = 4*3584 + 2*(16*256) + 2*(8*256) + 3*14336 = 69,632
    # A_layer = 2 * 4*2048 * 69,632 B = 1,088 MiB
    assert _estimate(gemma2_9b, grad_checkpoint=False) == pytest.approx(
        42 * 1_088.0 + _LOGITS, rel=1e-9
    )


def test_gemma2_differs_from_the_n_h_d_k_equals_h_assumption(
    gemma2_9b: ModelConfig,
) -> None:
    # Same model with the Llama-shaped assumption baked in (d_k = h/n_h = 224).
    as_if_square = _model_config(
        hidden_size=3584,
        num_layers=42,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=14336,
    )

    assert _estimate(gemma2_9b, grad_checkpoint=False) > _estimate(
        as_if_square, grad_checkpoint=False
    )


def test_exact_bracket_reduces_to_the_six_h_form_when_n_h_d_k_equals_h(
    llama: ModelConfig,
) -> None:
    gamma, tokens = 2, _GOLDEN_BATCH * _GOLDEN_SEQ
    six_h_bracket = (
        6 * llama.hidden_size
        + 2 * llama.hidden_size * llama.num_kv_heads // llama.num_attention_heads
        + 3 * llama.intermediate_size
    )
    a_layer_mib = gamma * tokens * six_h_bracket / 1024**2

    assert a_layer_mib == pytest.approx(_A_LAYER_FLASH, rel=1e-9)
    assert _estimate(llama, grad_checkpoint=False) == pytest.approx(
        llama.num_layers * a_layer_mib + _LOGITS, rel=1e-9
    )


def test_checkpoint_store_is_two_hidden_state_tensors_per_layer(
    llama: ModelConfig,
) -> None:
    """The measured multiplier is 2, not 1. Bumping it to 1 moved worst-case error
    on the 20-run T4 set from 4.8% to 10.4%."""
    resident = _estimate(llama) - max(_LOGITS, _A_LAYER_FLASH)

    assert resident == pytest.approx(
        2 * 2 * llama.num_layers * _GOLDEN_BATCH * _GOLDEN_SEQ * llama.hidden_size
        / 1024**2,
        rel=1e-9,
    )


def test_checkpointed_peak_is_a_max_not_a_sum(llama: ModelConfig) -> None:
    """The LM-head hump and one layer's recompute are both transient and never
    overlap, so summing them over-counts."""
    summed = _CKPT_STORE + _LOGITS + _A_LAYER_FLASH

    assert _estimate(llama) < summed
    assert _estimate(llama) == pytest.approx(
        _CKPT_STORE + max(_LOGITS, _A_LAYER_FLASH), rel=1e-9
    )


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
