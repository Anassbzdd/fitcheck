from __future__ import annotations
import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.memory.activations import (
    _ActivationParts,
    _activation_parts,
    estimate_activation_memory,
)

_GOLDEN_BATCH = 4
_GOLDEN_SEQ = 2048
_GOLDEN_VOCAB = 128_256


_LOGITS = 4 * 4.0 * _GOLDEN_BATCH * _GOLDEN_SEQ * _GOLDEN_VOCAB / 1024**2  # 16,032

# Under checkpointing the peak is 2 * L * gamma * b * s * h (resident) plus whichever
# of the LM-head hump or one layer's recompute is larger -- they never coexist.
_CKPT_STORE = 2 * 2 * 32 * _GOLDEN_BATCH * _GOLDEN_SEQ * 4096 / 1024**2  # 4,096
# bracket = 12h + 3*(n_h*d_k) + 1*(n_kv*d_k) + 3*d_ff = 105,472 for Llama-3.1-8B.
# The hidden-width total of 15 is measured (task 9.2); see docs/SPEC.md Component 5.
_SCORE_MATRIX = 1_024.0  # gamma * b * n_h * s^2, one copy
_A_LAYER_FLASH = 1_648.0
_A_LAYER_NO_FLASH = _A_LAYER_FLASH + 9 * _SCORE_MATRIX      # ckpt: one layer's peak
_A_LAYER_RETAINED = _A_LAYER_FLASH + 2.9 * _SCORE_MATRIX    # no ckpt: every layer keeps

# Llama-3.1-8B has a 128k vocabulary, so the LM-head hump wins both ways here and
# Flash Attention does not move A_act at all at this shape.
_A_ACT_CKPT_FLASH = _CKPT_STORE + max(_LOGITS, _A_LAYER_FLASH)          # 20,128
_A_ACT_CKPT_NO_FLASH = _CKPT_STORE + max(_LOGITS, _A_LAYER_NO_FLASH)    # 20,128
_A_ACT_NO_CKPT_FLASH = 32 * _A_LAYER_FLASH + _LOGITS
_A_ACT_NO_CKPT_NO_FLASH = 32 * _A_LAYER_RETAINED + _LOGITS


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


def test_flash_attn_off_adds_the_retained_copies_in_every_layer(
    llama: ModelConfig,
) -> None:
    delta = _estimate(llama, grad_checkpoint=False, flash_attn=False) - _estimate(
        llama, grad_checkpoint=False
    )

    assert delta == pytest.approx(32 * 2.9 * _SCORE_MATRIX, rel=1e-9)


def test_checkpointed_layer_hump_still_charges_nine_copies(
    llama: ModelConfig,
) -> None:
    seq_len, batch_size = 8192, 1
    score_matrix = 2 * batch_size * 32 * seq_len**2 / 1024**2
    a_layer_flash = 2 * batch_size * seq_len * 105_472 / 1024**2
    store = 2 * 2 * 32 * batch_size * seq_len * 4096 / 1024**2
    logits = 4 * 4.0 * batch_size * seq_len * _GOLDEN_VOCAB / 1024**2

    eager = _estimate(
        llama, seq_len=seq_len, batch_size=batch_size, flash_attn=False
    )

    assert eager == pytest.approx(
        store + max(logits, a_layer_flash + 9 * score_matrix), rel=1e-9
    )


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
    bracket = 12 * 4096 + 3 * 4096 + 1 * 1024 + 3 * (4 * 4096)
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
    # bracket = 12*3584 + 3*(16*256) + 1*(8*256) + 3*14336 = 100,352
    # A_layer = 2 * 4*2048 * 100,352 B = 1,568 MiB
    assert _estimate(gemma2_9b, grad_checkpoint=False) == pytest.approx(
        42 * 1_568.0 + _LOGITS, rel=1e-9
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


def test_exact_bracket_reduces_to_the_hidden_width_form_when_n_h_d_k_equals_h(
    llama: ModelConfig,
) -> None:
    """15 hidden-width tensors, not 6: 12h + 3*(n_h d_k) collapses to 15h whenever
    n_h * d_k == h, which is every model except Gemma-2 in the ground-truth set."""
    gamma, tokens = 2, _GOLDEN_BATCH * _GOLDEN_SEQ
    hidden_width_bracket = (
        15 * llama.hidden_size
        + 1 * llama.hidden_size * llama.num_kv_heads // llama.num_attention_heads
        + 3 * llama.intermediate_size
    )
    a_layer_mib = gamma * tokens * hidden_width_bracket / 1024**2

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


def _parts(config: ModelConfig, *, flash_attn: bool = True) -> _ActivationParts:
    return _activation_parts(config, _GOLDEN_BATCH, _GOLDEN_SEQ, flash_attn, 2.0)


def test_activation_parts_are_the_golden_terms(llama: ModelConfig) -> None:
    parts = _parts(llama)

    assert parts.layer_bytes / 1024**2 == pytest.approx(_A_LAYER_FLASH, rel=1e-9)
    assert parts.logits_bytes / 1024**2 == pytest.approx(_LOGITS, rel=1e-9)
    assert parts.checkpoint_store_bytes / 1024**2 == pytest.approx(_CKPT_STORE, rel=1e-9)
    assert parts.attention_matrix_bytes / 1024**2 == pytest.approx(
        9 * _SCORE_MATRIX, rel=1e-9
    )
    assert parts.retained_attention_matrix_bytes / 1024**2 == pytest.approx(
        2.9 * _SCORE_MATRIX, rel=1e-9
    )


def test_attention_matrix_part_is_nine_gamma_and_only_without_flash(
    llama: ModelConfig,
) -> None:
    eager = _parts(llama, flash_attn=False)
    assert eager.layer_bytes / 1024**2 == pytest.approx(_A_LAYER_NO_FLASH, rel=1e-9)
    assert _parts(llama).layer_bytes / 1024**2 == pytest.approx(_A_LAYER_FLASH, rel=1e-9)
    assert eager.logits_bytes == _parts(llama).logits_bytes
    assert eager.checkpoint_store_bytes == _parts(llama).checkpoint_store_bytes


def test_retained_score_matrix_is_smaller_than_the_transient_one(
    llama: ModelConfig,
) -> None:
    """Two regimes, two constants. With checkpointing on, one layer is live and
    peaks at 9 copies. With it off, every layer stays live and keeps ~2.9 -- so
    charging 9 copies in all L layers at once, as fitcheck used to, cannot happen.
    Measured on a T4: worst-case error on that branch fell from 98.3% to 5.9%."""
    eager = _parts(llama, flash_attn=False)

    assert eager.retained_layer_bytes < eager.layer_bytes
    assert eager.retained_layer_bytes / 1024**2 == pytest.approx(
        _A_LAYER_RETAINED, rel=1e-9
    )


def test_flash_attention_makes_the_two_regimes_identical(llama: ModelConfig) -> None:
    flash = _parts(llama, flash_attn=True)

    assert flash.retained_layer_bytes == flash.layer_bytes


def test_parts_reconstruct_the_public_estimate(llama: ModelConfig) -> None:
    for flash_attn in (True, False):
        parts = _parts(llama, flash_attn=flash_attn)
        checkpointed = parts.checkpoint_store_bytes + max(
            parts.logits_bytes, parts.layer_bytes
        )
        uncheckpointed = (
            llama.num_layers * parts.retained_layer_bytes + parts.logits_bytes
        )

        assert _estimate(llama, flash_attn=flash_attn) == pytest.approx(
            checkpointed / 1024**2, rel=1e-9
        )
        assert _estimate(
            llama, flash_attn=flash_attn, grad_checkpoint=False
        ) == pytest.approx(uncheckpointed / 1024**2, rel=1e-9)


def test_no_checkpointing_branch_is_measured_not_derived(llama: ModelConfig) -> None:
    """Task 9.2 measured this branch on a T4 over 11 rows and 9 models, so the
    'derived, not measured' caveat 8.3 attached to it is gone. Keeping a warning
    that says the branch is unmeasured would now be false."""
    assert _estimate(llama, flash_attn=False, grad_checkpoint=False) == pytest.approx(
        _A_ACT_NO_CKPT_NO_FLASH, rel=1e-9
    )
    assert _estimate(llama, flash_attn=True, grad_checkpoint=False) == pytest.approx(
        _A_ACT_NO_CKPT_FLASH, rel=1e-9
    )
