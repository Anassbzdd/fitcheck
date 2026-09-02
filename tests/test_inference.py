from __future__ import annotations
import inspect
import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.memory.inference import estimate_inference_memory
from fitcheck.utils import bytes_to_mib

# Llama-3.1-8B, the golden model. P is the same count the training components use.
_LLAMA_31_8B_PARAMS = 8_030_261_248
_W_BASE_FP16 = bytes_to_mib(_LLAMA_31_8B_PARAMS * 2)  # 15,316.51

# 2 * 32 layers * 8 kv heads * 128 head_dim * 2048 tokens * 1 request * 2 bytes
_KV_2048_FP16 = 256.0
_KV_BYTES_PER_TOKEN_FP16 = 2 * 32 * 8 * 128 * 2  # 131,072 -> 0.125 MiB/token


def _model_config(
    *,
    num_params: int = _LLAMA_31_8B_PARAMS,
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
        num_params=num_params,
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
def mha() -> ModelConfig:
    """Same shape as `llama` but no GQA: 32 KV heads instead of 8."""
    return _model_config(num_kv_heads=32)


@pytest.fixture
def gemma2_9b() -> ModelConfig:
    """head_dim 256 != 3584 / 16 -- the cache must read head_dim, not hidden_size."""
    return _model_config(
        hidden_size=3584,
        num_layers=42,
        num_attention_heads=16,
        num_kv_heads=8,
        intermediate_size=14336,
        head_dim=256,
    )


def test_golden_llama_31_8b_fp16_single_request(llama: ModelConfig) -> None:
    result = estimate_inference_memory(llama, "fp16", 2048, 1)

    assert result == pytest.approx(_W_BASE_FP16 + _KV_2048_FP16, rel=1e-6)
    assert result == pytest.approx(15_572.5078125, rel=1e-9)


def test_kv_cache_is_the_only_term_above_the_weights(llama: ModelConfig) -> None:
    weights_plus_one_token = estimate_inference_memory(llama, "fp16", 1, 1)
    weights_plus_context = estimate_inference_memory(llama, "fp16", 2048, 1)

    assert weights_plus_context - weights_plus_one_token == pytest.approx(
        _KV_2048_FP16 - bytes_to_mib(_KV_BYTES_PER_TOKEN_FP16), rel=1e-9
    )


def test_no_activation_optimizer_or_gradient_terms(llama: ModelConfig) -> None:
    """Serving is weights plus a small cache -- nothing is kept for a backward pass."""
    result = estimate_inference_memory(llama, "fp16", 2048, 1)

    assert result < _W_BASE_FP16 * 1.02


def test_kv_cache_scales_linearly_with_num_concurrent(llama: ModelConfig) -> None:
    one = estimate_inference_memory(llama, "fp16", 2048, 1)
    eight = estimate_inference_memory(llama, "fp16", 2048, 8)

    assert eight - _W_BASE_FP16 == pytest.approx(8 * _KV_2048_FP16, rel=1e-6)
    assert eight - one == pytest.approx(7 * _KV_2048_FP16, rel=1e-6)


def test_num_concurrent_is_in_the_formula_not_just_the_signature(
    llama: ModelConfig,
) -> None:
    one = estimate_inference_memory(llama, "fp16", 2048, 1)
    four = estimate_inference_memory(llama, "fp16", 2048, 4)

    assert four > one


def test_kv_cache_scales_linearly_with_seq_len(llama: ModelConfig) -> None:
    short_cache = estimate_inference_memory(llama, "fp16", 2048, 1) - _W_BASE_FP16
    long_cache = estimate_inference_memory(llama, "fp16", 8192, 1) - _W_BASE_FP16

    assert long_cache == pytest.approx(4 * short_cache, rel=1e-6)
    assert long_cache == pytest.approx(1024.0, rel=1e-6)


def test_seq_len_and_num_concurrent_are_interchangeable_in_the_cache(
    llama: ModelConfig,
) -> None:
    """Both are linear factors on the same tensor: 4 x 2048 == 1 x 8192 of cache."""
    wide = estimate_inference_memory(llama, "fp16", 2048, 4)
    deep = estimate_inference_memory(llama, "fp16", 8192, 1)

    assert wide == pytest.approx(deep, rel=1e-9)


def test_kv_cache_uses_num_kv_heads_not_num_attention_heads(
    llama: ModelConfig, mha: ModelConfig
) -> None:
    gqa_cache = estimate_inference_memory(llama, "fp16", 2048, 1) - _W_BASE_FP16
    mha_cache = estimate_inference_memory(mha, "fp16", 2048, 1) - _W_BASE_FP16

    assert mha_cache == pytest.approx(4 * gqa_cache, rel=1e-6)
    assert gqa_cache == pytest.approx(_KV_2048_FP16, rel=1e-6)


def test_kv_cache_uses_declared_head_dim_not_hidden_size_over_heads(
    gemma2_9b: ModelConfig,
) -> None:
    cache = estimate_inference_memory(gemma2_9b, "fp16", 2048, 1) - _W_BASE_FP16

    expected = bytes_to_mib(2 * 42 * 8 * 256 * 2048 * 1 * 2)
    assert cache == pytest.approx(expected, rel=1e-9)
    assert cache == pytest.approx(672.0, rel=1e-9)

    wrong = bytes_to_mib(2 * 42 * 8 * (3584 // 16) * 2048 * 1 * 2)
    assert cache != pytest.approx(wrong, rel=1e-3)


def test_leading_two_is_k_and_v_once_not_twice(llama: ModelConfig) -> None:
    cache = estimate_inference_memory(llama, "fp16", 2048, 1) - _W_BASE_FP16

    per_tensor = bytes_to_mib(32 * 8 * 128 * 2048 * 2)
    assert cache == pytest.approx(2 * per_tensor, rel=1e-9)
    assert cache != pytest.approx(4 * per_tensor, rel=1e-3)


@pytest.mark.parametrize(
    ("precision", "bytes_per_element"),
    [("fp32", 4.0), ("fp16", 2.0), ("bf16", 2.0), ("int8", 1.0)],
)
def test_both_terms_follow_the_serving_dtype(
    llama: ModelConfig, precision: str, bytes_per_element: float
) -> None:
    result = estimate_inference_memory(llama, precision, 2048, 1)

    expected_weights = bytes_to_mib(_LLAMA_31_8B_PARAMS * bytes_per_element)
    expected_cache = bytes_to_mib(2 * 32 * 8 * 128 * 2048 * bytes_per_element)
    assert result == pytest.approx(expected_weights + expected_cache, rel=1e-9)


def test_fp32_is_exactly_double_fp16(llama: ModelConfig) -> None:
    fp16 = estimate_inference_memory(llama, "fp16", 2048, 4)
    fp32 = estimate_inference_memory(llama, "fp32", 2048, 4)

    assert fp32 == pytest.approx(2 * fp16, rel=1e-9)


def test_num_concurrent_defaults_to_one(llama: ModelConfig) -> None:
    assert estimate_inference_memory(llama, "fp16", 2048) == (
        estimate_inference_memory(llama, "fp16", 2048, 1)
    )


def test_signature_matches_the_spec() -> None:
    parameters = inspect.signature(estimate_inference_memory).parameters

    assert list(parameters) == ["config", "precision", "seq_len", "num_concurrent"]


@pytest.mark.parametrize("precision", ["FP16", "  fp16  ", "Fp16"])
def test_precision_is_normalized(llama: ModelConfig, precision: str) -> None:
    result = estimate_inference_memory(llama, precision, 2048, 1)

    assert result == pytest.approx(_W_BASE_FP16 + _KV_2048_FP16, rel=1e-6)


def test_returns_mib_not_mb(llama: ModelConfig) -> None:
    cache = estimate_inference_memory(llama, "fp16", 2048, 1) - _W_BASE_FP16

    assert cache == pytest.approx(256.0, rel=1e-9)
    assert cache != pytest.approx(2 * 32 * 8 * 128 * 2048 * 2 / 1_000_000, rel=1e-3)


@pytest.mark.parametrize("bad_seq_len", [0, -1, True, 12.5, "2048", None])
def test_rejects_invalid_seq_len(llama: ModelConfig, bad_seq_len: object) -> None:
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        estimate_inference_memory(llama, "fp16", bad_seq_len, 1)


@pytest.mark.parametrize("bad_concurrent", [0, -1, True, 2.5, "4", None])
def test_rejects_invalid_num_concurrent(
    llama: ModelConfig, bad_concurrent: object
) -> None:
    with pytest.raises(ValueError, match="num_concurrent must be a positive integer"):
        estimate_inference_memory(llama, "fp16", 2048, bad_concurrent)


def test_rejects_unsupported_precision(llama: ModelConfig) -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        estimate_inference_memory(llama, "fp4", 2048, 1)


def test_rejects_non_string_precision(llama: ModelConfig) -> None:
    with pytest.raises(ValueError, match="precision must be a string"):
        estimate_inference_memory(llama, 2, 2048, 1)


@pytest.mark.parametrize("bad_config", [None, "llama-3.1-8b", 42, {"hidden_size": 4096}])
def test_rejects_non_model_config(bad_config: object) -> None:
    with pytest.raises(ValueError, match="config must be a ModelConfig"):
        estimate_inference_memory(bad_config, "fp16", 2048, 1)
