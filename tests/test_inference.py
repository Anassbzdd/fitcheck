from __future__ import annotations
import inspect
import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.memory.inference import InferenceMemory, estimate_inference_memory
from fitcheck.utils import bytes_to_mib

_LLAMA_31_8B_PARAMS = 8_030_261_248
_W_BASE_FP16 = bytes_to_mib(_LLAMA_31_8B_PARAMS * 2) 

_KV_2048_FP16 = 256.0
_KV_BYTES_PER_TOKEN_FP16 = 2 * 32 * 8 * 128 * 2 


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
    return _model_config(num_kv_heads=32)


@pytest.fixture
def gemma2_9b() -> ModelConfig:
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

    assert result.weight_mib == pytest.approx(15_316.5078125, rel=1e-9)
    assert result.kv_cache_mib == pytest.approx(_KV_2048_FP16, rel=1e-9)
    assert result.total_mib == pytest.approx(_W_BASE_FP16 + _KV_2048_FP16, rel=1e-6)
    assert result.total_mib == pytest.approx(15_572.5078125, rel=1e-9)


def test_kv_cache_is_the_only_term_above_the_weights(llama: ModelConfig) -> None:
    weights_plus_one_token = estimate_inference_memory(llama, "fp16", 1, 1).total_mib
    weights_plus_context = estimate_inference_memory(llama, "fp16", 2048, 1).total_mib

    assert weights_plus_context - weights_plus_one_token == pytest.approx(
        _KV_2048_FP16 - bytes_to_mib(_KV_BYTES_PER_TOKEN_FP16), rel=1e-9
    )


def test_no_activation_optimizer_or_gradient_terms(llama: ModelConfig) -> None:
    result = estimate_inference_memory(llama, "fp16", 2048, 1)

    assert result.total_mib < _W_BASE_FP16 * 1.02


def test_kv_cache_scales_linearly_with_num_concurrent(llama: ModelConfig) -> None:
    one = estimate_inference_memory(llama, "fp16", 2048, 1)
    eight = estimate_inference_memory(llama, "fp16", 2048, 8)

    assert eight.kv_cache_mib == pytest.approx(8 * _KV_2048_FP16, rel=1e-6)
    assert eight.total_mib - one.total_mib == pytest.approx(
        7 * _KV_2048_FP16, rel=1e-6
    )


def test_num_concurrent_is_in_the_formula_not_just_the_signature(
    llama: ModelConfig,
) -> None:
    one = estimate_inference_memory(llama, "fp16", 2048, 1)
    four = estimate_inference_memory(llama, "fp16", 2048, 4)

    assert four.total_mib > one.total_mib


def test_kv_cache_scales_linearly_with_seq_len(llama: ModelConfig) -> None:
    short_cache = estimate_inference_memory(llama, "fp16", 2048, 1).kv_cache_mib
    long_cache = estimate_inference_memory(llama, "fp16", 8192, 1).kv_cache_mib

    assert long_cache == pytest.approx(4 * short_cache, rel=1e-6)
    assert long_cache == pytest.approx(1024.0, rel=1e-6)


def test_seq_len_and_num_concurrent_are_interchangeable_in_the_cache(
    llama: ModelConfig,
) -> None:
    wide = estimate_inference_memory(llama, "fp16", 2048, 4)
    deep = estimate_inference_memory(llama, "fp16", 8192, 1)

    assert wide.total_mib == pytest.approx(deep.total_mib, rel=1e-9)


def test_kv_cache_uses_num_kv_heads_not_num_attention_heads(
    llama: ModelConfig, mha: ModelConfig
) -> None:
    gqa_cache = estimate_inference_memory(llama, "fp16", 2048, 1).kv_cache_mib
    mha_cache = estimate_inference_memory(mha, "fp16", 2048, 1).kv_cache_mib

    assert mha_cache == pytest.approx(4 * gqa_cache, rel=1e-6)
    assert gqa_cache == pytest.approx(_KV_2048_FP16, rel=1e-6)


def test_kv_cache_uses_declared_head_dim_not_hidden_size_over_heads(
    gemma2_9b: ModelConfig,
) -> None:
    cache = estimate_inference_memory(gemma2_9b, "fp16", 2048, 1).kv_cache_mib

    expected = bytes_to_mib(2 * 42 * 8 * 256 * 2048 * 1 * 2)
    assert cache == pytest.approx(expected, rel=1e-9)
    assert cache == pytest.approx(672.0, rel=1e-9)

    wrong = bytes_to_mib(2 * 42 * 8 * (3584 // 16) * 2048 * 1 * 2)
    assert cache != pytest.approx(wrong, rel=1e-3)


def test_leading_two_is_k_and_v_once_not_twice(llama: ModelConfig) -> None:
    cache = estimate_inference_memory(llama, "fp16", 2048, 1).kv_cache_mib

    per_tensor = bytes_to_mib(32 * 8 * 128 * 2048 * 2)
    assert cache == pytest.approx(2 * per_tensor, rel=1e-9)
    assert cache != pytest.approx(4 * per_tensor, rel=1e-3)


@pytest.mark.parametrize(
    ("precision", "bytes_per_element"),
    [("fp32", 4.0), ("fp16", 2.0), ("bf16", 2.0)],
)
def test_both_terms_follow_the_compute_dtype_when_unquantized(
    llama: ModelConfig, precision: str, bytes_per_element: float
) -> None:
    result = estimate_inference_memory(llama, precision, 2048, 1)

    expected_weights = bytes_to_mib(_LLAMA_31_8B_PARAMS * bytes_per_element)
    expected_cache = bytes_to_mib(2 * 32 * 8 * 128 * 2048 * bytes_per_element)
    assert result.weight_mib == pytest.approx(expected_weights, rel=1e-9)
    assert result.kv_cache_mib == pytest.approx(expected_cache, rel=1e-9)


def test_fp32_is_exactly_double_fp16(llama: ModelConfig) -> None:
    fp16 = estimate_inference_memory(llama, "fp16", 2048, 4)
    fp32 = estimate_inference_memory(llama, "fp32", 2048, 4)

    assert fp32.total_mib == pytest.approx(2 * fp16.total_mib, rel=1e-9)


def test_num_concurrent_defaults_to_one(llama: ModelConfig) -> None:
    assert estimate_inference_memory(llama, "fp16", 2048) == (
        estimate_inference_memory(llama, "fp16", 2048, 1)
    )


def test_signature_matches_the_spec() -> None:
    parameters = inspect.signature(estimate_inference_memory).parameters

    assert list(parameters) == [
        "config",
        "precision",
        "seq_len",
        "num_concurrent",
        "quantization",
        "double_quant",
    ]


@pytest.mark.parametrize("precision", ["FP16", "  fp16  ", "Fp16"])
def test_precision_is_normalized(llama: ModelConfig, precision: str) -> None:
    result = estimate_inference_memory(llama, precision, 2048, 1)

    assert result.total_mib == pytest.approx(_W_BASE_FP16 + _KV_2048_FP16, rel=1e-6)


def test_returns_mib_not_mb(llama: ModelConfig) -> None:
    cache = estimate_inference_memory(llama, "fp16", 2048, 1).kv_cache_mib

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


# --- Gap 2 from task 6.4: precision and quantization are two axes, not one ---


def test_quantization_defaults_to_none(llama: ModelConfig) -> None:
    assert estimate_inference_memory(
        llama, "fp16", 2048, 1
    ) == estimate_inference_memory(llama, "fp16", 2048, 1, "none", False)


def test_quantizing_the_base_leaves_the_cache_at_the_compute_dtype(
    llama: ModelConfig,
) -> None:
    """The whole point of the split: --quant nf4 must not shrink the KV cache 4x."""
    unquantized = estimate_inference_memory(llama, "fp16", 2048, 1)
    nf4 = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")

    assert nf4.kv_cache_mib == pytest.approx(unquantized.kv_cache_mib, rel=1e-9)
    assert nf4.kv_cache_mib == pytest.approx(_KV_2048_FP16, rel=1e-9)
    assert nf4.weight_mib < unquantized.weight_mib


def test_nf4_weights_are_packed_plus_scales_plus_a_float_slice(
    llama: ModelConfig,
) -> None:
    result = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")

    float_params = llama.num_unquantized_params
    quantized_params = _LLAMA_31_8B_PARAMS - float_params

    packed = bytes_to_mib(quantized_params * 0.5)
    scales = bytes_to_mib(quantized_params * 4 / 64)
    float_slice = bytes_to_mib(float_params * 2)

    assert result.weight_mib == pytest.approx(packed + scales + float_slice, rel=1e-9)
    assert result.weight_mib == pytest.approx(5_748.5078125, rel=1e-6)


def test_scale_overhead_is_not_skipped_under_quantization(llama: ModelConfig) -> None:
    """`--precision int4` used to bill 0.5 B/param and no absmax at all."""
    nf4 = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")

    quantized_params = _LLAMA_31_8B_PARAMS - llama.num_unquantized_params
    naive = bytes_to_mib(quantized_params * 0.5) + bytes_to_mib(
        llama.num_unquantized_params * 2
    )
    assert nf4.weight_mib - naive == pytest.approx(
        bytes_to_mib(quantized_params * 4 / 64), rel=1e-9
    )


def test_double_quant_halves_the_scale_overhead(llama: ModelConfig) -> None:
    plain = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4", False)
    doubled = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4", True)

    quantized_params = _LLAMA_31_8B_PARAMS - llama.num_unquantized_params
    saved = bytes_to_mib(quantized_params * 4 / 64) / 2
    assert plain.weight_mib - doubled.weight_mib == pytest.approx(saved, rel=1e-6)


def test_embeddings_and_lm_head_are_never_quantized(llama: ModelConfig) -> None:
    """A 128k-vocab model keeps 1.05B params out of the 4-bit slice."""
    nf4 = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")

    everything_quantized = bytes_to_mib(
        _LLAMA_31_8B_PARAMS * 0.5 + _LLAMA_31_8B_PARAMS * 4 / 64
    )
    assert nf4.weight_mib > everything_quantized


def test_serving_keeps_the_float_slice_in_the_compute_dtype_not_fp32(
    llama: ModelConfig,
) -> None:
    """Training upcasts embeddings to fp32; serving has no
    `prepare_model_for_kbit_training`, so billing 4 B/param here would over-count."""
    nf4_fp16 = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")
    nf4_fp32 = estimate_inference_memory(llama, "fp32", 2048, 1, "nf4")

    upcast_cost = bytes_to_mib(llama.num_unquantized_params * 2)
    assert nf4_fp32.weight_mib - nf4_fp16.weight_mib == pytest.approx(
        upcast_cost, rel=1e-6
    )
    assert upcast_cost == pytest.approx(2_004.5, rel=1e-3)


def test_int8_halves_the_quantized_slice_against_fp16(llama: ModelConfig) -> None:
    nf4 = estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")
    int8 = estimate_inference_memory(llama, "fp16", 2048, 1, "int8")

    assert int8.weight_mib > nf4.weight_mib
    assert int8.weight_mib == pytest.approx(9_076.5078125, rel=1e-6)


def test_quantized_and_unquantized_caches_are_identical(llama: ModelConfig) -> None:
    caches = {
        quantization: estimate_inference_memory(
            llama, "fp16", 4096, 3, quantization
        ).kv_cache_mib
        for quantization in ("none", "nf4", "int8")
    }
    assert len(set(caches.values())) == 1


def test_rejects_a_storage_dtype_as_the_compute_precision(llama: ModelConfig) -> None:
    """`--precision int4` is exactly the mistake gap 2 describes."""
    for storage_dtype in ("int4", "nf4", "int8", "fp8"):
        with pytest.raises(ValueError, match="Unsupported precision"):
            estimate_inference_memory(llama, storage_dtype, 2048, 1)


@pytest.mark.parametrize("bad_quantization", ["fp4", "gptq", "awq", ""])
def test_rejects_unsupported_quantization(
    llama: ModelConfig, bad_quantization: str
) -> None:
    with pytest.raises(ValueError, match="Unsupported quantization"):
        estimate_inference_memory(llama, "fp16", 2048, 1, bad_quantization)


@pytest.mark.parametrize("bad_quantization", [None, 4, True])
def test_rejects_non_string_quantization(
    llama: ModelConfig, bad_quantization: object
) -> None:
    with pytest.raises(ValueError, match="quantization must be a string"):
        estimate_inference_memory(llama, "fp16", 2048, 1, bad_quantization)


@pytest.mark.parametrize("bad_flag", [None, 1, "yes"])
def test_rejects_non_boolean_double_quant(
    llama: ModelConfig, bad_flag: object
) -> None:
    with pytest.raises(ValueError, match="double_quant must be a boolean"):
        estimate_inference_memory(llama, "fp16", 2048, 1, "nf4", bad_flag)


@pytest.mark.parametrize("quantization", ["NF4", "  nf4  ", "Nf4"])
def test_quantization_is_normalized(llama: ModelConfig, quantization: str) -> None:
    normalized = estimate_inference_memory(llama, "fp16", 2048, 1, quantization)

    assert normalized == estimate_inference_memory(llama, "fp16", 2048, 1, "nf4")


def test_returns_the_two_terms_separately_so_overhead_can_be_priced(
    llama: ModelConfig,
) -> None:
    """Gap 1: a caller needs the split to call `estimate_overhead(weights, kv)`."""
    result = estimate_inference_memory(llama, "fp16", 2048, 1)

    assert isinstance(result, InferenceMemory)
    assert result.total_mib == result.weight_mib + result.kv_cache_mib


def test_rejects_quantizing_a_model_with_no_quantizable_weights() -> None:
    """Degenerate config: nothing left once embeddings and norms are carved out."""
    all_embeddings = _model_config(num_params=1_000)

    with pytest.raises(ValueError, match="no quantizable weights"):
        estimate_inference_memory(all_embeddings, "fp16", 2048, 1, "nf4")
