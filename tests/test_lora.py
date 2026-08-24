from __future__ import annotations
import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.memory.lora import (
    LORA_TARGETS_FULL,
    LORA_TARGETS_MINIMAL,
    LORA_TARGETS_STANDARD,
    estimate_lora_memory,
)
from fitcheck.utils import bytes_to_mib

_MIB = 1024**2


def _model_config(
    *,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int = 32000,
    tie_word_embeddings: bool = False,
    name: str = "test-model",
) -> ModelConfig:
    return ModelConfig(
        name=name,
        num_params=0,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=hidden_size // num_attention_heads,
        tie_word_embeddings=tie_word_embeddings,
    )


@pytest.fixture
def llama_31_8b() -> ModelConfig:
    return _model_config(
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        name="Llama-3.1-8B",
    )


@pytest.fixture
def mha_config() -> ModelConfig:
    return _model_config(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_kv_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
    )


@pytest.fixture
def mqa_config() -> ModelConfig:
    return _model_config(
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=1,
        intermediate_size=5504,
        vocab_size=32000,
    )


def test_estimate_lora_memory_llama_31_8b_qkvo_matches_worked_example(
    llama_31_8b: ModelConfig,
) -> None:
    result = estimate_lora_memory(llama_31_8b, rank=64, targets=LORA_TARGETS_STANDARD, precision="bf16")

    total_params = 32 * 1_703_936
    assert total_params == 54_525_952
    expected_mib = round(total_params * 2) / _MIB
    assert result == pytest.approx(expected_mib, rel=1e-9)
    assert result == pytest.approx(104, rel=0.05)


def test_estimate_lora_memory_is_gqa_aware_kv_uses_kv_heads_not_hidden_size(
    llama_31_8b: ModelConfig,
) -> None:
    result = estimate_lora_memory(llama_31_8b, rank=64, targets=["k_proj"], precision="bf16")

    kv_out = llama_31_8b.num_kv_heads * llama_31_8b.head_dim
    assert kv_out == 1024
    expected_params = llama_31_8b.num_layers * 64 * (llama_31_8b.hidden_size + kv_out)
    expected_mib = bytes_to_mib(round(expected_params * 2))
    assert result == pytest.approx(expected_mib, rel=1e-9)

    wrong_params = llama_31_8b.num_layers * 64 * (llama_31_8b.hidden_size + llama_31_8b.hidden_size)
    wrong_mib = bytes_to_mib(round(wrong_params * 2))
    assert result < wrong_mib


def test_estimate_lora_memory_mha_kv_heads_equal_hidden_size(mha_config: ModelConfig) -> None:
    result = estimate_lora_memory(mha_config, rank=8, targets=["k_proj"], precision="fp16")

    expected_params = mha_config.num_layers * 8 * (mha_config.hidden_size + mha_config.hidden_size)
    expected_mib = bytes_to_mib(round(expected_params * 2))
    assert result == pytest.approx(expected_mib, rel=1e-9)


def test_estimate_lora_memory_mqa_single_kv_head(mqa_config: ModelConfig) -> None:
    result = estimate_lora_memory(mqa_config, rank=16, targets=["v_proj"], precision="bf16")

    kv_out = mqa_config.num_kv_heads * mqa_config.head_dim
    assert kv_out == mqa_config.head_dim  # single KV head
    expected_params = mqa_config.num_layers * 16 * (mqa_config.hidden_size + kv_out)
    expected_mib = bytes_to_mib(round(expected_params * 2))
    assert result == pytest.approx(expected_mib, rel=1e-9)


@pytest.mark.parametrize(
    ("targets", "expected_target_count"),
    [
        (LORA_TARGETS_MINIMAL, 2),
        (LORA_TARGETS_STANDARD, 4),
        (LORA_TARGETS_FULL, 7),
    ],
)
def test_estimate_lora_memory_target_presets_scale_with_target_count(
    llama_31_8b: ModelConfig, targets: tuple[str, ...], expected_target_count: int
) -> None:
    assert len(targets) == expected_target_count

    result = estimate_lora_memory(llama_31_8b, rank=16, targets=targets, precision="bf16")
    assert result > 0


def test_estimate_lora_memory_full_targets_exceed_standard_targets(llama_31_8b: ModelConfig) -> None:
    minimal = estimate_lora_memory(llama_31_8b, rank=16, targets=LORA_TARGETS_MINIMAL, precision="bf16")
    standard = estimate_lora_memory(llama_31_8b, rank=16, targets=LORA_TARGETS_STANDARD, precision="bf16")
    full = estimate_lora_memory(llama_31_8b, rank=16, targets=LORA_TARGETS_FULL, precision="bf16")

    assert minimal < standard < full


def test_estimate_lora_memory_mlp_targets_use_intermediate_size(llama_31_8b: ModelConfig) -> None:
    result = estimate_lora_memory(llama_31_8b, rank=16, targets=["down_proj"], precision="bf16")

    expected_params = llama_31_8b.num_layers * 16 * (
        llama_31_8b.intermediate_size + llama_31_8b.hidden_size
    )
    expected_mib = bytes_to_mib(round(expected_params * 2))
    assert result == pytest.approx(expected_mib, rel=1e-9)


def test_estimate_lora_memory_scales_linearly_with_rank(llama_31_8b: ModelConfig) -> None:
    r16 = estimate_lora_memory(llama_31_8b, rank=16, targets=LORA_TARGETS_STANDARD, precision="bf16")
    r32 = estimate_lora_memory(llama_31_8b, rank=32, targets=LORA_TARGETS_STANDARD, precision="bf16")

    assert r32 == pytest.approx(r16 * 2, rel=1e-9)


@pytest.mark.parametrize(
    ("precision", "bytes_per_param"),
    [("fp32", 4.0), ("fp16", 2.0), ("bf16", 2.0)],
)
def test_estimate_lora_memory_respects_precision(
    llama_31_8b: ModelConfig, precision: str, bytes_per_param: float
) -> None:
    result = estimate_lora_memory(llama_31_8b, rank=16, targets=["q_proj"], precision=precision)

    q_out = llama_31_8b.num_attention_heads * llama_31_8b.head_dim
    expected_params = llama_31_8b.num_layers * 16 * (llama_31_8b.hidden_size + q_out)
    expected_mib = bytes_to_mib(round(expected_params * bytes_per_param))
    assert result == pytest.approx(expected_mib, rel=1e-9)


@pytest.mark.parametrize("bad_rank", [0, -1, True, 12.5, "16"])
def test_estimate_lora_memory_rejects_invalid_rank(llama_31_8b: ModelConfig, bad_rank: object) -> None:
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        estimate_lora_memory(llama_31_8b, rank=bad_rank, targets=LORA_TARGETS_MINIMAL, precision="bf16")


def test_estimate_lora_memory_rejects_empty_targets(llama_31_8b: ModelConfig) -> None:
    with pytest.raises(ValueError, match="at least one LoRA target"):
        estimate_lora_memory(llama_31_8b, rank=16, targets=[], precision="bf16")


def test_estimate_lora_memory_rejects_unknown_target(llama_31_8b: ModelConfig) -> None:
    with pytest.raises(ValueError, match="Unsupported LoRA target"):
        estimate_lora_memory(llama_31_8b, rank=16, targets=["mlp_proj"], precision="bf16")


def test_estimate_lora_memory_rejects_duplicate_targets(llama_31_8b: ModelConfig) -> None:
    with pytest.raises(ValueError, match="Duplicate LoRA target"):
        estimate_lora_memory(llama_31_8b, rank=16, targets=["q_proj", "q_proj"], precision="bf16")


def test_estimate_lora_memory_rejects_unsupported_precision(llama_31_8b: ModelConfig) -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        estimate_lora_memory(llama_31_8b, rank=16, targets=["q_proj"], precision="fp4")