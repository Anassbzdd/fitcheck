from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import httpx
import pytest
from huggingface_hub.errors import GatedRepoError
from fitcheck import config_parser
from fitcheck.config_parser import (
    UnsupportedModelError,
    _reported_param_count,
    fetch_model_config,
)


@pytest.mark.network
def test_fetch_model_config_parses_llama_31_8b() -> None:
    config = fetch_model_config("meta-llama/Llama-3.1-8B")

    assert config.name == "Llama-3.1-8B"
    assert config.hidden_size == 4096
    assert config.num_layers == 32
    assert config.num_attention_heads == 32
    assert config.num_kv_heads == 8
    assert config.intermediate_size == 14336
    assert config.vocab_size == 128256
    assert config.head_dim == 128
    assert config.tie_word_embeddings is False
    assert config.num_params == pytest.approx(8_030_261_248, rel=0.00001)


def test_fetch_model_config_parses_llama_31_8b_offline(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)

    config = fetch_model_config("meta-llama/Llama-3.1-8B")

    assert config.name == "Llama-3.1-8B"
    assert config.num_kv_heads == 8
    assert config.head_dim == 128
    assert config.num_params == pytest.approx(8_030_261_248, rel=0.00001)


def test_fetch_model_config_defaults_missing_kv_heads_to_mha(
    fake_config_download: Callable[[dict[str, Any]], None],
    mha_config: dict[str, Any],
) -> None:
    fake_config_download(mha_config)

    config = fetch_model_config("openai-community/gpt2")

    assert config.num_kv_heads == config.num_attention_heads == 12
    assert config.head_dim == 64
    expected_attention_params = 4 * config.hidden_size**2
    expected_mlp_params = 3 * config.hidden_size * config.intermediate_size
    expected_norm_params = 2 * config.hidden_size
    expected_embedding_params = config.vocab_size * config.hidden_size
    expected = (
        expected_embedding_params
        + config.num_layers * (expected_attention_params + expected_mlp_params + expected_norm_params)
        + config.hidden_size
        + expected_embedding_params  
    )
    assert config.num_params == expected


def test_fetch_model_config_handles_mqa(
    fake_config_download: Callable[[dict[str, Any]], None],
    mqa_config: dict[str, Any],
) -> None:
    fake_config_download(mqa_config)

    config = fetch_model_config("tiiuae/falcon-7b")

    assert config.num_kv_heads == 1
    assert config.num_attention_heads == 16
    assert config.head_dim == 128


def test_fetch_model_config_handles_tied_embeddings(
    fake_config_download: Callable[[dict[str, Any]], None],
    tied_embeddings_config: dict[str, Any],
) -> None:
    """tie_word_embeddings=True must count the embedding table once, not twice."""
    fake_config_download(tied_embeddings_config)

    tied = fetch_model_config("fake-org/tied-model")
    assert tied.tie_word_embeddings is True

    untied_config = dict(tied_embeddings_config, tie_word_embeddings=False)
    fake_config_download(untied_config)
    untied = fetch_model_config("fake-org/untied-model")
    assert untied.tie_word_embeddings is False

    embedding_params = tied_embeddings_config["vocab_size"] * tied_embeddings_config["hidden_size"]
    assert untied.num_params - tied.num_params == embedding_params


_GEMMA_2_9B_PARAMS = 9_241_404_928


def test_declared_head_dim_wins_over_hidden_size_over_heads(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    """Gemma-2-9B declares head_dim 256; hidden_size // heads would give 224."""
    fake_config_download(gemma_2_9b_config)

    config = fetch_model_config("google/gemma-2-9b")

    assert config.head_dim == 256
    assert config.head_dim != config.hidden_size // config.num_attention_heads


def test_head_dim_falls_back_to_hidden_size_over_heads_when_absent(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    without_head_dim = {
        key: value for key, value in gemma_2_9b_config.items() if key != "head_dim"
    }
    fake_config_download(without_head_dim)

    config = fetch_model_config("fake-org/no-head-dim")

    assert config.head_dim == 3584 // 16 == 224


def test_declared_head_dim_lifts_the_divisibility_requirement(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    """Divisibility constrains deriving head_dim, not the model itself."""
    indivisible = dict(gemma_2_9b_config, hidden_size=3585)
    fake_config_download(indivisible)

    config = fetch_model_config("fake-org/indivisible")

    assert config.hidden_size % config.num_attention_heads != 0
    assert config.head_dim == 256


def test_absent_tie_word_embeddings_means_tied_for_gemma(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    assert "tie_word_embeddings" not in gemma_2_9b_config
    fake_config_download(gemma_2_9b_config)

    config = fetch_model_config("google/gemma-2-9b")

    assert config.tie_word_embeddings is True


def test_absent_tie_word_embeddings_defaults_untied_for_unknown_architecture(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    """Unrecognised model_type keeps the conservative (over-counting) default."""
    unknown = dict(gemma_2_9b_config, model_type="not-a-real-architecture")
    fake_config_download(unknown)

    config = fetch_model_config("fake-org/unknown-arch")

    assert config.tie_word_embeddings is False


def test_an_explicit_tie_word_embeddings_beats_the_architecture_default(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    fake_config_download(dict(gemma_2_9b_config, tie_word_embeddings=False))

    config = fetch_model_config("fake-org/gemma-untied")

    assert config.tie_word_embeddings is False


def test_gemma_2_9b_counts_924b_not_993b(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    """Both traps compound: head_dim 224 + untied would over-count by 7.4%."""
    fake_config_download(gemma_2_9b_config)

    config = fetch_model_config("google/gemma-2-9b")

    assert config.num_params == _GEMMA_2_9B_PARAMS
    assert config.num_params / 1e9 == pytest.approx(9.24, abs=0.005)
    assert config.num_params < 9_500_000_000


def test_attention_params_use_head_dim_not_hidden_size_squared(
    fake_config_download: Callable[[dict[str, Any]], None],
    gemma_2_9b_config: dict[str, Any],
) -> None:
    fake_config_download(gemma_2_9b_config)
    config = fetch_model_config("google/gemma-2-9b")

    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_kv_heads * config.head_dim
    expected_attention = 2 * config.hidden_size * q_out + 2 * config.hidden_size * kv_out

    embedding = config.vocab_size * config.hidden_size
    mlp = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    expected = (
        embedding
        + config.num_layers * (expected_attention + mlp + norms)
        + config.hidden_size
    )

    assert q_out == 4096 != config.hidden_size
    assert config.num_params == expected


def test_generalized_attention_formula_leaves_llama_untouched(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    """n_h * d_k == h for Llama, so 2h*n_h*d_k + 2h*n_kv*d_k == 2h^2 + 2h*n_kv*d_k."""
    fake_config_download(llama_31_8b_config)

    config = fetch_model_config("meta-llama/Llama-3.1-8B")

    assert config.num_attention_heads * config.head_dim == config.hidden_size
    assert config.num_params == 8_030_261_248


@pytest.mark.parametrize("missing_field", ["hidden_size", "vocab_size", "num_hidden_layers"])
def test_fetch_model_config_missing_required_field_raises(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
    missing_field: str,
) -> None:
    broken_config = dict(llama_31_8b_config)
    del broken_config[missing_field]
    fake_config_download(broken_config)

    with pytest.raises(ValueError, match=missing_field):
        fetch_model_config("fake-org/broken-model")


def test_fetch_model_config_missing_intermediate_size_falls_back_to_4h(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_without_ffn_size = dict(llama_31_8b_config)
    del config_without_ffn_size["intermediate_size"]
    fake_config_download(config_without_ffn_size)

    config = fetch_model_config("fake-org/tied-model")

    assert config.intermediate_size == 4 * config.hidden_size
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "'intermediate_size' not in config.json" in captured.err
    assert "4 x hidden_size" in captured.err


def test_fetch_model_config_hidden_size_not_divisible_by_heads_raises(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    bad_config = dict(llama_31_8b_config, num_attention_heads=33)
    fake_config_download(bad_config)

    with pytest.raises(ValueError, match="divisible"):
        fetch_model_config("fake-org/bad-heads-model")


def test_fetch_model_config_kv_heads_exceeding_attention_heads_raises(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    bad_config = dict(llama_31_8b_config, num_key_value_heads=64)
    fake_config_download(bad_config)

    with pytest.raises(ValueError, match="cannot exceed"):
        fetch_model_config("fake-org/bad-kv-heads-model")


@pytest.mark.parametrize("model_id", ["", "   "])
def test_fetch_model_config_empty_model_id_raises(model_id: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fetch_model_config(model_id)


def _install_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_gated(*, repo_id: str, filename: str, token: str | None = None) -> str:
        fake_response = httpx.Response(
            403, request=httpx.Request("GET", f"https://huggingface.co/{repo_id}")
        )
        raise GatedRepoError("gated", response=fake_response)

    monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _raise_gated)


def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_fetch_model_config_gated_repo_raises_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_token_env(monkeypatch)
    _install_gated(monkeypatch)

    with pytest.raises(RuntimeError, match="gated"):
        fetch_model_config("meta-llama/some-gated-model")


def test_gated_error_without_a_token_points_at_login_and_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_token_env(monkeypatch)
    _install_gated(monkeypatch)

    with pytest.raises(RuntimeError, match="no token was found") as excinfo:
        fetch_model_config("meta-llama/some-gated-model")

    message = str(excinfo.value)
    assert "hf auth login" in message
    assert "HF_TOKEN" in message


def test_gated_error_with_a_token_blames_access_not_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token that is present but unapproved is a different problem, and a
    'run hf auth login' message sends the user in the wrong direction."""
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    _install_gated(monkeypatch)

    with pytest.raises(RuntimeError, match="does not grant access") as excinfo:
        fetch_model_config("meta-llama/some-gated-model")

    assert "hf auth login" not in str(excinfo.value)


def test_gated_error_never_echoes_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_token_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    _install_gated(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_model_config("meta-llama/some-gated-model")

    assert "hf_secret_value" not in str(excinfo.value)
    assert "hf_secret_value" not in repr(excinfo.value)


@pytest.mark.parametrize(
    "variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
)
def test_environment_token_is_forwarded_to_the_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str
) -> None:
    """Kaggle and Colab export a token instead of caching a CLI login."""
    _clear_token_env(monkeypatch)
    monkeypatch.setenv(variable, "  hf_from_env  ")
    seen: dict[str, Any] = {}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_MINIMAL_CONFIG), encoding="utf-8")

    def _capture(*, repo_id: str, filename: str, token: str | None = None) -> str:
        seen["token"] = token
        return str(config_path)

    monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _capture)
    fetch_model_config("some/model")

    assert seen["token"] == "hf_from_env"


def test_no_environment_token_forwards_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """None lets huggingface_hub fall back to a cached `hf auth login`."""
    _clear_token_env(monkeypatch)
    seen: dict[str, Any] = {}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_MINIMAL_CONFIG), encoding="utf-8")

    def _capture(*, repo_id: str, filename: str, token: str | None = None) -> str:
        seen["token"] = token
        return str(config_path)

    monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _capture)
    fetch_model_config("some/model")

    assert seen["token"] is None


_MINIMAL_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 100,
    "tie_word_embeddings": False,
}


_MIXTRAL_8X7B_CONFIG = {
    "model_type": "mixtral",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}

_QWEN3_30B_A3B_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "vocab_size": 151936,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}

_GPT_OSS_20B_CONFIG = {
    "model_type": "gpt_oss",
    "hidden_size": 2880,
    "num_hidden_layers": 24,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "intermediate_size": 2880,
    "vocab_size": 201088,
    "num_local_experts": 32,
    "num_experts_per_tok": 4,
}

_DEEPSEEK_V2_LITE_CONFIG = {
    "model_type": "deepseek_v2",
    "hidden_size": 2048,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "intermediate_size": 10944,
    "moe_intermediate_size": 1408,
    "vocab_size": 102400,
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
}

_SMOLVLM_CONFIG = {
    "model_type": "idefics3",
    "text_config": {
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 8192,
        "vocab_size": 49280,
    },
    "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
}


@pytest.mark.parametrize(
    ("model_id", "config", "expected"),
    [
        ("mistralai/Mixtral-8x7B-v0.1", _MIXTRAL_8X7B_CONFIG, "Mixture-of-Experts"),
        ("Qwen/Qwen3-30B-A3B", _QWEN3_30B_A3B_CONFIG, "Mixture-of-Experts"),
        ("openai/gpt-oss-20b", _GPT_OSS_20B_CONFIG, "Mixture-of-Experts"),
        ("deepseek-ai/DeepSeek-V2-Lite", _DEEPSEEK_V2_LITE_CONFIG, "Mixture-of-Experts"),
        ("HuggingFaceTB/SmolVLM-Instruct", _SMOLVLM_CONFIG, "nests"),
    ],
)
def test_unsupported_architectures_are_refused(
    fake_config_download: Callable[[dict[str, Any]], None],
    model_id: str,
    config: dict[str, Any],
    expected: str,
) -> None:
    fake_config_download(config)

    with pytest.raises(UnsupportedModelError, match=expected) as excinfo:
        fetch_model_config(model_id)

    message = str(excinfo.value)
    assert model_id in message
    assert "OOM" in message


@pytest.mark.parametrize(
    ("model_id", "config", "declared_key"),
    [
        ("mistralai/Mixtral-8x7B-v0.1", _MIXTRAL_8X7B_CONFIG, "num_local_experts"),
        ("Qwen/Qwen3-30B-A3B", _QWEN3_30B_A3B_CONFIG, "num_experts"),
        ("openai/gpt-oss-20b", _GPT_OSS_20B_CONFIG, "num_local_experts"),
        ("deepseek-ai/DeepSeek-V2-Lite", _DEEPSEEK_V2_LITE_CONFIG, "n_routed_experts"),
    ],
)
def test_moe_refusal_names_the_key_it_saw(
    fake_config_download: Callable[[dict[str, Any]], None],
    model_id: str,
    config: dict[str, Any],
    declared_key: str,
) -> None:
    fake_config_download(config)

    with pytest.raises(UnsupportedModelError) as excinfo:
        fetch_model_config(model_id)

    assert declared_key in str(excinfo.value)
    assert "num_experts_per_tok" in str(excinfo.value)


def test_nested_multimodal_refusal_says_nested_not_malformed(
    fake_config_download: Callable[[dict[str, Any]], None],
) -> None:
    fake_config_download(_SMOLVLM_CONFIG)

    with pytest.raises(UnsupportedModelError) as excinfo:
        fetch_model_config("HuggingFaceTB/SmolVLM-Instruct")

    message = str(excinfo.value)
    assert "text_config" in message
    assert "nested, not missing" in message
    assert "must be a positive integer" not in message


def test_unsupported_model_error_is_a_value_error() -> None:
    assert issubclass(UnsupportedModelError, ValueError)


@pytest.mark.parametrize(
    ("model_id", "config"),
    [
        ("meta-llama/Llama-3.1-8B", "llama_31_8b_config"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mqa_config"),
        ("openai-community/gpt2", "mha_config"),
        ("google/gemma-2-9b", "gemma_2_9b_config"),
        ("HuggingFaceTB/SmolLM2-1.7B", "tied_embeddings_config"),
    ],
)
def test_supported_dense_models_are_not_refused(
    request: pytest.FixtureRequest,
    fake_config_download: Callable[[dict[str, Any]], None],
    model_id: str,
    config: str,
) -> None:
    fake_config_download(request.getfixturevalue(config))

    assert fetch_model_config(model_id).num_params > 0


def test_a_vision_tower_alongside_top_level_dims_passes_the_parse_gate(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(
        dict(llama_31_8b_config, vision_config={"hidden_size": 1280, "depth": 32})
    )
    assert fetch_model_config("Qwen/Qwen2.5-VL-7B-Instruct").num_params > 0



_LLAMA_31_8B_PARAMS = 8_030_261_248

_PHI_2_CONFIG = {
    "model_type": "phi",
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 10240,
    "vocab_size": 51200,
    "tie_word_embeddings": False,
}
_PHI_2_DERIVED = 3_617_753_600
_PHI_2_HUB = 2_779_683_840


def test_hub_count_is_preferred_over_the_derived_one(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    hub_count = _LLAMA_31_8B_PARAMS + 57_344 

    fake_config_download(llama_31_8b_config, hub_param_count=hub_count)

    assert fetch_model_config("meta-llama/Llama-3.1-8B").num_params == hub_count


def test_hub_count_agreeing_exactly_leaves_the_golden_number_alone(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config, hub_param_count=_LLAMA_31_8B_PARAMS)

    assert fetch_model_config("meta-llama/Llama-3.1-8B").num_params == _LLAMA_31_8B_PARAMS


def test_offline_fallback_keeps_the_derived_count(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config, hub_param_count=None)

    assert fetch_model_config("meta-llama/Llama-3.1-8B").num_params == _LLAMA_31_8B_PARAMS


def test_offline_fallback_warns_that_the_count_is_derived(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config_download(llama_31_8b_config, hub_param_count=None)

    fetch_model_config("meta-llama/Llama-3.1-8B")

    stderr = capsys.readouterr().err
    assert "derived from config.json" in stderr
    assert "SwiGLU" in stderr


def test_phi_2_style_over_count_is_refused(
    fake_config_download: Callable[..., None],
) -> None:
    fake_config_download(_PHI_2_CONFIG, hub_param_count=_PHI_2_HUB)

    with pytest.raises(UnsupportedModelError) as excinfo:
        fetch_model_config("microsoft/phi-2")

    message = str(excinfo.value)
    assert "microsoft/phi-2" in message
    assert f"{_PHI_2_HUB:,}" in message
    assert f"{_PHI_2_DERIVED:,}" in message
    assert "+30.1%" in message
    assert "over-counts" in message


_QWEN2_5_VL_7B_CONFIG = {
    "model_type": "qwen2_5_vl",
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
    "vision_config": {"hidden_size": 1280, "depth": 32, "intermediate_size": 3420},
}
_QWEN2_5_VL_7B_DERIVED = 7_615_487_488
_QWEN2_5_VL_7B_HUB = 8_292_166_656


def test_flat_multimodal_is_refused_by_the_hub_cross_check(
    fake_config_download: Callable[..., None],
) -> None:
    fake_config_download(
        _QWEN2_5_VL_7B_CONFIG, hub_param_count=_QWEN2_5_VL_7B_HUB
    )

    with pytest.raises(UnsupportedModelError) as excinfo:
        fetch_model_config("Qwen/Qwen2.5-VL-7B-Instruct")

    message = str(excinfo.value)
    assert f"{_QWEN2_5_VL_7B_DERIVED:,}" in message
    assert "-8.2%" in message
    assert "OOM" in message


def test_disagreement_refusal_does_not_average_the_two(
    fake_config_download: Callable[..., None],
) -> None:
    fake_config_download(_PHI_2_CONFIG, hub_param_count=_PHI_2_HUB)

    with pytest.raises(UnsupportedModelError, match="refuses instead of averaging"):
        fetch_model_config("microsoft/phi-2")


@pytest.mark.parametrize(
    ("hub_count", "refused"),
    [
        (int(_LLAMA_31_8B_PARAMS / 1.02) + 1, False),  # just inside the 2% edge
        (int(_LLAMA_31_8B_PARAMS / 1.021), True),      # just past it
        (int(_LLAMA_31_8B_PARAMS / 0.98) - 1, False),
        (int(_LLAMA_31_8B_PARAMS / 0.979), True),
    ],
)
def test_two_percent_is_the_boundary(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
    hub_count: int,
    refused: bool,
) -> None:
    fake_config_download(llama_31_8b_config, hub_param_count=hub_count)

    if refused:
        with pytest.raises(UnsupportedModelError):
            fetch_model_config("meta-llama/Llama-3.1-8B")
    else:
        assert fetch_model_config("meta-llama/Llama-3.1-8B").num_params == hub_count


def test_moe_is_refused_before_the_hub_is_asked(
    fake_config_download: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parse gate runs first, so a refused model costs no metadata call."""
    calls: list[str] = []

    def _record(model_id: str, token: str | None) -> int | None:
        calls.append(model_id)
        return None

    fake_config_download(_MIXTRAL_8X7B_CONFIG)
    monkeypatch.setattr("fitcheck.config_parser._reported_param_count", _record)

    with pytest.raises(UnsupportedModelError):
        fetch_model_config("mistralai/Mixtral-8x7B-v0.1")

    assert calls == []


class _FakeSafetensors:
    def __init__(self, total: Any) -> None:
        self.total = total


class _FakeModelInfo:
    def __init__(self, safetensors: Any) -> None:
        self.safetensors = safetensors


def _install_fake_api(monkeypatch: pytest.MonkeyPatch, result: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class _FakeApi:
        def __init__(self, token: str | None = None) -> None:
            seen["token"] = token

        def model_info(self, model_id: str, expand: list[str] | None = None) -> Any:
            seen["model_id"] = model_id
            seen["expand"] = expand
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr("fitcheck.config_parser.HfApi", _FakeApi)
    return seen


def test_reported_param_count_reads_safetensors_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_api(
        monkeypatch, _FakeModelInfo(_FakeSafetensors(_LLAMA_31_8B_PARAMS))
    )

    assert (
        _reported_param_count("meta-llama/Llama-3.1-8B", "hf_tok") == _LLAMA_31_8B_PARAMS
    )
    assert seen["token"] == "hf_tok"
    assert seen["expand"] == ["safetensors"]


@pytest.mark.parametrize(
    "result",
    [
        OSError("no network"),
        RuntimeError("hub is down"),
        _FakeModelInfo(None),
        _FakeModelInfo(_FakeSafetensors(None)),
        _FakeModelInfo(_FakeSafetensors(0)),
        _FakeModelInfo(_FakeSafetensors(True)),
    ],
)
def test_reported_param_count_returns_none_when_the_hub_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, result: Any
) -> None:
    _install_fake_api(monkeypatch, result)

    assert _reported_param_count("some/model", None) is None


def test_the_hub_cross_check_downloads_no_weights(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole premise: a metadata call plus config.json, and no other file."""
    fake_config_download(llama_31_8b_config)
    _install_fake_api(monkeypatch, _FakeModelInfo(_FakeSafetensors(_LLAMA_31_8B_PARAMS)))
    monkeypatch.setattr(
        "fitcheck.config_parser._reported_param_count", _reported_param_count
    )
    downloaded: list[str] = []
    stubbed = config_parser.hf_hub_download

    def _record(*, repo_id: str, filename: str, token: str | None = None) -> str:
        downloaded.append(filename)
        return stubbed(repo_id=repo_id, filename=filename, token=token)

    monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _record)

    assert fetch_model_config("meta-llama/Llama-3.1-8B").num_params == _LLAMA_31_8B_PARAMS
    assert downloaded == ["config.json"]


@pytest.mark.network
@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("meta-llama/Llama-3.1-8B", 8_030_261_248),
        ("HuggingFaceTB/SmolLM2-1.7B", 1_711_376_384),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1_100_048_384),
        ("Qwen/Qwen2.5-1.5B", 1_543_714_304),
    ],
)
def test_num_params_matches_the_hub_for_dense_models(
    model_id: str, expected: int
) -> None:
    assert fetch_model_config(model_id).num_params == expected


@pytest.mark.network
@pytest.mark.parametrize(
    "model_id", ["microsoft/phi-2", "Qwen/Qwen2.5-VL-7B-Instruct"]
)
def test_architectures_outside_the_formula_are_refused_against_the_real_hub(
    model_id: str,
) -> None:
    with pytest.raises(UnsupportedModelError, match="tolerance"):
        fetch_model_config(model_id)
