from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import httpx
import pytest
from huggingface_hub.errors import GatedRepoError
from fitcheck.config_parser import fetch_model_config


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
    printed = capsys.readouterr().out
    assert "'intermediate_size' not in config.json" in printed
    assert "4 x hidden_size" in printed


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
