from __future__ import annotations
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


def test_fetch_model_config_gated_repo_raises_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_gated(*, repo_id: str, filename: str) -> str:
        fake_response = httpx.Response(
            403, request=httpx.Request("GET", f"https://huggingface.co/{repo_id}")
        )
        raise GatedRepoError("gated", response=fake_response)

    monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _raise_gated)

    with pytest.raises(RuntimeError, match="gated"):
        fetch_model_config("meta-llama/some-gated-model")
