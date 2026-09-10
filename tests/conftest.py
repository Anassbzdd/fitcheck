from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import pytest


@pytest.fixture(autouse=True)
def offline_hub_param_count(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("network") is not None:
        return

    def _no_hub_count(model_id: str, token: str | None) -> int | None:
        return None

    monkeypatch.setattr("fitcheck.config_parser._reported_param_count", _no_hub_count)


@pytest.fixture
def llama_31_8b_config() -> dict[str, Any]:
    return {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def mha_config() -> dict[str, Any]:
    return {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 50257,
    }


@pytest.fixture
def mqa_config() -> dict[str, Any]:
    return {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 1,
        "intermediate_size": 5504,
        "vocab_size": 32000,
    }


@pytest.fixture
def gemma_2_9b_config() -> dict[str, Any]:
    """Verbatim from google/gemma-2-9b: declares head_dim, omits tie_word_embeddings.

    head_dim 256 != 3584 / 16 = 224, and the absent tie flag means *tied*, not untied.
    Both traps in one config, which is why it is the fixture for 2.6.
    """
    return {
        "model_type": "gemma2",
        "hidden_size": 3584,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "intermediate_size": 14336,
        "vocab_size": 256000,
    }


@pytest.fixture
def tied_embeddings_config() -> dict[str, Any]:
    return {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 2816,
        "vocab_size": 32000,
        "tie_word_embeddings": True,
    }


@pytest.fixture
def fake_config_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., None]:
    def _install(config: dict[str, Any], hub_param_count: int | None = None) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        def _fake_hf_hub_download(
            *, repo_id: str, filename: str, token: str | None = None
        ) -> str:
            return str(config_path)

        def _fake_reported_param_count(model_id: str, token: str | None) -> int | None:
            return hub_param_count

        monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _fake_hf_hub_download)
        monkeypatch.setattr(
            "fitcheck.config_parser._reported_param_count", _fake_reported_param_count
        )

    return _install
