from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import pytest


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
) -> Callable[[dict[str, Any]], None]:
    def _install(config: dict[str, Any]) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        def _fake_hf_hub_download(*, repo_id: str, filename: str) -> str:
            return str(config_path)

        monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _fake_hf_hub_download)

    return _install
