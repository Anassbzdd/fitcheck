"""Integration tests for Hugging Face model-config parsing."""

from __future__ import annotations

import pytest

from fitcheck.config_parser import fetch_model_config


def test_fetch_model_config_parses_llama_31_8b() -> None:
    """Fetch Llama's real config.json and derive its GQA-aware parameter count."""
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
    assert config.num_params == pytest.approx(8.03e9, rel=0.005)
