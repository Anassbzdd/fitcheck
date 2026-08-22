from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_params: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    tie_word_embeddings: bool


@dataclass(frozen=True)
class _ParsedFields:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    tie_word_embeddings: bool


def _required_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config.json field '{field}' must be a positive integer")
    return value


def _num_kv_heads(raw: Mapping[str, Any], num_attention_heads: int) -> int:
    if raw.get("num_key_value_heads") is None:
        return num_attention_heads
    return _required_int(raw, "num_key_value_heads")


def _parse_fields(raw: Mapping[str, Any]) -> _ParsedFields:
    hidden_size = _required_int(raw, "hidden_size")
    num_layers = _required_int(raw, "num_hidden_layers")
    num_attention_heads = _required_int(raw, "num_attention_heads")
    num_kv_heads = _num_kv_heads(raw, num_attention_heads)
    intermediate_size = _required_int(raw, "intermediate_size")
    vocab_size = _required_int(raw, "vocab_size")

    if hidden_size % num_attention_heads != 0:
        raise ValueError("'hidden_size' must be divisible by 'num_attention_heads'")
    if num_kv_heads > num_attention_heads:
        raise ValueError("'num_key_value_heads' cannot exceed 'num_attention_heads'")

    return _ParsedFields(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=hidden_size // num_attention_heads,
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )


def _attention_param_count(
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:    
    return 2 * hidden_size**2 + 2 * hidden_size * num_kv_heads * head_dim


def _count_params(fields: _ParsedFields) -> int:
    attention_params = _attention_param_count(
        fields.hidden_size,
        fields.num_kv_heads,
        fields.head_dim,
    )
    mlp_params = 3 * fields.hidden_size * fields.intermediate_size  
    norm_params = 2 * fields.hidden_size
    embedding_params = fields.vocab_size * fields.hidden_size
    lm_head_params = 0 if fields.tie_word_embeddings else embedding_params

    return (
        embedding_params
        + fields.num_layers * (attention_params + mlp_params + norm_params)
        + fields.hidden_size  # final normalization
        + lm_head_params
    )


def fetch_model_config(model_id: str) -> ModelConfig:
    if not model_id or not model_id.strip():
        raise ValueError("model_id must be a non-empty Hugging Face repository ID")

    normalized_model_id = model_id.strip()
    try:
        config_path = hf_hub_download(repo_id=normalized_model_id, filename="config.json")
    except GatedRepoError as error:
        raise RuntimeError(
            "This model is gated on Hugging Face.\n"
            "Accept its license on the model page, then run: hf auth login"
        ) from error
    with Path(config_path).open(encoding="utf-8") as config_file:
        raw: Mapping[str, Any] = json.load(config_file)

    fields = _parse_fields(raw)

    return ModelConfig(
        name=normalized_model_id.rstrip("/").split("/")[-1],
        num_params=_count_params(fields),
        hidden_size=fields.hidden_size,
        num_layers=fields.num_layers,
        num_attention_heads=fields.num_attention_heads,
        num_kv_heads=fields.num_kv_heads,
        intermediate_size=fields.intermediate_size,
        vocab_size=fields.vocab_size,
        head_dim=fields.head_dim,
        tie_word_embeddings=fields.tie_word_embeddings,
    )
