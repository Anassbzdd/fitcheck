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


def _required_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config.json field '{field}' must be a positive integer")
    return value


def _attention_param_count(
    hidden_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    if num_kv_heads == num_attention_heads:
        return 4 * hidden_size**2
    if num_kv_heads == 1:
        # MQA: Q/O are h × h; K/V are h × head_dim.
        return 2 * hidden_size**2 + 2 * hidden_size * head_dim

    # GQA: Q/O are h × h; K/V are h × (n_kv × head_dim).
    return 2 * hidden_size**2 + 2 * hidden_size * num_kv_heads * head_dim


def _count_params(raw: Mapping[str, Any]) -> int:
    hidden_size = _required_int(raw, "hidden_size")
    num_layers = _required_int(raw, "num_hidden_layers")
    num_attention_heads = _required_int(raw, "num_attention_heads")
    num_kv_heads = _required_int(raw, "num_key_value_heads")
    if isinstance(num_kv_heads, bool) or not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        raise ValueError("config.json field 'num_key_value_heads' must be a positive integer")
    intermediate_size = _required_int(raw, "intermediate_size")
    vocab_size = _required_int(raw, "vocab_size")

    if hidden_size % num_attention_heads != 0:
        raise ValueError("'hidden_size' must be divisible by 'num_attention_heads'")

    head_dim = hidden_size // num_attention_heads
    if num_kv_heads > num_attention_heads:
        raise ValueError("'num_key_value_heads' cannot exceed 'num_attention_heads'")

    attention_params = _attention_param_count(
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        head_dim,
    )
    mlp_params = 3 * hidden_size * intermediate_size  # gate, up, and down projections
    norm_params = 2 * hidden_size  # input and post-attention norm per layer
    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if raw.get("tie_word_embeddings", False) else embedding_params

    return (
        embedding_params
        + num_layers * (attention_params + mlp_params + norm_params)
        + hidden_size  # final normalization
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

    hidden_size = _required_int(raw, "hidden_size")
    num_attention_heads = _required_int(raw, "num_attention_heads")
    num_kv_heads = _required_int(raw, "num_key_value_heads")
    if isinstance(num_kv_heads, bool) or not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        raise ValueError("config.json field 'num_key_value_heads' must be a positive integer")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("'hidden_size' must be divisible by 'num_attention_heads'")

    return ModelConfig(
        name=normalized_model_id.rstrip("/").split("/")[-1],
        num_params=_count_params(raw),
        hidden_size=hidden_size,
        num_layers=_required_int(raw, "num_hidden_layers"),
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=_required_int(raw, "intermediate_size"),
        vocab_size=_required_int(raw, "vocab_size"),
        head_dim=hidden_size // num_attention_heads,
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
    )
