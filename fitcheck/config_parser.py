from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError

_TIES_WORD_EMBEDDINGS_BY_DEFAULT = frozenset(
    {"gemma", "gemma2", "gemma3", "gemma3_text"}
)


_MOE_KEYS = (
    "num_experts_per_tok",
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
)


_PARAM_COUNT_TOLERANCE = 0.02


class UnsupportedModelError(ValueError):
    """The config parsed fine, but its architecture is outside fitcheck's memory model.

    A subclass of `ValueError` so existing callers keep catching it, but a distinct
    type so the CLI and the REPL can print the message without the "could not read
    config.json" prefix: the file was read, it is the model that is refused.
    """


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

    @property
    def num_unquantized_params(self) -> int:
        embedding_params = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embedding_params *= 2
        norm_params = self.num_layers * 2 * self.hidden_size + self.hidden_size
        return embedding_params + norm_params


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


def _intermediate_size(raw: Mapping[str, Any], hidden_size: int) -> int:
    if raw.get("intermediate_size") is None:
        print(
            "Warning: 'intermediate_size' not in config.json - "
            "assuming 4 x hidden_size (can be 10-30% off the model's actual FFN size).",
            file=sys.stderr,
        )
        return 4 * hidden_size
    return _required_int(raw, "intermediate_size")


def _head_dim(raw: Mapping[str, Any], hidden_size: int, num_attention_heads: int) -> int:
    if raw.get("head_dim") is None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "'hidden_size' must be divisible by 'num_attention_heads' when "
                "config.json does not declare 'head_dim'"
            )
        return hidden_size // num_attention_heads
    return _required_int(raw, "head_dim")


def _tie_word_embeddings(raw: Mapping[str, Any]) -> bool:
    declared = raw.get("tie_word_embeddings")
    if declared is not None:
        return bool(declared)
    return str(raw.get("model_type", "")).strip().casefold() in _TIES_WORD_EMBEDDINGS_BY_DEFAULT


def _reject_unsupported(raw: Mapping[str, Any], model_id: str) -> None:
    declared_moe_keys = [key for key in _MOE_KEYS if raw.get(key) is not None]
    if declared_moe_keys:
        raise UnsupportedModelError(
            f"'{model_id}' is a Mixture-of-Experts model (config.json declares "
            f"{', '.join(declared_moe_keys)}). fitcheck's weight and activation formulas "
            "assume a dense FFN and would under-count total parameters by 80-90% - in the "
            "direction that reports a fit where the run would OOM. MoE is not supported; "
            "see docs/SPEC.md section 3.7."
        )

    if "text_config" in raw and "hidden_size" not in raw:
        raise UnsupportedModelError(
            f"'{model_id}' is a multimodal model that nests its language-model dimensions "
            "under 'text_config'. The fields are nested, not missing. fitcheck estimates "
            "text-only decoder models and does not model a vision tower, so any number it "
            "produced here would leave the vision tower out - an under-count, in the "
            "direction that reports a fit where the run would OOM. "
            "See docs/SPEC.md section 3.7."
        )


def _parse_fields(raw: Mapping[str, Any]) -> _ParsedFields:
    hidden_size = _required_int(raw, "hidden_size")
    num_layers = _required_int(raw, "num_hidden_layers")
    num_attention_heads = _required_int(raw, "num_attention_heads")
    num_kv_heads = _num_kv_heads(raw, num_attention_heads)
    intermediate_size = _intermediate_size(raw, hidden_size)
    vocab_size = _required_int(raw, "vocab_size")
    head_dim = _head_dim(raw, hidden_size, num_attention_heads)

    if num_kv_heads > num_attention_heads:
        raise ValueError("'num_key_value_heads' cannot exceed 'num_attention_heads'")

    return _ParsedFields(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=head_dim,
        tie_word_embeddings=_tie_word_embeddings(raw),
    )


def _attention_param_count(
    hidden_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    q_out = num_attention_heads * head_dim
    kv_out = num_kv_heads * head_dim
    return 2 * hidden_size * q_out + 2 * hidden_size * kv_out


def _count_params(fields: _ParsedFields) -> int:
    attention_params = _attention_param_count(
        fields.hidden_size,
        fields.num_attention_heads,
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
        + fields.hidden_size 
        + lm_head_params
    )


def _reported_param_count(model_id: str, token: str | None) -> int | None:
    try:
        info = HfApi(token=token).model_info(model_id, expand=["safetensors"])
    except Exception: 
        return None
    total = getattr(getattr(info, "safetensors", None), "total", None)
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        return None
    return total


def _reconcile_param_count(model_id: str, reported: int | None, derived: int) -> int:
    if reported is None:
        print(
            f"Warning: could not read the Hub's parameter count for '{model_id}' - "
            "using the count derived from config.json, which assumes a dense 3-matrix "
            "SwiGLU MLP and no biases (exact for Llama-shaped models, off by tens of "
            "percent for others).",
            file=sys.stderr,
        )
        return derived

    disagreement = (derived - reported) / reported
    if abs(disagreement) > _PARAM_COUNT_TOLERANCE:
        direction = (
            "over-counts, which over-states the memory needed"
            if disagreement > 0
            else "under-counts - the direction that reports a fit where the run would OOM"
        )
        raise UnsupportedModelError(
            f"'{model_id}' has {reported:,} parameters according to the Hub, but "
            f"fitcheck's config.json formula derives {derived:,} - a disagreement of "
            f"{disagreement:+.1%}, past the {_PARAM_COUNT_TOLERANCE:.0%} tolerance "
            f"({direction}). That formula assumes a dense decoder with a 3-matrix "
            "SwiGLU MLP and no biases, and the activation and LoRA formulas assume the "
            "same shape, so a gap this large means the whole memory model is off, not "
            "just the parameter count. fitcheck refuses instead of averaging the two. "
            "See docs/SPEC.md section 3.3."
        )
    return reported


def _hub_token() -> str | None:
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(variable, "").strip()
        if token:
            return token
    return None


def fetch_model_config(model_id: str) -> ModelConfig:
    if not model_id or not model_id.strip():
        raise ValueError("model_id must be a non-empty Hugging Face repository ID")

    normalized_model_id = model_id.strip()
    token = _hub_token()
    try:
        config_path = hf_hub_download(
            repo_id=normalized_model_id, filename="config.json", token=token
        )
    except GatedRepoError as error:
        if token is None:
            raise RuntimeError(
                f"'{normalized_model_id}' is gated on Hugging Face and no token was found.\n"
                "Accept its license on the model page, then either run `hf auth login` or "
                "export HF_TOKEN.\n"
                "On Kaggle or Colab, set HF_TOKEN from your notebook secrets before running."
            ) from error
        raise RuntimeError(
            f"'{normalized_model_id}' is gated on Hugging Face. A token was found, but it "
            "does not grant access to this repository.\n"
            "Accept the license on the model page with the same account that issued the "
            "token, and check the token has 'read' permission."
        ) from error
    with Path(config_path).open(encoding="utf-8") as config_file:
        raw: Mapping[str, Any] = json.load(config_file)

    _reject_unsupported(raw, normalized_model_id)
    fields = _parse_fields(raw)
    num_params = _reconcile_param_count(
        normalized_model_id,
        _reported_param_count(normalized_model_id, token),
        _count_params(fields),
    )

    return ModelConfig(
        name=normalized_model_id.rstrip("/").split("/")[-1],
        num_params=num_params,
        hidden_size=fields.hidden_size,
        num_layers=fields.num_layers,
        num_attention_heads=fields.num_attention_heads,
        num_kv_heads=fields.num_kv_heads,
        intermediate_size=fields.intermediate_size,
        vocab_size=fields.vocab_size,
        head_dim=fields.head_dim,
        tie_word_embeddings=fields.tie_word_embeddings,
    )
