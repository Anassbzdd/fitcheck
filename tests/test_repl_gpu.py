from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import pytest
from fitcheck import repl
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.estimator import TrainingConfig
from fitcheck.gpu_db import GpuSpec, get_gpu
from rich.console import Console


@pytest.fixture
def llama_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(llama_31_8b_config)
    return fetch_model_config("meta-llama/Llama-3.1-8B")


@pytest.fixture
def qlora_training() -> TrainingConfig:
    return TrainingConfig(
        precision="bf16",
        quantization="nf4",
        optimizer="adamw",
        optimizer_dtype="fp32",
        batch_size=1,
        seq_len=2048,
        lora_rank=64,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        grad_checkpoint=True,
        flash_attn=True,
    )


class _Repl:
    """A REPL session wired to a string buffer, driven one line at a time."""

    def __init__(self, model: ModelConfig, training: TrainingConfig, gpu: GpuSpec) -> None:
        self.buffer = io.StringIO()
        console = Console(file=self.buffer, width=200, no_color=True, legacy_windows=False)
        self.session = repl._Session(
            console=console,
            glyphs=repl._ASCII_GLYPHS,
            ascii_only=True,
            model=model,
            model_id="meta-llama/Llama-3.1-8B",
            gpu=gpu,
            training=training,
        )

    def run(self, line: str) -> str:
        before = self.buffer.tell()
        repl._dispatch(self.session, line)
        self.buffer.seek(before)
        return " ".join(self.buffer.read().split())


def test_gpu_flag_uses_the_database_key(llama_model: ModelConfig) -> None:
    assert repl._gpu_flag(get_gpu("4090")) == "--gpu 4090"


def test_gpu_flag_reproduces_a_custom_card(llama_model: ModelConfig) -> None:
    assert repl._gpu_flag(get_gpu("mycard", 24_000)) == "--gpu mycard --vram-mib 24000"


def test_optimize_command_names_the_session_gpu(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    session = _Repl(llama_model, qlora_training, get_gpu("4090"))
    assert "memory --gpu 4090 --batch-size" in session.run("optimize")


def test_optimize_command_keeps_a_one_shot_gpu_override(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    session = _Repl(llama_model, qlora_training, get_gpu("a100-80"))
    session.run("memory --gpu 4090")

    assert "memory --gpu 4090 --batch-size" in session.run("optimize")
    assert session.session.gpu == get_gpu("a100-80")


def test_rescue_command_keeps_the_gpu_override(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    session = _Repl(llama_model, qlora_training, get_gpu("a100-80"))
    session.run("memory --gpu t4")

    output = session.run("optimize")
    assert "Nothing fits at batch_size 1" in output
    assert "memory --gpu t4 --batch-size 1" in output


def test_compare_prices_every_card_separately(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    session = _Repl(llama_model, qlora_training, get_gpu("t4"))
    output = session.run("compare 4090")

    assert "Peak (MiB)" in output
    assert "Peak is identical on every card" not in output
    assert "Peak moves with the card" in output


def test_compare_still_states_equality_when_the_peaks_match(
    llama_model: ModelConfig, qlora_training: TrainingConfig
) -> None:
    session = _Repl(llama_model, qlora_training, get_gpu("4090"))
    output = session.run("compare a100-80")

    assert "Peak is identical on every card" in output
