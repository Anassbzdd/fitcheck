from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import pytest
from fitcheck.config_parser import HubUnavailableError

_MODEL_ID = "meta-llama/Llama-3.1-8B"


def _load_app() -> Any:
    pytest.importorskip("gradio", reason="install the web extra to test the Gradio app")
    return import_module("app")


def _training_callback(app_module: Any) -> tuple[Any, ...]:
    return app_module.training_callback(
        _MODEL_ID,
        "4090",
        1,
        512,
        "nf4",
        False,
        "bf16",
        16,
        "standard",
        False,
        "adamw",
        "fp32",
        True,
        False,
        1,
        None,
    )


def test_app_imports_and_creates_demo() -> None:
    gradio = pytest.importorskip("gradio", reason="install the web extra to test the Gradio app")
    app_module = import_module("app")
    assert isinstance(app_module.demo, gradio.Blocks)
    assert isinstance(app_module.create_demo(), gradio.Blocks)


def test_training_callback_runs_with_mocked_hub_access(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)
    app_module = _load_app()

    outputs = _training_callback(app_module)

    assert len(outputs) == 7
    assert "role=\"status\"" in outputs[0]
    assert "Predicted peak" in outputs[1]
    assert outputs[2]
    assert outputs[-1] == ""
    assert "Traceback" not in repr(outputs)


def test_training_callback_shows_hub_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = _load_app()

    def unavailable(_: str) -> None:
        raise HubUnavailableError("Could not reach the Hugging Face Hub. Try again.")

    monkeypatch.setattr("fitcheck.web.fetch_model_config", unavailable)

    outputs = _training_callback(app_module)

    assert len(outputs) == 7
    assert "Could not reach the Hugging Face Hub" in outputs[-1]
    assert outputs[2] == []
    assert outputs[5] == []
    assert "Traceback" not in repr(outputs)
