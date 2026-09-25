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


def _training_callback(
    app_module: Any,
    gpu_name: str = "4090",
    custom_vram_mib: object | None = None,
) -> tuple[Any, ...]:
    return app_module.training_callback(
        _MODEL_ID,
        gpu_name,
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
        custom_vram_mib,
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


def test_preset_gpu_ignores_custom_vram_value(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)
    app_module = _load_app()

    for stale_value in (0, "0", "12345"):
        outputs = _training_callback(app_module, "4090", stale_value)
        assert outputs[-1] == ""
        assert "23,500 MiB" in outputs[1]


def test_custom_gpu_requires_and_uses_total_vram(
    fake_config_download: Callable[..., None],
    llama_31_8b_config: dict[str, Any],
) -> None:
    fake_config_download(llama_31_8b_config)
    app_module = _load_app()

    for empty_value in (None, "", "  "):
        missing = _training_callback(app_module, "custom", empty_value)
        assert missing[-1] == "Enter total VRAM in MiB for Custom GPU."

    for invalid_value in ("0", "12.5", "abc"):
        invalid = _training_callback(app_module, "custom", invalid_value)
        assert "positive whole number" in invalid[-1]

    valid = _training_callback(app_module, "custom", "12000")
    assert valid[-1] == ""
    assert "11,400 MiB" in valid[1]


def test_vram_field_is_hidden_until_custom_gpu_is_selected() -> None:
    app_module = _load_app()
    components = app_module.demo.config["components"]

    gpu_fields = [
        component for component in components
        if component["props"].get("label") == "GPU preset"
    ]
    assert len(gpu_fields) == 3
    assert all(
        ("Custom GPU", "custom") in component["props"]["choices"]
        for component in gpu_fields
    )

    vram_fields = [
        component for component in components
        if component["props"].get("label") == "Custom GPU total VRAM (MiB)"
    ]
    assert len(vram_fields) == 3
    assert all(component["type"] == "textbox" for component in vram_fields)
    assert all(component["props"]["visible"] is False for component in vram_fields)
    assert all(component["props"]["value"] == "" for component in vram_fields)
    assert all(component["props"].get("placeholder") for component in vram_fields)

    assert app_module._custom_vram_visibility("4090")["visible"] is False
    assert app_module._custom_vram_visibility("custom")["visible"] is True
