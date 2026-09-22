from __future__ import annotations

from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result
from fitcheck import repl
from fitcheck.cli import main
from fitcheck.config_parser import (
    HubTransportError,
    HubUnavailableError,
    fetch_model_config,
)
from fitcheck.display import make_console
from huggingface_hub.errors import RepositoryNotFoundError

_MODEL_ID = "meta-llama/Llama-3.1-8B"
_EXIT_ERROR = 2

_REQUEST = httpx.Request("GET", f"https://huggingface.co/{_MODEL_ID}")

_HUB_SPEAKS_HTTPX = HubTransportError is httpx.HTTPError
_needs_httpx_transport = pytest.mark.skipif(
    not _HUB_SPEAKS_HTTPX,
    reason="huggingface_hub < 1.0 speaks requests, whose transport errors are already OSError",
)

_TRANSPORT_FAILURES = (
    httpx.ReadTimeout("timed out", request=_REQUEST),
    httpx.ConnectTimeout("connect timed out", request=_REQUEST),
    httpx.ConnectError("name resolution failed", request=_REQUEST),
    httpx.ProxyError("proxy refused", request=_REQUEST),
    httpx.RemoteProtocolError("peer closed the connection", request=_REQUEST),
)


def _install(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    def _raise(**kwargs: Any) -> str:
        raise error

    monkeypatch.setattr("fitcheck.config_parser.hf_hub_download", _raise)


def _repl_session() -> repl._Session:
    return repl._Session(
        console=make_console(no_color=True),
        glyphs=repl._ASCII_GLYPHS,
        ascii_only=True,
    )


def _crashed(result: Result) -> bool:
    """True when an exception reached the terminal instead of an error message."""
    return result.exception is not None and not isinstance(result.exception, SystemExit)




@pytest.mark.parametrize("failure", _TRANSPORT_FAILURES, ids=lambda e: type(e).__name__)
@_needs_httpx_transport
def test_transport_failures_become_hub_unavailable(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    """httpx errors are not OSError, so nothing downstream was catching them."""
    _install(monkeypatch, failure)

    with pytest.raises(HubUnavailableError):
        fetch_model_config(_MODEL_ID)


@_needs_httpx_transport
def test_the_message_names_the_model_and_offers_a_way_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, httpx.ReadTimeout("timed out", request=_REQUEST))

    with pytest.raises(HubUnavailableError) as excinfo:
        fetch_model_config(_MODEL_ID)

    message = str(excinfo.value)
    assert _MODEL_ID in message
    assert "ReadTimeout" in message
    assert "HF_HUB_OFFLINE=1" in message


def test_an_answer_from_the_hub_is_not_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """404 / 401 / gated carry the Hub's own message; a retry hint would mislead."""
    _install(
        monkeypatch,
        RepositoryNotFoundError(
            "404 Client Error. Repository Not Found", response=httpx.Response(404, request=_REQUEST)
        ),
    )

    with pytest.raises(RepositoryNotFoundError):
        fetch_model_config(_MODEL_ID)


def test_programmer_errors_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, TypeError("hf_hub_download() got an unexpected keyword"))

    with pytest.raises(TypeError):
        fetch_model_config(_MODEL_ID)




@pytest.mark.parametrize(
    "argv",
    [
        [_MODEL_ID, "--gpu", "4090"],
        ["infer", _MODEL_ID],
        ["advise", _MODEL_ID, "--seq-lens", "2048"],
    ],
    ids=["estimate", "infer", "advise"],
)
@_needs_httpx_transport
def test_cli_reports_a_timeout_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    _install(monkeypatch, httpx.ReadTimeout("timed out", request=_REQUEST))

    result = CliRunner().invoke(main, argv)

    assert not _crashed(result)
    assert result.exit_code == _EXIT_ERROR
    assert "Could not reach the Hugging Face Hub" in result.output
    assert "HF_HUB_OFFLINE=1" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("failure", _TRANSPORT_FAILURES, ids=lambda e: type(e).__name__)
@_needs_httpx_transport
def test_cli_handles_every_transport_failure_the_same_way(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    _install(monkeypatch, failure)

    result = CliRunner().invoke(main, [_MODEL_ID, "--gpu", "4090"])

    assert not _crashed(result)
    assert result.exit_code == _EXIT_ERROR
    assert "Could not reach the Hugging Face Hub" in result.output


def test_repository_not_found_keeps_its_own_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        RepositoryNotFoundError(
            "404 Client Error. Repository Not Found", response=httpx.Response(404, request=_REQUEST)
        ),
    )

    result = CliRunner().invoke(main, [_MODEL_ID, "--gpu", "4090"])

    assert not _crashed(result)
    assert result.exit_code == _EXIT_ERROR
    assert "Could not read config.json" in result.output
    assert "Could not reach the Hugging Face Hub" not in result.output


def test_malformed_input_still_fails_as_a_usage_error() -> None:
    result = CliRunner().invoke(main, [_MODEL_ID, "--quant", "gptq"])

    assert not _crashed(result)
    assert result.exit_code == _EXIT_ERROR
    assert "is not one of" in result.output




@_needs_httpx_transport
def test_repl_reports_a_timeout_as_an_ordinary_repl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_ReplError is printed and the session survives; anything else prints a bare type."""
    _install(monkeypatch, httpx.ReadTimeout("timed out", request=_REQUEST))
    session = _repl_session()

    with pytest.raises(repl._ReplError) as excinfo:
        repl._cmd_model(session, [_MODEL_ID])

    assert "Could not reach the Hugging Face Hub" in str(excinfo.value)
    assert "HF_HUB_OFFLINE=1" in str(excinfo.value)
    assert session.model is None


def test_repl_keeps_the_repository_not_found_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        RepositoryNotFoundError(
            "404 Client Error. Repository Not Found", response=httpx.Response(404, request=_REQUEST)
        ),
    )
    session = _repl_session()

    with pytest.raises(repl._ReplError, match="Could not read config"):
        repl._cmd_model(session, [_MODEL_ID])
