from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from click import UsageError
from click.testing import CliRunner
from fitcheck import repl
from fitcheck.cli import main
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.display import make_console
from fitcheck.estimator import (
    ServingConfig,
    TrainingConfig,
    estimate,
    estimate_inference,
)
from fitcheck.gpu_db import get_gpu
from fitcheck.memory.inference import estimate_inference_memory
from fitcheck.validation import (
    COMPUTE_PRECISIONS,
    QUANTIZATIONS,
    double_quant_conflict,
    validate_double_quant,
    validate_flag,
    validate_precision,
    validate_quantization,
)

_STORAGE_DTYPES = ("nf4", "int8", "int4", "fp8")
_EXIT_USAGE_ERROR = 2


@pytest.fixture
def llama_model(
    fake_config_download: Callable[[dict[str, Any]], None],
    llama_31_8b_config: dict[str, Any],
) -> ModelConfig:
    fake_config_download(llama_31_8b_config)
    return fetch_model_config("meta-llama/Llama-3.1-8B")




@pytest.mark.parametrize("precision", COMPUTE_PRECISIONS)
def test_validate_precision_accepts_every_compute_dtype(precision: str) -> None:
    assert validate_precision(precision) == precision


@pytest.mark.parametrize("storage_dtype", _STORAGE_DTYPES)
def test_validate_precision_rejects_storage_dtypes(storage_dtype: str) -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        validate_precision(storage_dtype)


@pytest.mark.parametrize("precision", ["BF16", "  bf16  ", "Bf16"])
def test_validate_precision_normalizes(precision: str) -> None:
    assert validate_precision(precision) == "bf16"


@pytest.mark.parametrize("bad_precision", [None, 16, True])
def test_validate_precision_rejects_non_strings(bad_precision: object) -> None:
    with pytest.raises(ValueError, match="precision must be a string"):
        validate_precision(bad_precision)  # type: ignore[arg-type]


@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_validate_quantization_accepts_every_storage_format(quantization: str) -> None:
    assert validate_quantization(quantization) == quantization


@pytest.mark.parametrize("bad_quantization", ["fp4", "gptq", "awq", "bf16", ""])
def test_validate_quantization_rejects_unsupported(bad_quantization: str) -> None:
    with pytest.raises(ValueError, match="Unsupported quantization"):
        validate_quantization(bad_quantization)


def test_validate_flag_rejects_non_booleans() -> None:
    with pytest.raises(ValueError, match="grad_checkpoint must be a boolean"):
        validate_flag("false", "grad_checkpoint")


def test_double_quant_is_supported_only_under_nf4() -> None:
    assert double_quant_conflict("nf4") is None
    assert "nothing to quantize" in str(double_quant_conflict("none"))
    assert "applies only to --quant nf4" in str(double_quant_conflict("int8"))


@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_validate_double_quant_rejects_everything_but_nf4(quantization: str) -> None:
    with pytest.raises(ValueError, match="double-quant"):
        validate_double_quant(True, quantization)


@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_validate_double_quant_off_is_always_fine(quantization: str) -> None:
    assert validate_double_quant(False, quantization) is False



_TRAINING_CASES: tuple[tuple[list[str], dict[str, Any], bool], ...] = (
    (["--quant", "nf4", "--double-quant"], {"quantization": "nf4", "double_quant": True}, True),
    (["--quant", "nf4"], {"quantization": "nf4"}, True),
    (["--quant", "int8"], {"quantization": "int8"}, True),
    (["--precision", "fp32"], {"precision": "fp32"}, True),
    (["--quant", "none", "--double-quant"], {"double_quant": True}, False),
    (["--quant", "int8", "--double-quant"], {"quantization": "int8", "double_quant": True}, False),
    (["--precision", "nf4"], {"precision": "nf4"}, False),
    (["--precision", "int8"], {"precision": "int8"}, False),
)

_SERVING_CASES: tuple[tuple[list[str], dict[str, Any], bool], ...] = (
    (["--quant", "nf4", "--double-quant"], {"quantization": "nf4", "double_quant": True}, True),
    (["--quant", "int8"], {"quantization": "int8"}, True),
    (["--quant", "none", "--double-quant"], {"double_quant": True}, False),
    (["--quant", "int8", "--double-quant"], {"quantization": "int8", "double_quant": True}, False),
    (["--precision", "nf4"], {"precision": "nf4"}, False),
)


def _python_accepts_training(model: ModelConfig, overrides: dict[str, Any]) -> bool:
    training = TrainingConfig(lora_rank=16, **overrides)
    try:
        estimate(model, training, get_gpu("4090"))
    except ValueError:
        return False
    return True


def _cli_accepts(argv: list[str]) -> bool:
    result = CliRunner().invoke(main, argv)
    return result.exit_code != _EXIT_USAGE_ERROR


def _repl_session() -> repl._Session:
    console = make_console(no_color=True)
    return repl._Session(console=console, glyphs=repl._ASCII_GLYPHS, ascii_only=True)


def _repl_accepts_training(model: ModelConfig, flags: list[str]) -> bool:
    session = _repl_session()
    try:
        ctx = repl._memory_command().make_context("memory", list(flags))
        training = repl._training_from_args(session, ctx)
        estimate(model, training, get_gpu("4090"))
    except (UsageError, ValueError, repl._ReplError):
        return False
    return True


def _repl_accepts_serving(model: ModelConfig, flags: list[str]) -> bool:
    session = _repl_session()
    try:
        ctx = repl._infer_command().make_context("infer", list(flags))
        serving = repl._serving_from_args(session, ctx)
        estimate_inference(model, serving, get_gpu("4090"))
    except (UsageError, ValueError, repl._ReplError):
        return False
    return True


@pytest.mark.parametrize(("flags", "overrides", "accepted"), _TRAINING_CASES)
def test_training_interfaces_agree(
    llama_model: ModelConfig,
    flags: list[str],
    overrides: dict[str, Any],
    accepted: bool,
) -> None:
    """Python, CLI and REPL take the same training configs, and refuse the same ones."""
    assert _python_accepts_training(llama_model, overrides) is accepted
    assert _cli_accepts(["meta-llama/Llama-3.1-8B", *flags]) is accepted
    assert _repl_accepts_training(llama_model, flags) is accepted


@pytest.mark.parametrize(("flags", "overrides", "accepted"), _SERVING_CASES)
def test_serving_interfaces_agree(
    llama_model: ModelConfig,
    flags: list[str],
    overrides: dict[str, Any],
    accepted: bool,
) -> None:
    """The serving surface enforces the same contract as the training one."""
    serving = ServingConfig(**overrides)
    try:
        estimate_inference(llama_model, serving, get_gpu("4090"))
        python_accepted = True
    except ValueError:
        python_accepted = False

    assert python_accepted is accepted
    assert _cli_accepts(["infer", "meta-llama/Llama-3.1-8B", *flags]) is accepted
    assert _repl_accepts_serving(llama_model, flags) is accepted


def test_json_output_refuses_what_text_output_refuses(llama_model: ModelConfig) -> None:
    for extra in ([], ["--json"]):
        result = CliRunner().invoke(
            main,
            ["meta-llama/Llama-3.1-8B", "--quant", "int8", "--double-quant", *extra],
        )
        assert result.exit_code == _EXIT_USAGE_ERROR
        assert "--double-quant" in result.output


def test_advise_refuses_the_same_pair(llama_model: ModelConfig) -> None:
    result = CliRunner().invoke(
        main,
        [
            "advise",
            "meta-llama/Llama-3.1-8B",
            "--seq-lens",
            "2048",
            "--quant",
            "int8",
            "--double-quant",
        ],
    )

    assert result.exit_code == _EXIT_USAGE_ERROR
    assert "--double-quant" in result.output


def test_low_level_inference_helper_refuses_it_too(llama_model: ModelConfig) -> None:
    with pytest.raises(ValueError, match="double-quant"):
        estimate_inference_memory(llama_model, "fp16", 2048, 1, "int8", True)




@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_sticky_double_quant_is_cleared_by_a_quant_that_cannot_use_it(
    quantization: str,
) -> None:
    session = _repl_session()
    session.training = TrainingConfig(quantization="nf4", double_quant=True)

    ctx = repl._memory_command().make_context("memory", ["--quant", quantization])
    training = repl._training_from_args(session, ctx)

    assert training.quantization == quantization
    assert training.double_quant is False


@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_sticky_serving_double_quant_is_cleared_the_same_way(quantization: str) -> None:
    session = _repl_session()
    session.serving = ServingConfig(quantization="nf4", double_quant=True)

    ctx = repl._infer_command().make_context("infer", ["--quant", quantization])
    serving = repl._serving_from_args(session, ctx)

    assert serving.quantization == quantization
    assert serving.double_quant is False


@pytest.mark.parametrize("quantization", ["none", "int8"])
def test_typing_both_on_one_line_is_still_an_error(quantization: str) -> None:
    session = _repl_session()

    ctx = repl._memory_command().make_context(
        "memory", ["--quant", quantization, "--double-quant"]
    )
    with pytest.raises(UsageError, match="--double-quant"):
        repl._training_from_args(session, ctx)
