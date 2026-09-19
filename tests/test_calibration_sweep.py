from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

_SWEEP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibration_sweep.py"


def _load_sweep() -> Any:
    spec = importlib.util.spec_from_file_location("calibration_sweep", _SWEEP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


_PROBE_INFO = {
    "torch": "2.14.0+cu130",
    "transformers": "5.17.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.48.0",
    "bitsandbytes_error": None,
    "cuda_available": True,
    "gpu": "Tesla T4",
    "capability": "sm_75",
    "total_mib": 15095,
    "arch_list": ["sm_75"],
}


def _args(**overrides: Any) -> Any:
    import argparse

    defaults = {
        "gpu": "t4",
        "quant": "none",
        "precision": "fp16",
        "lora_r": 32,
        "tag": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _command_options(command: list[str]) -> tuple[dict[str, str], set[str]]:
    options: dict[str, str] = {}
    flags: set[str] = set()
    tokens = command[3:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        follower = tokens[index + 1] if index + 1 < len(tokens) else None
        if follower is not None and not follower.startswith("--"):
            options[token] = follower
            index += 2
        else:
            flags.add(token)
            index += 1
    return options, flags


def _payload_for(command: list[str], **overrides: Any) -> dict[str, Any]:
    options, flags = _command_options(command)
    run = {
        "fitcheck_version": "0.3.0",
        "gpu_key": options["--gpu"].strip().casefold(),
        "gpu_name": "Tesla T4",
        "kernel": "flash" if "--flash-attn" in flags else "eager",
        "attn_impl": options.get("--attn-impl", "eager"),
        "seq_len": int(options["--seq-len"]),
        "batch_size": int(options["--batch-size"]),
        "precision": options["--precision"],
        "quantization": options["--quant"],
        "double_quant": "--double-quant" in flags,
        "optimizer": options["--optimizer"],
        "lora_rank": int(options["--lora-r"]),
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "grad_checkpoint": "--grad-checkpoint" in flags,
    }
    run.update(overrides)
    return {
        "model_id": command[2],
        "gpu": "Tesla T4",
        "config": "LoRA r=32",
        "run": run,
        "measured": {"process_mib": 4096.0},
        "error_pct": {"tensors": -1.2, "process": 2.4},
    }


class _FakeRun:
    def __init__(
        self,
        failures: set[str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.failures = failures or set()
        self.overrides = overrides or {}
        self.rows: list[list[str]] = []

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if "-c" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_PROBE_INFO), ""
            )
        self.rows.append(command)
        if command[2] in self.failures:
            return subprocess.CompletedProcess(command, 1, "", "CUDA out of memory")
        payload = _payload_for(command, **self.overrides)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(**kwargs: Any) -> _FakeRun:
        runner = _FakeRun(**kwargs)
        monkeypatch.setattr(sweep.subprocess, "run", runner)
        return runner

    return _install


def _sweep_argv(out: Path, **extra: str) -> list[str]:
    argv = ["--gpu", "t4", "--out", str(out), "--kernels", "eager", "--no-repeats"]
    for name, value in extra.items():
        argv += [f"--{name.replace('_', '-')}", value]
    return argv


# ---------------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------------


def test_the_identity_carries_every_field_that_changes_a_measurement() -> None:
    row = sweep.identity("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 1024, "eager", _args())
    assert set(row) == set(sweep.IDENTITY_FIELDS)


@pytest.mark.parametrize(
    "first,second",
    [
        ({"quant": "none"}, {"quant": "nf4"}),
        ({"precision": "fp16"}, {"precision": "bf16"}),
        ({"lora_r": 32}, {"lora_r": 8}),
        ({"gpu": "t4"}, {"gpu": "4090"}),
    ],
)
def test_configurations_the_old_name_collapsed_now_get_their_own_file(
    first: dict[str, Any], second: dict[str, Any]
) -> None:
    model, batch_size, seq_len, kernel = "TinyLlama/TinyLlama-1.1B", 2, 1024, "eager"
    left = sweep.identity(model, batch_size, seq_len, kernel, _args(**first))
    right = sweep.identity(model, batch_size, seq_len, kernel, _args(**second))
    assert sweep.slug(left) != sweep.slug(right)


def test_every_configuration_in_a_wide_grid_gets_its_own_filename() -> None:
    names = set()
    total = 0
    for gpu in ("t4", "4090"):
        for quant in ("none", "nf4", "int8"):
            for precision in ("fp16", "bf16"):
                for rank in (8, 32):
                    for kernel in ("eager", "flash"):
                        for model, batch_size, seq_len in sweep.GRID:
                            for repeat in (1, 2, 3):
                                args = _args(
                                    gpu=gpu,
                                    quant=quant,
                                    precision=precision,
                                    lora_r=rank,
                                )
                                row = sweep.identity(
                                    model, batch_size, seq_len, kernel, args
                                )
                                names.add(sweep.slug(row, repeat=repeat))
                                total += 1
    assert len(names) == total


def test_two_models_with_the_same_repository_name_do_not_collide() -> None:
    left = sweep.identity("org-a/Llama-3", 2, 1024, "eager", _args())
    right = sweep.identity("org-b/Llama-3", 2, 1024, "eager", _args())
    assert sweep.slug(left) != sweep.slug(right)


def test_a_filename_is_stable_and_has_no_path_separators() -> None:
    row = sweep.identity("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 1024, "eager", _args())
    name = sweep.slug(row, tag="-nf4", repeat=2)
    assert name == sweep.slug(row, tag="-nf4", repeat=2)
    assert "/" not in name and "\\" not in name
    assert name.endswith(sweep.fingerprint(row))


# ---------------------------------------------------------------------------------
# What lands in the file
# ---------------------------------------------------------------------------------


def test_each_row_stores_the_identity_it_was_measured_under(
    tmp_path: Path, fake_run: Any
) -> None:
    fake_run()
    assert sweep.main(_sweep_argv(tmp_path, quant="nf4", precision="bf16")) == 0

    written = sorted(tmp_path.glob("*.json"))
    assert len(written) == len(sweep.GRID)
    for path in written:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["sweep"]["identity"]
        assert set(row) == set(sweep.IDENTITY_FIELDS)
        assert (row["quantization"], row["precision"]) == ("nf4", "bf16")
        assert payload["sweep"]["fingerprint"] == sweep.fingerprint(row)
        assert path.stem.endswith(payload["sweep"]["fingerprint"])
        assert not sweep.identity_diffs(sweep.recorded_identity(payload), row)


def test_the_sweep_asks_for_the_knobs_its_identity_claims(
    tmp_path: Path, fake_run: Any
) -> None:
    runner = fake_run()
    sweep.main(_sweep_argv(tmp_path))

    options, flags = _command_options(runner.rows[0])
    assert options["--optimizer"] == sweep.OPTIMIZER
    assert options["--lora-targets"] == sweep.LORA_TARGETS_PRESET
    assert ("--grad-checkpoint" in flags) is sweep.GRAD_CHECKPOINT
    assert ("--double-quant" in flags) is sweep.DOUBLE_QUANT


# ---------------------------------------------------------------------------------
# Skipping
# ---------------------------------------------------------------------------------


def test_a_matching_file_is_skipped_and_not_re_measured(
    tmp_path: Path, fake_run: Any
) -> None:
    fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 0

    runner = fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 0
    assert runner.rows == []


def test_a_row_written_before_the_identity_block_is_still_skipped(
    tmp_path: Path, fake_run: Any
) -> None:
    fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 0
    for path in tmp_path.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("sweep")
        path.write_text(json.dumps(payload), encoding="utf-8")

    runner = fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 0
    assert runner.rows == []


def test_a_file_whose_identity_does_not_match_is_never_silently_skipped(
    tmp_path: Path, fake_run: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 0
    victim = sorted(tmp_path.glob("*.json"))[0]
    stale = json.loads(victim.read_text(encoding="utf-8"))
    stale["sweep"]["identity"]["seq_len"] += 1
    stale["run"]["seq_len"] += 1
    victim.write_text(json.dumps(stale), encoding="utf-8")
    capsys.readouterr()

    runner = fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 1

    output = capsys.readouterr().out
    assert "STALE FILE" in output
    assert "seq_len" in output
    assert runner.rows == []  # and it was not re-measured over the top of it
    assert json.loads(victim.read_text(encoding="utf-8")) == stale


def test_an_unreadable_file_is_reported_rather_than_skipped(
    tmp_path: Path, fake_run: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 0
    victim = sorted(tmp_path.glob("*.json"))[0]
    victim.write_text("truncated by a dead session {", encoding="utf-8")
    capsys.readouterr()

    fake_run()
    assert sweep.main(_sweep_argv(tmp_path)) == 1
    assert "STALE FILE" in capsys.readouterr().out


# ---------------------------------------------------------------------------------
# Failure is failure
# ---------------------------------------------------------------------------------


def test_one_failed_row_fails_the_sweep(
    tmp_path: Path, fake_run: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run(failures={sweep.GRID[0][0]})
    assert sweep.main(_sweep_argv(tmp_path)) == 1

    output = capsys.readouterr().out
    assert "failed:" in output
    written = {path.stem.split("-ckpt-")[1] for path in tmp_path.glob("*.json")}
    assert all(not name.startswith("HuggingFaceTB_SmolLM2-135M-") for name in written)


def test_a_sweep_that_stops_early_says_so_and_fails(
    tmp_path: Path, fake_run: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run(failures={model for model, _, _ in sweep.GRID})
    assert sweep.main(_sweep_argv(tmp_path)) == 1

    output = capsys.readouterr().out
    assert "STOPPING" in output
    assert "never attempted" in output
    assert list(tmp_path.glob("*.json")) == []


def test_a_row_that_measured_something_else_is_rejected_not_archived(
    tmp_path: Path, fake_run: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run(overrides={"grad_checkpoint": False})
    assert sweep.main(_sweep_argv(tmp_path)) == 1

    output = capsys.readouterr().out
    assert "different run than the one asked for" in output
    assert "grad_checkpoint" in output
    assert list(tmp_path.glob("*.json")) == []


def test_a_preflight_failure_still_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sweep, "preflight", lambda args: None)
    assert sweep.main(_sweep_argv(tmp_path)) == 2
