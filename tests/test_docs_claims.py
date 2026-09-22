from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from fitcheck.config_parser import ModelConfig
from fitcheck.estimator import TrainingConfig, estimate
from fitcheck.gpu_db import get_gpu
from fitcheck.overhead_db import OVERHEAD_DB

_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = _ROOT / "data" / "measurements"
_MANIFEST = _ARCHIVE / "manifest.json"
_MEMORY = _ROOT / "fitcheck" / "memory"

_README = _ROOT / "README.md"
_CLAUDE = _ROOT / "CLAUDE.md"
_CONTRIBUTING = _ROOT / "CONTRIBUTING.md"

_CHECK_ARGS = [
    "-m",
    "fitcheck.calibrate",
    str(_MANIFEST),
    "--check",
    "--role",
    "calibration",
    "--role",
    "repeat",
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace(" ", " ")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _measurement_files() -> dict[str, list[dict]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))["runs"]
        for path in sorted(_ARCHIVE.glob("*.json"))
        if path != _MANIFEST
    }


def _role_counts() -> Counter[str]:
    return Counter(row["role"] for row in _manifest()["rows"])


def _run_check() -> str:
    result = subprocess.run(
        [sys.executable, *_CHECK_ARGS],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _collect_test_counts() -> tuple[int, int]:
    tail = re.compile(r"(\d+)/(\d+) tests collected|^(\d+) tests collected", re.M)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-cov",
            "--collect-only",
            "-m",
            "network",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    match = tail.search(result.stdout)
    assert match is not None, f"could not read a collection count from:\n{result.stdout}"
    if match.group(1) is not None:
        network, total = int(match.group(1)), int(match.group(2))
    else:
        network = total = int(match.group(3))
    return total - network, network


def _rescore_archive(
    roles: set[str] | None = None,
) -> dict[str, tuple[float, float]]:
    files = _measurement_files()
    scored: dict[str, tuple[float, float]] = {}

    for entry in _manifest()["rows"]:
        if roles is not None and entry["role"] not in roles:
            continue
        row = files[entry["file"]][entry["index"]]
        config = row.get("model_config")
        if config is None:
            continue
        run = row["run"]
        report = estimate(
            ModelConfig(**config),
            TrainingConfig(
                precision=run["precision"],
                quantization=run["quantization"],
                double_quant=run["double_quant"],
                optimizer=run["optimizer"],
                lora_rank=run["lora_rank"],
                lora_targets=list(run["lora_targets"]),
                batch_size=run["batch_size"],
                seq_len=run["seq_len"],
                grad_checkpoint=run["grad_checkpoint"],
                flash_attn=run["kernel"] == "flash",
            ),
            get_gpu(run["gpu_key"]),
        )
        measured = row["measured"]
        allocated = measured["peak_allocated_mib"]
        process = measured["peak_reserved_mib"] + measured["cuda_context_mib"]
        tensors = report.total_mib - report.overhead_mib
        scored[entry["run_id"]] = (
            100.0 * (tensors - allocated) / allocated,
            100.0 * (report.total_mib - process) / process,
        )
    return scored


def _claims_if_present(doc: Path, *claims: str) -> None:
    # Check internal scaffolding when present; tracked docs are the CI fallback.
    if doc.exists():
        _claims(doc, *claims)


def _claims(doc: Path, *claims: str) -> None:
    text = _flat(_text(doc))
    missing = [claim for claim in claims if _flat(claim) not in text]
    assert not missing, (
        f"{doc.name} does not carry these claims as written -- the artifacts moved, "
        f"so update the prose to match:\n  "
        + "\n  ".join(repr(claim) for claim in missing)
    )


def test_the_archive_size_claims_match_the_archive() -> None:
    roles = _role_counts()
    rows = sum(len(runs) for runs in _measurement_files().values())
    scorable = roles["calibration"] + roles["repeat"]

    assert rows == sum(roles.values())

    _claims(
        _README,
        f"{rows} archived runs",
        f"{roles['calibration']} fitted",
        f"{scorable} calibration/repeat rows scored",
        f"{roles['excluded']} legacy rows",
    )
    _claims_if_present(
        _CLAUDE,
        f"{rows} archived rows",
        f"{roles['calibration']} rows fitted, {scorable} scored",
    )


def test_the_shipped_profile_count_is_never_described_as_empty() -> None:
    gpus = sorted({key[0] for key in OVERHEAD_DB})
    assert len(OVERHEAD_DB) == 4 and gpus == ["t4"], (
        f"'four T4 profiles' and now hold {len(OVERHEAD_DB)} over {gpus}"
    )

    for doc in (_README, _CLAUDE):
        if not doc.exists():
            continue
        text = _text(doc).casefold()
        assert "still empty" not in text, f"{doc.name} still calls OVERHEAD_DB empty"

    _claims(_README, "four T4 profiles")


def test_the_headline_accuracy_claim_is_the_output_of_the_documented_command() -> None:
    check = _run_check()
    readme = _text(_README)

    command = " ".join(
        ["python", *_CHECK_ARGS[:2], "data/measurements/manifest.json", *_CHECK_ARGS[3:]]
    )
    assert command in _flat(readme), (
        "the README must quote the exact command whose output it publishes:\n" + command
    )

    for line in check.splitlines():
        if line.strip():
            assert _flat(line) in _flat(readme), (
                f"`--check` prints a line the README does not:\n  {line}\n"
                f"full output:\n{check}"
            )


def test_the_summary_error_figures_are_the_rescored_archive() -> None:
    scored = _rescore_archive({"calibration", "repeat"})
    assert len(scored) == 57, f"expected 57 offline-scorable rows, got {len(scored)}"

    tensors = [pair[0] for pair in scored.values()]
    process = [pair[1] for pair in scored.values()]

    def worst(values: list[float]) -> float:
        return max(abs(value) for value in values)

    def mean(values: list[float]) -> float:
        return sum(abs(value) for value in values) / len(values)

    _claims(
        _README,
        f"| **tensors** — the five physical formulas | **{worst(tensors):.1f}%** "
        f"| {mean(tensors):.1f}% |",
        f"| **process** — the full total, what the verdict uses "
        f"| **{worst(process):.1f}%** | {mean(process):.1f}% |",
    )


def test_the_final_holdout_claims_match_the_imported_rows() -> None:
    scored = _rescore_archive({"holdout"})
    assert len(scored) == 12

    tensors = [pair[0] for pair in scored.values()]
    process = [pair[1] for pair in scored.values()]
    worst_tensor = max(abs(value) for value in tensors)
    mean_process = sum(abs(value) for value in process) / len(process)
    worst_under = min(process)

    _claims(
        _README,
        f"**±{worst_tensor:.1f}%**",
        f"process MAE is **{mean_process:.1f}%**",
        f"worst under-prediction is **−{abs(worst_under):.1f}%**",
        "**12/12 correct**",
        "one Tesla T4 (sm_75), NF4, FP16 compute, LoRA r=16",
    )


def test_the_measurement_stack_is_described_per_session_not_in_one_lump() -> None:
    sessions = _manifest()["sessions"]
    counts = Counter(row["session"] for row in _manifest()["rows"])
    assert len(sessions) == 4

    for name, session in sessions.items():
        stack = (
            f"| `{name}` | {counts[name]} | torch {session['torch']} | "
            f"transformers {session['transformers']} | peft {session['peft']} |"
        )
        _claims(_README, stack)

    assert {session["gpu_name"] for session in sessions.values()} == {"Tesla T4"}
    _claims(
        _README,
        f"All {sum(counts.values())} archived rows are one Tesla T4 (sm_75)",
    )


def test_the_cuda_context_claim_matches_every_archived_row() -> None:
    measured = {
        row["measured"]["cuda_context_mib"]
        for runs in _measurement_files().values()
        for row in runs
    }
    assert measured == {140.875, 141.0}, measured
    counts = Counter(
        row["measured"]["cuda_context_mib"]
        for runs in _measurement_files().values()
        for row in runs
    )
    _claims(
        _README,
        f"140.875 MiB on {counts[140.875]} of them and 141.0 MiB on the other ten",
    )


def test_the_test_counts_are_the_suites_own_counts() -> None:
    offline, network = _collect_test_counts()
    modules = sorted(
        path.stem for path in _MEMORY.glob("*.py") if path.stem != "__init__"
    )
    assert len(modules) == 7, modules

    for doc in (_README, _CONTRIBUTING):
        _claims(
            doc,
            f"{offline} offline tests",
            f"{network} tests marked `network`",
            "100% line coverage on all seven `memory/` modules",
        )


@pytest.mark.parametrize("doc", [_README, _CLAUDE, _CONTRIBUTING])
def test_no_document_still_quotes_a_superseded_population(doc: Path) -> None:
    if not doc.exists():
        pytest.skip(f"{doc.name} is gitignored and absent from this checkout")
    stale = re.compile(r"(?<!\w)33 (runs|measured runs|archived)(?!\w)")
    for number, line in enumerate(_text(doc).splitlines(), start=1):
        if stale.search(line) and "2026-09-12" not in line and "historic" not in line:
            pytest.fail(
                f"{doc.name}:{number} quotes the superseded 33-run population without "
                f"dating it as historical:\n  {line.strip()}"
            )
