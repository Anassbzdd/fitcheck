from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from fitcheck.calibrate import (
    ROLES,
    CalibrationError,
    calibrate,
    load_manifest,
    load_runs,
)

_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = _ROOT / "data" / "measurements"
_MANIFEST = _ARCHIVE / "manifest.json"
_OVERHEAD_DB = _ROOT / "fitcheck" / "overhead_db.py"

_LITERAL = re.compile(
    r"^OVERHEAD_DB: dict\[tuple\[str, str, str\], OverheadProfile\] = \{.*?^\}",
    re.S | re.M,
)


def _manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _archived_row_count() -> int:
    total = 0
    for path in sorted(_ARCHIVE.glob("*.json")):
        if path == _MANIFEST:
            continue
        total += len(json.loads(path.read_text(encoding="utf-8"))["runs"])
    return total


def test_every_archived_row_has_exactly_one_declared_role() -> None:
    rows = _manifest()["rows"]
    pointers = Counter((row["file"], row["index"]) for row in rows)

    assert len(rows) == _archived_row_count()
    assert [pointer for pointer, n in pointers.items() if n > 1] == []
    assert set(pointers) == {
        (path.name, index)
        for path in sorted(_ARCHIVE.glob("*.json"))
        if path != _MANIFEST
        for index in range(len(json.loads(path.read_text(encoding="utf-8"))["runs"]))
    }


def test_run_ids_are_unique_and_every_role_is_known() -> None:
    rows = _manifest()["rows"]
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert {row["role"] for row in rows} <= set(ROLES)
    for row in rows:
        if row["role"] == "excluded":
            assert row.get("reason"), f"{row['run_id']} is excluded with no reason"


def test_every_row_names_the_session_it_was_measured_in() -> None:
    manifest = _manifest()
    sessions = manifest["sessions"]
    for row in manifest["rows"]:
        assert row["session"] in sessions, row["run_id"]
    for name, session in sessions.items():
        for field in ("measured_on", "gpu_name", "torch", "transformers", "peft"):
            assert field in session, f"{name} does not record {field}"


def test_only_calibration_rows_are_loaded_by_default() -> None:
    declared = Counter(row["role"] for row in _manifest()["rows"])
    loaded = load_manifest(_MANIFEST)

    assert len(loaded) == declared["calibration"]
    assert {run.role for run in loaded} == {"calibration"}


def test_repeats_and_excluded_rows_are_reachable_but_never_by_default() -> None:
    declared = Counter(row["role"] for row in _manifest()["rows"])

    assert len(load_manifest(_MANIFEST, ["repeat"])) == declared["repeat"]
    assert len(load_manifest(_MANIFEST, ["excluded"])) == declared["excluded"]
    assert len(load_manifest(_MANIFEST, ROLES)) == len(_manifest()["rows"])


def test_a_glob_over_the_archive_yields_the_declared_set_not_the_directory() -> None:
    """Manifest-covered files are not loaded a second time through the glob."""
    globbed = load_runs(sorted(_ARCHIVE.glob("*.json")))
    declared = load_manifest(_MANIFEST)

    assert [run.run_id for run in globbed] == [run.run_id for run in declared]


def test_mixing_hump_and_legacy_rows_in_one_group_is_refused() -> None:
    """A fitted group cannot mix rows with and without activation humps."""
    runs = load_manifest(_MANIFEST)
    legacy = load_manifest(_MANIFEST, ["excluded"])
    contaminated = [run for run in runs if run.key == legacy[0].key] + [legacy[0]]

    with pytest.raises(CalibrationError, match="carry no activation humps"):
        calibrate(contaminated, min_runs=2)


def test_a_stale_manifest_entry_fails_the_load(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["rows"][0]["identity"]["batch_size"] += 1
    for path in _ARCHIVE.glob("*.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CalibrationError, match="manifest is stale"):
        load_manifest(tmp_path / "manifest.json", ROLES)


def test_a_manifest_from_a_future_schema_is_refused(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 99, "rows": []}), encoding="utf-8"
    )
    with pytest.raises(CalibrationError, match="schema_version"):
        load_manifest(tmp_path / "manifest.json")


def test_the_documented_command_reproduces_the_shipped_database() -> None:
    """One command, byte for byte. This is what makes the constants auditable.

    If this fails, either somebody edited `OVERHEAD_DB` by hand or the declared row
    set changed. Both are fine to do -- but they have to be done together, through
    the manifest, and the diff has to show the coefficients moving.
    """
    emitted = subprocess.run(
        [
            sys.executable,
            "-m",
            "fitcheck.calibrate",
            str(_MANIFEST),
            "--emit-python",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    shipped = _LITERAL.search(_OVERHEAD_DB.read_text(encoding="utf-8"))
    assert shipped is not None, "OVERHEAD_DB literal not found in overhead_db.py"
    assert emitted == shipped.group(0)


def test_the_shipped_source_strings_name_the_manifest_they_came_from() -> None:
    from fitcheck.overhead_db import OVERHEAD_DB

    manifest_id = _manifest()["manifest_id"]
    for key, profile in OVERHEAD_DB.items():
        assert f"manifest {manifest_id}" in profile.source, key
        assert profile.runs > 0, key


def test_every_shipped_profile_traces_back_to_declared_rows() -> None:
    from fitcheck.overhead_db import OVERHEAD_DB

    counts = Counter(run.key for run in load_manifest(_MANIFEST))
    assert set(counts) == set(OVERHEAD_DB)
    for key, profile in OVERHEAD_DB.items():
        assert profile.runs == counts[key], key
