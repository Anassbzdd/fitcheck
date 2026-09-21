"""`python -m fitcheck` -- the second way in, and the one no entry point covers."""

from __future__ import annotations

import runpy
import subprocess
import sys

import pytest


def test_running_the_package_as_a_module_invokes_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["fitcheck", "--list-gpus"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("fitcheck", run_name="__main__")

    assert exit_info.value.code in (0, None)
    assert "fitcheck GPU database" in capsys.readouterr().out


def test_the_module_entry_point_works_in_a_real_interpreter() -> None:
    """Covers what an in-process run cannot: the package actually being importable."""
    result = subprocess.run(
        [sys.executable, "-m", "fitcheck", "--list-gpus"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "4090" in result.stdout
