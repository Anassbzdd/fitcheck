from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from fitcheck.calibrate import (
    DEFAULT_MIN_RUNS,
    CalibrationError,
    CalibrationRun,
    calibrate,
    fit_group,
    load_runs,
    main,
    parse_run,
    parse_runs,
    render_check,
    render_python,
    render_report,
    score,
)
from fitcheck.memory.overhead import estimate_overhead
from fitcheck.overhead_db import DEFAULT_OVERHEAD_PROFILE, OverheadProfile

_ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "measurements"
_T4_ROWS = _ARCHIVE / "t4-qlora-2026-09-01.json"


# ---------------------------------------------------------------------------------
# Fixtures: synthetic runs with a known answer
# ---------------------------------------------------------------------------------

_TRUE_BASE = 140.0
_TRUE_FRAGMENTATION = 0.18


def _run(
    *,
    basis_mib: float,
    seq_len: int = 2048,
    kernel: str = "eager",
    gpu_key: str = "t4",
    base: float = _TRUE_BASE,
    fragmentation: float = _TRUE_FRAGMENTATION,
    slope: float = 0.0,
    noise_mib: float = 0.0,
    model_id: str = "org/model",
) -> CalibrationRun:
    """A run whose process total is exactly base + frag(seq) * (W + A) + tensors."""
    from math import log2

    tensors = 1_000.0 + basis_mib
    effective = fragmentation + slope * log2(seq_len / 2048)
    process = tensors + base + effective * basis_mib + noise_mib
    return CalibrationRun(
        source="<test>",
        model_id=model_id,
        gpu_key=gpu_key,
        gpu_name="Tesla T4",
        kernel=kernel,
        seq_len=seq_len,
        batch_size=1,
        weight_mib=basis_mib * 0.25,
        activation_mib=basis_mib * 0.75,
        tensors_mib=tensors,
        allocated_mib=tensors,
        reserved_mib=process - 141.0,
        context_mib=141.0,
        process_mib=process,
        fitcheck_version="0.3.0",
    )


def _synthetic_group(**overrides: object) -> list[CalibrationRun]:
    return [
        _run(basis_mib=basis, **overrides)  # type: ignore[arg-type]
        for basis in (1_000.0, 2_500.0, 4_000.0, 6_500.0, 9_000.0)
    ]


def _payload(**run_overrides: object) -> dict[str, object]:
    run = {
        "fitcheck_version": "0.3.0",
        "gpu_key": "t4",
        "gpu_name": "Tesla T4",
        "kernel": "eager",
        "seq_len": 2048,
        "batch_size": 2,
    }
    run.update(run_overrides)
    return {
        "model_id": "org/model",
        "gpu": "Tesla T4",
        "run": run,
        "measured": {
            "peak_allocated_mib": 5_000.0,
            "peak_reserved_mib": 6_000.0,
            "cuda_context_mib": 141.0,
        },
        "predicted": {
            "weight_mib": 1_000.0,
            "activation_mib": 3_000.0,
            "overhead_mib": 700.0,
            "total_mib": 5_700.0,
        },
    }


# ---------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------


def test_parse_run_reduces_a_measure_py_payload_to_the_fit_inputs() -> None:
    run = parse_run(_payload())

    assert run.key == ("t4", "eager", "none")
    assert run.basis_mib == pytest.approx(4_000.0)
    # tensors = predicted total minus the overhead the prediction already carried.
    assert run.tensors_mib == pytest.approx(5_000.0)
    # process = the allocator pool plus the context outside it.
    assert run.process_mib == pytest.approx(6_141.0)
    assert run.residual_mib == pytest.approx(1_141.0)
    assert run.fragmentation == pytest.approx(0.2)


def test_parse_runs_accepts_a_bare_object_a_list_and_a_runs_wrapper() -> None:
    one = _payload()

    assert len(parse_runs(one)) == 1
    assert len(parse_runs([one, one])) == 2
    assert len(parse_runs({"_provenance": "...", "runs": [one, one, one]})) == 3


def test_a_run_without_a_prediction_is_refused_with_the_reason() -> None:
    payload = _payload()
    del payload["predicted"]

    with pytest.raises(CalibrationError, match="--no-predict"):
        parse_run(payload)


def test_a_run_with_no_gpu_key_is_refused() -> None:
    payload = _payload()
    payload["run"].pop("gpu_key")  # type: ignore[union-attr]

    with pytest.raises(CalibrationError, match="gpu_key"):
        parse_run(payload)


def test_a_run_with_an_unknown_kernel_is_refused() -> None:
    with pytest.raises(CalibrationError, match=re.escape("run.kernel")):
        parse_run(_payload(kernel="sdpa"))


def test_a_run_with_a_bad_seq_len_is_refused() -> None:
    with pytest.raises(CalibrationError, match=re.escape("run.seq_len")):
        parse_run(_payload(seq_len=0))


def test_a_run_with_a_non_finite_measurement_is_refused() -> None:
    # json.load() reads the bare literals NaN and Infinity, and one of either would
    # poison every coefficient fitted from the group without a word.
    payload = _payload()
    payload["measured"]["peak_reserved_mib"] = float("nan")  # type: ignore[index]

    with pytest.raises(CalibrationError, match="must be a finite number"):
        parse_run(payload)


def test_load_runs_refuses_a_file_holding_a_nan_literal(tmp_path: Path) -> None:
    document = tmp_path / "nan.json"
    document.write_text(
        json.dumps(_payload()).replace('"peak_allocated_mib": 5000.0', '"peak_allocated_mib": NaN'),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="must be a finite number"):
        load_runs([document])


def test_load_runs_reads_json_and_json_lines(tmp_path: Path) -> None:
    document = tmp_path / "one.json"
    document.write_text(json.dumps([_payload(), _payload()]), encoding="utf-8")

    lines = tmp_path / "many.jsonl"
    lines.write_text(
        "\n".join(json.dumps(_payload()) for _ in range(3)), encoding="utf-8"
    )

    assert len(load_runs([document])) == 2
    assert len(load_runs([lines])) == 3
    assert len(load_runs([document, lines])) == 5


def test_load_runs_reports_the_file_that_could_not_be_read(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(CalibrationError, match=re.escape("broken.json")):
        load_runs([broken])

    with pytest.raises(CalibrationError, match=re.escape("missing.json")):
        load_runs([tmp_path / "missing.json"])


# ---------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------


def test_the_fit_recovers_the_constants_it_was_generated_from() -> None:
    fit = fit_group(_synthetic_group())

    assert fit.profile.base_context_mib == pytest.approx(_TRUE_BASE, abs=0.5)
    assert fit.profile.fragmentation == pytest.approx(_TRUE_FRAGMENTATION, abs=1e-3)
    assert fit.worst_abs_pct < 0.01


def test_the_fit_recovers_a_sequence_slope_when_the_rows_span_enough_lengths() -> None:
    runs = [
        _run(basis_mib=basis, seq_len=seq_len, slope=0.05)
        for seq_len in (512, 1024, 2048, 4096)
        for basis in (1_500.0, 5_000.0)
    ]

    fit = fit_group(runs)

    assert fit.profile.fragmentation_per_octave == pytest.approx(0.05, abs=1e-3)
    assert fit.profile.seq_len_min == 512
    assert fit.profile.seq_len_max == 4096
    assert fit.worst_abs_pct < 0.01


def test_a_slope_is_not_fitted_from_too_few_rows() -> None:
    """Five points and three parameters traces the noise; the T4 eager group proved it."""
    runs = [
        _run(basis_mib=basis, seq_len=seq_len, slope=0.05)
        for seq_len, basis in (
            (512, 1_500.0),
            (1_024, 2_000.0),
            (2_048, 4_000.0),
            (2_048, 6_000.0),
            (4_096, 9_000.0),
        )
    ]

    assert fit_group(runs).profile.fragmentation_per_octave == 0.0


def test_a_slope_that_would_go_negative_inside_its_own_range_is_dropped() -> None:
    runs = [
        _run(basis_mib=basis, seq_len=seq_len, fragmentation=0.02, slope=0.30)
        for seq_len in (512, 1024, 2048, 4096)
        for basis in (1_500.0, 5_000.0)
    ]

    profile = fit_group(runs).profile

    assert profile.fragmentation_per_octave == 0.0
    assert profile.fragmentation >= 0.0


def test_the_fitted_constants_are_never_negative() -> None:
    """Rows whose overhead shrinks with size still have to produce a physical profile."""
    runs = [
        _run(basis_mib=basis, base=50.0, fragmentation=0.0, noise_mib=-basis * 0.02)
        for basis in (1_000.0, 3_000.0, 6_000.0, 9_000.0)
    ]

    profile = fit_group(runs).profile

    assert profile.base_context_mib >= 0.0
    assert profile.fragmentation >= 0.0


def test_degenerate_rows_are_refused_rather_than_fitted() -> None:
    runs = [_run(basis_mib=4_000.0) for _ in range(5)]

    with pytest.raises(CalibrationError, match="degenerate"):
        fit_group(runs)


def test_the_safety_margin_only_lifts_the_context_term() -> None:
    plain = fit_group(_synthetic_group()).profile
    padded = fit_group(_synthetic_group(), safety_mib=200.0).profile

    assert padded.base_context_mib == pytest.approx(plain.base_context_mib + 200.0)
    assert padded.fragmentation == plain.fragmentation


def test_a_fitted_profile_records_its_own_worst_case_both_ways() -> None:
    runs = _synthetic_group()
    runs[0] = _run(basis_mib=1_000.0, noise_mib=300.0)
    runs[-1] = _run(basis_mib=9_000.0, noise_mib=-300.0)

    profile = fit_group(runs).profile

    assert profile.worst_over_pct > 0.0
    assert profile.worst_under_pct < 0.0
    assert profile.runs == len(runs)
    assert "runs" in profile.source


# ---------------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------------


def test_groups_are_fitted_separately_per_gpu_and_kernel() -> None:
    runs = [
        *_synthetic_group(kernel="eager", fragmentation=0.25),
        *_synthetic_group(kernel="flash", fragmentation=0.08),
        *_synthetic_group(gpu_key="p100", kernel="eager", fragmentation=0.12),
    ]

    result = calibrate(runs)

    assert set(result.fits) == {
        ("t4", "eager", "none"),
        ("t4", "flash", "none"),
        ("p100", "eager", "none"),
    }
    assert result.fits[("t4", "eager", "none")].profile.fragmentation == pytest.approx(
        0.25, abs=1e-3
    )
    assert result.fits[("t4", "flash", "none")].profile.fragmentation == pytest.approx(
        0.08, abs=1e-3
    )
    assert result.fits[("p100", "eager", "none")].profile.fragmentation == pytest.approx(
        0.12, abs=1e-3
    )


def test_a_group_below_the_minimum_is_skipped_not_fitted() -> None:
    runs = [
        *_synthetic_group(),
        _run(basis_mib=2_000.0, kernel="flash"),
        _run(basis_mib=4_000.0, kernel="flash"),
    ]

    result = calibrate(runs, min_runs=DEFAULT_MIN_RUNS)

    assert set(result.fits) == {("t4", "eager", "none")}
    assert set(result.skipped) == {("t4", "flash", "none")}
    assert len(result.skipped[("t4", "flash", "none")]) == 2


@pytest.mark.parametrize("bad_value", [1, 0, -3, True, "4", 4.0])
def test_calibrate_rejects_a_bad_minimum(bad_value: object) -> None:
    with pytest.raises(ValueError, match="min_runs must be an integer >= 2"):
        calibrate(_synthetic_group(), min_runs=bad_value)


def test_calibrate_rejects_a_non_numeric_safety_margin() -> None:
    with pytest.raises(ValueError, match="safety_mib must be a number"):
        calibrate(_synthetic_group(), safety_mib="200")


# ---------------------------------------------------------------------------------
# Scoring the shipped constants
# ---------------------------------------------------------------------------------


def test_score_grades_the_shipped_constants_without_fitting() -> None:
    runs = _synthetic_group()

    scored = score(runs)

    assert set(scored) == {("t4", "eager", "none")}
    for run, error in zip(
        sorted(runs, key=lambda r: r.basis_mib), scored[("t4", "eager", "none")], strict=True
    ):
        predicted = run.tensors_mib + estimate_overhead(
            run.weight_mib, run.activation_mib, DEFAULT_OVERHEAD_PROFILE, run.seq_len
        )
        assert error == pytest.approx(
            100.0 * (predicted - run.process_mib) / run.process_mib
        )


def test_score_can_grade_a_candidate_profile_instead() -> None:
    runs = _synthetic_group()
    perfect = OverheadProfile(
        gpu="Tesla T4",
        kernel="eager",
        base_context_mib=_TRUE_BASE,
        fragmentation=_TRUE_FRAGMENTATION,
        runs=len(runs),
    )

    errors = score(runs, {("t4", "eager", "none"): perfect})[("t4", "eager", "none")]

    assert max(abs(error) for error in errors) < 1e-9


# ---------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------


def test_the_emitted_literal_evaluates_back_to_the_fitted_profiles() -> None:
    result = calibrate(_synthetic_group())
    namespace: dict[str, object] = {"OverheadProfile": OverheadProfile}

    exec(render_python(result), namespace)

    rebuilt = namespace["OVERHEAD_DB"]
    assert rebuilt == {key: fit.profile for key, fit in result.fits.items()}


def test_the_report_names_every_group_and_every_row() -> None:
    result = calibrate(
        [*_synthetic_group(), _run(basis_mib=2_000.0, kernel="flash")]
    )

    report = render_report(result)

    assert "t4 / eager" in report
    assert "SKIPPED" in report
    assert report.count("model") >= 5  # the row table lists every fitted run
    assert "worst process-tier error" in report


def test_render_check_summarises_one_line_per_group() -> None:
    text = render_check(score(_synthetic_group()))

    assert "t4/eager" in text
    assert "worst process-tier error with the shipped constants" in text


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def _write(tmp_path: Path, runs: list[dict[str, object]]) -> Path:
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs), encoding="utf-8")
    return path


def _payloads_spanning_sizes() -> list[dict[str, object]]:
    payloads = []
    for activation in (1_000.0, 2_000.0, 4_000.0, 8_000.0, 12_000.0):
        payload = _payload()
        payload["predicted"] = {  # type: ignore[index]
            "weight_mib": 1_000.0,
            "activation_mib": activation,
            "overhead_mib": 500.0 + 0.05 * (1_000.0 + activation),
            "total_mib": 2_000.0 + activation + 500.0 + 0.05 * (1_000.0 + activation),
        }
        payload["measured"] = {  # type: ignore[index]
            "peak_allocated_mib": 2_000.0 + activation,
            "peak_reserved_mib": (2_000.0 + activation) * 1.15,
            "cuda_context_mib": 141.0,
        }
        payloads.append(payload)
    return payloads


def test_cli_prints_a_report_and_exits_zero(tmp_path, capsys) -> None:
    path = _write(tmp_path, _payloads_spanning_sizes())

    assert main([str(path)]) == 0
    assert "t4 / eager" in capsys.readouterr().out


def test_cli_emit_python_prints_a_pasteable_literal(tmp_path, capsys) -> None:
    path = _write(tmp_path, _payloads_spanning_sizes())

    assert main([str(path), "--emit-python"]) == 0
    assert capsys.readouterr().out.startswith("OVERHEAD_DB")


def test_cli_check_grades_the_shipped_constants(tmp_path, capsys) -> None:
    path = _write(tmp_path, _payloads_spanning_sizes())

    assert main([str(path), "--check"]) == 0
    assert "shipped constants" in capsys.readouterr().out


def test_cli_exits_one_when_nothing_reaches_the_minimum(tmp_path, capsys) -> None:
    path = _write(tmp_path, _payloads_spanning_sizes()[:2])

    assert main([str(path)]) == 1
    assert "--min-runs" in capsys.readouterr().err


def test_cli_exits_one_when_the_fit_misses_the_error_budget(tmp_path, capsys) -> None:
    payloads = _payloads_spanning_sizes()
    # Bend one row away from the line so the fit has a residual to fail on.
    payloads[0]["measured"]["peak_reserved_mib"] *= 1.4  # type: ignore[index]
    path = _write(tmp_path, payloads)

    assert main([str(path), "--max-error-pct", "0.0001"]) == 1
    assert "exceeds" in capsys.readouterr().err


def test_cli_exits_two_on_unreadable_input(tmp_path, capsys) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("nope", encoding="utf-8")

    assert main([str(broken)]) == 2
    assert "calibrate:" in capsys.readouterr().err


# ---------------------------------------------------------------------------------
# The archived T4 rows
# ---------------------------------------------------------------------------------


def test_the_archived_t4_rows_parse_and_group_as_expected() -> None:
    runs = load_runs([_T4_ROWS])

    assert len(runs) == 10
    # The archive is QLoRA, so quantization -- now part of the key -- is nf4.
    assert {run.key for run in runs} == {("t4", "eager", "nf4"), ("t4", "flash", "nf4")}
    # The measured CUDA context is nowhere near the 500 MiB the default profile bills.
    assert all(130.0 <= run.context_mib <= 150.0 for run in runs)
    # And the measured fragmentation is nowhere near the flat 5% either.
    assert all(run.fragmentation > 0.05 for run in runs)


def test_the_archived_t4_rows_fit_and_the_flash_group_meets_the_budget() -> None:
    """The half of 9.3 the data in hand can settle.

    `flash` fits inside the 8% the task asks for. `eager` does not, on five rows --
    which is why `OVERHEAD_DB` is still empty. If this test starts failing because
    `eager` got better, that is the sweep landing, not a regression.
    """
    result = calibrate(load_runs([_T4_ROWS]))

    assert set(result.fits) == {("t4", "eager", "nf4"), ("t4", "flash", "nf4")}
    assert result.fits[("t4", "flash", "nf4")].worst_abs_pct < 8.0
    assert result.fits[("t4", "eager", "nf4")].worst_abs_pct > 8.0


def test_fitting_the_archived_rows_beats_the_constants_fitcheck_ships() -> None:
    runs = load_runs([_T4_ROWS])
    result = calibrate(runs)
    shipped = score(runs)

    for key, fit in result.fits.items():
        shipped_worst = max(abs(error) for error in shipped[key])
        assert fit.worst_abs_pct < shipped_worst


# ---------------------------------------------------------------------------------
# Gradient checkpointing: the hump form does not exist without it
# ---------------------------------------------------------------------------------


def _hump_run(*, grad_checkpoint: bool | None, basis_mib: float) -> CalibrationRun:
    return replace(
        _run(basis_mib=basis_mib),
        grad_checkpoint=grad_checkpoint,
        logits_mib=basis_mib * 0.5,
        layer_mib=basis_mib * 0.2,
    )


def test_the_checkpointing_flag_survives_parsing() -> None:
    assert parse_run(_payload(grad_checkpoint=False)).grad_checkpoint is False
    assert parse_run(_payload(grad_checkpoint=True)).grad_checkpoint is True
    assert parse_run(_payload()).grad_checkpoint is None


@pytest.mark.parametrize("value", ["false", 0, 1, "true"])
def test_a_non_boolean_checkpointing_flag_is_refused_not_coerced(value: object) -> None:
    with pytest.raises(CalibrationError, match="grad_checkpoint"):
        parse_run(_payload(grad_checkpoint=value))


@pytest.mark.parametrize("flag", [False, None])
def test_the_hump_form_refuses_rows_that_did_not_run_checkpointing(
    flag: bool | None,
) -> None:
    runs = [
        _hump_run(grad_checkpoint=flag, basis_mib=basis)
        for basis in (1_000.0, 2_500.0, 4_000.0, 6_500.0)
    ]

    with pytest.raises(CalibrationError, match="gradient checkpointing ON"):
        fit_group(runs)


def test_a_group_that_disagrees_on_checkpointing_is_refused() -> None:
    runs = [
        _hump_run(grad_checkpoint=True, basis_mib=1_000.0),
        _hump_run(grad_checkpoint=True, basis_mib=2_500.0),
        _hump_run(grad_checkpoint=False, basis_mib=4_000.0),
    ]

    with pytest.raises(CalibrationError, match="disagree on gradient checkpointing"):
        fit_group(runs)


def test_checkpointing_off_rows_cannot_move_a_checkpointing_on_fit() -> None:
    clean = [
        _hump_run(grad_checkpoint=True, basis_mib=basis)
        for basis in (1_000.0, 2_500.0, 4_000.0, 6_500.0)
    ]
    baseline = fit_group(clean).profile

    with pytest.raises(CalibrationError):
        fit_group([*clean, _hump_run(grad_checkpoint=False, basis_mib=9_000.0)])

    assert fit_group(clean).profile == baseline
