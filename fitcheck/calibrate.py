# Phase 3: fit the C_overhead constants from measured runs
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from math import log2
from pathlib import Path
from typing import Any

from fitcheck.memory.overhead import estimate_overhead
from fitcheck.overhead_db import (
    DEFAULT_OVERHEAD_PROFILE,
    REFERENCE_SEQ_LEN,
    OverheadProfile,
    get_overhead_profile,
)

DEFAULT_MIN_RUNS = 4
_MIN_SEQ_LEVELS_FOR_SLOPE = 3
_MIN_RUNS_FOR_SLOPE = 6

_KERNELS = ("eager", "flash")


class CalibrationError(ValueError):
    """A measurement file could not be read as calibration input."""


@dataclass(frozen=True)
class CalibrationRun:
    source: str
    model_id: str
    gpu_key: str
    gpu_name: str
    kernel: str
    seq_len: int
    batch_size: int
    weight_mib: float
    activation_mib: float
    tensors_mib: float
    allocated_mib: float
    reserved_mib: float
    context_mib: float
    process_mib: float
    fitcheck_version: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.gpu_key, self.kernel)

    @property
    def basis_mib(self) -> float:
        return self.weight_mib + self.activation_mib

    @property
    def residual_mib(self) -> float:
        return self.process_mib - self.tensors_mib

    @property
    def fragmentation(self) -> float:
        if self.allocated_mib <= 0:
            return 0.0
        return self.reserved_mib / self.allocated_mib - 1.0


@dataclass(frozen=True)
class GroupFit:
    profile: OverheadProfile
    runs: tuple[CalibrationRun, ...]
    errors_pct: tuple[float, ...]

    @property
    def worst_over_pct(self) -> float:
        return max([error for error in self.errors_pct if error > 0] or [0.0])

    @property
    def worst_under_pct(self) -> float:
        return min([error for error in self.errors_pct if error < 0] or [0.0])

    @property
    def worst_abs_pct(self) -> float:
        return max((abs(error) for error in self.errors_pct), default=0.0)

    @property
    def mean_abs_pct(self) -> float:
        if not self.errors_pct:
            return 0.0
        return sum(abs(error) for error in self.errors_pct) / len(self.errors_pct)


@dataclass(frozen=True)
class CalibrationResult:
    fits: dict[tuple[str, str], GroupFit]
    skipped: dict[tuple[str, str], tuple[CalibrationRun, ...]]

    @property
    def worst_abs_pct(self) -> float:
        return max((fit.worst_abs_pct for fit in self.fits.values()), default=0.0)


# ---------------------------------------------------------------------------------
# Reading measure.py output
# ---------------------------------------------------------------------------------


def _require(payload: dict[str, Any], *path: str) -> Any:
    node: Any = payload
    for step in path:
        if not isinstance(node, dict) or step not in node:
            raise CalibrationError(
                f"missing '{'.'.join(path)}'. Calibration needs measure.py output "
                f"produced with --json and WITHOUT --no-predict: the fit compares a "
                f"prediction against a measurement, so it needs both halves."
            )
        node = node[step]
    return node


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"'{name}' must be a number, got {value!r}")
    return float(value)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"'{name}' must be a positive integer, got {value!r}")
    return value


def parse_run(payload: dict[str, Any], source: str = "<memory>") -> CalibrationRun:
    if not isinstance(payload, dict):
        raise CalibrationError(f"{source}: expected a JSON object per run")

    run = _require(payload, "run")
    if not isinstance(run, dict):
        raise CalibrationError(f"{source}: 'run' must be an object")

    gpu_key = run.get("gpu_key")
    if not isinstance(gpu_key, str) or not gpu_key.strip():
        raise CalibrationError(
            f"{source}: 'run.gpu_key' is missing. Re-run measure.py with --gpu <key>; "
            f"a row with no card cannot be filed under one."
        )

    kernel = run.get("kernel")
    if kernel not in _KERNELS:
        raise CalibrationError(
            f"{source}: 'run.kernel' must be one of {_KERNELS}, got {kernel!r}"
        )

    predicted = _require(payload, "predicted")
    measured = _require(payload, "measured")

    reserved_mib = _as_float(
        _require(payload, "measured", "peak_reserved_mib"), "peak_reserved_mib"
    )
    context_mib = _as_float(
        _require(payload, "measured", "cuda_context_mib"), "cuda_context_mib"
    )
    total_mib = _as_float(_require(payload, "predicted", "total_mib"), "total_mib")
    overhead_mib = _as_float(
        _require(payload, "predicted", "overhead_mib"), "overhead_mib"
    )

    return CalibrationRun(
        source=source,
        model_id=str(payload.get("model_id", "?")),
        gpu_key=gpu_key.strip().casefold(),
        gpu_name=str(run.get("gpu_name") or payload.get("gpu") or gpu_key),
        kernel=kernel,
        seq_len=_as_int(run.get("seq_len"), "run.seq_len"),
        batch_size=_as_int(run.get("batch_size", 1), "run.batch_size"),
        weight_mib=_as_float(predicted["weight_mib"], "weight_mib"),
        activation_mib=_as_float(predicted["activation_mib"], "activation_mib"),
        tensors_mib=total_mib - overhead_mib,
        allocated_mib=_as_float(measured["peak_allocated_mib"], "peak_allocated_mib"),
        reserved_mib=reserved_mib,
        context_mib=context_mib,
        process_mib=reserved_mib + context_mib,
        fitcheck_version=str(run.get("fitcheck_version", "?")),
    )


def parse_runs(payload: Any, source: str = "<memory>") -> list[CalibrationRun]:
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return parse_runs(payload["runs"], source)
    if isinstance(payload, dict):
        return [parse_run(payload, source)]
    if isinstance(payload, list):
        return [parse_run(item, f"{source}[{i}]") for i, item in enumerate(payload)]
    raise CalibrationError(f"{source}: expected a JSON object or array of objects")


def load_runs(paths: Iterable[str | Path]) -> list[CalibrationRun]:
    runs: list[CalibrationRun] = []
    for path in paths:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise CalibrationError(f"{path}: cannot read ({error})") from error

        try:
            runs.extend(parse_runs(json.loads(text), str(path)))
            continue
        except json.JSONDecodeError:
            pass

        lines = [line for line in text.splitlines() if line.strip()]
        try:
            documents = [json.loads(line) for line in lines]
        except json.JSONDecodeError as error:
            raise CalibrationError(
                f"{path}: not valid JSON or JSON lines ({error})"
            ) from error
        for index, document in enumerate(documents, 1):
            runs.extend(parse_runs(document, f"{path}:{index}"))

    if not runs:
        raise CalibrationError("no measurement rows found")
    return runs


# ---------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------


def _solve(matrix: list[list[float]], target: list[float]) -> list[float] | None:
    columns = len(matrix[0])
    system = [
        [
            sum(row[i] * row[j] for row in matrix)
            for j in range(columns)
        ]
        + [sum(row[i] * y for row, y in zip(matrix, target, strict=True))]
        for i in range(columns)
    ]

    for i in range(columns):
        pivot = max(range(i, columns), key=lambda r: abs(system[r][i]))
        system[i], system[pivot] = system[pivot], system[i]
        if abs(system[i][i]) < 1e-9:
            return None
        for r in range(columns):
            if r == i:
                continue
            factor = system[r][i] / system[i][i]
            for c in range(i, columns + 1):
                system[r][c] -= factor * system[i][c]

    return [system[i][columns] / system[i][i] for i in range(columns)]


def _design_row(run: CalibrationRun, with_slope: bool) -> list[float]:
    row = [1.0, run.basis_mib]
    if with_slope:
        row.append(run.basis_mib * log2(run.seq_len / REFERENCE_SEQ_LEN))
    return row


def _fit_coefficients(
    runs: Sequence[CalibrationRun], with_slope: bool
) -> tuple[float, float, float] | None:
    matrix = [_design_row(run, with_slope) for run in runs]
    target = [run.residual_mib for run in runs]

    solution = _solve(matrix, target)
    if solution is None:
        return None

    base = solution[0]
    fragmentation = solution[1]
    slope = solution[2] if with_slope else 0.0

    if base < 0.0:
        reduced = _solve([row[1:] for row in matrix], target)
        if reduced is None:
            return None
        base = 0.0
        fragmentation = reduced[0]
        slope = reduced[1] if with_slope else 0.0

    if fragmentation < 0.0:
        base = max(sum(target) / len(target), 0.0)
        fragmentation = 0.0
        slope = 0.0

    return base, fragmentation, slope


def _errors_pct(
    runs: Sequence[CalibrationRun], profile: OverheadProfile
) -> tuple[float, ...]:
    errors = []
    for run in runs:
        predicted = run.tensors_mib + estimate_overhead(
            run.weight_mib, run.activation_mib, profile, run.seq_len
        )
        errors.append(100.0 * (predicted - run.process_mib) / run.process_mib)
    return tuple(errors)


def fit_group(
    runs: Sequence[CalibrationRun],
    *,
    safety_mib: float = 0.0,
    source: str = "",
) -> GroupFit:
    if not runs:
        raise CalibrationError("cannot fit an empty group")

    ordered = sorted(runs, key=lambda run: (run.seq_len, run.model_id))
    with_slope = (
        len({run.seq_len for run in ordered}) >= _MIN_SEQ_LEVELS_FOR_SLOPE
        and len(ordered) >= _MIN_RUNS_FOR_SLOPE
    )

    coefficients = _fit_coefficients(ordered, with_slope)
    if coefficients is None and with_slope:
        coefficients = _fit_coefficients(ordered, False)
    if coefficients is None:
        raise CalibrationError(
            f"{ordered[0].gpu_key}/{ordered[0].kernel}: the rows are degenerate "
            f"(every run has the same W_base + A_act), so the two constants cannot "
            f"be separated. Vary the batch size, the model, or the sequence length."
        )

    seq_min = min(run.seq_len for run in ordered)
    seq_max = max(run.seq_len for run in ordered)

    base, fragmentation, slope = coefficients
    if slope and not _slope_is_usable(fragmentation, slope, seq_min, seq_max):
        fallback = _fit_coefficients(ordered, False)
        if fallback is not None:
            base, fragmentation, slope = fallback

    profile = OverheadProfile(
        gpu=ordered[0].gpu_name,
        kernel=ordered[0].kernel,
        base_context_mib=round(base + safety_mib, 2),
        fragmentation=round(fragmentation, 5),
        fragmentation_per_octave=round(slope, 5),
        seq_len_min=seq_min,
        seq_len_max=seq_max,
        runs=len(ordered),
        source=source or _default_source(ordered),
    )

    errors = _errors_pct(ordered, profile)
    profile = replace(
        profile,
        worst_over_pct=round(max([e for e in errors if e > 0] or [0.0]), 1),
        worst_under_pct=round(min([e for e in errors if e < 0] or [0.0]), 1),
    )

    return GroupFit(profile=profile, runs=tuple(ordered), errors_pct=errors)


def _slope_is_usable(
    fragmentation: float, slope: float, seq_min: int, seq_max: int
) -> bool:
    return all(
        fragmentation + slope * log2(seq / REFERENCE_SEQ_LEN) >= 0.0
        for seq in (seq_min, seq_max)
    )


def _default_source(runs: Sequence[CalibrationRun]) -> str:
    models = sorted({run.model_id.split("/")[-1] for run in runs})
    lengths = sorted({run.seq_len for run in runs})
    versions = sorted({run.fitcheck_version for run in runs})
    return (
        f"{len(runs)} runs, {len(models)} models, seq {lengths[0]}-{lengths[-1]}, "
        f"fitcheck {'/'.join(versions)}"
    )


def calibrate(
    runs: Iterable[CalibrationRun],
    *,
    min_runs: int = DEFAULT_MIN_RUNS,
    safety_mib: float = 0.0,
) -> CalibrationResult:
    if isinstance(min_runs, bool) or not isinstance(min_runs, int) or min_runs < 2:
        raise ValueError("min_runs must be an integer >= 2")
    if isinstance(safety_mib, bool) or not isinstance(safety_mib, (int, float)):
        raise ValueError("safety_mib must be a number")

    groups: dict[tuple[str, str], list[CalibrationRun]] = {}
    for run in runs:
        groups.setdefault(run.key, []).append(run)

    fits: dict[tuple[str, str], GroupFit] = {}
    skipped: dict[tuple[str, str], tuple[CalibrationRun, ...]] = {}
    for key in sorted(groups):
        group = groups[key]
        if len(group) < min_runs:
            skipped[key] = tuple(group)
            continue
        fits[key] = fit_group(group, safety_mib=float(safety_mib))

    return CalibrationResult(fits=fits, skipped=skipped)


def score(
    runs: Iterable[CalibrationRun],
    profiles: dict[tuple[str, str], OverheadProfile] | None = None,
) -> dict[tuple[str, str], tuple[float, ...]]:
    grouped: dict[tuple[str, str], list[CalibrationRun]] = {}
    for run in runs:
        grouped.setdefault(run.key, []).append(run)

    scored: dict[tuple[str, str], tuple[float, ...]] = {}
    for key, group in sorted(grouped.items()):
        if profiles is not None:
            profile = profiles.get(key, DEFAULT_OVERHEAD_PROFILE)
        else:
            profile = get_overhead_profile(key[0], key[1] == "flash")
        scored[key] = _errors_pct(group, profile)
    return scored


# ---------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------


def render_report(result: CalibrationResult) -> str:
    lines: list[str] = []

    for key, fit in sorted(result.fits.items()):
        profile = fit.profile
        lines.append(f"{key[0]} / {key[1]}  ({profile.runs} runs)")
        lines.append(f"  base context          {profile.base_context_mib:>10,.1f} MiB")
        lines.append(f"  fragmentation @{REFERENCE_SEQ_LEN:<5} {profile.fragmentation:>10.4f}")
        lines.append(
            f"  per octave of seq     {profile.fragmentation_per_octave:>10.4f}"
            f"   (seq {profile.seq_len_min}-{profile.seq_len_max})"
        )
        lines.append(
            f"  process error         worst {fit.worst_abs_pct:+.1f}%  "
            f"mean {fit.mean_abs_pct:.1f}%  "
            f"(over {fit.worst_over_pct:+.1f}%, under {fit.worst_under_pct:+.1f}%)"
        )
        lines.append("")
        lines.append(
            f"    {'model':<26} {'seq':>5} {'W+A':>9} {'measured':>9} "
            f"{'frag':>7} {'err':>7}"
        )
        for run, error in zip(fit.runs, fit.errors_pct, strict=True):
            lines.append(
                f"    {run.model_id.split('/')[-1][:26]:<26} {run.seq_len:>5} "
                f"{run.basis_mib:>9,.0f} {run.process_mib:>9,.0f} "
                f"{100 * run.fragmentation:>6.1f}% {error:>+6.1f}%"
            )
        lines.append("")

    for key, group in sorted(result.skipped.items()):
        lines.append(
            f"{key[0]} / {key[1]}  SKIPPED -- {len(group)} run(s), "
            f"below the minimum. Not fitted, so this pair keeps the default profile."
        )

    if result.fits:
        lines.append(f"worst process-tier error across all fits: {result.worst_abs_pct:.1f}%")

    return "\n".join(lines)


def render_python(result: CalibrationResult) -> str:
    """The `OVERHEAD_DB` literal, ready to paste into `fitcheck/overhead_db.py`."""
    lines = ["OVERHEAD_DB: dict[tuple[str, str], OverheadProfile] = {"]
    for key, fit in sorted(result.fits.items()):
        profile = fit.profile
        lines.append(f'    ("{key[0]}", "{key[1]}"): OverheadProfile(')
        lines.append(f'        gpu="{profile.gpu}",')
        lines.append(f'        kernel="{profile.kernel}",')
        lines.append(f"        base_context_mib={profile.base_context_mib},")
        lines.append(f"        fragmentation={profile.fragmentation},")
        lines.append(
            f"        fragmentation_per_octave={profile.fragmentation_per_octave},"
        )
        lines.append(f"        seq_len_min={profile.seq_len_min},")
        lines.append(f"        seq_len_max={profile.seq_len_max},")
        lines.append(f"        runs={profile.runs},")
        lines.append(f"        worst_over_pct={profile.worst_over_pct},")
        lines.append(f"        worst_under_pct={profile.worst_under_pct},")
        lines.append(f'        source="{profile.source}",')
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def render_check(scored: dict[tuple[str, str], tuple[float, ...]]) -> str:
    lines = [f"{'group':<18} {'runs':>5} {'worst':>8} {'mean':>8}"]
    worst_overall = 0.0
    for key, errors in sorted(scored.items()):
        worst = max((abs(e) for e in errors), default=0.0)
        mean = sum(abs(e) for e in errors) / len(errors) if errors else 0.0
        worst_overall = max(worst_overall, worst)
        lines.append(
            f"{key[0] + '/' + key[1]:<18} {len(errors):>5} {worst:>7.1f}% {mean:>7.1f}%"
        )
    lines.append("")
    lines.append(f"worst process-tier error with the shipped constants: {worst_overall:.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fitcheck.calibrate",
        description=(
            "Fit the C_overhead constants in fitcheck/overhead_db.py from "
            "scripts/measure.py --json output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical use:\n"
            "  python -m fitcheck.calibrate runs/*.json\n"
            "  python -m fitcheck.calibrate runs/*.json --emit-python\n"
            "  python -m fitcheck.calibrate runs/*.json --check\n"
        ),
    )
    parser.add_argument("runs", nargs="+", help="measure.py --json files")
    parser.add_argument(
        "--emit-python",
        action="store_true",
        help="Print the OVERHEAD_DB literal instead of the report.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Score the constants fitcheck already ships, without fitting.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=DEFAULT_MIN_RUNS,
        help=f"Rows a (GPU, kernel) pair needs before it is fitted. "
        f"(default: {DEFAULT_MIN_RUNS})",
    )
    parser.add_argument(
        "--safety-mib",
        type=float,
        default=0.0,
        help=(
            "Add this much to every fitted base context. fitcheck biases high on "
            "purpose -- a false 'fits' costs the user an OOM, a false 'does not fit' "
            "costs them a smaller batch. (default: 0)"
        ),
    )
    parser.add_argument(
        "--max-error-pct",
        type=float,
        default=None,
        help="Exit non-zero if the worst process-tier error exceeds this.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        runs = load_runs(args.runs)
    except CalibrationError as error:
        print(f"calibrate: {error}", file=sys.stderr)
        return 2

    if args.check:
        scored = score(runs)
        print(render_check(scored))
        worst = max(
            (abs(e) for errors in scored.values() for e in errors), default=0.0
        )
    else:
        try:
            result = calibrate(
                runs, min_runs=args.min_runs, safety_mib=args.safety_mib
            )
        except CalibrationError as error:
            print(f"calibrate: {error}", file=sys.stderr)
            return 2

        if not result.fits:
            print(
                "calibrate: no (GPU, kernel) pair reached --min-runs "
                f"{args.min_runs}; nothing fitted.",
                file=sys.stderr,
            )
            return 1

        print(render_python(result) if args.emit_python else render_report(result))
        worst = result.worst_abs_pct

    if args.max_error_pct is not None and worst > args.max_error_pct:
        print(
            f"calibrate: worst error {worst:.1f}% exceeds "
            f"--max-error-pct {args.max_error_pct:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
