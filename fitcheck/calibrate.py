# Phase 3: fit the C_overhead constants from measured runs
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from math import isfinite, log2
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
_MIN_RUNS_PER_HUMP_BRANCH = 3

_KERNELS = ("eager", "flash")

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
ROLES = ("calibration", "holdout", "repeat", "excluded")
DEFAULT_ROLES = ("calibration",)


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
    quantization: str = "none"
    grad_checkpoint: bool | None = None
    logits_mib: float | None = None
    layer_mib: float | None = None
    run_id: str = ""
    role: str = ""
    provenance: str = ""

    @property
    def label(self) -> str:
        return self.run_id or self.source

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.gpu_key, self.kernel, self.quantization)

    @property
    def basis_mib(self) -> float:
        return self.weight_mib + self.activation_mib

    @property
    def residual_mib(self) -> float:
        return self.process_mib - self.tensors_mib

    @property
    def has_humps(self) -> bool:
        return self.logits_mib is not None and self.layer_mib is not None

    @property
    def hump_mib(self) -> float:
        """min(A_logits, A_layer) -- the leftover the allocator over-reserves for."""
        if self.logits_mib is None or self.layer_mib is None:
            raise CalibrationError(f"{self.source}: row carries no activation humps")
        return min(self.logits_mib, self.layer_mib)

    @property
    def layer_wins(self) -> bool:
        if self.logits_mib is None or self.layer_mib is None:
            raise CalibrationError(f"{self.source}: row carries no activation humps")
        return self.layer_mib > self.logits_mib

    @property
    def over_reserve_mib(self) -> float:
        """What the fit actually explains: process minus tensors minus the context.

        `context_mib` is measured, not fitted, so it comes straight off the row.
        """
        return self.process_mib - self.tensors_mib - self.context_mib

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
    fits: dict[tuple[str, str, str], GroupFit]
    skipped: dict[tuple[str, str, str], tuple[CalibrationRun, ...]]

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
    if not isfinite(value):
        raise CalibrationError(f"'{name}' must be a finite number, got {value!r}")
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

    quantization = run.get("quantization", "none")
    if not isinstance(quantization, str) or not quantization.strip():
        raise CalibrationError(
            f"{source}: 'run.quantization' must be a non-empty string, "
            f"got {quantization!r}"
        )

    grad_checkpoint = run.get("grad_checkpoint")
    if grad_checkpoint is not None and not isinstance(grad_checkpoint, bool):
        raise CalibrationError(
            f"{source}: 'run.grad_checkpoint' must be true or false, "
            f"got {grad_checkpoint!r}"
        )

    logits_mib = predicted.get("activation_logits_mib")
    layer_mib = predicted.get("activation_layer_mib")
    if logits_mib is not None:
        logits_mib = _as_float(logits_mib, "activation_logits_mib")
    if layer_mib is not None:
        layer_mib = _as_float(layer_mib, "activation_layer_mib")

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
        quantization=quantization.strip().casefold(),
        grad_checkpoint=grad_checkpoint,
        logits_mib=logits_mib,
        layer_mib=layer_mib,
    )


def parse_runs(payload: Any, source: str = "<memory>") -> list[CalibrationRun]:
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return parse_runs(payload["runs"], source)
    if isinstance(payload, dict):
        return [parse_run(payload, source)]
    if isinstance(payload, list):
        return [parse_run(item, f"{source}[{i}]") for i, item in enumerate(payload)]
    raise CalibrationError(f"{source}: expected a JSON object or array of objects")


# ---------------------------------------------------------------------------------
# The manifest: which archived rows are allowed to be fitted
# ---------------------------------------------------------------------------------


def is_manifest(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and "schema_version" in payload
        and isinstance(payload.get("rows"), list)
    )


def _model_config_sha256(row: dict[str, Any]) -> str | None:
    config = row.get("model_config")
    if config is None:
        return None
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _check_identity(entry: dict[str, Any], row: dict[str, Any], run_id: str) -> None:
    declared_model = entry.get("model_id")
    if declared_model is not None and declared_model != row.get("model_id"):
        raise CalibrationError(
            f"{run_id}: manifest says model_id {declared_model!r}, the archived row "
            f"says {row.get('model_id')!r}. The manifest is stale."
        )

    run = row.get("run")
    identity = entry.get("identity") or {}
    if identity and not isinstance(run, dict):
        raise CalibrationError(f"{run_id}: the archived row has no 'run' block")
    for field, declared in identity.items():
        actual = run.get(field) if isinstance(run, dict) else None
        if isinstance(declared, list):
            actual = list(actual) if isinstance(actual, list) else actual
        if declared != actual:
            raise CalibrationError(
                f"{run_id}: manifest declares {field}={declared!r}, the archived row "
                f"has {actual!r}. The manifest is stale."
            )

    declared_hash = entry.get("model_config_sha256")
    actual_hash = _model_config_sha256(row)
    if declared_hash != actual_hash:
        raise CalibrationError(
            f"{run_id}: the archived model_config does not match the manifest "
            f"({declared_hash} vs {actual_hash}). The measurement changed underneath "
            f"the declaration."
        )


def load_manifest(
    path: str | Path, roles: Iterable[str] = DEFAULT_ROLES
) -> list[CalibrationRun]:
    path = Path(path)
    wanted = tuple(roles)
    for name in wanted:
        if name not in ROLES:
            raise CalibrationError(f"unknown role {name!r}; expected one of {ROLES}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CalibrationError(f"{path}: cannot read ({error})") from error
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{path}: not valid JSON ({error})") from error

    if not is_manifest(payload):
        raise CalibrationError(f"{path}: not a manifest (no schema_version / rows)")

    version = payload.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise CalibrationError(
            f"{path}: manifest schema_version {version!r}, this fitcheck reads "
            f"{MANIFEST_SCHEMA_VERSION}"
        )

    manifest_id = str(payload.get("manifest_id", path.stem))
    provenance = f"manifest {manifest_id}"
    cache: dict[Path, list[dict[str, Any]]] = {}

    runs: list[CalibrationRun] = []
    seen: set[str] = set()
    for position, entry in enumerate(payload["rows"]):
        if not isinstance(entry, dict):
            raise CalibrationError(f"{path}: rows[{position}] is not an object")

        run_id = entry.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise CalibrationError(f"{path}: rows[{position}] has no run_id")
        if run_id in seen:
            raise CalibrationError(f"{path}: duplicate run_id {run_id!r}")
        seen.add(run_id)

        role = entry.get("role")
        if role not in ROLES:
            raise CalibrationError(
                f"{run_id}: role {role!r} is not one of {ROLES}"
            )
        if role == "excluded" and not entry.get("reason"):
            raise CalibrationError(
                f"{run_id}: an excluded row must say why it is excluded"
            )
        if role not in wanted:
            continue

        source_file = entry.get("file")
        index = entry.get("index")
        if not isinstance(source_file, str) or not isinstance(index, int):
            raise CalibrationError(
                f"{run_id}: needs 'file' and an integer 'index' into its 'runs' array"
            )

        target = path.parent / source_file
        if target not in cache:
            try:
                document = json.loads(target.read_text(encoding="utf-8"))
            except OSError as error:
                raise CalibrationError(
                    f"{run_id}: cannot read {target} ({error})"
                ) from error
            except json.JSONDecodeError as error:
                raise CalibrationError(
                    f"{run_id}: {target} is not valid JSON ({error})"
                ) from error
            rows = document.get("runs") if isinstance(document, dict) else document
            if not isinstance(rows, list):
                raise CalibrationError(f"{run_id}: {target} holds no 'runs' array")
            cache[target] = rows

        rows = cache[target]
        if not 0 <= index < len(rows):
            raise CalibrationError(
                f"{run_id}: index {index} is outside {source_file} "
                f"({len(rows)} rows)"
            )

        row = rows[index]
        _check_identity(entry, row, run_id)
        runs.append(
            replace(
                parse_run(row, f"{source_file}[{index}]"),
                run_id=run_id,
                role=role,
                provenance=provenance,
            )
        )

    if not runs:
        raise CalibrationError(
            f"{path}: no rows with role {'/'.join(wanted)}. "
            f"The manifest declares {len(payload['rows'])} rows in total."
        )
    return runs


def _manifest_coverage(path: Path, payload: dict[str, Any]) -> set[Path]:
    covered: set[Path] = set()
    for entry in payload.get("rows", []):
        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
            covered.add((path.parent / entry["file"]).resolve())
    return covered


def load_runs(
    paths: Iterable[str | Path], roles: Iterable[str] = DEFAULT_ROLES
) -> list[CalibrationRun]:
    resolved: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        resolved.append(path)

    manifests: list[Path] = []
    covered: set[Path] = set()
    for path in resolved:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if is_manifest(payload):
            manifests.append(path)
            covered |= _manifest_coverage(path, payload)

    runs: list[CalibrationRun] = []
    for path in manifests:
        runs.extend(load_manifest(path, roles))

    for path in resolved:
        if path in manifests:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise CalibrationError(f"{path}: cannot read ({error})") from error

        if path.resolve() in covered:
            continue

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


def _origin_slope(runs: Sequence[CalibrationRun]) -> float | None:
    """Least squares through the origin: over_reserve = F * min(A_logits, A_layer).

    No intercept, because the intercept is the CUDA context and that is measured on
    every row rather than fitted. Letting it float was the pre-9.3 defect: the
    constant column is collinear with the size column, so the two traded against each
    other and a single dropped row could move F by 172%.
    """
    denominator = sum(run.hump_mib**2 for run in runs)
    if denominator <= 0.0:
        return None
    numerator = sum(run.hump_mib * run.over_reserve_mib for run in runs)
    return max(numerator / denominator, 0.0)


def _fit_hump(
    runs: Sequence[CalibrationRun],
) -> tuple[float, float | None] | None:
    """Fit the hump form, splitting by which hump wins the checkpointed peak.

    The leftover is a different shape depending on which one wins -- the LM-head
    logits are one enormous block, a layer's activations are many medium ones -- so
    they get separate coefficients whenever both branches have enough rows. With only
    one branch populated (every `flash` group, where the score matrix is gone and
    `A_layer` always loses) a single pooled coefficient is fitted and the layer-win
    slot is left empty for `fragmentation_for` to fall back on.
    """
    logits_win = [run for run in runs if not run.layer_wins]
    layer_win = [run for run in runs if run.layer_wins]

    if (
        len(logits_win) >= _MIN_RUNS_PER_HUMP_BRANCH
        and len(layer_win) >= _MIN_RUNS_PER_HUMP_BRANCH
    ):
        first = _origin_slope(logits_win)
        second = _origin_slope(layer_win)
        if first is None or second is None:
            return None
        return first, second

    pooled = _origin_slope(runs)
    if pooled is None:
        return None
    return pooled, None


def _errors_pct(
    runs: Sequence[CalibrationRun], profile: OverheadProfile
) -> tuple[float, ...]:
    errors = []
    for run in runs:
        predicted = run.tensors_mib + estimate_overhead(
            run.weight_mib,
            run.activation_mib,
            profile,
            run.seq_len,
            logits_mib=run.logits_mib,
            layer_mib=run.layer_mib,
        )
        errors.append(100.0 * (predicted - run.process_mib) / run.process_mib)
    return tuple(errors)


def _group_label(runs: Sequence[CalibrationRun]) -> str:
    first = runs[0]
    return f"{first.gpu_key}/{first.kernel}/quant={first.quantization}"


def _flag_counts(runs: Sequence[CalibrationRun]) -> str:
    counts: dict[str, int] = {}
    for run in runs:
        counts[repr(run.grad_checkpoint)] = counts.get(repr(run.grad_checkpoint), 0) + 1
    return ", ".join(f"{n}x {flag}" for flag, n in sorted(counts.items()))


def fit_group(
    runs: Sequence[CalibrationRun],
    *,
    safety_mib: float = 0.0,
    source: str = "",
) -> GroupFit:
    if not runs:
        raise CalibrationError("cannot fit an empty group")

    ordered = sorted(runs, key=lambda run: (run.seq_len, run.model_id))


    flags = {run.grad_checkpoint for run in ordered}
    if len(flags) > 1:
        raise CalibrationError(
            f"{_group_label(ordered)}: the rows disagree on gradient checkpointing "
            f"({_flag_counts(ordered)}). A mixed group cannot be fitted as one -- "
            f"split it in the manifest, or exclude the odd rows."
        )


    legacy = [run for run in ordered if not run.has_humps]
    if legacy and len(legacy) != len(ordered):
        names = ", ".join(run.label for run in legacy[:3])
        more = "" if len(legacy) <= 3 else f", +{len(legacy) - 3} more"
        raise CalibrationError(
            f"{_group_label(ordered)}: "
            f"{len(legacy)} of {len(ordered)} rows carry no activation humps "
            f"({names}{more}). The hump form and the legacy proportional form cannot "
            f"be fitted together -- exclude the older rows in the manifest, or fit "
            f"them as their own group."
        )

    if all(run.has_humps for run in ordered):
        if ordered[0].grad_checkpoint is not True:
            raise CalibrationError(
                f"{_group_label(ordered)}: the hump form of C_overhead is only "
                f"defined with gradient checkpointing ON, and these rows report "
                f"grad_checkpoint={ordered[0].grad_checkpoint!r}. Exclude them in "
                f"the manifest, or fit them with a separately justified model."
            )
        hump = _fit_hump(ordered)
        if hump is not None:
            return _hump_group_fit(ordered, hump, safety_mib, source)

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
        quantization=ordered[0].quantization,
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


def _hump_group_fit(
    ordered: Sequence[CalibrationRun],
    hump: tuple[float, float | None],
    safety_mib: float,
    source: str,
) -> GroupFit:
    logits_win_f, layer_win_f = hump

    # The CUDA context is measured on every row, not fitted. Taking the largest keeps
    # the profile on the conservative side if rows from two sessions ever disagree;
    # on the 66 T4 rows behind the shipped profiles they are identical at 140.875.
    context_mib = max(run.context_mib for run in ordered)

    profile = OverheadProfile(
        gpu=ordered[0].gpu_name,
        kernel=ordered[0].kernel,
        quantization=ordered[0].quantization,
        base_context_mib=round(context_mib + safety_mib, 3),
        hump_fragmentation_logits_win=round(logits_win_f, 4),
        hump_fragmentation_layer_win=(
            None if layer_win_f is None else round(layer_win_f, 4)
        ),
        seq_len_min=min(run.seq_len for run in ordered),
        seq_len_max=max(run.seq_len for run in ordered),
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
    provenance = sorted({run.provenance for run in runs if run.provenance})
    trail = f", {'/'.join(provenance)}" if provenance else ""
    return (
        f"{len(runs)} runs, {len(models)} models, seq {lengths[0]}-{lengths[-1]}, "
        f"fitcheck {'/'.join(versions)}{trail}"
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

    groups: dict[tuple[str, str, str], list[CalibrationRun]] = {}
    for run in runs:
        groups.setdefault(run.key, []).append(run)

    fits: dict[tuple[str, str, str], GroupFit] = {}
    skipped: dict[tuple[str, str, str], tuple[CalibrationRun, ...]] = {}
    for key in sorted(groups):
        group = groups[key]
        if len(group) < min_runs:
            skipped[key] = tuple(group)
            continue
        fits[key] = fit_group(group, safety_mib=float(safety_mib))

    return CalibrationResult(fits=fits, skipped=skipped)


def score(
    runs: Iterable[CalibrationRun],
    profiles: dict[tuple[str, str, str], OverheadProfile] | None = None,
) -> dict[tuple[str, str, str], tuple[float, ...]]:
    grouped: dict[tuple[str, str, str], list[CalibrationRun]] = {}
    for run in runs:
        grouped.setdefault(run.key, []).append(run)

    scored: dict[tuple[str, str, str], tuple[float, ...]] = {}
    for key, group in sorted(grouped.items()):
        if profiles is not None:
            profile = profiles.get(key, DEFAULT_OVERHEAD_PROFILE)
        else:
            profile = get_overhead_profile(key[0], key[1] == "flash", key[2])
        scored[key] = _errors_pct(group, profile)
    return scored


# ---------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------


def render_report(result: CalibrationResult) -> str:
    lines: list[str] = []

    for key, fit in sorted(result.fits.items()):
        profile = fit.profile
        lines.append(f"{key[0]} / {key[1]} / quant={key[2]}  ({profile.runs} runs)")
        lines.append(
            f"  base context          {profile.base_context_mib:>10,.3f} MiB"
            f"   (MEASURED, not fitted)"
        )
        if profile.uses_hump_form:
            lines.append(
                f"  F, logits hump wins   "
                f"{profile.hump_fragmentation_logits_win:>10.4f}"
            )
            if profile.hump_fragmentation_layer_win is None:
                lines.append(
                    "  F, layer hump wins            -- no rows; "
                    "falls back to the logits-win coefficient"
                )
            else:
                lines.append(
                    f"  F, layer hump wins    "
                    f"{profile.hump_fragmentation_layer_win:>10.4f}"
                )
            lines.append(
                f"  form                  C_overhead = B + F * min(A_logits, A_layer)"
                f"   (seq {profile.seq_len_min}-{profile.seq_len_max})"
            )
        else:
            lines.append(
                f"  fragmentation @{REFERENCE_SEQ_LEN:<5} {profile.fragmentation:>10.4f}"
            )
            lines.append(
                f"  per octave of seq     {profile.fragmentation_per_octave:>10.4f}"
                f"   (seq {profile.seq_len_min}-{profile.seq_len_max})"
            )
            lines.append(
                "  form                  LEGACY B + F * (W_base + A_act) -- these rows"
                " carry no activation humps"
            )
        lines.append(
            f"  process error         worst {fit.worst_abs_pct:+.1f}%  "
            f"mean {fit.mean_abs_pct:.1f}%  "
            f"(over {fit.worst_over_pct:+.1f}%, under {fit.worst_under_pct:+.1f}%)"
        )
        lines.append("")
        lines.append(
            f"    {'model':<26} {'seq':>5} {'min hump':>9} {'measured':>9} "
            f"{'frag':>7} {'err':>7}"
        )
        for run, error in zip(fit.runs, fit.errors_pct, strict=True):
            size = run.hump_mib if run.has_humps else run.basis_mib
            lines.append(
                f"    {run.model_id.split('/')[-1][:26]:<26} {run.seq_len:>5} "
                f"{size:>9,.0f} {run.process_mib:>9,.0f} "
                f"{100 * run.fragmentation:>6.1f}% {error:>+6.1f}%"
            )
        lines.append("")

    for key, group in sorted(result.skipped.items()):
        lines.append(
            f"{key[0]} / {key[1]} / quant={key[2]}  SKIPPED -- {len(group)} run(s), "
            f"below the minimum. Not fitted, so this pair keeps the default profile."
        )

    if result.fits:
        lines.append(f"worst process-tier error across all fits: {result.worst_abs_pct:.1f}%")

    return "\n".join(lines)


def render_python(result: CalibrationResult) -> str:
    """The `OVERHEAD_DB` literal, ready to paste into `fitcheck/overhead_db.py`."""
    lines = ["OVERHEAD_DB: dict[tuple[str, str, str], OverheadProfile] = {"]
    for key, fit in sorted(result.fits.items()):
        profile = fit.profile
        lines.append(
            f'    ("{key[0]}", "{key[1]}", "{key[2]}"): OverheadProfile('
        )
        lines.append(f'        gpu="{profile.gpu}",')
        lines.append(f'        kernel="{profile.kernel}",')
        lines.append(f'        quantization="{profile.quantization}",')
        lines.append(f"        base_context_mib={profile.base_context_mib},")
        if profile.uses_hump_form:
            lines.append(
                f"        hump_fragmentation_logits_win="
                f"{profile.hump_fragmentation_logits_win},"
            )
            lines.append(
                f"        hump_fragmentation_layer_win="
                f"{profile.hump_fragmentation_layer_win},"
            )
        else:
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


def render_check(scored: dict[tuple[str, str, str], tuple[float, ...]]) -> str:
    lines = [f"{'group':<24} {'runs':>5} {'worst':>8} {'mean':>8}"]
    worst_overall = 0.0
    for key, errors in sorted(scored.items()):
        worst = max((abs(e) for e in errors), default=0.0)
        mean = sum(abs(e) for e in errors) / len(errors) if errors else 0.0
        worst_overall = max(worst_overall, worst)
        lines.append(
            f"{'/'.join(key):<24} {len(errors):>5} {worst:>7.1f}% {mean:>7.1f}%"
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
            "Reproduce the shipped constants:\n"
            "  python -m fitcheck.calibrate data/measurements/manifest.json"
            " --emit-python\n"
            "\n"
            "Typical use:\n"
            "  python -m fitcheck.calibrate data/measurements/manifest.json\n"
            "  python -m fitcheck.calibrate data/measurements/manifest.json --check\n"
            "  python -m fitcheck.calibrate runs/*.json        # ad-hoc, undeclared\n"
        ),
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "a manifest.json (preferred -- only the roles you ask for are read), a "
            "directory holding one, or raw measure.py --json files."
        ),
    )
    parser.add_argument(
        "--role",
        action="append",
        choices=[*ROLES, "all"],
        help=(
            "Manifest roles to read. Repeatable. (default: calibration, so a repeat "
            "or a hold-out cannot enter a fit by accident)"
        ),
    )
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

    roles: tuple[str, ...] = DEFAULT_ROLES
    if args.role:
        roles = ROLES if "all" in args.role else tuple(dict.fromkeys(args.role))

    try:
        runs = load_runs(args.runs, roles)
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
