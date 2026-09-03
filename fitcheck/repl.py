# Interactive REPL (Mode B)
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from shlex import split as shell_split
from typing import Callable, Sequence

import click
from click.core import ParameterSource
from rich.box import SIMPLE_HEAD
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.display import (
    _ASCII_GLYPHS,
    _UNICODE_GLYPHS,
    _Glyphs,
    _config_line,
    _gpu_line,
    _model_line,
    _serving_line,
    make_console,
    render_explanation,
    render_gpu_table,
    render_inference_report,
    render_report,
    render_verbose_detail,
    use_ascii_glyphs,
)
from fitcheck.estimator import (
    InferenceReport,
    MemoryReport,
    ServingConfig,
    TrainingConfig,
    estimate,
    estimate_inference,
)
from fitcheck.gpu_db import GPU_DB, GpuSpec, get_gpu

try:
    import readline
except ImportError:
    pass

_PROMPT = "[bold cyan]fitcheck[/bold cyan] > "
_TIGHT_HEADROOM_FRACTION = 0.20
_TARGET_EFFECTIVE_BATCH = 16
_SAFETY_FRACTION = 0.75

_MEMORY_EXCLUDED = frozenset({"model_id", "list_gpus", "no_color", "version"})
_INFER_EXCLUDED = frozenset({"model_id", "no_color"})
_COMPARE_INFER_FLAGS = frozenset({"--infer", "--serve", "--serving"})

_OFF_SWITCHES: dict[str, str] = {
    "no_flash_attn": "flash_attn",
    "no_grad_checkpoint": "grad_checkpoint",
    "no_double_quant": "double_quant",
}
_ON_SWITCHES = {on: off for off, on in _OFF_SWITCHES.items()}

_MEMORY_HELP = """\
Estimate peak VRAM for the loaded model on the current GPU.

Flags are the CLI's and they persist for the session, so the next `memory` only needs
what changed. Omit --gpu to use the GPU set by `gpu` (it is a one-shot override here,
not a session change). `reset` restores every default.
"""

_INFER_HELP = """\
Estimate serving VRAM for the loaded model: resident weights plus the KV cache.

Flags are `fitcheck infer`'s and they persist for the session, in their own set:
serving compute defaults to fp16 and training to bf16, so `memory` and `infer` never
share a value. Omit --gpu to use the GPU set by `gpu` (a one-shot override here, not a
session change). `reset` restores every default.
"""


class _ReplError(Exception):
    """A message for the user that should not end the session."""


class _ExitRepl(Exception):
    """Raised by `exit` / `quit` / EOF."""


@dataclass(frozen=True)
class _Result:
    """A computed report, with the exact inputs it was computed from."""

    report: MemoryReport
    training: TrainingConfig
    gpu: GpuSpec


@dataclass(frozen=True)
class _InferenceResult:
    """A computed serving report, with the exact inputs it was computed from."""

    report: InferenceReport
    serving: ServingConfig
    gpu: GpuSpec


@dataclass
class _Session:
    console: Console
    glyphs: _Glyphs
    ascii_only: bool
    model: ModelConfig | None = None
    gpu: GpuSpec | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    last: _Result | None = None
    last_inference: _InferenceResult | None = None

    def invalidate(self) -> None:
        """Drop the last reports: they were computed against a model/GPU that just changed."""
        self.last = None
        self.last_inference = None

_SESSION_COMMANDS: dict[str, click.Command] = {}


def _cli_module():
    from fitcheck import cli

    return cli


def _session_command(
    name: str, source: click.Command, excluded: frozenset[str], help_text: str
) -> click.Command:
    command = _SESSION_COMMANDS.get(name)
    if command is not None:
        return command

    params = [param for param in source.params if param.name not in excluded]
    present = {param.name for param in params}
    params.extend(
        click.Option(
            [f"--{off.replace('_', '-')}"],
            is_flag=True,
            help=f"Clear a sticky --{on.replace('_', '-')}.",
        )
        for off, on in _OFF_SWITCHES.items()
        if on in present
    )
    command = click.Command(
        name,
        params=params,
        help=help_text,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    _SESSION_COMMANDS[name] = command
    return command


def _memory_command() -> click.Command:
    return _session_command(
        "memory", _cli_module().estimate_command, _MEMORY_EXCLUDED, _MEMORY_HELP
    )


def _infer_command() -> click.Command:
    return _session_command(
        "infer", _cli_module().infer_command, _INFER_EXCLUDED, _INFER_HELP
    )


def _typed(ctx: click.Context, name: str) -> bool:
    """True when the user actually typed this flag on this line."""
    return ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE


_FIELDS = {
    "quant": "quantization",
    "precision": "precision",
    "optimizer": "optimizer",
    "optimizer_dtype": "optimizer_dtype",
    "batch_size": "batch_size",
    "seq_len": "seq_len",
    "grad_accum": "grad_accum_steps",
}


def _training_from_args(session: _Session, ctx: click.Context) -> TrainingConfig:
    """Fold the flags typed on this line into the session's stored training config."""
    cli = _cli_module()
    current = session.training
    params = ctx.params

    for off, on in _OFF_SWITCHES.items():
        if _typed(ctx, off) and _typed(ctx, on):
            raise _ReplError(
                f"--{on.replace('_', '-')} and --{off.replace('_', '-')} "
                "contradict each other."
            )

    def sticky(name: str) -> object:
        return params[name] if _typed(ctx, name) else getattr(current, _FIELDS[name])

    def flag(name: str) -> bool:
        if _typed(ctx, name):
            return True
        if _typed(ctx, _ON_SWITCHES[name]):
            return False
        return bool(getattr(current, name))

    quant = str(sticky("quant"))
    precision = str(sticky("precision"))
    grad_checkpoint = flag("grad_checkpoint")

    if _typed(ctx, "qlora"):
        if not _typed(ctx, "quant"):
            quant = "nf4"
        if not _typed(ctx, "precision"):
            precision = "bf16"
        if not (_typed(ctx, "grad_checkpoint") or _typed(ctx, "no_grad_checkpoint")):
            grad_checkpoint = True

    if _typed(ctx, "no_lora"):
        lora_rank = None
    elif _typed(ctx, "lora_r"):
        lora_rank = params["lora_r"]
    elif _typed(ctx, "lora_targets") and current.lora_rank is None:
        lora_rank = params["lora_r"]
    else:
        lora_rank = current.lora_rank

    lora_targets = (
        cli._parse_lora_targets(params["lora_targets"])
        if _typed(ctx, "lora_targets")
        else list(current.lora_targets)
    )

    double_quant = flag("double_quant")
    if quant == "none" and not _typed(ctx, "double_quant"):
        double_quant = False

    optimizer = str(sticky("optimizer"))
    cli._validate_combination(ctx, quant, double_quant, lora_rank is None, optimizer)

    return TrainingConfig(
        precision=precision,
        quantization=quant,
        double_quant=double_quant,
        optimizer=optimizer,
        optimizer_dtype=str(sticky("optimizer_dtype")),
        batch_size=int(sticky("batch_size")),
        seq_len=int(sticky("seq_len")),
        lora_rank=lora_rank,
        lora_targets=lora_targets,
        grad_checkpoint=grad_checkpoint,
        flash_attn=flag("flash_attn"),
        grad_accum_steps=int(sticky("grad_accum")),
    )


_SERVING_FIELDS = {
    "quant": "quantization",
    "precision": "precision",
    "seq_len": "seq_len",
    "num_concurrent": "num_concurrent",
}


def _serving_from_args(session: _Session, ctx: click.Context) -> ServingConfig:
    """Fold the flags typed on this line into the session's stored serving config."""
    current = session.serving
    params = ctx.params

    if _typed(ctx, "double_quant") and _typed(ctx, "no_double_quant"):
        raise _ReplError("--double-quant and --no-double-quant contradict each other.")

    def sticky(name: str) -> object:
        field_name = _SERVING_FIELDS[name]
        return params[name] if _typed(ctx, name) else getattr(current, field_name)

    quant = str(sticky("quant"))

    if _typed(ctx, "double_quant"):
        double_quant = True
    elif _typed(ctx, "no_double_quant") or quant == "none":
        double_quant = False
    else:
        double_quant = current.double_quant

    _cli_module()._validate_serving_combination(quant, double_quant)

    return ServingConfig(
        precision=str(sticky("precision")),
        quantization=quant,
        double_quant=double_quant,
        seq_len=int(sticky("seq_len")),
        num_concurrent=int(sticky("num_concurrent")),
    )


def _gpu_from_args(session: _Session, ctx: click.Context) -> GpuSpec:
    """`--gpu` / `--vram-mib` override this one estimate; `gpu <name>` sets the session."""
    if _typed(ctx, "gpu") or _typed(ctx, "vram_mib"):
        return _lookup_gpu(
            ctx.params["gpu"] if _typed(ctx, "gpu") else None, ctx.params["vram_mib"]
        )
    return _require_gpu(session)


def _lookup_gpu(name: str | None, vram_mib: int | None = None) -> GpuSpec:
    try:
        return get_gpu(name, vram_mib)
    except ValueError as error:
        if name and vram_mib is None:
            raise _ReplError(
                f"Unknown GPU '{name}'. Run `gpus` for the database, or give the card's "
                f"size: `gpu {name} --vram-mib 24000`."
            ) from error
        raise _ReplError(str(error)) from error


def _require_model(session: _Session) -> ModelConfig:
    if session.model is None:
        if session.gpu is None:
            raise _ReplError(
                "Run `model <id>` and `gpu <name>` first, e.g. "
                "`model meta-llama/Llama-3.1-8B` then `gpu 4090`."
            )
        raise _ReplError(
            "No model loaded. Run `model <id>` first, e.g. "
            "`model meta-llama/Llama-3.1-8B`."
        )
    return session.model


def _require_gpu(session: _Session) -> GpuSpec:
    if session.gpu is None:
        raise _ReplError("No GPU set. Run `gpu <name>` first. `gpus` lists them.")
    return session.gpu


def _current_result(session: _Session) -> _Result:
    """The last report, computing one from the session's state if none exists yet."""
    model = _require_model(session)
    gpu = _require_gpu(session)
    if session.last is None:
        session.last = _Result(estimate(model, session.training, gpu), session.training, gpu)
    return session.last


def _run_inference(
    model: ModelConfig, serving: ServingConfig, gpu: GpuSpec
) -> InferenceReport:
    """Component 7 + 6 for one card, with a bad combination reported, not raised."""
    try:
        return estimate_inference(model, serving, gpu)
    except ValueError as error:
        raise _ReplError(str(error)) from error


def _current_inference(session: _Session) -> _InferenceResult:
    """The last serving report, computing one from the session's state if none exists."""
    model = _require_model(session)
    gpu = _require_gpu(session)
    if session.last_inference is None:
        session.last_inference = _InferenceResult(
            _run_inference(model, session.serving, gpu), session.serving, gpu
        )
    return session.last_inference


def _mib(value: float) -> str:
    return f"{value:,.0f}"


def _percent(part: float, whole: float) -> str:
    return f"{(part / whole * 100) if whole > 0 else 0.0:.0f}%"


def _leader_grid(rows: Sequence[tuple[str, RenderableType]]) -> Table:
    """Dotted-leader rows, matching the `explain` panel's visual language."""
    width = max(len(label) for label, _ in rows) + 2

    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim")
    grid.add_column()
    for label, value in rows:
        grid.add_row(Text(f"{label} {'.' * (width - len(label))}"), value)
    return grid


def _verdict_text(report: MemoryReport | InferenceReport, glyphs: _Glyphs) -> Text:
    headroom = report.headroom_mib / report.gpu_capacity_mib if report.gpu_capacity_mib else 0.0
    if not report.fits:
        return Text(
            f"{glyphs.oom} over by {_mib(-report.headroom_mib)} MiB", style="red"
        )
    if headroom <= _TIGHT_HEADROOM_FRACTION:
        return Text(f"{glyphs.tight} fits, barely", style="yellow")
    return Text(f"{glyphs.fits} fits", style="green")


def _ok(session: _Session, message: str) -> None:
    check = "OK" if session.ascii_only else "✓"
    session.console.print(Text.assemble((f"{check} ", "green"), message))

def _cmd_model(session: _Session, args: list[str]) -> None:
    if len(args) != 1:
        raise _ReplError("Usage: model <huggingface_id>, e.g. `model mistralai/Mistral-7B-v0.3`.")

    model_id = args[0]
    try:
        with session.console.status(f"Fetching config.json for {model_id} ..."):
            config = fetch_model_config(model_id)
    except (RuntimeError, ValueError, OSError) as error:
        raise _ReplError(f"Could not read config.json for '{model_id}': {error}") from error

    session.model = config
    session.invalidate()
    _ok(session, f"Loaded {_model_line(config, session.glyphs)}")


def _cmd_gpu(session: _Session, args: list[str]) -> None:
    rest = list(args)
    vram_mib: int | None = None

    if "--vram-mib" in rest:
        index = rest.index("--vram-mib")
        try:
            vram_mib = int(rest[index + 1])
        except (IndexError, ValueError) as error:
            raise _ReplError(
                "--vram-mib needs a positive integer, e.g. `gpu mycard --vram-mib 24000`."
            ) from error
        del rest[index : index + 2]

    if len(rest) > 1 or (not rest and vram_mib is None):
        raise _ReplError("Usage: gpu <name> [--vram-mib N]. `gpus` lists the database.")

    session.gpu = _lookup_gpu(rest[0] if rest else None, vram_mib)
    session.invalidate()
    _ok(session, f"Target GPU set to {_gpu_line(session.gpu, session.glyphs)}")


def _cmd_memory(session: _Session, args: list[str]) -> None:
    ctx = _memory_command().make_context("memory", list(args))

    model = _require_model(session)
    training = _training_from_args(session, ctx)
    gpu = _gpu_from_args(session, ctx)

    try:
        report = estimate(model, training, gpu)
    except ValueError as error:
        raise _ReplError(str(error)) from error

    session.training = training
    session.last = _Result(report, training, gpu)

    if ctx.params["as_json"]:
        session.console.print_json(
            data=_cli_module().report_to_dict(report, model, gpu, training)
        )
        return

    session.console.print(
        render_report(report, model, gpu, training, ascii_only=session.ascii_only)
    )
    if ctx.params["verbose"]:
        session.console.print(
            render_verbose_detail(model, training, ascii_only=session.ascii_only)
        )
    if ctx.params["explain"]:
        session.console.print(
            render_explanation(report, model, training, ascii_only=session.ascii_only)
        )


def _cmd_infer(session: _Session, args: list[str]) -> None:
    ctx = _infer_command().make_context("infer", list(args))

    model = _require_model(session)
    serving = _serving_from_args(session, ctx)
    gpu = _gpu_from_args(session, ctx)

    report = _run_inference(model, serving, gpu)

    session.serving = serving
    session.last_inference = _InferenceResult(report, serving, gpu)

    if ctx.params["as_json"]:
        session.console.print_json(
            data=_cli_module().inference_report_to_dict(report, model, gpu, serving)
        )
        return

    session.console.print(
        render_inference_report(
            report, model, gpu, serving, ascii_only=session.ascii_only
        )
    )


def _cmd_explain(session: _Session, args: list[str]) -> None:
    if args:
        raise _ReplError(
            "explain takes no arguments. To change the config first, use "
            "`memory <flags> --explain`."
        )

    result = _current_result(session)
    session.console.print(
        render_explanation(
            result.report, _require_model(session), result.training,
            ascii_only=session.ascii_only,
        )
    )


def _cmd_optimize(session: _Session, args: list[str]) -> None:
    if args:
        raise _ReplError("optimize takes no arguments.")
    session.console.print(_render_optimize(session, _current_result(session)))


def _cmd_compare(session: _Session, args: list[str]) -> None:
    serving_mode = any(token.casefold() in _COMPARE_INFER_FLAGS for token in args)
    names = [
        token
        for token in args
        if token != "--gpu" and token.casefold() not in _COMPARE_INFER_FLAGS
    ]
    if not names:
        raise _ReplError(
            "Usage: compare <gpu> [<gpu> ...] [--infer], e.g. `compare 3090 t4 a100-40`."
        )

    if serving_mode:
        serving_result = _current_inference(session)
        session.console.print(
            _render_infer_compare(
                session, serving_result, _compare_specs(serving_result.gpu, names)
            )
        )
        return

    result = _current_result(session)
    session.console.print(
        _render_compare(session, result, _compare_specs(result.gpu, names))
    )


def _compare_specs(current: GpuSpec, names: Sequence[str]) -> list[GpuSpec]:
    """The card the report was computed on, then each named card, once each."""
    specs = [current]
    for name in names:
        spec = _lookup_gpu(name)
        if spec not in specs:
            specs.append(spec)
    return specs


def _cmd_show(session: _Session, args: list[str]) -> None:
    if args:
        raise _ReplError("show takes no arguments.")

    rows: list[tuple[str, RenderableType]] = []
    rows.append(
        (
            "Model",
            Text(_model_line(session.model, session.glyphs))
            if session.model is not None
            else Text("not loaded. Run `model <id>`", style="dim"),
        )
    )
    rows.append(
        (
            "GPU",
            Text(_gpu_line(session.gpu, session.glyphs))
            if session.gpu is not None
            else Text("not set. Run `gpu <name>`", style="dim"),
        )
    )
    rows.append(("Config", Text(_config_line(session.training, session.glyphs))))
    rows.append(("Serving", Text(_serving_line(session.serving, session.glyphs))))

    for label, result in (
        ("Last memory", session.last),
        ("Last infer", session.last_inference),
    ):
        if result is None:
            continue
        rows.append(
            (
                label,
                Text.assemble(
                    f"{_mib(result.report.total_mib)} MiB peak on {result.gpu.name}  ",
                    _verdict_text(result.report, session.glyphs),
                ),
            )
        )

    session.console.print(
        Panel(
            _leader_grid(rows),
            title="session",
            title_align="left",
            border_style="dim",
            padding=(1, 2),
        )
    )


def _cmd_gpus(session: _Session, args: list[str]) -> None:
    if args:
        raise _ReplError("gpus takes no arguments.")
    session.console.print(render_gpu_table())


def _cmd_reset(session: _Session, args: list[str]) -> None:
    if args:
        raise _ReplError("reset takes no arguments.")
    session.training = TrainingConfig()
    session.serving = ServingConfig()
    session.invalidate()
    _ok(
        session,
        "Training and serving flags back to defaults. Model and GPU are unchanged.",
    )


def _cmd_help(session: _Session, args: list[str]) -> None:
    session.console.print(_render_help(session))


def _cmd_exit(session: _Session, args: list[str]) -> None:
    raise _ExitRepl


_COMMANDS: dict[str, Callable[[_Session, list[str]], None]] = {
    "model": _cmd_model,
    "gpu": _cmd_gpu,
    "memory": _cmd_memory,
    "infer": _cmd_infer,
    "explain": _cmd_explain,
    "optimize": _cmd_optimize,
    "compare": _cmd_compare,
    "show": _cmd_show,
    "gpus": _cmd_gpus,
    "reset": _cmd_reset,
    "help": _cmd_help,
    "exit": _cmd_exit,
}

_ALIASES = {
    "quit": "exit",
    "q": "exit",
    "?": "help",
    "h": "help",
    "mem": "memory",
    "serve": "infer",
    "inference": "infer",
    "kv": "infer",
    "config": "show",
    "state": "show",
    "list-gpus": "gpus",
    "explain-last": "explain",
}


def _recommended_batch(max_batch_size: int) -> int:
    """Largest power of two inside a safety margin below the ceiling.

    Powers of two because that is what people actually set, and a margin because the
    last batch that fits leaves nothing for a long sample or an allocator spike.
    """
    target = max(1, int(max_batch_size * _SAFETY_FRACTION))
    batch = 1
    while batch * 2 <= target:
        batch *= 2
    return batch


def _recommended_accum(batch_size: int, training: TrainingConfig) -> int:
    current_effective = training.batch_size * training.grad_accum_steps
    target = max(_TARGET_EFFECTIVE_BATCH, current_effective)
    return max(1, -(-target // batch_size))


def _render_optimize(session: _Session, result: _Result) -> Panel:
    model = _require_model(session)
    report, training, gpu = result.report, result.training, result.gpu

    if report.max_batch_size == 0:
        body: RenderableType = _render_rescue(session, model, training, gpu)
    else:
        batch_size = _recommended_batch(report.max_batch_size)
        accum = _recommended_accum(batch_size, training)
        recommended = replace(training, batch_size=batch_size, grad_accum_steps=accum)

        at_max = estimate(model, replace(training, batch_size=report.max_batch_size), gpu)
        at_recommended = estimate(model, recommended, gpu)

        rows: list[tuple[str, RenderableType]] = [
            (
                f"Max micro-batch at seq {training.seq_len:,}",
                Text.assemble(
                    (f"{report.max_batch_size}", "bold"),
                    (
                        f"   {_mib(at_max.headroom_mib)} MiB "
                        f"({_percent(at_max.headroom_mib, at_max.gpu_capacity_mib)}) left",
                        "dim",
                    ),
                ),
            ),
            (
                "Recommended micro-batch",
                Text.assemble(
                    (f"{batch_size}", "bold green"),
                    (
                        f"   {_mib(at_recommended.headroom_mib)} MiB "
                        f"({_percent(at_recommended.headroom_mib, at_recommended.gpu_capacity_mib)})"
                        " left",
                        "dim",
                    ),
                ),
            ),
            (
                "Effective batch",
                Text.assemble(
                    f"{batch_size * accum}",
                    (
                        f"   {batch_size} x {accum} accumulation steps, which cost 0 MiB",
                        "dim",
                    ),
                ),
            ),
            (
                "Predicted peak",
                Text(
                    f"{_mib(at_recommended.total_mib)} MiB of "
                    f"{_mib(at_recommended.gpu_capacity_mib)} MiB"
                ),
            ),
            (
                "Command",
                Text(
                    f"memory --batch-size {batch_size} --grad-accum {accum}",
                    style="cyan",
                ),
            ),
        ]

        note = Text(
            f"The ceiling is {report.max_batch_size}, not the recommendation: it is the "
            "largest batch the estimate allows, so a long sample or allocator "
            "fragmentation has nowhere to go. Accumulation buys back the effective "
            "batch for free.",
            style="dim",
        )
        body = Group(_leader_grid(rows), Text(""), note)

    return Panel(
        body,
        title=f"optimize {session.glyphs.arrow} {gpu.name}",
        title_align="left",
        border_style="green" if report.max_batch_size else "red",
        padding=(1, 2),
    )

_LEVERS: tuple[
    tuple[
        Callable[[TrainingConfig], bool],
        Callable[[TrainingConfig], TrainingConfig],
        Callable[[TrainingConfig], str],
    ],
    ...,
] = (
    (
        lambda t: not t.flash_attn,
        lambda t: replace(t, flash_attn=True),
        lambda t: "--flash-attn",
    ),
    (
        lambda t: not t.grad_checkpoint,
        lambda t: replace(t, grad_checkpoint=True),
        lambda t: "--grad-checkpoint",
    ),
    (
        lambda t: t.quantization == "none" and t.lora_rank is not None,
        lambda t: replace(t, quantization="nf4", double_quant=True),
        lambda t: "--quant nf4 --double-quant",
    ),
    (
        lambda t: t.seq_len >= 512,
        lambda t: replace(t, seq_len=t.seq_len // 2),
        lambda t: f"--seq-len {t.seq_len}",
    ),
    (
        lambda t: t.optimizer.strip().casefold() != "adam8bit",
        lambda t: replace(t, optimizer="adam8bit"),
        lambda t: "--optimizer adam8bit",
    ),
)


def _render_rescue(
    session: _Session, model: ModelConfig, training: TrainingConfig, gpu: GpuSpec
) -> RenderableType:
    """Nothing fits even at batch_size 1 — apply levers until something does."""
    current = replace(training, batch_size=1)
    latest = estimate(model, current, gpu)

    headline = Text(
        f"Nothing fits at batch_size 1: {_mib(latest.total_mib)} MiB needed, "
        f"{_mib(latest.gpu_capacity_mib)} MiB usable on {gpu.name}.",
        style="bold",
    )

    applied: list[str] = []
    steps: list[tuple[str, MemoryReport]] = []

    for applies, apply, label in _LEVERS:
        if not applies(current):
            continue
        current = apply(current)
        applied.append(label(current))
        latest = estimate(model, current, gpu)
        steps.append((f"+ {applied[-1]}", latest))
        if latest.fits:
            break

    width = max((len(_mib(report.total_mib)) for _, report in steps), default=0)
    rows: list[tuple[str, RenderableType]] = [
        (
            label,
            Text.assemble(
                f"{_mib(report.total_mib):>{width}} MiB   ",
                _verdict_text(report, session.glyphs),
            ),
        )
        for label, report in steps
    ]

    if not rows:
        return Group(headline, Text(""), _bigger_gpu_note(latest.total_mib))

    if not latest.fits:
        return Group(
            headline, Text(""), _leader_grid(rows), Text(""),
            _bigger_gpu_note(latest.total_mib),
        )

    return Group(
        headline,
        Text(""),
        _leader_grid(rows),
        Text(""),
        Text.assemble(
            ("Try: ", "dim"),
            (f"memory --batch-size 1 {' '.join(applied)}", "cyan"),
        ),
    )


def _bigger_gpu_note(total_mib: float) -> Text:
    candidates = sorted(
        {spec for spec in GPU_DB.values() if spec.usable_mib >= total_mib},
        key=lambda spec: spec.usable_mib,
    )
    if not candidates:
        return Text(
            f"No single card in the database holds {_mib(total_mib)} MiB. This "
            "configuration needs multi-GPU sharding, which fitcheck does not model.",
            style="dim",
        )
    spec = candidates[0]
    return Text(
        f"Smallest card in the database that would hold this configuration "
        f"({_mib(total_mib)} MiB): {spec.name}, {_mib(spec.usable_mib)} MiB usable.",
        style="dim",
    )


def _compare_panel(
    session: _Session,
    title: str,
    header_line: str,
    ceiling_header: str,
    rows: Sequence[tuple[GpuSpec, MemoryReport | InferenceReport, str]],
    peak_mib: float,
) -> Panel:
    """One table for both modes: the peak never moves, only the ceiling does."""
    table = Table(box=SIMPLE_HEAD, pad_edge=False, expand=True)
    table.add_column("GPU")
    table.add_column("Usable (MiB)", justify="right")
    table.add_column("Headroom (MiB)", justify="right")
    table.add_column("Used", justify="right")
    table.add_column(ceiling_header, justify="right")
    table.add_column("Verdict")

    for spec, report, ceiling in rows:
        table.add_row(
            spec.name,
            _mib(spec.usable_mib),
            Text(
                _mib(report.headroom_mib),
                style="green" if report.fits else "red",
            ),
            _percent(report.total_mib, report.gpu_capacity_mib),
            ceiling,
            _verdict_text(report, session.glyphs),
        )

    dash = "-" if session.ascii_only else "—"
    footer = Text.assemble(
        ("Peak is identical on every card: ", "dim"),
        (f"{_mib(peak_mib)} MiB", "bold"),
        (f" {dash} only the ceiling moves.", "dim"),
    )

    return Panel(
        Group(Text(header_line), Text(""), table, footer),
        title=title,
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    )


def _render_compare(
    session: _Session, result: _Result, specs: Sequence[GpuSpec]
) -> Panel:
    model = _require_model(session)
    training = result.training

    rows: list[tuple[GpuSpec, MemoryReport | InferenceReport, str]] = []
    for spec in specs:
        report = estimate(model, training, spec)
        rows.append((spec, report, str(report.max_batch_size)))

    return _compare_panel(
        session,
        "compare",
        _config_line(training, session.glyphs),
        f"Max bs @ {training.seq_len:,}",
        rows,
        result.report.total_mib,
    )


def _render_infer_compare(
    session: _Session, result: _InferenceResult, specs: Sequence[GpuSpec]
) -> Panel:
    model = _require_model(session)
    serving = result.serving

    rows: list[tuple[GpuSpec, MemoryReport | InferenceReport, str]] = []
    for spec in specs:
        report = _run_inference(model, serving, spec)
        rows.append((spec, report, str(report.max_concurrent)))

    return _compare_panel(
        session,
        "compare infer",
        _serving_line(serving, session.glyphs),
        f"Max concurrent @ {serving.seq_len:,}",
        rows,
        result.report.total_mib,
    )

_HELP_ROWS: tuple[tuple[str, str], ...] = (
    ("model <id>", "Fetch a model's config.json from HuggingFace (~2 KB, no weights)"),
    ("gpu <name> [--vram-mib N]", "Set the target GPU"),
    ("memory [flags]", "Estimate peak training VRAM. Same flags as the CLI; they stick"),
    ("infer [flags]", "Estimate serving VRAM: weights + KV cache. Flags stick too"),
    ("explain", "Name the largest component, price every toggle"),
    ("optimize", "Largest micro-batch that fits, plus a config worth running"),
    ("compare <gpu> ... [--infer]", "The same config across other cards"),
    ("show", "Current model, GPU, flags, and the last estimates"),
    ("reset", "Training and serving flags back to defaults"),
    ("gpus", "Print the GPU database"),
    ("help", "This table"),
    ("exit / quit", "Leave"),
)


def _render_help(session: _Session) -> Panel:
    table = Table(box=SIMPLE_HEAD, pad_edge=False, expand=True)
    table.add_column("Command", style="cyan")
    table.add_column("What it does")
    for command, description in _HELP_ROWS:
        table.add_row(Text(command), Text(description))

    notes = Text.assemble(
        ("Flags persist. ", "bold"),
        "`memory --qlora --lora-r 64 --batch-size 4 --seq-len 2048 --flash-attn`, then "
        "just `memory --batch-size 8` to move one dial. Turn a sticky flag off with "
        "--no-flash-attn / --no-grad-checkpoint / --no-double-quant, or `reset` for all "
        "of them. Every report header echoes the flags in force.\n"
        "`memory --help` lists all of them; `memory --verbose` adds the per-layer "
        "breakdown and `memory --json` prints the machine-readable payload.\n"
        "`infer` has its own sticky set, because serving computes in fp16 where training "
        "defaults to bf16: `infer --quant nf4 --seq-len 8192 --concurrent 8`, then "
        "`compare a100-40 h100 --infer` for the requests each card holds.",
        style="dim",
    )

    return Panel(
        Group(table, Text(""), notes),
        title="commands",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    )


def _render_banner(session: _Session) -> Panel:
    quickstart = Table.grid(padding=(0, 2))
    quickstart.add_column(style="dim", justify="right")
    quickstart.add_column(style="cyan")
    for step, command in enumerate(
        (
            "model meta-llama/Llama-3.1-8B",
            "gpu 4090",
            "memory --qlora --lora-r 64 --batch-size 4 --flash-attn",
        ),
        start=1,
    ):
        quickstart.add_row(str(step), Text(command))

    return Panel(
        Group(
            Text.assemble(
                ("fitcheck interactive. ", "bold"),
                (
                    "No GPU is touched: every figure is math from config.json.",
                    "dim",
                ),
            ),
            Text(""),
            quickstart,
            Text(""),
            Text("`help` lists every command, `exit` leaves.", style="dim"),
        ),
        title="fitcheck",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    )

def _resolve_command(token: str) -> Callable[[_Session, list[str]], None]:
    """Look the command up case-insensitively, but quote the user's spelling back."""
    canonical = _ALIASES.get(token.casefold(), token.casefold())
    handler = _COMMANDS.get(canonical)
    if handler is not None:
        return handler

    if "/" in token:
        raise _ReplError(f"Unknown command '{token}'. Did you mean `model {token}`?")

    suggestions = get_close_matches(canonical, list(_COMMANDS), n=1)
    hint = f" Did you mean `{suggestions[0]}`?" if suggestions else ""
    raise _ReplError(f"Unknown command '{token}'.{hint} Type `help` for the list.")


def _dispatch(session: _Session, line: str) -> None:
    try:
        tokens = shell_split(line)
    except ValueError as error:
        raise _ReplError(f"Could not parse that line: {error}") from error

    if not tokens:
        return
    _resolve_command(tokens[0])(session, tokens[1:])


def _print_error(session: _Session, message: str) -> None:
    session.console.print(Text.assemble(("error  ", "bold red"), message))


def _read_line(session: _Session) -> str:
    line = session.console.input(_PROMPT)
    if not sys.stdin.isatty():
        session.console.print(line)
    return line

def run_repl(
    console: Console | None = None,
    training: TrainingConfig | None = None,
    gpu: GpuSpec | None = None,
) -> int:
    """Run Mode B until the user leaves. `training` / `gpu` seed the session.

    `fitcheck` with flags but no MODEL_ID lands here with those flags already in
    force, which is why the banner echoes them: seeded state that nothing shows is
    state the user will forget by the third command.
    """
    target = console if console is not None else make_console()
    ascii_only = use_ascii_glyphs(target)
    session = _Session(
        console=target,
        glyphs=_ASCII_GLYPHS if ascii_only else _UNICODE_GLYPHS,
        ascii_only=ascii_only,
        gpu=gpu,
        training=training if training is not None else TrainingConfig(),
    )

    session.console.print(_render_banner(session))
    if gpu is not None:
        _ok(session, f"Target GPU set to {_gpu_line(gpu, session.glyphs)}")
    if session.training != TrainingConfig():
        _ok(
            session,
            f"Flags carried in from the command line: "
            f"{_config_line(session.training, session.glyphs)}",
        )

    while True:
        try:
            line = _read_line(session)
        except EOFError:
            session.console.print()
            break
        except KeyboardInterrupt:
            session.console.print()
            continue

        try:
            _dispatch(session, line)
        except _ExitRepl:
            break
        except _ReplError as error:
            _print_error(session, str(error))
        except click.ClickException as error:
            _print_error(session, error.format_message())
        except click.exceptions.Exit:
            pass 
        except click.Abort:
            _print_error(session, "aborted")
        except KeyboardInterrupt:
            session.console.print()
            _print_error(session, "interrupted")
        except Exception as error: 
            _print_error(session, f"{type(error).__name__}: {error}")

    session.console.print("Goodbye!")
    return 0
