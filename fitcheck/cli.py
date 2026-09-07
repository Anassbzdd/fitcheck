# click commands & option groups
from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import click
from click.core import ParameterSource
from rich.console import Console

from fitcheck.advisor import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_LORA_RANKS,
    AdvisorReport,
    FrontierPoint,
    SweepSpec,
    advise,
)
from fitcheck.config_parser import ModelConfig, fetch_model_config
from fitcheck.display import (
    make_console,
    render_advisor_report,
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
    activation_breakdown,
    estimate,
    estimate_inference,
    trainable_params,
)
from fitcheck.gpu_db import GpuSpec, get_gpu
from fitcheck.memory.lora import (
    LORA_TARGETS_FULL,
    LORA_TARGETS_MINIMAL,
    LORA_TARGETS_STANDARD,
)
from fitcheck.repl import run_repl

_DEFAULT_GPU = "4090"
_EXIT_DOES_NOT_FIT = 1
_EXIT_ERROR = 2
_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "minimal": LORA_TARGETS_MINIMAL,
    "standard": LORA_TARGETS_STANDARD,
    "full": LORA_TARGETS_FULL,
}
_HELP_EPILOG = """\
Exit codes: 0 the config fits, 1 it does not fit, 2 the estimate could not be run
(bad flags, unknown model or GPU). A prediction that does not fit is a verdict, not
an error, so `fitcheck ... && accelerate launch ...` guards a training run. The REPL
always exits 0.

Serving instead of training? `fitcheck infer MODEL [OPTIONS]` prices resident weights
plus the KV cache. Run `fitcheck infer --help` for its flags.

Not sure what to run? `fitcheck advise MODEL --seq-lens 1024,2048` sweeps batch size
and LoRA rank instead of pricing one config, and prints what each axis costs and
where the wall is. Run `fitcheck advise --help` for its flags.

Every figure is analytical, computed from config.json alone. No GPU is touched and
no weights are downloaded.
"""

_ADVISE_EPILOG = """\
Exit codes: 0 at least one swept config fits, 1 none of them fits, 2 the sweep could
not be run. `advise` says what each axis costs and where the wall is; the REPL's
`optimize` says what to run with the flags you are already holding.

`--seq-lens` is required on purpose. Nothing in config.json tells fitcheck a model's
real context length, so any default would sweep past it in silence. Pass
`--max-seq-len` as well and every swept length above it is rejected instead of priced.

tokens/step is `batch_size x seq_len` -- the work in one optimizer step, NOT a speed.
fitcheck has no throughput model and will never draw one.
"""

_INFER_EPILOG = """\
Exit codes are the same as the training command: 0 means it fits, 1 means it
does not fit, and 2 means the estimate could not be run. Use `fitcheck --list-gpus`
to see the supported GPUs.

Inference does not need optimizer states, gradients, or training activations.
It mainly needs the model weights and the KV cache. Each request is assumed to
use the full `--seq-len`, so the estimate is a safe upper bound for an engine
that reserves the KV cache in advance. For paged KV-cache engines such as vLLM,
the actual memory usage can be lower.
"""


class _EstimateError(click.ClickException):
    exit_code = _EXIT_ERROR


def _package_version() -> str:
    try:
        return version("fitcheck-llm")
    except PackageNotFoundError:
        return "unknown"


def _explicit(ctx: click.Context, name: str) -> bool:
    return ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE


def _parse_lora_targets(value: str) -> list[str]:
    preset = _TARGET_PRESETS.get(value.strip().casefold())
    if preset is not None:
        return list(preset)

    targets: list[str] = []
    for token in value.split(","):
        name = token.strip().casefold()
        if not name:
            continue
        module = name if name.endswith("_proj") else f"{name}_proj"
        if module not in LORA_TARGETS_FULL:
            supported = ", ".join(LORA_TARGETS_FULL)
            raise click.BadParameter(
                f"unknown LoRA target '{token.strip()}'. Presets: "
                f"{', '.join(_TARGET_PRESETS)}. Modules: {supported}.",
                param_hint="--lora-targets",
            )
        if module in targets:
            raise click.BadParameter(
                f"duplicate LoRA target '{module}'", param_hint="--lora-targets"
            )
        targets.append(module)

    if not targets:
        raise click.BadParameter(
            "at least one target module is required", param_hint="--lora-targets"
        )
    return targets


def _parse_int_list(value: str, param_hint: str) -> list[int]:
    values: list[int] = []
    for token in value.split(","):
        text = token.strip()
        if not text:
            continue
        try:
            number = int(text)
        except ValueError:
            raise click.BadParameter(
                f"'{text}' is not a whole number", param_hint=param_hint
            ) from None
        if number < 1:
            raise click.BadParameter(
                f"{number} is not a positive value", param_hint=param_hint
            )
        if number not in values:
            values.append(number)

    if not values:
        raise click.BadParameter(
            "at least one value is required", param_hint=param_hint
        )
    return values


def _validate_serving_combination(quant: str, double_quant: bool) -> None:
    """The one check the training and serving surfaces share, so they cannot drift."""
    if double_quant and quant == "none":
        raise click.UsageError(
            "--double-quant has nothing to quantize under --quant none. It halves the "
            "NF4/INT8 scale overhead, so pair it with --quant nf4."
        )


def _validate_combination(
    ctx: click.Context,
    quant: str,
    double_quant: bool,
    no_lora: bool,
    optimizer: str,
) -> None:
    if no_lora and quant != "none":
        raise click.UsageError(
            f"--quant {quant} with --no-lora is not modelled by fitcheck. Its "
            "quantized path assumes the base model stays frozen while only adapters "
            "train, which is what the weight and optimizer formulas are derived for. "
            "Quantized base models can be trained (QAT, straight-through estimators); "
            "fitcheck simply does not model those memory profiles. Drop --no-lora, or "
            "use --quant none."
        )

    if no_lora:
        for option, flag in (("lora_r", "--lora-r"), ("lora_targets", "--lora-targets")):
            if _explicit(ctx, option):
                raise click.UsageError(
                    f"--no-lora conflicts with {flag}: full fine-tuning trains every "
                    "parameter, so there is no adapter to configure."
                )

    _validate_serving_combination(quant, double_quant)

    if _explicit(ctx, "optimizer_dtype") and optimizer != "adamw":
        raise click.UsageError(
            f"--optimizer-dtype applies only to --optimizer adamw. {optimizer} has a "
            "fixed state dtype."
        )


def _resolve_gpu(ctx: click.Context, gpu: str | None, vram_mib: int | None) -> GpuSpec:
    try:
        if vram_mib is not None:
            return get_gpu(gpu if _explicit(ctx, "gpu") else None, vram_mib)
        return get_gpu(gpu or _DEFAULT_GPU)
    except ValueError as error:
        raise _EstimateError(str(error)) from error


def _enter_repl(
    ctx: click.Context,
    console: Console,
    training: TrainingConfig,
    gpu: str | None,
    vram_mib: int | None,
) -> int:
    for option, flag in (
        ("as_json", "--json"),
        ("verbose", "--verbose"),
        ("explain", "--explain"),
    ):
        if _explicit(ctx, option):
            raise click.UsageError(
                f"{flag} formats one estimate, and without a MODEL_ID there is "
                f"nothing to format. Name a model, or enter the REPL and run "
                f"`memory {flag}` there."
            )

    seeded_gpu = (
        _resolve_gpu(ctx, gpu, vram_mib)
        if _explicit(ctx, "gpu") or _explicit(ctx, "vram_mib")
        else None
    )
    return run_repl(console, training=training, gpu=seeded_gpu)


def _load_model_config(model_id: str) -> ModelConfig:
    try:
        return fetch_model_config(model_id)
    except (RuntimeError, ValueError, OSError) as error:
        raise _EstimateError(
            f"Could not read config.json for '{model_id}': {error}"
        ) from error


def report_to_dict(
    report: MemoryReport,
    config: ModelConfig,
    gpu: GpuSpec,
    training: TrainingConfig,
) -> dict[str, Any]:
    parts = activation_breakdown(config, training)

    def mib(value: float) -> float:
        return round(value, 2)

    return {
        "fitcheck_version": _package_version(),
        "model": asdict(config),
        "gpu": asdict(gpu),
        "training": asdict(training),
        "trainable_params": trainable_params(config, training),
        "memory_mib": {
            "weights": mib(report.weight_mib),
            "lora": mib(report.lora_mib),
            "optimizer": mib(report.optimizer_mib),
            "gradients": mib(report.gradient_mib),
            "activations": mib(report.activation_mib),
            "overhead": mib(report.overhead_mib),
            "total": mib(report.total_mib),
        },
        "activations_per_layer_mib": mib(parts["layer_mib"]),
        "verdict": {
            "fits": report.fits,
            "gpu_capacity_mib": mib(report.gpu_capacity_mib),
            "headroom_mib": mib(report.headroom_mib),
            "headroom_pct": mib(
                100.0 * report.headroom_mib / report.gpu_capacity_mib
                if report.gpu_capacity_mib
                else 0.0
            ),
            "max_batch_size": report.max_batch_size,
            "effective_batch_size": report.effective_batch_size,
        },
        "savings_hints": list(report.savings_hints),
    }


def inference_report_to_dict(
    report: InferenceReport,
    config: ModelConfig,
    gpu: GpuSpec,
    serving: ServingConfig,
) -> dict[str, Any]:
    def mib(value: float) -> float:
        return round(value, 2)

    return {
        "fitcheck_version": _package_version(),
        "model": asdict(config),
        "gpu": asdict(gpu),
        "serving": asdict(serving),
        "memory_mib": {
            "weights": mib(report.weight_mib),
            "kv_cache": mib(report.kv_cache_mib),
            "overhead": mib(report.overhead_mib),
            "total": mib(report.total_mib),
        },
        "kv_cache_mib_per_request": mib(report.kv_mib_per_request),
        "kv_cache_mib_per_token": round(report.kv_mib_per_token, 6),
        "verdict": {
            "fits": report.fits,
            "gpu_capacity_mib": mib(report.gpu_capacity_mib),
            "headroom_mib": mib(report.headroom_mib),
            "headroom_pct": mib(
                100.0 * report.headroom_mib / report.gpu_capacity_mib
                if report.gpu_capacity_mib
                else 0.0
            ),
            "max_concurrent": report.max_concurrent,
        },
    }


def _frontier_point_to_dict(point: FrontierPoint) -> dict[str, Any]:
    return {
        "batch_size": point.batch_size,
        "seq_len": point.seq_len,
        "lora_rank": point.lora_rank,
        "tokens_per_step": point.tokens_per_step,
        "total_mib": round(point.total_mib, 2),
        "fits": point.fits,
        "command": point.command,
        "equivalent_splits": [
            [batch_size, seq_len] for batch_size, seq_len in point.equivalent_splits
        ],
    }


def advisor_report_to_dict(
    report: AdvisorReport, config: ModelConfig
) -> dict[str, Any]:
    def mib(value: float) -> float:
        return round(value, 2)

    return {
        "fitcheck_version": _package_version(),
        "model": asdict(config),
        "gpu": asdict(report.gpu),
        "sweep": {
            "batch_sizes": list(report.sweep.batch_sizes),
            "seq_lens": list(report.sweep.seq_lens),
            "lora_ranks": list(report.sweep.lora_ranks),
        },
        "anchor": asdict(report.anchor),
        "grid_size": report.grid_size,
        "fitting_count": report.fitting_count,
        "frontier": [_frontier_point_to_dict(point) for point in report.frontier],
        "recommended": (
            _frontier_point_to_dict(report.recommended)
            if report.recommended is not None
            else None
        ),
        "ceilings": [
            {
                "axis": ceiling.axis,
                "max_value": ceiling.max_value,
                "total_mib_at_max": mib(ceiling.total_mib_at_max),
            }
            for ceiling in report.ceilings
        ],
        "prices": [
            {
                "axis": price.axis,
                "from_value": price.from_value,
                "to_value": price.to_value,
                "delta_mib": mib(price.delta_mib),
            }
            for price in report.prices
        ],
        "verdict": {
            "fits": report.fitting_count > 0,
            "gpu_capacity_mib": mib(float(report.gpu.usable_mib)),
        },
    }


class _DefaultContext(click.Context):
    @property
    def command_path(self) -> str:
        return super().command_path.strip()


class _DefaultCommand(click.Command):
    context_class = _DefaultContext


@click.command(
    cls=_DefaultCommand,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=_HELP_EPILOG,
)
@click.argument("model_id", required=False)
@click.option(
    "--quant",
    type=click.Choice(["none", "nf4", "int8"]),
    default="none",
    show_default=True,
    help="BASE MODEL storage format.",
)
@click.option(
    "--double-quant",
    is_flag=True,
    help="NF4 double quantization (halves the scale overhead).",
)
@click.option(
    "--qlora",
    is_flag=True,
    help="Shorthand for --quant nf4 --precision bf16 --grad-checkpoint. "
    "An explicit flag always wins over it.",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="bf16",
    show_default=True,
    help="COMPUTE dtype: LoRA weights, gradients, activations.",
)
@click.option(
    "--lora-r", type=click.IntRange(min=1), default=16, show_default=True,
    help="LoRA rank.",
)
@click.option(
    "--no-lora", is_flag=True, help="Full fine-tuning: every parameter is trainable."
)
@click.option(
    "--lora-targets",
    default="standard",
    show_default=True,
    help="Preset (minimal | standard | full) or comma-separated modules, "
    "e.g. 'q,k,v,o' or 'q_proj,v_proj'.",
)
@click.option(
    "--batch-size", type=click.IntRange(min=1), default=1, show_default=True,
    help="MICRO-batch size: what one forward/backward sees.",
)
@click.option(
    "--grad-accum", type=click.IntRange(min=1), default=1, show_default=True,
    help="Accumulation steps. Display only, costs no memory.",
)
@click.option(
    "--seq-len", type=click.IntRange(min=1), default=2048, show_default=True,
    help="Sequence length.",
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "adam8bit", "sgd", "sgd-momentum"]),
    default="adamw",
    show_default=True,
)
@click.option(
    "--optimizer-dtype",
    type=click.Choice(["fp32", "bf16"]),
    default="fp32",
    show_default=True,
    help="AdamW state dtype. fp32 states cost 8 bytes/param, bf16 states 4.",
)
@click.option("--grad-checkpoint", is_flag=True, help="Enable gradient checkpointing.")
@click.option("--flash-attn", is_flag=True, help="Enable Flash Attention.")
@click.option(
    "--gpu", default=None, help=f"GPU name from the database  [default: {_DEFAULT_GPU}]"
)
@click.option(
    "--vram-mib",
    type=click.IntRange(min=1),
    default=None,
    help="VRAM override for a GPU not in the database (usable is taken as 95%).",
)
@click.option(
    "--list-gpus", is_flag=True, help="Print the GPU database and exit."
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (for CI/CD).")
@click.option("--no-color", is_flag=True, help="Disable colored output.")
@click.option("--verbose", is_flag=True, help="Show the per-layer breakdown.")
@click.option(
    "--explain", is_flag=True, help="Plain-English breakdown plus savings hints."
)
@click.version_option(_package_version(), "-V", "--version", prog_name="fitcheck")
@click.pass_context
def estimate_command(
    ctx: click.Context,
    model_id: str | None,
    quant: str,
    double_quant: bool,
    qlora: bool,
    precision: str,
    lora_r: int,
    no_lora: bool,
    lora_targets: str,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    optimizer: str,
    optimizer_dtype: str,
    grad_checkpoint: bool,
    flash_attn: bool,
    gpu: str | None,
    vram_mib: int | None,
    list_gpus: bool,
    as_json: bool,
    no_color: bool,
    verbose: bool,
    explain: bool,
) -> None:
    console = make_console(no_color=no_color)

    if list_gpus:
        console.print(render_gpu_table())
        return

    if qlora:
        if not _explicit(ctx, "quant"):
            quant = "nf4"
        if not _explicit(ctx, "precision"):
            precision = "bf16"
        if not _explicit(ctx, "grad_checkpoint"):
            grad_checkpoint = True

    _validate_combination(ctx, quant, double_quant, no_lora, optimizer)

    training = TrainingConfig(
        precision=precision,
        quantization=quant,
        double_quant=double_quant,
        optimizer=optimizer,
        optimizer_dtype=optimizer_dtype,
        batch_size=batch_size,
        seq_len=seq_len,
        lora_rank=None if no_lora else lora_r,
        lora_targets=_parse_lora_targets(lora_targets),
        grad_checkpoint=grad_checkpoint,
        flash_attn=flash_attn,
        grad_accum_steps=grad_accum,
    )

    if model_id is None:
        ctx.exit(_enter_repl(ctx, console, training, gpu, vram_mib))

    gpu_spec = _resolve_gpu(ctx, gpu, vram_mib)
    model_config = _load_model_config(model_id)

    try:
        report = estimate(model_config, training, gpu_spec)
    except ValueError as error:
        raise click.UsageError(str(error)) from error

    if as_json:
        click.echo(
            json.dumps(report_to_dict(report, model_config, gpu_spec, training), indent=2)
        )
    else:
        ascii_only = use_ascii_glyphs(console)
        console.print(
            render_report(report, model_config, gpu_spec, training, ascii_only=ascii_only)
        )
        if verbose:
            console.print(
                render_verbose_detail(model_config, training, ascii_only=ascii_only)
            )
        if explain:
            console.print(
                render_explanation(
                    report, model_config, training, ascii_only=ascii_only
                )
            )

    ctx.exit(0 if report.fits else _EXIT_DOES_NOT_FIT)


@click.command(
    "infer",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=_INFER_EPILOG,
    short_help="Estimate VRAM for serving a model (weights + KV cache).",
)
@click.argument("model_id")
@click.option(
    "--quant",
    type=click.Choice(["none", "nf4", "int8"]),
    default="none",
    show_default=True,
    help="BASE MODEL storage format. Embeddings, LM head and norms stay unquantized.",
)
@click.option(
    "--double-quant",
    is_flag=True,
    help="NF4 double quantization (halves the scale overhead).",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp16",
    show_default=True,
    help="COMPUTE dtype: the KV cache and the unquantized weight slice. "
    "A 4-bit deployment still serves an fp16 cache, so this is NOT --quant.",
)
@click.option(
    "--seq-len",
    type=click.IntRange(min=1),
    default=2048,
    show_default=True,
    help="Context length one request holds.",
)
@click.option(
    "--concurrent",
    "--num-concurrent",
    "num_concurrent",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Requests in flight. Interchangeable with --seq-len: 4 x 2048 costs what "
    "1 x 8192 costs.",
)
@click.option(
    "--gpu", default=None, help=f"GPU name from the database  [default: {_DEFAULT_GPU}]"
)
@click.option(
    "--vram-mib",
    type=click.IntRange(min=1),
    default=None,
    help="VRAM override for a GPU not in the database (usable is taken as 95%).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (for CI/CD).")
@click.option("--no-color", is_flag=True, help="Disable colored output.")
@click.pass_context
def infer_command(
    ctx: click.Context,
    model_id: str,
    quant: str,
    double_quant: bool,
    precision: str,
    seq_len: int,
    num_concurrent: int,
    gpu: str | None,
    vram_mib: int | None,
    as_json: bool,
    no_color: bool,
) -> None:
    """Estimate VRAM for serving MODEL_ID: resident weights plus the KV cache."""
    console = make_console(no_color=no_color)

    _validate_serving_combination(quant, double_quant)

    serving = ServingConfig(
        precision=precision,
        quantization=quant,
        double_quant=double_quant,
        seq_len=seq_len,
        num_concurrent=num_concurrent,
    )

    gpu_spec = _resolve_gpu(ctx, gpu, vram_mib)
    model_config = _load_model_config(model_id)

    try:
        report = estimate_inference(model_config, serving, gpu_spec)
    except ValueError as error:
        raise click.UsageError(str(error)) from error

    if as_json:
        click.echo(
            json.dumps(
                inference_report_to_dict(report, model_config, gpu_spec, serving),
                indent=2,
            )
        )
    else:
        console.print(
            render_inference_report(
                report,
                model_config,
                gpu_spec,
                serving,
                ascii_only=use_ascii_glyphs(console),
            )
        )

    ctx.exit(0 if report.fits else _EXIT_DOES_NOT_FIT)


@click.command(
    "advise",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=_ADVISE_EPILOG,
    short_help="Sweep batch size and LoRA rank: what each axis costs, and the wall.",
)
@click.argument("model_id")
@click.option(
    "--seq-lens",
    required=True,
    help="REQUIRED. Sequence lengths to sweep, e.g. '1024,2048,4096'. No default: "
    "config.json does not tell fitcheck a model's real context length.",
)
@click.option(
    "--max-seq-len",
    type=click.IntRange(min=1),
    default=None,
    help="The model's real context length. Any --seq-lens value above it is "
    "rejected instead of priced.",
)
@click.option(
    "--batch-sizes",
    default=",".join(str(value) for value in DEFAULT_BATCH_SIZES),
    show_default=True,
    help="Micro-batch sizes to sweep.",
)
@click.option(
    "--lora-ranks",
    default=",".join(str(value) for value in DEFAULT_LORA_RANKS),
    show_default=True,
    help="LoRA ranks to sweep.",
)
@click.option(
    "--quant",
    type=click.Choice(["none", "nf4", "int8"]),
    default="none",
    show_default=True,
    help="BASE MODEL storage format. Held fixed across the sweep.",
)
@click.option(
    "--double-quant",
    is_flag=True,
    help="NF4 double quantization (halves the scale overhead).",
)
@click.option(
    "--qlora",
    is_flag=True,
    help="Shorthand for --quant nf4 --precision bf16 --grad-checkpoint. "
    "An explicit flag always wins over it.",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="bf16",
    show_default=True,
    help="COMPUTE dtype: LoRA weights, gradients, activations.",
)
@click.option(
    "--lora-targets",
    default="standard",
    show_default=True,
    help="Preset (minimal | standard | full) or comma-separated modules, "
    "e.g. 'q,k,v,o' or 'q_proj,v_proj'.",
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "adam8bit", "sgd", "sgd-momentum"]),
    default="adamw",
    show_default=True,
)
@click.option(
    "--optimizer-dtype",
    type=click.Choice(["fp32", "bf16"]),
    default="fp32",
    show_default=True,
    help="AdamW state dtype. fp32 states cost 8 bytes/param, bf16 states 4.",
)
@click.option("--grad-checkpoint", is_flag=True, help="Enable gradient checkpointing.")
@click.option("--flash-attn", is_flag=True, help="Enable Flash Attention.")
@click.option(
    "--gpu", default=None, help=f"GPU name from the database  [default: {_DEFAULT_GPU}]"
)
@click.option(
    "--vram-mib",
    type=click.IntRange(min=1),
    default=None,
    help="VRAM override for a GPU not in the database (usable is taken as 95%).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON (for CI/CD).")
@click.option("--no-color", is_flag=True, help="Disable colored output.")
@click.pass_context
def advise_command(
    ctx: click.Context,
    model_id: str,
    seq_lens: str,
    max_seq_len: int | None,
    batch_sizes: str,
    lora_ranks: str,
    quant: str,
    double_quant: bool,
    qlora: bool,
    precision: str,
    lora_targets: str,
    optimizer: str,
    optimizer_dtype: str,
    grad_checkpoint: bool,
    flash_attn: bool,
    gpu: str | None,
    vram_mib: int | None,
    as_json: bool,
    no_color: bool,
) -> None:
    console = make_console(no_color=no_color)

    if qlora:
        if not _explicit(ctx, "quant"):
            quant = "nf4"
        if not _explicit(ctx, "precision"):
            precision = "bf16"
        if not _explicit(ctx, "grad_checkpoint"):
            grad_checkpoint = True

    _validate_combination(ctx, quant, double_quant, False, optimizer)

    swept_batch_sizes = _parse_int_list(batch_sizes, "--batch-sizes")
    swept_seq_lens = _parse_int_list(seq_lens, "--seq-lens")
    swept_lora_ranks = _parse_int_list(lora_ranks, "--lora-ranks")

    if max_seq_len is not None:
        too_long = [value for value in swept_seq_lens if value > max_seq_len]
        if too_long:
            raise click.UsageError(
                f"--seq-lens {', '.join(f'{value:,}' for value in too_long)} exceed "
                f"--max-seq-len {max_seq_len:,}. Pricing a context the model cannot "
                "serve would recommend a run that cannot be trained."
            )

    sweep = SweepSpec(
        batch_sizes=tuple(swept_batch_sizes),
        seq_lens=tuple(swept_seq_lens),
        lora_ranks=tuple(swept_lora_ranks),
    )
    base = TrainingConfig(
        precision=precision,
        quantization=quant,
        double_quant=double_quant,
        optimizer=optimizer,
        optimizer_dtype=optimizer_dtype,
        batch_size=min(swept_batch_sizes),
        seq_len=min(swept_seq_lens),
        lora_rank=min(swept_lora_ranks),
        lora_targets=_parse_lora_targets(lora_targets),
        grad_checkpoint=grad_checkpoint,
        flash_attn=flash_attn,
    )

    gpu_spec = _resolve_gpu(ctx, gpu, vram_mib)
    model_config = _load_model_config(model_id)

    try:
        report = advise(model_config, base, gpu_spec, sweep, model_id=model_id)
    except ValueError as error:
        raise click.UsageError(str(error)) from error

    if as_json:
        click.echo(json.dumps(advisor_report_to_dict(report, model_config), indent=2))
    else:
        console.print(
            render_advisor_report(
                report, model_config, ascii_only=use_ascii_glyphs(console)
            )
        )

    ctx.exit(0 if report.fitting_count else _EXIT_DOES_NOT_FIT)


class _FitcheckGroup(click.Group):
    _DEFAULT_COMMAND = "estimate"
    _DEFAULTED = "fitcheck.default_command"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args or args[0] not in self.commands:
            ctx.meta[self._DEFAULTED] = True
            args = [self._DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        name, command, rest = super().resolve_command(ctx, args)
        if ctx.meta.pop(self._DEFAULTED, False):
            return "", command, rest
        return name, command, rest


main = _FitcheckGroup(
    name="fitcheck",
    commands={
        "estimate": estimate_command,
        "infer": infer_command,
        "advise": advise_command,
    },
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
