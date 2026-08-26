from __future__ import annotations

import re
from dataclasses import dataclass

from rich.box import SIMPLE_HEAD
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fitcheck.config_parser import ModelConfig
from fitcheck.estimator import MemoryReport, TrainingConfig, _count_lora_params
from fitcheck.gpu_db import GpuSpec, list_gpus
from fitcheck.memory.activations import estimate_activation_memory
from fitcheck.utils import precision_to_bytes

_TIGHT_HEADROOM_FRACTION = 0.20
_BAR_WIDTH = 44
_STYLE_FITS = "green"
_STYLE_TIGHT = "yellow"
_STYLE_OOM = "red"
_STYLE_LABEL = "dim"
_SAVINGS_PATTERN = re.compile(r"saves\s+([\d,]+)\s+MiB")


@dataclass(frozen=True)
class _Glyphs:
    bar_filled: str
    bar_empty: str
    fits: str
    tight: str
    oom: str
    hint: str
    arrow: str
    separator: str


_UNICODE_GLYPHS = _Glyphs("█", "░", "✅", "⚠️", "❌", "💡", "→", " · ")
_ASCII_GLYPHS = _Glyphs("#", ".", "[OK]", "[!]", "[X]", ">", "->", " | ")


def _fraction(part: float, whole: float) -> float:
    return part / whole if whole > 0 else 0.0


def _mib(value: float) -> str:
    return f"{value:,.0f}"


def _percent(fraction: float, decimals: int = 1) -> str:
    return f"{fraction * 100:.{decimals}f}%"


def _attention_label(config: ModelConfig) -> str:
    if config.num_kv_heads == config.num_attention_heads:
        return f"{config.num_attention_heads} heads (MHA)"
    if config.num_kv_heads == 1:
        return f"{config.num_attention_heads} heads (MQA)"
    return f"{config.num_attention_heads} heads, GQA {config.num_kv_heads} KV heads"


def _params_label(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B params"
    return f"{num_params / 1_000_000:.0f}M params"


def _model_line(config: ModelConfig, glyphs: _Glyphs) -> str:
    return glyphs.separator.join(
        (
            config.name,
            _params_label(config.num_params),
            f"{config.num_layers} layers",
            _attention_label(config),
        )
    )


def _gpu_line(gpu: GpuSpec, glyphs: _Glyphs) -> str:
    return glyphs.separator.join(
        (
            gpu.name,
            f"{_mib(gpu.usable_mib)} MiB usable of {_mib(gpu.vram_mib)} MiB",
        )
    )


def _adapter_label(training: TrainingConfig) -> str:
    if training.lora_rank is None:
        return "full fine-tune"

    targets = ",".join(
        target.removesuffix("_proj") for target in training.lora_targets
    )
    prefix = "QLoRA" if training.quantization != "none" else "LoRA"
    return f"{prefix} r={training.lora_rank} [{targets}]"


def _batch_label(training: TrainingConfig) -> str:
    if training.grad_accum_steps > 1:
        effective = training.batch_size * training.grad_accum_steps
        return (
            f"bs {training.batch_size} x {training.grad_accum_steps} accum "
            f"= {effective} effective"
        )
    return f"bs {training.batch_size}"


def _optimizer_label(training: TrainingConfig) -> str:
    if training.optimizer.strip().casefold() == "adamw":
        return f"adamw ({training.optimizer_dtype} states)"
    return training.optimizer


def _config_line(training: TrainingConfig, glyphs: _Glyphs) -> str:
    return glyphs.separator.join(
        (
            _adapter_label(training),
            _batch_label(training),
            f"seq {training.seq_len:,}",
            training.precision,
            _optimizer_label(training),
        )
    )


def _weights_label(training: TrainingConfig | None) -> str:
    if training is None:
        return "Base model weights"
    if training.quantization != "none":
        suffix = " + double quant" if training.double_quant else ""
        return f"Base model weights ({training.quantization.upper()}{suffix})"
    return f"Base model weights ({training.precision})"


def _lora_label(training: TrainingConfig | None) -> str:
    if training is not None and training.lora_rank is None:
        return "LoRA adapter (none, full fine-tune)"
    return "LoRA adapter (trainable)"


def _activation_label(training: TrainingConfig | None) -> str:
    if training is None:
        return "Activations"

    checkpoint = "grad ckpt" if training.grad_checkpoint else "no ckpt"
    flash = "flash attn" if training.flash_attn else "no flash"
    return f"Activations ({checkpoint}, {flash})"


def _component_rows(
    report: MemoryReport, training: TrainingConfig | None
) -> list[tuple[str, float]]:
    return [
        (_weights_label(training), report.weight_mib),
        (_lora_label(training), report.lora_mib),
        ("Optimizer states", report.optimizer_mib),
        ("Gradients", report.gradient_mib),
        (_activation_label(training), report.activation_mib),
        ("CUDA context + buffers", report.overhead_mib),
    ]


def _header_grid(
    config: ModelConfig, gpu: GpuSpec, training: TrainingConfig | None, glyphs: _Glyphs
) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=_STYLE_LABEL, justify="right")
    grid.add_column()
    grid.add_row("Model", Text(_model_line(config, glyphs)))
    grid.add_row("GPU", Text(_gpu_line(gpu, glyphs)))
    if training is not None:
        grid.add_row("Config", Text(_config_line(training, glyphs)))
    return grid


def _component_table(
    report: MemoryReport, training: TrainingConfig | None, verdict_style: str
) -> Table:
    table = Table(box=SIMPLE_HEAD, pad_edge=False, expand=True)
    table.add_column("Component")
    table.add_column("Memory (MiB)", justify="right")
    table.add_column("% of Total", justify="right")

    for label, value in _component_rows(report, training):
        table.add_row(
            label, _mib(value), _percent(_fraction(value, report.total_mib))
        )

    table.add_section()
    table.add_row(
        Text("TOTAL (predicted peak)", style="bold"),
        Text(_mib(report.total_mib), style=f"bold {verdict_style}"),
        "",
    )
    table.add_row("GPU capacity (usable)", _mib(report.gpu_capacity_mib), "")
    table.add_row(
        "Headroom",
        Text(_mib(report.headroom_mib), style=verdict_style),
        _percent(_fraction(report.headroom_mib, report.gpu_capacity_mib), decimals=0),
    )
    return table


def _verdict_style(report: MemoryReport) -> str:
    if not report.fits:
        return _STYLE_OOM
    headroom = _fraction(report.headroom_mib, report.gpu_capacity_mib)
    return _STYLE_FITS if headroom > _TIGHT_HEADROOM_FRACTION else _STYLE_TIGHT


def _usage_bar(report: MemoryReport, verdict_style: str, glyphs: _Glyphs) -> Text:
    used = _fraction(report.total_mib, report.gpu_capacity_mib)
    filled = min(_BAR_WIDTH, max(1, round(used * _BAR_WIDTH))) if used > 0 else 0

    bar = Text()
    bar.append(glyphs.bar_filled * filled, style=verdict_style)
    bar.append(glyphs.bar_empty * (_BAR_WIDTH - filled), style=_STYLE_LABEL)
    bar.append(
        f"  {_percent(used, decimals=0)} of {_mib(report.gpu_capacity_mib)} MiB",
        style=_STYLE_LABEL,
    )
    return bar


def _verdict_line(report: MemoryReport, verdict_style: str, glyphs: _Glyphs) -> Text:
    headroom = _percent(
        _fraction(report.headroom_mib, report.gpu_capacity_mib), decimals=0
    )

    if not report.fits:
        icon, message = (
            glyphs.oom,
            f"DOES NOT FIT {glyphs.arrow} over by {_mib(-report.headroom_mib)} MiB",
        )
    elif verdict_style == _STYLE_TIGHT:
        icon, message = (
            glyphs.tight,
            f"FITS, BARELY {glyphs.arrow} only {_mib(report.headroom_mib)} MiB "
            f"({headroom}) headroom",
        )
    else:
        icon, message = (
            glyphs.fits,
            f"FITS {glyphs.arrow} {_mib(report.headroom_mib)} MiB ({headroom}) "
            "headroom remaining",
        )

    return Text(f"{icon} {message}", style=f"bold {verdict_style}")


def _batch_suggestion(report: MemoryReport, training: TrainingConfig | None) -> str:
    if report.max_batch_size == 0:
        return (
            "Even batch_size=1 does not fit: shorten --seq-len, add "
            "--grad-checkpoint / --flash-attn, or quantize the base with --quant nf4."
        )

    if training is None:
        return (
            f"Max micro-batch size at this sequence length: {report.max_batch_size}."
        )
    if report.max_batch_size > training.batch_size:
        return (
            f"You could increase batch_size to {report.max_batch_size} before "
            "hitting the memory ceiling."
        )
    if report.max_batch_size < training.batch_size:
        return (
            f"Drop batch_size to {report.max_batch_size} to fit, then use "
            "--grad-accum to keep the effective batch at "
            f"{report.effective_batch_size}."
        )
    return (
        f"batch_size {training.batch_size} is already the maximum that fits at this "
        "sequence length."
    )


def _hint_savings_mib(hint: str) -> float:
    match = _SAVINGS_PATTERN.search(hint)
    return float(match.group(1).replace(",", "")) if match else 0.0


def _best_savings_hint(report: MemoryReport, glyphs: _Glyphs) -> str | None:
    if not report.savings_hints:
        return None

    best_hint = max(report.savings_hints, key=_hint_savings_mib)
    if _hint_savings_mib(best_hint) <= 0:
        return None
    return best_hint.replace("->", glyphs.arrow)


def _suggestion_grid(
    report: MemoryReport, training: TrainingConfig | None, glyphs: _Glyphs
) -> Table:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style=_STYLE_LABEL)
    grid.add_column(style=_STYLE_LABEL, overflow="fold")
    grid.add_row(glyphs.hint, Text(_batch_suggestion(report, training)))

    savings_hint = _best_savings_hint(report, glyphs)
    if savings_hint is not None:
        grid.add_row(glyphs.hint, Text(savings_hint))
    return grid


def render_report(
    report: MemoryReport,
    config: ModelConfig,
    gpu: GpuSpec,
    training: TrainingConfig | None = None,
    *,
    ascii_only: bool = False,
) -> Panel:
    glyphs = _ASCII_GLYPHS if ascii_only else _UNICODE_GLYPHS
    verdict_style = _verdict_style(report)

    body: list[RenderableType] = [
        _header_grid(config, gpu, training, glyphs),
        _component_table(report, training, verdict_style),
        _usage_bar(report, verdict_style, glyphs),
        Text(""),
        _verdict_line(report, verdict_style, glyphs),
        _suggestion_grid(report, training, glyphs),
    ]

    return Panel(
        Group(*body),
        title="fitcheck",
        title_align="left",
        border_style=verdict_style,
        padding=(1, 2),
    )


def make_console(*, no_color: bool = False) -> Console:
    return Console(color_system=None) if no_color else Console()


def use_ascii_glyphs(console: Console) -> bool:
    """True when the console encoding cannot represent the Unicode glyph set."""
    return "utf" not in (console.encoding or "").casefold()


def print_report(
    report: MemoryReport,
    config: ModelConfig,
    gpu: GpuSpec,
    training: TrainingConfig | None = None,
    *,
    console: Console | None = None,
) -> None:
    target_console = console if console is not None else make_console()
    target_console.print(
        render_report(
            report, config, gpu, training, ascii_only=use_ascii_glyphs(target_console)
        )
    )


def activation_breakdown(
    config: ModelConfig, training: TrainingConfig
) -> dict[str, float]:
    """Split A_act into its per-layer parts, in MiB.

    Every figure comes from re-running `estimate_activation_memory` with one input
    changed, never from re-deriving the formula here: A_layer is the no-checkpoint
    total over L, and the attention-matrix term is the flash-off minus flash-on
    difference. A formula fix in activations.py therefore lands here for free.
    """

    def activations(grad_checkpoint: bool, flash_attn: bool) -> float:
        return estimate_activation_memory(
            config,
            training.batch_size,
            training.seq_len,
            grad_checkpoint,
            flash_attn,
            training.precision,
        )

    layers = config.num_layers
    layer_mib = activations(False, training.flash_attn) / layers
    checkpointed_mib = activations(True, training.flash_attn)

    return {
        "layer_mib": layer_mib,
        "attention_matrix_mib": (activations(False, False) - activations(False, True))
        / layers,
        "all_layers_mib": layer_mib * layers,
        "stored_inputs_mib": checkpointed_mib - layer_mib,
        "checkpointed_mib": checkpointed_mib,
    }


def trainable_params(config: ModelConfig, training: TrainingConfig) -> int:
    """Trainable parameter count: the LoRA adapters, or every parameter for full FT."""
    if training.lora_rank is None:
        return config.num_params
    return _count_lora_params(config, training.lora_rank, training.lora_targets)


def _geometry_grid(config: ModelConfig, training: TrainingConfig) -> Table:
    trainable = trainable_params(config, training)
    ffn_ratio = config.intermediate_size / config.hidden_size
    gamma = precision_to_bytes(training.precision)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=_STYLE_LABEL, justify="right")
    grid.add_column()
    for label, value in (
        ("hidden_size", f"{config.hidden_size:,}"),
        ("layers", f"{config.num_layers:,}"),
        (
            "attention",
            f"{config.num_attention_heads} Q heads, {config.num_kv_heads} KV heads, "
            f"head_dim {config.head_dim}",
        ),
        (
            "intermediate_size",
            f"{config.intermediate_size:,}  ({ffn_ratio:.2f} x hidden_size)",
        ),
        (
            "vocab_size",
            f"{config.vocab_size:,}  "
            f"(tied embeddings: {'yes' if config.tie_word_embeddings else 'no'})",
        ),
        ("parameters", f"{config.num_params:,}"),
        (
            "trainable",
            f"{trainable:,}  ({_percent(_fraction(trainable, config.num_params))})",
        ),
        ("activation dtype", f"{training.precision} ({gamma:g} bytes/element)"),
    ):
        grid.add_row(label, Text(value))
    return grid


def _activation_detail_table(
    config: ModelConfig, training: TrainingConfig, ascii_only: bool
) -> Table:
    parts = activation_breakdown(config, training)
    layers = config.num_layers
    gamma_bsh = "gamma*b*s*h" if ascii_only else "γbsh"

    table = Table(box=SIMPLE_HEAD, pad_edge=False, expand=True)
    table.add_column(
        f"Per-layer activations (b={training.batch_size}, s={training.seq_len:,})"
    )
    table.add_column("MiB", justify="right")

    table.add_row("A_layer, all saved tensors for one layer", _mib(parts["layer_mib"]))
    attention_note = (
        "avoided by Flash Attention"
        if training.flash_attn
        else "included, no Flash Attention"
    )
    table.add_row(
        f"  attention matrix (b, n_h, s, s): {attention_note}",
        _mib(parts["attention_matrix_mib"]),
    )

    table.add_section()
    if training.grad_checkpoint:
        table.add_row(
            f"x {layers} layers, no checkpointing (what you are NOT paying)",
            Text(_mib(parts["all_layers_mib"]), style=_STYLE_LABEL),
        )
        table.add_row(
            f"Stored layer inputs (L x {gamma_bsh})", _mib(parts["stored_inputs_mib"])
        )
        table.add_row("+ one recomputed layer", _mib(parts["layer_mib"]))
        charged_mib = parts["checkpointed_mib"]
    else:
        table.add_row(f"x {layers} layers, every layer's tensors kept", "")
        charged_mib = parts["all_layers_mib"]

    table.add_row(
        Text("= A_act charged", style="bold"), Text(_mib(charged_mib), style="bold")
    )
    return table


def _largest_component(report: MemoryReport) -> tuple[str, float]:
    components = {
        "base model weights": report.weight_mib,
        "the LoRA adapter": report.lora_mib,
        "optimizer states": report.optimizer_mib,
        "gradients": report.gradient_mib,
        "activations": report.activation_mib,
        "CUDA overhead": report.overhead_mib,
    }
    return max(components.items(), key=lambda item: item[1])


def _why_largest(name: str, config: ModelConfig, training: TrainingConfig) -> str:
    if name == "base model weights":
        if training.quantization == "none":
            return (
                f"the frozen base model held in {training.precision}. Quantizing it "
                "with --quant nf4 cuts this to roughly a quarter."
            )
        return (
            f"the frozen base model at {training.quantization.upper()} plus its "
            "quantization scales. That is close to the floor for a model this size."
        )
    if name == "activations":
        if training.grad_checkpoint:
            return (
                f"{config.num_layers} stored layer inputs plus one recomputed layer, "
                f"at micro-batch {training.batch_size} x seq {training.seq_len:,}. "
                "Lowering --batch-size or --seq-len is the lever here."
            )
        return (
            f"all {config.num_layers} layers keep their full set of saved tensors. "
            "--grad-checkpoint trades compute for most of this."
        )
    if name == "optimizer states":
        if training.optimizer.strip().casefold() == "adamw":
            return (
                f"AdamW keeps momentum and variance in {training.optimizer_dtype} "
                f"(8 bytes per trainable param) even though you train in "
                f"{training.precision}."
            )
        return f"{training.optimizer} states over the trainable parameters."
    if name == "gradients":
        return f"one .grad tensor per trainable parameter, in {training.precision}."
    if name == "the LoRA adapter":
        return (
            f"rank {training.lora_rank} across {len(training.lora_targets)} target "
            "modules per layer. A smaller --lora-r shrinks it linearly."
        )
    return (
        "the 500 MiB CUDA context floor plus 5% of weights and activations — it grows "
        "with everything else, so it is never the thing to optimize first."
    )


def _toggle_table(report: MemoryReport, glyphs: _Glyphs) -> Table:
    # Hints arrive as prose ("--flash-attn OFF: costs +1,075 MiB (currently ON)");
    # split once on the colon to get a label column and a price column.
    parsed = [
        tuple(part.strip() for part in hint.split(":", 1))
        if ":" in hint
        else (hint, "")
        for hint in report.savings_hints
    ]
    if not parsed:
        return Table.grid()

    width = max(len(label) for label, _ in parsed) + 2

    table = Table.grid(padding=(0, 1))
    table.add_column(style=_STYLE_LABEL)
    table.add_column()
    for label, price in parsed:
        leader = "." * (width - len(label))
        table.add_row(
            Text(f"{label.replace('->', glyphs.arrow)} {leader}"), Text(price)
        )
    return table


def render_explanation(
    report: MemoryReport,
    config: ModelConfig,
    training: TrainingConfig,
    *,
    ascii_only: bool = False,
) -> Panel:
    """Render `--explain`: which component dominates and why, then the price of each
    toggle. Every figure is a total-memory delta the estimator already computed by
    re-running itself with one flag flipped — no new math happens here."""
    glyphs = _ASCII_GLYPHS if ascii_only else _UNICODE_GLYPHS
    name, value = _largest_component(report)

    share = _percent(_fraction(value, report.total_mib), 0)
    dash = "-" if ascii_only else "—"
    headline = Text.assemble(
        (f"Largest component: {name} ({_mib(value)} MiB, {share}) {dash} ", "bold"),
        _why_largest(name, config, training),
    )

    return Panel(
        Group(headline, Text(""), _toggle_table(report, glyphs)),
        title="explain",
        title_align="left",
        border_style=_STYLE_LABEL,
        padding=(1, 2),
    )


def render_gpu_table() -> Table:
    """Render the GPU database as a table, for `--list-gpus`."""
    table = Table(box=SIMPLE_HEAD, pad_edge=False, title="fitcheck GPU database")
    table.add_column("--gpu")
    table.add_column("Name")
    table.add_column("VRAM (MiB)", justify="right")
    table.add_column("Usable (MiB)", justify="right")
    table.add_column("Usable %", justify="right")

    for key, spec in list_gpus():
        table.add_row(
            key,
            spec.name,
            f"{spec.vram_mib:,}",
            f"{spec.usable_mib:,}",
            _percent(_fraction(spec.usable_mib, spec.vram_mib), 0),
        )
    return table


def render_verbose_detail(
    config: ModelConfig, training: TrainingConfig, *, ascii_only: bool = False
) -> Panel:
    """Render the per-layer breakdown behind `--verbose`: model geometry, parameter
    counts, and how A_act decomposes into per-layer terms."""
    glyphs = _ASCII_GLYPHS if ascii_only else _UNICODE_GLYPHS

    return Panel(
        Group(
            _geometry_grid(config, training),
            Text(""),
            _activation_detail_table(config, training, ascii_only),
        ),
        title=f"detail {glyphs.arrow} {config.name}",
        title_align="left",
        border_style=_STYLE_LABEL,
        padding=(1, 2),
    )
