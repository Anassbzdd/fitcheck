"""Gradio web interface for the FitCheck estimators."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from html import escape
from typing import Any

import gradio as gr
from fitcheck.advisor import DEFAULT_BATCH_SIZES, DEFAULT_LORA_RANKS
from fitcheck.config_parser import HubUnavailableError, UnsupportedModelError
from fitcheck.gpu_db import GPU_DB
from fitcheck.web import (
    AdvisorResult,
    ServingResult,
    TrainingResult,
    advise_training_request,
    estimate_serving_request,
    estimate_training_request,
)
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

_LOGGER = logging.getLogger("fitcheck.web")
_REPOSITORY_URL = "https://github.com/Anassbzdd/fitcheck"
_SPEC_URL = f"{_REPOSITORY_URL}/blob/main/docs/SPEC.md"
_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"

_GPU_CHOICES = [(spec.name, key) for key, spec in GPU_DB.items()]
_STATUS_STYLE = {
    "SAFE": ("safe", "The estimate passes FitCheck's safety policy."),
    "UNCERTAIN": (
        "uncertain",
        "The point estimate fits, but its safety margin is not enough to call it safe.",
    ),
    "DOES NOT FIT": ("fail", "The predicted peak is above the usable GPU capacity."),
    "FITS": ("safe", "The resident-memory estimate fits the usable GPU capacity."),
    "NO FITTING CONFIGURATION": (
        "fail",
        "No configuration in this sweep fits the usable GPU capacity.",
    ),
    "SWEEP COMPLETE": ("safe", "At least one configuration in this sweep fits."),
}
_EMPTY_STATUS = (
    '<div class="fc-status fc-status-empty"><strong>READY</strong>'
    "<span>Run an estimate to see results.</span></div>"
)
_CSS = """
.fc-status { display:flex; gap:1rem; align-items:center; padding:1rem 1.2rem;
  border-radius:14px; border:1px solid; font-size:1.05rem; }
.fc-status strong { font-size:1.15rem; letter-spacing:.04em; }
.fc-status-safe { color:#14532d; background:#f0fdf4; border-color:#86efac; }
.fc-status-uncertain { color:#854d0e; background:#fffbeb; border-color:#fcd34d; }
.fc-status-fail { color:#991b1b; background:#fef2f2; border-color:#fca5a5; }
.fc-status-empty { color:#334155; background:#f8fafc; border-color:#cbd5e1; }
"""


def _status_card(status: str) -> str:
    """Render a status from a fixed allowlist; no request text enters HTML."""
    style, explanation = _STATUS_STYLE[status]
    return (
        f'<div class="fc-status fc-status-{style}" role="status">'
        f"<strong>{escape(status)}</strong><span>{escape(explanation)}</span></div>"
    )


def _empty_outputs(
    error: str, output_count: int, table_positions: Sequence[int]
) -> tuple[Any, ...]:
    """Clear every result field so a failed request cannot leave stale data."""
    values: list[Any] = [""] * output_count
    values[0] = _EMPTY_STATUS
    values[-1] = error
    for position in table_positions:
        values[position] = []
    return tuple(values)


def _error_text(error: Exception, operation: str) -> str:
    if isinstance(error, (UnsupportedModelError, HubUnavailableError, ValueError)):
        return str(error)
    if isinstance(error, RepositoryNotFoundError):
        return (
            "Could not read this model's config.json. Check the repository ID and confirm "
            "that it is public or that the Space owner has configured Hub access."
        )
    if isinstance(error, HfHubHTTPError):
        return (
            "Hugging Face returned an access or request error. Check the model ID and "
            "repository access, then try again."
        )
    if isinstance(error, RuntimeError) and "is gated on Hugging Face" in str(error):
        # fetch_model_config wraps gated access in RuntimeError with this stable wording.
        return str(error)
    _LOGGER.exception("Unexpected FitCheck %s request failure.", operation)
    return "The estimate could not be completed. Please check the settings and try again."


def _component_rows(report: Any) -> list[list[str | float]]:
    components = (
        ("Weights", report.weight_mib),
        ("LoRA", report.lora_mib),
        ("Optimizer", report.optimizer_mib),
        ("Gradients", report.gradient_mib),
        ("Activations", report.activation_mib),
        ("Overhead", report.overhead_mib),
        ("Total", report.total_mib),
    )
    return [
        [name, value, 100.0 if name == "Total" else value / report.total_mib * 100.0]
        for name, value in components
    ]


def _summary(report: Any) -> str:
    headroom_percent = (
        f"{report.headroom_mib / report.gpu_capacity_mib:+.1%}"
        if report.gpu_capacity_mib > 0
        else "not available (usable capacity is 0 MiB)"
    )
    return (
        f"**Predicted peak:** {report.total_mib:,.2f} MiB  \n"
        f"**Usable GPU capacity:** {report.gpu_capacity_mib:,.0f} MiB  \n"
        f"**Headroom:** {report.headroom_mib:+,.2f} MiB ({headroom_percent})"
    )


def training_callback(
    model_id: str,
    gpu_name: str,
    batch_size: object,
    seq_len: object,
    quantization: str,
    double_quant: bool,
    precision: str,
    lora_rank: object,
    target_preset: str,
    full_finetuning: bool,
    optimizer: str,
    optimizer_dtype: str,
    grad_checkpoint: bool,
    flash_attn: bool,
    grad_accum_steps: object,
    custom_vram_mib: object | None,
) -> tuple[Any, ...]:
    try:
        result: TrainingResult = estimate_training_request(
            model_id,
            gpu_name,
            batch_size=batch_size,
            seq_len=seq_len,
            quantization=quantization,
            double_quant=double_quant,
            precision=precision,
            lora_rank=lora_rank,
            target_preset=target_preset,
            full_finetuning=full_finetuning,
            optimizer=optimizer,
            optimizer_dtype=optimizer_dtype,
            grad_checkpoint=grad_checkpoint,
            flash_attn=flash_attn,
            grad_accum_steps=grad_accum_steps,
            custom_vram_mib=custom_vram_mib,
        )
    except Exception as error:
        return _empty_outputs(_error_text(error, "training"), 7, (2, 5))

    report = result.report
    status = report.verdict.upper().replace("_", " ")
    if status not in _STATUS_STYLE:
        status = "UNCERTAIN"
    recommendation = (
        "**Maximum micro-batch (point estimate):** "
        f"{report.max_batch_size:,}  \n"
        "**Recommended micro-batch:** "
        f"{report.recommended_batch_size or 0:,}  \n"
        f"**Effective batch size:** {report.effective_batch_size:,}  \n"
    )
    if report.recommendation_basis == "validated_reserve":
        recommendation += (
            "FitCheck applies the conservative reserve available for this validated "
            "configuration scope."
        )
    else:
        recommendation += (
            "No matching final-validation reserve applies; this recommendation keeps "
            "margin inside the point-estimate ceiling."
        )
    warning_text = "\n".join(report.warnings) if report.warnings else "No additional warnings."
    hints_text = "\n".join(f"• {hint}" for hint in report.savings_hints)
    detail_keys = (
        ("layer_mib", "Layer peak"),
        ("logits_mib", "LM-head logits"),
        ("checkpoint_store_mib", "Checkpoint store"),
        ("resident_hump_mib", "Larger live hump"),
    )
    activation_rows = [
        [label, result.activation_details[key]] for key, label in detail_keys
    ]
    return (
        _status_card(status),
        _summary(report),
        _component_rows(report),
        recommendation,
        f"Warnings\n{warning_text}\n\nSavings hints\n{hints_text}",
        activation_rows,
        "",
    )


def serving_callback(
    model_id: str,
    gpu_name: str,
    seq_len: object,
    num_concurrent: object,
    quantization: str,
    double_quant: bool,
    precision: str,
    custom_vram_mib: object | None,
) -> tuple[Any, ...]:
    try:
        result: ServingResult = estimate_serving_request(
            model_id,
            gpu_name,
            seq_len=seq_len,
            num_concurrent=num_concurrent,
            quantization=quantization,
            double_quant=double_quant,
            precision=precision,
            custom_vram_mib=custom_vram_mib,
        )
    except Exception as error:
        return _empty_outputs(_error_text(error, "serving"), 5, (2,))

    report = result.report
    status = "FITS" if report.fits else "DOES NOT FIT"
    summary = (
        f"**Resident-memory total:** {report.total_mib:,.2f} MiB  \n"
        f"**Usable GPU capacity:** {report.gpu_capacity_mib:,.0f} MiB  \n"
        f"**Headroom:** {report.headroom_mib:+,.2f} MiB  \n"
        f"**KV cache per request:** {report.kv_mib_per_request:,.2f} MiB  \n"
        f"**KV cache per token:** {report.kv_mib_per_token:,.4f} MiB  \n"
        f"**Maximum fitting concurrency:** {report.max_concurrent:,}  \n\n"
        "This is a resident-memory estimate. It does not model decode-step transient "
        "memory or throughput."
    )
    rows = [
        ["Resident weights", report.weight_mib],
        ["KV cache", report.kv_cache_mib],
        ["Overhead", report.overhead_mib],
        ["Total", report.total_mib],
    ]
    warnings = "\n".join(report.warnings) if report.warnings else "No additional warnings."
    return _status_card(status), summary, rows, warnings, ""


def advisor_callback(
    model_id: str,
    gpu_name: str,
    batch_sizes: Sequence[object],
    seq_lens: Sequence[object],
    lora_ranks: Sequence[object],
    max_context_length: object | None,
    quantization: str,
    double_quant: bool,
    precision: str,
    target_preset: str,
    optimizer: str,
    optimizer_dtype: str,
    grad_checkpoint: bool,
    flash_attn: bool,
    custom_vram_mib: object | None,
) -> tuple[Any, ...]:
    try:
        result: AdvisorResult = advise_training_request(
            model_id,
            gpu_name,
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            lora_ranks=lora_ranks,
            max_context_length=max_context_length,
            quantization=quantization,
            double_quant=double_quant,
            precision=precision,
            target_preset=target_preset,
            optimizer=optimizer,
            optimizer_dtype=optimizer_dtype,
            grad_checkpoint=grad_checkpoint,
            flash_attn=flash_attn,
            custom_vram_mib=custom_vram_mib,
        )
    except Exception as error:
        return _empty_outputs(_error_text(error, "advisor"), 7, (2, 3, 4))

    report = result.report
    has_fit = report.fitting_count > 0
    status = "SWEEP COMPLETE" if has_fit else "NO FITTING CONFIGURATION"
    recommended = report.recommended
    recommended_text = (
        "No fitting frontier point was found."
        if recommended is None
        else (
            f"**Recommended frontier point:** batch {recommended.batch_size}, "
            f"sequence {recommended.seq_len}, rank {recommended.lora_rank}; "
            f"{recommended.tokens_per_step:,} tokens/step at "
            f"{recommended.total_mib:,.2f} MiB."
        )
    )
    summary = (
        f"**Fitting configurations:** {report.fitting_count:,} / {report.grid_size:,}  \n"
        f"{recommended_text}"
    )
    frontier_rows = [
        [
            point.tokens_per_step,
            point.batch_size,
            point.seq_len,
            point.lora_rank,
            point.total_mib,
            point.command,
        ]
        for point in report.frontier
    ]
    ceiling_rows = [
        [ceiling.axis, ceiling.max_value, ceiling.total_mib_at_max]
        for ceiling in report.ceilings
    ]
    price_rows = [
        [price.axis, price.from_value, price.to_value, price.delta_mib]
        for price in report.prices
    ]
    note = (
        "Tokens/step is work per optimizer step (micro-batch × sequence length), "
        "not throughput. Frontier rows include runnable CLI commands."
    )
    return _status_card(status), summary, frontier_rows, ceiling_rows, price_rows, note, ""


def create_demo() -> gr.Blocks:
    """Create the FitCheck Gradio app without starting a server."""
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=_CSS,
        title="FitCheck",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# FitCheck\n"
            "Estimate LLM training and serving VRAM from Hugging Face model metadata — "
            "no weights, GPU, or PyTorch required.\n\n"
            "Sizing tool, not an OOM guarantee or throughput benchmark.\n\n"
            f"[Repository]({_REPOSITORY_URL}) · [Technical specification]({_SPEC_URL})"
        )

        with gr.Tabs():
            with gr.Tab("Training"):
                with gr.Row():
                    training_model = gr.Textbox(
                        label="Hugging Face model ID", value=_MODEL_DEFAULT
                    )
                    training_gpu = gr.Dropdown(
                        label="GPU preset",
                        choices=_GPU_CHOICES,
                        value="4090",
                        allow_custom_value=False,
                    )
                with gr.Row():
                    training_batch = gr.Number(
                        label="Micro-batch size", value=1, precision=0, minimum=1
                    )
                    training_seq = gr.Number(
                        label="Sequence length (tokens)",
                        value=2048,
                        precision=0,
                        minimum=1,
                    )
                with gr.Accordion("Advanced training settings", open=False):
                    with gr.Row():
                        training_quant = gr.Dropdown(
                            ["none", "nf4", "int8"],
                            value="nf4",
                            label="Base storage format",
                        )
                        training_double = gr.Checkbox(
                            label="NF4 double quantization", value=False
                        )
                        training_precision = gr.Dropdown(
                            ["fp32", "fp16", "bf16"],
                            value="bf16",
                            label="Compute precision",
                        )
                    with gr.Row():
                        training_rank = gr.Number(
                            label="LoRA rank", value=16, precision=0, minimum=1
                        )
                        training_targets = gr.Dropdown(
                            ["minimal", "standard", "full"],
                            value="standard",
                            label="LoRA target preset",
                        )
                        training_full = gr.Checkbox(
                            label="Full fine-tuning (no LoRA)", value=False
                        )
                    with gr.Row():
                        training_optimizer = gr.Dropdown(
                            ["adamw", "adam8bit", "sgd", "sgd-momentum"],
                            value="adamw",
                            label="Optimizer",
                        )
                        training_optimizer_dtype = gr.Dropdown(
                            ["fp32", "bf16"],
                            value="fp32",
                            label="Optimizer state dtype",
                        )
                    with gr.Row():
                        training_checkpoint = gr.Checkbox(
                            label="Gradient checkpointing", value=True
                        )
                        training_flash = gr.Checkbox(
                            label="Flash Attention", value=False
                        )
                        training_accum = gr.Number(
                            label="Gradient accumulation steps",
                            value=1,
                            precision=0,
                            minimum=1,
                        )
                    training_vram = gr.Number(
                        label="Custom total VRAM (MiB), optional",
                        value=None,
                        precision=0,
                        minimum=1,
                        info=(
                            "Overrides the selected GPU preset; usable capacity is 95% "
                            "of this total."
                        ),
                    )
                training_run = gr.Button("Estimate training memory", variant="primary")
                training_status = gr.HTML(value=_EMPTY_STATUS)
                training_summary = gr.Markdown()
                training_components = gr.Dataframe(
                    headers=["Component", "MiB", "Share of predicted peak (%)"],
                    datatype=["str", "number", "number"],
                    interactive=False,
                    label="Training component breakdown",
                )
                training_recommendation = gr.Markdown()
                training_activation = gr.Dataframe(
                    headers=["Activation detail", "MiB"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="Activation details",
                )
                training_notes = gr.Textbox(
                    label="Warnings and savings hints", lines=7, interactive=False
                )
                training_error = gr.Textbox(label="Request message", interactive=False)
                training_inputs = [
                    training_model,
                    training_gpu,
                    training_batch,
                    training_seq,
                    training_quant,
                    training_double,
                    training_precision,
                    training_rank,
                    training_targets,
                    training_full,
                    training_optimizer,
                    training_optimizer_dtype,
                    training_checkpoint,
                    training_flash,
                    training_accum,
                    training_vram,
                ]
                training_run.click(
                    fn=training_callback,
                    inputs=training_inputs,
                    outputs=[
                        training_status,
                        training_summary,
                        training_components,
                        training_recommendation,
                        training_notes,
                        training_activation,
                        training_error,
                    ],
                    concurrency_id="fitcheck-estimates",
                    concurrency_limit=4,
                )

            with gr.Tab("Serving"):
                with gr.Row():
                    serving_model = gr.Textbox(
                        label="Hugging Face model ID", value=_MODEL_DEFAULT
                    )
                    serving_gpu = gr.Dropdown(
                        label="GPU preset",
                        choices=_GPU_CHOICES,
                        value="4090",
                        allow_custom_value=False,
                    )
                with gr.Row():
                    serving_seq = gr.Number(
                        label="Sequence length (tokens)",
                        value=2048,
                        precision=0,
                        minimum=1,
                    )
                    serving_concurrency = gr.Number(
                        label="Concurrent requests", value=1, precision=0, minimum=1
                    )
                with gr.Accordion("Advanced serving settings", open=False):
                    with gr.Row():
                        serving_quant = gr.Dropdown(
                            ["none", "nf4", "int8"],
                            value="none",
                            label="Base storage format",
                        )
                        serving_double = gr.Checkbox(
                            label="NF4 double quantization", value=False
                        )
                        serving_precision = gr.Dropdown(
                            ["fp32", "fp16", "bf16"],
                            value="fp16",
                            label="Weight precision",
                        )
                    serving_vram = gr.Number(
                        label="Custom total VRAM (MiB), optional",
                        value=None,
                        precision=0,
                        minimum=1,
                        info=(
                            "Overrides the selected GPU preset; usable capacity is 95% "
                            "of this total."
                        ),
                    )
                serving_run = gr.Button("Estimate serving memory", variant="primary")
                serving_status = gr.HTML(value=_EMPTY_STATUS)
                serving_summary = gr.Markdown()
                serving_components = gr.Dataframe(
                    headers=["Resident component", "MiB"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="Serving memory breakdown",
                )
                serving_warnings = gr.Textbox(
                    label="Warnings", lines=4, interactive=False
                )
                serving_error = gr.Textbox(label="Request message", interactive=False)
                serving_run.click(
                    fn=serving_callback,
                    inputs=[
                        serving_model,
                        serving_gpu,
                        serving_seq,
                        serving_concurrency,
                        serving_quant,
                        serving_double,
                        serving_precision,
                        serving_vram,
                    ],
                    outputs=[
                        serving_status,
                        serving_summary,
                        serving_components,
                        serving_warnings,
                        serving_error,
                    ],
                    concurrency_id="fitcheck-estimates",
                    concurrency_limit=4,
                )

            with gr.Tab("Advisor"):
                with gr.Row():
                    advisor_model = gr.Textbox(
                        label="Hugging Face model ID", value=_MODEL_DEFAULT
                    )
                    advisor_gpu = gr.Dropdown(
                        label="GPU preset",
                        choices=_GPU_CHOICES,
                        value="4090",
                        allow_custom_value=False,
                    )
                with gr.Row():
                    advisor_batches = gr.CheckboxGroup(
                        choices=list(DEFAULT_BATCH_SIZES),
                        value=list(DEFAULT_BATCH_SIZES),
                        label="Micro-batch sizes",
                    )
                    advisor_sequences = gr.CheckboxGroup(
                        choices=[512, 1024, 2048, 4096, 8192],
                        value=[1024, 2048],
                        label="Sequence lengths",
                    )
                    advisor_ranks = gr.CheckboxGroup(
                        choices=list(DEFAULT_LORA_RANKS),
                        value=list(DEFAULT_LORA_RANKS),
                        label="LoRA ranks",
                    )
                advisor_context = gr.Number(
                    label="Known maximum context length (optional)",
                    value=None,
                    precision=0,
                    minimum=1,
                )
                with gr.Accordion("Advanced sweep settings", open=False):
                    with gr.Row():
                        advisor_quant = gr.Dropdown(
                            ["none", "nf4", "int8"],
                            value="nf4",
                            label="Base storage format",
                        )
                        advisor_double = gr.Checkbox(
                            label="NF4 double quantization", value=False
                        )
                        advisor_precision = gr.Dropdown(
                            ["fp32", "fp16", "bf16"],
                            value="bf16",
                            label="Compute precision",
                        )
                    with gr.Row():
                        advisor_targets = gr.Dropdown(
                            ["minimal", "standard", "full"],
                            value="standard",
                            label="LoRA target preset",
                        )
                        advisor_optimizer = gr.Dropdown(
                            ["adamw", "adam8bit", "sgd", "sgd-momentum"],
                            value="adamw",
                            label="Optimizer",
                        )
                        advisor_optimizer_dtype = gr.Dropdown(
                            ["fp32", "bf16"],
                            value="fp32",
                            label="Optimizer state dtype",
                        )
                    with gr.Row():
                        advisor_checkpoint = gr.Checkbox(
                            label="Gradient checkpointing", value=True
                        )
                        advisor_flash = gr.Checkbox(
                            label="Flash Attention", value=False
                        )
                    advisor_vram = gr.Number(
                        label="Custom total VRAM (MiB), optional",
                        value=None,
                        precision=0,
                        minimum=1,
                        info=(
                            "Overrides the selected GPU preset; usable capacity is 95% "
                            "of this total."
                        ),
                    )
                advisor_run = gr.Button("Run advisor sweep", variant="primary")
                advisor_status = gr.HTML(value=_EMPTY_STATUS)
                advisor_summary = gr.Markdown()
                advisor_frontier = gr.Dataframe(
                    headers=[
                        "Tokens/step",
                        "Batch",
                        "Sequence",
                        "LoRA rank",
                        "Peak MiB",
                        "Runnable CLI command",
                    ],
                    datatype=["number", "number", "number", "number", "number", "str"],
                    interactive=False,
                    label="Frontier",
                )
                with gr.Row():
                    advisor_ceilings = gr.Dataframe(
                        headers=["Axis", "Exact ceiling", "Peak MiB at ceiling"],
                        datatype=["str", "number", "number"],
                        interactive=False,
                        label="Exact ceilings",
                    )
                    advisor_prices = gr.Dataframe(
                        headers=["Axis", "From", "To", "Memory change (MiB)"],
                        datatype=["str", "number", "number", "number"],
                        interactive=False,
                        label="Per-axis prices",
                    )
                advisor_note = gr.Markdown(
                    "Tokens/step is work per optimizer step, not throughput."
                )
                advisor_error = gr.Textbox(label="Request message", interactive=False)
                advisor_run.click(
                    fn=advisor_callback,
                    inputs=[
                        advisor_model,
                        advisor_gpu,
                        advisor_batches,
                        advisor_sequences,
                        advisor_ranks,
                        advisor_context,
                        advisor_quant,
                        advisor_double,
                        advisor_precision,
                        advisor_targets,
                        advisor_optimizer,
                        advisor_optimizer_dtype,
                        advisor_checkpoint,
                        advisor_flash,
                        advisor_vram,
                    ],
                    outputs=[
                        advisor_status,
                        advisor_summary,
                        advisor_frontier,
                        advisor_ceilings,
                        advisor_prices,
                        advisor_note,
                        advisor_error,
                    ],
                    concurrency_id="fitcheck-estimates",
                    concurrency_limit=4,
                )

    demo.queue(default_concurrency_limit=4)
    return demo


demo = create_demo()


if __name__ == "__main__":
    demo.launch(analytics_enabled=False)
