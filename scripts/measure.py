from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

MIB = 1024.0**2

_PRECISIONS = ("fp32", "fp16", "bf16")
_QUANTIZATIONS = ("none", "nf4", "int8")
_OPTIMIZERS = ("adamw", "adam8bit", "sgd", "sgd-momentum")
_TARGET_PRESETS = {
    "minimal": ("q_proj", "v_proj"),
    "standard": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "full": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}
_ALL_TARGETS = _TARGET_PRESETS["full"]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="measure.py",
        description="Measure real peak VRAM for one LoRA/QLoRA training step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The markdown row this prints is meant to be pasted straight into the\n"
            "validation matrix in README.md (task 6.2)."
        ),
    )
    parser.add_argument(
        "model_id", help="Hugging Face model ID, e.g. mistralai/Mistral-7B-v0.3"
    )

    parser.add_argument(
        "--quant",
        choices=_QUANTIZATIONS,
        default=None,
        help="BASE MODEL storage format. (default: none)",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        default=None,
        help="NF4 double quantization.",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Shorthand for --quant nf4 --precision bf16 --grad-checkpoint.",
    )
    parser.add_argument(
        "--precision",
        choices=_PRECISIONS,
        default=None,
        help="COMPUTE dtype: LoRA weights, gradients, activations. (default: bf16)",
    )

    parser.add_argument(
        "--lora-r", type=int, default=None, help="LoRA rank. (default: 16)"
    )
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning.")
    parser.add_argument(
        "--lora-targets",
        default="standard",
        help="Preset (minimal|standard|full) or comma-separated modules.",
    )

    parser.add_argument(
        "--batch-size", type=int, default=1, help="MICRO-batch size. (default: 1)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length. (default: 2048)"
    )
    parser.add_argument("--optimizer", choices=_OPTIMIZERS, default="adamw")
    parser.add_argument("--optimizer-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        default=None,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--flash-attn", action="store_true", help="Enable Flash Attention 2."
    )

    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU key from fitcheck's database, for the predicted verdict.",
    )
    parser.add_argument(
        "--attn-impl",
        default=None,
        help=(
            "Override the attention implementation passed to transformers. "
            "Default: flash_attention_2 with --flash-attn, else eager."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help=(
            "Steps to run before measuring. Must be >=1: optimizer states are "
            "allocated lazily on the first .step(). (default: 1)"
        ),
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=1,
        help="Steps to run under measurement. (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the result as JSON instead of a human report.",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Skip the fitcheck prediction; measure only.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args(argv)
    _apply_qlora_shorthand(args)
    _validate_args(parser, args)
    return args


def _apply_qlora_shorthand(args: argparse.Namespace) -> None:
    if args.qlora:
        if args.quant is None:
            args.quant = "nf4"
        if args.precision is None:
            args.precision = "bf16"
        if args.grad_checkpoint is None:
            args.grad_checkpoint = True

    args.quant = args.quant or "none"
    args.precision = args.precision or "bf16"
    args.double_quant = bool(args.double_quant)
    args.grad_checkpoint = bool(args.grad_checkpoint)
    args.lora_rank = (
        None if args.no_lora else (args.lora_r if args.lora_r is not None else 16)
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.no_lora and args.quant != "none":
        parser.error(
            f"--quant {args.quant} with --no-lora is not modelled by fitcheck; "
            "you cannot backprop into frozen quantized weights."
        )
    if args.no_lora and args.lora_r is not None:
        parser.error("--no-lora conflicts with --lora-r.")
    if args.double_quant and args.quant != "nf4":
        parser.error("--double-quant applies only to --quant nf4.")
    if args.optimizer_dtype != "fp32" and args.optimizer != "adamw":
        parser.error("--optimizer-dtype applies only to --optimizer adamw.")
    for name in ("batch_size", "seq_len", "measure_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.warmup_steps < 1:
        parser.error(
            "--warmup-steps must be >= 1: optimizer states are allocated lazily on "
            "the first .step(), so a zero-warmup peak understates S_optim."
        )
    if args.lora_rank is not None and args.lora_rank < 1:
        parser.error("--lora-r must be >= 1")


def parse_lora_targets(value: str) -> list[str]:
    preset = _TARGET_PRESETS.get(value.strip().casefold())
    if preset is not None:
        return list(preset)

    targets: list[str] = []
    for token in value.split(","):
        name = token.strip().casefold()
        if not name:
            continue
        module = name if name.endswith("_proj") else f"{name}_proj"
        if module not in _ALL_TARGETS:
            raise SystemExit(
                f"measure.py: unknown LoRA target '{token.strip()}'. "
                f"Presets: {', '.join(_TARGET_PRESETS)}. "
                f"Modules: {', '.join(_ALL_TARGETS)}."
            )
        if module not in targets:
            targets.append(module)
    if not targets:
        raise SystemExit("measure.py: at least one LoRA target module is required")
    return targets


# ---------------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------------


def _torch_dtype(precision: str):
    import torch

    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def _attn_implementation(args: argparse.Namespace) -> str:
    """Resolve the attention kernel.

    Without --flash-attn this deliberately returns "eager", not the transformers
    default of "sdpa". SDPA silently selects a memory-efficient kernel that never
    materializes the b*n_h*s^2 score matrix -- precisely the term fitcheck adds when
    Flash Attention is off. Measuring "no flash" under SDPA would compare fitcheck's
    eager formula against a flash-like kernel and report a large phantom error.
    """
    if args.attn_impl:
        return args.attn_impl
    return "flash_attention_2" if args.flash_attn else "eager"


def build_model(args: argparse.Namespace, targets: list[str]):
    from transformers import AutoConfig, AutoModelForCausalLM

    compute_dtype = _torch_dtype(args.precision)
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "attn_implementation": _attn_implementation(args),
        "trust_remote_code": args.trust_remote_code,
    }

    if args.quant != "none":
        from transformers import BitsAndBytesConfig

        if args.quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    if args.quant == "none":
        model = model.cuda()

    model.config.use_cache = False

    if args.lora_rank is not None:
        from peft import LoraConfig, get_peft_model

        if args.quant != "none":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.grad_checkpoint
            )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    hf_config = AutoConfig.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code
    )
    return model, hf_config


def build_optimizer(model, args: argparse.Namespace):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("measure.py: no trainable parameters -- nothing to measure")

    if args.optimizer == "adamw":
        if args.optimizer_dtype == "fp32":
            for p in params:
                if p.dtype != torch.float32:
                    p.data = p.data.float()
        return torch.optim.AdamW(params, lr=1e-4)
    if args.optimizer == "adam8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=1e-4)
    momentum = 0.9 if args.optimizer == "sgd-momentum" else 0.0
    return torch.optim.SGD(params, lr=1e-4, momentum=momentum)


def observed_optimizer_state_bytes(optimizer) -> tuple[float, str]:
    import torch

    total = 0
    dtypes: set[str] = set()
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.numel() > 1:
                total += value.numel() * value.element_size()
                dtypes.add(str(value.dtype).replace("torch.", ""))
    return total / MIB, ("+".join(sorted(dtypes)) if dtypes else "none")


def observed_gradient_bytes(model) -> float:
    """Resident gradient bytes for the trainable parameters.

    Must be read after .backward() and before zero_grad(set_to_none=True), which
    drops .grad entirely. Together with the optimizer-state and after-load figures
    this is what lets A_act be measured by subtraction instead of inferred by hand.
    """
    total = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            total += param.grad.numel() * param.grad.element_size()
    return total / MIB


# ---------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------


@dataclass
class Measurement:
    cuda_context_mib: float
    cuda_context_init_mib: float
    after_load_allocated_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    process_mib: float
    peak_forward_mib: float
    peak_backward_mib: float
    peak_optimizer_mib: float
    optimizer_state_mib: float
    optimizer_state_dtype: str
    gradient_mib: float
    sdpa_backend: str
    trainable_params: int
    total_params: int
    step_seconds: float
    weight_breakdown: dict[str, float]


def _is_4bit(param) -> bool:
    if getattr(param, "quant_state", None) is not None:
        return True
    return any(cls.__name__ == "Params4bit" for cls in type(param).__mro__)


def _logical_param_count(model) -> int:
    total = 0
    for p in model.parameters():
        if _is_4bit(p):
            state = getattr(p, "quant_state", None)
            shape = getattr(state, "shape", None)
            if shape is not None:
                n = 1
                for dim in shape:
                    n *= dim
                total += n
            else:
                total += p.numel() * 2
        else:
            total += p.numel()
    return total


def resident_weight_breakdown(model) -> dict[str, float]:
    buckets = {"nf4_packed": 0, "quant_scales": 0, "unquantized_fp32": 0, "other": 0}
    for p in model.parameters():
        nbytes = p.numel() * p.element_size()
        if _is_4bit(p):
            buckets["nf4_packed"] += nbytes
            state = getattr(p, "quant_state", None)
            for attr in ("absmax", "code", "offset", "state2"):
                value = getattr(state, attr, None)
                inner = getattr(value, "absmax", None)
                if inner is not None:
                    buckets["quant_scales"] += inner.numel() * inner.element_size()
                if hasattr(value, "numel"):
                    buckets["quant_scales"] += value.numel() * value.element_size()
        elif p.dtype.is_floating_point and p.element_size() == 4:
            buckets["unquantized_fp32"] += nbytes
        else:
            buckets["other"] += nbytes
    return {k: v / MIB for k, v in buckets.items()}


def _cuda_context_mib() -> float:
    """Device memory held by the process but not by the caching allocator.

    Read this AFTER the measured steps, not straight after torch.cuda.init().
    cuBLAS/cuDNN workspaces and lazily-loaded kernel images land during the first
    real matmul, so an init-time reading understates the true footprint and makes
    the process tier look better than it is.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / MIB


def _install_gqa_sdpa_shim() -> None:
    """Expand grouped-query K/V before SDPA, so the fused kernels accept them.

    With no attention_mask, transformers hands SDPA un-expanded K/V plus
    enable_gqa=True. The memory-efficient backend refuses that ("both fused kernels
    require query, key and value to have the same num_heads"), flash needs sm_80 and
    cuDNN attention is off -- so every GQA model dies with "No available kernel".

    Expanding by hand is what the eager path already does via repeat_kv, so the
    eager-vs-SDPA contrast still differs only by the score matrix, which is the whole
    point of the comparison. Idempotent, and only installed for --attn-impl sdpa.
    """
    import torch

    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    if getattr(original, "_fitcheck_gqa_shim", False):
        return

    def patched(query, key, value, *args, **kwargs):
        if key.size(-3) != query.size(-3):
            groups = query.size(-3) // key.size(-3)
            key = key.repeat_interleave(groups, dim=-3)
            value = value.repeat_interleave(groups, dim=-3)
            kwargs.pop("enable_gqa", None)
        return original(query, key, value, *args, **kwargs)

    patched._fitcheck_gqa_shim = True
    functional.scaled_dot_product_attention = patched


def _sdpa_backend(args: argparse.Namespace):
    """Pin SDPA to its memory-efficient backend, and say which one is in use.

    Plain "sdpa" is free to fall back to the *math* backend, which does materialize
    the (b, n_h, s, s) score matrix. That would silently void the one comparison
    --attn-impl sdpa exists for, so pin it and fail loudly instead.

    Returns a (factory, label) pair; the factory makes a fresh context manager.
    """
    import contextlib

    if _attn_implementation(args) != "sdpa":
        return (lambda: contextlib.nullcontext()), "n/a (not sdpa)"

    _install_gqa_sdpa_shim()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return (lambda: contextlib.nullcontext()), "unpinned (torch.nn.attention missing)"

    return (
        lambda: sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    ), "EFFICIENT_ATTENTION (+ GQA expand shim)"


def _preflight_device(args: argparse.Namespace) -> None:
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    capability = major * 10 + minor
    name = torch.cuda.get_device_name(0)

    if args.precision == "bf16" and capability < 80:
        raise SystemExit(
            f"measure.py: {name} is sm_{capability} and has no native bf16 "
            f"(that needs sm_80 / Ampere or newer).\n"
            f"  Re-run with --precision fp16. On a T4 that is the correct compute "
            f"dtype, and fitcheck models fp16 and bf16 identically at 2 bytes, so "
            f"the prediction is unchanged.\n"
            f"  Note --qlora implies bf16, so pass it as: --qlora --precision fp16"
        )

    if args.flash_attn and capability < 80:
        if not args.attn_impl:
            raise SystemExit(
                f"measure.py: {name} is sm_{capability}; Flash Attention 2 needs "
                f"sm_80 or newer.\n"
                f"  Either drop --flash-attn, or pass --attn-impl sdpa. SDPA's "
                f"memory-efficient kernel also never materializes the b*n_h*s^2 "
                f"score matrix, so it measures the same thing fitcheck's flash_attn "
                f"branch predicts, and it runs on sm_75."
            )
        print(
            f"measure.py: {name} is sm_{capability}, so Flash Attention 2 is "
            f"unavailable. --flash-attn is being applied to the PREDICTION only; "
            f"the measured kernel is '{args.attn_impl}'."
        )

    visible = torch.cuda.device_count()
    if visible > 1:
        print(
            f"measure.py: {visible} GPUs visible; measuring device 0 ({name}) only. "
            f"This is by design -- fitcheck predicts single-device memory, so a "
            f"sharded or DDP run would not be comparable to the prediction."
        )


def measure(args: argparse.Namespace, targets: list[str]) -> Measurement:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "measure.py: no CUDA device visible -- this harness needs a real GPU"
        )

    _preflight_device(args)

    torch.manual_seed(args.seed)
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    context_init_mib = _cuda_context_mib()
    sdpa_ctx, sdpa_label = _sdpa_backend(args)

    model, hf_config = build_model(args, targets)
    torch.cuda.synchronize()
    after_load_mib = torch.cuda.memory_allocated() / MIB

    weight_breakdown = resident_weight_breakdown(model)

    optimizer = build_optimizer(model, args)
    total_params = _logical_param_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    vocab_size = getattr(hf_config, "vocab_size", 32000)
    batch = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device="cuda", dtype=torch.long
    )

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        optimizer.step()

    def phase_resolved_step() -> tuple[float, float, float, float]:
        """One step with the peak counter reset between phases.

        reset_peak_memory_stats() rebases the peak to whatever is currently live,
        so each phase peak includes the resident baseline -- which is what makes the
        three numbers comparable to one another and to peak_allocated.
        """
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        out = model(input_ids=batch, labels=batch)
        torch.cuda.synchronize()
        forward_peak = torch.cuda.max_memory_allocated() / MIB

        torch.cuda.reset_peak_memory_stats()
        out.loss.backward()
        torch.cuda.synchronize()
        backward_peak = torch.cuda.max_memory_allocated() / MIB
        grad_mib = observed_gradient_bytes(model)

        torch.cuda.reset_peak_memory_stats()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_peak = torch.cuda.max_memory_allocated() / MIB
        return forward_peak, backward_peak, optimizer_peak, grad_mib

    with sdpa_ctx():
        try:
            for _ in range(args.warmup_steps):
                one_step()
        except RuntimeError as exc:
            if "No available kernel" not in str(exc):
                raise
            raise SystemExit(
                f"measure.py: SDPA has no usable kernel for this model on "
                f"{torch.cuda.get_device_name(0)} (backend pinned to {sdpa_label}).\n"
                f"  The warnings printed above say which kernel was rejected and why.\n"
                f"  Drop --attn-impl sdpa to measure the eager path instead; the "
                f"resulting row is still valid, it just does not isolate the "
                f"b*n_h*s^2 score matrix."
            ) from exc
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.measure_steps):
            one_step()
        torch.cuda.synchronize()
        step_seconds = (time.perf_counter() - started) / args.measure_steps

        # Read the headline peaks before the extra instrumented step, so these stay
        # exactly what they were before phase splitting was added.
        peak_allocated = torch.cuda.max_memory_allocated() / MIB
        peak_reserved = torch.cuda.max_memory_reserved() / MIB

        forward_peak, backward_peak, optimizer_peak, grad_mib = phase_resolved_step()

    opt_mib, opt_dtype = observed_optimizer_state_bytes(optimizer)
    context_mib = _cuda_context_mib()

    return Measurement(
        cuda_context_mib=context_mib,
        cuda_context_init_mib=context_init_mib,
        after_load_allocated_mib=after_load_mib,
        peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,
        process_mib=peak_reserved + context_mib,
        peak_forward_mib=forward_peak,
        peak_backward_mib=backward_peak,
        peak_optimizer_mib=optimizer_peak,
        optimizer_state_mib=opt_mib,
        optimizer_state_dtype=opt_dtype,
        gradient_mib=grad_mib,
        sdpa_backend=sdpa_label,
        trainable_params=trainable,
        total_params=total_params,
        step_seconds=step_seconds,
        weight_breakdown=weight_breakdown,
    )


# ---------------------------------------------------------------------------------
# Prediction (imports fitcheck, which never imports torch -- the dependency is one-way)
# ---------------------------------------------------------------------------------


def predict(args: argparse.Namespace, targets: list[str]):
    from fitcheck.config_parser import fetch_model_config
    from fitcheck.estimator import TrainingConfig, estimate
    from fitcheck.gpu_db import get_gpu

    training = TrainingConfig(
        precision=args.precision,
        quantization=args.quant,
        double_quant=args.double_quant,
        optimizer=args.optimizer,
        optimizer_dtype=args.optimizer_dtype,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lora_rank=args.lora_rank,
        lora_targets=targets,
        grad_checkpoint=args.grad_checkpoint,
        flash_attn=args.flash_attn,
    )
    model_config = fetch_model_config(args.model_id)
    return model_config, estimate(model_config, training, get_gpu(args.gpu))


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------


def _error_pct(predicted: float, actual: float) -> float:
    if actual <= 0:
        return float("nan")
    return (predicted - actual) / actual * 100.0


def _measured_activation_mib(m: Measurement) -> float:
    """A_act by subtraction: everything in the peak that is not weights or state.

    Weights, optimizer states and gradients are all measured directly, so whatever
    is left in max_memory_allocated() is the activation memory.
    """
    return (
        m.peak_allocated_mib
        - m.after_load_allocated_mib
        - m.optimizer_state_mib
        - m.gradient_mib
    )


def config_label(args: argparse.Namespace, targets: list[str]) -> str:
    if args.lora_rank is None:
        bits = ["full FT"]
    else:
        short = ",".join(t.replace("_proj", "") for t in targets)
        prefix = "QLoRA" if args.quant == "nf4" else "LoRA"
        bits = [f"{prefix} r={args.lora_rank} [{short}]"]
    bits.append(f"bs={args.batch_size}")
    bits.append(f"seq={args.seq_len}")
    bits.append(args.precision)
    bits.append(args.optimizer)
    if args.grad_checkpoint:
        bits.append("ckpt")
    if args.attn_impl:
        # The kernel was overridden, so name it rather than claiming FA2. A row
        # labelled "FA2" that actually ran under SDPA would be a mislabelled result.
        bits.append(f"attn={args.attn_impl}")
    else:
        bits.append("FA2" if args.flash_attn else "no FA")
    return ", ".join(bits)


def markdown_row(
    args: argparse.Namespace,
    targets: list[str],
    m: Measurement,
    report,
    gpu_name: str,
) -> str:
    tensors_predicted = report.total_mib - report.overhead_mib
    err = _error_pct(tensors_predicted, m.peak_allocated_mib)
    return (
        f"| {args.model_id} | {gpu_name} | {config_label(args, targets)} "
        f"| {tensors_predicted:,.0f} | {m.peak_allocated_mib:,.0f} | {err:+.1f}% |"
    )


def render(
    args: argparse.Namespace,
    targets: list[str],
    m: Measurement,
    model_config,
    report,
    gpu_name: str,
) -> str:
    import torch

    lines: list[str] = []
    add = lines.append

    add("")
    add("=" * 78)
    add(f"  {args.model_id}  on  {torch.cuda.get_device_name(0)}")
    add(f"  {config_label(args, targets)}")
    add("=" * 78)
    add("")
    add(
        f"  torch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"python {platform.python_version()} | {platform.system()}"
    )
    add(
        f"  attention: {_attn_implementation(args)}   "
        f"warmup: {args.warmup_steps}   measured steps: {args.measure_steps}"
    )
    add(f"  sdpa backend: {m.sdpa_backend}")
    add(f"  params: {m.total_params:,} logical | {m.trainable_params:,} trainable")
    if model_config is not None:
        base = m.total_params - m.trainable_params
        delta = base - model_config.num_params
        flag = "OK" if abs(delta) <= max(1, model_config.num_params // 1000) else "MISMATCH"
        add(f"          base {base:,} vs fitcheck P {model_config.num_params:,}"
            f"  ({delta:+,})  [{flag}]")
        if flag == "MISMATCH":
            add("          ^ config_parser's param count disagrees with the loaded "
                "model; every per-param term is built on P, so fix this first.")
    add(
        f"  optimizer states: {m.optimizer_state_mib:,.0f} MiB observed, "
        f"dtype {m.optimizer_state_dtype}"
    )
    add(f"  step time: {m.step_seconds:.2f}s")
    add("")
    add("  MEASURED")
    add(f"    CUDA context (at peak)           {m.cuda_context_mib:>12,.0f} MiB")
    add(f"    CUDA context (at init, for ref)  {m.cuda_context_init_mib:>12,.0f} MiB")
    add(f"    allocated after load             {m.after_load_allocated_mib:>12,.0f} MiB")
    add(f"    gradients (after backward)       {m.gradient_mib:>12,.0f} MiB")
    add(f"    peak allocated  (tensor bytes)   {m.peak_allocated_mib:>12,.0f} MiB")
    add(f"    peak reserved   (allocator pool) {m.peak_reserved_mib:>12,.0f} MiB")
    add(f"    process total   (reserved+ctx)   {m.process_mib:>12,.0f} MiB")
    add("")
    add("  PEAK BY PHASE  (which part of the step actually is the peak)")
    add(f"    forward   (logits + loss)        {m.peak_forward_mib:>12,.0f} MiB")
    add(f"    backward  (recompute + attn)     {m.peak_backward_mib:>12,.0f} MiB")
    add(f"    optimizer step                   {m.peak_optimizer_mib:>12,.0f} MiB")
    add("")
    add("    fitcheck SUMS the forward and backward humps into one A_act. If the peak")
    add("    moves between phases as seq_len grows, they do not coexist and summing")
    add("    them is the wrong shape, not just the wrong coefficient.")
    add("")

    if report is None:
        add("  PREDICTED: skipped (--no-predict)")
        add("")
        return "\n".join(lines)

    tensors_predicted = report.total_mib - report.overhead_mib
    allocator_predicted = report.total_mib - 500.0

    add("  PREDICTED (fitcheck)")
    add(f"    W_base   base model weights      {report.weight_mib:>12,.0f} MiB")
    add(f"    W_lora   adapter weights         {report.lora_mib:>12,.0f} MiB")
    add(f"    S_optim  optimizer states        {report.optimizer_mib:>12,.0f} MiB")
    add(f"    G_grad   gradients               {report.gradient_mib:>12,.0f} MiB")
    add(f"    A_act    activations             {report.activation_mib:>12,.0f} MiB")
    add(f"    C_over   context + fragmentation {report.overhead_mib:>12,.0f} MiB")
    add(f"    {'TOTAL':<32}{report.total_mib:>12,.0f} MiB")
    add("")
    add("  PREDICTED vs MEASURED  (each tier compares like with like)")
    add(f"    {'tier':<12}{'predicted':>12}{'measured':>12}{'error':>10}")
    for tier, predicted, actual in (
        ("tensors", tensors_predicted, m.peak_allocated_mib),
        ("allocator", allocator_predicted, m.peak_reserved_mib),
        ("process", report.total_mib, m.process_mib),
    ):
        add(
            f"    {tier:<12}{predicted:>12,.0f}{actual:>12,.0f}"
            f"{_error_pct(predicted, actual):>+9.1f}%"
        )
    add("")
    add("    tensors   = six-component formula minus C_overhead, vs max_memory_allocated()")
    add("    allocator = formula minus the 500 MiB context, vs max_memory_reserved()")
    add("    process   = the full fitcheck total, vs what nvidia-smi would show")
    add("")

    add("  COMPONENT SPOT-CHECKS  (isolates which term is wrong when a tier is off)")
    for label, predicted, actual in (
        (
            "weights (W_base + W_lora)",
            report.weight_mib + report.lora_mib,
            m.after_load_allocated_mib,
        ),
        ("optimizer states (S_optim)", report.optimizer_mib, m.optimizer_state_mib),
        ("gradients (G_grad)", report.gradient_mib, m.gradient_mib),
        ("activations (A_act)", report.activation_mib, _measured_activation_mib(m)),
    ):
        add(
            f"    {label:<28}{predicted:>10,.0f}{actual:>12,.0f}"
            f"{_error_pct(predicted, actual):>+9.1f}%"
        )
    add("")
    add("    A_act measured = peak allocated - after load - S_optim - G_grad. This is")
    add("    the subtraction that used to be done by hand; it is the row to read first")
    add("    when a tier is off, because every other component is measured directly.")
    add("")
    add("  RESIDENT WEIGHT BYTES BY STORAGE  (fitcheck bills every param one rate)")
    for key, label in (
        ("nf4_packed", "NF4 packed linears"),
        ("quant_scales", "quantization scales"),
        ("unquantized_fp32", "unquantized, held fp32"),
        ("other", "other"),
    ):
        add(f"    {label:<28}{m.weight_breakdown.get(key, 0.0):>22,.0f} MiB")
    add("")
    add("    embeddings and lm_head are NOT quantized by bitsandbytes, and peft's")
    add("    prepare_model_for_kbit_training upcasts them to fp32. Any large")
    add("    unquantized_fp32 figure is weight the W_base formula bills at 0.5 B/param.")
    add("")
    add("  MATRIX ROW  (paste into README.md)")
    add("")
    add("  " + markdown_row(args, targets, m, report, gpu_name))
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = parse_lora_targets(args.lora_targets)

    model_config = report = None
    gpu_name = args.gpu or "unknown"
    if not args.no_predict:
        from fitcheck.gpu_db import get_gpu

        model_config, report = predict(args, targets)
        gpu_name = get_gpu(args.gpu).name

    m = measure(args, targets)

    if args.as_json:
        payload: dict[str, Any] = {
            "model_id": args.model_id,
            "gpu": gpu_name,
            "config": config_label(args, targets),
            "measured": asdict(m),
        }
        if report is not None:
            payload["predicted"] = {
                "weight_mib": report.weight_mib,
                "lora_mib": report.lora_mib,
                "optimizer_mib": report.optimizer_mib,
                "gradient_mib": report.gradient_mib,
                "activation_mib": report.activation_mib,
                "overhead_mib": report.overhead_mib,
                "total_mib": report.total_mib,
            }
            payload["measured"]["activation_mib"] = _measured_activation_mib(m)
            payload["error_pct"] = {
                "tensors": _error_pct(
                    report.total_mib - report.overhead_mib, m.peak_allocated_mib
                ),
                "allocator": _error_pct(report.total_mib - 500.0, m.peak_reserved_mib),
                "process": _error_pct(report.total_mib, m.process_mib),
                "activation": _error_pct(
                    report.activation_mib, _measured_activation_mib(m)
                ),
            }
            payload["markdown_row"] = markdown_row(args, targets, m, report, gpu_name)
        print(json.dumps(payload, indent=2))
    else:
        print(render(args, targets, m, model_config, report, gpu_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
