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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="measure_infer.py",
        description=(
            "Measure real peak VRAM for one SERVING workload: resident weights plus a "
            "KV cache filled to --seq-len tokens for --num-concurrent sequences."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The inference twin of measure.py. fitcheck models serving as\n"
            "weights + KV cache + C_overhead, so the cache is measured twice here:\n"
            "directly, by walking the cache object, and by subtraction from the\n"
            "allocator. They should agree; a gap is something else staying resident."
        ),
    )
    parser.add_argument("model_id", help="Hugging Face model ID.")

    parser.add_argument(
        "--precision",
        choices=_PRECISIONS,
        default="fp16",
        help="COMPUTE dtype, and the KV cache dtype with it. (default: fp16)",
    )
    parser.add_argument(
        "--quant",
        choices=_QUANTIZATIONS,
        default="none",
        help="BASE MODEL storage format. (default: none)",
    )
    parser.add_argument(
        "--double-quant", action="store_true", help="NF4 double quantization."
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Cached tokens per sequence at the peak. (default: 2048)",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=1,
        help="Concurrent sequences sharing the cache. (default: 1)",
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=256,
        help=(
            "Tokens per prefill forward pass. Smaller keeps the transient logits "
            "tensor small, so the peak stays a serving peak rather than a prefill "
            "artefact. (default: 256)"
        ),
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=4,
        help=(
            "Single-token steps at the end, so the measured cache is one a real "
            "decoder actually appended to. (default: 4)"
        ),
    )

    parser.add_argument("--gpu", default=None, help="GPU key for the prediction.")
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Attention implementation. Default: whatever transformers picks.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on. Only 'cuda' produces a publishable row.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--no-predict", action="store_true", help="Skip the fitcheck prediction."
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.double_quant and args.quant != "nf4":
        parser.error("--double-quant applies only to --quant nf4.")
    for name in ("seq_len", "num_concurrent", "prefill_chunk"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.decode_steps < 1:
        parser.error(
            "--decode-steps must be >= 1: a cache that was only ever prefilled has "
            "never been grown by a decode step, which is the shape serving runs in."
        )
    if args.decode_steps >= args.seq_len:
        parser.error("--decode-steps must be < --seq-len")


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


def build_model(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModelForCausalLM

    compute_dtype = _torch_dtype(args.precision)
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl

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
        model = model.to(args.device)

    model.eval()
    model.config.use_cache = True
    hf_config = AutoConfig.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code
    )
    return model, hf_config


def _new_cache():
    """A growable KV cache, across the transformers versions that ship one.

    Returning None is a valid answer: older versions accept the legacy tuple cache,
    which the forward pass creates on its own and hands back.
    """
    try:
        from transformers import DynamicCache
    except ImportError:
        return None
    return DynamicCache()


def _iter_cache_tensors(cache: Any):
    """Every tensor reachable from the cache object.

    Walks the object graph rather than naming attributes, because the cache layout
    has been a list of tuples, then key_cache/value_cache lists, then a list of
    per-layer objects. The walk survives all three.
    """
    import torch

    seen: set[int] = set()
    stack: list[Any] = [cache]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if torch.is_tensor(obj):
            yield obj
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        attrs = getattr(obj, "__dict__", None)
        if attrs:
            stack.extend(attrs.values())


def cache_tensor_bytes(cache: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _iter_cache_tensors(cache))


def cached_token_count(cache: Any, fallback: int) -> int:
    method = getattr(cache, "get_seq_length", None)
    if callable(method):
        try:
            value = int(method())
        except Exception:
            value = 0
        if value > 0:
            return value
    return fallback


# ---------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------


@dataclass
class InferenceMeasurement:
    cuda_context_mib: float
    cuda_context_init_mib: float
    after_load_allocated_mib: float
    kv_cache_direct_mib: float
    kv_cache_by_subtraction_mib: float
    resident_after_fill_mib: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    process_mib: float
    peak_prefill_mib: float
    peak_decode_mib: float
    cached_tokens_per_sequence: int
    kv_dtype: str
    total_params: int
    prefill_seconds: float
    decode_seconds_per_token: float


def _cuda_context_mib(device: str) -> float:
    import torch

    if device != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / MIB


def _allocated_mib(device: str) -> float:
    import torch

    if device != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / MIB


def _peak_allocated_mib(device: str) -> float:
    import torch

    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / MIB


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


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


def _forward(model, input_ids, cache, past_len: int, device: str):
    """One forward pass that appends to `cache`, keeping the logits transient small.

    Only the last position's logits matter for serving, and the (b, s, V) tensor is
    otherwise the largest thing in the run -- it would hide the cache in the peak.
    """
    import torch

    batch, width = input_ids.shape
    attention_mask = torch.ones(
        (batch, past_len + width), dtype=torch.long, device=device
    )
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "cache_position": torch.arange(
            past_len, past_len + width, device=device, dtype=torch.long
        ),
    }
    if cache is not None:
        kwargs["past_key_values"] = cache

    for name in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(**kwargs, **{name: 1})
        except TypeError:
            continue
    return model(**kwargs)


def measure(args: argparse.Namespace) -> InferenceMeasurement:
    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "measure_infer.py: no CUDA device visible -- this harness needs a real GPU"
        )

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()
    _reset_peak(device)
    context_init_mib = _cuda_context_mib(device)

    model, hf_config = build_model(args)
    _sync(device)
    after_load_mib = _allocated_mib(device)
    total_params = _logical_param_count(model)

    vocab_size = getattr(hf_config, "vocab_size", 32000)
    batch = args.num_concurrent
    prefill_tokens = args.seq_len - args.decode_steps

    cache = _new_cache()

    _reset_peak(device)
    started = time.perf_counter()
    with torch.no_grad():
        past = 0
        while past < prefill_tokens:
            width = min(args.prefill_chunk, prefill_tokens - past)
            chunk = torch.randint(
                0, vocab_size, (batch, width), device=device, dtype=torch.long
            )
            out = _forward(model, chunk, cache, past, device)
            if cache is None:
                cache = out.past_key_values
            past += width
            del out
        _sync(device)
        prefill_seconds = time.perf_counter() - started
        prefill_peak = _peak_allocated_mib(device)

        _reset_peak(device)
        started = time.perf_counter()
        for _ in range(args.decode_steps):
            token = torch.randint(
                0, vocab_size, (batch, 1), device=device, dtype=torch.long
            )
            out = _forward(model, token, cache, past, device)
            if cache is None:
                cache = out.past_key_values
            past += 1
            del out
        _sync(device)
        decode_seconds = (time.perf_counter() - started) / args.decode_steps
        decode_peak = _peak_allocated_mib(device)

    kv_direct_mib = cache_tensor_bytes(cache) / MIB
    kv_dtype = "unknown"
    for tensor in _iter_cache_tensors(cache):
        kv_dtype = str(tensor.dtype).replace("torch.", "")
        break

    resident_mib = _allocated_mib(device)
    peak_reserved = torch.cuda.max_memory_reserved() / MIB if device == "cuda" else 0.0

    return InferenceMeasurement(
        cuda_context_mib=_cuda_context_mib(device),
        cuda_context_init_mib=context_init_mib,
        after_load_allocated_mib=after_load_mib,
        kv_cache_direct_mib=kv_direct_mib,
        kv_cache_by_subtraction_mib=resident_mib - after_load_mib,
        resident_after_fill_mib=resident_mib,
        peak_allocated_mib=max(prefill_peak, decode_peak),
        peak_reserved_mib=peak_reserved,
        process_mib=peak_reserved + _cuda_context_mib(device),
        peak_prefill_mib=prefill_peak,
        peak_decode_mib=decode_peak,
        cached_tokens_per_sequence=cached_token_count(cache, past),
        kv_dtype=kv_dtype,
        total_params=total_params,
        prefill_seconds=prefill_seconds,
        decode_seconds_per_token=decode_seconds,
    )


# ---------------------------------------------------------------------------------
# Prediction (imports fitcheck, which never imports torch -- the dependency is one-way)
# ---------------------------------------------------------------------------------


def predict(args: argparse.Namespace):
    from fitcheck.config_parser import fetch_model_config
    from fitcheck.estimator import ServingConfig, estimate_inference
    from fitcheck.gpu_db import get_gpu

    serving = ServingConfig(
        precision=args.precision,
        quantization=args.quant,
        double_quant=args.double_quant,
        seq_len=args.seq_len,
        num_concurrent=args.num_concurrent,
    )
    model_config = fetch_model_config(args.model_id)
    return model_config, estimate_inference(model_config, serving, get_gpu(args.gpu))


# ---------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------


def _error_pct(predicted: float, actual: float) -> float:
    if actual <= 0:
        return float("nan")
    return (predicted - actual) / actual * 100.0


def config_label(args: argparse.Namespace) -> str:
    bits = [f"serve seq={args.seq_len}", f"concurrent={args.num_concurrent}"]
    bits.append(args.precision)
    bits.append(f"quant={args.quant}" + (" +dq" if args.double_quant else ""))
    return ", ".join(bits)


def markdown_row(
    args: argparse.Namespace, m: InferenceMeasurement, report, gpu_name: str
) -> str:
    tensors_predicted = report.total_mib - report.overhead_mib
    err = _error_pct(tensors_predicted, m.peak_allocated_mib)
    return (
        f"| {args.model_id} | {gpu_name} | {config_label(args)} "
        f"| {tensors_predicted:,.0f} | {m.peak_allocated_mib:,.0f} | {err:+.1f}% |"
    )


def render(
    args: argparse.Namespace, m: InferenceMeasurement, model_config, report, gpu_name: str
) -> str:
    import torch

    lines: list[str] = []
    add = lines.append
    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else args.device

    add("")
    add("=" * 78)
    add(f"  {args.model_id}  serving on  {device_name}")
    add(f"  {config_label(args)}")
    add("=" * 78)
    add("")
    add(
        f"  torch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"python {platform.python_version()} | {platform.system()}"
    )
    add(f"  params: {m.total_params:,} logical")
    add(
        f"  cached tokens/sequence: {m.cached_tokens_per_sequence:,} "
        f"(asked for {args.seq_len:,})   kv dtype: {m.kv_dtype}"
    )
    add(
        f"  prefill: {m.prefill_seconds:.2f}s   "
        f"decode: {m.decode_seconds_per_token * 1000:.1f} ms/token"
    )
    add("")
    add("  MEASURED")
    add(f"    CUDA context (at peak)           {m.cuda_context_mib:>12,.0f} MiB")
    add(f"    allocated after load             {m.after_load_allocated_mib:>12,.0f} MiB")
    add(f"    KV cache, walked directly        {m.kv_cache_direct_mib:>12,.0f} MiB")
    add(f"    KV cache, by subtraction         {m.kv_cache_by_subtraction_mib:>12,.0f} MiB")
    add(f"    resident after fill              {m.resident_after_fill_mib:>12,.0f} MiB")
    add(f"    peak allocated  (tensor bytes)   {m.peak_allocated_mib:>12,.0f} MiB")
    add(f"    peak reserved   (allocator pool) {m.peak_reserved_mib:>12,.0f} MiB")
    add(f"    process total   (reserved+ctx)   {m.process_mib:>12,.0f} MiB")
    add("")
    add(f"    peak during prefill              {m.peak_prefill_mib:>12,.0f} MiB")
    add(f"    peak during decode               {m.peak_decode_mib:>12,.0f} MiB")
    add("")
    add("    The two KV numbers must agree. Direct walks the cache object; subtraction")
    add("    is resident-minus-weights. A gap is something else still resident.")
    add("")

    if report is None:
        add("  PREDICTED: skipped (--no-predict)")
        add("")
        return "\n".join(lines)

    add("  PREDICTED (fitcheck infer)")
    add(f"    W        resident weights        {report.weight_mib:>12,.0f} MiB")
    add(f"    KV       cache                   {report.kv_cache_mib:>12,.0f} MiB")
    add(f"    C_over   context + fragmentation {report.overhead_mib:>12,.0f} MiB")
    add(f"    {'TOTAL':<32}{report.total_mib:>12,.0f} MiB")
    add(f"    kv per request                   {report.kv_mib_per_request:>12,.1f} MiB")
    add(f"    kv per token                     {report.kv_mib_per_token:>12,.3f} MiB")
    add("")
    add("  PREDICTED vs MEASURED  (each tier compares like with like)")
    add(f"    {'tier':<12}{'predicted':>12}{'measured':>12}{'error':>10}")
    for tier, predicted, actual in (
        ("tensors", report.total_mib - report.overhead_mib, m.peak_allocated_mib),
        ("allocator", report.total_mib - 500.0, m.peak_reserved_mib),
        ("process", report.total_mib, m.process_mib),
    ):
        add(
            f"    {tier:<12}{predicted:>12,.0f}{actual:>12,.0f}"
            f"{_error_pct(predicted, actual):>+9.1f}%"
        )
    add("")
    add("  COMPONENT SPOT-CHECKS  (isolates which term is wrong when a tier is off)")
    for label, predicted, actual in (
        ("weights (W)", report.weight_mib, m.after_load_allocated_mib),
        ("KV cache", report.kv_cache_mib, m.kv_cache_direct_mib),
    ):
        add(
            f"    {label:<28}{predicted:>10,.0f}{actual:>12,.0f}"
            f"{_error_pct(predicted, actual):>+9.1f}%"
        )
    add("")
    add("  MATRIX ROW  (paste into README.md)")
    add("")
    add("  " + markdown_row(args, m, report, gpu_name))
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    model_config = report = None
    gpu_name = args.gpu or "unknown"
    if not args.no_predict:
        from fitcheck.gpu_db import get_gpu

        model_config, report = predict(args)
        gpu_name = get_gpu(args.gpu).name

    m = measure(args)

    if args.as_json:
        payload: dict[str, Any] = {
            "model_id": args.model_id,
            "gpu": gpu_name,
            "config": config_label(args),
            "measured": asdict(m),
        }
        if report is not None:
            payload["predicted"] = {
                "weight_mib": report.weight_mib,
                "kv_cache_mib": report.kv_cache_mib,
                "overhead_mib": report.overhead_mib,
                "total_mib": report.total_mib,
            }
            payload["error_pct"] = {
                "tensors": _error_pct(
                    report.total_mib - report.overhead_mib, m.peak_allocated_mib
                ),
                "process": _error_pct(report.total_mib, m.process_mib),
                "kv_cache": _error_pct(report.kv_cache_mib, m.kv_cache_direct_mib),
                "weights": _error_pct(report.weight_mib, m.after_load_allocated_mib),
            }
            payload["markdown_row"] = markdown_row(args, m, report, gpu_name)
        print(json.dumps(payload, indent=2))
    else:
        print(render(args, m, model_config, report, gpu_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
