from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MEASURE = Path(__file__).resolve().parent / "measure.py"

GRID: tuple[tuple[str, int, int], ...] = (
    ("HuggingFaceTB/SmolLM2-135M", 2, 512),
    ("HuggingFaceTB/SmolLM2-360M", 1, 4096),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 512),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 1024),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 2048),
    ("HuggingFaceTB/SmolLM2-135M", 1, 4096),
    ("HuggingFaceTB/SmolLM2-1.7B", 4, 1024),
    ("Qwen/Qwen2.5-1.5B-Instruct", 2, 1024),
)

REPEAT = ("HuggingFaceTB/SmolLM2-1.7B", 4, 1024)
REPEATS = 3

_ABORT_AFTER_FAILURES = 3


def _slug(model_id: str, batch_size: int, seq_len: int, kernel: str, tag: str) -> str:
    name = model_id.split("/")[-1].replace(".", "-")
    return f"{name}-bs{batch_size}-seq{seq_len}-{kernel}{tag}"


def _normalise_tag(tag: str) -> str:
    """`--tag nf4`, `--tag -nf4` and `--tag=-nf4` all mean the same thing.

    argparse reads a value starting with `-` as another flag, so `--tag -nf4` fails
    with "expected one argument" before this script ever runs. Accepting the bare word
    and adding the separator here is the difference between a flag that works and one
    that needs its own footnote.
    """
    trimmed = tag.strip().lstrip("-")
    return f"-{trimmed}" if trimmed else ""


_PROBE = """
import json, sys, traceback
try:
    import torch, transformers, peft
    try:
        import bitsandbytes
        bnb, bnb_error = bitsandbytes.__version__, None
    except BaseException as bnb_failure:
        bnb, bnb_error = None, "%s: %s" % (type(bnb_failure).__name__, bnb_failure)
    p = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bnb,
        "bitsandbytes_error": bnb_error,
        "cuda_available": torch.cuda.is_available(),
        "gpu": p.name,
        "capability": "sm_%d%d" % (p.major, p.minor),
        "total_mib": round(p.total_memory / 1024**2),
        "arch_list": torch.cuda.get_arch_list(),
    }))
except BaseException as error:
    traceback.print_exc()
    root = error
    while root.__cause__ or root.__context__:
        root = root.__cause__ or root.__context__
    print("ROOT CAUSE: %s: %s" % (type(root).__name__, root), file=sys.stderr)
    sys.exit(1)
"""

_GENERIC_ADVICE = (
    "The measurement stack could not be imported, so every row would fail the same\n"
    "way. On a managed GPU image (Kaggle, Colab) the stack is already installed --\n"
    "do NOT `pip install -U` over it. See scripts/requirements-measure.txt."
)

_TORCHAO_ADVICE = (
    "An OLD torchao is installed. peft raises on that, but skips torchao silently\n"
    "when it is absent -- and nothing in this project uses it. So REMOVE it:\n"
    "\n    pip uninstall -y torchao\n"
    "\nDo not upgrade it instead: torchao pulls a matching torch, and replacing a\n"
    "GPU image's torch breaks torchvision, torchaudio and the CUDA toolkit with it."
)

_TORCHVISION_MISMATCH_ADVICE = (
    "torchvision does not match torch. `operator torchvision::nms does not exist`\n"
    "means torchvision was compiled against a different torch than the one loaded,\n"
    "which happens when `pip install -U` replaces a GPU image's torch. transformers\n"
    "imports torchvision for image models, so every text model dies with it too.\n"
    "\nTHE REAL FIX is a clean image. On Kaggle a kernel restart is NOT enough --\n"
    "pip changes live in the container, so use Run > Factory reset (or stop the\n"
    "session from the sidebar and reopen), then install nothing but `pip install -e .`\n"
    "plus `pip uninstall -y torchao`.\n"
    "\nIf a clean image is genuinely unavailable, `pip uninstall -y torchvision\n"
    "torchaudio` also clears it: transformers skips them when they are absent, and\n"
    "nothing here measures an image or an audio model. Note in that case that the\n"
    "rows were taken on a different torch than the rest of the archive."
)

_BITSANDBYTES_MISSING_ADVICE = (
    "bitsandbytes is NOT installed, and `--quant nf4` / `--quant int8` load the base\n"
    "model through it. Kaggle and Colab ship torch, transformers and peft but not this\n"
    "one, so an unquantized grid runs on a bare image and a quantized grid cannot.\n"
    "\n    pip install bitsandbytes\n"
    "\nWithout `-U`. A plain install leaves the image's torch alone -- bitsandbytes\n"
    "only requires a torch, and the image already has one that satisfies it -- while\n"
    "`-U` upgrades torch itself and takes torchvision down with it (see the torchvision\n"
    "note above). No kernel restart is needed: every row runs in a fresh subprocess.\n"
    "\nOr take the unquantized grid instead, which needs nothing installed:\n"
    "\n    --quant none   (and drop --tag nf4)"
)

_BITSANDBYTES_BROKEN_ADVICE = (
    "bitsandbytes is installed but will not import, so every quantized row dies in the\n"
    "same place. The usual cause is a bitsandbytes built against a different torch or a\n"
    "different CUDA than this image's. Reinstall it against the torch that is here:\n"
    "\n    pip uninstall -y bitsandbytes && pip install bitsandbytes\n"
    "\nNever `pip install -U torch` to satisfy it -- that breaks the image, and then\n"
    "every row fails for a second, less obvious reason on top of this one.\n"
    "\nOr run `--quant none`, which does not import bitsandbytes at all."
)

_BROKEN_TORCH_ADVICE = (
    "torch looks broken or mismatched with this image. The usual cause is a\n"
    "`pip install -U` that replaced the image's CUDA build with a generic PyPI one;\n"
    "`Could not import module 'LlamaConfig'` is a symptom of it, not a transformers\n"
    "bug. pip cannot undo this reliably -- restart from a clean image (on Kaggle:\n"
    "Run > Factory reset; a kernel restart keeps the broken packages) and install\n"
    "nothing but `pip install -e .`."
)


def _diagnose(stderr: str) -> str:
    """Name the fix, not just the error.

    Each of these cost a real Kaggle session. A preflight that only says "something
    is wrong" is barely better than the twenty stack traces it replaced, and the
    bottom of an import chain like this one is fifty lines below the symptom.
    """

    if "requires bitsandbytes" in stderr or "No module named 'bitsandbytes'" in stderr:
        return _BITSANDBYTES_MISSING_ADVICE
    if "torchvision::nms" in stderr or ("torchvision" in stderr and "operator" in stderr):
        return _TORCHVISION_MISMATCH_ADVICE
    if "torchao" in stderr:
        return _TORCHAO_ADVICE
    if (
        "LlamaConfig" in stderr
        or "BloomPreTrainedModel" in stderr
        or "Torch not compiled with CUDA" in stderr
    ):
        return _BROKEN_TORCH_ADVICE
    return _GENERIC_ADVICE


def preflight(args: argparse.Namespace) -> str | None:
    """Check the measurement stack once, instead of failing the same way 20 times.

    Every dependency problem here is global: a missing library or an incompatible
    version kills every row identically. Finding that out on row 1 costs a second;
    finding it out on row 20 costs however long the models took to download.
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("PREFLIGHT FAILED -- not running the grid.\n")
        for line in result.stderr.strip().splitlines()[-3:]:
            print(f"  {line}")
        print(f"\n{_diagnose(result.stderr)}")
        return None

    info = json.loads(result.stdout)
    if not info["cuda_available"]:
        print("PREFLIGHT FAILED -- torch imports, but sees no CUDA device.\n")
        print(_BROKEN_TORCH_ADVICE)
        return None
    if args.quant != "none" and info["bitsandbytes"] is None:
        failure = info["bitsandbytes_error"] or ""
        print(
            f"PREFLIGHT FAILED -- `--quant {args.quant}` needs bitsandbytes and it "
            f"did not import.\n"
        )
        if failure:
            print(f"  {failure}\n")
        print(
            _BITSANDBYTES_MISSING_ADVICE
            if not failure or "No module named" in failure
            else _BITSANDBYTES_BROKEN_ADVICE
        )
        return None

    print(
        f"{info['gpu']} ({info['capability']}, {info['total_mib']:,} MiB) | "
        f"torch {info['torch']} | transformers {info['transformers']} | "
        f"peft {info['peft']} | "
        f"bitsandbytes {info['bitsandbytes'] or 'absent'}"
    )
    if info["capability"] not in info["arch_list"]:
        print(
            f"  WARNING: this torch build lists {info['arch_list']} and not "
            f"{info['capability']}. Kernels may fail to launch on this card."
        )
    return str(result.stdout)


def _row_args(
    model_id: str,
    batch_size: int,
    seq_len: int,
    kernel: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(MEASURE),
        model_id,
        "--gpu",
        args.gpu,
        "--quant",
        args.quant,
        "--precision",
        args.precision,
        "--lora-r",
        str(args.lora_r),
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--grad-checkpoint",
        "--json",
    ]
    if kernel == "flash":
        command += ["--flash-attn", "--attn-impl", "sdpa"]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="fitcheck GPU key, e.g. t4")
    parser.add_argument("--quant", default="none", choices=("none", "nf4", "int8"))
    parser.add_argument("--precision", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--out", default="runs", help="directory for the row files")
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the filenames, e.g. 'nf4'. A leading dash is optional.",
    )
    parser.add_argument(
        "--kernels",
        default="eager,flash",
        help="comma-separated subset of eager,flash",
    )
    parser.add_argument(
        "--no-repeats", action="store_true", help="skip the repeat block"
    )
    args = parser.parse_args(argv)
    args.tag = _normalise_tag(args.tag)

    if preflight(args) is None:
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]

    plan: list[tuple[str, int, int, str, str]] = []
    for kernel in kernels:
        for model_id, batch_size, seq_len in GRID:
            plan.append((model_id, batch_size, seq_len, kernel, args.tag))
        if not args.no_repeats:
            model_id, batch_size, seq_len = REPEAT
            for repeat in range(2, REPEATS + 1):
                plan.append(
                    (model_id, batch_size, seq_len, kernel, f"{args.tag}-r{repeat}")
                )

    done, failed = 0, []
    consecutive_failures = 0
    for index, (model_id, batch_size, seq_len, kernel, tag) in enumerate(plan, 1):
        target = out / f"{_slug(model_id, batch_size, seq_len, kernel, tag)}.json"
        if target.exists():
            print(f"[{index}/{len(plan)}] {target.name}  (already done, skipping)")
            done += 1
            continue

        print(f"[{index}/{len(plan)}] {target.name} ...", flush=True)
        started = time.time()
        result = subprocess.run(
            _row_args(model_id, batch_size, seq_len, kernel, args),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    FAILED ({result.returncode}). Last stderr lines:")
            for line in result.stderr.strip().splitlines()[-6:]:
                print(f"      {line}")
            failed.append(target.name)
            consecutive_failures += 1
            if consecutive_failures >= _ABORT_AFTER_FAILURES:
                print(
                    f"\nSTOPPING: {consecutive_failures} rows failed in a row. This is "
                    f"an environment problem, not a memory limit. Rows already "
                    f"written are kept and will be skipped.\n"
                )
                print(_diagnose(result.stderr))
                break
            continue

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            print(f"    FAILED: stdout was not JSON ({error})")
            print(f"      first 200 chars of stdout: {result.stdout[:200]!r}")
            failed.append(target.name)
            consecutive_failures += 1
            continue

        consecutive_failures = 0

        target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        errors = payload.get("error_pct", {})
        print(
            f"    ok in {time.time() - started:.0f}s  "
            f"tensors {errors.get('tensors', 0):+.1f}%  "
            f"process {errors.get('process', 0):+.1f}%"
        )
        done += 1

    print(f"\n{done}/{len(plan)} rows written to {out}/")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
