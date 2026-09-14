"""Drive `measure.py` over the C_overhead calibration grid (task 9.3).

One subprocess per row, on purpose. The CUDA context and the caching allocator's
high-water mark are both process-scoped: running two configurations in one interpreter
would measure the second one on top of whatever the first left in the pool, and the
context reading would be whichever run touched the most kernels. A fresh process per
row is the only way `cuda_context_mib` and `peak_reserved_mib` mean what they say.

Writes one JSON file per row into `--out`, which is the shape
`python -m fitcheck.calibrate <out>/*.json` expects. Rows that fail (OOM, a model that
will not download) are reported and skipped rather than taking the sweep down with
them.

    python scripts/calibration_sweep.py --gpu t4
    python scripts/calibration_sweep.py --gpu p100-16 --quant none
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MEASURE = Path(__file__).resolve().parent / "measure.py"

# (model, batch_size, seq_len). Run under both kernels, so each line is two rows.
#
# Chosen to spread `W_base + A_act` as widely as 16 GB allows and to put four distinct
# sequence lengths in every kernel group -- the sequence slope needs three levels and
# six rows before `calibrate.py` will fit it at all, and seq 4096 is the row the audit
# singles out as the worst fragmentation this project has measured.
GRID: tuple[tuple[str, int, int], ...] = (
    ("HuggingFaceTB/SmolLM2-135M", 2, 512),
    ("HuggingFaceTB/SmolLM2-360M", 1, 4096),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 512),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 1024),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2, 2048),
    # 135M rather than 1.1B at seq 4096: the eager score matrix is 9*gamma*b*n_h*s^2,
    # which for TinyLlama at 4096 is ~9.7 GiB on its own and would OOM a 14.9 GiB T4.
    # The point of the row is the sequence length, not the model.
    ("HuggingFaceTB/SmolLM2-135M", 1, 4096),
    ("HuggingFaceTB/SmolLM2-1.7B", 4, 1024),
    ("Qwen/Qwen2.5-1.5B-Instruct", 2, 1024),
)

# Run this one three times per kernel. Its eager row is the outlier that decides
# whether the T4 eager group can be fitted at all: it reserved 30.9% more than it
# allocated while its SDPA twin reserved 11.9%, off an identical tensor peak. Three
# repeats say whether that is a mechanism or run-to-run allocator variance, and no
# amount of extra configurations will answer it -- only repeats of this one.
REPEAT = ("HuggingFaceTB/SmolLM2-1.7B", 4, 1024)
REPEATS = 3

# One failed row is a data point -- an out-of-memory config is a real answer. Three in a
# row is a broken environment saying the same thing over and over, and the remaining
# rows would only say it again after paying for the model downloads.
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


# Run in a subprocess. It prints the stack description as JSON on success, and on
# failure prints the traceback followed by the ROOT CAUSE as the last line -- the real
# error in a transformers import chain sits fifty lines below the symptom, and a tail
# of the traceback shows only the symptom. `Could not import module 'LlamaConfig'` is
# the symptom; `operator torchvision::nms does not exist` is the cause.
_PROBE = """
import json, sys, traceback
try:
    import torch, transformers, peft
    p = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
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
    if "torchvision::nms" in stderr or "torchvision" in stderr and "operator" in stderr:
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
    print(
        f"{info['gpu']} ({info['capability']}, {info['total_mib']:,} MiB) | "
        f"torch {info['torch']} | transformers {info['transformers']} | "
        f"peft {info['peft']}"
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
        # --flash-attn is the fitcheck knob; --attn-impl sdpa is the kernel that
        # actually runs, because FA2 needs sm_80 and neither T4 nor P100 has it.
        # SDPA's memory-efficient backend never materializes the score matrix, which
        # is the branch fitcheck's flash_attn path predicts.
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
                # An out-of-memory row is a data point; three in a row is a broken
                # environment, and the next seventeen will say exactly the same thing.
                print(
                    f"\nSTOPPING: {consecutive_failures} rows failed in a row. This is "
                    f"an environment problem, not a memory limit -- fix the error above "
                    f"and re-run. Rows already written are kept and will be skipped."
                )
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
