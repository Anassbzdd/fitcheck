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


def _slug(model_id: str, batch_size: int, seq_len: int, kernel: str, tag: str) -> str:
    name = model_id.split("/")[-1].replace(".", "-")
    return f"{name}-bs{batch_size}-seq{seq_len}-{kernel}{tag}"


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
        "--tag", default="", help="suffix for the filenames, e.g. -nf4"
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
            continue

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            print(f"    FAILED: stdout was not JSON ({error})")
            failed.append(target.name)
            continue

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
