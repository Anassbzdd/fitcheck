# Measured runs

Raw ground truth for the `C_overhead` calibration (task 9.3) and its out-of-sample
verdict validation. Not packaged — the wheel ships only `fitcheck/`.

Five of fitcheck's six components are derivations you can check on paper. The sixth is
not: the CUDA context and the caching allocator's fragmentation belong to a driver and
a card, not to a `config.json`. These files are the only evidence fitcheck has about
them, so they live in the repo rather than in a notebook cell somebody re-runs.

## The manifest decides what gets fitted

`manifest.json` declares **every** archived row and gives each one a role. It is the
input to the fit; the measurement files are just where the numbers live.

| role | meaning |
|:---|:---|
| `calibration` | fitted. Every shipped coefficient comes from these rows and nothing else. |
| `holdout` | scored, never fitted. Grades the constants out of sample. |
| `repeat` | a re-measurement of a config already declared under another `run_id`. Kept as evidence, never fitted — three bit-identical copies of one config would carry 3× weight. |
| `excluded` | unusable, and the row says why. |

Each entry carries a stable `run_id`, a pointer (`file` + `index`) to the raw row, the
session it was measured in, and the full identity of the run: GPU, kernel, quantization,
double-quant, precision, optimizer, LoRA rank and targets, gradient checkpointing, batch
size, sequence length, model, and a SHA-256 of the `model_config` it was measured with.
`sessions` holds the software stack (torch / transformers / peft / bitsandbytes) for each
measurement session.

Loading verifies all of it against the archived row. **A stale manifest fails the load
rather than silently re-labelling a row** — so re-ordering a measurement file, or editing
a measured number, is loud.

Roles are a decision, not a heuristic. A new row means a new manifest entry, by hand.

## Producing a row

```bash
python scripts/measure.py <model> --gpu t4 --qlora --precision fp16 \
    --lora-r 32 --batch-size 2 --seq-len 2048 --json >> runs.jsonl
```

Always pass `--gpu <key>`: a row with no card cannot be filed under one. Never pass
`--no-predict`: the fit compares a prediction against a measurement and needs both.

`--json` prints one indented document per run, so appending several of them gives a file
whose rows span many physical lines. The loader reads it document by document, not line by
line, so `>>` works: a file of appended `--json` output and a one-object-per-line `.jsonl`
both load. A truncated run — a half-written document — is refused with the file named.

`scripts/calibration_sweep.py` drives the same script over the whole grid and adds a
`sweep` block to each row: the canonical identity it asked for, a fingerprint of that
identity, and the repeat index. The file name is built from the same identity, so two
configurations cannot share a path, and the sweep checks a row's own `run` block against
what it asked for before archiving it.

Then add the row to a file here **and** declare it in `manifest.json`. A row nobody
classified is a row that can drift into a fit unnoticed, and `tests/test_manifest.py`
fails if the manifest and the archive disagree by even one row.

## Fitting

```bash
# reproduces fitcheck/overhead_db.py byte for byte
python -m fitcheck.calibrate data/measurements/manifest.json --emit-python

python -m fitcheck.calibrate data/measurements/manifest.json            # report
python -m fitcheck.calibrate data/measurements/manifest.json --check    # grade what ships today
python -m fitcheck.calibrate data/measurements/manifest.json --role holdout --check
```

Only `calibration` rows are read unless you ask for more with `--role`. Passing a
directory (`data/measurements`) or the old glob (`data/measurements/*.json`) finds the
manifest and gives the same declared set — the manifest wins, and the files it covers
are not read twice.

`--emit-python` prints the `OVERHEAD_DB` literal for `fitcheck/overhead_db.py`. Paste
it, then re-run `--check` to confirm the shipped constants score the way the fit said
they would. `tests/test_manifest.py` holds that first command to a character-for-character
match against what is shipped, so the two cannot drift apart.

## What a row needs to be fittable

Since task 9.3 the `C_overhead` fit is keyed **(GPU, kernel, quantization)** and regresses on
`min(A_logits, A_layer)`, so a row must carry `predicted.activation_logits_mib` and
`predicted.activation_layer_mib`. `measure.py` emits both (and `model_config`, so the archive
re-scores with no network). Rows older than 9.3 have neither.

Mixing the two is now refused: a group whose rows do not all carry humps raises a
calibration error naming the offenders. It used to fall back to the legacy proportional
form silently, so **one** legacy row could decide the form for an entire group while the
group still reported its full row count.

Rows with gradient checkpointing **off** cannot be fitted with this form at all — without the
`max()` in `A_act` there is no losing hump, which is the whole mechanism. `measure.py` emits the
two humps whatever `grad_checkpoint` says, so their presence is not evidence that the `max()`
existed: the parser therefore keeps `run.grad_checkpoint` (a real boolean — a string is refused,
not coerced) and the fitter refuses a hump fit unless every row in the group ran with it on. A
group whose rows disagree on the flag is refused too, rather than averaged into one profile.

## Files

| file | card | rows | declared | note |
|:---|:---|---:|:---|:---|
| `manifest.json` | — | 79 entries | — | The roles. Not a measurement file. |
| `t4-phase3-2026-09-16.json` | Tesla T4 (sm_75) | 17 | 17 calibration | LoRA r=32 [q,k,v,o], fp16, checkpointing on, both quants. The rows that broke two confounds: the **first batch ladder in the project** (TinyLlama seq 1024 nf4, bs 1/2/4/8/12 flash and 1/2/4/8 eager) and seq-2048 anchors on Qwen2.5-1.5B and SmolLM2-1.7B, which separated sequence length from model size. Carries `activation_logits_mib` / `activation_layer_mib` — the two humps the `C_overhead` fit needs. `cuda_context_mib` is 140.875, identical to the 2026-09-15 session on a different torch, which is what makes the two safe to fit together. |
| `t4-sweep-2026-09-15.json` | Tesla T4 (sm_75) | 40 | 32 calibration, 8 repeat | LoRA r=32 [q,k,v,o], fp16, checkpointing on, eager and SDPA, seq 512–4096, bs 1/2/4, **both `--quant none` and `nf4`**. One session, one stack; `cuda_context_mib` is 140.875 on every row. The 8 repeats (`-r2`/`-r3`) came back bit-identical, which is how we know the allocator noise floor on this card is zero. |
| `t4-qlora-2026-09-01.json` | Tesla T4 (sm_75) | 10 | 10 excluded | QLoRA r=32, fp16, checkpointing on, eager and SDPA, seq 512–2048. Measured half transcribed from `fitcheck.ipynb`; predicted half recomputed. Excluded because the rows predate task 9.3: no activation humps and no embedded `model_config`, so they can neither be fitted with the hump form nor re-scored offline. See `_provenance` in the file. |
| `final-t4-20260921.json` | Tesla T4 (sm_75) | 12 | 12 holdout | Final successful accuracy rows: NF4, FP16, LoRA r=16 [q,k,v,o], AdamW FP32, checkpointing, seq 1024, both kernels. Scored only after the coefficients were frozen; never fitted. |
| `validation/final-t4-20260921-verdicts.json` | Tesla T4 (sm_75) | 12 outcomes | separate evidence | Six point-estimate safe ceilings and six unsafe ceilings against a 14,000 MiB budget. Includes four actual OOMs and two completed runs over budget; never fitted. |

**49 rows are fitted.** That is the number behind every shipped coefficient.

## Holdout and boundary evidence

The final 12 successful accuracy rows now carry role `holdout` in `manifest.json`. The
default calibration command reads only `calibration` rows; run
`python -m fitcheck.calibrate data/measurements/manifest.json --role holdout --check`
to score them after the coefficients are frozen. The separate boundary file preserves
the 12 safe/unsafe outcomes because OOM rows do not have a comparable peak measurement
for coefficient fitting. The final evidence is narrow: one T4, NF4/FP16, LoRA r=16,
AdamW FP32, checkpointing, seq=1024.

The older notebook holdout quoted in historical sections is a different, uncommitted
archive and remains historical; it is not used to describe the v0.3.1 beta gate.

## Adding a card

A second card is the one gap task 9.3 left open, and what task 11.1 is about: every one of the
49 fitted rows is a Tesla T4, and `B` is card-specific by construction, so the whole point of
keying on the GPU is lost while only one GPU has been fitted. Measure the same grid, drop the
file here, declare the rows in `manifest.json`, re-fit, and open a PR — the `OVERHEAD_DB` entry
is the only source change, and it is generated, not typed.
