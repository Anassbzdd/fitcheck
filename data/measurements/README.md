# Measured runs

Raw ground truth for the `C_overhead` calibration (task 9.3). Not packaged — the wheel
ships only `fitcheck/`.

Five of fitcheck's six components are derivations you can check on paper. The sixth is
not: the CUDA context and the caching allocator's fragmentation belong to a driver and
a card, not to a `config.json`. These files are the only evidence fitcheck has about
them, so they live in the repo rather than in a notebook cell somebody re-runs.

## What a file contains

A JSON array of `scripts/measure.py --json` payloads, or a `{"_provenance": ..., "runs":
[...]}` wrapper when the rows need a note about where they came from. JSON lines also
works, which is what `>> runs.jsonl` produces.

The fit needs four numbers per row and will refuse a file missing any of them:

| field | why |
|:---|:---|
| `predicted.weight_mib`, `predicted.activation_mib` | fragmentation is billed as a fraction of `W_base + A_act` |
| `predicted.total_mib`, `predicted.overhead_mib` | their difference is the tensors-tier prediction, the part `C_overhead` has to close |
| `measured.peak_reserved_mib` | the allocator pool, which is what fragmentation actually costs |
| `measured.cuda_context_mib` | the process memory *outside* the pool — read after the step, not at init |
| `run.gpu_key`, `run.kernel`, `run.seq_len` | the fit keys on all three |

`run.*` only exists on rows produced after task 9.3. Older rows have to be assembled by
hand, which is why there is exactly one such file and it says so in its provenance.

## Producing a row

```bash
python scripts/measure.py <model> --gpu t4 --qlora --precision fp16 \
    --lora-r 32 --batch-size 2 --seq-len 2048 --json >> runs.jsonl
```

Always pass `--gpu <key>`: a row with no card cannot be filed under one. Never pass
`--no-predict`: the fit compares a prediction against a measurement and needs both.

## Fitting

```bash
python -m fitcheck.calibrate data/measurements/*.json            # report
python -m fitcheck.calibrate data/measurements/*.json --check    # grade what ships today
python -m fitcheck.calibrate data/measurements/*.json --emit-python
```

`--emit-python` prints the `OVERHEAD_DB` literal for `fitcheck/overhead_db.py`. Paste
it, then re-run `--check` to confirm the shipped constants score the way the fit said
they would.

## Files

| file | card | rows | note |
|:---|:---|---:|:---|
| `t4-phase3-2026-09-16.json` | Tesla T4 (sm_75) | 17 | LoRA r=32 [q,k,v,o], fp16, checkpointing on, both quants. The rows that broke two confounds: the **first batch ladder in the project** (TinyLlama seq 1024 nf4, bs 1/2/4/8/12 flash and 1/2/4/8 eager) and seq-2048 anchors on Qwen2.5-1.5B and SmolLM2-1.7B, which separated sequence length from model size. Carries `activation_logits_mib` / `activation_layer_mib` — the two humps the `C_overhead` fit needs. `cuda_context_mib` is 140.875, identical to the 2026-09-15 session on a different torch, which is what makes the two safe to fit together. |
| `t4-sweep-2026-09-15.json` | Tesla T4 (sm_75) | 40 | LoRA r=32 [q,k,v,o], fp16, checkpointing on, eager and SDPA, seq 512–4096, bs 1/2/4, **both `--quant none` and `nf4`**. One session, one stack; `cuda_context_mib` is 140.875 on every row. 8 rows are deliberate repeats (`-r2`/`-r3`) and came back bit-identical — exclude them before fitting or they carry 3× weight. These are the rows that found the per-quantization activation profiles. |
| `t4-qlora-2026-09-01.json` | Tesla T4 (sm_75) | 10 | QLoRA r=32, fp16, checkpointing on, eager and SDPA, seq 512–2048. Measured half transcribed from `fitcheck.ipynb`; predicted half recomputed. See `_provenance` in the file. |

## What a row needs to be fittable

Since task 9.3 the `C_overhead` fit is keyed **(GPU, kernel, quantization)** and regresses on
`min(A_logits, A_layer)`, so a row must carry `predicted.activation_logits_mib` and
`predicted.activation_layer_mib`. `measure.py` emits both (and `model_config`, so the archive
re-scores with no network). Rows older than 9.3 have neither; they still parse, and
`fit_group` falls back to the legacy proportional form for any group where one is missing.

Rows with gradient checkpointing **off** cannot be fitted with this form at all — without the
`max()` in `A_act` there is no losing hump, which is the whole mechanism.

## Adding a card

A second card is the one gap task 9.3 left open, and what task 11.1 is about: every one of the 66
rows behind the shipped profiles is a Tesla T4, and `B` is card-specific by construction, so the
whole point of keying on the GPU is lost while only one GPU has been fitted. Measure the same grid,
drop the file here, re-fit, and open a PR — the `OVERHEAD_DB` entry is the only source change.
