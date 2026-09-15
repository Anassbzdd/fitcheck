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
| `t4-sweep-2026-09-15.json` | Tesla T4 (sm_75) | 40 | LoRA r=32 [q,k,v,o], fp16, checkpointing on, eager and SDPA, seq 512–4096, bs 1/2/4, **both `--quant none` and `nf4`**. One session, one stack; `cuda_context_mib` is 140.875 on every row. 8 rows are deliberate repeats (`-r2`/`-r3`) and came back bit-identical — exclude them before fitting or they carry 3× weight. These are the rows that found the per-quantization activation profiles. |
| `t4-qlora-2026-09-01.json` | Tesla T4 (sm_75) | 10 | QLoRA r=32, fp16, checkpointing on, eager and SDPA, seq 512–2048. Measured half transcribed from `fitcheck.ipynb`; predicted half recomputed. See `_provenance` in the file. |

## Adding a card

A second card is what task 9.3 needs and what task 11.1 is about: the whole point of
keying the profile on the GPU is lost if only one GPU has ever been fitted. Measure
the same grid, drop the file here, re-fit, and open a PR — the `OVERHEAD_DB` entry is
the only source change.
