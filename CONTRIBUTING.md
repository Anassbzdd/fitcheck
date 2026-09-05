# Contributing to fitcheck

Fork, branch off `main`, open a PR. Please keep changes to one memory component per PR where
possible — the modules under `fitcheck/memory/` are deliberately independent so a formula can be
argued about in isolation.

## The most useful thing you can contribute

**A measured row on hardware that is not a Tesla T4.**

Every number in the validation matrix comes from one T4 (sm_75), which means BF16 and real
FlashAttention-2 — both need sm_80 or newer — have never been exercised, and the 500 MiB
CUDA-context constant has been checked exactly once. If you have an Ampere or newer GPU, one run of
`scripts/measure.py` is worth more to this project than any feature:

```bash
pip install -r scripts/requirements-measure.txt
python scripts/measure.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --qlora --precision fp16 --lora-r 32 --batch-size 2 --seq-len 1024 --gpu t4
```

It prints prediction vs measurement at all three tiers, a per-component spot-check, and a markdown
row ready to paste. Open it with the
[measurement issue template](.github/ISSUE_TEMPLATE/measurement.yml).

The paths with **no measured row at all** are listed under "What is not measured" in the README.
The largest gaps: gradient checkpointing off, `--quant none` / `--quant int8`, full fine-tuning,
FP32 compute, `--double-quant`, sequences beyond 2048, and the whole `fitcheck infer` path.

## The bar for a merge

- `pytest --cov=fitcheck --cov-report=term-missing -m "not network"` is green. Currently 331
  offline tests, with 100% line coverage on all seven `memory/` modules; ≥80% there is the floor.
  The `-m "not network"` filter is not optional: it skips the one test that fetches the gated
  `meta-llama/Llama-3.1-8B` for real, which fails without an `HF_TOKEN`. The offline tests cover
  the same parsing against a fixture.

- **Any change to a formula updates its module, its test, and `docs/SPEC.md` in the same PR.** The
  Llama-3.1-8B golden numbers in the SPEC appendix are the reference set — if a change moves them,
  say so explicitly in the PR description. A formula whose derivation and implementation disagree is
  how this project got a 36% error once already.

- Type hints and docstrings on public functions, dataclasses for configs, MiB returned as `float`.
  Linting and type checking aren't wired up yet; if you want to add `ruff` and `mypy` configs,
  that's a welcome PR on its own.

## Two constraints that are not negotiable

1. **`fitcheck` never imports `torch`, `peft` or `bitsandbytes`** — not lazily, not inside a `try`.
   An estimate must cost a few KB of `config.json` and no GPU; that is the whole product. Those
   libraries belong only in `scripts/measure.py`, which is not a runtime dependency and is not
   installed by `pip install fitcheck-llm`. The dependency runs one way: `measure.py` imports
   `fitcheck`, never the reverse.

2. **Only `config.json` is ever fetched.** Never weights, never a checkpoint.

Runtime dependencies are `click`, `rich` and `huggingface-hub`. Adding a fourth is a decision, not a
detail — raise it in an issue first.

## Units

Report MiB (1024²) everywhere, never MB (10⁶). Compute in bytes and convert once, at the boundary.
The 4.9% gap between the two is enough on its own to flip a fits/doesn't-fit verdict near the edge
of a card — and it is exactly how the T4 entry in `gpu_db.py` ended up claiming more usable memory
than the card physically has.

## Where things live

`docs/SPEC.md` is the source of truth for formulas and the golden numbers. `docs/Blueprint.md`
carries the derivations and the educational walkthrough. `docs/SPEC.md` §3.8 explains what
`scripts/measure.py` does and why each part of it matters — read it before trusting a measurement,
including your own.
