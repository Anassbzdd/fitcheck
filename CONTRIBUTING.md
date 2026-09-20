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

### Calibrating your card

`C_overhead` — the CUDA context plus the caching allocator's fragmentation — is the one component
that cannot be derived from a `config.json`, because it belongs to a driver and a card rather than
to a model. It is fitted per (GPU, attention kernel) and shipped as data in
`fitcheck/overhead_db.py`, so adding your card is a one-line source change:

```bash
python scripts/calibration_sweep.py --gpu <key>      # ~20 rows, unattended, one process each
python -m fitcheck.calibrate runs/*.json             # ad-hoc look at the fit and its residuals
```

Each row file is named after the **whole identity** of the run — card, kernel, quantization,
precision, optimizer, LoRA rank, checkpointing, model, batch size, sequence length — followed by a
fingerprint of that identity, so two configurations can never land on the same path. The identity
is written into the row as well (`sweep.identity`), the sweep refuses to skip or overwrite a file
whose identity does not match the row being asked for, and it rejects a row whose own `run` block
disagrees with what was requested. **A sweep that did not complete every planned row exits
non-zero**, so a grid with holes in it cannot be mistaken for a finished one.

Then commit the rows and **declare them**. `data/measurements/manifest.json` gives every archived
row a role — `calibration`, `holdout`, `repeat` or `excluded` (with a reason) — and only
`calibration` rows are fitted. Add one entry per row, then:

```bash
python -m fitcheck.calibrate data/measurements/manifest.json            # the fit and its residuals
python -m fitcheck.calibrate data/measurements/manifest.json --emit-python   # the OVERHEAD_DB literal
python -m fitcheck.calibrate data/measurements/manifest.json --check    # grade what now ships
```

Paste the `--emit-python` output over the `OVERHEAD_DB` literal in `fitcheck/overhead_db.py`.
**Do not hand-edit it**: `tests/test_manifest.py` re-runs that command and compares character for
character, so a typed coefficient fails CI. It also fails if the archive and the manifest disagree
by a single row, or if a manifest entry no longer matches the measurement it points at.

See the README under `data/measurements/` for the row format and for what makes a row fittable. A
fitted profile without its rows in the repo is a number nobody can check; a row without a declared
role is a number that can change the fit without anybody deciding it should.

The paths with **no measured row at all** are listed under "What is not measured" in the README.
The largest gaps: any card that is not a T4 (which is also what `C_overhead` needs before a
per-card fit means anything), a second `--quant int8` model, and an `fitcheck infer` concurrency
sweep.

## The bar for a merge

- `pytest --cov=fitcheck --cov-report=term-missing -m "not network"` is green. Currently 684
  offline tests, with 100% line coverage on all seven `memory/` modules; ≥80% there is the floor.
  The `-m "not network"` filter is not optional: it skips the 7 tests marked `network`, which hit
  the Hub for real — two of them the gated `meta-llama/Llama-3.1-8B`, which fails without an
  `HF_TOKEN`. The offline tests cover the same parsing against a fixture.

- **Numbers in the docs are checked, not trusted.** `tests/test_docs_claims.py` recomputes every
  quantitative claim in `README.md`, `CLAUDE.md` and this file from the artifacts they describe —
  the measurement archive, `fitcheck/overhead_db.py`, and the test suite's own collection — and
  fails with the number you should have written. If you add a test or a measured row, run it.

- **Any change to a formula updates its module, its test, and `docs/SPEC.md` in the same PR.** The
  Llama-3.1-8B golden numbers in the SPEC appendix are the reference set — if a change moves them,
  say so explicitly in the PR description. A formula whose derivation and implementation disagree is
  how this project got a 36% error once already.

- Type hints and docstrings on public functions, dataclasses for configs, MiB returned as `float`.

- **`ruff check .` and `mypy --strict fitcheck/` are both clean.** CI runs them as their own job,
  so a PR that fails either one does not merge. Both come with `pip install -e ".[dev]"`:

  ```bash
  ruff check .              # add --fix for the mechanical ones
  mypy --strict fitcheck/
  ```

  The configuration lives in `[tool.ruff]` and `[tool.mypy]` in `pyproject.toml`. Two deliberate
  choices there: the notebooks are excluded, because they are dated measurement artifacts rather
  than maintained source, and `RUF001-003` are off, because the formulas are written with the same
  symbols `docs/SPEC.md` uses (γ, ×) and ASCII lookalikes would make the two disagree. `--strict`
  covers `fitcheck/` only — `scripts/measure.py` imports `torch`, which is not installed in CI and
  must never become a dependency of the package.

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
