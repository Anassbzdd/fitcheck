# `fitcheck` — Task Checklist

> Ordered implementation steps. Check off as you go.
> Each task maps to a concrete deliverable. No task depends on a later one.

---

## Week 0 — Derive Before You Code

> **Rule: if you can't derive it on paper, you haven't learned it yet.**

- [x] **0.1** Derive parameter counting formula from `config.json` fields on paper
  - Embedding, attention (Q/K/V/O with GQA dims), MLP (gate/up/down), LayerNorm, LM head
  - Verify against `model.num_parameters()` for Llama-3.1-8B (8.03B)
- [x] **0.2** Derive LoRA param count formula on paper (GQA-aware)
  - Work through Llama 3.1-8B: q_proj (4096→4096), k_proj (4096→1024), v_proj (4096→1024), o_proj (4096→4096)
  - Compute: $32 \times 1{,}703{,}936 = 54.5\text{M}$ params → 104 MiB
- [x] **0.3** Derive QLoRA quantization overhead on paper
  - NF4: 0.5 bytes/param + scale overhead (2 bytes per block of 64)
  - Double quantization: ~50% reduction in scale overhead
- [x] **0.4** Derive activation memory formula on paper
  - List all **12** saved tensors per layer with their exact shapes (11 with Flash Attention)
  - Six of them are $(b,s,h)$ — hence the $6h$ term, not $5h$ or $4h$. See Blueprint Component 5.
  - Know what is *not* saved and why: pre-RoPE Q/K, `o_proj`'s output, residual adds
  - Write $A_{layer}$ formula with and without Flash Attention
  - Write $A_{checkpointed}$ formula (practical: every-layer checkpointing)
- [x] **0.5** Derive optimizer state sizes on paper
  - AdamW FP32: 8 bytes/param (m + v). Adam8bit: 2 bytes/param. SGD: 4 or 0.
- [x] **0.6** Work through full Llama-3.1-8B example on paper end-to-end
  - Config: QLoRA r=64, bs=4, seq=2048, BF16, AdamW, grad ckpt, Flash Attn, 4090
  - Must arrive at: 4,068.45 + 104 + 416 + 104 + 3,136 + 860.22 = **8,688.67 MiB** (displays as 8,689)
  - Fits a 4090 (23,500 usable) with 14,811 MiB / 63% headroom; max micro-batch **21**
  - Work in bytes throughout, convert to MiB once — see SPEC Appendix "Units discipline"
- [x] **0.7** Study Flash Attention memory model (2–3 hours)
  - Understand: standard attention is $O(s^2)$ in HBM, Flash Attention is $O(s)$
  - Know: tiling in SRAM, no materialization of full attention matrix

---

## Day 1 — Project Skeleton & Packaging

- [x] **1.1** Choose PyPI name: `fitcheck-llm` (confirmed selected)
- [x] **1.2** Create `pyproject.toml` with project metadata
  - Name, version (0.1.0), description, author, license (MIT), Python ≥3.10
  - Dependencies: `click`, `rich`, `huggingface-hub`
  - Dev dependencies: `pytest`, `pytest-cov`
  - `[project.scripts]` entry: `fitcheck = "fitcheck.cli:main"`
- [x] **1.3** Create package skeleton
  ```
  fitcheck/__init__.py
  fitcheck/__main__.py
  fitcheck/cli.py          (stub: just --help works)
  fitcheck/repl.py         (stub: prints "coming soon")
  fitcheck/config_parser.py
  fitcheck/estimator.py
  fitcheck/memory/__init__.py
  fitcheck/memory/weights.py
  fitcheck/memory/lora.py
  fitcheck/memory/optimizer.py
  fitcheck/memory/gradients.py
  fitcheck/memory/activations.py
  fitcheck/memory/overhead.py
  fitcheck/gpu_db.py
  fitcheck/display.py
  fitcheck/utils.py
  tests/conftest.py
  ```
- [x] **1.4** Install locally in editable mode
  ```bash
  pip install -e ".[dev]"
  ```
- [x] **1.5** Verify: `fitcheck --help` prints help text, `python -m fitcheck --help` works

---

## Day 2 — Config Parser & GPU Database

- [x] **2.1** Implement `config_parser.py`
  - `fetch_model_config(model_id: str) -> ModelConfig`
  - Uses `hf_hub_download` to get `config.json` only (~2KB)
  - Parses: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `tie_word_embeddings`
  - Computes `num_params` from config fields (no weight download)
  - Computes `head_dim` = `hidden_size // num_attention_heads` — **superseded by 2.6**: that is the
    fallback, not the rule. Do not re-implement it as the rule.
- [x] **2.2** Implement `gpu_db.py`
  - Dict mapping short names to `GpuSpec(name, vram_mib, usable_mib)`
  - GPUs: 22 entries — consumer, older/cloud, workstation, datacenter. `gpu_db.py` is the
    authority; the roster is documented in SPEC §3.4
  - `get_gpu(name: str) -> GpuSpec` with friendly error on unknown GPU
  - Support `--vram-mib` override for unlisted GPUs
- [x] **2.3** Implement `utils.py`
  - `bytes_to_mib(b: int) -> float`
  - `precision_to_bytes(precision: str) -> float`
  - `optimizer_bytes_per_param(optimizer: str, optimizer_dtype: str = "fp32") -> int`
    (`optimizer_dtype` selects the AdamW row: fp32 → 8, bf16 → 4)
- [x] **2.4** Write `tests/test_config_parser.py`
  - Test with a mock `config.json` for Llama-3.1-8B
  - Verify param count ≈ 8.03B
- [x] **2.5** Write `tests/test_gpu_db.py`
  - Test known GPUs return correct specs
  - Test unknown GPU raises clean error
- [x] **2.6** Fix two silent `config_parser.py` assumptions (SPEC §3.3, "Two fields that are not what they look like")
  - `head_dim`: use the config field when present; `hidden_size // num_attention_heads` is only the fallback
  - `tie_word_embeddings`: absent ≠ untied. Per-architecture default (Gemma-2 ties), `False` for unknown `model_type`
  - Generalize $P_{attn}$ to $2h \cdot n_h d_k + 2h \cdot n_{kv} d_k$ (identical for Llama, right for Gemma)
  - Verify: Gemma-2-9B counts **9.24B**, not 9.93B. Llama-3.1-8B must stay at exactly 8,030,261,248
  - Done: `_head_dim()`, `_tie_word_embeddings()`, generalized `_attention_param_count()`.
    Divisibility is now checked only when `head_dim` is *derived* — a config that declares it
    need not satisfy `h % n_h == 0`. 9 new tests + a `gemma_2_9b_config` fixture; `config_parser.py`
    at 100% coverage. Checked against the live Hub: Llama-3.1-8B **8,030,261,248**,
    Gemma-2-9B **9,241,404,928**, Qwen2.5-14B 14.77B

---

## Day 3 — All 6 Memory Modules

- [x] **3.1** Implement `memory/weights.py`
  - `estimate_weight_memory(num_params, precision, quantization_config) -> float` (MiB)
  - Handle: FP32, FP16, BF16, INT8, INT4
  - Handle: QLoRA scale overhead, double quantization
- [x] **3.2** Implement `memory/lora.py`
  - `estimate_lora_memory(config, rank, targets, precision) -> float` (MiB)
  - GQA-aware: k_proj and v_proj use `num_kv_heads × head_dim` as d_out
  - Support target lists: minimal (q,v), standard (q,k,v,o), full (q,k,v,o,gate,up,down)
- [x] **3.3** Implement `memory/optimizer.py`
  - `estimate_optimizer_memory(trainable_params, optimizer, is_lora, optimizer_dtype, precision) -> float` (MiB)
  - AdamW: 8 bytes/param (`optimizer_dtype="bf16"` → 4). Adam8bit: 2. SGD+momentum: 4. SGD: 0.
  - Full FT **in mixed precision only**: add master weight copy (+4 bytes/param).
    Under `precision="fp32"` there is no master copy — $W_{base}$ already holds FP32 params.
    See SPEC Component 3 for the indicator and the 16 bytes/param invariant.
- [x] **3.4** Implement `memory/gradients.py`
  - `estimate_gradient_memory(trainable_params, precision) -> float` (MiB)
  - BF16/FP16: 2 bytes/param. FP32: 4 bytes/param.
- [x] **3.5** Implement `memory/activations.py`
  - `estimate_activation_memory(config, batch_size, seq_len, grad_checkpoint, flash_attn, precision) -> float` (MiB)
  - $A_{layer} = \gamma bs\left[6h + 2h\frac{n_{kv}}{n_h} + 3d_{ff}\right] + \gamma bn_hs^2 \cdot \mathbb{1}[\text{no FA}]$
  - $\gamma$ = `precision_to_bytes(precision)` — **never hardcode 2**; FP32 doubles every term
  - Two paths: flash_attn ON (no $s^2$ term) vs OFF
  - Two paths: grad_checkpoint ON ($L \times \gamma bsh + A_{layer}$) vs OFF ($L \times A_{layer}$)
  - Read `intermediate_size` from config, never assume `4h`
- [x] **3.6** Implement `memory/overhead.py`
  - `estimate_overhead(weight_memory, activation_memory) -> float` (MiB)
  - Formula: $500 + 0.05 \times (W_{base} + A_{act})$
- [x] **3.7** Implement `estimator.py`
  - `estimate(model_config, training_config, gpu_spec) -> MemoryReport`
  - Calls all 6 modules and sums them. The orchestrator owns everything no single module can decide:
    - **Pick $P_{trainable}$**: LoRA param count when `lora_rank` is set, else `num_params` (full FT)
    - Thread `is_lora` into `estimate_optimizer_memory` (full FT adds the FP32 master copy)
    - Feed `precision` (compute dtype) to lora / gradients / activations; `quantization` to weights
    - **`max_batch_size` by bisection** — re-run the whole estimate for candidate $b$, take the largest
      that fits, and always **floor**. Do not extrapolate linearly: $C_{overhead}$ depends on $A_{act}(b)$,
      and for the worked example the true answer is 21.99, which rounds to the wrong side.
    - `effective_batch_size = batch_size * grad_accum_steps` (display only — costs zero memory)
    - `savings_hints`: re-run the estimate with one flag flipped per hint (see 4.7)
- [x] **3.8** Write tests for each memory module
  - `test_weights.py`: Llama-3.1-8B NF4 -> 4,068 MiB (+/-5%)
  - `test_lora.py`: r=64 on [q,k,v,o] with GQA -> 104 MiB (+/-5%)
  - `test_optimizer.py`: 54.5M params x AdamW -> 416 MiB
  - `test_gradients.py`: 54.5M params x BF16 -> 104 MiB
  - `test_activations.py`: Llama config, bs=4, seq=2048, grad_ckpt, flash -> 3,136 MiB (+/-10%)
    - also assert flash OFF -> 4,160 MiB, and that fp32 doubles the result
  - `test_overhead.py`: W_base=4,068.45 + A_act=3,136 -> 860.22 MiB; floor is 500 MiB at zero input
  - `test_end_to_end.py`: full Llama config -> total **8,689 MiB** (+/-10%), `max_batch_size == 21`
- [ ] **3.10** Generalize the activation bracket in `memory/activations.py` (SPEC Component 5, "Exact bracket")
  - **Smaller than it looks:** the KV term is already exact — `kv_width = num_kv_heads * head_dim`.
    The only change is `6 * hidden_size` → `4 * hidden_size + 2 * (num_attention_heads * head_dim)`
  - Target: $4h + 2 n_h d_k + 2 n_{kv} d_k + 3 d_{ff}$ — reduces to the $6h$ form when $n_h d_k = h$
  - Golden path must come out bit-identical: `A_layer` 1,088 MiB, `A_act` 3,136 MiB
  - Add a Gemma-2-9B case to `test_activations.py` ($n_h d_k = 4096$ vs $h = 3584$). Its
    `_model_config` helper hardcodes `head_dim=hidden_size // num_attention_heads` — that has to
    become a parameter or the new case cannot be expressed

---

## Day 4 — CLI, REPL, and Display

- [x] **4.1** Implement `display.py`
  - `render_report(report: MemoryReport, config: ModelConfig, gpu: GpuSpec)` → rich Panel + Table
  - Colored bar: green (fits with >20% headroom), yellow (fits <20% headroom), red (doesn't fit)
  - Pass/Fail verdict with headroom percentage
  - Max batch size suggestion
  - Component percentage column
- [x] **4.2** Implement `cli.py` (Mode A — full)
  - `@click.command` with all options from SPEC §3.5
  - Calls `fetch_model_config` → `estimate` → `render_report`
  - `--json` flag for machine-readable output
  - `--no-color` flag
  - `--verbose` flag for per-layer detail
- [x] **4.3** Implement `repl.py` (Mode B)
  - Session state: current `ModelConfig`, `GpuSpec`, last `MemoryReport`
  - Commands:
    - `model <id>` → fetch and store config, print confirmation
    - `gpu <name>` → lookup and store GPU, print confirmation
    - `memory [flags]` → compute and display report (same flags as CLI)
    - `explain` → generate plain-English explanation of last report
    - `optimize` → suggest max batch_size and recommended config
    - `compare --gpu <name>` → show side-by-side with different GPU
    - `help` → list commands
    - `exit` / `quit` → exit
  - Error handling: "Run `model` and `gpu` first" if missing context
  - Beyond the spec (see SPEC §3.5 Mode B, updated to match):
    - **The `memory` flags are literally the CLI's `click.Option` objects**, reused in a
      second command, so the two surfaces cannot drift.
    - **Flags stick** across `memory` calls, with `--no-flash-attn` / `--no-grad-checkpoint`
      / `--no-double-quant` and `reset` to undo them. `--gpu` is a one-shot override;
      only `gpu <name>` moves the session.
    - `compare` takes **several** GPUs (`compare 3090 t4 a100-40`) and states the point
      out loud: peak is identical on every card, only the ceiling moves.
    - `optimize` recommends a power-of-two micro-batch at ~75% of the ceiling plus the
      accumulation steps to keep the effective batch, and when nothing fits it walks a
      lever ladder (flash → ckpt → nf4 → seq/2 → adam8bit) until something does.
    - Extra commands: `show`, `reset`, `gpus`; aliases `mem`, `q`, `?`, `config`.
- [x] **4.4** Wire REPL into `cli.py`
  - `fitcheck` with no arguments → enters REPL
  - `fitcheck <model_id> [flags]` → one-liner mode
  - Mode is chosen by the presence of `MODEL_ID`, not the absence of flags (SPEC §3.5 Mode B):
    - `fitcheck --qlora --gpu 4090` **seeds** the session instead of erroring — the flags are
      sticky in the REPL anyway, so dropping them would be silent data loss
    - Only an explicit `--gpu` / `--vram-mib` presets the session GPU; Mode A's 4090 default
      would set a card the user never named
    - `--json` / `--verbose` / `--explain` without a `MODEL_ID` are a usage error (exit 2)
      pointing at `memory --json` in-session; flag validation runs before entry
    - Banner echoes the seeded GPU and flags. REPL always exits 0
- [ ] **4.5** Manual test: run `fitcheck meta-llama/Llama-3.1-8B --gpu 4090 --lora-r 64 --batch-size 4 --seq-len 2048 --precision bf16 --optimizer adamw --grad-checkpoint --flash-attn`
  - Screenshot the output
- [ ] **4.6** Manual test: run `fitcheck` (REPL mode)
  - Walk through: `model` → `gpu` → `memory` → `explain` → `optimize` → `compare` → `exit`

- [x] **4.7** `--explain` flag + savings hints (see SPEC §3.5 "explain output contract")
  - Landed with 4.2, since SPEC §3.5 lists `--explain` in the Mode A surface. The REPL's own
    `explain` command is still 4.3's to wire — it calls `render_explanation()`.
  - Promote `explain` out of the REPL: `--explain` works in CLI mode too
  - Default output gets one hint line; `--explain` prints the full breakdown
  - Name the largest component and say *why* it's large
  - Price each toggle by re-running the estimator with that one flag flipped — no new math:
    `adam8bit` −312, `--flash-attn` OFF +1,075, `--grad-checkpoint` OFF +33,264, `--grad-accum` 0
  - The `--grad-accum ... 0 MiB` line is load-bearing: it kills the most common memory myth

- [x] **4.8** Wire `--list-gpus` and `--vram-mib`
  - Landed with 4.2 (both are in the SPEC §3.5 option surface). `list_gpus()` already existed
    in `gpu_db.py`; `display.render_gpu_table()` renders it.
  - Both already work in `get_gpu()`; this is CLI plumbing only
  - Add `list_gpus()` helper to `gpu_db.py`, render as a rich table
  - `--list-gpus` is the first thing a new user reaches for

---

## Day 5 — Tests, Polish, Publish

- [ ] **5.1** Run full test suite
  ```bash
  pytest --cov=fitcheck --cov-report=term-missing
  ```
  - Target: ≥80% coverage on `memory/` modules — currently **100%** on all six, 222 tests green
  - All tests pass
  - Write `tests/test_utils.py` — it does not exist yet, and `utils.py` holds the whole
    `optimizer_bytes_per_param` β table (adamw 8 / bf16 states 4 / adam8bit 2 / sgd 4 / sgd 0)
- [ ] **5.2** Add `README.md`
  - What it does (1 paragraph)
  - Terminal screenshot (Mode A output)
  - Interactive REPL demo (Mode B)
  - Installation: `pip install fitcheck-llm`
  - Usage examples (both modes)
  - Comparison table vs. existing tools
  - Validation matrix (with TBD columns)
  - "How it works" section (link to SPEC.md or summary)
  - Contributing guidelines
  - License (MIT)
- [x] **5.3** Add `.gitignore`, `LICENSE` — both present

- [ ] **5.4** Add GitHub Actions CI — `.github/workflows/ci.yml`
  - `pytest --cov=fitcheck -m "not network"` on Python 3.10 / 3.11 / 3.12, ubuntu-latest, push + PR
  - **`-m "not network"` is required, not optional.** `test_config_parser.py` has one
    `@pytest.mark.network` test that fetches `meta-llama/Llama-3.1-8B` for real. That repo is
    **gated**: it passes locally because this machine is authenticated, and will fail on a runner
    with no `HF_TOKEN`. The 221 offline tests cover the same parsing via `fake_config_download`.
  - Upload coverage, add the badge to README
  - Highest resume signal per hour of work in this whole plan — do not skip it
  - SPEC §4 bullet 5 makes a green run a v0.1 ship condition, so it lands **before** the publish below

- [ ] **5.5** Ship v0.1.0 with an honest accuracy banner (SPEC §4, v0.1 bullet 3)
  - README states plainly: estimates are **analytical and not yet validated** against real measurements,
    target ±20%, validation matrix landing in v0.2
  - **Hold the r/LocalLLaMA post (8.1) until the matrix has ≥3 real rows.** That audience checks numbers,
    and unvalidated claims there are very hard to walk back.
- [ ] **5.6** Build and publish to PyPI
  ```bash
  python -m build
  twine upload dist/*
  ```
- [ ] **5.7** Verify: `pip install fitcheck-llm` from PyPI works on a clean venv
- [ ] **5.8** Verify: `fitcheck --help` works after PyPI install
- [ ] **5.9** Git tag `v0.1.0`, push to GitHub

---

## Week 2 — Validation & Inference Mode

- [ ] **6.0** Ship `scripts/measure.py` — the ground-truth harness (do this FIRST)
  - Loads model + PEFT, runs one full train step, reports `torch.cuda.max_memory_allocated()`
    and `max_memory_reserved()` (see Blueprint "How to Measure Ground Truth")
  - Prints a ready-to-paste markdown row for the validation matrix
  - Ship a **Colab T4 notebook** wrapper so validation needs no owned hardware — this unblocks
    6.1 entirely if you don't have a 4090 handy
  - Add `.github/ISSUE_TEMPLATE/measurement.yml` so users can submit their own rows
  - This turns the validation matrix from a chore into a contribution funnel: the matrix is what makes
    the tool trustworthy, and every submitted row is a user with a reason to watch the repo

- [ ] **6.1** Run ≥3 real training jobs with `scripts/measure.py`
  - Llama-3.1-8B on RTX 4090: QLoRA r=64, bs=4, seq=2048, FA2 → predicted **8,689 MiB**
  - Mistral-7B on T4: QLoRA r=32, bs=2, seq=1024, no FA
  - One other config (Qwen or Gemma)
- [ ] **6.2** Fill validation matrix in README with predicted vs. actual vs. error %
- [ ] **6.3** If any estimate is >±20% off, debug and adjust formulas
  - First suspect if Llama-3.1-8B measures **high**: NF4 scale dtype. v0.1 models `absmax` as FP16
    (239 MiB); `bitsandbytes` uses FP32 without double quant (479 MiB). See SPEC Component 1,
    "Known simplification — scale dtype". Changing it moves the whole golden set and the pinned tests.
  - Second suspect: activations. SPEC Component 5 is a principled ±10-15% estimate, not a guarantee
- [ ] **6.4** Implement `memory/inference.py`
  - `estimate_inference_memory(config, precision, seq_len, num_concurrent) -> float`
  - Weights + KV cache: $\text{KV} = 2 \times L \times n_{kv} \times d_k \times s \times n_{concurrent} \times \gamma$
  - The leading 2 is K and V — **once**, not twice. $n_{concurrent}$ is the batch dimension: it is in
    the signature and it belongs in the formula.
  - No activations, no optimizer, no gradients — inference saves nothing for backward
- [ ] **6.5** Add `fitcheck infer <model> [flags]` CLI command
- [ ] **6.6** Add inference commands to REPL
- [ ] **6.7** Update README with inference mode examples
- [ ] **6.8** Publish `v0.2.0` to PyPI

---

## Week 3–4 — Config Advisor & Polish

- [ ] **7.1** Implement `advisor.py`
  - Sweep (batch_size, lora_r, seq_len) parameter space
  - Report Pareto frontier: throughput vs. memory
  - Suggest optimal config for given model + GPU
- [ ] **7.2** Add `fitcheck advise <model> --gpu <gpu>` CLI command
- [ ] **7.3** Add `advise` command to REPL
- [ ] **7.4** Publish `v0.3.0`

---

## Week 5+ — Launch & Community

- [ ] **8.1** Post to r/LocalLLaMA with terminal screenshots
- [ ] **8.2** Post to X/Twitter with demo GIF
- [ ] **8.3** Deploy HuggingFace Gradio Space
- [ ] **8.4** Begin calibration mode (Phase 3)
- [ ] **8.5** Collect community GPU measurements for validation matrix
