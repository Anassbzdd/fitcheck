<p align="center">
  <img src="https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/logo.jpg" alt="fitcheck" width="150">
</p>

<h1 align="center">fitcheck</h1>

<p align="center">
  Predict how much VRAM your fine-tuning job will need — LoRA, QLoRA, or full fine-tuning — before you launch it.
</p>

<p align="center">
  <a href="https://github.com/Anassbzdd/fitcheck/actions/workflows/ci.yml"><img src="https://github.com/Anassbzdd/fitcheck/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/fitcheck-llm/"><img src="https://img.shields.io/pypi/v/fitcheck-llm.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/fitcheck-llm/"><img src="https://img.shields.io/pypi/pyversions/fitcheck-llm.svg" alt="Python"></a>
  <a href="https://github.com/Anassbzdd/fitcheck/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

`fitcheck` reads a model's `config.json` from the Hugging Face Hub (~2 KB, never the weights)
and computes peak training memory as a sum of six components: base model weights, LoRA adapter
weights, optimizer states, gradients, activations, and CUDA runtime overhead. Every number is
arithmetic over `hidden_size`, `num_hidden_layers`, `intermediate_size`, `num_key_value_heads`
and your training flags, so no GPU, no CUDA install, and no model download is involved — the
tool runs the same on a laptop as on the machine you're sizing for. You get the total, the
per-component breakdown, a fits/doesn't-fit verdict against a specific card, and the largest
micro-batch that still fits.

`fitcheck infer` does the same for the other half of the job: serving a trained model. There
the cost is the weights that stay in memory plus the KV cache, from the same config and the
same math. And `fitcheck advise` tries many settings instead of pricing one — what each knob
costs, how far each one can go before it stops fitting, and the configs right at that edge.

> **Accuracy status (v0.3.0, 2026-09-02): measured, and the measurements moved the formulas.**
> Ten real training runs on a Tesla T4 — three models, three sequence lengths, both attention
> kernels — put the five physical components within **3.4%** of measured peak (mean 0.8%). The full
> total, which includes the CUDA-overhead heuristic and is what the fits/doesn't-fit verdict uses, is
> within **14.7%** (mean 5.5%). All of the remaining error is in that one heuristic. Getting here
> changed three things in the activation formula and moved the reference numbers — see
> [Validation](#validation) for the table and [what is still unmeasured](#what-is-not-measured).

---

## Contents

- [Installation](#installation) · [Hugging Face access](#hugging-face-access) · [Model support](#model-support)
- The four commands: [`fitcheck`](#mode-a--one-liner) (estimate) · [REPL](#mode-b--interactive-repl) · [`infer`](#inference--fitcheck-infer) (serving) · [`advise`](#config-advisor--fitcheck-advise) (sweep)
- [Usage — all the flags and examples](#usage) · [Troubleshooting](#troubleshooting)
- [How it compares](#how-it-compares) · [Validation — predicted vs measured](#validation) · [How it works](#how-it-works) · [Contributing](#contributing)

---

## Installation

```bash
pip install fitcheck-llm
```

Python 3.10+. Runtime dependencies are `click`, `rich`, and `huggingface-hub` — no torch, no
CUDA.

### Hugging Face access

Most models need no authentication at all — `fitcheck Qwen/Qwen2.5-14B` and
`fitcheck mistralai/Mistral-7B-v0.3` work on a fresh machine with no token and no login.

**Gated repos are the exception, and that includes Llama and Gemma** — the models used in most
of the examples here. For those, accept the license on the model page, then either log in:

```bash
hf auth login
```

or set the token in the environment, which is what you want in CI or a container:

```bash
export HF_TOKEN=hf_...
```

Without it you get a clear error rather than a stack trace:

```
Error: Could not read config.json for 'meta-llama/Llama-3.1-8B': This model is gated on
Hugging Face. Accept its license on the model page, then run: hf auth login
```

Once a `config.json` is in the Hub cache, `fitcheck` runs offline.

---

## Model support

The parser handles dense decoder-only transformers with a gated (SwiGLU-style) MLP — Llama,
Mistral, Qwen2/2.5, Gemma-2/3 and anything config-shaped like them. It reads `head_dim` when the
config declares one rather than assuming `hidden_size / num_attention_heads`, and it never
assumes `intermediate_size == 4 × hidden_size`; both assumptions are wrong on Gemma-2.

Not modelled: MoE architectures (Mixtral, Qwen3-MoE, gpt-oss, DeepSeek), encoder-decoder models,
sliding-window attention, `torch.compile`, and multi-GPU sharding (FSDP / DeepSpeed ZeRO). See
[SPEC.md § 3.7](https://github.com/Anassbzdd/fitcheck/blob/main/docs/SPEC.md) for the full limitations table.

**Refused, not guessed at.** Two of those shapes are detectable from `config.json` alone, so
`fitcheck` exits **2** with an explanation instead of printing a plausible number:

| Shape | Detected by | What it would have said |
|:---|:---|:---|
| Mixture-of-Experts | `num_experts_per_tok`, `num_local_experts`, `num_experts`, `n_routed_experts` | Mixtral-8x7B as 7.24B instead of 46.7B — it counts one expert out of eight. Under-counts run 80–90% across Mixtral, Qwen3-30B-A3B, gpt-oss-20b and DeepSeek-V2-Lite |
| Multimodal with nested dimensions (SmolVLM) | `text_config` present, `hidden_size` absent | `'hidden_size' must be a positive integer` — true of the top level, and misleading: the fields are nested, not missing |

```console
$ fitcheck mistralai/Mixtral-8x7B-v0.1 --gpu 4090 --qlora
Error: 'mistralai/Mixtral-8x7B-v0.1' is a Mixture-of-Experts model (config.json declares
num_experts_per_tok, num_local_experts). fitcheck's weight and activation formulas assume a
dense FFN and would under-count total parameters by 80-90% - in the direction that reports a
fit where the run would OOM. MoE is not supported; see docs/SPEC.md section 3.7.
$ echo $?
2
```

> **Parsing is not endorsement — still check this list.** The refusal catches what the config
> file makes obvious. A shape it cannot see is still estimated silently: `Qwen/Qwen2.5-VL-7B-Instruct`
> keeps its text dimensions at the top level next to `vision_config`, so it parses like a dense
> decoder and comes out about 8% low, with the vision tower left out and nothing on screen saying
> so. Encoder-decoder and sliding-window models are the same class of gap.

---

## Mode A — one-liner

One command, one answer. This is a public mirror of Llama-3.1-8B, so it runs with no login:

```bash
fitcheck NousResearch/Meta-Llama-3.1-8B --qlora --lora-r 64 --batch-size 4 --seq-len 2048 --optimizer adamw --flash-attn --gpu 4090
```

![fitcheck Mode A output: component breakdown for Llama-3.1-8B QLoRA on an RTX 4090](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/mode-a-output.png)

Exit code is `0` if the config fits, `1` if it doesn't, `2` if the estimate couldn't be run — so
`fitcheck ... && accelerate launch ...` works as a guard in front of a training job. An
architecture `fitcheck` cannot model is a `2`, not a silent `0`, so the guard holds there too.

> The screenshot was taken with `meta-llama/Llama-3.1-8B`, the official repo. That one is
> **gated**, like every Llama and Gemma repo, so it needs `hf auth login` or an `HF_TOKEN`
> first — see [Hugging Face access](#hugging-face-access). The `NousResearch` mirror in the
> command above ships the same `config.json`, so every number matches. Public models like
> `Qwen/Qwen2.5-14B` and `mistralai/Mistral-7B-v0.3` need no token either.

---

## Mode B — interactive REPL

Run `fitcheck` with no model ID and you get a session instead. Flags typed at the `memory`
prompt stick, so moving one dial doesn't mean retyping the whole line.

![fitcheck Mode B session: banner, then model and gpu commands, then a memory estimate for Llama-3.1-8B QLoRA on an RTX 4090](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/mode-b-session.png)

`help` lists the command surface:

![fitcheck REPL help: the model, gpu, memory, infer, advise, explain, optimize, compare, show, reset, gpus, help and exit commands](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/mode-b-help.png)

`explain` names the largest component and prices every toggle by re-running the whole estimate with
one flag flipped — never by hand-summing component deltas, so the 5% that CUDA overhead picks up is
included automatically. Two results here matter most. Gradient accumulation costs **0 MiB**, because
gradients accumulate in place. And for this config, turning Flash Attention off also costs **0 MiB**:
under checkpointing the peak is the *larger* of the LM-head hump and one layer's recompute, and with a
128k vocabulary the LM head wins either way. A tool that promised a saving there would be wrong.

![fitcheck REPL explain output: the largest component named, followed by the cost of flipping each flag](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/mode-b-explain.png)

`compare` puts the same config on several cards, and leads with the point — the peak is
identical everywhere, only the ceiling moves, so the max micro-batch column is the interesting
one.

![fitcheck REPL compare output: RTX 4090, RTX 3090 and Tesla T4 side by side, none of them fitting, with max micro-batch 2, 2 and 0](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/mode-b-compare.png)

Also available: `optimize` (largest micro-batch that fits, plus a config actually worth
running), `advise` / `sweep` (the whole map at once — see below), `show`, `reset`, and `gpus`.

---

## Inference — `fitcheck infer`

Serving a model is a different budget from training one. There are no gradients, no
optimizer states and no saved activations. What stays resident is the weights plus the KV
cache, and the cache grows with every request you keep in flight.

```bash
fitcheck infer NousResearch/Meta-Llama-3.1-8B --gpu 4090
```

![fitcheck infer output: Llama-3.1-8B served in fp16 on an RTX 4090 — 15,317 MiB of weights, a 256 MiB KV cache, 16,851 MiB resident, fits with 28% headroom](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/infer-cli.png)

That is the ungated Llama-3.1-8B mirror, so it runs with no token — see
[Hugging Face access](#hugging-face-access).

The same thing is a REPL command, on the model and GPU already loaded. Its flags are sticky
like `memory`'s, but they are a **separate set** — serving computes in fp16 where training
defaults to bf16, so the two never share a value. That makes re-pricing the same model one
short line:

![fitcheck infer with NF4 double quantization: weights fall to 5,541 MiB and the total to 6,586 MiB, 72% headroom](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/infer-nf4.png)

4-bit weights take the same 8B model from 16,851 MiB down to 6,586 MiB: the weights line
falls from 15,317 to 5,541 MiB and the CUDA buffers shrink with it. The KV cache does not
move at all, because `--quant` is the **weight** format and `--precision` is the **compute**
dtype — a 4-bit deployment still serves an fp16 cache.

`compare ... --infer` puts one serving config on several cards. The peak is identical on all
of them, so the interesting column is how many concurrent requests each card can hold:

![fitcheck compare --infer: the same NF4 config on an RTX 4090, A100 40GB and Tesla T4, holding 63, 123 and 28 concurrent requests](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/infer-compare.png)

The cache is the part people under-budget. `fitcheck` prints its price per token and per
request — 0.125 MiB and 256 MiB for Llama-3.1-8B at 2,048 tokens — and `--seq-len` and
`--concurrent` are interchangeable: 4 requests of 2,048 tokens cost exactly what 1 request of
8,192 costs. Every request is assumed to hold its full context, so the number is a worst
case; a paged engine like vLLM allocates less until the cache fills up.

---

## Config advisor — `fitcheck advise`

A breakdown tells you what one config costs. It does not tell you which dial to turn. `advise`
sweeps batch size, sequence length and LoRA rank together and answers the two questions that
actually decide a run: **what does each axis cost**, and **how far can each one go before it
stops fitting**.

```bash
fitcheck advise meta-llama/Llama-3.1-8B --gpu 4090 --qlora --flash-attn \
  --lora-targets q,k,v,o --seq-lens 512,1024,2048,4096,8192
```

![fitcheck advise output: what is held fixed and what is swept, the frontier with a runnable command per row, the per-axis ceilings, and the price of each axis](https://raw.githubusercontent.com/Anassbzdd/fitcheck/main/docs/images/advise-cli.png)

The screen is four blocks: what is **held fixed** versus what is **swept**, the **frontier**
(the configs at the edge of what fits, one pasteable command per row), **the wall** (the exact
ceiling on each axis), and **the price of each axis**.

For Llama-3.1-8B QLoRA on a 4090, the price table settles the argument on its own. The
table below is anchored at **rank 64**; the screenshot above prices the same axes from the
frontier's own anchor, **rank 256**, so its rank rows are four times larger (−1,664 and
+3,328 MiB). The tokens/step rows are identical either way.

| Change from `batch 2 × seq 2048`, rank 64 | Cost | New total |
|:---|---:|---:|
| rank 64 → 32 | **−416 MiB** | 19,623.87 |
| rank 64 → 128 | **+832 MiB** | 20,871.87 |
| tokens/step ÷2 (batch 1) | **−5,284 MiB** | 14,756.27 |
| tokens/step ×2 (batch 4 *or* seq 4096) | **+10,567 MiB** | 30,607.07 — does not fit |

At rank 64, doubling tokens/step costs about **12.7×** what doubling the rank costs (at rank
256 it is still 3.2×). Rank is not the knob stopping you — and the ceiling proves it: at rank
64 the wall is 4,096 tokens/step, and dropping all the way to rank 8 does not buy a single
extra token. Meanwhile at 4,096 tokens/step the rank can go up to **330** before the card
runs out.

Two details that make the numbers trustworthy. Every ceiling is found by **bisection over the
full estimator**, not read off the grid — this grid stops at rank 256 while the real wall is
330, so a grid-only answer would be wrong. And `--seq-lens` is **required**, because there is
no honest default: sweeping past a model's real context length in silence would be worse than
asking. `--max-seq-len` is a guard that rejects a too-long swept length, not a generator.

`advise` is also a REPL command (alias `sweep`), and it shares the sticky training flags with
`memory` — `advise --qlora --flash-attn` sets them for both, so there is no second copy to
drift. In a session `--seq-lens` stops being required once you have given it, because a
session remembers.

`advise` says **what each axis costs and where the wall is**; the REPL's `optimize` says
**what to run**. They are deliberately different questions.

---

## Usage

### The flags you need first

Everything else has a sane default. `fitcheck --help` lists the full set.

| Flag | What it means | Default |
|:---|:---|:---|
| `--qlora` | Shorthand for `--quant nf4 --precision bf16 --grad-checkpoint`. The usual starting point. | off |
| `--quant` | How the base model is **stored** — `none`, `nf4` (4-bit), `int8`. | `none` |
| `--precision` | The **compute** dtype — LoRA weights, gradients, activations. Not the same axis as `--quant`. | `bf16` |
| `--lora-r` | LoRA rank. Higher = more trainable weights, more memory. | `16` |
| `--lora-targets` | Which layers get an adapter: `minimal` (q,v), `standard` (q,k,v,o), `full` (adds gate,up,down), or your own list like `q,k,v,o`. | `standard` |
| `--batch-size` | **Micro**-batch: what one forward/backward sees. This drives activation memory. | `1` |
| `--seq-len` | Sequence length in tokens. | `2048` |
| `--grad-checkpoint` | Recompute activations instead of storing them. Usually the biggest single saving. | off |
| `--flash-attn` | Skip the attention score matrix. Sometimes saves nothing — see [below](#a-result-worth-knowing). | off |
| `--gpu` | Target card. `--list-gpus` prints all 22. Use `--vram-mib` for a card not in the list. | `4090` |

`--grad-accum` is display-only: gradient accumulation costs **0 MiB**, because gradients
accumulate in place.

### Mode A

```bash
# QLoRA on a 4090 — the shorthand expands to --quant nf4 --precision bf16 --grad-checkpoint
fitcheck meta-llama/Llama-3.1-8B --qlora --lora-r 64 --batch-size 4 --seq-len 2048 --flash-attn

# All seven target modules, 8-bit optimizer, on a 16 GB T4
fitcheck mistralai/Mistral-7B-v0.3 --qlora --lora-r 32 --lora-targets full --optimizer adam8bit --batch-size 2 --seq-len 1024 --flash-attn --gpu t4

# Full fine-tuning in mixed precision (adds the FP32 master weight copy).
# This one reports "doesn't fit" and exits 1 — which is the useful answer.
fitcheck Qwen/Qwen2.5-14B --no-lora --precision bf16 --batch-size 1 --gpu a100-80

# Why is it that big, and what would each knob save?
fitcheck meta-llama/Llama-3.1-8B --qlora --lora-r 64 --batch-size 4 --explain

# A card that isn't in the database
fitcheck meta-llama/Llama-3.1-8B --qlora --vram-mib 32768

# Machine-readable, for CI
fitcheck meta-llama/Llama-3.1-8B --qlora --batch-size 4 --json
```

`--list-gpus` prints the 22-card database. `--verbose` adds the per-layer activation breakdown.
`--no-color` for logs. `-V` for the version. `fitcheck --help` has the full option surface.

### Mode B

```bash
fitcheck                          # bare session
fitcheck --qlora --gpu 4090       # flags without a MODEL_ID seed the session
```

```
model meta-llama/Llama-3.1-8B     # fetch config.json
gpu 4090                          # set the target card
memory --qlora --lora-r 64 --batch-size 4 --flash-attn
memory --batch-size 8             # flags are sticky; only the batch size changes
explain                           # largest component + price of every toggle
optimize                          # a batch size worth running, not just the ceiling
compare 3090 t4 a100-40           # same config, several cards
infer --quant nf4 --double-quant  # serving instead of training: weights + KV cache
compare a100-40 t4 --infer        # the serving config across cards
advise --seq-lens 1024,2048,4096  # sweep the knobs: axis prices, ceilings, frontier
reset                             # flags back to defaults
```

### Inference

```bash
# Weights + KV cache for one 2,048-token request
fitcheck infer meta-llama/Llama-3.1-8B --gpu 4090

# 4-bit serving, with double quantization for the smaller scale overhead
fitcheck infer meta-llama/Llama-3.1-8B --quant nf4 --double-quant --gpu 4090

# 8 concurrent requests at 8k context -- 14,919 MiB, still fits a 4090
fitcheck infer meta-llama/Llama-3.1-8B --quant nf4 --double-quant --seq-len 8192 --concurrent 8

# Doesn't fit: fp16 weights alone are 15,317 MiB and a T4 has 14,000 usable. Exits 1.
fitcheck infer meta-llama/Llama-3.1-8B --gpu t4

# Machine-readable, for CI
fitcheck infer meta-llama/Llama-3.1-8B --quant nf4 --json
```

Exit codes are the training command's: `0` fits, `1` doesn't fit, `2` couldn't run. `--gpu`,
`--vram-mib` and `--no-color` behave the same too. `fitcheck infer --help` has the full flag
list.

### Advisor

```bash
# The full map: axis prices, per-axis ceilings, and the frontier
fitcheck advise meta-llama/Llama-3.1-8B --gpu 4090 --qlora --flash-attn --seq-lens 512,1024,2048,4096,8192

# Narrow the sweep to the axes you can actually change
fitcheck advise meta-llama/Llama-3.1-8B --gpu 4090 --qlora --seq-lens 2048 --batch-sizes 1,2,4,8

# Refuse to price a context length the model cannot serve
fitcheck advise meta-llama/Llama-3.1-8B --gpu 4090 --qlora --seq-lens 8192,16384 --max-seq-len 8192

# Machine-readable, for CI
fitcheck advise meta-llama/Llama-3.1-8B --gpu 4090 --qlora --seq-lens 2048,4096 --json
```

`--seq-lens` is required in Mode A and optional in the session. Exit codes: `0` at least one
swept config fits, `1` none of them does, `2` couldn't run — which includes a `--seq-lens`
past `--max-seq-len`. `fitcheck advise --help` has the full flag list.

---

## Troubleshooting

**`fitcheck: command not found` after installing.** The package installs a script into your
Python environment's `bin` / `Scripts` folder. If that folder is not on your `PATH`, run it as a
module instead: `python -m fitcheck ...`.

**`This model is gated on Hugging Face`.** Accept the licence on the model page, then
`hf auth login` or set `HF_TOKEN`. See [Hugging Face access](#hugging-face-access).

**`Repository Not Found for url: ...`.** Either the model ID has a typo, or the repo is private.
The ID must be the full `owner/name`, exactly as it appears on the Hub.

**`Error: Unknown GPU 'x'`.** Run `fitcheck --list-gpus` for the 22 supported names. For a card
that is not in the list, give the VRAM directly: `--vram-mib 32768`.

**No internet.** `fitcheck` needs the network once per model, to fetch `config.json` (a few KB).
After that the file is in the Hub cache and the same command works offline.

**`is a Mixture-of-Experts model` or `nests its language-model dimensions`.** Not a bug and not a
bad model ID — `fitcheck` refuses these rather than under-count them by 80–90% and tell you a
doomed run fits. There is no override flag, because the number behind it would be wrong. See
[Model support](#model-support).

**The number looks wrong for my model.** First check it is a supported architecture — see
[Model support](#model-support). MoE and nested-multimodal configs are refused outright, but
encoder-decoder, sliding-window and flat multimodal models (Qwen2.5-VL) still parse and return a
wrong answer with no warning.

**My real run used a different amount.** Expect the total to be within about 15%, and see
[Validation](#validation) for where that error comes from. Two common causes are outside the
model: another process on the same card, and a serving engine like vLLM that pre-allocates a
fixed share of VRAM.

---

## How it compares

| | needs a GPU? | component breakdown? | LoRA / QLoRA training? | empirically validated? |
|:---|:---|:---|:---|:---|
| **fitcheck** | no | yes — all 6 | yes, GQA-aware | **yes — 10 measured runs** (see below) |
| [`accelerate estimate-memory`](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator) | no | weights + a coarse training multiplier | no | not published |
| [HF Model Memory Usage Space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) | no | same, in a web UI | no | not published |
| [vram.asmirnov.xyz](https://vram.asmirnov.xyz/) | no | yes, but outdated and has many issues | partial | not published |

None of these need a GPU — that isn't the differentiator, and claiming it would be dishonest.
The gaps `fitcheck` fills are LoRA/QLoRA-native accounting (adapter memory, optimizer states
sized to trainable params only, NF4 scale overhead), GQA-aware dimensions for `k_proj`/`v_proj`
and the K/V activations, and a CLI that exits nonzero so CI can gate on it.

The honest differentiator is the measured predicted-vs-actual table below. None of the alternatives
publish one. That is not a claim that they are wrong — it is a claim that nobody can tell, including
their authors. `fitcheck`'s numbers have been checked against real training runs, the checks found
three real bugs, and the gaps that remain are listed rather than hidden.

## Validation

Ten real training runs, all reproducible from [`fitcheck.ipynb`](https://github.com/Anassbzdd/fitcheck/blob/main/fitcheck.ipynb) with
[`scripts/measure.py`](https://github.com/Anassbzdd/fitcheck/blob/main/scripts/measure.py). One Tesla T4 (sm_75), FP16 compute, QLoRA r=32
[q,k,v,o], AdamW with FP32 states, gradient checkpointing on. Every run loads the real model,
applies real LoRA adapters, and runs real training steps.

### What "error" means here

PyTorch has two memory counters and neither of them is "VRAM used". Comparing the wrong pair makes a
correct formula look broken, so the harness compares three times, like with like:

| tier | predicted | measured | what it grades |
|:---|:---|:---|:---|
| **tensors** | the six components **minus** `C_overhead` | `max_memory_allocated()` | the five physical formulas |
| **process** | the full `fitcheck` total | `max_memory_reserved()` + CUDA context | what you actually see, and what the verdict uses |

Both are reported below, because showing only the flattering one would be dishonest.

### Results

| Model | Config | Attention | Predicted | Measured | Tensors err | Process err |
|:---|:---|:---|---:|---:|---:|---:|
| TinyLlama-1.1B | bs=2, seq=512 | eager | 1,834 | 1,849 | −0.8% | +9.4% |
| TinyLlama-1.1B | bs=2, seq=1024 | eager | 2,778 | 2,876 | **−3.4%** | −7.2% |
| TinyLlama-1.1B | bs=2, seq=2048 | eager | 6,702 | 6,655 | +0.7% | −3.2% |
| TinyLlama-1.1B | bs=2, seq=512 | SDPA (no s²) | 1,834 | 1,847 | −0.7% | +6.7% |
| TinyLlama-1.1B | bs=2, seq=1024 | SDPA (no s²) | 2,510 | 2,528 | −0.7% | +3.1% |
| TinyLlama-1.1B | bs=2, seq=2048 | SDPA (no s²) | 3,862 | 3,897 | −0.9% | −4.9% |
| SmolLM2-1.7B | bs=4, seq=1024 | eager | 5,280 | 5,297 | −0.3% | **−14.7%** |
| SmolLM2-1.7B | bs=4, seq=1024 | SDPA (no s²) | 5,280 | 5,281 | −0.0% | −0.2% |
| Qwen2.5-1.5B | bs=2, seq=1024 | eager | 6,810 | 6,831 | −0.3% | +1.7% |
| Qwen2.5-1.5B | bs=2, seq=1024 | SDPA (no s²) | 6,810 | 6,823 | −0.2% | +3.7% |

Predicted and Measured are MiB on the tensors tier.

| tier | max abs error | mean abs error |
|:---|---:|---:|
| **tensors** — the five physical formulas | **3.4%** | 0.8% |
| `A_act` alone, measured by subtraction | 4.6% | 0.9% |
| **process** — the full total | **14.7%** | 5.5% |

**How to read this.** The physical formulas are right to a few percent. `C_overhead` is not — the
process tier differs from the tensors tier only by that one heuristic, and every bit of the extra
error is there. Its fragmentation model assumes a flat 5%; measured fragmentation ran from 6% to 32%,
and is worst under eager attention because transient score matrices churn the allocator pool. That is
the next thing to fix.

Errors are signed as `(predicted − measured) / measured`, so a **negative** number means fitcheck
predicted **less** than reality — the unsafe direction.

### What these runs changed

They were not a rubber stamp. Three things in the activation formula were wrong, and each was found
by measurement rather than by re-reading the derivation:

1. **`A_act` summed the LM-head hump and the layer recompute.** They never coexist — the FP32 logits
   are freed before the backward pass reaches a decoder layer — so the peak is a `max`, not a sum.
2. **The checkpoint store was `L·γbsh`.** Non-reentrant checkpointing keeps two hidden-state tensors
   per layer, not one, so it is `2L·γbsh`.
3. **The eager attention score matrix was billed once.** It is materialized about nine times across
   forward and backward, including two FP32 softmax copies — `9γ`, not `γ`.

Across the wider development set of twenty runs, worst-case error fell from **36.1% to 4.8%**.

The third one was found by running the same config twice, changing only the attention kernel: the
difference between the two peaks *is* the cost of the score matrix. On a T4 that meant using SDPA's
memory-efficient backend, which — like Flash Attention — never builds the matrix, but unlike Flash
Attention runs on pre-Ampere hardware.

### A result worth knowing

**Flash Attention often saves no memory at all under gradient checkpointing.** The score matrix is
transient inside one layer's recompute, and `A_act` takes the max of that against the LM-head hump.
For a large-vocabulary model the LM head wins, so removing the matrix changes the peak by nothing —
measured at 16 MiB out of 5,297 on SmolLM2, and exactly 0 MiB for the Llama-3.1-8B reference config.
Flash still buys speed, and it starts buying memory once the sequence is long enough for the layer
hump to overtake the LM head.

### What is not measured

Be as clear about the gaps as about the results:

- **Any GPU other than this T4.** No Ampere or newer card, so no BF16 and no real FlashAttention-2 —
  the flash code path is validated only through SDPA's memory-efficient backend as a stand-in.
- **Gradient checkpointing off.** `L × A_layer + A_logits` is derived, never measured.
- **`--quant none`, `--quant int8`, full fine-tuning, FP32 compute.** Code paths with no measured row.
- **Sequences beyond 2048**, where the `9γ` coefficient multiplies an `s²` term.
- **`fitcheck infer` in full.** Every measured row is a training run. The serving path
  (resident weights + KV cache) shares the weight formula, which is measured, but the cache
  term and the serving overhead constant have no measured row of their own.
- **Every GPU entry except the T4.** The T4 used to claim 15,360 MiB usable of 16,384 when the card
  actually reports 14,912 MiB total — vendor GB treated as GiB, unsafe direction — and is now
  corrected to its measured `14_912 / 14_000`. It is the only row in `gpu_db.py` backed by a
  measurement. The rest are estimates, and `h200` / `b200` deliberately take the `usable_mib` that
  is safe under the pessimistic reading of their vendor GB until someone measures them.

Reproducing any of this needs one command:

```bash
python scripts/measure.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --qlora --precision fp16 --lora-r 32 --batch-size 2 --seq-len 1024 --gpu t4
```

`docs/SPEC.md` §3.8 explains what the harness does and why each part of it matters.

---

## How it works

Peak VRAM is modelled as `W_base + W_lora + S_optim + G_grad + A_act + C_overhead`, one module
per term under [`fitcheck/memory/`](https://github.com/Anassbzdd/fitcheck/blob/main/fitcheck/memory/): base weights (only the transformer linears
are packed under `--quant` — the embeddings, LM head and norms stay unquantized and get upcast to
FP32, plus one FP32 NF4 scale per block of 64), LoRA adapters (`r × (d_in + d_out)` per target, with
`k_proj`/`v_proj` narrowed to `num_kv_heads × head_dim` under GQA), optimizer states (trainable
params only — 8 bytes/param for AdamW, whose states stay FP32 even when you train in BF16),
gradients, activations, and CUDA overhead. `estimator.py` orchestrates the six and returns a
`MemoryReport`.

Two dtype rules catch people out, and both come from the same peft call. On a **quantized** base,
`prepare_model_for_kbit_training` upcasts everything trainable to FP32 — so QLoRA's adapters are
FP32, not the bf16 you asked for, and the gradients follow them there. `--precision` is the
**compute** dtype and drives activations; `--quant` is how the base is **stored**. They are separate
axes on purpose.

Activations are the hard term and the one worth reading about. `A_layer` sums the twelve tensors
autograd saves per decoder layer, plus the `(b, n_h, s, s)` attention score matrix when Flash
Attention is off — billed at **9γ**, because eager attention materializes it about nine times across
forward and backward. `A_logits` is four FP32 copies of the `(b, s, V)` tensor, and for a
large-vocabulary model it is usually the single biggest line in the whole budget.

Under gradient checkpointing the peak is **not** a sum of those:

```
A_act = 2L·γ·b·s·h  +  max(A_logits, A_layer)
```

Only the checkpoints are resident for the whole backward pass. The LM-head hump and one layer's
recompute are both transient and never overlap, so the peak takes whichever is larger — adding them
prices a moment that never happens. This is measured, not assumed, and getting it wrong is what made
v0.1.1 under-predict by up to 36%.

`max_batch_size` is found by bisecting the whole estimator and flooring, never by extrapolating from
one point: `total(b)` is piecewise linear, with a kink wherever that `max` flips branches.

Serving reuses the same weight term and adds one of its own:
`2 · L · (n_kv × head_dim) · s · concurrent · bytes` for the KV cache, GQA-narrowed like the
rest. `max_concurrent` is found the same way `max_batch_size` is — by bisecting the whole estimate
and flooring, not by dividing free space by the per-request cache. That shortcut gives too high
a number, because `C_overhead` is a percentage of a total that itself grows with the cache.

See [SPEC.md](https://github.com/Anassbzdd/fitcheck/blob/main/docs/SPEC.md) for the full memory model, and
[Blueprint.md](https://github.com/Anassbzdd/fitcheck/blob/main/docs/Blueprint.md) for the derivations.

---

## Contributing

Fork, branch off `main`, open a PR. Please keep changes to one memory component per PR where
possible — the modules are deliberately independent so a formula can be argued about in
isolation.

The bar for a merge:

- `pytest --cov=fitcheck --cov-report=term-missing -m "not network"` is green. Currently 371
  offline tests, with 100% line coverage on all seven `memory/` modules; ≥80% there is the
  floor. The `-m "not network"` filter is not optional: it skips the one test that fetches the
  gated `meta-llama/Llama-3.1-8B` for real, which fails without an `HF_TOKEN`. The offline
  tests cover the same parsing against a fixture.
- Any change to a formula updates its module, its test, and `docs/SPEC.md` in the same PR. The
  Llama-3.1-8B golden numbers in the SPEC appendix are the reference set — if a change moves
  them, say so explicitly in the PR description.
- Type hints and docstrings on public functions, dataclasses for configs, MiB returned as
  `float`. Linting and type checking aren't wired up yet; if you want to add `ruff` and `mypy`
  configs, that's a welcome PR on its own.

[CONTRIBUTING.md](https://github.com/Anassbzdd/fitcheck/blob/main/CONTRIBUTING.md) has the full version, including the two non-negotiable
constraints (no `torch` in the package, `config.json` only).

The most useful thing you can contribute right now is **a measured row on hardware that is not a
Tesla T4**. Every number in the validation table comes from one card, which means BF16 and real
FlashAttention-2 (both need sm_80 or newer) have never been exercised, and the 500 MiB CUDA-context
constant has been checked exactly once. If you have an Ampere or newer GPU, one run of
`scripts/measure.py` is worth more to this project than any feature — open it with the
[measurement issue template](https://github.com/Anassbzdd/fitcheck/blob/main/.github/ISSUE_TEMPLATE/measurement.yml).

---

## License

MIT. See [LICENSE](https://github.com/Anassbzdd/fitcheck/blob/main/LICENSE).
