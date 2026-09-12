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

`fitcheck` reads a model's `config.json` from the Hugging Face Hub (~2 KB, never the weights) and
its parameter count from the Hub's metadata (also never the weights), then computes peak training
memory as a sum of six components: base model weights, LoRA adapter
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

Python 3.10+. Runtime dependencies are `click>=8.1`, `rich>=13.0` and `huggingface-hub>=0.25` —
no torch, no CUDA.

The `huggingface-hub` floor is exact, not cautious: `GatedRepoError` is only exported from
`huggingface_hub.errors` in 0.25 and later, so any older version crashes `fitcheck` on import. A CI
job installs the lowest version of every declared dependency on Python 3.10 and runs the test
suite there, so the floor stays honest.

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

Once a `config.json` is in the Hub cache, `fitcheck` runs offline. The one thing it loses offline
is the Hub parameter count (see [Model support](#model-support)); it falls back to the count
derived from `config.json` and warns on stderr that it did.

---

## Model support

The parser handles dense decoder-only transformers with a gated (SwiGLU-style) MLP — Llama,
Mistral, Qwen2/2.5, Gemma-2/3 and anything config-shaped like them. It reads `head_dim` when the
config declares one rather than assuming `hidden_size / num_attention_heads`, and it never
assumes `intermediate_size == 4 × hidden_size`; both assumptions are wrong on Gemma-2.

Not modelled: MoE architectures (Mixtral, Qwen3-MoE, gpt-oss, DeepSeek), non-gated 2-matrix MLPs
(phi-2), multimodal models, encoder-decoder models, sliding-window attention, `torch.compile`, and
multi-GPU sharding (FSDP / DeepSpeed ZeRO). See
[SPEC.md § 3.7](https://github.com/Anassbzdd/fitcheck/blob/main/docs/SPEC.md) for the full limitations table.

**Refused, not guessed at.** When a model is outside what the formulas cover, `fitcheck` exits **2**
with an explanation instead of printing a plausible number. Two shapes are caught from
`config.json` keys; a third is caught by checking the parameter count against the Hub:

| Shape | Detected by | What it would have said |
|:---|:---|:---|
| Mixture-of-Experts | `num_experts_per_tok`, `num_local_experts`, `num_experts`, `n_routed_experts` | Mixtral-8x7B as 7.24B instead of 46.7B — it counts one expert out of eight. Under-counts run 80–90% across Mixtral, Qwen3-30B-A3B, gpt-oss-20b and DeepSeek-V2-Lite |
| Multimodal with nested dimensions (SmolVLM) | `text_config` present, `hidden_size` absent | `'hidden_size' must be a positive integer` — true of the top level, and misleading: the fields are nested, not missing |
| Anything whose real parameter count is >2% from the formula | the Hub's `safetensors.total`, compared against the count derived from `config.json` | `microsoft/phi-2` **+30.1%** (a 2-matrix `fc1`/`fc2` MLP charged as 3-matrix SwiGLU) and `Qwen/Qwen2.5-VL-7B-Instruct` **−8.2%** (vision tower left out) |

```console
$ fitcheck mistralai/Mixtral-8x7B-v0.1 --gpu 4090 --qlora
Error: 'mistralai/Mixtral-8x7B-v0.1' is a Mixture-of-Experts model (config.json declares
num_experts_per_tok, num_local_experts). fitcheck's weight and activation formulas assume a
dense FFN and would under-count total parameters by 80-90% - in the direction that reports a
fit where the run would OOM. MoE is not supported; see docs/SPEC.md section 3.7.
$ echo $?
2
```

**Where the parameter count comes from.** `fitcheck` asks the Hub for the model's real parameter
count (`safetensors.total` — one metadata call, still no weights) and uses that. The count derived
from `config.json` is kept as the cross-check: the derivation assumes a 3-matrix SwiGLU MLP with no
biases, so when the two disagree by more than 2%, the architecture is outside what the *activation*
and *LoRA* formulas assume as well. That is refused, not averaged and not quietly patched with the
Hub's number — a correct weight term next to a wrong activation term is still a wrong answer.

```console
$ fitcheck microsoft/phi-2 --gpu 4090 --qlora
Error: 'microsoft/phi-2' has 2,779,683,840 parameters according to the Hub, but fitcheck's
config.json formula derives 3,617,753,600 - a disagreement of +30.1%, past the 2% tolerance
(over-counts, which over-states the memory needed). [...] fitcheck refuses instead of
averaging the two. See docs/SPEC.md section 3.3.
$ echo $?
2
```

> **Parsing is not endorsement — still check this list.** The two checks catch what the config file
> makes obvious and what the parameter count gives away. A shape that shows up in neither is still
> estimated silently: encoder-decoder models, and sliding-window attention (Gemma-2 alternates
> sliding and full layers, and the parameter count is identical either way). Offline the parameter
> cross-check cannot run at all — `fitcheck` falls back to the derived count, says so on stderr, and
> phi-2 and Qwen2.5-VL are estimated instead of refused.

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

**No internet.** `fitcheck` needs the network once per model, to fetch `config.json` (a few KB)
and to read the model's parameter count from the Hub's metadata (no weights, either way). After
that the file is in the Hub cache and the same command works offline, with the parameter count
falling back to the `config.json` formula and a warning saying so.

**`is a Mixture-of-Experts model` or `nests its language-model dimensions`.** Not a bug and not a
bad model ID — `fitcheck` refuses these rather than under-count them by 80–90% and tell you a
doomed run fits. There is no override flag, because the number behind it would be wrong. See
[Model support](#model-support).

**`parameters according to the Hub, but fitcheck's config.json formula derives ...`.** Same idea,
one level deeper. The parameter count the Hub reports and the one the formula derives are more
than 2% apart, which means the model's MLP or its components are not the shape the activation and
LoRA formulas assume. `fitcheck` refuses instead of adopting the Hub's number, because only the
weight term would be fixed by that. No override flag, for the same reason.

**The number looks wrong for my model.** First check it is a supported architecture — see
[Model support](#model-support). MoE and nested-multimodal configs are refused outright, and so is
anything whose parameter count disagrees with the Hub by more than 2% (phi-2, Qwen2.5-VL). What
still parses and returns a wrong answer with no warning is the class the count does not give
away: encoder-decoder and sliding-window models.

**My real run used a different amount.** Expect the total to be within about 15%, and see
[Validation](#validation) for where that error comes from. Two common causes are outside the
model: another process on the same card, and a serving engine like vLLM that pre-allocates a
fixed share of VRAM.

---

## How it compares

| | needs a GPU? | component breakdown? | LoRA / QLoRA training? | empirically validated? |
|:---|:---|:---|:---|:---|
| **fitcheck** | no | yes — all 6 | yes, GQA-aware | **yes — 33 measured runs** (see below) |
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

**33 real training and serving runs on one Tesla T4 (sm_75), FP16 compute.** Every run loads the real
model, applies real LoRA adapters, and runs real steps — nothing here is simulated. Reproducible from
[`fitcheck.ipynb`](https://github.com/Anassbzdd/fitcheck/blob/main/fitcheck.ipynb) and
[`fitcheck_infer.ipynb`](https://github.com/Anassbzdd/fitcheck/blob/main/fitcheck_infer.ipynb) with
[`scripts/measure.py`](https://github.com/Anassbzdd/fitcheck/blob/main/scripts/measure.py).

**Library versions are part of the result.** Attention internals change between majors, so mixing rows
from different stacks in one table without saying so would be misleading. The 23 newer rows ran on:

```
torch 2.10.0+cu128 | transformers 5.0.0 | peft 0.19.1 | Python 3.12 | CUDA 12.8
```

The ten original rows ran on an earlier stack (torch 2.x / transformers 4.x). They are kept in their
own table below for that reason, not merged into one list.

### What "error" means here

PyTorch has two memory counters and neither of them is "VRAM used". Comparing the wrong pair makes a
correct formula look broken, so the harness compares three times, like with like:

| tier | predicted | measured | what it grades |
|:---|:---|:---|:---|
| **tensors** | the six components **minus** `C_overhead` | `max_memory_allocated()` | the five physical formulas |
| **process** | the full `fitcheck` total | `max_memory_reserved()` + CUDA context | what you actually see, and what the verdict uses |

Both are reported below, because showing only the flattering one would be dishonest.

Errors are signed as `(predicted − measured) / measured`, so a **negative** number means fitcheck
predicted **less** than reality — the unsafe direction, the one that lets a run OOM after you were
told it would fit.

### Results — training, gradient checkpointing ON

QLoRA r=32 [q,k,v,o], AdamW with FP32 states. Predicted and Measured are MiB on the tensors tier.

| Model | Config | Attention | Predicted | Measured | Tensors err | Process err |
|:---|:---|:---|---:|---:|---:|---:|
| TinyLlama-1.1B | bs=2, seq=512 | eager | 1,834 | 1,849 | −0.8% | +9.4% |
| TinyLlama-1.1B | bs=2, seq=1024 | eager | 2,849 | 2,876 | −1.0% | −5.2% |
| TinyLlama-1.1B | bs=2, seq=2048 | eager | 6,844 | 6,655 | +2.8% | −1.3% |
| TinyLlama-1.1B | bs=2, seq=512 | SDPA (no s²) | 1,834 | 1,847 | −0.7% | +6.7% |
| TinyLlama-1.1B | bs=2, seq=1024 | SDPA (no s²) | 2,510 | 2,528 | −0.7% | +3.1% |
| TinyLlama-1.1B | bs=2, seq=2048 | SDPA (no s²) | 3,862 | 3,897 | −0.9% | −4.9% |
| SmolLM2-1.7B | bs=4, seq=1024 | eager | 5,280 | 5,297 | −0.3% | **−14.7%** |
| SmolLM2-1.7B | bs=4, seq=1024 | SDPA (no s²) | 5,280 | 5,281 | −0.0% | −0.2% |
| Qwen2.5-1.5B | bs=2, seq=1024 | eager | 6,810 | 6,831 | −0.3% | +1.7% |
| Qwen2.5-1.5B | bs=2, seq=1024 | SDPA (no s²) | 6,810 | 6,823 | −0.2% | +3.7% |

Two newer rows, on the 2026-09 stack, cover paths the ten above do not — full fine-tuning, and a
sequence past 2048. These use LoRA/full FT on an fp16 base, not QLoRA:

| Model | Config | Attention | Predicted | Measured | Tensors err | Process err |
|:---|:---|:---|---:|---:|---:|---:|
| llama-160m | **full fine-tuning**, bs=2, seq=1024 | eager | 3,550 | 3,377 | +5.1% | −7.9% |
| SmolLM2-360M | r=32, bs=1, **seq=4096** | eager | 5,765 | 4,909 | **+17.4%** | −12.0% |

### Results — training, gradient checkpointing OFF

This is the branch that had never been measured, and it is the **default**. LoRA r=32 [q,k,v,o] on an
unquantized fp16 base — so these rows also close `--quant none`.

| Model | Config | Attention | Predicted | Measured | Tensors err | Process err |
|:---|:---|:---|---:|---:|---:|---:|
| SmolLM2-135M | bs=2, seq=512 | SDPA (no s²) | 1,868 | 1,846 | +1.2% | +22.2% |
| SmolLM2-135M | bs=4, seq=512 | SDPA (no s²) | 3,424 | 3,314 | +3.3% | +11.5% |
| SmolLM2-135M | bs=1, seq=2048 | SDPA (no s²) | 3,424 | 3,314 | +3.3% | +11.5% |
| SmolLM2-1.7B | bs=1, seq=512 | SDPA (no s²) | 5,184 | 5,112 | +1.4% | +8.9% |
| SmolLM2-1.7B | bs=1, seq=512 | eager | 6,298 | 6,215 | +1.3% | +9.7% |
| Qwen2.5-0.5B | bs=2, seq=512 | SDPA (no s²) | 4,702 | 4,504 | +4.4% | +2.7% |
| Qwen2.5-0.5B | bs=2, seq=512 | eager | 5,677 | 5,486 | +3.5% | +3.2% |
| Qwen2.5-1.5B | bs=1, seq=512 | SDPA (no s²) | 5,636 | 5,552 | +1.5% | +8.7% |
| Qwen2.5-1.5B | bs=1, seq=512 | eager | 6,124 | 5,999 | +2.1% | +7.2% |
| gemma-2-2b | bs=1, seq=512 | SDPA (no s²) | 8,767 | 8,993 | −2.5% | +2.8% |
| gemma-2-2b | bs=1, seq=512 | eager | 9,069 | 9,356 | −3.1% | +1.6% |

The first three rows are the same 2,048 tokens arranged three ways. All three measured **3,001 MiB of
activations, identical to the MiB**, while `s²` spans 16× across them — which is how we know the
memory the old formula was missing is per-token, and not hiding in an `s²` term.

### Results — serving (`fitcheck infer`)

| Model | Config | Concurrent | Tensors err |
|:---|:---|---:|---:|
| TinyLlama-1.1B | fp16, seq=2048 | 1 | −2.8% |
| SmolLM2-1.7B | nf4, seq=2048 | 4 | −3.4% |
| Qwen2.5-1.5B | fp16, seq=2048 | 8 | −8.9% |
| TinyLlama-1.1B | fp16, seq=2048 | 16 | **−23.2%** |

**The KV cache formula is exact — +0.0% on all four**, including heavy GQA (Qwen2.5-1.5B, 2 KV heads)
and an NF4 base, with weights inside 0.1%. Every bit of the error above is the *transient* work of a
decode step, which `fitcheck infer` does not model at all: at 16 concurrent requests the peak was
3,650 MiB against 2,812 MiB resident, so 838 MiB of it. It grows with concurrency, which is the
direction that hurts, because nobody serves at concurrency 1. **`fitcheck infer` prints a warning
above four concurrent requests** rather than presenting the number as fact.

### Summary

| tier | max abs error | mean abs error |
|:---|---:|---:|
| **tensors** — the five physical formulas, all 23 training rows | **17.4%** | 2.2% |
| tensors, excluding the one seq-4096 row | **5.1%** | 1.5% |
| **process** — the full total | **22.2%** | 7.5% |

**How to read this.** The physical formulas are right to a few percent. `C_overhead` is not — the
process tier differs from the tensors tier only by that one heuristic, and essentially all of the
extra error is there. Its fragmentation model assumes a flat 5%; measured `reserved/allocated` ran
from **7% to 49%**, worst at long sequences and under eager attention, because big short-lived tensors
churn the allocator pool. The `_BASE_CONTEXT_MIB` constant of 500 is separately wrong — the T4
measured **133–147 MiB** on every one of the 33 runs — and the two errors partly cancel, so they have
to be fitted together or fixing one will visibly worsen the other. That is the next thing to fix.

The one tensors-tier row above 10% is an overhead row too: SmolLM2-360M at seq 4096 reserved
7,304 MiB against 4,909 allocated, the worst fragmentation this project has measured.

### What these runs changed

They were not a rubber stamp. **Five** things in the formulas were wrong, and every one was found by
measurement rather than by re-reading the derivation.

From the first twenty runs:

1. **`A_act` summed the LM-head hump and the layer recompute.** They never coexist — the FP32 logits
   are freed before the backward pass reaches a decoder layer — so the peak is a `max`, not a sum.
2. **The checkpoint store was `L·γbsh`.** Non-reentrant checkpointing keeps two hidden-state tensors
   per layer, not one, so it is `2L·γbsh`.
3. **The eager attention score matrix was billed once.** It is materialized about nine times across
   forward and backward, including two FP32 softmax copies — `9γ`, not `γ`.

Worst-case error across those twenty fell from **36.1% to 4.8%**.

From the thirteen newest runs:

4. **The no-checkpointing branch was wrong twice, in opposite directions.** The per-layer bracket was
   about 2.5× too small, *and* the `9γ` score matrix — a constant fitted with checkpointing on, where
   only one layer is ever live — was charged in all `L` layers at once. The over-count was the bigger
   of the two, so the total looked "safely too high" and the under-count stayed hidden, until the
   SDPA rows, which have no score matrix at all, exposed it alone at **−32.6%** — the unsafe
   direction. Measured: the bracket's hidden-width total is **15**, not 6, and a layer *retains*
   about **2.9** score-matrix copies when it is not the one being differentiated. Worst-case error on
   that branch fell from **98.3% to 5.9%**, and the checkpointed branch did not move.
5. **LoRA adapters are FP32 even on an unquantized base.** All 14 LoRA runs showed gradients at
   *exactly* −50% and adapter weights at exactly half — nine of them with `--quant none`. peft's
   `autocast_adapter_dtype=True` default upcasts adapters whenever the base is fp16 or bf16, not only
   when it is quantized, and a gradient follows its parameter's dtype. The project's own documented
   rule said "whenever the base is quantized"; measurement widened it to "always".

Two more results are worth recording because they are *not* bugs:

- **Full fine-tuning splits its terms differently from the harness, and the total is still exact.**
  fitcheck keeps the FP32 master weight copy inside `S_optim` at 12 bytes/param; the harness upcasts
  the parameters in place, so the master copy lands in weights instead. That reads as +50% / −50% /
  −50% across three terms and sums to **2,479 MiB against 2,479 measured, exact to the MiB**. Real
  mixed-precision training keeps a separate master copy, so fitcheck's split is the realistic one and
  the harness is the unusual one. Correcting any one of those terms would break a perfect total.
- **The `--double-quant` constant was settled by measuring, not by deriving.** Measured scale bytes:
  96 MiB without it, **24 MiB** with. The spec's derivation predicted 24.4; the shipped `×0.5`
  predicted 48. `W_base + W_lora` on that run moved from +1.9% to **−0.1%**.

The `9γ` result was found by running the same config twice, changing only the attention kernel: the
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

Be as clear about the gaps as about the results. This list is much shorter than it was — gradient
checkpointing off, `--quant none`, `--double-quant`, full fine-tuning, seq 4096 and the serving path
all have measured rows now — and what is left mostly needs hardware nobody here has.

- **Any GPU other than this T4.** All 33 runs are one Tesla T4 (sm_75). That means **no BF16 and no
  real FlashAttention-2** — sm_75 supports neither — so the flash code path is validated only through
  SDPA's memory-efficient backend as a stand-in. This is now the single largest gap in the project.
- **`--quant int8` beyond one model.** One TinyLlama run measured activations **−56.6%** low. The
  cause is identified: `prepare_model_for_kbit_training` upcasts the layer norms to FP32 and
  LLM.int8() takes that FP32 input at every linear, so fitcheck now bills int8 activations at `γ=4`
  regardless of `--precision`. That takes activations on the measured row from **−56.6% to −9.3%**,
  and the tensors tier from −38.9% to −5.7%. The rest is LLM.int8()'s own fp16 outlier buffers,
  which are not modelled. **fitcheck warns on `--quant int8`** and calls the
  figure a lower bound. One model is exactly how a 36% error happened here once before; a second int8
  row on a different model is owed before the mechanism can be called general.
- **Serving transients, and serving above 16 concurrent requests.** The KV cache formula is exact, but
  the decode-step transient is not modelled and reached −23.2% at 16 concurrent. No coefficient is
  fitted for it, deliberately: 838 MiB is far larger than any mechanism yet identified — the logits
  row is about 2 MiB at that shape — and one data point cannot settle it. `fitcheck infer` warns
  above four concurrent requests instead. Closing it properly needs a concurrency sweep.
- **Sequences beyond 4096**, where the eager coefficient multiplies an `s²` term.
- **MoE models**, which are refused rather than estimated — see [Model support](#model-support).
- **Every GPU entry except the T4.** The T4 used to claim 15,360 MiB usable of 16,384 when the card
  actually reports 14,912 MiB total — vendor GB treated as GiB, unsafe direction — and is now
  corrected to its measured `14_912 / 14_000`. It is the only row in `gpu_db.py` backed by a
  measurement. The rest are estimates, and `h200` / `b200` deliberately take the `usable_mib` that
  is safe under the pessimistic reading of their vendor GB.

One open question is recorded rather than hidden: the activation bracket's split between `h` and
`n_h·d_k` — shipped as `12h + 3·n_h·d_k` — is decided by **one model**. Gemma-2 is the only measured
model where those two widths differ, and it also has four layer norms per layer instead of two, which
pushes the same coefficient. One model, two effects. The hidden-width *total* of 15 is solid; the
split between the two halves is not, and a second model with `n_h·d_k ≠ h` would settle it.

Reproducing any of this needs one command:

```bash
python scripts/measure.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --qlora --precision fp16 --lora-r 32 --batch-size 2 --seq-len 1024 --gpu t4
```

`docs/SPEC.md` §3.8 explains what the harness does and why each part of it matters.

---

## How it works

`P`, the parameter count every weight-side term starts from, is the Hub's own `safetensors.total`.
A count derived from `config.json` runs alongside it as a cross-check, and a disagreement past 2%
is refused rather than reconciled — see [Model support](#model-support).

Peak VRAM is modelled as `W_base + W_lora + S_optim + G_grad + A_act + C_overhead`, one module
per term under [`fitcheck/memory/`](https://github.com/Anassbzdd/fitcheck/blob/main/fitcheck/memory/): base weights (only the transformer linears
are packed under `--quant` — the embeddings, LM head and norms stay unquantized and get upcast to
FP32, plus one FP32 NF4 scale per block of 64), LoRA adapters (`r × (d_in + d_out)` per target, with
`k_proj`/`v_proj` narrowed to `num_kv_heads × head_dim` under GQA), optimizer states (trainable
params only — 8 bytes/param for AdamW, whose states stay FP32 even when you train in BF16),
gradients, activations, and CUDA overhead. `estimator.py` orchestrates the six and returns a
`MemoryReport`.

Two dtype rules catch people out. **LoRA adapters are held in FP32 on every path** — not the bf16
you asked for — and gradients follow their parameter's dtype, so they are FP32 too. Two different
peft mechanisms land in the same place: `prepare_model_for_kbit_training` on a quantized base, and
`get_peft_model`'s `autocast_adapter_dtype=True` default on an unquantized fp16/bf16 one. Measured on
14 runs out of 14. And `--precision` is the **compute** dtype, driving activations, while `--quant`
is how the base is **stored** — separate axes on purpose. (The one place they interact:
`--quant int8` forces activations to FP32, because bitsandbytes feeds LLM.int8() FP32 inputs.)

Activations are the hard term and the one worth reading about. `A_layer` charges **15 hidden-width
tensors** per decoder layer plus `3 × intermediate_size` for the FFN — the count is measured, not
derived; the derivation in `docs/Blueprint.md` names six of the fifteen and the other nine are an
open question — plus the `(b, n_h, s, s)` attention score matrix when Flash Attention is off.
That score matrix has **two** coefficients, because it is two different quantities: with gradient
checkpointing on, one layer is live and peaks at about **9** copies of it; with checkpointing off,
every layer is live and each one *retains* about **2.9**. Using the first number in the second
situation over-estimated by 98%. `A_logits` is four FP32 copies of the `(b, s, V)` tensor, and for a
large-vocabulary model it is usually the single biggest line in the whole budget.

Under gradient checkpointing the peak is **not** a sum of those:

```
A_act = 2L·γ·b·s·h  +  max(A_logits, A_layer)
```

Only the checkpoints are resident for the whole backward pass. The LM-head hump and one layer's
recompute are both transient and never overlap, so the peak takes whichever is larger — adding them
prices a moment that never happens. This is measured, not assumed, and getting it wrong is what made
v0.1.1 under-predict by up to 36%. With checkpointing **off**, every layer keeps its own saved set
and the formula becomes `L × A_layer + A_logits` with the retained score-matrix count.

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
