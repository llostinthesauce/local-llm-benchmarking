# Model Notes

Landscape survey as of **2026-07-28**, filtered to what is worth benchmarking on
a single machine. Sizes assume ~64 GB unified memory, which caps you around
45 GB of weights once you leave room for KV cache at long context.

Model claims below come from vendor announcements and model cards. Nothing here
is a measured result from this repo — that is what running it produces.

---

## Worth testing now

### Poolside Laguna XS 2.1 — 33B-A3B MoE

Released 2026-07-02, OpenMDW-1.1, 256K context. Purpose-built for agentic coding
on a single machine, which makes it the most direct rival to a Qwen3.6-35B-A3B
setup — same active-parameter budget, same use case, different training.

```bash
hf download poolside/Laguna-XS-2.1-GGUF --include "*Q4_K_M*" \
  --local-dir ~/.lmstudio/models/poolside/Laguna-XS-2.1-GGUF
```

Official GGUF from Poolside (Q4_K_M is their recommended local default);
community MLX at `AtomicChat/Laguna-XS-2.1-MLX-5bit`. Already in
`configs/model_catalog.json` as `laguna_xs_33b_moe` (aliases: `laguna`,
`laguna-xs`).

**Why benchmark it:** a same-size, same-purpose competitor is the cleanest
comparison you can run. Head-to-head on `gsm8k` + `humaneval` + throughput
answers whether the daily driver should change.

### Cohere North Mini Code 1.0 — 30B-A3B MoE

Released 2026-06-09, Apache-2.0, 256K context / 64K max output. Tuned for agentic
software engineering and terminal work.

> **Blocked on tooling.** The GGUF uses the `cohere2moe` architecture, which
> stock llama.cpp cannot load until PR #24260 merges. Build from that branch, or
> use the official `w4a16` / `fp8` checkpoints on a different runtime.

In the catalog as `north_mini_code_30b_moe` (aliases: `north`, `north-mini`).

### Gemma 4 — re-download what you already have

On **2026-07-15/16** Google republished the Gemma 4 weights **under the same
names, with no version bump**: chat-template fixes, tool-calling bug fixes, a
vision token-budget option (`max_soft_tokens` up to 1120 for sharper OCR), and
Flash Attention 4 support on Hopper GPUs.

Any Gemma 4 checkout pulled before 2026-07-16 is materially different from what
the repo now serves. The FA4 gains are NVIDIA-only and irrelevant on Apple
Silicon, but **the chat-template and tool-calling fixes are not** — those affect
every backend.

If you have not re-pulled since mid-July, do that before trusting any Gemma
number, old or new. It also makes a genuinely interesting benchmark: same name,
same quant, two different sets of weights, measured with the same harness.

---

## Too large for 64 GB

Everything else released in this window is frontier-scale. Listed so the survey is
complete, not because you can run them:

| Model | Org | Date | Size | Context |
|---|---|---|---|---|
| Kimi K3 | Moonshot AI | 2026-07-17 | 2.8T total | 1M |
| Inkling | Thinking Machines Lab | 2026-07-15 | 975B / 41B active | — |
| Hunyuan Hy3 | Tencent | 2026-07-06 | 295B / 21B active | 256K |
| Nemotron 3 Ultra | NVIDIA | 2026-06-04 | 550B / 55B active | 1M |
| GLM-5.2 | Z.ai | 2026-06-16 | 753B / ~40B active | 1M |
| DeepSeek V4 Pro | DeepSeek | 2026-04-24 | 1.6T / 49B active | 1M |

Hunyuan Hy3 is the closest to feasible at ~150 GB for 4-bit — still roughly triple
a 64 GB budget. A Thunderbolt-5 multi-Mac cluster is the only local path to these.

Mistral has confirmed a new "fat but sparse" MoE family entering early access in
July 2026, with a broader release expected later in the summer. Parameter count,
license, and benchmarks are all undisclosed. **Worth watching** — if it lands in
the 30–40B-total range it goes straight onto the test list.

---

## What to measure when you add one

A new model is not "better" because it is newer. Run the join:

```bash
bash scripts/serve_local.sh laguna --backend llamacpp &
python3 scripts/bench_quality.py --model laguna --evals tier1 tier2 --fetch
python3 scripts/bench_llamacpp_api.py --model laguna --passes micro normal high
python3 scripts/aggregate_results.py results/
```

Three questions the combined report answers that a throughput number cannot:

1. **Is the advertised context real?** `niah` by context length. A 256K claim
   that collapses at 32K changes which model you pick for large-repo work.
2. **Did the quant survive?** Compare `gsm8k` and `ifeval_local` across quants of
   the *same* model. Arithmetic and constraint-following break first.
3. **Is it reproducible?** `determinism` at temp 0. A stack that is fast but
   irreproducible makes every other number you collect noisy.

---

## Sources

- [LLM Releases tracker](https://www.llm-releases.com/)
- [July 2026 open-weight wave](https://www.digitalapplied.com/blog/open-weight-model-wave-july-2026-momentum-tracker)
- [poolside/Laguna-XS-2.1](https://huggingface.co/poolside/Laguna-XS-2.1) · [GGUF](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF) · [MLX 5-bit](https://huggingface.co/AtomicChat/Laguna-XS-2.1-MLX-5bit)
- [CohereLabs/North-Mini-Code-1.0](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) · [GGUF](https://huggingface.co/bartowski/North-Mini-Code-1.0-GGUF)
- [Gemma 4 stealth update](https://the-decoder.com/gemma-4-gets-a-stealth-update-that-fixes-tool-calling-bugs-and-truncated-responses-under-the-same-name/)
- [Mistral open-weight launch confirmation](https://aiweekly.co/alerts/mistral-confirms-open-weight-model-launch-as-arr-tops-400m)
