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

In `configs/model_catalog.json` as `laguna_xs_33b_moe` (aliases: `laguna`,
`laguna-xs`). Discovery matches it automatically once the file is on disk.

> **Tested 2026-07-28 and it does not work through the chat endpoint on
> llama.cpp b10090.** The model loads and generates; the OpenAI-compatible chat
> API returns empty content.

What we measured on `Laguna-XS-2.1-Q4_K_M.gguf`:

| Probe | Result |
|---|---|
| Model loads | yes — `general.architecture = laguna`, pre-tokenizer `laguna` |
| `POST /completions` (raw, no template) | **works** — correct Python, ~75–82 tok/s |
| `POST /v1/chat/completions` | `content: ""`, `finish_reason: "length"`, full token budget consumed |
| With `reasoning_format: "none"` | `content: "<think>"` and nothing further |
| `chat_template_kwargs: {enable_thinking: false}` | no change |
| Empty system message (template's documented opt-out) | no change |
| Server-level `--reasoning-format none` | no change |

Two things in the GGUF explain it. The embedded template ends every prompt with
`<assistant><think>` — it opens a thinking block unconditionally, ignoring its
own `enable_thinking` default of `false`. The model then emits a second literal
`<think>` and never closes it, so llama.cpp's reasoning parser consumes the rest
of the generation and hands back nothing. Compounding it, the metadata sets
`bos_token_id = 2` and `eos_token_id = 2` — **the same id for both** — with
`add_bos_token = true`.

**Practical status:** benchmarkable through `/completions` only, which this
harness does not use (every eval speaks the chat API, deliberately, because that
is how the model gets served in practice). Re-test when either llama.cpp gains
a `laguna` chat-format handler or Poolside ships a corrected template. The
community MLX build (`AtomicChat/Laguna-XS-2.1-MLX-5bit`, ~23 GB) uses a
different template path and has not been tested here — that is the next thing to
try.

### Cohere North Mini Code 1.0 — 30B-A3B MoE

Released 2026-06-09, Apache-2.0, 256K context / 64K max output. Tuned for agentic
software engineering and terminal work.

> **Blocked on tooling.** The GGUF uses the `cohere2moe` architecture, which
> stock llama.cpp cannot load until PR #24260 merges. Build from that branch, or
> use the official `w4a16` / `fp8` checkpoints on a different runtime.

In the catalog as `north_mini_code_30b_moe` (aliases: `north`, `north-mini`).

### Gemma 4 — the re-pull, actually measured

On **2026-07-15/16** Google republished Gemma 4 **under the same names with no
version bump**: chat-template fixes, tool-calling fixes, a vision token-budget
option, and Flash Attention 4 on Hopper GPUs. The obvious advice is "re-pull
everything before trusting any Gemma number."

We tested that advice on `unsloth/gemma-4-26B-A4B-it-qat-GGUF` (UD-Q4_K_XL),
comparing a 2026-06-12 checkout against a fresh 2026-07-28 download:

| | Stale (Jun 12) | Fresh (Jul 28) |
|---|---|---|
| File size | 14,249,045,120 B | 14,249,047,104 B |
| `ifeval_local` scored / truncated | 2 / 2 | 2 / 2 |
| Which prompts truncated | cases 2, 3 | cases 2, 3 |
| Per-case latency | ~50 s | ~50 s |

**The file changed by 1,984 bytes. The behaviour did not** — same prompts
truncate, same scores, same timings. So for this community GGUF, the re-pull
bought nothing measurable. Either the requantization predates Google's change,
or the change does not survive into this quant.

Take this as the general lesson: *"newer weights are better" is a hypothesis,
and this repo exists to test hypotheses like it rather than repeat them.* The
advice may still hold for the official Google checkpoints or the MLX
conversions — those have not been tested here.

**Separately, and reproducibly:** Gemma 4 26B A4B enters very long reasoning on
constraint-following prompts. Asked to write a paragraph without commas, it
consumed a full 3,072-token budget in `reasoning_content` and emitted no answer
at all — on both stale and fresh weights. Budget accordingly
(`bench_quality.py --token-scale 4` or higher), and see
[BENCHMARKS.md](BENCHMARKS.md) on why the harness reports these as unscored
rather than wrong.

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
