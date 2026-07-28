# Local LLM Benchmarking

**A benchmark harness for locally-served LLMs that measures both how fast a model runs and whether it still gives correct answers.**

Built for Apple Silicon, but every benchmark talks plain OpenAI-compatible HTTP, so it works against llama.cpp, MLX, Ollama, vLLM, or anything else that serves `/v1/chat/completions`.

```bash
git clone https://github.com/llostinthesauce/local-llm-benchmarking
cd local-llm-benchmarking
pip install -r requirements.txt
python3 scripts/discover_models.py --roots ~/.lmstudio/models --write configs/models.local.json
python3 bench_tui.py
```

---

## Why this exists

Most local-LLM benchmarking stops at tokens per second. That number is easy to
produce and it is genuinely useful — but on its own it cannot answer the question
people actually have, which is *"should I serve this quant?"*

A 3-bit quantization can be 40% faster than the 4-bit and still be the wrong
choice, because somewhere in that compression it started dropping carries in
arithmetic, ignoring "exactly three bullet points", or losing the middle of its
own context window. **None of those failures change the throughput number.** They
are invisible to every benchmark that only counts tokens.

So this repo measures both, and joins them into one table. The shape of that
report looks like this (**illustrative layout — not measured results; run it on
your own hardware to get real numbers**):

| Model | Quant | gen t/s | TTFT s | niah | ifeval | gsm8k | Eval mean |
|---|---|---:|---:|---:|---:|---:|---:|
| example_moe | Q4_K_XL | 59.3 | 1.84 | 1.000 | 0.917 | 0.884 | 0.934 |
| example_moe | Q3_K_M | 71.4 | 1.52 | 0.733 | 0.833 | 0.712 | 0.759 |

The second row is faster. It is also the one that quietly stopped retrieving from
long context. That trade is a decision — but you can only make it if you measured
both columns.

## What's in the box

**Four speed passes**, from a 1K-context smoke check to a run that fills the
model's full advertised context window:

| Pass | Context | Output | Intent |
|---|---|---|---|
| `micro` | 1K | 128 tok | Smoke check |
| `normal` | 16K | 1K tok | Practical coding prompt |
| `high` | 64K | 2K tok | Hard architecture prompt |
| `max` | fills `ctx_cap` | 4K tok | Long-context stress |

**Seven quality evals**, in two tiers. Every one is scored by code — no LLM
judge, no API key, no rubric that can drift between runs.

*Tier 1 generates its own data and runs fully offline:*

| Eval | Measures | Catches |
|---|---|---|
| `niah` | Needle-in-a-haystack retrieval across the context window | A `ctx_cap` that is advertised but not real |
| `ifeval_local` | Verifiable instruction following (IFEval method) | Constraint-following degrading before fluency does |
| `determinism` | Byte-identical output across repeats at temp 0 | Non-deterministic kernels, KV-cache bugs, samplers that ignore temp=0 |

*Tier 2 uses the standard public datasets, after an explicit `--fetch`:*

| Eval | Dataset | Scoring |
|---|---|---|
| `gsm8k` | Grade-school math word problems | Exact match on the final number |
| `humaneval` | 164 Python problems | pass@1 by running the reference tests |
| `ifeval` | Google's 541-prompt IFEval set | Programmatic constraint verifiers |
| `mmlu_pro` | Ten-option academic knowledge | Exact match on the chosen letter |

Nothing downloads unless you ask. The serving scripts launch with
`HF_HUB_OFFLINE=1` specifically so a benchmark can never silently pull weights
mid-run and invalidate its own numbers.

## Quickstart

**Requirements:** macOS on Apple Silicon (for the MLX backends), Python 3.9+,
`brew install llama.cpp` for GGUF, `pip install mlx mlx-lm mlx-vlm` for MLX.

```bash
pip install -r requirements.txt

# Point the registry at wherever your weights live:
python3 scripts/discover_models.py --roots ~/.lmstudio/models --write configs/models.local.json
python3 scripts/model_registry.py list

# Confirm the pipeline is sound before spending an hour on a run:
python3 scripts/smoke_test.py
```

Then either drive it interactively:

```bash
python3 bench_tui.py
```

or run the pieces directly:

```bash
# Serve a model (registry-driven; picks the right engine automatically)
bash scripts/serve_local.sh qwen35 --backend llamacpp

# Speed
python3 scripts/bench_llamacpp_api.py --model qwen35 --passes micro normal

# Quality — offline suite, nothing downloaded
python3 scripts/bench_quality.py --model qwen35 --url http://127.0.0.1:8080/v1

# Quality — add the public datasets (one-time download)
python3 scripts/bench_quality.py --model qwen35 --evals tier1 tier2 --fetch

# Join everything into one report
python3 scripts/aggregate_results.py results/
```

There is also a unified CLI and a zero-dependency browser GUI:

```bash
python3 llm.py serve qwen35     # or: llm.py list / status / bench / smoke
python3 llm.py web              # chat + model lifecycle at http://127.0.0.1:7860
```

## How it fits together

```
                     configs/model_catalog.json     (public: match rules)
                                 |
                    scripts/discover_models.py      (scan disk)
                                 |
                     configs/models.local.json      (local: real paths, git-ignored)
                                 |
                    scripts/model_registry.py       (alias -> path + engine + sampling)
                        /                 \
        scripts/serve_local.sh          bench_tui.py / llm.py
        (llama.cpp | mlx_lm | mlx_vlm)          |
                        \                 /
                    OpenAI-compatible HTTP endpoint
                        /                 \
            scripts/bench_*.py          scripts/bench_quality.py
              (throughput)                 (evals/ suite)
                        \                 /
                  scripts/aggregate_results.py
                        summary.md + summary.json
```

The registry is the hinge. One alias (`qwen35`) resolves to a path, a context
cap, sampling defaults, and the engine that can actually load it — and serving,
benchmarking, and the GUI all read the same entry, so they cannot drift apart.

## Adding a model

Two files, and usually only one of them:

```bash
# 1. Put the weights under your model root
hf download poolside/Laguna-XS-2.1-GGUF --local-dir ~/.lmstudio/models/poolside/Laguna-XS-2.1-GGUF

# 2. Re-scan
python3 scripts/discover_models.py --roots ~/.lmstudio/models --write configs/models.local.json
```

Anything unrecognized is still registered as a `custom` family, so a new model
is benchmarkable immediately. To give it a proper name, context cap, and sampling
defaults, add a family to `configs/model_catalog.json`. Full walkthrough in
**[docs/ADDING_MODELS.md](docs/ADDING_MODELS.md)**.

## Safety notes

- **`humaneval` executes code the model under test wrote.** That is inherent to
  the benchmark. Mitigations (subprocess, rlimits, timeout, scratch dir, guard
  prelude) stop *buggy* completions from doing damage; they are **not** a
  security boundary against a deliberately malicious one. It is off by default
  and needs `--allow-code-execution`. Evaluating an untrusted model belongs in a
  VM or container.
- **The web GUI is unauthenticated** and can start and stop model servers. It
  binds `127.0.0.1` and refuses anything else without `--allow-remote`.
- **No secrets, no personal paths.** `scripts/smoke_test.py` fails the build if a
  git-tracked file contains a local path, and flags untracked files that would
  leak one if committed.

## Documentation

| Doc | What's in it |
|---|---|
| **[docs/ADDING_MODELS.md](docs/ADDING_MODELS.md)** | Registering new weights, catalog fields, engine routing |
| **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** | What each eval measures, how it's scored, how to read the output |
| **[docs/MODEL_NOTES.md](docs/MODEL_NOTES.md)** | Current model landscape and what's worth testing |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Tests, conventions, adding an eval |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest evals/ webgui/ -q
python3 scripts/smoke_test.py
```

The eval tests assert on scoring behaviour specifically — a benchmark whose
grader is subtly wrong is worse than no benchmark, because it produces confident
numbers.

## License

MIT
