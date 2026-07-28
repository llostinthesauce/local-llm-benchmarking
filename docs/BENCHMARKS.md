# Benchmarks

Two independent measurements, joined at the end.

- **Speed** — four passes of increasing context and output length, recording
  throughput, time-to-first-token, and peak memory.
- **Quality** — seven evals, all scored by code. No judge model, no API key, no
  rubric that can drift between runs.

The join is the point. Either half alone will mislead you.

---

## Design rules

Everything here follows four rules, and they explain most of the odd decisions
elsewhere in the repo.

**1. The grader must not be able to be wrong.** Every eval is exact-match,
programmatic constraint checking, or test execution. An LLM judge introduces a
second model's failures into your measurement of the first, and its verdicts
change when the judge is updated — which makes historical runs incomparable.

**2. Offline by default.** Tier 1 generates its own data. A benchmark that pauses
to download has already invalidated its own timing, and an air-gapped machine is
a normal place to be evaluating local models.

**3. Deterministic.** Every generated eval is seeded. Two runs of `niah` on
different machines produce byte-identical prompts, so scores are comparable
across hardware.

**4. Report what was actually sent.** Prompt sizing uses an estimated tokens-per-word
ratio, but reports come from the server's own `prompt_tokens`. Estimates size the
work; measurements describe it.

---

## Speed passes

| Pass | Context | Output | Prompt |
|---|---|---|---|
| `micro` | 1K | 128 tok | Fibonacci function |
| `normal` | 16K | 1K tok | Async rate-limited API client |
| `high` | 64K budget | 2K tok | Django monolith → microservices architecture |
| `max` | fills `ctx_cap` | 4K tok | Distributed time-series DB spec, with synthetic filler to fill the window |

All prompt text lives in `scripts/prompts.py` and is imported everywhere. It is
never duplicated — `smoke_test.py` fails the build if it is, because a prompt
that drifts between two runners silently makes their numbers incomparable.

### Token counting honesty

Not every backend reports real token counts. Results carry a `token_count_method`
and the aggregator ranks it:

| Method | Rank | Meaning |
|---|---|---|
| `hf_tokenizer`, `mlx_native`, `llama_bench_native` | 5 | Real count |
| `openai_usage` | 4 | Server's `usage` block |
| `word_fallback*` | 1–2 | **Estimated from whitespace** |

Anything at rank ≤ 3 is flagged `approximate`, and the aggregator reports a
separate "trusted" winner alongside the raw fastest. A model whose throughput was
computed from word counts can look 20% faster than one measured properly — the
distinction is not pedantry.

---

## Quality evals

### Tier 1 — offline, self-generating

#### `niah` — needle-in-a-haystack

Builds a deterministic prose haystack at a target token length, inserts numeric
needles at fixed relative depths, and asks for them back. Scored on exact digit
match.

Reported per context length, which is the shape that shows *where* a window stops
working:

```json
"by_context": { "1024": 1.0, "16384": 1.0, "65536": 0.6, "131072": 0.2 }
```

That model has a real working context somewhere between 16K and 64K, whatever its
`ctx_cap` claims. The haystack is varied prose, not a repeated token: repetition
compresses in ways that make retrieval artificially easy.

#### `ifeval_local` — verifiable instruction following

Locally authored prompts, each carrying constraints a function can check: word
counts, valid JSON with required keys, no commas, exact bullet counts, forbidden
words, casing, `***`-separated paragraphs.

Two scores, matching the IFEval paper:

- **strict** — the response satisfies every constraint as written
- **loose** — the same check after stripping code fences and boilerplate openers
  ("Sure! Here is…"), which otherwise fail formatting rules for reasons unrelated
  to instruction following

Instruction-following degrades before fluency does. A 4-bit model that still
writes clean prose will start ignoring "exactly three bullet points" well before
anything else looks wrong.

#### `determinism` — reproducibility at temp 0

Same prompt, `temperature=0`, N repeats. Greedy decoding is a pure function;
identical inputs must give identical outputs. When they do not, something in the
stack is non-deterministic and every other number inherits that noise.

- **exact_match_rate** — fraction byte-identical to the first response
- **prefix_stability** — mean shared-prefix length, which localizes the drift.
  Divergence at token 3000 is a long-context or cache problem; at token 5 it is a
  sampler still sampling despite `temperature=0`.

### Tier 2 — public datasets, opt-in

Nothing downloads without `--fetch`. Files land in `evals/.cache/` (git-ignored)
and are recorded in a manifest with their SHA-256, so an upstream change to a
supposedly-immutable file gets surfaced loudly.

| Eval | Source | Scoring |
|---|---|---|
| `gsm8k` | openai/grade-school-math | Exact match on `#### <n>`, falling back to the last number |
| `humaneval` | openai/human-eval | pass@1 by executing the reference tests |
| `ifeval` | google-research IFEval | The `ifeval_local` verifiers, applied to the real prompt set |
| `mmlu_pro` | TIGER-Lab/MMLU-Pro via the HF datasets-server | Exact match on the chosen letter |

**Two caveats worth stating plainly:**

- `ifeval` scores only the instruction ids this repo implements verifiers for,
  and reports that fraction as `coverage`. Scores are comparable **across models
  you run yourself**, but not to published IFEval numbers.
- `mmlu_pro` samples a subset. Sampling is seeded, so the subset is identical
  across models — but a 150-question sample has real confidence intervals, and
  a 2-point gap between two models is noise.

#### HumanEval executes model-written code

That is inherent to the benchmark — functional correctness cannot be checked
without running the function. Mitigations: subprocess per problem, wall-clock
timeout, CPU and address-space rlimits, scratch working directory, and a guard
prelude that nulls `os.system`, `subprocess.*`, and `shutil.rmtree` in the child.

**None of that is a security boundary.** It stops a buggy completion from doing
damage. It does not stop a deliberately malicious one — the child is a normal
process running as you, with network and filesystem access. Evaluating a model
you do not trust belongs in a VM or container.

Requires `--allow-code-execution` on top of `--fetch`.

---

## Running them

```bash
# Offline suite
python3 scripts/bench_quality.py --model qwen35 --url http://127.0.0.1:8080/v1

# Everything, including downloads and code execution
python3 scripts/bench_quality.py --model qwen35 \
    --evals all --fetch --allow-code-execution

# Quick pass while iterating
python3 scripts/bench_quality.py --model qwen35 --limit 20

# See the plan without contacting anything
python3 scripts/bench_quality.py --model qwen35 --evals all --dry-run
```

Useful flags: `--ctx-cap` (bounds `niah`), `--limit` (caps cases per eval),
`--repeats` (determinism repeats), `--seed`, `--backend` / `--quant` (labels that
end up in the CSV and drive the join).

## Reading the output

```bash
python3 scripts/aggregate_results.py results/
```

Produces `summary.md` and `summary.json` with:

- **Speed × Accuracy** — the combined leaderboard, sorted by eval mean then
  throughput. Joined on `(family_id, quant)`, deliberately *not* on backend:
  accuracy is a property of the weights, so a family's scores apply to whichever
  engine served it fastest.
- **Quality Evals** — per model × eval, with case counts, errors, mean score, and
  strict pass rate
- **Serving Verdict** — fastest and *trusted-fastest* variants
- **MTP deltas** — speculative-decoding speedup where measured

### Two numbers that are easy to confuse

**Mean vs strict.** `mean_score` gives partial credit (3 of 4 needles found =
0.75). `strict_pass_rate` counts only fully-correct cases. A large gap means the
model is *nearly* right a lot — usually a formatting problem rather than a
knowledge one.

**Errors vs failures.** `errors` counts requests that never completed — a dead
socket, a timeout, an HTTP 500. Those are excluded from the score rather than
counted as wrong, because blaming the model for a transport failure understates
it. A run with a high error count should be re-run, not interpreted.

> One aggregation detail worth knowing, because it bit this repo during
> development: throughput averages skip zeros (0 t/s means "never measured"),
> while score averages must include them (0.0 means "got it wrong"). They use
> different helpers — `_mean` and `_score_mean` — and there is a regression test
> pinning the difference.
