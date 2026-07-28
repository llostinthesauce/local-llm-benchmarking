# Contributing

## Setup

```bash
pip install -r requirements.txt
python3 scripts/discover_models.py --roots ~/.lmstudio/models --write configs/models.local.json
python3 scripts/smoke_test.py
```

## Before opening a PR

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest evals/ webgui/ -q
python3 scripts/smoke_test.py
```

`smoke_test.py` is a static gate, not a correctness proof. It checks that
referenced scripts exist, every file compiles, prompts are not duplicated across
runners, CSV schemas agree, and — importantly — that no git-tracked file contains
a local path.

It also flags files that are neither tracked nor ignored. Those are one
`git add -A` away from being published without review, so add them to
`.gitignore` or commit them deliberately.

## Adding an eval

1. Write the module in `evals/offline/` (self-generating) or `evals/datasets/`
   (needs a download).
2. Export `build_cases(**kwargs) -> List[Case]` and `score(case, response) -> Score`.
   Accept `**_ignored` — the runner passes shared kwargs to every eval.
3. Register it in `evals/registry.py`.
4. Optionally export `summarize(rows) -> dict` for a breakdown in the report.
5. **Write tests for the scorer**, in `evals/test_evals.py`.

Step 5 is the one that matters. A benchmark whose grader is subtly wrong is worse
than no benchmark, because it produces confident numbers that nobody questions.
Every test should assert on something that would change a reported score if it
regressed. `test_niah_scores_exact_digits_only` is the model: it pins that a
near-miss digit sequence does *not* count as retrieval.

If an eval only makes sense across cases (like `determinism`, which compares
repeats), export `score_all(cases, responses) -> List[Score]` and set it on the
registry entry.

### Rules for new evals

- **No LLM judges.** Exact match, programmatic verification, or test execution.
  A judge model injects a second model's failures into your measurement and its
  verdicts change when it is updated, which makes historical runs incomparable.
- **Deterministic.** Seed anything generated, so two machines produce identical
  prompts.
- **Offline unless tier 2.** Tier-1 evals must not touch the network. Tier-2 evals
  must fail with an actionable message when `allow_fetch` is False, never hang on
  a socket.
- **Declare code execution.** Set `needs_code_execution=True` and gate on an
  explicit flag.

## Conventions

- Python 3.9-compatible syntax. `from __future__ import annotations` is already
  used throughout, so `list[str]` in annotations is fine — but avoid `match` and
  runtime `X | Y`.
- Standard library only in anything that must run before a virtualenv is active:
  `model_registry.py`, `serve_local.sh`, `webgui/serve.py`, and the `evals/` core.
- Prompts live in `scripts/prompts.py`. Never duplicate prompt text — the smoke
  test fails the build if you do, because a prompt that drifts between two runners
  silently makes their numbers incomparable.
- Files stay focused. 200–400 lines is typical; 800 is the ceiling.
- Comments explain *why*, especially where the code looks wrong but is not. The
  `_mean` / `_score_mean` split in `aggregate_results.py` is the canonical
  example — two averaging functions that look redundant and are not.

## Commits

```
<type>: <description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.

## Reporting benchmark results

If you are sharing numbers, include the quant, the backend, and the
`token_count_method`. Throughput computed from word-count fallback can read ~20%
high, so a comparison that omits it is not a comparison.
