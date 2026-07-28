#!/usr/bin/env python3
"""
Determinism — does the same prompt at temperature 0 give the same answer twice?

At temperature 0 with a fixed seed, greedy decoding is a pure function of the
weights and the KV state. Identical inputs must produce identical outputs. When
they do not, something in the serving stack is non-deterministic, and every other
number you measure inherits that noise:

  * non-deterministic Metal/CUDA reduction order in a fused kernel
  * a quantized KV cache whose dequant path depends on batch composition
  * continuous batching letting an unrelated request perturb attention
  * a sampler still sampling despite temperature=0 (a common server bug)
  * numerical instability introduced by an aggressive quant

This is cheap to run and catches real breakage that throughput and accuracy
benchmarks both miss — a stack can be fast and score fine on averages while
being irreproducible run to run.

Reported as two numbers per prompt group:
  exact_match_rate — fraction of repeats byte-identical to the first response
  prefix_stability — mean shared-prefix length as a fraction of response length,
                     which localizes *where* divergence starts. Drift at token
                     3000 is a long-context or cache problem; drift at token 5
                     is a sampler problem.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..core import Case, Score

# Deliberately varied: a short factual answer, a structured one, and a long
# generation. Divergence often shows up only as output length grows.
PROMPTS = [
    ("short_factual", "In one sentence, what problem does a bloom filter solve?", 128),
    ("structured", "List the five stages of a TCP connection teardown as a numbered list.", 256),
    ("code", "Write a Python function that merges two sorted lists into one sorted list.", 512),
    ("long_form", "Explain how gradient checkpointing trades compute for memory during training. Write about 400 words.", 900),
]

DEFAULT_REPEATS = 3


def build_cases(repeats: int = DEFAULT_REPEATS, limit: int = 0, **_ignored: object) -> List[Case]:
    """`repeats` copies of each prompt. Identical prompt, distinct case_id."""
    specs = PROMPTS[:limit] if limit else PROMPTS
    cases: List[Case] = []
    for group, prompt, max_tokens in specs:
        for index in range(max(2, repeats)):
            cases.append(
                Case(
                    case_id=f"determinism_{group}_r{index}",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    meta={"group": group, "repeat": index},
                )
            )
    return cases


def _shared_prefix_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    shared = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        shared += 1
    return shared / longest


def score_all(cases: List[Case], responses: List[str]) -> List[Score]:
    """Compare every repeat against the first response in its group."""
    baseline: Dict[str, str] = {}
    for case, response in zip(cases, responses):
        group = str(case.meta["group"])
        if int(case.meta["repeat"]) == 0 or group not in baseline:
            baseline.setdefault(group, response)

    scores: List[Score] = []
    for case, response in zip(cases, responses):
        group = str(case.meta["group"])
        reference = baseline.get(group, response)
        if int(case.meta["repeat"]) == 0:
            scores.append(Score(value=1.0, passed=True, detail="baseline repeat"))
            continue
        identical = response == reference
        ratio = _shared_prefix_ratio(reference, response)
        detail = "identical to baseline" if identical else f"diverged at {ratio:.1%} of response"
        scores.append(Score(value=1.0 if identical else ratio, passed=identical, detail=detail))
    return scores


def score(case: Case, response: str) -> Score:
    """Unused — determinism is only meaningful across repeats (see score_all)."""
    return Score(value=0.0, passed=False, detail="requires score_all")


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_group: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("group") or "?")].append(row)

    out: Dict[str, object] = {}
    for group, group_rows in sorted(by_group.items()):
        comparisons = [r for r in group_rows if int(r.get("repeat") or 0) > 0]
        if not comparisons:
            continue
        exact = sum(1 for r in comparisons if bool(r.get("passed")))
        out[group] = {
            "exact_match_rate": round(exact / len(comparisons), 4),
            "prefix_stability": round(
                sum(float(r.get("score") or 0.0) for r in comparisons) / len(comparisons), 4
            ),
        }
    return {"by_prompt": out}
