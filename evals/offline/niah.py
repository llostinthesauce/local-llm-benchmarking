#!/usr/bin/env python3
"""
Needle-in-a-Haystack / RULER-lite — does the advertised context window work?

Every model card claims a context length and every registry entry copies it into
`ctx_cap`. That number describes what the weights *can* address; it does not
prove the served stack — quantized KV cache, RoPE scaling, a llama.cpp `-c`
flag, a chunked-prefill bug — still retrieves from the far end of it. Throughput
benchmarks never catch this, because a model that has silently lost the middle of
its context generates tokens at exactly the same speed.

Method (Kamradt's NIAH, extended toward RULER's multi-key variant):
  1. build a deterministic prose haystack of a target token length
  2. insert one or more "needles" at fixed relative depths
  3. ask for the needle values back
  4. score exact match on the digits

The haystack is generated from a seeded PRNG over a fixed vocabulary, so runs are
byte-identical across machines with no download. It is deliberately varied prose
rather than a repeated token: repetition compresses in ways that make retrieval
artificially easy and does not resemble the long documents users actually paste.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..core import Case, Score

# Roughly how many BPE tokens one whitespace word costs for the filler prose
# below. Calibrated against the served prompt_tokens the API reports back
# (Gemma 4 / Qwen3 SentencePiece-style vocabularies land near 1.15); it only
# sizes the haystack, and the report always shows the server's own
# prompt_tokens as ground truth for what was actually sent.
TOKENS_PER_WORD = 1.15

# Fraction of a context budget held back for the instruction wrapper and the
# answer. Applied proportionally rather than as a flat count: a fixed 512-token
# reserve consumes half of a 1024-token case, which would quietly turn a "1K
# context" data point into a 500-token one.
RESERVE_FRACTION = 0.12
MIN_RESERVE_TOKENS = 96
MAX_RESERVE_TOKENS = 1024


def _reserve_for(context_len: int) -> int:
    return int(min(MAX_RESERVE_TOKENS, max(MIN_RESERVE_TOKENS, context_len * RESERVE_FRACTION)))

SUBJECTS = [
    "the archivist", "a field technician", "the night operator", "our supplier",
    "the review board", "a visiting researcher", "the logistics team", "her deputy",
    "the standards committee", "a contract auditor", "the harbor master", "their analyst",
]
VERBS = [
    "documented", "revised", "questioned", "approved", "deferred", "catalogued",
    "escalated", "rejected", "annotated", "forwarded", "reconciled", "withdrew",
]
OBJECTS = [
    "the quarterly variance report", "an unlabeled inventory crate",
    "the revised shipping manifest", "three unresolved maintenance tickets",
    "the calibration log for bay four", "a discrepancy in the ledger",
    "the updated evacuation route", "her notes from the site visit",
    "the vendor's amended invoice", "a draft of the retention policy",
    "the sealed envelope from legal", "an overdue equipment recall",
]
CLAUSES = [
    "before the end of the fiscal quarter", "without notifying the regional office",
    "after the second inspection failed", "despite the earlier objection",
    "pending confirmation from the registrar", "under the revised handling procedure",
    "while the primary system was offline", "in accordance with the 1998 amendment",
    "once the backup generator came online", "as the weather window closed",
]

# Needle topics kept semantically distinct so a multi-needle prompt cannot be
# answered by pattern-matching a single nearby number.
NEEDLE_TOPICS = [
    ("Lisbon", "harbor access"), ("Tampere", "cold storage"), ("Nagoya", "freight dock"),
    ("Valparaiso", "customs gate"), ("Reykjavik", "fuel depot"), ("Kigali", "transit hub"),
]

DEFAULT_DEPTHS = (0.05, 0.25, 0.5, 0.75, 0.95)
DEFAULT_CONTEXTS = (1024, 4096, 16384, 65536, 131072)


def _sentence(rng: random.Random) -> str:
    return (
        f"{rng.choice(SUBJECTS).capitalize()} {rng.choice(VERBS)} "
        f"{rng.choice(OBJECTS)} {rng.choice(CLAUSES)}."
    )


def _haystack_words(target_tokens: int, rng: random.Random) -> List[str]:
    """Generate prose until it is approximately `target_tokens` long."""
    target_words = max(1, int(target_tokens / TOKENS_PER_WORD))
    words: List[str] = []
    while len(words) < target_words:
        words.extend(_sentence(rng).split())
    return words[:target_words]


def _needle_text(city: str, kind: str, code: int) -> str:
    return f"Note: the {kind} clearance code for {city} is {code}."


def _insert_at_depths(
    words: List[str], needles: List[Tuple[str, str, int]], depths: List[float]
) -> str:
    """Splice needles into the word stream at the given relative depths.

    Inserting back-to-front keeps earlier indices valid as the list grows.
    """
    placements = sorted(
        ((int(depth * len(words)), needle) for depth, needle in zip(depths, needles)),
        key=lambda item: item[0],
        reverse=True,
    )
    out = list(words)
    for index, (city, kind, code) in placements:
        out[index:index] = _needle_text(city, kind, code).split()
    return " ".join(out)


def build_cases(
    ctx_cap: int = 131072,
    contexts: Tuple[int, ...] = DEFAULT_CONTEXTS,
    depths: Tuple[float, ...] = DEFAULT_DEPTHS,
    needles: int = 1,
    seed: int = 20260728,
    **_ignored: object,
) -> List[Case]:
    """One Case per (context length x depth) that fits inside ctx_cap.

    A proportional reserve leaves room for the instruction wrapper and the
    answer so a case never overflows the window it is meant to be testing.
    """
    cases: List[Case] = []
    usable = [c for c in contexts if c <= ctx_cap]
    if not usable:
        usable = [max(512, min(contexts[0], ctx_cap))]

    for context_len in usable:
        haystack_tokens = max(128, context_len - _reserve_for(context_len))
        for depth in depths:
            rng = random.Random(f"{seed}:{context_len}:{depth}:{needles}")
            words = _haystack_words(haystack_tokens, rng)

            topics = rng.sample(NEEDLE_TOPICS, k=min(needles, len(NEEDLE_TOPICS)))
            chosen = [(city, kind, rng.randint(1000000, 9999999)) for city, kind in topics]

            # Spread multiple needles evenly around the requested depth so a
            # multi-needle case probes the whole window, not one hot spot.
            if len(chosen) == 1:
                placed_depths = [depth]
            else:
                span = 0.8 / len(chosen)
                placed_depths = [min(0.98, max(0.02, depth - 0.4 + span * i)) for i in range(len(chosen))]

            body = _insert_at_depths(words, chosen, placed_depths)
            wanted = ", ".join(f"the {kind} clearance code for {city}" for city, kind, _ in chosen)
            prompt = (
                "Below is a long document. Read it carefully.\n\n"
                "<document>\n" + body + "\n</document>\n\n"
                f"Question: What is {wanted}?\n"
                "Answer with the number only. If there are several, list them "
                "comma-separated in the order asked. Do not explain."
            )
            cases.append(
                Case(
                    case_id=f"niah_ctx{context_len}_d{int(depth * 100):02d}_n{len(chosen)}",
                    prompt=prompt,
                    max_tokens=96,
                    temperature=0.0,
                    meta={
                        "expected": [str(code) for _, _, code in chosen],
                        "context_len": context_len,
                        "depth": depth,
                        "needles": len(chosen),
                    },
                )
            )
    return cases


def score(case: Case, response: str) -> Score:
    """Fraction of needles whose exact digits appear in the answer.

    Substring matching rather than strict equality: models reliably retrieve the
    value but ignore "number only", and formatting compliance is what the IFEval
    suite measures. Here we are testing retrieval.
    """
    expected: List[str] = case.meta["expected"]
    if not expected:
        return Score.binary(False, "no expected values")
    digits = "".join(ch if ch.isdigit() else " " for ch in response)
    found = {token for token in digits.split() if token}
    hits = [value for value in expected if value in found]
    value = len(hits) / len(expected)
    detail = f"{len(hits)}/{len(expected)} needles at depth {case.meta['depth']:.2f}"
    return Score(value=value, passed=len(hits) == len(expected), detail=detail)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Per-context-length retrieval rate — the shape that exposes where a
    context window actually stops working."""
    by_context: Dict[int, List[float]] = {}
    for row in rows:
        context_len = int(row.get("context_len") or 0)
        by_context.setdefault(context_len, []).append(float(row.get("score") or 0.0))
    return {
        "by_context": {
            str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_context.items()) if v
        }
    }
