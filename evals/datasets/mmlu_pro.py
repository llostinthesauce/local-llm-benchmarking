#!/usr/bin/env python3
"""
MMLU-Pro — harder, ten-option multiple choice knowledge (Wang et al., 2024).

MMLU itself is saturated and noisy; MMLU-Pro rebuilds it with ten options
instead of four and removes the items that were unanswerable or mislabeled.
Random guessing drops from 25% to 10%, which makes the signal far cleaner at the
sample sizes a local rig can actually afford to run.

Fetched through the Hugging Face datasets-server REST API rather than the
`datasets` library, so the only dependency is the standard library. The server
caps a request at 100 rows, so a sample is assembled from paged requests and each
page is cached separately.

Scoring is exact-match on the chosen letter. Models are asked to answer with a
bare letter; a trailing "Answer: C" is also accepted, since that formatting habit
is about instruction following, which the IFEval suite measures separately.
"""
from __future__ import annotations

import random
import re
import string
from typing import Dict, List

from ..core import Case, Score
from . import fetch

DATASET = "TIGER-Lab/MMLU-Pro"
PAGE_SIZE = 100
BASE_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET.replace('/', '%2F')}&config=default&split=test"
)

LETTERS = string.ascii_uppercase[:10]  # A-J

INSTRUCTION = (
    "Answer the following multiple-choice question. "
    "Respond with the single letter of the correct option and nothing else."
)

# "Answer: C", "**C**", "(C)", or a bare "C" — checked in that order.
_LABELLED = re.compile(r"(?:answer|final answer)\s*[:\-]?\s*\(?\*{0,2}([A-J])\b", re.IGNORECASE)
_BRACKETED = re.compile(r"\(([A-J])\)")
_BARE = re.compile(r"\b([A-J])\b")


def _pages_needed(limit: int) -> int:
    return max(1, (limit + PAGE_SIZE - 1) // PAGE_SIZE)


def build_cases(limit: int = 150, seed: int = 20260728, allow_fetch: bool = False,
                **_ignored: object) -> List[Case]:
    rows: List[dict] = []
    # Pull a few extra pages so the deterministic sample has room to choose from
    # a wider slice than a single contiguous block of the test split.
    for page in range(_pages_needed(limit) + 2):
        offset = page * PAGE_SIZE
        url = f"{BASE_URL}&offset={offset}&length={PAGE_SIZE}"
        try:
            payload = fetch.fetch_json(url, allow_fetch, f"mmlu_pro_p{page:03d}.json")
        except fetch.FetchDisabled:
            raise
        except Exception as exc:  # noqa: BLE001 - a partial fetch is still usable
            print(f"  WARNING: MMLU-Pro page {page} unavailable ({exc})")
            break
        page_rows = [entry.get("row", {}) for entry in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)

    rng = random.Random(seed)
    usable = [r for r in rows if r.get("options") and r.get("answer")]
    if limit and limit < len(usable):
        usable = rng.sample(usable, limit)

    cases: List[Case] = []
    for index, row in enumerate(usable):
        options = [str(o) for o in row["options"]]
        rendered = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options) if i < len(LETTERS))
        cases.append(
            Case(
                case_id=f"mmlu_pro_{row.get('question_id', index)}",
                prompt=f"{INSTRUCTION}\n\nQuestion: {row['question']}\n\n{rendered}\n\nAnswer:",
                max_tokens=16,
                temperature=0.0,
                meta={
                    "expected": str(row["answer"]).strip().upper()[:1],
                    "category": str(row.get("category", "unknown")),
                },
            )
        )
    return cases


def extract_letter(text: str) -> str:
    stripped = text.strip()
    for pattern in (_LABELLED, _BRACKETED):
        match = pattern.search(stripped)
        if match:
            return match.group(1).upper()
    match = _BARE.search(stripped)
    return match.group(1).upper() if match else ""


def score(case: Case, response: str) -> Score:
    expected = str(case.meta["expected"])
    got = extract_letter(response)
    if not got:
        return Score.binary(False, f"no letter found (expected {expected})")
    return Score.binary(got == expected, f"got {got}, expected {expected}")


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_category: Dict[str, List[float]] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        by_category.setdefault(category, []).append(float(row.get("score") or 0.0))
    return {
        "by_category": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_category.items()) if v
        }
    }
