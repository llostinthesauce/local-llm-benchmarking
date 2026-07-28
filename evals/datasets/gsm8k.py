#!/usr/bin/env python3
"""
GSM8K — grade-school math word problems (Cobbe et al., 2021).

The canonical multi-step arithmetic reasoning benchmark: 1,319 test problems,
each with a single integer answer after a `####` marker. Scoring is exact match
on that number, so there is no judge and no ambiguity.

Useful locally because arithmetic is the first thing aggressive quantization
breaks. A model that still writes fluent prose at 3-bit will start dropping
carries, and GSM8K surfaces that as a double-digit accuracy drop while
throughput looks unchanged.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List

from ..core import Case, Score
from . import fetch

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
FILENAME = "gsm8k_test.jsonl"

# Answers appear as "#### 42" at the end of the reference solution.
_GOLD = re.compile(r"####\s*(-?[\d,]+)")
# Model answers: take the last number, which is where the final answer lands
# whether or not the model was asked to show work.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

INSTRUCTION = (
    "Solve the problem. Reason step by step, then give the final numeric answer "
    "on its own final line in the form: #### <number>"
)


def build_cases(limit: int = 200, seed: int = 20260728, allow_fetch: bool = False,
                **_ignored: object) -> List[Case]:
    path: Path = fetch.ensure(FILENAME, URL, allow_fetch)
    rows = fetch.read_jsonl(path)

    # Deterministic sample so partial runs stay comparable across models.
    rng = random.Random(seed)
    if limit and limit < len(rows):
        rows = rng.sample(rows, limit)

    cases: List[Case] = []
    for index, row in enumerate(rows):
        match = _GOLD.search(str(row.get("answer", "")))
        if not match:
            continue
        cases.append(
            Case(
                case_id=f"gsm8k_{index:04d}",
                prompt=f"{INSTRUCTION}\n\nProblem: {row['question']}",
                max_tokens=768,
                temperature=0.0,
                meta={"expected": match.group(1).replace(",", "")},
            )
        )
    return cases


def _final_number(text: str) -> str:
    """Prefer the #### marker; otherwise fall back to the last number seen."""
    marked = _GOLD.search(text)
    if marked:
        return marked.group(1).replace(",", "")
    numbers = _NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else ""


def _normalize(value: str) -> str:
    """Compare numerically so 42, 42.0 and 042 all match."""
    try:
        number = float(value)
    except ValueError:
        return value.strip()
    return str(int(number)) if number.is_integer() else str(number)


def score(case: Case, response: str) -> Score:
    expected = str(case.meta["expected"])
    got = _final_number(response)
    if not got:
        return Score.binary(False, f"no number in response (expected {expected})")
    passed = _normalize(got) == _normalize(expected)
    return Score.binary(passed, f"got {got}, expected {expected}")
