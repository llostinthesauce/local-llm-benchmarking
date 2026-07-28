#!/usr/bin/env python3
"""
IFEval — Google's verifiable instruction-following set (Zhou et al., 2023).

541 prompts, each carrying machine-checkable constraints. This module runs the
real prompt set against the verifier implementations in
`evals/offline/ifeval_local.py`, covering the instruction ids this repo knows how
to check and skipping the rest rather than guessing.

That subsetting is deliberate and it means one thing you should keep in mind:
scores here are comparable *across models you run yourself*, but they are not
directly comparable to published IFEval numbers, which score all instruction
categories. The `coverage` field in the summary records exactly what fraction of
the set was scored.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from ..core import Case, Score
from ..offline import ifeval_local
from . import fetch

URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)
FILENAME = "ifeval_input_data.jsonl"


def _build_verifier(instruction_id: str, kwargs: dict):
    """Map an IFEval instruction id onto a local verifier, or None if unsupported."""
    args = kwargs or {}

    def get_int(*names: str):
        for name in names:
            value = args.get(name)
            if isinstance(value, int):
                return value
        return None

    if instruction_id == "length_constraints:number_words":
        count = get_int("num_words")
        relation = args.get("relation")
        if count is None:
            return None
        if relation == "at least":
            return ifeval_local.word_count_at_least(count)
        if relation == "less than":
            return ifeval_local.word_count_at_most(count)
        return None

    if instruction_id == "punctuation:no_comma":
        return ifeval_local.no_commas()

    if instruction_id == "change_case:english_lowercase":
        return ifeval_local.all_lowercase()

    if instruction_id == "change_case:english_capital":
        return ifeval_local.all_uppercase()

    if instruction_id == "startend:end_checker":
        phrase = args.get("end_phrase")
        return ifeval_local.ends_with(str(phrase)) if phrase else None

    if instruction_id == "startend:quotation":
        return ifeval_local.wrapped_in_double_quotes()

    if instruction_id == "keywords:frequency":
        keyword = args.get("keyword")
        count = get_int("frequency")
        relation = args.get("relation")
        if keyword and count is not None and relation == "at least":
            return ifeval_local.keyword_at_least(str(keyword), count)
        return None

    if instruction_id == "keywords:forbidden_words":
        words = args.get("forbidden_words")
        if isinstance(words, list) and words:
            return ifeval_local.forbidden_words(tuple(str(w) for w in words))
        return None

    if instruction_id == "detectable_format:number_bullet_lists":
        count = get_int("num_bullets")
        return ifeval_local.exact_bullet_count(count) if count is not None else None

    if instruction_id == "detectable_format:number_highlighted_sections":
        return None  # highlight syntax not implemented; skip rather than guess

    if instruction_id == "detectable_content:number_placeholders":
        count = get_int("num_placeholders")
        return ifeval_local.placeholder_count_at_least(count) if count is not None else None

    if instruction_id == "detectable_format:title":
        return ifeval_local.title_in_angle_brackets()

    if instruction_id == "detectable_format:json_format":
        # No key list in the spec: any valid JSON object satisfies it.
        return ifeval_local.valid_json_with_keys(())

    if instruction_id == "length_constraints:number_paragraphs":
        count = get_int("num_paragraphs")
        return ifeval_local.exact_paragraph_count(count) if count is not None else None

    return None


def _verifiers_for(row: dict) -> List[Tuple[str, Callable[[str], bool]]]:
    ids = row.get("instruction_id_list") or []
    kwargs_list = row.get("kwargs") or []
    out: List[Tuple[str, Callable[[str], bool]]] = []
    for index, instruction_id in enumerate(ids):
        kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        verifier = _build_verifier(str(instruction_id), kwargs or {})
        if verifier is not None:
            out.append((str(instruction_id), verifier))
    return out


_VERIFIERS: Dict[str, List[Tuple[str, Callable[[str], bool]]]] = {}
_STATS: Dict[str, int] = {"total_rows": 0, "scored_rows": 0}


def build_cases(limit: int = 150, seed: int = 20260728, allow_fetch: bool = False,
                **_ignored: object) -> List[Case]:
    path: Path = fetch.ensure(FILENAME, URL, allow_fetch)
    rows = fetch.read_jsonl(path)
    _STATS["total_rows"] = len(rows)

    scoreable = []
    for row in rows:
        verifiers = _verifiers_for(row)
        if verifiers:
            scoreable.append((row, verifiers))
    _STATS["scored_rows"] = len(scoreable)

    rng = random.Random(seed)
    if limit and limit < len(scoreable):
        scoreable = rng.sample(scoreable, limit)

    cases: List[Case] = []
    for index, (row, verifiers) in enumerate(scoreable):
        case_id = f"ifeval_{row.get('key', index)}"
        _VERIFIERS[case_id] = verifiers
        cases.append(
            Case(
                case_id=case_id,
                prompt=str(row["prompt"]),
                max_tokens=1024,
                temperature=0.0,
                meta={"instruction_ids": [name for name, _ in verifiers]},
            )
        )
    return cases


def score(case: Case, response: str) -> Score:
    verifiers = _VERIFIERS.get(case.case_id)
    if not verifiers:
        return Score(value=0.0, passed=False, detail="no supported verifiers")

    loose_text = ifeval_local.loosen(response)
    strict_hits, loose_hits, failed = 0, 0, []
    for name, verify in verifiers:
        strict_ok = verify(response)
        loose_ok = strict_ok or verify(loose_text)
        strict_hits += int(strict_ok)
        loose_hits += int(loose_ok)
        if not loose_ok:
            failed.append(name)

    total = len(verifiers)
    detail = "all met" if not failed else "failed: " + ", ".join(failed)
    return Score(value=loose_hits / total, passed=strict_hits == total, detail=detail)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = _STATS.get("total_rows", 0)
    scored = _STATS.get("scored_rows", 0)
    return {
        "coverage": {
            "dataset_prompts": total,
            "prompts_with_supported_verifiers": scored,
            "fraction": round(scored / total, 4) if total else 0.0,
            "note": "Not comparable to published IFEval; only supported instruction ids are scored.",
        }
    }
