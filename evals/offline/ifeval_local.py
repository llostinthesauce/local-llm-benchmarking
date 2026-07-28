#!/usr/bin/env python3
"""
Verifiable instruction following — IFEval methodology, zero download.

Google's IFEval works because every instruction it uses is checkable by code:
"write at least 400 words", "respond in valid JSON", "use no commas". There is
no judge model, no rubric, no API key, and therefore no way for the grader to be
wrong. That property is worth more on a local rig than the specific 541 prompts,
so this module reimplements the taxonomy with locally authored prompts and keeps
the real dataset available as a tier-2 eval (`evals/datasets/ifeval.py`).

Why it matters for quantization: instruction-following degrades before fluency
does. A 4-bit model that still writes clean prose will start ignoring "exactly
three bullet points" well before its perplexity looks suspicious.

Two scores are reported, matching the paper:
  strict — the response satisfies every constraint as written
  loose  — the same check after stripping markdown fences and boilerplate
           openers ("Sure! Here is..."), which otherwise fail formatting rules
           for reasons unrelated to instruction following
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Tuple

from ..core import Case, Score

Verifier = Callable[[str], bool]


# --------------------------------------------------------------------------
# Constraint verifiers. Each takes the response text and returns pass/fail.
# --------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def word_count_at_least(n: int) -> Verifier:
    return lambda text: len(_words(text)) >= n


def word_count_at_most(n: int) -> Verifier:
    return lambda text: len(_words(text)) <= n


def valid_json_with_keys(keys: Tuple[str, ...]) -> Verifier:
    def check(text: str) -> bool:
        candidate = _strip_code_fence(text).strip()
        try:
            obj = json.loads(candidate)
        except ValueError:
            return False
        return isinstance(obj, dict) and all(k in obj for k in keys)

    return check


def no_commas() -> Verifier:
    return lambda text: "," not in text


def all_lowercase() -> Verifier:
    return lambda text: text == text.lower()


def all_uppercase() -> Verifier:
    return lambda text: text == text.upper()


def ends_with(phrase: str) -> Verifier:
    return lambda text: text.strip().rstrip(".").strip().endswith(phrase.rstrip(".").strip())


def keyword_at_least(word: str, n: int) -> Verifier:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return lambda text: len(pattern.findall(text)) >= n


def forbidden_words(words: Tuple[str, ...]) -> Verifier:
    patterns = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
    return lambda text: not any(p.search(text) for p in patterns)


def exact_bullet_count(n: int) -> Verifier:
    pattern = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)
    return lambda text: len(pattern.findall(text)) == n


def exact_paragraph_count(n: int) -> Verifier:
    """Paragraphs separated by a literal *** divider, as IFEval specifies."""
    return lambda text: len([p for p in text.split("***") if p.strip()]) == n


def wrapped_in_double_quotes() -> Verifier:
    def check(text: str) -> bool:
        stripped = text.strip()
        return len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')

    return check


def placeholder_count_at_least(n: int) -> Verifier:
    pattern = re.compile(r"\[[^\]\n]+\]")
    return lambda text: len(pattern.findall(text)) >= n


def title_in_angle_brackets() -> Verifier:
    return lambda text: re.search(r"<<[^>\n]+>>", text) is not None


def no_bullet_lists() -> Verifier:
    return lambda text: re.search(r"^\s*[-*+]\s+\S", text, re.MULTILINE) is None


# --------------------------------------------------------------------------
# Loose-mode normalization
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)

# Two distinct shapes, kept separate on purpose:
#   interjections ("Sure!", "Certainly,") are self-delimiting and always safe
#   lead-ins ("Here is the config:") are only boilerplate when a colon marks
#     where the preamble ends — without one, "here is the answer" may well BE
#     the answer, and eating it would corrupt the text being graded.
_OPENERS = re.compile(
    r"^(?:"
    r"(?:sure|certainly|of course|absolutely|got it)[,!.]?"
    r"|(?:here(?:'s| is| are)|below is)[^\n:]{0,60}:"
    r")\s*",
    re.IGNORECASE,
)


def _strip_code_fence(text: str) -> str:
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text


def loosen(text: str) -> str:
    """Remove wrappers that fail formatting rules for reasons unrelated to
    whether the model followed the instruction.

    Openers are stripped repeatedly, not once: "Sure! Here is the answer:" is two
    stacked boilerplate phrases, and leaving the second one behind would still
    fail a strict `starts_with` or word-count constraint. Bounded so a response
    made entirely of openers cannot spin.
    """
    out = _strip_code_fence(text).strip()
    for _ in range(3):
        stripped = _OPENERS.sub("", out, count=1).strip()
        if stripped == out:
            break
        out = stripped
    return out.strip()


# --------------------------------------------------------------------------
# Prompt set
# --------------------------------------------------------------------------

_SPEC: List[Dict[str, object]] = [
    {
        "id": "json_config",
        "prompt": (
            "Output a JSON object describing a database connection pool. It must "
            "have exactly the keys \"host\", \"port\", \"max_connections\", and "
            "\"timeout_seconds\". Respond with the JSON only — no prose, no code fence."
        ),
        "constraints": [("valid_json_with_keys", valid_json_with_keys(
            ("host", "port", "max_connections", "timeout_seconds")))],
    },
    {
        "id": "no_commas_summary",
        "prompt": (
            "Explain what a write-ahead log does in a database. "
            "Write at least 60 words. Do not use any commas anywhere in your response."
        ),
        "constraints": [
            ("no_commas", no_commas()),
            ("word_count_at_least_60", word_count_at_least(60)),
        ],
    },
    {
        "id": "lowercase_ending",
        "prompt": (
            "Describe the trade-off between latency and throughput in one paragraph. "
            "Your entire response must be in lowercase letters — no capitals at all. "
            "Finish with exactly this sentence: that is the trade-off."
        ),
        "constraints": [
            ("all_lowercase", all_lowercase()),
            ("ends_with", ends_with("that is the trade-off")),
        ],
    },
    {
        "id": "three_bullets",
        "prompt": (
            "List reasons to prefer a mixture-of-experts model for local inference. "
            "Use exactly three bullet points, each starting with '- '. Nothing else."
        ),
        "constraints": [("exact_bullet_count_3", exact_bullet_count(3))],
    },
    {
        "id": "keyword_repetition",
        "prompt": (
            "Write a short paragraph about model quantization. "
            "Use the word 'precision' at least four times. Keep it under 120 words."
        ),
        "constraints": [
            ("keyword_precision_4", keyword_at_least("precision", 4)),
            ("word_count_at_most_120", word_count_at_most(120)),
        ],
    },
    {
        "id": "forbidden_terms",
        "prompt": (
            "Explain in at least 50 words how a KV cache speeds up autoregressive "
            "generation. You may not use the words 'cache', 'memory', or 'store' "
            "anywhere in your answer."
        ),
        "constraints": [
            ("forbidden_words", forbidden_words(("cache", "memory", "store"))),
            ("word_count_at_least_50", word_count_at_least(50)),
        ],
    },
    {
        "id": "two_paragraphs",
        "prompt": (
            "Compare dense and sparse transformer architectures. Write exactly two "
            "paragraphs separated by the divider *** on its own line."
        ),
        "constraints": [("exact_paragraph_count_2", exact_paragraph_count(2))],
    },
    {
        "id": "quoted_answer",
        "prompt": (
            "What is speculative decoding? Answer in one or two sentences. "
            "Wrap your entire response in double quotation marks."
        ),
        "constraints": [("wrapped_in_double_quotes", wrapped_in_double_quotes())],
    },
    {
        "id": "titled_placeholders",
        "prompt": (
            "Draft a short internal note about scheduling a benchmark run. "
            "Give it a title wrapped in double angle brackets, like <<Title Here>>. "
            "Include at least three placeholders in square brackets, like [date]."
        ),
        "constraints": [
            ("title_in_angle_brackets", title_in_angle_brackets()),
            ("placeholder_count_at_least_3", placeholder_count_at_least(3)),
        ],
    },
    {
        "id": "uppercase_short",
        "prompt": (
            "Name the single biggest bottleneck when running a large language model "
            "on a laptop. Your entire response must be in capital letters and "
            "at most 15 words."
        ),
        "constraints": [
            ("all_uppercase", all_uppercase()),
            ("word_count_at_most_15", word_count_at_most(15)),
        ],
    },
    {
        "id": "prose_no_bullets",
        "prompt": (
            "Explain why throughput benchmarks alone cannot detect a broken "
            "quantization. Write at least 80 words in flowing prose. "
            "Do not use any bulleted or numbered lists."
        ),
        "constraints": [
            ("no_bullet_lists", no_bullet_lists()),
            ("word_count_at_least_80", word_count_at_least(80)),
        ],
    },
    {
        "id": "json_no_prose",
        "prompt": (
            "Return a JSON object with the keys \"model\", \"quant\", and \"verdict\" "
            "summarizing a hypothetical benchmark result. Output valid JSON only, "
            "and use no commas inside any string value."
        ),
        "constraints": [
            ("valid_json_with_keys", valid_json_with_keys(("model", "quant", "verdict"))),
        ],
    },
]


def build_cases(limit: int = 0, **_ignored: object) -> List[Case]:
    """One Case per prompt spec. `limit` truncates for quick runs."""
    specs = _SPEC[:limit] if limit else _SPEC
    cases: List[Case] = []
    for spec in specs:
        cases.append(
            Case(
                case_id=f"ifeval_local_{spec['id']}",
                prompt=str(spec["prompt"]),
                max_tokens=768,
                temperature=0.0,
                meta={"spec_id": spec["id"], "constraint_count": len(spec["constraints"])},
            )
        )
    return cases


_BY_ID = {str(spec["id"]): spec for spec in _SPEC}


def score(case: Case, response: str) -> Score:
    """Fraction of constraints satisfied. `passed` requires strict compliance."""
    spec = _BY_ID.get(str(case.meta.get("spec_id")))
    if spec is None:
        return Score.binary(False, "unknown spec")

    constraints: List[Tuple[str, Verifier]] = spec["constraints"]  # type: ignore[assignment]
    loose_text = loosen(response)

    strict_hits, loose_hits, failed = 0, 0, []
    for name, verify in constraints:
        strict_ok = verify(response)
        loose_ok = strict_ok or verify(loose_text)
        strict_hits += int(strict_ok)
        loose_hits += int(loose_ok)
        if not loose_ok:
            failed.append(name)

    total = len(constraints)
    detail = "all constraints met" if not failed else "failed: " + ", ".join(failed)
    if strict_hits < loose_hits:
        detail += " (passed only after loose normalization)"
    return Score(value=loose_hits / total, passed=strict_hits == total, detail=detail)
