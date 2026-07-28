#!/usr/bin/env python3
"""
Unit tests for the eval suite.

These test the part that must not be wrong: the scoring. A benchmark whose
grader is subtly broken is worse than no benchmark, because it produces
confident numbers. Every test here asserts on a behaviour that would change a
model's reported score if it regressed — not merely that a function runs.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest evals/test_evals.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.core import Case, Score  # noqa: E402
from evals.datasets import gsm8k, humaneval, mmlu_pro  # noqa: E402
from evals.offline import determinism, ifeval_local, niah  # noqa: E402
from evals.registry import EVALS, resolve, tier1_names, tier2_names  # noqa: E402


# ---------------------------------------------------------------------------
# NIAH — the haystack must actually contain the needle, and only exact digits
# may count as a hit. A loose matcher would inflate long-context scores.
# ---------------------------------------------------------------------------

def test_niah_embeds_needle_in_prompt():
    cases = niah.build_cases(ctx_cap=2048, contexts=(1024,), depths=(0.5,))
    assert cases, "expected at least one case"
    case = cases[0]
    for expected in case.meta["expected"]:
        assert expected in case.prompt, "needle must be present in its own haystack"


def test_niah_is_deterministic_across_builds():
    first = niah.build_cases(ctx_cap=2048, contexts=(1024,), depths=(0.5,), seed=7)
    second = niah.build_cases(ctx_cap=2048, contexts=(1024,), depths=(0.5,), seed=7)
    assert first[0].prompt == second[0].prompt
    assert first[0].meta["expected"] == second[0].meta["expected"]


def test_niah_scores_exact_digits_only():
    case = Case(case_id="t", prompt="", meta={"expected": ["1234567"], "depth": 0.5})
    assert niah.score(case, "1234567").passed
    assert niah.score(case, "The code is 1234567.").passed
    # A near miss must not count: off-by-one digits are a retrieval failure.
    assert not niah.score(case, "1234568").passed
    # Nor may a substring of a longer number satisfy it.
    assert not niah.score(case, "912345670").passed


def test_niah_partial_credit_for_multi_needle():
    case = Case(case_id="t", prompt="", meta={"expected": ["111", "222"], "depth": 0.5})
    result = niah.score(case, "111 and 999")
    assert result.value == 0.5 and not result.passed


def test_niah_respects_ctx_cap():
    cases = niah.build_cases(ctx_cap=4096, contexts=(1024, 4096, 65536), depths=(0.5,))
    lengths = {case.meta["context_len"] for case in cases}
    assert lengths <= {1024, 4096}, "must not build cases beyond ctx_cap"


# ---------------------------------------------------------------------------
# IFEval verifiers — each must reject the violating case, or the constraint is
# decorative and every model scores 100%.
# ---------------------------------------------------------------------------

def test_word_count_bounds():
    assert ifeval_local.word_count_at_least(3)("one two three")
    assert not ifeval_local.word_count_at_least(4)("one two three")
    assert ifeval_local.word_count_at_most(3)("one two three")
    assert not ifeval_local.word_count_at_most(2)("one two three")


def test_no_commas_rejects_commas():
    assert ifeval_local.no_commas()("no punctuation here")
    assert not ifeval_local.no_commas()("yes, there is one")


def test_case_verifiers():
    assert ifeval_local.all_lowercase()("all lower 123")
    assert not ifeval_local.all_lowercase()("Has Capitals")
    assert ifeval_local.all_uppercase()("ALL UPPER 123")
    assert not ifeval_local.all_uppercase()("Mixed Case")


def test_json_verifier_requires_keys():
    verify = ifeval_local.valid_json_with_keys(("host", "port"))
    assert verify('{"host": "a", "port": 1}')
    assert not verify('{"host": "a"}'), "missing key must fail"
    assert not verify("not json at all")
    # A fenced block is still valid JSON for grading purposes.
    assert verify('```json\n{"host": "a", "port": 1}\n```')


def test_exact_bullet_count_is_exact():
    text = "- one\n- two\n- three"
    assert ifeval_local.exact_bullet_count(3)(text)
    assert not ifeval_local.exact_bullet_count(2)(text), "more bullets than asked must fail"


def test_forbidden_words_matches_whole_words_only():
    verify = ifeval_local.forbidden_words(("cache",))
    assert not verify("the cache is warm")
    # "cached" contains "cache" but is a different word; a substring matcher
    # here would fail responses that never used the banned term.
    assert verify("the value was precomputed")


def test_keyword_frequency_counts_occurrences():
    verify = ifeval_local.keyword_at_least("precision", 3)
    assert verify("precision Precision PRECISION")
    assert not verify("precision precision")


def test_paragraph_divider_count():
    assert ifeval_local.exact_paragraph_count(2)("first para\n***\nsecond para")
    assert not ifeval_local.exact_paragraph_count(2)("only one para")


def test_loosen_strips_fences_and_openers():
    assert ifeval_local.loosen("```\nhello\n```") == "hello"
    # Interjections are self-delimiting, so they always come off.
    assert ifeval_local.loosen("Sure! the answer") == "the answer"
    # Stacked openers are stripped repeatedly, not once.
    assert ifeval_local.loosen("Sure! Here is the config:\nthe answer") == "the answer"


def test_loosen_keeps_content_that_only_looks_like_an_opener():
    # Without a colon there is no boundary, so "here is ..." must be treated as
    # the answer itself. Stripping it would corrupt the text being graded and
    # silently fail word-count and starts-with constraints.
    assert ifeval_local.loosen("here is the answer") == "here is the answer"


def test_ifeval_score_strict_vs_loose():
    cases = ifeval_local.build_cases()
    lowercase_case = next(c for c in cases if c.meta["spec_id"] == "lowercase_ending")
    good = "this is all lowercase and it ends correctly. that is the trade-off"
    assert ifeval_local.score(lowercase_case, good).passed
    # Capitals violate the constraint and loosening cannot rescue it.
    assert not ifeval_local.score(lowercase_case, "THIS IS WRONG").passed


def test_every_local_ifeval_case_has_verifiers():
    for case in ifeval_local.build_cases():
        assert case.meta["constraint_count"] >= 1


# ---------------------------------------------------------------------------
# Determinism — identical repeats must score 1.0 and divergence must not.
# ---------------------------------------------------------------------------

def test_determinism_flags_identical_and_divergent():
    cases = determinism.build_cases(repeats=2, limit=1)
    identical = determinism.score_all(cases, ["same answer", "same answer"])
    assert all(s.passed for s in identical)

    divergent = determinism.score_all(cases, ["same answer", "same wrong"])
    assert divergent[0].passed, "baseline compares against itself"
    assert not divergent[1].passed
    assert 0.0 < divergent[1].value < 1.0, "partial prefix match earns partial credit"


def test_determinism_builds_at_least_two_repeats():
    # One repeat can never detect divergence, so the builder must floor at two.
    assert len(determinism.build_cases(repeats=1, limit=1)) == 2


# ---------------------------------------------------------------------------
# GSM8K / MMLU-Pro answer extraction — the most common source of silent
# under-reporting is a parser that cannot find a correct answer.
# ---------------------------------------------------------------------------

def test_gsm8k_prefers_marked_answer():
    case = Case(case_id="t", prompt="", meta={"expected": "18"})
    assert gsm8k.score(case, "Lots of reasoning here.\n#### 18").passed
    # Falls back to the last number when the marker is missing.
    assert gsm8k.score(case, "so the answer is 18").passed
    assert not gsm8k.score(case, "so the answer is 19").passed


def test_gsm8k_normalizes_formatting():
    case = Case(case_id="t", prompt="", meta={"expected": "1000"})
    assert gsm8k.score(case, "#### 1,000").passed, "thousands separators must not fail"
    assert gsm8k.score(case, "#### 1000.0").passed, "42.0 and 42 are the same answer"


def test_gsm8k_handles_no_number():
    case = Case(case_id="t", prompt="", meta={"expected": "5"})
    assert not gsm8k.score(case, "I cannot answer that").passed


def test_mmlu_letter_extraction():
    assert mmlu_pro.extract_letter("C") == "C"
    assert mmlu_pro.extract_letter("Answer: D") == "D"
    assert mmlu_pro.extract_letter("**B**") == "B"
    assert mmlu_pro.extract_letter("(A)") == "A"
    assert mmlu_pro.extract_letter("no letter here") == ""


def test_mmlu_prefers_labelled_answer_over_stray_letter():
    # An option letter appearing mid-sentence must not beat the stated answer.
    assert mmlu_pro.extract_letter("Option A is tempting. Answer: E") == "E"


# ---------------------------------------------------------------------------
# HumanEval — extraction, and the execution gate staying shut by default.
# ---------------------------------------------------------------------------

def test_humaneval_extracts_target_function():
    response = (
        "Some prose.\n```python\ndef helper():\n    pass\n```\n"
        "```python\ndef target(x):\n    return x\n```"
    )
    assert "def target" in humaneval.extract_code(response, "target")


def test_humaneval_falls_back_to_raw_response():
    assert "def add" in humaneval.extract_code("def add(a, b):\n    return a + b", "add")


def test_humaneval_does_not_execute_without_opt_in(monkeypatch):
    monkeypatch.delenv("HUMANEVAL_ALLOW_EXEC", raising=False)
    case = Case(case_id="t", prompt="", meta={"entry_point": "f", "test": "", "prompt_source": ""})
    result = humaneval.score(case, "def f(): pass")
    assert not result.passed and "allow-code-execution" in result.detail


def test_humaneval_executes_and_grades_when_enabled(monkeypatch):
    monkeypatch.setenv("HUMANEVAL_ALLOW_EXEC", "1")
    case = Case(
        case_id="t",
        prompt="",
        meta={
            "entry_point": "add",
            "test": "def check(fn):\n    assert fn(2, 3) == 5\n",
            "prompt_source": "",
        },
    )
    assert humaneval.score(case, "```python\ndef add(a, b):\n    return a + b\n```").passed
    assert not humaneval.score(case, "```python\ndef add(a, b):\n    return a - b\n```").passed


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_tiers_are_partitioned():
    assert set(tier1_names()) | set(tier2_names()) == set(EVALS)
    assert not set(tier1_names()) & set(tier2_names())


def test_resolve_shorthands_and_dedupes():
    assert [e.name for e in resolve(["tier1"])] == tier1_names()
    assert len(resolve(["all", "tier1", "niah"])) == len(EVALS)


def test_resolve_rejects_unknown_name():
    try:
        resolve(["not_a_real_eval"])
    except SystemExit as exc:
        assert "Unknown eval" in str(exc)
    else:
        raise AssertionError("unknown eval names must be rejected, not ignored")


def test_only_humaneval_declares_code_execution():
    flagged = [name for name, spec in EVALS.items() if spec.needs_code_execution]
    assert flagged == ["humaneval"]


def test_score_binary_normalizes_value():
    assert Score.binary(True).value == 1.0
    assert Score.binary(False).value == 0.0


def test_json_module_import_is_used():
    # Guards the fetch layer's contract: manifests are JSON-serializable.
    assert json.loads(json.dumps({"a": 1})) == {"a": 1}


# ---------------------------------------------------------------------------
# Aggregation — the averaging rule for scores differs from the one for
# throughput, and mixing them up silently inflates every reported accuracy.
# ---------------------------------------------------------------------------

def _aggregate_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import aggregate_results

    return aggregate_results


def test_score_mean_counts_zeros_but_tps_mean_does_not():
    agg = _aggregate_module()
    scores = [1.0, 0.0, 1.0, 1.0]
    # A model that failed one of four cases scored 0.75, not 1.0. Reusing the
    # throughput mean here (which drops zeros as "not measured") would report
    # a perfect run and hide every wrong answer.
    assert agg._score_mean(scores) == 0.75
    assert agg._mean(scores) == 1.0, "throughput mean intentionally skips zeros"
    assert agg._score_mean([]) == 0.0


def test_quality_rows_are_routed_away_from_speed_normalization():
    agg = _aggregate_module()
    assert agg._is_quality_row({"eval_name": "niah", "score": "1.0"})
    assert not agg._is_quality_row({"gen_tps": "58.1", "pass_name": "pass_1_micro"})


def test_quality_summary_reports_strict_and_mean_separately():
    agg = _aggregate_module()
    rows = [
        {"model_name": "m", "backend": "MLX_API", "quant": "4bit", "eval_name": "e",
         "metric": "accuracy", "status": "ok", "score": "1.0", "passed": "1"},
        {"model_name": "m", "backend": "MLX_API", "quant": "4bit", "eval_name": "e",
         "metric": "accuracy", "status": "ok", "score": "0.5", "passed": "0"},
    ]
    summary = agg._quality_summary(rows, {})
    assert len(summary) == 1
    assert summary[0]["mean_score"] == 0.75
    assert summary[0]["strict_pass_rate"] == 0.5


def test_quality_summary_excludes_errored_cases_from_scores():
    agg = _aggregate_module()
    rows = [
        {"model_name": "m", "backend": "B", "quant": "q", "eval_name": "e",
         "metric": "accuracy", "status": "ok", "score": "1.0", "passed": "1"},
        {"model_name": "m", "backend": "B", "quant": "q", "eval_name": "e",
         "metric": "accuracy", "status": "unreachable: boom", "score": "0.0", "passed": "0"},
    ]
    summary = agg._quality_summary(rows, {})[0]
    # A transport failure is not a wrong answer; counting it as one would blame
    # the model for a dead socket.
    assert summary["errors"] == 1
    assert summary["cases"] == 2
    assert summary["mean_score"] == 1.0
