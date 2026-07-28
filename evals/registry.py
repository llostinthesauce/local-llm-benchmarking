#!/usr/bin/env python3
"""
The eval catalog. Adding a benchmark means adding one entry here.

Tier 1 generates its own data and runs with no network — that is the default
suite, and it is what makes this harness useful on an air-gapped machine.
Tier 2 needs a public dataset and only runs after an explicit `--fetch`.
"""
from __future__ import annotations

from typing import Dict, List

from .core import Eval
from .datasets import gsm8k, humaneval, ifeval, mmlu_pro
from .offline import determinism, ifeval_local, niah

EVALS: Dict[str, Eval] = {
    # -- tier 1: offline, self-generating ----------------------------------
    "niah": Eval(
        name="niah",
        tier=1,
        description="Needle-in-a-haystack retrieval across the advertised context window",
        build_cases=niah.build_cases,
        score=niah.score,
        metric="retrieval_rate",
    ),
    "ifeval_local": Eval(
        name="ifeval_local",
        tier=1,
        description="Programmatically verifiable instruction following (IFEval method, local prompts)",
        build_cases=ifeval_local.build_cases,
        score=ifeval_local.score,
        metric="constraint_rate",
    ),
    "determinism": Eval(
        name="determinism",
        tier=1,
        description="Byte-identical output across repeats at temperature 0",
        build_cases=determinism.build_cases,
        score=determinism.score,
        score_all=determinism.score_all,
        metric="stability",
    ),
    # -- tier 2: public datasets, opt-in download --------------------------
    "gsm8k": Eval(
        name="gsm8k",
        tier=2,
        description="Grade-school math word problems, exact-match final answer",
        build_cases=gsm8k.build_cases,
        score=gsm8k.score,
    ),
    "humaneval": Eval(
        name="humaneval",
        tier=2,
        description="Python functional correctness, pass@1 by executing reference tests",
        build_cases=humaneval.build_cases,
        score=humaneval.score,
        metric="pass@1",
        needs_code_execution=True,
    ),
    "ifeval": Eval(
        name="ifeval",
        tier=2,
        description="Google IFEval prompt set scored with the supported verifiers",
        build_cases=ifeval.build_cases,
        score=ifeval.score,
        metric="constraint_rate",
    ),
    "mmlu_pro": Eval(
        name="mmlu_pro",
        tier=2,
        description="Ten-option multiple-choice knowledge across academic categories",
        build_cases=mmlu_pro.build_cases,
        score=mmlu_pro.score,
    ),
}

# Per-eval extra reporting, when the module provides it.
SUMMARIZERS = {
    "niah": niah.summarize,
    "determinism": determinism.summarize,
    "ifeval": ifeval.summarize,
    "mmlu_pro": mmlu_pro.summarize,
}


def tier1_names() -> List[str]:
    return [name for name, spec in EVALS.items() if spec.tier == 1]


def tier2_names() -> List[str]:
    return [name for name, spec in EVALS.items() if spec.tier == 2]


def resolve(names: List[str]) -> List[Eval]:
    """Expand names plus the 'tier1' / 'tier2' / 'all' shorthands."""
    selected: List[str] = []
    for name in names:
        if name == "all":
            selected.extend(EVALS)
        elif name == "tier1":
            selected.extend(tier1_names())
        elif name == "tier2":
            selected.extend(tier2_names())
        elif name in EVALS:
            selected.append(name)
        else:
            raise SystemExit(
                f"Unknown eval '{name}'. Available: {', '.join(sorted(EVALS))} "
                f"(or tier1, tier2, all)"
            )
    seen, ordered = set(), []
    for name in selected:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return [EVALS[name] for name in ordered]
