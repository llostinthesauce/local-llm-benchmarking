#!/usr/bin/env python3
"""
bench_quality.py — run the quality evals against a served model.

Speed benchmarks and quality evals answer different questions, and only the pair
is decision-grade: 60 tok/s is not a win if the quant lost 12 points of GSM8K.
This runner writes a CSV with the same run/model/backend columns the speed
benchmarks use, so `aggregate_results.py` can join them into one leaderboard.

Examples
--------
    # Offline suite against a running llama-server, nothing downloaded:
    python3 scripts/bench_quality.py --model qwen35 --url http://127.0.0.1:8080/v1

    # Add the public datasets (one-time download into evals/.cache/):
    python3 scripts/bench_quality.py --model qwen35 --evals tier1 tier2 --fetch

    # Include HumanEval, which executes model-written code (read the warning):
    python3 scripts/bench_quality.py --model qwen35 --evals humaneval \\
        --fetch --allow-code-execution

    # See what would run without touching a server:
    python3 scripts/bench_quality.py --model qwen35 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.core import Case, EvalClient, Score  # noqa: E402
from evals.datasets.fetch import FetchDisabled  # noqa: E402
from evals.registry import EVALS, SUMMARIZERS, resolve  # noqa: E402

FIELDS = [
    "timestamp", "run_id", "model_name", "backend", "eval_name", "case_id",
    "score", "passed", "metric", "detail", "prompt_tokens", "completion_tokens",
    "latency_s", "status", "quant", "context_len", "depth", "group", "repeat",
    "category", "response_chars",
]

DEFAULT_EVALS = ["tier1"]


def _model_slug(model: str) -> str:
    """Filename-safe short name.

    mlx_vlm.server requires the model id to be an absolute filesystem path, so
    `--model` is routinely a full path under the user's home. Using it verbatim
    would stamp that path into every result filename, so take the basename.
    """
    name = model.rstrip("/").split("/")[-1] or "model"
    return "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in name)[:80]


def _csv_append(rows: List[Dict[str, Any]], path: Path) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _case_row(case: Case, score: Score, response, eval_name: str, metric: str,
              base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **base,
        "eval_name": eval_name,
        "case_id": case.case_id,
        "score": round(score.value, 4),
        "passed": int(score.passed),
        "metric": metric,
        "detail": score.detail[:300],
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_s": round(response.latency_s, 3),
        "status": "ok" if response.ok else response.error[:120],
        "context_len": case.meta.get("context_len", ""),
        "depth": case.meta.get("depth", ""),
        "group": case.meta.get("group", ""),
        "repeat": case.meta.get("repeat", ""),
        "category": case.meta.get("category", ""),
        "response_chars": len(response.text),
    }


def run_eval(spec, client: EvalClient, base: Dict[str, Any], build_kwargs: Dict[str, Any],
             dry_run: bool, verbose: bool) -> Dict[str, Any]:
    """Execute one eval end to end and return its summary block."""
    try:
        cases = spec.build_cases(**build_kwargs)
    except FetchDisabled as exc:
        print(f"  SKIP {spec.name}: {exc}")
        return {"eval": spec.name, "status": "skipped_no_data", "cases": 0}
    except Exception as exc:  # noqa: BLE001 - one bad eval must not kill the run
        print(f"  SKIP {spec.name}: build failed ({exc})")
        return {"eval": spec.name, "status": f"skipped_build_error:{exc}", "cases": 0}

    if not cases:
        print(f"  SKIP {spec.name}: produced no cases")
        return {"eval": spec.name, "status": "skipped_empty", "cases": 0}

    if dry_run:
        print(f"  [DRY RUN] {spec.name}: {len(cases)} case(s), metric={spec.metric}")
        return {"eval": spec.name, "status": "dry_run", "cases": len(cases)}

    print(f"\n  {spec.name} — {len(cases)} case(s) · {spec.description}")
    responses, texts = [], []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        response = client.complete(case)
        responses.append(response)
        texts.append(response.text)
        if verbose or index % 25 == 0 or index == len(cases):
            state = "ok" if response.ok else response.error[:60]
            print(f"    [{index}/{len(cases)}] {case.case_id} · {state}")

    # Evals whose cases are only meaningful together (determinism) score in bulk.
    if spec.score_all is not None:
        scores = spec.score_all(cases, texts)
    else:
        scores = [
            spec.score(case, text) if response.ok
            else Score(value=0.0, passed=False, detail=response.error[:200])
            for case, text, response in zip(cases, texts, responses)
        ]

    rows = [
        _case_row(case, score, response, spec.name, spec.metric, base)
        for case, score, response in zip(cases, scores, responses)
    ]

    scored = [s for s, r in zip(scores, responses) if r.ok]
    mean = sum(s.value for s in scored) / len(scored) if scored else 0.0
    strict = sum(1 for s in scored if s.passed) / len(scored) if scored else 0.0
    errors = sum(1 for r in responses if not r.ok)

    summary: Dict[str, Any] = {
        "eval": spec.name,
        "status": "ok" if errors < len(responses) else "all_requests_failed",
        "tier": spec.tier,
        "metric": spec.metric,
        "cases": len(cases),
        "errors": errors,
        "mean_score": round(mean, 4),
        "strict_pass_rate": round(strict, 4),
        "wall_s": round(time.perf_counter() - started, 1),
    }

    summarizer = SUMMARIZERS.get(spec.name)
    if summarizer is not None:
        summary["breakdown"] = summarizer(rows)

    print(
        f"  → {spec.name}: {spec.metric} {mean:.3f} · strict {strict:.3f} "
        f"· {errors} error(s) · {summary['wall_s']}s"
    )
    return {"summary": summary, "rows": rows}


def run_suite(
    model: str,
    eval_names: List[str],
    output_dir: Path,
    api_base: str = "http://127.0.0.1:8080/v1",
    api_key: str = "",
    backend: str = "unknown",
    quant: str = "?",
    ctx_cap: int = 131072,
    limit: int = 0,
    repeats: int = 3,
    seed: int = 20260728,
    allow_fetch: bool = False,
    allow_code_execution: bool = False,
    timeout: int = 600,
    verbose: bool = False,
) -> Path:
    """Run the named evals and write CSV + JSON. Returns the CSV path.

    Exposed as a function so the TUI can drive the suite the same way it drives
    the speed benchmarks, rather than shelling out and reparsing stdout.
    """
    specs = resolve(eval_names)
    if allow_code_execution:
        os.environ["HUMANEVAL_ALLOW_EXEC"] = "1"

    client = EvalClient(api_base=api_base, model=model, api_key=api_key, timeout=timeout)
    run_id = str(uuid.uuid4())[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "model_name": model,
        "backend": backend.upper(),
        "quant": quant,
    }
    build_kwargs: Dict[str, Any] = {
        "seed": seed,
        "allow_fetch": allow_fetch,
        "ctx_cap": ctx_cap,
        "repeats": repeats,
    }
    if limit:
        build_kwargs["limit"] = limit

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        result = run_eval(spec, client, base, dict(build_kwargs), False, verbose)
        if "rows" in result:
            all_rows.extend(result["rows"])
            summaries.append(result["summary"])
        else:
            summaries.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(model)
    csv_path = output_dir / f"quality_{slug}_{stamp}.csv"
    json_path = output_dir / f"quality_{slug}_{stamp}.json"
    if all_rows:
        _csv_append(all_rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model,
                "backend": backend,
                "quant": quant,
                "endpoint": api_base,
                "generated_at": datetime.now().isoformat(),
                "evals": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run quality evals against an OpenAI-compatible local server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Model id the server expects")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="", help="Bearer token, if the server needs one")
    parser.add_argument("--evals", nargs="+", default=DEFAULT_EVALS,
                        help=f"Eval names, or tier1/tier2/all. Known: {', '.join(sorted(EVALS))}")
    parser.add_argument("--backend", default="unknown", help="Backend label for the CSV")
    parser.add_argument("--quant", default="?", help="Quant label for the CSV")
    parser.add_argument("--ctx-cap", type=int, default=131072, help="Context cap for NIAH sizing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap cases per eval (0 = each eval's own default)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats for the determinism eval")
    parser.add_argument("--seed", type=int, default=20260728, help="Sampling seed")
    parser.add_argument("--fetch", action="store_true",
                        help="Permit downloading tier-2 datasets into evals/.cache/")
    parser.add_argument("--allow-code-execution", action="store_true",
                        help="Permit HumanEval to execute model-written code (see its docstring)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "quality")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout (s)")
    parser.add_argument("--verbose", action="store_true", help="Print every case")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan, contact nothing")
    args = parser.parse_args()

    specs = resolve(args.evals)

    if args.allow_code_execution:
        os.environ["HUMANEVAL_ALLOW_EXEC"] = "1"
    elif any(spec.needs_code_execution for spec in specs):
        print(
            "NOTE: HumanEval is selected but --allow-code-execution was not passed.\n"
            "      Its cases will run and be reported as unscored. That flag lets the\n"
            "      harness execute code written by the model under test — read\n"
            "      evals/datasets/humaneval.py before enabling it.\n"
        )

    client = EvalClient(api_base=args.url, model=args.model, api_key=args.api_key,
                        timeout=args.timeout)
    if not args.dry_run:
        problem = client.probe()
        if problem:
            raise SystemExit(
                f"{problem}\n"
                f"Start a server first, e.g.:  bash scripts/serve_local.sh {args.model} --backend llamacpp"
            )

    print(f"Quality evals · model={_model_slug(args.model)}")
    print(f"  endpoint : {args.url}")
    print(f"  evals    : {', '.join(spec.name for spec in specs)}")

    if args.dry_run:
        base = {"timestamp": "", "run_id": "dry", "model_name": args.model,
                "backend": args.backend.upper(), "quant": args.quant}
        build_kwargs: Dict[str, Any] = {
            "seed": args.seed, "allow_fetch": args.fetch,
            "ctx_cap": args.ctx_cap, "repeats": args.repeats,
        }
        if args.limit:
            build_kwargs["limit"] = args.limit
        for spec in specs:
            run_eval(spec, client, base, dict(build_kwargs), True, args.verbose)
        print("\nDry run complete — nothing was sent.")
        return

    csv_path = run_suite(
        model=args.model,
        eval_names=args.evals,
        output_dir=args.output_dir,
        api_base=args.url,
        api_key=args.api_key,
        backend=args.backend,
        quant=args.quant,
        ctx_cap=args.ctx_cap,
        limit=args.limit,
        repeats=args.repeats,
        seed=args.seed,
        allow_fetch=args.fetch,
        allow_code_execution=args.allow_code_execution,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {csv_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
