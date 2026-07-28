#!/usr/bin/env python3
"""
Aggregate benchmark CSVs into a promotion-ready report.

This reads a run directory such as results/exhaustive_YYYYMMDD_HHMMSS, handles
the current unified speed CSVs plus native llama.cpp probe CSVs, and writes:

- summary.json: normalized rows, per-variant metrics, recommendations
- summary.md: human-readable leaderboard and serving guidance
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_registry

API_BACKENDS = {"LLAMACPP_API", "MLX_API", "MLX_VLM_API"}
RAW_BACKENDS = {
    "LLAMACPP_RAW", "MLX_DIRECT", "GGUF",
}
PASS_ORDER = {
    "pass_1_micro": 1,
    "pass_2_normal": 2,
    "pass_3_high": 3,
    "pass_4_max": 4,
}

RELIABILITY_RANK = {
    "hf_tokenizer": 5,
    "mlx_native": 5,
    "llama_bench_native": 5,
    "openai_usage": 4,
    "word_fallback_with_reasoning": 2,
    "word_fallback_no_reasoning": 2,
    "word_fallback": 1,
    "word_fallback_no_reasoning_untrusted": 1,
    "none": 0,
}

APPROXIMATE_THRESHOLD = 3  # ranks <= this are considered approximate/untrusted


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clean_model_key(name: str) -> str:
    value = name.split("/")[-1]
    value = value.replace(".gguf", "")
    value = re.sub(r"(?i)-UD", "", value)
    value = re.sub(r"(?i)-Q[0-9]+(?:_[A-Z0-9]+)*", "", value)
    value = re.sub(r"(?i)-[0-9]+bit", "", value)
    value = re.sub(r"(?i)-NVFP[0-9]+", "", value)
    value = re.sub(r"(?i)-MXFP[0-9]+", "", value)
    value = re.sub(r"(?i)-A[0-9]B", "", value)
    value = re.sub(r"(?i)-GGUF", "", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower()


def _registry_lookup(config: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in model_registry.iter_models(config):
        keys = {
            row["name"],
            row["path"],
            row["selector"],
            row["family_id"],
            _clean_model_key(row["name"]),
            _clean_model_key(row["path"]),
        }
        for key in keys:
            if key:
                lookup[str(key).lower()] = row
    return lookup


def _registry_match(row: Dict[str, Any], lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    model = row.get("model_name", "")
    keys = [model, model.split("/")[-1], _clean_model_key(model)]
    for key in keys:
        match = lookup.get(str(key).lower())
        if match:
            return match
    return None


def _iter_csv_rows(run_dir: Path) -> Iterable[tuple[Path, Dict[str, str]]]:
    for path in sorted(run_dir.rglob("*.csv")):
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    yield path, row
        except Exception:
            continue


def _is_quality_row(row: Dict[str, str]) -> bool:
    """Quality CSVs carry per-case accuracy, not per-pass throughput.

    They share the run/model/backend columns so the two can be joined, but they
    have no gen_tps or ctx_used and must not be fed through _normalize().
    """
    return "eval_name" in row and "score" in row


def _quality_summary(rows: List[Dict[str, str]], lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse per-case rows into one record per (model, backend, eval)."""
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        model_name = row.get("model_name") or "?"
        match = _registry_match({"model_name": model_name}, lookup)
        family_id = match["family_id"] if match else _clean_model_key(model_name)
        key = (family_id, row.get("backend") or "?", row.get("quant") or "?", row.get("eval_name") or "?")
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (family_id, backend, quant, eval_name), group in sorted(grouped.items()):
        scored = [r for r in group if (r.get("status") or "") == "ok"]
        errors = len(group) - len(scored)
        # None, not 0.0, when nothing was scorable. An eval whose every response
        # was truncated or errored has no score — recording it as zero would
        # drag the model's combined mean down for a failure that was the
        # harness's (a token budget) rather than the model's.
        mean = round(_score_mean([_float(r.get("score")) for r in scored]), 4) if scored else None
        strict = round(_score_mean([_float(r.get("passed")) for r in scored]), 4) if scored else None
        out.append({
            "family_id": family_id,
            "backend": backend,
            "quant": quant,
            "eval_name": eval_name,
            "metric": (scored[0].get("metric") if scored else "") or "accuracy",
            "cases": len(group),
            "scored": len(scored),
            "errors": errors,
            "mean_score": mean,
            "strict_pass_rate": strict,
            "model_name": group[0].get("model_name", ""),
        })
    return out


def _combined_leaderboard(variants: List[Dict[str, Any]],
                          quality: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join throughput against accuracy on (family_id, quant).

    This is the join the whole suite exists for. Tokens per second is a
    production number only if the weights still answer correctly, and a quant
    that scores well is irrelevant if it runs at 4 tok/s. Neither table alone
    supports a serving decision; side by side they do.

    Backend is deliberately not part of the key: accuracy is a property of the
    weights, so a family's eval scores apply to whichever engine served it
    fastest.
    """
    by_variant: Dict[tuple, Dict[str, Any]] = {}
    for variant in variants:
        if variant.get("ok_passes", 0) <= 0:
            continue
        key = (variant["family_id"], variant.get("quant", "?"))
        current = by_variant.get(key)
        if current is None or variant.get("avg_gen_tps", 0) > current.get("avg_gen_tps", 0):
            by_variant[key] = variant

    by_quality: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in quality:
        by_quality[(record["family_id"], record["quant"])][record["eval_name"]] = record

    rows: List[Dict[str, Any]] = []
    for key in sorted(set(by_variant) | set(by_quality)):
        family_id, quant = key
        variant = by_variant.get(key)
        evals = by_quality.get(key, {})
        scorable = [r["mean_score"] for r in evals.values() if r["mean_score"] is not None]
        rows.append({
            "family_id": family_id,
            "quant": quant,
            "backend": variant["backend"] if variant else "",
            "gen_tps_mean": round(variant["avg_gen_tps"], 2) if variant else None,
            "ttft_s_mean": round(variant["avg_ttft_s"], 3) if variant else None,
            "token_trust": variant.get("token_trust") if variant else None,
            "evals": {name: record["mean_score"] for name, record in sorted(evals.items())
                      if record["mean_score"] is not None},
            "eval_mean": round(_score_mean(scorable), 4) if scorable else None,
            "has_speed": variant is not None,
            "has_quality": bool(evals),
        })
    rows.sort(key=lambda r: (-(r["eval_mean"] or 0), -(r["gen_tps_mean"] or 0)))
    return rows


def _mtp_label(row: Dict[str, str]) -> str:
    """Normalize the serving-config dimension across schemas.

    bench_*.py records mtp=on|off. config_compare records config=mtp_on|mtp_off
    (or kv_fp16|kv_q8 for the KV suite — passed through so KV variants stay distinct).
    """
    mtp = (row.get("mtp") or "").strip().lower()
    if mtp in ("on", "off"):
        return mtp
    config = (row.get("config") or "").strip().lower()
    if config == "mtp_on":
        return "on"
    if config == "mtp_off":
        return "off"
    if config:
        return config
    return "off"


def _normalize(path: Path, row: Dict[str, str], lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    backend = row.get("backend") or path.parent.name.upper()
    model_name = row.get("model_name") or row.get("model") or path.stem
    match = _registry_match({"model_name": model_name}, lookup)
    family_id = match["family_id"] if match else _clean_model_key(model_name)
    pass_name = row.get("pass_name") or row.get("test_name") or row.get("prompt_id") or path.parent.name
    status = row.get("status") or "unknown"
    gen_tps = _float(row.get("gen_tps"))
    if not gen_tps:
        gen_tps = _float(row.get("tokens_per_second") or row.get("total_tps"))
    prompt_tps = _float(row.get("prompt_tps"))
    if not prompt_tps:
        prompt_tps = _float(row.get("prompt_tps") or row.get("prompt_tps"))
    return {
        "source": str(path),
        "timestamp": row.get("timestamp", ""),
        "run_id": row.get("run_id", ""),
        "model_name": model_name,
        "family_id": family_id,
        "backend": backend,
        "kind": "api" if backend in API_BACKENDS else "raw",
        "pass_name": pass_name,
        "pass_order": PASS_ORDER.get(pass_name, 99),
        "ctx_cap": _int(row.get("ctx_cap")),
        "ctx_used": _int(row.get("ctx_used")),
        "prompt_tokens": _int(row.get("prompt_tokens") or row.get("n_prompt")),
        "gen_tokens": _int(row.get("gen_tokens") or row.get("n_gen")),
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "ttft_s": _float(row.get("ttft_s")),
        "peak_mem_pct": _float(row.get("peak_mem_pct") or row.get("peak_mem_percent")),
        "status": status,
        "ok": status.startswith("ok") or "clamped_ctx_" in status,
        "quant": row.get("quant") or (match.get("quant") if match else "?"),
        "concurrency": _int(row.get("concurrency") or row.get("parallel"), 1),
        "mtp": _mtp_label(row),
        "draft_tokens": _int(row.get("draft_tokens")),
        "draft_accepted_tokens": _int(row.get("draft_accepted_tokens")),
        # config_compare stores the ratio in draft_accept; bench_*.py in draft_accept_rate
        "draft_accept_rate": _float(row.get("draft_accept_rate") or row.get("draft_accept")),
        "recall_pass": row.get("recall_pass", ""),
        "artifact": row.get("artifact", ""),
        "token_count_method": row.get("token_count_method", ""),
    }


def _mean(values: Iterable[float]) -> float:
    """Mean of positive values only.

    Correct for throughput: a 0 t/s reading means the pass never produced a
    measurement, and averaging it in would understate the model. Do NOT use this
    for accuracy — see _score_mean.
    """
    vals = [v for v in values if v > 0]
    return sum(vals) / len(vals) if vals else 0.0


def _score_mean(values: Iterable[float]) -> float:
    """Plain arithmetic mean, zeros included.

    Eval scores are the opposite case from throughput: a 0.0 means the model got
    the answer wrong, which is a result, not a missing measurement. Dropping
    zeros here would report a model that failed half its cases as scoring 1.0.
    """
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _variant_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (row["family_id"], row["backend"], row["quant"], row["model_name"], row["mtp"])


def _score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [r for r in rows if r["ok"]]
    all_status_ok = [r for r in rows if r["status"] == "ok"]

    duplicate_passes: list[str] = []
    by_pass_latest: Dict[str, Dict[str, Any]] = {}
    for r in sorted(ok_rows, key=lambda x: x["timestamp"]):
        pname = r["pass_name"]
        if pname in by_pass_latest:
            duplicate_passes.append(pname)
        by_pass_latest[pname] = r

    micro = by_pass_latest.get("pass_1_micro", {})
    normal = by_pass_latest.get("pass_2_normal", {})
    high = by_pass_latest.get("pass_3_high", {})
    maxp = by_pass_latest.get("pass_4_max", {})

    gen_values = [r["gen_tps"] for r in ok_rows]
    mem_values = [r["peak_mem_pct"] for r in rows if r["peak_mem_pct"]]
    ttft_values = [r["ttft_s"] for r in ok_rows if r["ttft_s"]]

    methods = [r.get("token_count_method", "none") for r in rows if r.get("token_count_method")]
    worst_rank = min((RELIABILITY_RANK.get(m, 0) for m in methods), default=0) if methods else 0
    best_rank = max((RELIABILITY_RANK.get(m, 0) for m in methods), default=0)

    is_approximate = worst_rank <= APPROXIMATE_THRESHOLD

    sources = sorted({r["source"] for r in rows if r.get("source")})

    all_zero = (
        all(r["status"] == "ok" for r in all_status_ok)
        and all(r.get("prompt_tps", 0) == 0.0 and r.get("gen_tps", 0) == 0.0 for r in all_status_ok)
        and len(all_status_ok) > 0
    )

    warnings: list[str] = []
    if duplicate_passes:
        warnings.append(f"Duplicate passes: {', '.join(sorted(set(duplicate_passes)))}; latest retained")
    if is_approximate:
        labels = ", ".join(sorted({r.get("token_count_method", "") for r in rows})) or "?"
        warnings.append(f"Token count approximate (worst={worst_rank}, best={best_rank}): {labels}")
    if all_zero:
        warnings.append("status=ok but prompt_tps and gen_tps are both zero; possible measurement failure")

    return {
        "family_id": rows[0]["family_id"],
        "model_name": rows[0]["model_name"],
        "backend": rows[0]["backend"],
        "kind": rows[0]["kind"],
        "quant": rows[0]["quant"],
        "mtp": rows[0]["mtp"],
        "avg_draft_accept_rate": round(_mean([r["draft_accept_rate"] for r in ok_rows]), 3),
        "ok_passes": len(ok_rows),
        "total_passes": len(rows),
        "total_rows": len(rows),
        "duplicate_pass_count": len(set(duplicate_passes)) if duplicate_passes else 0,
        "duplicate_passes": sorted(set(duplicate_passes)),
        "sources": sources,
        "avg_gen_tps": round(_mean(gen_values), 2),
        "micro_gen_tps": round(_float(micro.get("gen_tps")), 2),
        "normal_gen_tps": round(_float(normal.get("gen_tps")), 2),
        "high_gen_tps": round(_float(high.get("gen_tps")), 2),
        "max_gen_tps": round(_float(maxp.get("gen_tps")), 2),
        "max_ctx_used": max((r["ctx_used"] for r in ok_rows), default=0),
        "avg_ttft_s": round(_mean(ttft_values), 3),
        "peak_mem_pct": round(max(mem_values), 1) if mem_values else 0.0,
        "statuses": sorted({r["status"] for r in rows if r["status"]}),
        "token_count_methods": sorted({r["token_count_method"] for r in rows if r.get("token_count_method")}),
        "worst_token_rank": worst_rank,
        "best_token_rank": best_rank,
        "token_trust": "trusted" if not is_approximate else "approximate",
        "warnings": warnings,
        "score": round(
            (_mean(gen_values) * 1.0)
            + (_float(high.get("gen_tps")) * 0.25)
            + (_float(maxp.get("gen_tps")) * 0.25)
            + (worst_rank * 0.5),
            2,
        ),
    }


def _best(items: List[Dict[str, Any]], *, kind: str | None = None,
           family_id: str | None = None, trusted_only: bool = False) -> Dict[str, Any] | None:
    candidates = items
    if kind:
        candidates = [i for i in candidates if i["kind"] == kind]
    if family_id:
        candidates = [i for i in candidates if i["family_id"] == family_id]
    if trusted_only:
        candidates = [i for i in candidates if i.get("token_trust") == "trusted"]
    candidates = [i for i in candidates if i["ok_passes"] > 0 and i["avg_gen_tps"] > 0]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x["score"], x["ok_passes"], -x["peak_mem_pct"]), reverse=True)[0]


def aggregate(run_dir: Path, config: Path) -> Dict[str, Any]:
    lookup = _registry_lookup(config)
    raw_rows = list(_iter_csv_rows(run_dir))
    quality_raw = [row for _, row in raw_rows if _is_quality_row(row)]
    rows = [_normalize(path, row, lookup) for path, row in raw_rows if not _is_quality_row(row)]
    grouped: Dict[tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_variant_key(row)].append(row)
    variants = [_score(v) for v in grouped.values()]
    variants.sort(key=lambda x: (x["kind"], -x["score"], x["family_id"], x["backend"]))

    best_raw_all = _best(variants, kind="raw", trusted_only=False)
    best_api_all = _best(variants, kind="api", trusted_only=False)
    best_api_trusted = _best(variants, kind="api", trusted_only=True)
    best_raw_trusted = _best(variants, kind="raw", trusted_only=True)

    approximate_api = [v for v in variants if v["kind"] == "api" and v.get("token_trust") == "approximate" and v["ok_passes"] > 0]
    approximate_raw = [v for v in variants if v["kind"] == "raw" and v.get("token_trust") == "approximate" and v["ok_passes"] > 0]
    duplicate_variants = [v for v in variants if v.get("duplicate_pass_count", 0) > 0]

    warnings: list[str] = []
    if best_api_all and best_api_all.get("token_trust") == "approximate":
        warnings.append(
            f"best_api_overall ({best_api_all['backend']} / {best_api_all['model_name']}) "
            f"uses approximate token counts ({', '.join(best_api_all.get('token_count_methods', []))}); "
            f"prefer trusted_api for measurement integrity"
        )
    if best_api_trusted is None:
        warnings.append("No API variant has fully trusted token counts; all API rows use approximate or fallback counting")
    if approximate_api:
        names = [v["model_name"][:30] for v in approximate_api[:5]]
        warnings.append(f"{len(approximate_api)} API variant(s) with approximate token counts: {', '.join(names)}")
    if duplicate_variants:
        warnings.append(f"{len(duplicate_variants)} variant(s) have duplicate passes (retries detected)")

    families = sorted({v["family_id"] for v in variants})
    family_recs = []
    for family in families:
        family_recs.append({
            "family_id": family,
            "best_raw": _best(variants, kind="raw", family_id=family, trusted_only=False),
            "best_api": _best(variants, kind="api", family_id=family, trusted_only=False),
            "best_api_trusted": _best(variants, kind="api", family_id=family, trusted_only=True),
        })

    mtp_deltas = _mtp_deltas(variants)
    quality = _quality_summary(quality_raw, lookup)
    combined = _combined_leaderboard(variants, quality)

    if quality and not variants:
        warnings.append("Quality results found with no speed results; the combined leaderboard has no throughput column")
    if variants and not quality:
        warnings.append("No quality results in this run — throughput alone cannot show whether a quant is still correct (see scripts/bench_quality.py)")

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "config": str(config),
            "row_count": len(rows),
            "variant_count": len(variants),
            "quality_row_count": len(quality_raw),
            "token_trust_cutoff": APPROXIMATE_THRESHOLD,
        },
        "warnings": warnings,
        "recommendations": {
            "fastest_raw_overall": best_raw_all,
            "fastest_api_overall": best_api_all,
            "trusted_raw_overall": best_raw_trusted,
            "trusted_api_overall": best_api_trusted,
            "by_family": family_recs,
        },
        "mtp_deltas": mtp_deltas,
        "quality": quality,
        "combined_leaderboard": combined,
        "variants": variants,
        "rows": rows,
    }


def _mtp_deltas(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair mtp=on vs mtp=off variants (same family/backend/quant) and report the speedup."""
    by_base: Dict[tuple[str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for v in variants:
        if v["mtp"] in ("on", "off"):
            by_base[(v["family_id"], v["backend"], v["quant"])][v["mtp"]] = v
    deltas = []
    for (family_id, backend, quant), pair in sorted(by_base.items()):
        off, on = pair.get("off"), pair.get("on")
        if not off or not on:
            continue
        off_tps, on_tps = off["avg_gen_tps"], on["avg_gen_tps"]
        pct = round((on_tps / off_tps - 1) * 100, 1) if off_tps else 0.0
        deltas.append({
            "family_id": family_id,
            "backend": backend,
            "quant": quant,
            "off_avg_gen_tps": off_tps,
            "on_avg_gen_tps": on_tps,
            "delta_pct": pct,
            "avg_draft_accept_rate": on.get("avg_draft_accept_rate", 0.0),
        })
    return deltas


def _fmt_num(value: Any, decimals: int = 2) -> str:
    val = _float(value)
    return f"{val:.{decimals}f}" if val else "-"


def _variant_label(item: Dict[str, Any] | None) -> str:
    if not item:
        return "No valid result"
    return f"{item['backend']} / {item['model_name']} / {item['quant']} ({item['avg_gen_tps']} t/s avg)"


def write_markdown(payload: Dict[str, Any], out_path: Path) -> None:
    meta = payload["metadata"]
    rec = payload["recommendations"]
    warnings = payload.get("warnings", [])

    lines = [
        "# Benchmark Aggregate",
        "",
        f"Generated: {meta['generated_at']}",
        f"Run dir: `{meta['run_dir']}`",
        f"Rows: {meta['row_count']}  ",
        f"Variants: {meta['variant_count']}",
        f"Token trust cutoff: <= rank {meta.get('token_trust_cutoff', 3)} = approximate",
        "",
    ]

    if warnings:
        lines += ["## Warnings", ""]
        for w in warnings:
            lines.append(f"- **WARNING:** {w}")
        lines.append("")

    combined = payload.get("combined_leaderboard") or []
    if combined:
        eval_names = sorted({name for row in combined for name in row["evals"]})
        lines += [
            "## Speed x Accuracy",
            "",
            "Throughput is only a serving argument when the weights still answer",
            "correctly. Sorted by mean eval score, then by throughput.",
            "",
            "| Model | Quant | Backend | gen t/s | TTFT s | "
            + " | ".join(eval_names)
            + " | Eval mean |",
            "|---|---|---|---:|---:|" + "---:|" * (len(eval_names) + 1),
        ]
        for row in combined:
            scores = " | ".join(
                _fmt_num(row["evals"].get(name), 3) if name in row["evals"] else "—"
                for name in eval_names
            )
            lines.append(
                f"| {row['family_id']} | {row['quant']} | {row['backend'] or '—'} "
                f"| {_fmt_num(row['gen_tps_mean'], 2) if row['gen_tps_mean'] is not None else '—'} "
                f"| {_fmt_num(row['ttft_s_mean'], 3) if row['ttft_s_mean'] is not None else '—'} "
                f"| {scores} | {_fmt_num(row['eval_mean'], 3) if row['eval_mean'] is not None else '—'} |"
            )
        lines.append("")

    quality = payload.get("quality") or []
    if quality:
        lines += [
            "## Quality Evals",
            "",
            "| Model | Quant | Backend | Eval | Metric | Scored | Unscorable | Mean | Strict pass |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
        for record in quality:
            mean = _fmt_num(record["mean_score"], 3) if record["mean_score"] is not None else "n/a"
            strict = (
                _fmt_num(record["strict_pass_rate"], 3)
                if record["strict_pass_rate"] is not None else "n/a"
            )
            lines.append(
                f"| {record['family_id']} | {record['quant']} | {record['backend']} "
                f"| {record['eval_name']} | {record['metric']} "
                f"| {record.get('scored', 0)}/{record['cases']} "
                f"| {record['errors']} | {mean} | {strict} |"
            )
        lines.append("")

    lines += [
        "## Serving Verdict",
        "",
        "| Recommendation | Variant | TPS | Token Trust |",
        "|---|---:|---:|---|",
    ]
    for label, result in [
        ("Fastest raw (any token count)", rec.get("fastest_raw_overall")),
        ("Trusted raw (reliable tokens)", rec.get("trusted_raw_overall")),
        ("Fastest API (any token count)", rec.get("fastest_api_overall")),
        ("**Trusted API (reliable tokens)**", rec.get("trusted_api_overall")),
    ]:
        if result:
            trust = result.get("token_trust", "?")
            lines.append(
                f"| {label} | {result['backend']} / {result['model_name']} / {result['quant']} "
                f"| {result['avg_gen_tps']} t/s | {trust} |"
            )
        else:
            lines.append(f"| {label} | *none* | - | - |")

    lines += [
        "",
        "> **Trusted API** is the recommended serving configuration for tool IDEs.",
        "> **Fastest API** is listed for reference but may over-report TPS due to approximate token counting.",
        "",
        "## Family Winners",
        "",
        "| Family | Best Raw | Best API | Best Trusted API |",
        "|---|---|---|---|",
    ]
    for family in rec["by_family"]:
        raw_label = _variant_label(family.get("best_raw"))
        api_label = _variant_label(family.get("best_api"))
        trusted_label = _variant_label(family.get("best_api_trusted"))
        lines.append(f"| `{family['family_id']}` | {raw_label} | {api_label} | {trusted_label} |")

    mtp_deltas = payload.get("mtp_deltas", [])
    if mtp_deltas:
        lines += [
            "",
            "## MTP Speedup (on vs off)",
            "",
            "| Family | Backend | Quant | Off TPS | On TPS | Δ | Draft Accept |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for d in mtp_deltas:
            lines.append(
                f"| `{d['family_id']}` | {d['backend']} | {d['quant']} "
                f"| {_fmt_num(d['off_avg_gen_tps'])} | {_fmt_num(d['on_avg_gen_tps'])} "
                f"| {d['delta_pct']:+.1f}% | {_fmt_num(d['avg_draft_accept_rate'], 3)} |"
            )

    lines += [
        "",
        "## Variant Leaderboard",
        "",
        "| Kind | Trust | Family | Backend | Model | Quant | MTP | OK | Avg TPS | Micro | Normal | High | Max | Draft | Max Ctx | TTFT | Peak Mem | Token Rank | Dup | Status |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in sorted(payload["variants"], key=lambda x: (x["kind"], -x["score"], x["family_id"])):
        status = ", ".join(item["statuses"]) or "-"
        trust = item.get("token_trust", "?")
        dup = str(item.get("duplicate_pass_count", 0))
        lines.append(
            f"| {item['kind']} | {trust} | `{item['family_id']}` | {item['backend']} | `{item['model_name']}` | {item['quant']} "
            f"| {item.get('mtp', 'off')} "
            f"| {item['ok_passes']}/{item['total_passes']} | {_fmt_num(item['avg_gen_tps'])} "
            f"| {_fmt_num(item['micro_gen_tps'])} | {_fmt_num(item['normal_gen_tps'])} "
            f"| {_fmt_num(item['high_gen_tps'])} | {_fmt_num(item['max_gen_tps'])} "
            f"| {_fmt_num(item.get('avg_draft_accept_rate'), 3)} "
            f"| {item['max_ctx_used']} | {_fmt_num(item['avg_ttft_s'], 3)} "
            f"| {_fmt_num(item['peak_mem_pct'], 1)} | {item['worst_token_rank']}/{item['best_token_rank']} | {dup} | {status} |"
        )

    lines.append("")
    out_path.write_text("\n".join(lines))


# Flat normalized long-format CSV: one row per measurement, fixed schema.
# This is the no-cruft artifact for downstream parsing — no per-backend column drift.
NORMALIZED_FIELDS = [
    "timestamp", "family_id", "model_name", "backend", "kind", "quant", "mtp",
    "pass_name", "ctx_used", "prompt_tps", "gen_tps", "ttft_s",
    "draft_accept_rate", "recall_pass", "peak_mem_pct", "status", "source",
]


def write_normalized_csv(payload: Dict[str, Any], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NORMALIZED_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in sorted(
            payload["rows"],
            key=lambda r: (r["family_id"], r["backend"], r["quant"], r["mtp"],
                           r["pass_order"], r["timestamp"]),
        ):
            w.writerow({k: row.get(k, "") for k in NORMALIZED_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark run results")
    parser.add_argument("run_dir", type=Path, help="Run directory containing benchmark CSVs")
    parser.add_argument("--config", type=Path, default=model_registry.DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"Run directory not found: {args.run_dir}")
    payload = aggregate(args.run_dir, args.config)
    out_json = args.output_json or args.run_dir / "summary.json"
    out_md = args.output_md or args.run_dir / "summary.md"
    out_csv = args.output_csv or args.run_dir / "results.csv"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(payload, out_md)
    write_normalized_csv(payload, out_csv)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
