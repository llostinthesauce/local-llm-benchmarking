#!/usr/bin/env python3
"""
bench_mlx_direct.py — Benchmark MLX models via mlx_lm.stream_generate directly.

Loads models from HF repos or local paths. Measures TTFT, gen TPS, prompt TPS,
and MLX peak memory using the stream_generate response metadata.

Supports MTP speculative decoding via --draft-model.

Usage:
  python3 scripts/bench_mlx_direct.py --model mlx-community/Qwen3.5-35B-A3B-4bit
  python3 scripts/bench_mlx_direct.py --model mlx-community/gemma-4-26b-4bit --draft-model mlx-community/gemma-4-26b-mtp-drafter
  python3 scripts/bench_mlx_direct.py --model /path/to/local/mlx_model --passes micro --dry-run
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

MLX_IMPORT_ERROR: Exception | None = None


def _load_mlx_modules():
    """Import MLX lazily so --dry-run works without Metal access."""
    global MLX_IMPORT_ERROR
    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
        return load, stream_generate, make_sampler, mx
    except Exception as exc:
        MLX_IMPORT_ERROR = exc
        return None

FIELDS = [
    "timestamp", "run_id", "model_name", "backend", "pass_name",
    "ctx_cap", "ctx_used", "prompt_tokens", "gen_tokens",
    "prompt_tps", "gen_tps", "ttft_s", "peak_mem_pct", "status",
    "quant", "concurrency", "mtp", "draft_tokens", "draft_accepted_tokens",
    "draft_accept_rate", "token_count_method", "generated_text",
]

DEFAULT_COOLDOWN = 30


def _safe_model_cap_ctx(model_size_gb: float, ctx_cap: int, mem_guard_pct: float = 80.0, architecture: str = "dense") -> int:
    """Estimate the largest ctx that fits within mem_guard."""
    if model_size_gb <= 0:
        return ctx_cap
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    avail_gb = total_gb * (mem_guard_pct / 100.0) - model_size_gb - 2.0
    if avail_gb <= 0:
        return 0
    if architecture == "moe":
        kv_per_token_bytes = 20_000
    else:
        kv_per_token_bytes = model_size_gb * 25_000
    max_ctx = int(avail_gb * (1024 ** 3) / kv_per_token_bytes)
    return min(max_ctx, ctx_cap)


def _estimate_mlx_size_gb(repo_or_path: str) -> float:
    """Best-effort size lookup for an MLX model: local path first, then HF cache."""
    p = Path(os.path.expanduser(repo_or_path))
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 3)
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_or_path.replace('/', '--')}"
    if cache.is_dir():
        return sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / (1024 ** 3)
    return 0.0

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from prompts import PASSES, build_prompt_for_pass


class MemSampler:
    def __init__(self):
        self.peak = psutil.virtual_memory().percent
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, psutil.virtual_memory().percent)
            time.sleep(0.1)
    def start(self): self._t.start()
    def stop(self) -> float:
        self._stop.set(); self._t.join(timeout=2)
        return round(self.peak, 1)


def _csv_append(row: dict, path: Path) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)


def run_benchmark(model_repo: str, passes: list[dict], output_dir: Path,
                  draft_model: str | None = None, ctx_cap: int = 131072,
                  temperature: float = 0.7, top_p: float = 0.8,
                  quant: str = "?", cooldown: int = DEFAULT_COOLDOWN,
                  mem_guard: float = 80.0, dry_run: bool = False,
                   architecture: str = "dense", model_size_gb: float = 0.0) -> Path:
    model_repo = os.path.expanduser(model_repo)
    model_name = model_repo.split("/")[-1]
    run_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"mlx_direct_{model_name}_{ts}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_size_gb <= 0:
        model_size_gb = _estimate_mlx_size_gb(model_repo)

    if dry_run:
        for p in passes:
            note = ""
            raw_ctx = p["ctx"]
            if raw_ctx == -1:
                raw_ctx = _safe_model_cap_ctx(model_size_gb, ctx_cap, mem_guard, architecture) if model_size_gb > 0 else ctx_cap
                if raw_ctx == 0:
                    note = " → skipped_kv_oom"
            elif architecture == "dense" and model_size_gb > 20.0 and raw_ctx > 0:
                safe = _safe_model_cap_ctx(model_size_gb, ctx_cap, mem_guard, architecture)
                if 0 < safe < raw_ctx:
                    note = f" → clamped to {safe}"
                    raw_ctx = safe
            ctx_used = min(raw_ctx, ctx_cap) if raw_ctx > 0 else 0
            if ctx_used > 0:
                _, target_prompt_tokens = build_prompt_for_pass(p, ctx_used)
            else:
                target_prompt_tokens = 0
            target = f" target_prompt~{target_prompt_tokens}" if target_prompt_tokens else ""
            print(f"  [DRY RUN] {p['id']}: ctx={ctx_used}{target} gen={p['gen_tokens']} arch={architecture} size={model_size_gb:.1f}GB{note}")
        print(f"  CSV → {out_csv}")
        return out_csv

    mlx_modules = _load_mlx_modules()
    if mlx_modules is None:
        print(f"ERROR: mlx_lm unavailable or Metal is inaccessible: {MLX_IMPORT_ERROR}")
        sys.exit(1)
    load, stream_generate, make_sampler, mx = mlx_modules

    sampler = make_sampler(temp=temperature, top_p=top_p)
    draft = None
    if draft_model:
        print(f"  Loading draft model: {draft_model} ...")
        draft, _ = load(draft_model, lazy=True)

    print(f"  Loading model: {model_repo} ...")
    model, tokenizer = load(model_repo, lazy=True)

    for i, p in enumerate(passes):
        raw_ctx = p["ctx"]
        clamp_status: str | None = None
        if raw_ctx == -1:
            raw_ctx = _safe_model_cap_ctx(model_size_gb, ctx_cap, mem_guard, architecture) if model_size_gb > 0 else ctx_cap
            if raw_ctx == 0:
                _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                    "model_name": model_name, "backend": "MLX_DIRECT", "pass_name": p["id"],
                    "ctx_cap": ctx_cap, "ctx_used": 0, "prompt_tokens": 0, "gen_tokens": 0,
                    "prompt_tps": 0.0, "gen_tps": 0.0, "ttft_s": 0.0,
                    "peak_mem_pct": round(psutil.virtual_memory().percent, 1),
                    "status": "skipped_kv_oom", "quant": quant, "concurrency": 1,
                    "mtp": "on" if draft else "off", "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": "none", "generated_text": ""}, out_csv)
                print(f"  SKIP {p['id']}: would OOM at ctx_cap={ctx_cap}")
                continue
        elif architecture == "dense" and model_size_gb > 20.0 and raw_ctx > 0:
            safe_ctx = _safe_model_cap_ctx(model_size_gb, ctx_cap, mem_guard, architecture)
            if 0 < safe_ctx < raw_ctx:
                clamp_status = f"clamped_ctx_{raw_ctx}_to_{safe_ctx}"
                raw_ctx = safe_ctx
        ctx_used = min(raw_ctx, ctx_cap)
        prompt_text, target_prompt_tokens = build_prompt_for_pass(p, ctx_used)

        mem_pct = psutil.virtual_memory().percent
        if mem_pct >= mem_guard:
            _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                "model_name": model_name, "backend": "MLX_DIRECT", "pass_name": p["id"],
                "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": 0, "gen_tokens": 0,
                "prompt_tps": 0.0, "gen_tps": 0.0, "ttft_s": 0.0, "peak_mem_pct": round(mem_pct, 1),
                "status": f"skipped_mem_{mem_pct:.0f}pct", "quant": quant, "concurrency": 1,
                    "mtp": "on" if draft else "off", "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": "none", "generated_text": ""}, out_csv)
            print(f"  SKIP {p['id']}: mem {mem_pct:.0f}%")
            continue

        mem_sampler = MemSampler(); mem_sampler.start()
        ttft = 0.0; pt_out = 0; gt_out = 0; pt_tps = 0.0; gen_tps_s = 0.0
        peak_mem_gb = 0.0; status = "ok"
        generated_chunks: list[str] = []

        try:
            t0 = time.perf_counter()
            response = None
            for response in stream_generate(
                model, tokenizer, prompt_text, max_tokens=p["gen_tokens"],
                sampler=sampler, draft_model=draft,
            ):
                if response.text:
                    generated_chunks.append(response.text)
                if ttft == 0 and response.text:
                    ttft = time.perf_counter() - t0
                pt_out = response.prompt_tokens
                pt_tps = response.prompt_tps
                gt_out = response.generation_tokens
                gen_tps_s = response.generation_tps
                peak_mem_gb = response.peak_memory

            if response and response.finish_reason not in ("stop", "length", None):
                status = f"done:{response.finish_reason}"

        except Exception as exc:
            status = f"error:{exc}"
        finally:
            peak = mem_sampler.stop()

        token_count_method = "mlx_native" if status == "ok" else "none"
        if clamp_status:
            status = f"ok_{clamp_status}"
        if target_prompt_tokens and pt_out and pt_out < int(target_prompt_tokens * 0.8):
            status = f"underfilled_context_prompt_{pt_out}_target_{target_prompt_tokens}"

        _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
            "model_name": model_name, "backend": "MLX_DIRECT", "pass_name": p["id"],
            "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": pt_out, "gen_tokens": gt_out,
            "prompt_tps": round(pt_tps, 2), "gen_tps": round(gen_tps_s, 2),
            "ttft_s": round(ttft, 3), "peak_mem_pct": peak,
            "status": status, "quant": quant, "concurrency": 1,
            "mtp": "on" if draft else "off", "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": token_count_method,
            "generated_text": "".join(generated_chunks)}, out_csv)

        ok = status == "ok"
        print(f"  {'OK' if ok else 'FAIL'} {p['id']}: ttft {ttft:.2f}s · "
              f"p {pt_tps:.0f} · g {gen_tps_s:.1f} t/s · mem {peak:.0f}%")
        if i < len(passes) - 1: time.sleep(cooldown)

    # Unload models
    del model, tokenizer, draft
    gc.collect()
    try: mx.metal.clear_cache()
    except Exception: pass

    print(f"  CSV → {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark MLX models via direct inference")
    ap.add_argument("--model", type=str, required=True, help="HF repo or local path to MLX model")
    ap.add_argument("--draft-model", type=str, help="Draft model for MTP speculative decoding")
    ap.add_argument("--output-dir", type=Path, default=Path("results/mlx_direct"))
    ap.add_argument("--passes", nargs="+", default=["micro", "normal", "high", "max"],
                    choices=["micro", "normal", "high", "max"])
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    ap.add_argument("--mem-guard", type=float, default=90.0)
    ap.add_argument("--ctx-cap", type=int, default=131072)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--quant", type=str, default="?", help="Quant label for CSV")
    ap.add_argument("--architecture", choices=["dense", "moe"], default="dense",
                    help="Architecture hint for ctx clamping (dense >20GB clamps high passes)")
    ap.add_argument("--model-size-gb", type=float, default=0.0,
                    help="Override model size for clamp math; auto-detected from HF cache when 0")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    selected_passes = [p for p in PASSES if p["id"].split("_")[2] in args.passes]
    run_benchmark(args.model, selected_passes, args.output_dir,
                  args.draft_model, args.ctx_cap, args.temperature, args.top_p,
                  args.quant, args.cooldown, args.mem_guard, args.dry_run,
                  args.architecture, args.model_size_gb)


if __name__ == "__main__":
    main()
