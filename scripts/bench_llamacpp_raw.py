#!/usr/bin/env python3
"""
bench_llamacpp_raw.py — Benchmark GGUF models via llama-bench (raw throughput).

Measures prompt_tps and gen_tps directly from llama-bench (Metal backend).
No API overhead — fastest possible measurement.

Usage:
  python3 scripts/bench_llamacpp_raw.py --model ~/path/to/model.gguf
  python3 scripts/bench_llamacpp_raw.py --model model.gguf --passes micro,normal --dry-run
  python3 scripts/bench_llamacpp_raw.py --config configs/models.local.json --all
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

FIELDS = [
    "timestamp", "run_id", "model_name", "backend", "pass_name",
    "ctx_cap", "ctx_used", "prompt_tokens", "gen_tokens",
    "prompt_tps", "gen_tps", "ttft_s", "peak_mem_pct", "status",
    "quant", "concurrency", "mtp", "draft_tokens", "draft_accepted_tokens",
    "draft_accept_rate", "token_count_method", "generated_text",
]

MAX_GEN_TPS = 300.0
DEFAULT_COOLDOWN = 30

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import model_registry
from prompts import PASSES


def _safe_model_cap_ctx(model_size_gb: float, ctx_cap: int, mem_guard_pct: float = 80.0, architecture: str = "dense") -> int:
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


def run_benchmark(model_path: str, passes: list[dict], output_dir: Path,
                  ctx_cap: int = 131072, model_size_gb: float = 0,
                  quant: str = "?", cooldown: int = DEFAULT_COOLDOWN,
                  mem_guard: float = 80.0, dry_run: bool = False,
                  architecture: str = "dense") -> Path:
    model_name = Path(model_path).name
    run_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"llamacpp_raw_{model_name.replace('.gguf','')}_{ts}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(passes):
        raw_ctx = p["ctx"]
        clamp_status: str | None = None
        if raw_ctx == -1:
            raw_ctx = _safe_model_cap_ctx(model_size_gb, ctx_cap, mem_guard, architecture)
            if raw_ctx == 0:
                _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                    "model_name": model_name, "backend": "LLAMACPP_RAW", "pass_name": p["id"],
                    "ctx_cap": ctx_cap, "ctx_used": 0, "prompt_tokens": 0, "gen_tokens": 0,
                    "prompt_tps": 0.0, "gen_tps": 0.0, "peak_mem_pct": round(psutil.virtual_memory().percent, 1),
                    "status": "skipped_kv_oom", "quant": quant, "concurrency": 1, "mtp": "off",
                    "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": "none"}, out_csv)
                continue
        elif architecture == "dense" and model_size_gb > 20.0 and raw_ctx > 0:
            safe_ctx = _safe_model_cap_ctx(model_size_gb, ctx_cap, mem_guard, architecture)
            if 0 < safe_ctx < raw_ctx:
                clamp_status = f"clamped_ctx_{raw_ctx}_to_{safe_ctx}"
                raw_ctx = safe_ctx
        ctx_used = min(raw_ctx, ctx_cap)

        if dry_run:
            print(f"  [DRY RUN] {p['id']}: ctx={ctx_used} gen={p['gen_tokens']}")
            continue

        mem_pct = psutil.virtual_memory().percent
        if mem_pct >= mem_guard:
            _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                "model_name": model_name, "backend": "LLAMACPP_RAW", "pass_name": p["id"],
                "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": 0, "gen_tokens": 0,
                "prompt_tps": 0.0, "gen_tps": 0.0, "peak_mem_pct": round(mem_pct, 1),
                "status": f"skipped_mem_{mem_pct:.0f}pct", "quant": quant, "concurrency": 1, "mtp": "off",
                "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": "none"}, out_csv)
            print(f"  SKIP {p['id']}: mem {mem_pct:.0f}%")
            continue

        cmd = ["llama-bench", "-m", model_path, "-p", str(ctx_used),
               "-n", str(p["gen_tokens"]), "-b", "512", "-r", "1", "--output", "csv"]
        sampler = MemSampler(); sampler.start()
        prompt_tps = gen_tps = 0.0; prompt_tokens_out = ctx_used
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                                  timeout=max(600, p["gen_tokens"] * 4))
            if proc.returncode == 0:
                lines = [l for l in proc.stdout.strip().splitlines()
                         if l.startswith("build_commit") or l.startswith('"')]
                if len(lines) > 1:
                    for row in csv.DictReader(lines):
                        avg = float(row.get("avg_ts", 0) or 0)
                        np_ = int(row.get("n_prompt", 0) or 0)
                        ng_ = int(row.get("n_gen", 0) or 0)
                        if np_ > 0 and ng_ == 0:
                            prompt_tps, prompt_tokens_out = avg, np_
                        elif ng_ > 0 and np_ == 0:
                            gen_tps = avg
            status = "ok" if proc.returncode == 0 else f"error:rc{proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as exc:
            status = f"error:{exc}"
        finally:
            peak = sampler.stop()

        if gen_tps > MAX_GEN_TPS:
            gen_tps = 0.0; status = "warn_bogus_tps"

        token_count_method = "llama_bench_native" if status in ("ok",) or (clamp_status and not status.startswith("error")) else "none"
        if clamp_status:
            status = f"ok_{clamp_status}"

        _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
            "model_name": model_name, "backend": "LLAMACPP_RAW", "pass_name": p["id"],
            "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": prompt_tokens_out,
            "gen_tokens": p["gen_tokens"], "prompt_tps": round(prompt_tps, 2),
            "gen_tps": round(gen_tps, 2), "ttft_s": 0.0, "peak_mem_pct": peak,
            "status": status, "quant": quant, "concurrency": 1, "mtp": "off",
            "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": token_count_method}, out_csv)

        ok = status == "ok"
        print(f"  {'OK' if ok else 'FAIL'} {p['id']}: p {prompt_tps:.0f} · g {gen_tps:.1f} t/s · mem {peak:.0f}%")
        if i < len(passes) - 1: time.sleep(cooldown)

    print(f"  CSV → {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark GGUF models via llama-bench")
    ap.add_argument("--model", type=str, help="Path to GGUF model file")
    ap.add_argument("--config", type=Path, default=model_registry.DEFAULT_CONFIG, help="Path to generated local model registry")
    ap.add_argument("--all", action="store_true", help="Run all GGUF models in config")
    ap.add_argument("--output-dir", type=Path, default=Path("results/llamacpp_raw"))
    ap.add_argument("--passes", nargs="+", default=["micro", "normal", "high", "max"],
                    choices=["micro", "normal", "high", "max"])
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    ap.add_argument("--mem-guard", type=float, default=80.0)
    ap.add_argument("--quant", type=str, default="?", help="Quant label for CSV")
    ap.add_argument("--ctx-cap", type=int, default=131072)
    ap.add_argument("--architecture", choices=["dense", "moe"], default="dense",
                    help="Architecture hint for ctx clamping (dense models clamp pass_3_high)")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    selected_passes = [p for p in PASSES if p["id"].split("_")[2] in args.passes]

    models = []
    if args.config and args.all:
        for row in model_registry.iter_models(args.config):
            if row["backend"] != "llamacpp" or not row["exists"]:
                continue
            path = row["path"]
            gb = Path(path).stat().st_size / (1024**3)
            models.append((path, row["ctx_cap"], gb, row.get("quant", "?"), row.get("architecture", "dense")))
    elif args.model:
        path = os.path.expanduser(args.model)
        if Path(path).exists():
            gb = Path(path).stat().st_size / (1024**3)
            models.append((path, args.ctx_cap, gb, args.quant, args.architecture))
        else:
            print(f"Model not found: {path}")
            sys.exit(1)
    else:
        ap.print_help()
        return

    for path, ctx_cap, size_gb, quant, arch in models:
        print(f"\n=== {Path(path).name} ({arch}) ===")
        run_benchmark(path, selected_passes, args.output_dir, ctx_cap, size_gb, quant,
                      args.cooldown, args.mem_guard, args.dry_run, arch)


if __name__ == "__main__":
    main()
