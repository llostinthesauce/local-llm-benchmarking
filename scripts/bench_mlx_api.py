#!/usr/bin/env python3
"""
bench_mlx_api.py — Benchmark via mlx_lm.server OpenAI-compatible API.

Starts mlx_lm.server as a subprocess, benchmarks via streaming chat completions.
Measures TTFT, gen TPS, peak memory. Supports MTP via --draft-model.

Usage:
  python3 scripts/bench_mlx_api.py --model mlx-community/Qwen3.5-35B-A3B-4bit
  python3 scripts/bench_mlx_api.py --model mlx-community/gemma-4-26b-4bit --draft-model mlx-community/gemma-4-26b-mtp-drafter
  python3 scripts/bench_mlx_api.py --model /path/to/local --passes micro --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
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
import requests

FIELDS = [
    "timestamp", "run_id", "model_name", "backend", "pass_name",
    "ctx_cap", "ctx_used", "prompt_tokens", "gen_tokens",
    "prompt_tps", "gen_tps", "ttft_s", "peak_mem_pct", "status",
    "quant", "concurrency", "mtp", "draft_tokens", "draft_accepted_tokens",
    "draft_accept_rate", "token_count_method", "generated_text",
]

DEFAULT_COOLDOWN = 30
DEFAULT_PORT = 8085

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from prompts import PASSES, build_prompt_for_pass, context_budget_for_pass


class MemSampler:
    def __init__(self, pids: list[int] | None = None):
        self.peak = psutil.virtual_memory().percent
        self._stop = threading.Event()
        self._pids = pids or []
        self._t = threading.Thread(target=self._loop, daemon=True)
    def _loop(self):
        while not self._stop.is_set():
            mem = psutil.virtual_memory().percent
            for pid in self._pids:
                try: mem = max(mem, psutil.Process(pid).memory_percent())
                except psutil.NoSuchProcess: pass
            self.peak = max(self.peak, mem)
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


def _start_server(model: str, port: int, draft_model: str | None = None,
                  decode_concurrency: int = 1) -> subprocess.Popen | None:
    cmd = [sys.executable, "-m", "mlx_lm.server", "--model", model,
           "--host", "127.0.0.1", "--port", str(port)]
    if draft_model:
        cmd += ["--draft-model", draft_model]
    if decode_concurrency > 1:
        cmd += ["--decode-concurrency", str(decode_concurrency)]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    api_base = f"http://127.0.0.1:{port}/v1"
    for i in range(60):
        time.sleep(1)
        if proc.poll() is not None:
            stderr_tail = proc.stderr.read().decode(errors="replace")[-500:] if proc.stderr else ""
            print(f"  ERROR: mlx_lm.server exited early (rc={proc.returncode}): {stderr_tail}")
            return None
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            if r.status_code == 200:
                print(f"  mlx_lm.server ready on port {port} (after {i+1}s)")
                return proc
        except Exception:
            pass
    print(f"  ERROR: mlx_lm.server failed to start on port {port}")
    proc.terminate()
    return None


def run_benchmark(model_repo: str, passes: list[dict], output_dir: Path,
                  draft_model: str | None = None, ctx_cap: int = 131072,
                  temperature: float = 0.7, top_p: float = 0.8,
                  port: int = DEFAULT_PORT, decode_concurrency: int = 1,
                  quant: str = "?", cooldown: int = DEFAULT_COOLDOWN,
                   mem_guard: float = 80.0, dry_run: bool = False) -> Path:
    model_repo = os.path.expanduser(model_repo)
    model_name = model_repo.split("/")[-1]
    full_model_id = model_repo  # mlx server expects full repo path
    run_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"mlx_api_{model_name}_{ts}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    api_base = f"http://127.0.0.1:{port}/v1"

    server_proc = None
    if not dry_run:
        server_proc = _start_server(model_repo, port, draft_model, decode_concurrency)
        if not server_proc:
            sys.exit(1)

    try:
        for i, p in enumerate(passes):
            ctx_used = context_budget_for_pass(p, ctx_cap)
            prompt_text, target_prompt_tokens = build_prompt_for_pass(p, ctx_used)
            mem_pct = psutil.virtual_memory().percent
            if mem_pct >= mem_guard:
                _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                    "model_name": model_name, "backend": "MLX_API", "pass_name": p["id"],
                    "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": 0, "gen_tokens": 0,
                    "prompt_tps": 0.0, "gen_tps": 0.0, "ttft_s": 0.0, "peak_mem_pct": round(mem_pct, 1),
                    "status": f"skipped_mem_{mem_pct:.0f}pct", "quant": quant, "concurrency": 1,
                    "mtp": "on" if draft_model else "off", "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": "none", "generated_text": ""}, out_csv)
                print(f"  SKIP {p['id']}: mem {mem_pct:.0f}%")
                continue

            if dry_run:
                target = f" target_prompt~{target_prompt_tokens}" if target_prompt_tokens else ""
                print(f"  [DRY RUN] {p['id']}: ctx={ctx_used}{target} gen={p['gen_tokens']}")
                continue

            payload = {"model": full_model_id,
                       "messages": [{"role": "user", "content": prompt_text}],
                       "max_tokens": p["gen_tokens"], "temperature": temperature,
                       "top_p": top_p, "stream": True,
                       "stream_options": {"include_usage": True}}
            request_timeout = max(600, min(3600, int(ctx_used / 100) + int(p["gen_tokens"] / 2)))

            sampler = MemSampler([server_proc.pid] if server_proc else []); sampler.start()
            gt_out = 0; pt_out = 0; ttft = 0.0; tps = 0.0; status = "ok"
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            usage_completion = 0
            usage_prompt = 0
            try:
                t0 = time.perf_counter(); ttft_started = False
                resp = requests.post(f"{api_base}/chat/completions", json=payload,
                                     stream=True, timeout=request_timeout)
                if resp.status_code != 200:
                    status = f"http_{resp.status_code}"
                else:
                    for line in resp.iter_lines():
                        if not line: continue
                        line_str = line if isinstance(line, str) else line.decode("utf-8", errors="ignore")
                        if line_str.startswith("data: "):
                            line_str = line_str[6:]
                        if line_str.strip() in ("[DONE]", ""):
                            continue
                        try: chunk = json.loads(line_str)
                        except Exception: continue
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            ct = usage.get("completion_tokens")
                            pt = usage.get("prompt_tokens")
                            if isinstance(ct, int) and ct > 0: usage_completion = ct
                            if isinstance(pt, int) and pt > 0: usage_prompt = pt
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content_text = delta.get("content") or delta.get("text") or ""
                        reasoning_text = delta.get("reasoning") or delta.get("reasoning_content") or ""
                        if content_text or reasoning_text:
                            if not ttft_started: ttft = time.perf_counter() - t0; ttft_started = True
                        if content_text:
                            content_parts.append(content_text)
                        if reasoning_text:
                            reasoning_parts.append(reasoning_text)
                elapsed = time.perf_counter() - t0
                if usage_completion > 0:
                    gt_out = usage_completion
                    pt_out = usage_prompt
                    token_count_method = "openai_usage"
                else:
                    if reasoning_parts:
                        if content_parts:
                            gt_out = len("".join(content_parts).split()) + len("".join(reasoning_parts).split())
                            token_count_method = "word_fallback_with_reasoning"
                        else:
                            gt_out = len("".join(reasoning_parts).split())
                            token_count_method = "word_fallback_no_reasoning_untrusted"
                    else:
                        gt_out = len("".join(content_parts).split())
                        token_count_method = "word_fallback"
                tps = gt_out / (elapsed - ttft) if (elapsed - ttft) > 0 and gt_out > 0 else 0.0
                if ttft == 0: status = "no_tokens"
            except Exception as exc:
                status = f"error:{exc}"
                token_count_method = "none"
            finally:
                peak = sampler.stop()
            if target_prompt_tokens and pt_out and pt_out < int(target_prompt_tokens * 0.8):
                status = f"underfilled_context_prompt_{pt_out}_target_{target_prompt_tokens}"

            _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                "model_name": model_name, "backend": "MLX_API", "pass_name": p["id"],
                "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": pt_out, "gen_tokens": gt_out,
                "prompt_tps": 0.0, "gen_tps": round(tps, 2), "ttft_s": round(ttft, 3),
                "peak_mem_pct": peak, "status": status, "quant": quant, "concurrency": 1,
                "mtp": "on" if draft_model else "off", "draft_tokens": 0, "draft_accepted_tokens": 0, "draft_accept_rate": 0.0, "token_count_method": token_count_method,
                "generated_text": "".join(content_parts) + "".join(reasoning_parts)}, out_csv)

            ok = status == "ok"
            print(f"  {'OK' if ok else 'FAIL'} {p['id']}: ttft {ttft:.2f}s · gen {tps:.1f} t/s · mem {peak:.0f}%")
            if i < len(passes) - 1: time.sleep(cooldown)

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait(timeout=10)

    print(f"  CSV → {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark via mlx_lm.server API")
    ap.add_argument("--model", type=str, required=True, help="HF repo or local path to MLX model")
    ap.add_argument("--draft-model", type=str, help="Draft model for MTP speculative decoding")
    ap.add_argument("--output-dir", type=Path, default=Path("results/mlx_api"))
    ap.add_argument("--passes", nargs="+", default=["micro", "normal", "high", "max"],
                    choices=["micro", "normal", "high", "max"])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--decode-concurrency", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    ap.add_argument("--mem-guard", type=float, default=90.0)
    ap.add_argument("--ctx-cap", type=int, default=131072)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--quant", type=str, default="?", help="Quant label for CSV")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    selected_passes = [p for p in PASSES if p["id"].split("_")[2] in args.passes]
    run_benchmark(args.model, selected_passes, args.output_dir,
                  args.draft_model, args.ctx_cap, args.temperature, args.top_p,
                  args.port, args.decode_concurrency, args.quant, args.cooldown,
                  args.mem_guard, args.dry_run)


if __name__ == "__main__":
    main()
