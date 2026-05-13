#!/usr/bin/env python3
"""
bench_llamacpp_api.py — Benchmark via llama-server OpenAI-compatible API.

Requires llama-server running with a model loaded. Benchmarks via streaming
chat completions. Measures TTFT, gen TPS, peak memory.

Usage:
  python3 scripts/bench_llamacpp_api.py --url http://127.0.0.1:8080/v1 --model model-name
  python3 scripts/bench_llamacpp_api.py --passes micro --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import requests
from urllib.parse import urlparse

FIELDS = [
    "timestamp", "run_id", "model_name", "backend", "pass_name",
    "ctx_cap", "ctx_used", "prompt_tokens", "gen_tokens",
    "prompt_tps", "gen_tps", "ttft_s", "peak_mem_pct", "status",
    "quant", "concurrency", "mtp", "draft_tokens", "draft_accepted_tokens",
    "draft_accept_rate", "token_count_method", "generated_text",
]

DEFAULT_COOLDOWN = 30

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from prompts import PASSES, build_prompt_for_pass, context_budget_for_pass


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


def run_benchmark(model_id: str, passes: list[dict], output_dir: Path,
                  api_base: str = "http://127.0.0.1:8080/v1",
                  ctx_cap: int = 131072, family: str = "?",
                  temperature: float = 0.7, top_p: float = 0.8,
                  quant: str = "?", cooldown: int = DEFAULT_COOLDOWN,
                  mem_guard: float = 80.0, dry_run: bool = False,
                  mtp: bool = False) -> Path:
    run_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"llamacpp_api_{model_id}_{ts}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        # Verify server is reachable. /health lives at the host root, not under /v1.
        parsed = urlparse(api_base)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        try:
            health = requests.get(health_url, timeout=3)
            if health.status_code != 200:
                print(f"WARNING: llama-server unhealthy at {health_url}")
        except Exception:
            print(f"WARNING: Cannot reach llama-server at {health_url}")

    for i, p in enumerate(passes):
        ctx_used = context_budget_for_pass(p, ctx_cap)
        prompt_text, target_prompt_tokens = build_prompt_for_pass(p, ctx_used)
        if dry_run:
            target = f" target_prompt~{target_prompt_tokens}" if target_prompt_tokens else ""
            print(f"  [DRY RUN] {p['id']}: ctx={ctx_used}{target} gen={p['gen_tokens']}")
            continue

        mem_pct = psutil.virtual_memory().percent
        if mem_pct >= mem_guard:
            _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
                "model_name": model_id, "backend": "LLAMACPP_API", "pass_name": p["id"],
                "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": 0, "gen_tokens": 0,
                "prompt_tps": 0.0, "gen_tps": 0.0, "ttft_s": 0.0, "peak_mem_pct": round(mem_pct, 1),
                "status": f"skipped_mem_{mem_pct:.0f}pct", "quant": quant, "concurrency": 1,
                "mtp": "on" if mtp else "off", "draft_tokens": 0, "draft_accepted_tokens": 0,
                "draft_accept_rate": 0.0, "token_count_method": "none", "generated_text": ""}, out_csv)
            print(f"  SKIP {p['id']}: mem {mem_pct:.0f}%")
            continue

        system_msg = "" if family == "gemma4" else "You are a helpful coding assistant."
        payload = {"model": model_id, "messages": [{"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text}], "max_tokens": p["gen_tokens"],
            "temperature": temperature, "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True}}
        request_timeout = max(600, min(3600, int(ctx_used / 100) + int(p["gen_tokens"] / 2)))

        def _run_streaming_pass(p: dict, attempt: int = 1) -> dict:
            sampler = MemSampler(); sampler.start()
            gt_out = 0; pt_out = 0; ttft = 0.0; tps = 0.0; status = "ok"
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            usage_completion = 0
            usage_prompt = 0
            draft_n = 0
            draft_n_accepted = 0
            token_count_method = "word_fallback"
            try:
                t0 = time.perf_counter(); ttft_started = False
                resp = requests.post(f"{api_base}/chat/completions", json=payload, stream=True, timeout=request_timeout)
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
                            ct = usage.get("completion_tokens") or usage.get("completion_tokens_detail", {}).get("completion_tokens")
                            pt = usage.get("prompt_tokens")
                            if isinstance(ct, int) and ct > 0: usage_completion = ct
                            if isinstance(pt, int) and pt > 0: usage_prompt = pt
                        timings = chunk.get("timings")
                        if isinstance(timings, dict):
                            dn = timings.get("draft_n")
                            da = timings.get("draft_n_accepted")
                            if isinstance(dn, int) and dn > draft_n: draft_n = dn
                            if isinstance(da, int) and da > draft_n_accepted: draft_n_accepted = da
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content_text = delta.get("content") or delta.get("text") or ""
                        reasoning_text = delta.get("reasoning_content") or delta.get("reasoning") or ""
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
                        token_count_method = "word_fallback_with_reasoning"
                        gt_out = len("".join(content_parts).split()) + len("".join(reasoning_parts).split())
                    else:
                        token_count_method = "word_fallback"
                        gt_out = len("".join(content_parts).split())
                tps = gt_out / (elapsed - ttft) if (elapsed - ttft) > 0 and gt_out > 0 else 0.0
                if ttft == 0: status = "no_tokens"
            except Exception as exc:
                status = f"error:{exc}"
                token_count_method = "none"
            finally:
                peak = sampler.stop()
            return {
                "peak": peak, "status": status, "pt_out": pt_out, "gt_out": gt_out,
                "ttft": ttft, "tps": tps, "token_count_method": token_count_method,
                "generated_text": "".join(content_parts) + "".join(reasoning_parts),
                "draft_n": draft_n, "draft_n_accepted": draft_n_accepted,
            }

        result = _run_streaming_pass(p)
        if result["status"] == "no_tokens":
            print(f"  Retry {p['id']} after no_tokens...")
            time.sleep(cooldown // 3)
            result = _run_streaming_pass(p, attempt=2)

        peak = result["peak"]
        status = result["status"]
        pt_out = result["pt_out"]
        gt_out = result["gt_out"]
        ttft = result["ttft"]
        tps = result["tps"]
        token_count_method = result["token_count_method"]
        generated_text = result["generated_text"]
        draft_tokens = result["draft_n"] if mtp else 0
        draft_accepted_tokens = result["draft_n_accepted"] if mtp else 0
        draft_accept_rate = (draft_accepted_tokens / draft_tokens) if draft_tokens > 0 else 0.0
        if target_prompt_tokens and pt_out and pt_out < int(target_prompt_tokens * 0.8):
            status = f"underfilled_context_prompt_{pt_out}_target_{target_prompt_tokens}"
        if mtp and status == "ok" and draft_tokens <= 0:
            status = "warn_mtp_no_draft_timings"

        _csv_append({"timestamp": datetime.now().isoformat(), "run_id": run_id,
            "model_name": model_id, "backend": "LLAMACPP_API", "pass_name": p["id"],
            "ctx_cap": ctx_cap, "ctx_used": ctx_used, "prompt_tokens": pt_out, "gen_tokens": gt_out,
            "prompt_tps": 0.0, "gen_tps": round(tps, 2), "ttft_s": round(ttft, 3),
            "peak_mem_pct": peak, "status": status, "quant": quant, "concurrency": 1,
            "mtp": "on" if mtp else "off", "draft_tokens": draft_tokens,
            "draft_accepted_tokens": draft_accepted_tokens,
            "draft_accept_rate": round(draft_accept_rate, 3),
            "token_count_method": token_count_method, "generated_text": generated_text}, out_csv)

        ok = status == "ok"
        print(f"  {'OK' if ok else 'FAIL'} {p['id']}: ttft {ttft:.2f}s · gen {tps:.1f} t/s · mem {peak:.0f}%")
        if i < len(passes) - 1: time.sleep(cooldown)

    print(f"  CSV → {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark via llama-server API")
    ap.add_argument("--model", type=str, required=True, help="Model name (as registered with server)")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:8080/v1", help="llama-server API base URL")
    ap.add_argument("--output-dir", type=Path, default=Path("results/llamacpp_api"))
    ap.add_argument("--passes", nargs="+", default=["micro", "normal", "high", "max"],
                    choices=["micro", "normal", "high", "max"])
    ap.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    ap.add_argument("--mem-guard", type=float, default=80.0)
    ap.add_argument("--ctx-cap", type=int, default=131072)
    ap.add_argument("--family", type=str, default="?", help="Model family (gemma4, qwen3, etc.)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--quant", type=str, default="?", help="Quant label for CSV")
    ap.add_argument("--mtp", action="store_true", help="Mark run as using MTP speculative decoding")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    selected_passes = [p for p in PASSES if p["id"].split("_")[2] in args.passes]
    run_benchmark(args.model, selected_passes, args.output_dir, args.url,
                  args.ctx_cap, args.family, args.temperature, args.top_p,
                  args.quant, args.cooldown, args.mem_guard, args.dry_run,
                  args.mtp)


if __name__ == "__main__":
    main()
