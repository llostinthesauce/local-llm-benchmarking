#!/usr/bin/env python3
"""
bench_config_compare.py — same model, two server configs, head to head.

Unlike bench_head_to_head.py (different models, MLX only), this isolates ONE
serving knob by running the identical model + prompt suite under two configs and
diffing throughput / memory / recall accuracy.

Suites:
  mtp   llama.cpp GGUF, MTP speculative decoding OFF vs ON (same .gguf file).
        Captures llama-server draft-acceptance from the server log.
  kv    MLX via mlx_vlm.server, KV cache fp16 vs q8 turboquant (same weights,
        same engine — only --kv-bits differs, so the delta is pure KV quant).

Reuses the prompt suite + recall scoring from bench_head_to_head.

Usage:
  python3 scripts/bench_config_compare.py --suite mtp --model qwen35-mtp
  python3 scripts/bench_config_compare.py --suite mtp --model gemma26
  python3 scripts/bench_config_compare.py --suite kv  --model qwen35
  python3 scripts/bench_config_compare.py --suite kv  --model gemma26
  python3 scripts/bench_config_compare.py --suite mtp --model qwen35-mtp --only code_debug code_algo
  python3 scripts/bench_config_compare.py --suite kv --model gemma26 --dry-run

Writes results/config_compare/<suite>_<model>_<ts>.csv  (+ .md transcript).
The CSV is the artifact to analyze.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_registry
import memutil
from bench_head_to_head import _build_prompts, _recall_pass  # reuse suite + scorer

try:
    import requests
except ImportError:
    raise SystemExit("pip install requests")
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    raise SystemExit("pip install rich")

HOST = "127.0.0.1"
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "config_compare"
OFFLINE_ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# Base llama-server flags, mirrored from serve_local.sh so GGUF runs are faithful.
LLAMA_BASE = ["-ngl", "all", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
              "--mlock", "--no-mmap", "-t", "8", "-tb", "16"]

CSV_FIELDS = [
    "timestamp", "suite", "model", "config", "prompt_id", "kind",
    "ttft_s", "gen_tps", "output_tokens", "recall_pass", "draft_accept",
    "peak_mem_pct", "status",
]

# Default prompt subsets per suite (override with --only). MTP cares about
# generation throughput; KV cares about long-context memory + recall fidelity.
SUITE_DEFAULT_PROMPTS = {
    "mtp": ["recall_32k", "code_algo", "code_debug", "code_arch"],
    "kv":  ["recall_32k", "recall_64k", "recall_128k", "code_debug"],
}


class MemSampler:
    """Peak system-memory % over the lifetime of one config run."""
    def __init__(self):
        self.peak = memutil.virtual_memory().percent
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, memutil.virtual_memory().percent)
            time.sleep(0.25)

    def start(self):
        self._t.start()

    def stop(self) -> float:
        self._stop.set()
        self._t.join(timeout=2)
        return round(self.peak, 1)


def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["lsof", "-i", f"tcp:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        for pid in out.splitlines():
            if pid.strip():
                subprocess.run(["kill", pid.strip()], check=False)
        time.sleep(1.5)
    except Exception:
        pass


def _wait_ready(port: int, proc: subprocess.Popen, label: str, timeout: int = 180) -> bool:
    for i in range(timeout):
        time.sleep(1)
        if proc.poll() is not None:
            console.print(f"  [red]{label} exited rc={proc.returncode} during load[/red]")
            return False
        try:
            r = requests.get(f"http://{HOST}:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                console.print(f"  [dim]{label} ready ({i + 1}s)[/dim]")
                return True
        except Exception:
            pass
    console.print(f"  [red]{label} not ready within {timeout}s[/red]")
    return False


def _infer(base_url: str, prompt: str, model_id: str, max_tokens: int, temperature: float) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text = ""
    thinking = ""
    ttft = None
    tokens = 0
    t0 = time.time()
    with requests.post(f"{base_url}/chat/completions", json=payload,
                       stream=True, timeout=900) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        for raw in resp.iter_lines(decode_unicode=True):
            line = raw if isinstance(raw, str) else raw.decode("utf-8", "ignore")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data in ("[DONE]", ""):
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(f"Model error: {chunk['error']}")
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            c = delta.get("content") or delta.get("text") or ""
            th = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if c or th:
                if ttft is None:
                    ttft = time.time() - t0
                text += c
                thinking += th
                tokens += 1
    if not text.strip() and thinking.strip():
        text = thinking
    elapsed = time.time() - t0
    gen_time = elapsed - (ttft or elapsed)
    tps = tokens / gen_time if gen_time > 0 and tokens > 1 else 0.0
    return {"text": text.strip(), "ttft": ttft or 0.0, "tps": tps, "tokens": tokens}


def _parse_draft_accept(log_path: Path, seen_offset: int) -> tuple[float | None, int]:
    """Return (latest draft-acceptance ratio, new byte offset) from a llama-server log."""
    try:
        data = log_path.read_bytes()
    except FileNotFoundError:
        return None, seen_offset
    new = data[seen_offset:].decode("utf-8", "ignore")
    matches = re.findall(r"draft acceptance\s*=\s*([0-9.]+)", new)
    ratio = float(matches[-1]) if matches else None
    return ratio, len(data)


# ── suite → list of (config_name, builder) ──────────────────────────────────

def _mtp_configs(row: dict, ctx: int) -> list[dict]:
    gguf = row["path"]
    is_gemma = row["family"] == "gemma4"
    n_max = str(row.get("spec_draft_n_max") or 2)
    # Separate draft head (Gemma) vs embedded/self-speculative (Qwen).
    draft = ""
    model_dir = Path(gguf).parent
    cands = sorted(model_dir.glob("mtp-*.gguf"))
    if cands:
        draft = str(cands[0])
    base = ["llama-server", "-m", gguf, "-c", str(ctx), "--host", HOST,
            "--port", "8080", *LLAMA_BASE]
    if is_gemma:
        base += ["--chat-template", "gemma2"]
    off = list(base)
    on = list(base) + ["--spec-type", "draft-mtp", "--spec-draft-n-max", n_max, "-np", "1"]
    if draft:
        on += ["--spec-draft-model", draft]
    return [
        {"name": "mtp_off", "cmd": off, "port": 8080, "model_id": gguf,
         "note": "speculative decoding disabled"},
        {"name": "mtp_on", "cmd": on, "port": 8080, "model_id": gguf,
         "note": f"draft-mtp n-max={n_max}" + (f" + draft head" if draft else " self-speculative")},
    ]


def _kv_configs(row: dict, ctx: int) -> list[dict]:
    path = row["path"]
    base = [sys.executable, "-m", "mlx_vlm.server", "--model", path,
            "--host", HOST, "--port", "8085"]
    return [
        {"name": "kv_fp16", "cmd": list(base), "port": 8085, "model_id": path,
         "note": "mlx_vlm KV fp16 (baseline)"},
        {"name": "kv_q8", "cmd": list(base) + ["--kv-bits", "8", "--kv-quant-scheme", "turboquant"],
         "port": 8085, "model_id": path, "note": "mlx_vlm KV q8 turboquant"},
    ]


def run(suite: str, model_sel: str, only: list[str] | None, ctx: int,
        temperature: float, dry_run: bool) -> None:
    backend = "llamacpp" if suite == "mtp" else "mlx"
    row = model_registry.resolve(model_sel, backend, model_registry.DEFAULT_CONFIG)
    if not row.get("exists"):
        raise SystemExit(f"Model '{model_sel}' ({backend}) not on disk: {row['path']}")
    if suite == "mtp" and not row.get("mtp_supported"):
        raise SystemExit(f"'{model_sel}' has no MTP head (mtp_supported=false). "
                         f"Use qwen35-mtp or gemma26.")

    if temperature < 0:
        temperature = float(row.get("temperature", 0.6))

    prompts = _build_prompts()
    chosen = only or SUITE_DEFAULT_PROMPTS[suite]
    bad = [p for p in chosen if p not in prompts]
    if bad:
        raise SystemExit(f"Unknown prompt IDs {bad}. Valid: {list(prompts)}")
    prompts = {k: prompts[k] for k in chosen}

    configs = _mtp_configs(row, ctx) if suite == "mtp" else _kv_configs(row, ctx)
    base_url = f"http://{HOST}:{configs[0]['port']}/v1"

    console.rule(f"[bold]{suite.upper()} compare — {row['name']}[/bold]")
    console.print(f"  configs: {' vs '.join(c['name'] for c in configs)}")
    console.print(f"  prompts: {', '.join(prompts)}")
    console.print(f"  temp={temperature}  ctx={ctx}\n")
    if dry_run:
        for c in configs:
            console.print(f"  [cyan]{c['name']}[/cyan] ({c['note']})")
            console.print(f"    {' '.join(str(x) for x in c['cmd'])}\n")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"{suite}_{model_sel.replace('/', '_')}_{ts}.csv"
    out_md = RESULTS_DIR / f"{suite}_{model_sel.replace('/', '_')}_{ts}.md"
    log_path = Path(f"/tmp/bench_cfg_{suite}_{ts}.log")

    rows_out: list[dict] = []
    transcripts: list[str] = []

    for c in configs:
        console.print(f"\n[bold cyan]══ {c['name']} ══[/bold cyan]  [dim]{c['note']}[/dim]")
        _kill_port(c["port"])
        if log_path.exists():
            log_path.unlink()
        log_f = open(log_path, "wb")
        proc = subprocess.Popen(c["cmd"], stdout=log_f, stderr=subprocess.STDOUT, env=OFFLINE_ENV)
        if not _wait_ready(c["port"], proc, c["name"]):
            for pid in prompts:
                rows_out.append({"timestamp": datetime.now().isoformat(), "suite": suite,
                                 "model": row["name"], "config": c["name"], "prompt_id": pid,
                                 "kind": prompts[pid]["kind"], "status": "server_failed"})
            proc.terminate()
            log_f.close()
            continue

        mem = MemSampler(); mem.start()
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        for pid, pd in prompts.items():
            approx = len(pd["text"]) // 4
            console.print(f"  {pd['label']} (~{approx:,} tok)… ", end="")
            sys.stdout.flush()
            rec = {"timestamp": datetime.now().isoformat(), "suite": suite,
                   "model": row["name"], "config": c["name"], "prompt_id": pid,
                   "kind": pd["kind"], "peak_mem_pct": "", "draft_accept": ""}
            try:
                r = _infer(base_url, pd["text"], c["model_id"], pd["max_tokens"], temperature)
                draft, log_offset = _parse_draft_accept(log_path, log_offset)
                passed = _recall_pass(r["text"], pd["answer"]) if pd["kind"] == "recall" else None
                rec.update({"ttft_s": round(r["ttft"], 3), "gen_tps": round(r["tps"], 2),
                            "output_tokens": r["tokens"],
                            "recall_pass": "" if passed is None else passed,
                            "draft_accept": "" if draft is None else round(draft, 4),
                            "status": "ok"})
                tag = ""
                if passed is not None:
                    tag = "[green]PASS[/green] " if passed else "[red]FAIL[/red] "
                da = f"  draft={draft:.2f}" if draft is not None else ""
                console.print(f"{tag}ttft={r['ttft']:.1f}s  {r['tps']:.1f}t/s{da}")
                transcripts.append(f"### [{c['name']}] {pd['label']}\n\n"
                                   f"ttft {r['ttft']:.1f}s · {r['tps']:.1f} t/s"
                                   f"{' · draft '+format(draft,'.3f') if draft is not None else ''}\n\n"
                                   f"```\n{r['text'][:4000]}\n```\n")
            except Exception as exc:
                rec["status"] = f"error:{str(exc)[:120]}"
                console.print(f"[red]ERR {exc}[/red]")
            rows_out.append(rec)

        peak = mem.stop()
        for rec in rows_out:
            if rec["config"] == c["name"] and not rec.get("peak_mem_pct"):
                rec["peak_mem_pct"] = peak
        console.print(f"  [dim]peak mem {peak}%[/dim]")
        proc.terminate()
        try:
            proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_f.close()
        time.sleep(3)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in rows_out:
            w.writerow(rec)
    with open(out_md, "w") as f:
        f.write(f"# {suite.upper()} config compare — {row['name']}\n\n{ts}\n\n")
        f.write("\n".join(transcripts))

    console.print(f"\n[bold green]CSV  → {out_csv}[/bold green]")
    console.print(f"[bold green]MD   → {out_md}[/bold green]")
    _print_summary(rows_out, configs, prompts, suite)


def _print_summary(rows_out, configs, prompts, suite) -> None:
    console.rule("[bold]Summary[/bold]")
    tbl = Table(show_header=True, header_style="bold", show_lines=True)
    tbl.add_column("Prompt", min_width=24)
    for c in configs:
        tbl.add_column(c["name"], justify="center")
    if len(configs) == 2:
        tbl.add_column("Δ gen_tps", justify="center")
    for pid, pd in prompts.items():
        cells = [pd["label"]]
        tps_vals = []
        for c in configs:
            rec = next((r for r in rows_out if r["config"] == c["name"] and r["prompt_id"] == pid), {})
            if rec.get("status") != "ok":
                cells.append("[red]—[/red]"); tps_vals.append(None); continue
            t = rec.get("gen_tps", 0)
            tps_vals.append(t)
            extra = ""
            if rec.get("recall_pass") not in ("", None):
                extra = " ✓" if rec["recall_pass"] is True else " ✗"
            if rec.get("draft_accept") not in ("", None):
                extra += f" d={rec['draft_accept']}"
            cells.append(f"{rec.get('ttft_s',0):.1f}s/{t:.1f}t/s{extra}")
        if len(configs) == 2 and tps_vals[0] and tps_vals[1]:
            delta = (tps_vals[1] / tps_vals[0] - 1) * 100
            cells.append(f"{delta:+.0f}%")
        elif len(configs) == 2:
            cells.append("—")
        tbl.add_row(*cells)
    console.print(tbl)


def main() -> None:
    ap = argparse.ArgumentParser(description="Same model, two server configs, head to head")
    ap.add_argument("--suite", required=True, choices=["mtp", "kv"])
    ap.add_argument("--model", required=True, help="Registry alias (qwen35-mtp, gemma26, qwen35, …)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="Prompt IDs to run (default: suite preset)")
    ap.add_argument("--ctx", type=int, default=0, help="llama.cpp context size (mtp suite); default per suite")
    ap.add_argument("--temp", type=float, default=-1.0, help="Sampling temp; default = model card value")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ctx = args.ctx or (131072 if args.suite == "kv" else 65536)
    run(args.suite, args.model, args.only, ctx, args.temp, args.dry_run)


if __name__ == "__main__":
    main()
