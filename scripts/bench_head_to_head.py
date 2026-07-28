#!/usr/bin/env python3
"""
MoE head-to-head: context recall + coding

Tests gemma4_26b_moe vs qwen3_35b_moe on:
  recall_32k   Needle-in-haystack at ~32K token depth
  recall_64k   Needle at ~64K token depth
  recall_128k  Needle at ~128K token depth
  code_algo    Implement a sliding context-window class
  code_debug   Find 5 bugs in an LRU cache
  code_arch    Design benchmark orchestrator ABCs

Recall tests score PASS/FAIL automatically.
Coding responses are printed and saved to results/head_to_head/.

Usage:
    python3 scripts/bench_head_to_head.py
    python3 scripts/bench_head_to_head.py --models qwen3_35b_moe gemma4_26b_moe
    python3 scripts/bench_head_to_head.py --only recall_32k code_debug
    python3 scripts/bench_head_to_head.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_registry

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

PORT = 8085
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}/v1"
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "head_to_head"

# ── filler text for context recall ──────────────────────────────────────────
# This paragraph is ~420 chars ≈ 105 tokens at 4 chars/token.
_FILLER_UNIT = (
    "This document describes the internal API for the DataStream processing service, "
    "version 3.4. Each endpoint follows REST conventions per RFC 7231. Request bodies "
    "use JSON encoding. Authentication uses bearer tokens in the Authorization header. "
    "Rate limits apply per API key. The service exposes endpoints for ingestion, "
    "transformation, and retrieval. All timestamps are UTC ISO-8601. Pagination uses "
    "cursor-based tokens, not page offsets.\n\n"
)
_CHARS_PER_TOKEN = 4
_UNIT_TOKENS = len(_FILLER_UNIT) // _CHARS_PER_TOKEN


def _make_filler(target_tokens: int, needle: str, needle_frac: float) -> str:
    n = max(1, target_tokens // _UNIT_TOKENS)
    needle_idx = max(0, int(n * needle_frac))
    parts: list[str] = []
    for i in range(n):
        if i == needle_idx:
            parts.append(f"\n\n{needle}\n\n")
        parts.append(_FILLER_UNIT)
    return "".join(parts)


def _recall_prompt(target_tokens: int, needle: str, question: str,
                   needle_frac: float = 0.5) -> str:
    filler = _make_filler(target_tokens, needle, needle_frac)
    return (
        f"{filler}\n\n"
        "Answer the following question with only the exact value — no explanation.\n\n"
        f"{question}"
    )


# ── buggy program for the debug prompt ───────────────────────────────────────

_BUGGY_CACHE = '''\
import threading
from collections import OrderedDict


class ModelResponseCache:
    """LRU cache for model API responses."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key, last=True)   # modifies _cache without lock
            return self._cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key, last=True)
            self._cache[key] = value
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=True)          # last=True evicts newest, not oldest

    def invalidate_prefix(self, prefix: str) -> int:
        removed = 0
        with self._lock:
            keys = [k for k in self._cache if k is prefix]   # `is` checks identity, not equality
            for k in keys:
                del self._cache[k]
                removed += 1
        return removed

    def warm(self, entries: dict[str, str]) -> None:
        for k, v in entries.items():
            self.put(k, v)
            if len(self._cache) >= self.capacity:      # >= stops one entry short of capacity
                break

    def get_many(self, keys: list[str]) -> dict[str, str]:
        return {k: self._cache[k] for k in keys if k in self._cache}  # no lock; dict comprehension unsafe under concurrent put()
'''


# ── prompt registry ───────────────────────────────────────────────────────────

def _build_prompts() -> dict[str, dict]:
    return {
        "recall_32k": {
            "label": "Context recall @ 32K tokens",
            "kind": "recall",
            "answer": "MAPLE-CLOUD-7734",
            "max_tokens": 512,
            "text": _recall_prompt(
                32_000,
                "SYSTEM: Recovery passphrase for backup vault B-7 is: MAPLE-CLOUD-7734.",
                "What is the recovery passphrase for backup vault B-7?",
                needle_frac=0.5,
            ),
        },
        "recall_64k": {
            "label": "Context recall @ 64K tokens",
            "kind": "recall",
            "answer": "ZENITH-KEY-0291",
            "max_tokens": 512,
            "text": _recall_prompt(
                64_000,
                "NOTICE: Internal service mesh token for region us-west is: ZENITH-KEY-0291.",
                "What is the service mesh token for region us-west?",
                needle_frac=0.25,
            ),
        },
        "recall_128k": {
            "label": "Context recall @ 128K tokens",
            "kind": "recall",
            "answer": "FALCON-NODE-K9",
            "max_tokens": 512,
            "text": _recall_prompt(
                128_000,
                "CRITICAL: Failover cluster node identifier: FALCON-NODE-K9. Required for DR activation.",
                "What is the failover cluster node identifier?",
                needle_frac=0.75,
            ),
        },
        "code_algo": {
            "label": "Code — sliding context window",
            "kind": "coding",
            "max_tokens": 6000,
            "text": (
                "Implement a `ContextWindow` class in Python for LLM token budget management.\n\n"
                "Requirements:\n"
                "1. `.add(chunk: str)` — appends a text chunk\n"
                "2. Enforces a `token_limit` (estimate: `len(text) // 4 == tokens`)\n"
                "3. Evicts the oldest chunks when the limit would be exceeded\n"
                "4. `.get() -> str` — returns current window as one string\n"
                "5. `.token_count` property — returns current estimate\n"
                "6. Thread-safe (single `threading.Lock`)\n\n"
                "Output only the Python class. No prose."
            ),
        },
        "code_debug": {
            "label": "Code — find 5 bugs in LRU cache",
            "kind": "coding",
            "max_tokens": 6000,
            "text": (
                "Find and fix all 5 bugs in this Python class. "
                "For each: (a) identify the exact line(s), "
                "(b) explain what is wrong, (c) show the corrected line(s).\n\n"
                f"```python\n{_BUGGY_CACHE}```"
            ),
        },
        "code_arch": {
            "label": "Code — benchmark orchestrator ABCs",
            "kind": "coding",
            "max_tokens": 6000,
            "text": (
                "Design Python abstract base classes for a local LLM benchmark orchestrator.\n\n"
                "Interfaces required:\n"
                "1. `ModelBackend(ABC)` — `start()`, `stop()`, "
                "`infer(prompt: str, max_tokens: int, temperature: float) -> InferResult`\n"
                "2. `InferResult` dataclass — `ttft_s: float`, `gen_tps: float`, "
                "`output_tokens: int`, `text: str`\n"
                "3. `BenchPass(ABC)` — `id: str`, "
                "`build_prompt(ctx_budget: int) -> str`, `max_output_tokens: int`\n"
                "4. `RunResult` dataclass — `backend_name: str`, `pass_id: str`, "
                "`result: InferResult`\n"
                "5. `Orchestrator` — takes `backends: list[ModelBackend]` and "
                "`passes: list[BenchPass]`, runs all combinations, returns "
                "`list[RunResult]`. New backends must not require modifying this class.\n\n"
                "ABCs and dataclasses only. Full type annotations. No implementation bodies."
            ),
        },
    }


# ── server lifecycle ──────────────────────────────────────────────────────────

def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["lsof", "-i", f"tcp:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        for pid in out.splitlines():
            if pid.strip():
                subprocess.run(["kill", pid.strip()], check=False)
        time.sleep(1)
    except Exception:
        pass


def _start_server(model_path: str) -> subprocess.Popen | None:
    _kill_port(PORT)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cmd = [sys.executable, "-m", "mlx_lm.server", "--model", model_path,
           "--host", HOST, "--port", str(PORT)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    for i in range(90):
        time.sleep(1)
        if proc.poll() is not None:
            err = (proc.stderr.read().decode(errors="replace")[-600:] if proc.stderr else "")
            console.print(f"  [red]Server exited (rc={proc.returncode}): {err}[/red]")
            return None
        try:
            r = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                console.print(f"  [dim]mlx_lm.server ready (after {i + 1}s)[/dim]")
                return proc
        except Exception:
            pass
    console.print("  [red]Server did not start within 90s[/red]")
    proc.terminate()
    return None


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(2)


# ── inference ─────────────────────────────────────────────────────────────────

def _infer(prompt: str, model_id: str, max_tokens: int = 512, temperature: float = 0.1) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text = ""
    thinking_buf = ""
    ttft: float | None = None
    token_count = 0
    t0 = time.time()

    with requests.post(f"{BASE_URL}/chat/completions", json=payload,
                       stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        for raw in resp.iter_lines(decode_unicode=True):
            line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")
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
            content = delta.get("content") or delta.get("text") or ""
            thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content or thinking:
                if ttft is None:
                    ttft = time.time() - t0
                if content:
                    text += content
                if thinking:
                    thinking_buf += thinking
                token_count += 1

    # if the model thought but never emitted content, surface the thinking as output
    if not text.strip() and thinking_buf.strip():
        text = thinking_buf

    elapsed = time.time() - t0
    gen_time = elapsed - (ttft or elapsed)
    tps = token_count / gen_time if gen_time > 0 and token_count > 1 else 0.0
    return {"text": text.strip(), "ttft": ttft or 0.0, "tps": tps, "tokens": token_count}


# ── scoring ───────────────────────────────────────────────────────────────────

def _recall_pass(response: str, expected: str) -> bool:
    return expected.lower() in response.lower()


# ── runner ────────────────────────────────────────────────────────────────────

def run(model_ids: list[str], only: list[str] | None = None, dry_run: bool = False) -> None:
    prompts = _build_prompts()
    if only:
        missing = [k for k in only if k not in prompts]
        if missing:
            raise SystemExit(f"Unknown prompt IDs: {missing}. Valid: {list(prompts)}")
        prompts = {k: prompts[k] for k in only}

    models = []
    for mid in model_ids:
        row = model_registry.resolve(mid, "mlx", model_registry.DEFAULT_CONFIG)
        if not row.get("exists"):
            raise SystemExit(f"Model '{mid}' not on disk. Check registry.")
        models.append({
            "id": mid, "path": row["path"],
            "name": row["name"], "ctx_cap": row["ctx_cap"],
        })

    if dry_run:
        console.rule("[bold]Dry run — approximate prompt sizes[/bold]")
        for pid, pd in prompts.items():
            approx = len(pd["text"]) // _CHARS_PER_TOKEN
            console.print(f"  {pid:<16}  ~{approx:>7,} tokens   {pd['label']}")
        console.print(f"\nModels: {', '.join(m['name'] for m in models)}")
        return

    results: dict[str, dict[str, dict]] = {m["id"]: {} for m in models}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for m in models:
        console.print(f"\n[bold cyan]══ {m['name']} ══[/bold cyan]")
        proc = _start_server(m["path"])
        if proc is None:
            for pid in prompts:
                results[m["id"]][pid] = {"error": "server failed to start"}
            continue

        for pid, pd in prompts.items():
            approx = len(pd["text"]) // _CHARS_PER_TOKEN
            console.print(f"  {pd['label']}  (~{approx:,} tok)… ", end="")
            sys.stdout.flush()
            try:
                r = _infer(pd["text"], model_id=m["path"], max_tokens=pd["max_tokens"], temperature=0.1)
                if pd["kind"] == "recall":
                    passed = _recall_pass(r["text"], pd["answer"])
                    color = "green" if passed else "red"
                    console.print(
                        f"[{color}]{'PASS' if passed else 'FAIL'}[/{color}]  "
                        f"ttft={r['ttft']:.1f}s  {r['tps']:.1f}t/s  "
                        f"→ {r['text'][:80]!r}"
                    )
                else:
                    console.print(f"ttft={r['ttft']:.1f}s  {r['tps']:.1f}t/s")
                results[m["id"]][pid] = r
            except Exception as exc:
                console.print(f"[red]ERROR: {exc}[/red]")
                results[m["id"]][pid] = {"error": str(exc)}

        _stop_server(proc)

    # ── summary table ─────────────────────────────────────────────────────────
    console.print("\n\n")
    console.rule("[bold]Summary[/bold]")
    tbl = Table(show_header=True, header_style="bold", show_lines=True)
    tbl.add_column("Test", min_width=34)
    for m in models:
        tbl.add_column(m["name"][:26], justify="center", min_width=22)

    for pid, pd in prompts.items():
        cells = [pd["label"]]
        for m in models:
            r = results[m["id"]].get(pid, {})
            if "error" in r:
                cells.append("[red]ERR[/red]")
            elif pd["kind"] == "recall":
                passed = _recall_pass(r.get("text", ""), pd["answer"])
                sym = "[green]PASS ✓[/green]" if passed else "[red]FAIL ✗[/red]"
                cells.append(f"{sym}\n{r.get('ttft', 0):.1f}s / {r.get('tps', 0):.1f}t/s")
            else:
                cells.append(f"{r.get('ttft', 0):.1f}s / {r.get('tps', 0):.1f}t/s")
        tbl.add_row(*cells)

    console.print(tbl)

    # ── coding responses: save to markdown, also print ────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = RESULTS_DIR / f"h2h_{ts}.md"
    with open(out_md, "w") as f:
        f.write(f"# Head-to-Head: {' vs '.join(m['name'] for m in models)}\n\n")
        f.write(f"Run: {ts}\n\n---\n\n")
        for pid, pd in prompts.items():
            if pd["kind"] != "coding":
                continue
            f.write(f"## {pd['label']}\n\n")
            for m in models:
                r = results[m["id"]].get(pid, {})
                f.write(f"### {m['name']}\n\n")
                f.write(f"ttft: {r.get('ttft', 0):.1f}s  |  gen: {r.get('tps', 0):.1f} t/s\n\n")
                f.write(f"```\n{r.get('text', '(no response)')}\n```\n\n")

    console.print(f"\nCoding responses → {out_md}\n")

    console.rule("[dim]Coding responses[/dim]")
    for pid, pd in prompts.items():
        if pd["kind"] != "coding":
            continue
        console.rule(f"[bold]{pd['label']}[/bold]", style="dim")
        for m in models:
            r = results[m["id"]].get(pid, {})
            console.print(f"\n[bold cyan]{m['name']}[/bold cyan]  "
                          f"[dim]ttft={r.get('ttft', 0):.1f}s  "
                          f"{r.get('tps', 0):.1f}t/s[/dim]\n")
            console.print(r.get("text", "[dim](no response)[/dim]"))
            console.print()


def main() -> None:
    ap = argparse.ArgumentParser(description="MoE head-to-head: context recall + coding")
    ap.add_argument(
        "--models", nargs="+", default=["gemma4_26b_moe", "qwen3_35b_moe"],
        metavar="FAMILY_ID",
        help="Registry family IDs to compare (default: gemma4_26b_moe qwen3_35b_moe)",
    )
    ap.add_argument(
        "--only", nargs="+", metavar="ID",
        help=(
            "Run only these prompt IDs. "
            "Valid: recall_32k, recall_64k, recall_128k, code_algo, code_debug, code_arch"
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Show prompt sizes without running")
    args = ap.parse_args()
    run(args.models, only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
