#!/usr/bin/env python3
"""
llm — unified local LLM CLI

  llm serve  [selector] [--backend] [--port] [--host] [--dry-run]
  llm bench  [--results-dir] [--max-pass]
  llm smoke  [--dry-run]
  llm list   [--backend]
  llm status
  llm web    [--host] [--port] [--no-open]

Delegates to existing scripts — nothing is re-implemented here.
  serve  → scripts/serve_local.sh  (os.execv; signal goes straight to server)
  menu   → scripts/llama_serve_menu.py  (interactive model+backend picker)
  bench  → bench_tui.py            (interactive five-backend TUI)
  smoke  → scripts/smoke_test.py
  list   → scripts/model_registry.py   (imported directly)
  status → HTTP health probes on inference ports
  web    → webgui/serve.py          (browser GUI over serve/status/chat)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import model_registry  # noqa: E402  (after sys.path insert)

SERVE_SH   = SCRIPTS / "serve_local.sh"
MENU_PY    = SCRIPTS / "llama_serve_menu.py"
BENCH_TUI  = ROOT / "bench_tui.py"
SMOKE_PY   = SCRIPTS / "smoke_test.py"
WEB_PY     = ROOT / "webgui" / "serve.py"

INFERENCE_PORTS = {
    8080: "llama.cpp",
    8085: "MLX  (mlx_lm / mlx_vlm)",
    8123: "oMLX (always-on)",
}

BACKENDS = ["llamacpp", "mlx", "mlx-kv", "mlx-vlm", "omlx"]


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python3"
    return str(venv) if venv.is_file() else sys.executable


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _probe(port: int) -> tuple[bool, int | None]:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
                return True, r.status
        except Exception:
            pass
    return False, None


def cmd_status(_args: argparse.Namespace) -> None:
    print("Inference port status")
    print("─" * 44)
    for port, label in INFERENCE_PORTS.items():
        up, code = _probe(port)
        state = f"UP  (HTTP {code})" if up else "down"
        print(f"  :{port}  {label:<26}  {state}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    backend_filter = getattr(args, "backend", "auto")
    rows = model_registry.iter_models(model_registry.DEFAULT_CONFIG)
    if backend_filter != "auto":
        rows = [r for r in rows if r["backend"] == backend_filter]

    print(f"{'BACKEND':<9} {'FAMILY':<22} {'QUANT':<8} {'ON-DISK':<8} ALIASES / PATH")
    print("─" * 100)
    for r in rows:
        aliases = ", ".join(r["aliases"])
        exists  = "yes" if r["exists"] else "MISSING"
        print(f"{r['backend']:<9} {r['family_id']:<22} {r['quant']:<8} {exists:<8} {aliases}")
        print(f"{'':>9} {'':>22} {'':>8} {'':>8} {r['path']}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def _auto_backend(selector: str) -> str | None:
    """Return the first on-disk backend for selector, or None."""
    for backend in ("llamacpp", "mlx"):
        try:
            row = model_registry.resolve(selector, backend, model_registry.DEFAULT_CONFIG)
            if row.get("exists"):
                return backend
        except SystemExit:
            pass
    return None


def cmd_serve(args: argparse.Namespace) -> None:
    selector: str | None = args.selector
    backend:  str | None = args.backend

    # No selector → hand off to the interactive menu (it handles model + backend pick).
    if not selector:
        cmd = [sys.executable, str(MENU_PY)]
        if backend:
            cmd += ["--backend", backend]
        if args.port:
            cmd += ["--port", args.port]
        if args.host:
            cmd += ["--host", args.host]
        if args.dry_run:
            cmd.append("--dry-run")
        os.execv(sys.executable, cmd)  # replaces this process

    # Selector given without backend → auto-detect.
    if not backend:
        backend = _auto_backend(selector)
        if backend is None:
            raise SystemExit(
                f"No on-disk model found for '{selector}'. "
                f"Run `llm list` to see what is available."
            )

    # omlx is menu-only (just points at an already-running server).
    if backend == "omlx":
        cmd = [sys.executable, str(MENU_PY), selector, "--backend", "omlx"]
        if args.dry_run:
            cmd.append("--dry-run")
        os.execv(sys.executable, cmd)

    # Direct launch via serve_local.sh — os.execv so signals reach the server.
    cmd = [str(SERVE_SH), selector, "--backend", backend]
    if args.port:
        cmd.append(args.port)
    if args.host:
        cmd += ["--host", args.host]
    if args.dry_run:
        cmd.append("--dry-run")
    os.execv(str(SERVE_SH), cmd)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    cmd = [_python(), str(BENCH_TUI)]
    if args.results_dir:
        cmd += ["--results-dir", str(args.results_dir)]
    if args.max_pass:
        cmd.append("--include-model-max-pass")
    subprocess.run(cmd, cwd=ROOT, check=False)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args: argparse.Namespace) -> None:
    cmd = [_python(), str(SMOKE_PY)]
    if args.dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=ROOT, check=False)


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

def cmd_web(args: argparse.Namespace) -> None:
    cmd = [_python(), str(WEB_PY)]
    if args.host:
        cmd += ["--host", args.host]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.idle_timeout is not None:
        cmd += ["--idle-timeout", str(args.idle_timeout)]
    if args.no_open:
        cmd.append("--no-open")
    os.execv(cmd[0], cmd)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="llm",
        description="Unified local LLM CLI — serve, bench, smoke, list, status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  llm serve                          # interactive model + backend picker
  llm serve gemma26                  # auto-detect backend
  llm serve gemma26 --backend mlx-kv # mlx_vlm + KV q8 turboquant
  llm serve gemma26 --dry-run
  llm bench
  llm smoke
  llm list
  llm list --backend mlx
  llm status
  llm web                            # browser GUI at http://127.0.0.1:7860
""",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # -- serve ----------------------------------------------------------------
    sp = sub.add_parser("serve", help="Launch a model server")
    sp.add_argument("selector", nargs="?",
                    help="Model alias, family id, or path (omit for interactive picker)")
    sp.add_argument("--backend", choices=BACKENDS,
                    help="Serving backend (auto-detected when omitted)")
    sp.add_argument("--port",  help="Port override")
    sp.add_argument("--host",  help="Host override (default: 127.0.0.1)")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_serve)

    # -- bench ----------------------------------------------------------------
    bp = sub.add_parser("bench", help="Run the interactive benchmark TUI")
    bp.add_argument("--results-dir", type=Path,
                    help="Where to write CSV results (default: results/)")
    bp.add_argument("--max-pass", action="store_true",
                    help="Pre-select the model-max context pass")
    bp.set_defaults(func=cmd_bench)

    # -- smoke ----------------------------------------------------------------
    kp = sub.add_parser("smoke", help="Run the benchmark pipeline smoke test")
    kp.add_argument("--dry-run", action="store_true",
                    help="Skip Python compilation check")
    kp.set_defaults(func=cmd_smoke)

    # -- list -----------------------------------------------------------------
    lp = sub.add_parser("list", help="List models from the registry")
    lp.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="auto")
    lp.set_defaults(func=cmd_list)

    # -- status ---------------------------------------------------------------
    tp = sub.add_parser("status", help="Check inference port health")
    tp.set_defaults(func=cmd_status)

    # -- web ------------------------------------------------------------------
    wp = sub.add_parser("web", help="Launch the browser GUI (serve/status/chat)")
    wp.add_argument("--host", help="Bind host (default: 127.0.0.1)")
    wp.add_argument("--port", type=int, help="Bind port (default: 7860)")
    wp.add_argument("--idle-timeout", type=int,
                    help="Seconds idle before auto-unload + shutdown (0 disables; default 300)")
    wp.add_argument("--no-open", action="store_true", help="Do not open a browser")
    wp.set_defaults(func=cmd_web)

    args = ap.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
