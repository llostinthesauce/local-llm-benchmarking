#!/usr/bin/env python3
"""
bench_tui.py - Interactive home base for the four local benchmark paths.

Backends:
  - llama.cpp raw via llama-bench
  - llama.cpp server via llama-server
  - MLX direct via mlx_lm.stream_generate
  - MLX server via mlx_lm.server
"""
from __future__ import annotations

import argparse
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _check_deps() -> None:
    import importlib.util

    missing = [
        package
        for package in ("questionary", "rich", "psutil", "requests")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install: pip install {' '.join(missing)}")
        sys.exit(1)


_check_deps()

import psutil
import questionary
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompts import PASSES as ALL_PASSES
import model_registry
import memutil

# Quality evals live outside scripts/, so the repo root has to be importable too.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from evals.registry import EVALS as EVAL_SPECS

DEFAULT_COOLDOWN = 120
DEFAULT_MEM_GUARD = 80.0
LOAD_LIMIT = 0.8


def _abort(value: Any) -> Any:
    if value is None:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise SystemExit(0)
    return value


def _cooldown(seconds: int) -> None:
    if seconds <= 0:
        return
    console.print(f"\n[dim]Cooldown {seconds}s - waiting for thermals/memory...[/dim]")
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(5)
        load1 = psutil.getloadavg()[0]
        cpus = psutil.cpu_count() or 1
        if (load1 / cpus) > LOAD_LIMIT:
            deadline += 10


def _discover_models(config_path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in model_registry.iter_models(config_path):
        if not row.get("exists", False):
            continue
        path = row["path"]
        p = Path(path)
        if row["backend"] == "llamacpp":
            size_gb = p.stat().st_size / (1024**3)
            mtp_supported = bool(row.get("mtp_supported", False))
            result.append(
                {
                    "kind": "gguf",
                    "path": path,
                    "name": p.name,
                    "size_gb": size_gb,
                    "ctx_cap": row["ctx_cap"],
                    "temperature": row.get("temperature", 0.7),
                    "top_p": row.get("top_p", 0.8),
                    "top_k": row.get("top_k", 20),
                    "family": row.get("family", "?"),
                    "family_id": row.get("family_id", "?"),
                    "quant": row.get("quant", "?"),
                    "mtp_supported": mtp_supported,
                    "architecture": row.get("architecture", "dense"),
                    "label": (
                        f"[{row.get('quant', '?')}] {row.get('family_id', '?')}"
                        f"{' [MTP]' if mtp_supported else ''} ({p.name[:30]}...) [{size_gb:.1f}GB]"
                    ),
                }
            )
        elif row["backend"] == "mlx":
            result.append(
                {
                    "kind": "mlx",
                    "mlx_repo": path,
                    "name": p.name,
                    "size_gb": 0.0,
                    "ctx_cap": row["ctx_cap"],
                    "temperature": row.get("temperature", 0.7),
                    "top_p": row.get("top_p", 0.8),
                    "top_k": row.get("top_k", 20),
                    "family": row.get("family", "?"),
                    "family_id": row.get("family_id", "?"),
                    "quant": row.get("quant", "?"),
                    "architecture": row.get("architecture", "dense"),
                    "mlx_server": row.get("mlx_server", ""),
                    "label": f"[MLX {row.get('quant', '?')}] {row.get('family_id', '?')} ({p.name})",
                }
            )
    return result


def _server_has_expected_model(expected_model: str) -> bool:
    try:
        response = requests.get("http://127.0.0.1:8080/v1/models", timeout=2)
        if response.status_code != 200:
            return False
        payload = response.json()
    except Exception:
        return False
    expected = expected_model.strip()
    for item in payload.get("data", []) + payload.get("models", []):
        if expected in {str(item.get("id", "")), str(item.get("model", "")), str(item.get("name", ""))}:
            return True
    return False


def _start_llama_server(model_path: str, expected_model: str) -> subprocess.Popen | None:
    serve_script = Path(__file__).parent / "scripts" / "serve_local.sh"
    proc = subprocess.Popen(
        ["bash", str(serve_script), model_path, "--backend", "llamacpp", "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for _ in range(120):
        if proc.poll() is not None:
            stderr_tail = proc.stderr.read().decode(errors="replace")[-500:] if proc.stderr else ""
            console.print(f"  [red]llama-server exited early (rc={proc.returncode}): {stderr_tail}[/red]")
            return None
        if _server_has_expected_model(expected_model):
            return proc
        time.sleep(1)
    proc.terminate()
    console.print(f"  [red]llama-server did not report expected model: {expected_model}[/red]")
    return None


def _wait_for_port(port: int, proc: subprocess.Popen, timeout_s: int = 600) -> bool:
    """Block until `port` can actually complete a request, or the child dies.

    Deliberately not /v1/models. llama-server answers that with 200 while the
    model is still loading and only then returns 503 "Loading model" from
    /v1/chat/completions — so /v1/models reports ready before it is. A benchmark
    that trusts it fires its first prompt into a server that has not finished
    allocating its KV cache, and at a 262K context that wedges for minutes.
    A one-token completion is the only probe that means the same thing on
    llama.cpp, mlx_lm and mlx_vlm alike.
    """
    deadline = time.time() + timeout_s
    payload = {
        "model": "readiness-probe",
        "messages": [{"role": "user", "content": "Reply with: OK"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=30
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _serve_for_eval(model: dict[str, Any]) -> tuple[subprocess.Popen | None, int, str]:
    """Launch the canonical server for a model and return (proc, port, model_id).

    Everything goes through serve_local.sh so the eval phase serves models
    exactly the way the rest of the repo does — same engine routing, same
    offline guards, same registry-pinned mlx_vlm decisions.
    """
    serve_script = Path(__file__).parent / "scripts" / "serve_local.sh"
    if model["kind"] == "gguf":
        selector, backend, port = model["path"], "llamacpp", 8080
        model_id = model["name"]
    else:
        selector, backend, port = model["mlx_repo"], "mlx", 8085
        # mlx_vlm.server requires the exact filesystem path as the model id;
        # mlx_lm.server accepts it too, so the path is correct for both.
        model_id = model["mlx_repo"]

    proc = subprocess.Popen(
        ["bash", str(serve_script), selector, "--backend", backend, "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if not _wait_for_port(port, proc):
        stderr_tail = proc.stderr.read().decode(errors="replace")[-400:] if proc.stderr else ""
        console.print(f"  [red]server for {model['name']} never came up: {stderr_tail}[/red]")
        _kill_server(proc, port)
        return None, port, model_id
    return proc, port, model_id


def _kill_server(proc: subprocess.Popen | None, port: int = 8080) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.5)
    console.print(f"  [yellow]Warning: port {port} is still occupied after stopping the server.[/yellow]")


def run_tui(results_dir: Path, config_v2: Path, preselect_model_cap: bool = False) -> None:
    mem_gb = memutil.virtual_memory().total / (1024**3)
    cpu = platform.processor() or platform.machine()
    console.print("\n\n")
    console.print(
        Panel(
            f"[bold cyan]Local LLM Benchmark Suite[/bold cyan]\n"
            f"[dim]{cpu} · {mem_gb:.0f} GB RAM · {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]",
            expand=False,
        )
    )

    models = _discover_models(config_v2)
    if not models:
        console.print(
            "[yellow]No local models found in the registry.[/yellow]\n"
            f"Create one with: python3 scripts/discover_models.py --roots ~/.lmstudio/models --write {config_v2}"
        )
        raise SystemExit(1)
    state: dict[str, Any] = {
        "backends": [],
        "sel_gguf": [],
        "sel_api": [],
        "sel_mlx": [],
        "sel_passes": [],
        "sel_evals": [],
        "eval_fetch": False,
        "eval_exec": False,
        "cooldown_s": DEFAULT_COOLDOWN,
        "mem_guard": DEFAULT_MEM_GUARD,
        "dry_run": False,
    }

    def _fmt_backends() -> str:
        labels = {
            "llamacpp": "llama.cpp raw",
            "llamacpp_api": "llama.cpp server",
            "mlx_direct": "MLX direct",
            "mlx_api": "MLX server",
            "mlx_vlm_api": "MLX-VLM server",
        }
        return ", ".join(labels[b] for b in state["backends"]) if state["backends"] else "none"

    def _fmt_models() -> str:
        total = len(state["sel_gguf"]) + len(state["sel_api"]) + len(state["sel_mlx"])
        return f"{total} variant(s)" if total else "none"

    def _fmt_passes() -> str:
        return ", ".join(p["id"] for p in state["sel_passes"]) if state["sel_passes"] else "none"

    def _fmt_evals() -> str:
        return ", ".join(state["sel_evals"]) if state["sel_evals"] else "none"

    nav_choices = [
        questionary.Choice("1. Backends - which engines to use", value="backends"),
        questionary.Choice("2. Models & Quants - which variants", value="models"),
        questionary.Choice("3. Passes - what test sizes", value="passes"),
        questionary.Choice("4. Quality evals - accuracy, not just speed", value="quality"),
        questionary.Choice("5. Settings - cooldown, memory, dry-run", value="settings"),
        questionary.Choice("6. Summary -> RUN", value="run"),
        questionary.Choice("Exit", value="exit"),
    ]

    while True:
        console.print("\n\n")
        console.rule("[bold]Navigation[/bold]")
        page = _abort(questionary.select("Go to:", choices=nav_choices).ask())
        if page == "exit":
            raise SystemExit(0)

        if page == "backends":
            prev = set(state["backends"]) or {"llamacpp"}
            choices = [
                questionary.Choice("llama.cpp raw (llama-bench)", value="llamacpp", checked="llamacpp" in prev),
                questionary.Choice("llama.cpp server (llama-server :8080)", value="llamacpp_api", checked="llamacpp_api" in prev),
                questionary.Choice("MLX direct (mlx_lm.stream_generate)", value="mlx_direct", checked="mlx_direct" in prev),
                questionary.Choice("MLX server (mlx_lm.server :8085)", value="mlx_api", checked="mlx_api" in prev),
                questionary.Choice("MLX-VLM server (mlx_vlm.server :8085)", value="mlx_vlm_api", checked="mlx_vlm_api" in prev),
            ]
            state["backends"] = _abort(questionary.checkbox("Backends:", choices=choices).ask()) or []
            console.print(f"  [green]Selected: {_fmt_backends()}[/green]")

        elif page == "models":
            if not state["backends"]:
                console.print("[yellow]Select at least one backend first.[/yellow]")
                continue

            state["sel_gguf"] = []
            state["sel_api"] = []
            state["sel_mlx"] = []

            gguf_opts = [m for m in models if m["kind"] == "gguf"]
            mlx_opts = [m for m in models if m["kind"] == "mlx"]

            if "llamacpp" in state["backends"] and gguf_opts:
                choices = [questionary.Choice(m["label"], value=m, checked=True) for m in gguf_opts]
                state["sel_gguf"] = _abort(questionary.checkbox("llama.cpp raw models:", choices=choices).ask()) or []

            if "llamacpp_api" in state["backends"] and gguf_opts:
                choices = [questionary.Choice(m["label"], value=m, checked=True) for m in gguf_opts]
                state["sel_api"] = _abort(questionary.checkbox("llama.cpp server models:", choices=choices).ask()) or []

            if any(b in state["backends"] for b in ("mlx_direct", "mlx_api", "mlx_vlm_api")) and mlx_opts:
                choices = [questionary.Choice(m["label"], value=m, checked=True) for m in mlx_opts]
                state["sel_mlx"] = _abort(questionary.checkbox("MLX models:", choices=choices).ask()) or []

            console.print(f"  [green]Models: {_fmt_models()}[/green]")

        elif page == "passes":
            choices = []
            for p in ALL_PASSES:
                checked = bool(state["sel_passes"] and p in state["sel_passes"])
                if not state["sel_passes"]:
                    checked = p.get("checked", True)
                if preselect_model_cap and p["id"] == "pass_4_max":
                    checked = True
                choices.append(questionary.Choice(p["label"], value=p, checked=checked))
            state["sel_passes"] = _abort(questionary.checkbox("Passes:", choices=choices).ask()) or []
            console.print(f"  [green]Passes: {_fmt_passes()}[/green]")

        elif page == "quality":
            choices = []
            for name, spec in sorted(EVAL_SPECS.items(), key=lambda kv: (kv[1].tier, kv[0])):
                gate = " [needs --fetch]" if spec.tier == 2 else ""
                if spec.needs_code_execution:
                    gate += " [executes model code]"
                choices.append(
                    questionary.Choice(
                        f"[tier {spec.tier}] {name} - {spec.description}{gate}",
                        value=name,
                        checked=name in state["sel_evals"],
                    )
                )
            state["sel_evals"] = _abort(questionary.checkbox("Quality evals:", choices=choices).ask()) or []

            selected = [EVAL_SPECS[name] for name in state["sel_evals"]]
            if any(spec.tier == 2 for spec in selected):
                state["eval_fetch"] = questionary.confirm(
                    "Tier-2 evals need a one-time dataset download. Allow it?",
                    default=state["eval_fetch"],
                ).ask() or False
            if any(spec.needs_code_execution for spec in selected):
                console.print(
                    "  [yellow]HumanEval scores by running code the model writes. "
                    "Mitigations are best-effort (subprocess, rlimits, timeout) and are "
                    "not a security boundary.[/yellow]"
                )
                state["eval_exec"] = questionary.confirm(
                    "Allow executing model-generated code?", default=False
                ).ask() or False
            console.print(f"  [green]Evals: {_fmt_evals()}[/green]")

        elif page == "settings":
            state["cooldown_s"] = int(
                _abort(
                    questionary.text(
                        "Cooldown between models (s):",
                        default=str(state["cooldown_s"]),
                        validate=lambda v: v.isdigit() or "Integer required",
                    ).ask()
                )
            )
            state["mem_guard"] = float(
                _abort(
                    questionary.text(
                        "Memory guard %:",
                        default=str(int(state["mem_guard"])),
                        validate=lambda v: v.replace(".", "").isdigit() or "Number required",
                    ).ask()
                )
            )
            state["dry_run"] = questionary.confirm("Dry run? (show plan, skip inference)", default=state["dry_run"]).ask() or False
            console.print(
                f"  [green]Cooldown: {state['cooldown_s']}s · "
                f"Mem guard: {state['mem_guard']:.0f}% · Dry run: {state['dry_run']}[/green]"
            )

        elif page == "run":
            total_models = len(state["sel_gguf"]) + len(state["sel_api"]) + len(state["sel_mlx"])
            total_runs = total_models * len(state["sel_passes"])
            eval_runs = total_models * len(state["sel_evals"])
            if total_runs == 0 and eval_runs == 0:
                console.print("[yellow]Nothing selected to run.[/yellow]")
                continue

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Setting", style="dim")
            table.add_column("Value", style="bold")
            table.add_row("Backends", _fmt_backends())
            table.add_row("Models", _fmt_models())
            table.add_row("Passes", _fmt_passes())
            table.add_row("Quality evals", _fmt_evals())
            table.add_row("Cooldown", f"{state['cooldown_s']}s")
            table.add_row("Mem guard", f"{state['mem_guard']:.0f}%")
            table.add_row("Dry run", "[bold yellow]Yes[/bold yellow]" if state["dry_run"] else "No")
            table.add_row("Total runs", str(total_runs))
            if eval_runs:
                table.add_row("Eval runs", f"{eval_runs} (model x eval)")
            console.print(table)

            if not (_abort(questionary.confirm("Proceed?", default=True).ask()) or False):
                console.print("[yellow]Aborted.[/yellow]")
                continue
            if state["dry_run"]:
                console.print("\n[bold yellow]Dry run complete - nothing was run.[/bold yellow]")
                continue

            results_dir.mkdir(parents=True, exist_ok=True)

            if "llamacpp" in state["backends"] and state["sel_gguf"]:
                from scripts.bench_llamacpp_raw import run_benchmark as run_llamacpp_raw

                out_dir = results_dir / "llamacpp_raw"
                console.print("\n[bold magenta]=== llama.cpp raw ===[/bold magenta]")
                for i, model in enumerate(state["sel_gguf"]):
                    console.print(f"\n [bold][{i + 1}/{len(state['sel_gguf'])}] {model['name']}[/bold]")
                    csv_path = run_llamacpp_raw(
                        model["path"],
                        state["sel_passes"],
                        out_dir,
                        ctx_cap=model["ctx_cap"],
                        model_size_gb=model.get("size_gb", 0),
                        quant=model.get("quant", "?"),
                        cooldown=state["cooldown_s"],
                        mem_guard=state["mem_guard"],
                        architecture=model.get("architecture", "dense"),
                    )
                    console.print(f" [green]CSV -> {csv_path}[/green]")
                    if i < len(state["sel_gguf"]) - 1:
                        _cooldown(state["cooldown_s"])

            if state["sel_gguf"] and state["sel_api"]:
                _cooldown(state["cooldown_s"])

            if "llamacpp_api" in state["backends"] and state["sel_api"]:
                from scripts.bench_llamacpp_api import run_benchmark as run_llamacpp_api

                out_dir = results_dir / "llamacpp_api"
                console.print("\n[bold cyan]=== llama.cpp server ===[/bold cyan]")
                for i, model in enumerate(state["sel_api"]):
                    console.print(f"\n [bold][{i + 1}/{len(state['sel_api'])}] {model['name']}[/bold]")
                    server = _start_llama_server(model["path"], model["name"])
                    if not server:
                        console.print(f" [red]Failed to start llama-server for {model['name']}[/red]")
                        continue
                    try:
                        csv_path = run_llamacpp_api(
                            model["name"],
                            state["sel_passes"],
                            out_dir,
                            ctx_cap=model["ctx_cap"],
                            family=model.get("family", "?"),
                            temperature=model.get("temperature", 0.7),
                            top_p=model.get("top_p", 0.8),
                            top_k=model.get("top_k", 20),
                            quant=model.get("quant", "?"),
                            cooldown=state["cooldown_s"],
                            mem_guard=state["mem_guard"],
                            mtp=bool(model.get("mtp_supported", False)),
                        )
                        console.print(f" [green]CSV -> {csv_path}[/green]")
                    finally:
                        _kill_server(server)
                    if i < len(state["sel_api"]) - 1:
                        _cooldown(state["cooldown_s"])

            if "mlx_direct" in state["backends"] and state["sel_mlx"]:
                from scripts.bench_mlx_direct import run_benchmark as run_mlx_direct

                out_dir = results_dir / "mlx_direct"
                console.print("\n[bold yellow]=== MLX direct ===[/bold yellow]")
                for i, model in enumerate(state["sel_mlx"]):
                    repo = model["mlx_repo"]
                    console.print(f"\n [bold][{i + 1}/{len(state['sel_mlx'])}] {repo}[/bold]")
                    csv_path = run_mlx_direct(
                        repo,
                        state["sel_passes"],
                        out_dir,
                        ctx_cap=model["ctx_cap"],
                        temperature=model.get("temperature", 0.7),
                        top_p=model.get("top_p", 0.8),
                        top_k=model.get("top_k", 20),
                        quant=model.get("quant", "?"),
                        cooldown=state["cooldown_s"],
                        mem_guard=state["mem_guard"],
                        architecture=model.get("architecture", "dense"),
                        model_size_gb=model.get("size_gb", 0),
                    )
                    console.print(f" [green]CSV -> {csv_path}[/green]")
                    if i < len(state["sel_mlx"]) - 1:
                        _cooldown(state["cooldown_s"])

            if "mlx_api" in state["backends"] and state["sel_mlx"]:
                from scripts.bench_mlx_api import run_benchmark as run_mlx_api

                out_dir = results_dir / "mlx_api"
                console.print("\n[bold cyan]=== MLX server ===[/bold cyan]")
                for i, model in enumerate(state["sel_mlx"]):
                    repo = model["mlx_repo"]
                    console.print(f"\n [bold][{i + 1}/{len(state['sel_mlx'])}] {repo}[/bold]")
                    server_type = "mlx_vlm" if model.get("mlx_server") == "mlx_vlm" else "mlx_lm"
                    csv_path = run_mlx_api(
                        repo,
                        state["sel_passes"],
                        out_dir,
                        ctx_cap=model["ctx_cap"],
                        temperature=model.get("temperature", 0.7),
                        top_p=model.get("top_p", 0.8),
                        top_k=model.get("top_k", 20),
                        quant=model.get("quant", "?"),
                        cooldown=state["cooldown_s"],
                        mem_guard=state["mem_guard"],
                        server=server_type,
                    )
                    console.print(f" [green]CSV -> {csv_path}[/green]")
                    if i < len(state["sel_mlx"]) - 1:
                        _cooldown(state["cooldown_s"])

            if "mlx_vlm_api" in state["backends"] and state["sel_mlx"]:
                from scripts.bench_mlx_api import run_benchmark as run_mlx_api

                out_dir = results_dir / "mlx_vlm_api"
                console.print("\n[bold cyan]=== MLX-VLM server ===[/bold cyan]")
                for i, model in enumerate(state["sel_mlx"]):
                    repo = model["mlx_repo"]
                    console.print(f"\n [bold][{i + 1}/{len(state['sel_mlx'])}] {repo}[/bold]")
                    csv_path = run_mlx_api(
                        repo,
                        state["sel_passes"],
                        out_dir,
                        ctx_cap=model["ctx_cap"],
                        temperature=model.get("temperature", 0.7),
                        top_p=model.get("top_p", 0.8),
                        top_k=model.get("top_k", 20),
                        quant=model.get("quant", "?"),
                        cooldown=state["cooldown_s"],
                        mem_guard=state["mem_guard"],
                        server="mlx_vlm",
                    )
                    console.print(f" [green]CSV -> {csv_path}[/green]")
                    if i < len(state["sel_mlx"]) - 1:
                        _cooldown(state["cooldown_s"])

            if state["sel_evals"]:
                from scripts.bench_quality import run_suite

                out_dir = results_dir / "quality"
                console.print("\n[bold green]=== Quality evals ===[/bold green]")
                # De-duplicate: a GGUF selected for both raw and server backends
                # is still one set of weights, and its accuracy does not depend
                # on which speed harness measured it.
                targets: list[dict[str, Any]] = []
                seen_paths: set[str] = set()
                for model in state["sel_gguf"] + state["sel_api"] + state["sel_mlx"]:
                    key = model.get("path") or model.get("mlx_repo", "")
                    if key and key not in seen_paths:
                        seen_paths.add(key)
                        targets.append(model)

                for i, model in enumerate(targets):
                    console.print(f"\n [bold][{i + 1}/{len(targets)}] {model['name']}[/bold]")
                    server, port, model_id = _serve_for_eval(model)
                    if not server:
                        console.print(f" [red]Skipping evals for {model['name']} — no server.[/red]")
                        continue
                    try:
                        csv_path = run_suite(
                            model=model_id,
                            eval_names=state["sel_evals"],
                            output_dir=out_dir,
                            api_base=f"http://127.0.0.1:{port}/v1",
                            backend="llamacpp_api" if model["kind"] == "gguf" else "mlx_api",
                            quant=model.get("quant", "?"),
                            ctx_cap=model["ctx_cap"],
                            allow_fetch=bool(state["eval_fetch"]),
                            allow_code_execution=bool(state["eval_exec"]),
                        )
                        console.print(f" [green]CSV -> {csv_path}[/green]")
                    finally:
                        _kill_server(server, port)
                    if i < len(targets) - 1:
                        _cooldown(state["cooldown_s"])

            console.print("\n[bold green]All done.[/bold green]")
            break


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Five-backend local LLM benchmark TUI")
    parser.add_argument(
        "--config-v2",
        type=Path,
        default=model_registry.DEFAULT_CONFIG,
        help="Generated local model registry",
    )
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--include-model-max-pass", action="store_true", help="Pre-select the max pass")
    args = parser.parse_args()

    try:
        run_tui(
            results_dir=args.results_dir,
            config_v2=args.config_v2,
            preselect_model_cap=args.include_model_max_pass,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
