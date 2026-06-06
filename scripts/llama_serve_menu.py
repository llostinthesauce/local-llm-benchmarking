#!/usr/bin/env python3
"""
Interactive and CLI launcher for local model serving.

This is the repo-owned replacement for hardcoded global launcher menus. Global
entrypoints should call this file, and this file resolves models from
configs/models.local.json via scripts/model_registry.py.
"""
from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import termios
import tty
from pathlib import Path

import model_registry


ROOT = Path(__file__).resolve().parents[1]
SERVE_SCRIPT = ROOT / "scripts" / "serve_local.sh"


def _rows() -> list[dict]:
    rows = model_registry.iter_models(model_registry.DEFAULT_CONFIG)
    families: dict[str, dict] = {}
    for row in rows:
        family_id = row["family_id"]
        if family_id not in families:
            aliases = row.get("aliases", [])
            label = aliases[0] if aliases else family_id
            families[family_id] = {
                "family_id": family_id,
                "label": label,
                "aliases": aliases,
                "backends": set(),
                "quant": row.get("quant", "?"),
            }
        # Only surface a backend if its model is actually on disk. A registry row
        # whose file is missing must not be offered — selecting it would either
        # error or (for MLX repo-ids) trigger a download.
        if row.get("exists"):
            families[family_id]["backends"].add(row["backend"])
    result = []
    for item in families.values():
        item["backends"] = sorted(item["backends"])
        result.append(item)
    return result


def _display_label(row: dict) -> str:
    backends = ", ".join(b for b in row["backends"] if b in ("llamacpp", "mlx"))
    return f"{row['family_id']} [{backends or 'unavailable'}]"


def _supported_backends(selector: str) -> list[str]:
    """Backends whose model for `selector` resolves AND exists on disk."""
    found = []
    for backend in ("llamacpp", "mlx"):
        try:
            row = model_registry.resolve(selector, backend, model_registry.DEFAULT_CONFIG)
        except SystemExit:
            continue
        if row.get("exists"):
            found.append(backend)
    return found


def _choose(items: list[dict], title: str, label_fn) -> dict:
    if not sys.stdin.isatty():
        raise SystemExit("Interactive selection requires a TTY. Pass a selector argument instead.")
    selected = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def render() -> None:
        sys.stdout.write("\x1b[2J\x1b[H")
        print(title)
        print("=" * len(title))
        for index, item in enumerate(items):
            marker = ">" if index == selected else " "
            print(f"{marker} {label_fn(item)}")
        print("\nUse up/down arrows, Enter to select, Ctrl+C to cancel.")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        render()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [])
            if not ready:
                continue
            char = os.read(fd, 3)
            if char in (b"\x1b[A", b"\x1bOA"):
                selected = max(0, selected - 1)
                render()
            elif char in (b"\x1b[B", b"\x1bOB"):
                selected = min(len(items) - 1, selected + 1)
                render()
            elif char in (b"\n", b"\r"):
                return items[selected]
            elif char == b"\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()


def _kill_port(port: str, dry_run: bool) -> None:
    try:
        output = subprocess.check_output(
            ["lsof", "-i", f"tcp:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    pids = [pid for pid in output.splitlines() if pid.strip()]
    if not pids:
        return
    if dry_run:
        print(f"DRY RUN: would kill port {port} pids: {', '.join(pids)}")
        return
    for pid in pids:
        subprocess.run(["kill", pid], check=False)


def _launch(selector: str, backend: str, port: str | None, host: str | None,
            dry_run: bool, kill_port: bool) -> None:
    cmd = [str(SERVE_SCRIPT), selector, "--backend", backend]
    if port:
        cmd.append(port)
    if host:
        cmd += ["--host", host]
    if dry_run:
        cmd.append("--dry-run")

    effective_port = port or ("8085" if backend == "mlx" else "8080")
    if kill_port:
        _kill_port(effective_port, dry_run)

    if dry_run:
        print("DRY RUN launcher:", " ".join(cmd))
        sys.stdout.flush()
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    os.execv(str(SERVE_SCRIPT), cmd)


def _print_list() -> None:
    for row in _rows():
        print(_display_label(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a local model API server")
    parser.add_argument("selector", nargs="?", help="Model alias, family id, path, or repo")
    parser.add_argument("--backend", choices=["llamacpp", "mlx"], help="Serving backend")
    parser.add_argument("--port", help="Port override")
    parser.add_argument("--host", help="Host override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-kill", action="store_true", help="Do not kill existing listener on the target port")
    parser.add_argument("--list", action="store_true", help="List serving families")
    args = parser.parse_args()

    if args.list:
        _print_list()
        return

    selector = args.selector
    backend = args.backend

    if not selector:
        family = _choose(_rows(), "Local model server", _display_label)
        selector = family["family_id"]
        supported = list(family["backends"])
    else:
        supported = _supported_backends(selector)

    if not backend:
        host_label = args.host or "127.0.0.1"
        labels = {
            "llamacpp": f"llama.cpp API on {host_label}:8080",
            "mlx": f"MLX API on {host_label}:8085",
        }
        # Only offer backends this model actually has on disk — a one-backend
        # model skips the prompt entirely instead of falsely showing both.
        backend_choices = [
            {"backend": b, "label": labels[b]} for b in ("llamacpp", "mlx") if b in supported
        ]
        if not backend_choices:
            raise SystemExit(
                f"No serving backend is available on disk for '{selector}'. "
                f"Check 'python3 scripts/model_registry.py list'."
            )
        if len(backend_choices) == 1:
            backend = backend_choices[0]["backend"]
            print(f"Backend: {backend} (only on-disk option for {selector})")
        elif args.selector:
            backend = backend_choices[0]["backend"]
        else:
            selected_backend = _choose(backend_choices, "Serving backend", lambda item: item["label"])
            backend = selected_backend["backend"]

    _launch(selector, backend, args.port, args.host, args.dry_run, not args.no_kill)


if __name__ == "__main__":
    main()
