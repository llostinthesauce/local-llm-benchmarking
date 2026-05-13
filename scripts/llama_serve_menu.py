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
        families[family_id]["backends"].add(row["backend"])
    result = []
    for item in families.values():
        item["backends"] = sorted(item["backends"])
        result.append(item)
    return result


def _display_label(row: dict) -> str:
    backends = ", ".join(b for b in row["backends"] if b in ("llamacpp", "mlx"))
    return f"{row['family_id']} [{backends}]"


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

    if not backend:
        host_label = args.host or "127.0.0.1"
        backend_choices = [{"backend": "llamacpp", "label": f"llama.cpp API on {host_label}:8080"}]
        try:
            model_registry.resolve(selector, "mlx", model_registry.DEFAULT_CONFIG)
            backend_choices.append({"backend": "mlx", "label": f"MLX API on {host_label}:8085"})
        except SystemExit:
            pass
        if args.selector:
            backend = "llamacpp"
        else:
            selected_backend = _choose(backend_choices, "Serving backend", lambda item: item["label"])
            backend = selected_backend["backend"]

    _launch(selector, backend, args.port, args.host, args.dry_run, not args.no_kill)


if __name__ == "__main__":
    main()
