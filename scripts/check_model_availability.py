#!/usr/bin/env python3
"""
Check local model registry availability without downloading anything.

GGUF rows are verified on disk. MLX rows are reported as configured repos/local
paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_registry


def _status_for_model(row: Dict[str, Any]) -> str:
    backend = row["backend"]
    path = row["path"]
    if backend == "llamacpp":
        return "ok" if Path(path).exists() else "missing"
    if backend == "mlx":
        expanded = os.path.expanduser(path)
        if Path(expanded).exists():
            return "ok_local"
        return "repo_configured"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check model registry availability")
    parser.add_argument("--config", type=Path, default=model_registry.DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit JSON rows")
    parser.add_argument("--fail-missing", action="store_true")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for row in model_registry.iter_models(args.config):
        if args.backend != "auto" and row["backend"] != args.backend:
            continue
        status = _status_for_model(row)
        rows.append({
            "backend": row["backend"],
            "family_id": row["family_id"],
            "name": row["name"],
            "quant": row["quant"],
            "status": status,
            "path": row["path"],
            "note": row.get("note", ""),
        })

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(
                f"{row['backend']:<9} {row['family_id']:<18} {row['quant']:<8} "
                f"{row['status']:<15} {row['path']}"
            )

    if args.fail_missing and any(row["status"] == "missing" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
