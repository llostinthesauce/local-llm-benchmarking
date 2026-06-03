#!/usr/bin/env python3
"""
Resolve local model aliases from configs/models.local.json.

This keeps serving scripts and benchmark scripts pointed at the same registry.
It intentionally uses only the Python standard library so it can run before a
project virtualenv is activated.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "models.local.json"
DEFAULT_CATALOG = ROOT / "configs" / "model_catalog.json"

LEGACY_ALIASES = {
    "qwen35": "qwen3_35b_moe",
    "qwen-moe-35b": "qwen3_35b_moe",
    "qwen35-mtp": "qwen3_35b_moe_mtp",
    "qwen35mtp": "qwen3_35b_moe_mtp",
    "qwen-moe-35b-mtp": "qwen3_35b_moe_mtp",
    "qwen27": "qwen3_27b_dense",
    "qwen-dense-27b": "qwen3_27b_dense",
    "gemma26": "gemma4_26b_moe",
    "gemma-26b": "gemma4_26b_moe",
    "gemma31": "gemma4_31b_dense",
    "gemma-31b": "gemma4_31b_dense",
    "gemmae4b": "gemma4_e4b_dense",
    "gemma-e4b": "gemma4_e4b_dense",
    "gemma4e4b": "gemma4_e4b_dense",
    "gemma4-e4b": "gemma4_e4b_dense",
    "granite-htiny": "granite_tiny",
    "granite-tiny": "granite_tiny",
    "qwen35-uncensored": "qwen3_35b_moe_uncensored",
    "qwen35uncensored": "qwen3_35b_moe_uncensored",
    "qwen35-nvfp4": "qwen3_35b_moe_nvfp4",
    "qwen35nvfp4": "qwen3_35b_moe_nvfp4",
}


def _load_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"Local model registry not found: {path}\n"
            f"Create it with:\n"
            f"  python3 scripts/discover_models.py --roots ~/.lmstudio/models --write {path}"
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")
    if data.get("_schema_version") != 2:
        raise SystemExit(f"Unsupported config schema in {path}: expected _schema_version=2")
    return data


FORBIDDEN_TEMPLATES = frozenset({"chatml"})


def _chat_template(family: str) -> str:
    if family == "gemma4":
        return "gemma2"
    if family == "qwen3":
        return ""
    return ""


def _validate_chat_template(family: str, template: str) -> None:
    if template and template.lower() in FORBIDDEN_TEMPLATES:
        raise SystemExit(
            f"Refusing to serve {family} model with --chat-template {template}. "
            f"Qwen/Granite GGUF models carry embedded templates; "
            f"forcing {template} corrupts tool calling."
        )


def _expand(path_or_repo: str) -> str:
    if path_or_repo.startswith("tools/") or path_or_repo.startswith("scripts/"):
        return str(ROOT / path_or_repo)
    return os.path.expanduser(path_or_repo)


def _family_aliases(family: dict[str, Any]) -> set[str]:
    family_id = family.get("id", "")
    aliases = {family_id}
    for alias, target in LEGACY_ALIASES.items():
        if target == family_id:
            aliases.add(alias)
    return aliases


def iter_models(config: Path) -> list[dict[str, Any]]:
    data = _load_config(config)
    rows: list[dict[str, Any]] = []
    for family in data.get("model_families", []):
        family_id = family.get("id", "")
        family_name = family.get("family", "")
        aliases = sorted(_family_aliases(family))
        chat_tpl = _chat_template(family_name)
        base = {
            "family_id": family_id,
            "family": family_name,
            "architecture": family.get("architecture", "dense"),
            "ctx_cap": int(family.get("ctx_cap", 131072)),
            "temperature": family.get("temperature", 0.7),
            "top_p": family.get("top_p", 0.8),
            "top_k": family.get("top_k", 20),
            "aliases": aliases,
            "chat_template": chat_tpl,
        }
        for entry in family.get("gguf", []):
            path = _expand(entry.get("path", ""))
            if not path:
                continue
            rows.append({
                **base,
                "backend": "llamacpp",
                "selector": path,
                "path": path,
                "name": Path(path).name,
                "quant": entry.get("quant", "?"),
                "exists": Path(path).exists(),
                "note": entry.get("_note", ""),
                "mtp_supported": entry.get("mtp_supported", False),
                "server_binary": _expand(entry.get("server_binary", "")),
                "mmproj_path": _expand(entry.get("mmproj_path", "")),
                "spec_type": entry.get("spec_type", ""),
                "spec_draft_n_max": entry.get("spec_draft_n_max", ""),
            })
        for entry in family.get("mlx", []):
            repo = entry.get("repo", "")
            if not repo:
                continue
            repo_path = _expand(repo)
            rows.append({
                **base,
                "backend": "mlx",
                "selector": repo,
                "path": repo_path,
                "name": repo_path.rstrip("/").split("/")[-1],
                "quant": entry.get("quant", "?"),
                "exists": Path(repo_path).exists() or Path(repo_path).is_symlink(),
                "note": entry.get("_note", ""),
                "mtp_supported": entry.get("mtp_supported", False),
                "draft_model": entry.get("draft_model", ""),
            })
    return rows


def resolve(selector: str, backend: str, config: Path) -> dict[str, Any]:
    selector_expanded = _expand(selector)
    selector_norm = selector.lower()
    target_family = LEGACY_ALIASES.get(selector_norm, selector_norm)
    candidates = [m for m in iter_models(config) if backend == "auto" or m["backend"] == backend]

    for model in candidates:
        if selector_expanded == model["path"] or selector == model["selector"]:
            return model
        if selector_norm == model["name"].lower():
            return model
        if selector_norm in {a.lower() for a in model["aliases"]}:
            return model
        if target_family == model["family_id"].lower():
            return model

    valid = ", ".join(sorted({a for m in candidates for a in m["aliases"]}))
    raise SystemExit(f"No {backend} model found for selector '{selector}'. Valid aliases: {valid}")


def _print_shell(model: dict[str, Any]) -> None:
    for key in (
        "backend",
        "path",
        "name",
        "family",
        "family_id",
        "ctx_cap",
        "quant",
        "chat_template",
        "temperature",
        "top_p",
        "top_k",
        "mtp_supported",
        "server_binary",
        "mmproj_path",
        "spec_type",
        "spec_draft_n_max",
    ):
        value = str(model.get(key, ""))
        print(f"MODEL_{key.upper()}={shlex.quote(value)}")


def cmd_list(args: argparse.Namespace) -> None:
    rows = [m for m in iter_models(args.config) if args.backend == "auto" or m["backend"] == args.backend]
    for model in rows:
        aliases = ",".join(model["aliases"])
        exists = "ok" if model["exists"] else "missing"
        print(
            f"{model['backend']:<9} {model['family_id']:<18} "
            f"{model['quant']:<8} {exists:<7} {aliases:<32} {model['path']}"
        )


def cmd_resolve(args: argparse.Namespace) -> None:
    model = resolve(args.selector, args.backend, args.config)
    if args.format == "json":
        print(json.dumps(model, indent=2, sort_keys=True))
    elif args.format == "shell":
        _print_shell(model)
    else:
        raise SystemExit(f"Unknown format: {args.format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve local model registry aliases")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List models")
    list_parser.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="auto")
    list_parser.set_defaults(func=cmd_list)

    resolve_parser = sub.add_parser("resolve", help="Resolve a selector")
    resolve_parser.add_argument("selector")
    resolve_parser.add_argument("--backend", choices=["auto", "llamacpp", "mlx"], default="llamacpp")
    resolve_parser.add_argument("--format", choices=["json", "shell"], default="json")
    resolve_parser.set_defaults(func=cmd_resolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
