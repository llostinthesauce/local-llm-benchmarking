#!/usr/bin/env python3
"""
Discover local GGUF and MLX models and write configs/models.local.json.

The committed catalog contains generic matching rules. This generated local
registry contains machine-specific paths and should stay out of git.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "model_catalog.json"
DEFAULT_OUTPUT = ROOT / "configs" / "models.local.json"


def _default_roots() -> list[Path]:
    env_roots = os.environ.get("BENCH_MODEL_ROOTS", "")
    if env_roots:
        return [Path(os.path.expanduser(p)) for p in env_roots.split(os.pathsep) if p]
    return [
        Path.home() / ".lmstudio" / "models",
        Path.home() / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]


def _load_catalog(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"Catalog not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")
    if data.get("_schema_version") != 1:
        raise SystemExit(f"Unsupported catalog schema in {path}: expected _schema_version=1")
    return data


def _safe_id(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "model"


def _infer_quant(path: Path) -> str:
    text = path.name
    patterns = [
        r"(Q\d+_[A-Za-z0-9_]+)",
        r"(IQ\d+_[A-Za-z0-9_]+)",
        r"\b(BF16|F16|F32)\b",
        r"\b([248]bit)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "?"


def _is_mlx_dir(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if (path / "tokenizer.json").is_file() or (path / "tokenizer_config.json").is_file():
        return True
    return any(path.glob("*.safetensors"))


def _skip_candidate(path: Path) -> bool:
    text = str(path).lower()
    skip_markers = (
        "mmproj",
        "embedding",
        "embed-",
        "-embed",
        "mtp",
    )
    return any(marker in text for marker in skip_markers)


def _scan_gguf(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found.extend(p for p in root.rglob("*.gguf") if p.is_file() and not _skip_candidate(p))
    return sorted(set(found))


def _scan_mlx(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            path = Path(current)
            if "config.json" in files and _is_mlx_dir(path) and not _skip_candidate(path):
                found.append(path)
                dirs[:] = []
    return sorted(set(found))


def _matches(path: Path, patterns: list[str]) -> bool:
    haystack = str(path)
    name = path.name
    return any(re.search(pattern, haystack, re.IGNORECASE) or re.search(pattern, name, re.IGNORECASE) for pattern in patterns)


def _family_base(family: dict[str, Any], defaults: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(
        [
            ("id", family["id"]),
            ("name", family.get("name", family["id"])),
            ("family", family.get("family", "custom")),
            ("architecture", family.get("architecture", "dense")),
            ("ctx_cap", int(family.get("ctx_cap", defaults.get("ctx_cap", 131072)))),
            ("temperature", family.get("temperature", defaults.get("temperature", 0.7))),
            ("top_p", family.get("top_p", defaults.get("top_p", 0.8))),
            ("top_k", family.get("top_k", defaults.get("top_k", 20))),
            ("gguf", []),
            ("mlx", []),
        ]
    )


def _add_unmatched_gguf(families: OrderedDict[str, OrderedDict[str, Any]], path: Path, defaults: dict[str, Any]) -> None:
    family_id = _safe_id(path.stem)
    base = _family_base(
        {
            "id": family_id,
            "name": path.stem,
            "family": "custom",
            "architecture": "dense",
        },
        defaults,
    )
    base["gguf"].append({"quant": _infer_quant(path), "path": str(path)})
    families[family_id] = base


def _add_unmatched_mlx(families: OrderedDict[str, OrderedDict[str, Any]], path: Path, defaults: dict[str, Any]) -> None:
    family_id = _safe_id(path.name)
    base = _family_base(
        {
            "id": family_id,
            "name": path.name,
            "family": "custom",
            "architecture": "dense",
        },
        defaults,
    )
    base["mlx"].append({"quant": _infer_quant(path), "repo": str(path)})
    families[family_id] = base


def discover(roots: list[Path], catalog_path: Path, include_unmatched: bool) -> dict[str, Any]:
    catalog = _load_catalog(catalog_path)
    defaults = catalog.get("defaults", {})
    families: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    matched_gguf: set[Path] = set()
    matched_mlx: set[Path] = set()
    gguf_paths = _scan_gguf(roots)
    mlx_paths = _scan_mlx(roots)

    for family in catalog.get("model_families", []):
        row = _family_base(family, defaults)
        for path in gguf_paths:
            if _matches(path, family.get("gguf_patterns", [])):
                row["gguf"].append({"quant": _infer_quant(path), "path": str(path)})
                matched_gguf.add(path)
        for path in mlx_paths:
            if _matches(path, family.get("mlx_patterns", [])):
                row["mlx"].append({"quant": _infer_quant(path), "repo": str(path)})
                matched_mlx.add(path)
        if row["gguf"] or row["mlx"]:
            families[row["id"]] = row

    if include_unmatched:
        for path in gguf_paths:
            if path not in matched_gguf:
                _add_unmatched_gguf(families, path, defaults)
        for path in mlx_paths:
            if path not in matched_mlx:
                _add_unmatched_mlx(families, path, defaults)

    return OrderedDict(
        [
            (
                "_comment",
                "Generated by scripts/discover_models.py. Machine-specific paths; do not commit.",
            ),
            ("_schema_version", 2),
            ("model_families", list(families.values())),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover local GGUF and MLX models")
    parser.add_argument("--roots", nargs="+", type=Path, default=_default_roots(), help="Model roots to scan")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--write", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--known-only", action="store_true", help="Skip unmatched custom models")
    parser.add_argument("--print", action="store_true", help="Print JSON to stdout instead of writing")
    args = parser.parse_args()

    roots = [Path(os.path.expanduser(str(p))).resolve() for p in args.roots]
    payload = discover(roots, args.catalog, include_unmatched=not args.known_only)
    text = json.dumps(payload, indent=2)

    if args.print:
        print(text)
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n")
        total = sum(len(f.get("gguf", [])) + len(f.get("mlx", [])) for f in payload["model_families"])
        print(f"Wrote {args.write} ({total} model variant(s))")


if __name__ == "__main__":
    main()
