#!/usr/bin/env python3
"""
Opt-in dataset downloader for the tier-2 evals.

The rest of this repo is aggressively offline — MLX servers launch with
`HF_HUB_OFFLINE=1` specifically so a benchmark can never silently pull weights
mid-run. Tier-2 evals need public datasets, so they follow the same rule from the
other direction: nothing downloads unless the caller passes `--fetch`, and every
tier-2 eval fails with a clear message rather than reaching for the network on
its own.

Downloads land in `evals/.cache/` (git-ignored) and are recorded in a manifest
with their SHA-256. Nothing is pinned to a hash shipped in this repo — that would
be a hash nobody can independently verify. Instead the first download records
what it saw, and any later change to a supposedly-immutable file is surfaced
loudly, which is the property that actually matters for reproducibility.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, List

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
MANIFEST = CACHE_DIR / "manifest.json"

USER_AGENT = "local-llm-benchmarking/1.0 (+https://github.com/llostinthesauce/local-llm-benchmarking)"


class FetchDisabled(RuntimeError):
    """Raised when a tier-2 eval needs data that has not been downloaded."""


def _load_manifest() -> Dict[str, str]:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def _save_manifest(manifest: Dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure(name: str, url: str, allow_fetch: bool) -> Path:
    """Return the local path for `name`, downloading from `url` if permitted.

    Raises FetchDisabled when the file is absent and `allow_fetch` is False, so
    an offline run fails fast with an actionable message instead of hanging on a
    socket timeout.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / name
    if target.is_file():
        return target

    if not allow_fetch:
        raise FetchDisabled(
            f"{name} is not downloaded. Tier-2 evals need a one-time fetch:\n"
            f"    python3 scripts/bench_quality.py --fetch ...\n"
            f"Source: {url}"
        )

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()

    if url.endswith(".gz"):
        payload = gzip.decompress(payload)

    digest = _sha256(payload)
    manifest = _load_manifest()
    previous = manifest.get(name)
    if previous and previous != digest:
        print(
            f"  WARNING: {name} changed upstream since the last fetch.\n"
            f"           was {previous}\n"
            f"           now {digest}\n"
            f"           Scores are not comparable across this change."
        )
    manifest[name] = digest
    manifest[f"{name}::url"] = url
    _save_manifest(manifest)

    target.write_bytes(payload)
    print(f"  fetched {name} ({len(payload) / 1024:.0f} KB, sha256 {digest[:16]}...)")
    return target


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def fetch_json(url: str, allow_fetch: bool, cache_name: str) -> dict:
    """Fetch and cache a JSON API response (used for the HF datasets-server)."""
    path = ensure(cache_name, url, allow_fetch)
    return json.loads(path.read_text(encoding="utf-8"))
