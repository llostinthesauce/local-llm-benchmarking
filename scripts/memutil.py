#!/usr/bin/env python3
"""Memory stats with a vm_stat/sysctl fallback for the macOS 27 psutil bug.

psutil.virtual_memory() can intermittently raise
RuntimeError: host_statistics64(HOST_VM_INFO64) syscall failed: (ipc/mig) array not large enough
on macOS 27 (Tahoe), especially when called from background threads. Fall
back to vm_stat + sysctl, which go through a different syscall path.
"""
from __future__ import annotations

import re
import subprocess
from types import SimpleNamespace

import psutil


def virtual_memory() -> SimpleNamespace:
    try:
        return psutil.virtual_memory()
    except RuntimeError:
        return _vm_stat_fallback()


def _vm_stat_fallback() -> SimpleNamespace:
    total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5).strip())
    out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    page_match = re.search(r"page size of (\d+) bytes", out)
    page_size = int(page_match.group(1)) if page_match else 4096

    pages: dict[str, int] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            pages[key.strip()] = int(value)

    free = pages.get("Pages free", 0) * page_size
    active = pages.get("Pages active", 0) * page_size
    inactive = pages.get("Pages inactive", 0) * page_size
    wired = pages.get("Pages wired down", 0) * page_size
    speculative = pages.get("Pages speculative", 0) * page_size
    available = free + inactive + speculative
    used = total - available
    percent = round(used / total * 100, 1) if total else 0.0

    return SimpleNamespace(
        total=total,
        available=available,
        percent=percent,
        used=used,
        free=free,
        active=active,
        inactive=inactive,
        wired=wired,
    )
