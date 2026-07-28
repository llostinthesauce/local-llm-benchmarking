#!/usr/bin/env python3
"""
Smoke test: static validation of the benchmark pipeline.
Run from repo root:
    python3 scripts/smoke_test.py
    python3 scripts/smoke_test.py --dry-run  # skip compile

Validates static/dry-run integrity only. Does NOT prove real benchmark correctness.
"""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
CONFIG_DIR = ROOT / "configs"

EXIT = 0
SKIP_DIRS = {"__pycache__", ".git", ".venv", "tools", "archive", ".remember", ".claude", "results", ".antigravitycli"}


def fail(msg: str) -> None:
    global EXIT
    print(f"  FAIL: {msg}")
    EXIT = 1


def ok(msg: str) -> None:
    print(f"  OK: {msg}")


def active_files(pattern: str):
    for path in sorted(ROOT.rglob(pattern)):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        yield path


def _git(*args: str) -> list[str]:
    """Run a git command in ROOT, returning stdout lines. Empty list if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines() if line]


def tracked_files() -> list[Path]:
    """Files git actually publishes. This — not a hand-maintained denylist — is
    the definition of 'public', so hygiene checks cannot drift out of date when
    a new private directory is added."""
    return [ROOT / rel for rel in _git("ls-files")]


def untracked_unignored_files() -> list[Path]:
    """Files that are neither tracked nor ignored: one `git add -A` from being public."""
    return [ROOT / rel for rel in _git("ls-files", "--others", "--exclude-standard")]


def repo_python_files() -> list[Path]:
    """This project's own .py files — tracked plus not-yet-committed, but never
    the git-ignored personal directories that happen to sit in the working tree."""
    candidates = tracked_files() + untracked_unignored_files()
    if not candidates:
        candidates = list(active_files("*.py"))
    return sorted(
        {p for p in candidates if p.suffix == ".py" and p.is_file() and "__pycache__" not in str(p)}
    )


def check_scripts_exist() -> None:
    print("\n--- Referenced scripts exist ---")
    referenced = set()
    for sh_file in sorted(SCRIPTS_DIR.glob("*.sh")):
        text = sh_file.read_text()
        for m in re.finditer(r'scripts/([\w_]+\.(?:py|sh))', text):
            referenced.add(m.group(1))
    for py_file in repo_python_files():
        text = py_file.read_text()
        for m in re.finditer(r'scripts/([\w_]+\.py)', text):
            referenced.add(m.group(1))
    for script in sorted(referenced):
        path = SCRIPTS_DIR / script
        if path.exists():
            ok(f"scripts/{script}")
        else:
            fail(f"scripts/{script} MISSING")


def check_python_compile() -> None:
    print("\n--- Python files compile ---")
    for py_file in repo_python_files():
        if py_file.name == "smoke_test.py":
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
            ok(f"{py_file.relative_to(ROOT)}")
        except py_compile.PyCompileError as exc:
            fail(f"{py_file.relative_to(ROOT)}: {exc}")


def check_prompts_load() -> None:
    print("\n--- Canonical prompts load ---")
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from prompts import PROMPTS, PASSES
        required = ["micro", "normal", "high", "max"]
        for key in required:
            if key not in PROMPTS:
                fail(f"PROMPTS missing key: {key}")
                continue
            if not isinstance(PROMPTS[key], str) or len(PROMPTS[key]) < 10:
                fail(f"PROMPTS[{key}] too short or not string")
                continue
            ok(f"PROMPTS[{key}]: {len(PROMPTS[key])} chars")
        pass_ids = {p["id"] for p in PASSES}
        expected = {"pass_1_micro", "pass_2_normal", "pass_3_high", "pass_4_max"}
        if pass_ids != expected:
            fail(f"PASSES IDs mismatch: {pass_ids} != {expected}")
        else:
            ok(f"PASSES: {len(PASSES)} entries")
        max_pass = next((p for p in PASSES if p["id"] == "pass_4_max"), None)
        if max_pass and max_pass.get("fill_context") is True and max_pass.get("ctx") == -1:
            ok("pass_4_max fills near ctx cap")
        else:
            fail("pass_4_max must be a real long-context fill pass")
        from prompts import build_prompt_for_pass, context_budget_for_pass
        ctx_used = context_budget_for_pass(max_pass, 262144) if max_pass else 0
        prompt, target = build_prompt_for_pass(max_pass, ctx_used) if max_pass else ("", 0)
        if target >= 245000 and len(prompt.split()) >= 245000:
            ok(f"max prompt builder targets {target} tokens")
        else:
            fail(f"max prompt builder too small: target={target}, words={len(prompt.split())}")
    except Exception as exc:
        fail(f"Failed to load prompts: {exc}")


def check_backend_runners() -> None:
    print("\n--- Backend-to-runner mapping ---")
    backends = {
        "LLAMACPP_RAW": SCRIPTS_DIR / "bench_llamacpp_raw.py",
        "LLAMACPP_API": SCRIPTS_DIR / "bench_llamacpp_api.py",
        "MLX_DIRECT": SCRIPTS_DIR / "bench_mlx_direct.py",
        "MLX_API": SCRIPTS_DIR / "bench_mlx_api.py",
        "MLX_VLM_API": SCRIPTS_DIR / "bench_mlx_api.py",
    }
    for backend, runner in sorted(backends.items()):
        if runner.exists():
            ok(f"{backend} -> {runner.name}")
        else:
            fail(f"{backend} -> {runner.name} MISSING")


def check_csv_columns() -> None:
    print("\n--- CSV column consistency ---")
    expected_fields = [
        "timestamp", "run_id", "model_name", "backend", "pass_name",
        "ctx_cap", "ctx_used", "prompt_tokens", "gen_tokens",
        "prompt_tps", "gen_tps", "ttft_s", "peak_mem_pct", "status",
        "quant", "concurrency", "mtp", "draft_tokens", "draft_accepted_tokens",
        "draft_accept_rate", "token_count_method", "generated_text",
    ]
    expected_set = set(expected_fields)
    # These are not pass-based throughput benches, so they carry their own CSV
    # schemas on purpose: config_compare A/Bs one model across two server
    # configs, head_to_head pits two models against each other, and quality
    # records per-case accuracy rather than tokens/sec.
    schema_exempt = {"bench_config_compare.py", "bench_head_to_head.py", "bench_quality.py"}
    for bench_file in sorted(SCRIPTS_DIR.glob("bench_*.py")):
        if bench_file.name in schema_exempt:
            continue
        text = bench_file.read_text()
        m = re.search(r'FIELDS\s*=\s*\[(.*?)\]', text, re.DOTALL)
        if not m:
            continue
        fields_str = m.group(1)
        fields = [f.strip().strip('"\'') for f in re.findall(r'"([^"]*)"', fields_str)]
        field_set = set(fields)
        missing = expected_set - field_set
        extra = field_set - expected_set
        if missing:
            fail(f"{bench_file.name}: missing columns {missing}")
        elif extra:
            ok(f"{bench_file.name}: has extra {extra}")
        else:
            ok(f"{bench_file.name}: columns match")


def check_registry_architecture() -> None:
    print("\n--- Registry architecture fields ---")
    catalog_path = CONFIG_DIR / "model_catalog.json"
    if not catalog_path.exists():
        fail("model_catalog.json missing")
        return
    data = json.loads(catalog_path.read_text())
    for fam in data.get("model_families", []):
        fid = fam.get("id", "?")
        arch = fam.get("architecture", "")
        if arch in ("dense", "moe"):
            ok(f"{fid}: architecture={arch}")
        else:
            fail(f"{fid}: architecture={arch!r} (expected dense|moe)")
        if fam.get("gguf_patterns") or fam.get("mlx_patterns"):
            ok(f"{fid}: discovery patterns present")
        else:
            fail(f"{fid}: missing discovery patterns")


def check_removed_backend_strings() -> None:
    print("\n--- Dead backend strings (active source only) ---")
    dead_patterns = [
        (r'LLAMA_BENCHY_LLAMACPP', "Llama Benchy backend string"),
        (r'LLAMA_BENCHY_MLX', "Llama Benchy backend string"),
        (r'llama_benchy', "llama_benchy reference"),
        (r'scripts/bench_llamacpp_batched\.py', "batched script reference"),
        (r'scripts/bench_llamacpp_parallel\.py', "parallel script reference"),
        (r'scripts/bench_llamacpp_perplexity\.py', "perplexity script reference"),
        (r'scripts/bench_llamacpp_speculative\.py', "speculative script reference"),
        (r'run_tool_eval', "tool eval reference"),
        (r'profile_mlx', "old MLX profiler reference"),
        (r'generate_opencode_config', "OpenCode generator reference"),
        # The removed feature was a cloud LLM-judge scorer. Match the identifiers
        # it would reintroduce, not the word "judge" in prose — the evals/ suite
        # documents at length that it deliberately has no judge, and a bare word
        # ban would forbid saying so.
        (r'(?i)judge[_-]?(?:model|prompt|score|client)', "LLM-judge scoring (replaced by the deterministic evals/ suite)"),
        (r'(?i)(?:gemini|openai|anthropic)[_-]?judge', "cloud judge reference"),
        (r'def\s+_?judge', "judge scoring function"),
        (r'--tool-eval', "tool eval orchestrator flag"),
        (r'"best_api_overall"', "deprecated best_api_overall key in recommendation dict"),
    ]
    found_any = False
    for pattern, label in dead_patterns:
        for file in repo_python_files():
            if file.name == "smoke_test.py":
                continue
            text = file.read_text()
            if re.search(pattern, text):
                fail(f"{file.relative_to(ROOT)}: contains {label}")
                found_any = True
        for file in active_files("*.sh"):
            text = file.read_text()
            if re.search(pattern, text):
                fail(f"{file.relative_to(ROOT)}: contains {label}")
                found_any = True
    if not found_any:
        ok("No dead backend strings found in active source files")


def check_aggressive_pkill() -> None:
    print("\n--- Aggressive pkill pattern (active source only) ---")
    pattern = re.compile(r'pkill\s+-f\s+["\']?llama-server')
    found = False
    for file in active_files("*"):
        if file.suffix not in (".py", ".sh"):
            continue
        if file.name == "smoke_test.py":
            continue
        text = file.read_text()
        if pattern.search(text):
            fail(f"{file.relative_to(ROOT)}: contains pkill -f llama-server")
            found = True
    if not found:
        ok("No aggressive pkill pattern found in active source files")


def check_duplicated_prompts() -> None:
    print("\n--- Duplicated prompt blocks (outside prompts.py) ---")
    from prompts import PROMPTS
    signature = PROMPTS["micro"][:80]
    dup_count = 0
    for py_file in active_files("*.py"):
        if py_file.name in ("prompts.py", "smoke_test.py", "__init__.py"):
            continue
        if "__pycache__" in str(py_file):
            continue
        text = py_file.read_text()
        if signature in text:
            dup_count += 1
            fail(f"{py_file.relative_to(ROOT)}: contains duplicated prompt text")
    if dup_count == 0:
        ok("No duplicated prompt blocks outside prompts.py")


def check_aggregate_duplicate_logic() -> None:
    print("\n--- Aggregate duplicate-pass detection ---")
    text = (SCRIPTS_DIR / "aggregate_results.py").read_text()
    if "duplicate_passes" in text and "duplicate_pass_count" in text:
        ok("Duplicate detection fields present in _score()")
    else:
        fail("Missing duplicate detection in aggregate_results.py")
    if "warnings" in text:
        ok("Warnings dict present in aggregate()")
    else:
        fail("Missing warnings in aggregate_results.py")


def check_aggregate_token_trust() -> None:
    print("\n--- Aggregate token trust logic ---")
    text = (SCRIPTS_DIR / "aggregate_results.py").read_text()
    if "trusted_only" in text:
        ok("trusted_only filtering in _best()")
    else:
        fail("Missing trusted_only parameter in _best()")
    if "trusted_api_overall" in text:
        ok("trusted_api_overall recommendation present")
    else:
        fail("Missing trusted_api_overall recommendation")
    if "fastest_api_overall" in text:
        ok("fastest_api_overall recommendation present")
    else:
        fail("Missing fastest_api_overall recommendation")
    if "APPROXIMATE_THRESHOLD" in text:
        ok("APPROXIMATE_THRESHOLD defined in aggregate_results.py")
    else:
        fail("Missing APPROXIMATE_THRESHOLD in aggregate_results.py")
    if "token_trust" in text and '"trusted"' in text and '"approximate"' in text:
        ok("token_trust field with trusted/approximate values")
    else:
        fail("Missing token_trust field in _score()")
    if "worst_token_rank" in text:
        ok("worst_token_rank used in scoring")
    else:
        fail("Missing worst_token_rank (still using best_rank only)")


def check_removed_sidecars() -> None:
    print("\n--- Removed sidecar files ---")
    removed = [
        SCRIPTS_DIR / "run_tool_eval.py",
        SCRIPTS_DIR / "profile_mlx.py",
        SCRIPTS_DIR / "generate_opencode_config.py",
        CONFIG_DIR / "tool_prompts.json",
        CONFIG_DIR / "tools_schema.json",
        CONFIG_DIR / "eval_prompts.json",
        CONFIG_DIR / "opencode_llamacpp_provider.generated.json",
    ]
    for path in removed:
        if path.exists():
            fail(f"{path.relative_to(ROOT)} should not exist")
        else:
            ok(f"{path.relative_to(ROOT)} removed")


def check_tui_imports() -> None:
    print("\n--- TUI imports standalone runners ---")
    tui_path = ROOT / "bench_tui.py"
    text = tui_path.read_text()
    if "from scripts.bench_llamacpp_api import run_benchmark" in text:
        ok("TUI imports bench_llamacpp_api")
    else:
        fail("TUI does not import bench_llamacpp_api")
    if "from scripts.bench_mlx_direct import run_benchmark" in text:
        ok("TUI imports bench_mlx_direct")
    else:
        fail("TUI does not import bench_mlx_direct")
    if "from scripts.bench_mlx_api import run_benchmark" in text:
        ok("TUI imports bench_mlx_api")
    else:
        fail("TUI does not import bench_mlx_api")
    if "from scripts.bench_llamacpp_raw import run_benchmark" in text:
        ok("TUI imports bench_llamacpp_raw")
    else:
        fail("TUI does not import bench_llamacpp_raw")
    # The TUI used to be speed-only, with quality work living in a since-removed
    # LLM-judge sidecar. The judge is gone for good; what replaced it is the
    # deterministic evals/ suite, which the TUI drives through run_suite().
    if "from scripts.bench_quality import run_suite" in text:
        ok("TUI wires the quality eval suite")
    else:
        fail("TUI does not import the quality eval suite")
    # Same precision rule as check_removed_backend_strings: match identifiers a
    # reintroduced judge would carry, not the word in a comment saying there
    # isn't one.
    if not re.search(r"(?i)judge[_-]?(?:model|prompt|score|client)", text):
        ok("TUI has no LLM-judge scoring")
    else:
        fail("TUI reintroduced LLM-judge scoring")
    if "pkill" not in text:
        ok("No pkill in TUI")
    else:
        fail("TUI still contains pkill")


def check_gitignore() -> None:
    print("\n--- .gitignore coverage ---")
    gitignore = ROOT / ".gitignore"
    text = gitignore.read_text()
    checks = ["__pycache__/", "*.pyc", ".DS_Store", "configs/models.local.json", "tools/", "results/"]
    for check in checks:
        if check in text:
            ok(f".gitignore has {check}")
        else:
            fail(f".gitignore missing {check}")


# Path fragments: a raw substring hit is always a real leak.
LEAK_SUBSTRINGS = [
    "/Users/",
    "/home/",
    "tools/llama.cpp-mtp",
    ".lmstudio/models/",
]

# Machine and account identifiers, loaded from a git-ignored file.
#
# These are inherently personal — hostnames, usernames, internal project
# codenames — so committing them would mean the public repo advertises exactly
# the private names it exists to keep out. Put yours in `.leakpatterns`, one per
# line (see .leakpatterns.example). Matching is on word boundaries: a bare
# substring search flags "formatting" for containing a hostname like "matti",
# and false positives are what train people to ignore the check.
LEAK_WORDS_FILE = ROOT / ".leakpatterns"


def _load_leak_words() -> list[str]:
    try:
        lines = LEAK_WORDS_FILE.read_text().splitlines()
    except OSError:
        return []
    return [w.strip() for w in lines if w.strip() and not w.startswith("#")]


LEAK_WORDS = _load_leak_words()

# Docs legitimately show `~/.lmstudio/models` as the conventional model root.
# `~`-prefixed references carry no username, so they are not a leak.
LEAK_ALLOWED_SUBSTRINGS = ["~/.lmstudio/models"]

SCANNABLE_SUFFIXES = {".py", ".sh", ".md", ".json", ".toml", ".txt", ".yml", ".yaml", ".html", ".css", ".js"}

_WORD_PATTERNS = [(word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)) for word in LEAK_WORDS]


def _leaks_in(text: str) -> list[str]:
    for allowed in LEAK_ALLOWED_SUBSTRINGS:
        text = text.replace(allowed, "")
    hits = [pattern for pattern in LEAK_SUBSTRINGS if pattern in text]
    hits.extend(word for word, pattern in _WORD_PATTERNS if pattern.search(text))
    return hits


def check_public_path_hygiene() -> None:
    print("\n--- Public path hygiene ---")
    files = tracked_files()
    if not files:
        ok("Skipped (not a git checkout) - hygiene is defined by what git tracks")
        return

    found = False
    for path in files:
        if path.name == "smoke_test.py":
            continue
        # A tracked symlink publishes its target string, which is a local path.
        if path.is_symlink():
            fail(f"{path.relative_to(ROOT)}: tracked symlink leaks its target {os.readlink(path)!r}")
            found = True
            continue
        if not path.is_file() or path.suffix not in SCANNABLE_SUFFIXES:
            continue
        for pattern in _leaks_in(path.read_text(errors="ignore")):
            fail(f"{path.relative_to(ROOT)}: contains local/development string {pattern!r}")
            found = True
    if not found:
        ok(f"No local/development paths in {len(files)} git-tracked files")


def check_untracked_not_ignored() -> None:
    """Anything neither tracked nor ignored is one `git add -A` away from being
    published without review. Fail on leaky content, warn on the rest."""
    print("\n--- Untracked files are ignored or clean ---")
    candidates = untracked_unignored_files()
    if not candidates:
        ok("No untracked, unignored files")
        return

    risky = []
    for path in candidates:
        if path.is_symlink():
            risky.append((path, f"symlink to {os.readlink(path)}"))
            continue
        if not path.is_file() or path.suffix not in SCANNABLE_SUFFIXES:
            continue
        leaks = _leaks_in(path.read_text(errors="ignore"))
        if leaks:
            risky.append((path, f"contains {leaks[0]!r}"))

    for path, why in risky:
        fail(f"{path.relative_to(ROOT)}: untracked and not ignored - {why}")
    if not risky:
        ok(f"{len(candidates)} untracked file(s), none leak local paths")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Skip Python compilation check")
    args = parser.parse_args()

    print("=" * 60)
    print("Benchmark Pipeline Smoke Test")
    print("=" * 60)
    print()
    print("Validates static/dry-run integrity. Does NOT prove real benchmark correctness.")
    print("Heavyweight model benchmarks must be run against real hardware to verify measurements.")
    print()

    check_scripts_exist()
    if not args.dry_run:
        check_python_compile()
    check_prompts_load()
    check_backend_runners()
    check_csv_columns()
    check_registry_architecture()
    check_removed_backend_strings()
    check_aggressive_pkill()
    check_duplicated_prompts()
    check_aggregate_duplicate_logic()
    check_aggregate_token_trust()
    check_removed_sidecars()
    check_tui_imports()
    check_gitignore()
    check_public_path_hygiene()
    check_untracked_not_ignored()

    print()
    if EXIT == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{EXIT} failures - see above")
        sys.exit(EXIT)


if __name__ == "__main__":
    main()
