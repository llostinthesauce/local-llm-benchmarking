#!/usr/bin/env python3
"""
HumanEval — functional correctness of generated code (Chen et al., 2021).

164 hand-written Python problems. Each is a signature plus docstring; the model
completes the body and the completion is scored by *running* the reference unit
tests. pass@1 with greedy decoding is the number labs quote.

  SAFETY: this eval executes code written by the model under test.

That is inherent to the benchmark — functional correctness cannot be checked
without running the function. The mitigations here are the same ones OpenAI's
own harness uses, and they are worth understanding before you enable it:

  * a separate subprocess per problem, so a crash or hang cannot take the
    harness down, with a hard wall-clock timeout
  * CPU-time and address-space rlimits, so a runaway loop or allocation dies
  * a scratch working directory, so relative-path writes stay contained
  * a guard prelude that removes the obvious destructive entry points
    (os.system, subprocess, shutil.rmtree, os.remove, ...) from the child

None of that is a security boundary. It stops a *buggy* completion from doing
damage; it does not stop a deliberately malicious one — the child is a normal
process running as you, and it can still reach the network and read your files.
Running this against a model you do not trust belongs in a VM or container.

Because of that, execution is off by default and requires an explicit
`--allow-code-execution` flag on top of `--fetch`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from ..core import Case, Score
from . import fetch

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
FILENAME = "humaneval.jsonl"

DEFAULT_TIMEOUT_S = 15
DEFAULT_MEMORY_MB = 2048

INSTRUCTION = (
    "Complete the following Python function. Output only the complete function "
    "definition inside a single ```python code block. Do not add explanations, "
    "examples, or test code."
)

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Removed from the child interpreter before the completion runs. This blocks the
# accidental `os.system("rm -rf ...")` a model sometimes emits when it
# hallucinates a shell step; it is not a sandbox.
#
# NOTE: the names below (os.system, subprocess.run, shutil.rmtree, ...) are
# assignment *targets* set to None inside the child, not calls. Nothing in this
# module invokes a shell; the harness itself uses subprocess.run with an argv
# list and no shell=True.
GUARD_PRELUDE = """
import builtins, os, shutil, subprocess
for _mod, _names in (
    (os, ("system", "remove", "unlink", "rmdir", "removedirs", "renames", "kill", "killpg")),
    (shutil, ("rmtree", "move", "chown")),
    (subprocess, ("run", "call", "check_call", "check_output", "Popen")),
):
    for _name in _names:
        if hasattr(_mod, _name):
            setattr(_mod, _name, None)
builtins.exit = None
builtins.quit = None
"""

RLIMIT_PRELUDE = """
import resource
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
try:
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except (ValueError, OSError):
    pass
"""


def build_cases(limit: int = 164, allow_fetch: bool = False, **_ignored: object) -> List[Case]:
    path: Path = fetch.ensure(FILENAME, URL, allow_fetch)
    rows = fetch.read_jsonl(path)
    # HumanEval is ordered and small; take a prefix rather than sampling so
    # partial runs cover the same problems for every model.
    if limit:
        rows = rows[:limit]

    cases: List[Case] = []
    for row in rows:
        cases.append(
            Case(
                case_id=str(row.get("task_id", "")).replace("/", "_") or f"humaneval_{len(cases)}",
                prompt=f"{INSTRUCTION}\n\n```python\n{row['prompt']}```",
                max_tokens=768,
                temperature=0.0,
                meta={
                    "prompt_source": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )
        )
    return cases


def extract_code(response: str, entry_point: str) -> str:
    """Pull the function definition out of a chat response.

    Models wrap code in fences inconsistently, so try the fenced block first and
    fall back to treating the whole response as code.
    """
    blocks = _FENCE.findall(response)
    for block in blocks:
        if f"def {entry_point}" in block:
            return block
    if blocks:
        return blocks[0]
    return response


def _build_program(case: Case, completion: str, memory_mb: int, cpu_s: int) -> str:
    return "\n".join(
        [
            RLIMIT_PRELUDE.format(cpu=cpu_s, mem=memory_mb * 1024 * 1024),
            GUARD_PRELUDE,
            completion,
            case.meta["test"],
            f"check({case.meta['entry_point']})",
            "print('__HUMANEVAL_PASS__')",
        ]
    )


def run_completion(
    case: Case,
    completion: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> Score:
    """Execute the completion against the reference tests in a child process."""
    program = _build_program(case, completion, memory_mb, timeout_s)
    with tempfile.TemporaryDirectory(prefix="humaneval_") as workdir:
        script = Path(workdir) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={"PATH": os.environ.get("PATH", ""), "HOME": workdir},
            )
        except subprocess.TimeoutExpired:
            return Score.binary(False, f"timeout after {timeout_s}s")
        except OSError as exc:
            return Score.binary(False, f"could not execute: {exc}")

    if "__HUMANEVAL_PASS__" in result.stdout:
        return Score.binary(True, "tests passed")
    error = (result.stderr or "").strip().splitlines()
    detail = error[-1][:200] if error else f"exit {result.returncode}"
    return Score.binary(False, detail)


def score(case: Case, response: str) -> Score:
    """Scoring runs code, so it is gated on an explicit opt-in.

    The runner sets HUMANEVAL_ALLOW_EXEC only when --allow-code-execution was
    passed. Without it the case is reported as unscored rather than silently
    counted as a failure, which would understate a model that answered fine.
    """
    if os.environ.get("HUMANEVAL_ALLOW_EXEC") != "1":
        return Score(value=0.0, passed=False, detail="skipped: --allow-code-execution not set")
    completion = extract_code(response, str(case.meta["entry_point"]))
    return run_completion(case, completion)


def load_raw(limit: int, allow_fetch: bool) -> List[dict]:
    """Escape hatch for tooling that wants the problems without Case wrapping."""
    path = fetch.ensure(FILENAME, URL, allow_fetch)
    rows = fetch.read_jsonl(path)
    return rows[:limit] if limit else rows


if __name__ == "__main__":  # tiny self-check on the extractor
    sample = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    print(json.dumps({"extracted": extract_code(sample, "add")}, indent=2))
