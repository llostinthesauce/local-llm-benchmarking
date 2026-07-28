#!/usr/bin/env python3
"""
Shared types and the OpenAI-compatible client used by every eval.

An eval is three things:
  1. `build_cases()` — turn a config into a deterministic list of Case objects
  2. the runner       — sends each Case to a server (this module)
  3. `score()`        — turn (Case, response text) into a Score

Nothing here talks to a specific backend. Any server that speaks
`POST /v1/chat/completions` works: llama-server, mlx_lm.server, mlx_vlm.server,
oMLX, Ollama, vLLM, or a hosted API.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Case:
    """One prompt plus everything needed to score the answer."""

    case_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    system: str = ""
    # Anything score() needs: the expected answer, constraint specs, depths...
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """Result of scoring one Case. `value` is always normalized to 0.0-1.0."""

    value: float
    passed: bool
    detail: str = ""

    @staticmethod
    def binary(passed: bool, detail: str = "") -> "Score":
        return Score(value=1.0 if passed else 0.0, passed=passed, detail=detail)


@dataclass
class Response:
    """What the server actually returned, plus timing."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class Eval:
    """A named, scoreable benchmark.

    tier 1 evals generate their own data and run with no network.
    tier 2 evals need a dataset download (see evals/datasets/fetch.py).
    """

    name: str
    tier: int
    description: str
    build_cases: Callable[..., List[Case]]
    score: Callable[[Case, str], Score]
    # Higher-is-better metric name reported in the summary.
    metric: str = "accuracy"
    needs_code_execution: bool = False
    # Set when a case cannot be scored on its own — the determinism eval, for
    # example, only means anything when repeats are compared against each other.
    # When present the runner calls this instead of per-case `score`.
    score_all: Optional[Callable[[List[Case], List[str]], List[Score]]] = None


class EvalClient:
    """Minimal OpenAI-compatible chat client.

    Standard library only, matching the rest of the repo's serving scripts, so
    the eval harness runs before any project virtualenv is active.
    """

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8080/v1",
        model: str = "default_model",
        api_key: str = "",
        timeout: int = 600,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, case: Case) -> Response:
        """Send one Case and return the assistant text.

        Reasoning models emit `reasoning_content` alongside `content`. Only
        `content` is scored: a model that reaches the right answer after a long
        think still answered correctly, and its scratchpad is not the answer.
        """
        messages: List[Dict[str, str]] = []
        if case.system:
            messages.append({"role": "system", "content": case.system})
        messages.append({"role": "user", "content": case.prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": case.max_tokens,
            "temperature": case.temperature,
            "top_p": case.top_p,
            "stream": False,
        }
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=body, method="POST", headers=self._headers()
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as raw:
                data = json.loads(raw.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            return Response(text="", error=f"http_{exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return Response(text="", error=f"unreachable: {exc}")
        except ValueError as exc:
            return Response(text="", error=f"bad_json: {exc}")

        elapsed = time.perf_counter() - start
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        return Response(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=elapsed,
        )

    def probe(self) -> Optional[str]:
        """Return an error string if the server is not usable, else None."""
        try:
            request = urllib.request.Request(f"{self.api_base}/models", headers=self._headers())
            with urllib.request.urlopen(request, timeout=5):
                return None
        except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
            return f"cannot reach {self.api_base}/models ({exc})"
