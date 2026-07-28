"""Quality evaluation suite for locally served LLMs.

Speed benchmarks answer "how fast?". These answer "is it still correct?" —
the question that decides whether a quantization is usable.
"""
from __future__ import annotations

from .core import Case, Eval, EvalClient, Response, Score

__all__ = ["Case", "Eval", "EvalClient", "Response", "Score"]
