#!/usr/bin/env python3
"""
Canonical benchmark prompts - SINGLE source of truth.

All benchmark scripts, the TUI, and the exhaustive runner must import PROMPTS
and PASSES from this module. Do NOT duplicate prompt text in any other file.

Prompt IDs are stable: micro, normal, high, max.
"""
from __future__ import annotations

LONG_CONTEXT_MARGIN_TOKENS = 8192
LONG_CONTEXT_MIN_PROMPT_TOKENS = 8192

PROMPTS = {
    "micro": (
        "Write a Python function `fib(n: int) -> int` that returns the nth "
        "Fibonacci number using an iterative approach with memoization. "
        "Include type hints and a docstring."
    ),
    "normal": (
        "You are building a rate-limited async API client. Implement a `TokenBucket` "
        "class with: configurable fill rate and burst capacity, async context manager "
        "support (`async with`), an `acquire()` method that blocks until a token is "
        "available, and exponential backoff with jitter for retries. Include type hints, "
        "docstrings, and a usage example calling a mock HTTP endpoint."
    ),
    "high": (
        "You are refactoring a legacy Django monolith with 50K daily active users into "
        "a microservices architecture. Address these pain points: slow deploys (45 min CI), "
        "DB lock contention on user/auth tables under write load, cascading failures when "
        "the payment processor times out. Design the target architecture covering: "
        "1) service decomposition (auth, payments, orders, notifications) with bounded contexts, "
        "2) inter-service communication — sync REST vs async Kafka and which domains use which, "
        "3) database-per-service with the saga pattern for cross-service transactions, "
        "4) observability — structured logging, OpenTelemetry distributed tracing, Prometheus SLOs, "
        "5) deployment — Kubernetes with ArgoCD, canary releases, health checks. "
        "Provide a detailed directory layout for the auth service in FastAPI, "
        "and pseudocode for the saga orchestrator handling a checkout flow "
        "spanning auth→payment→inventory→notification. Discuss idempotency, "
        "compensating transactions, and the outbox pattern."
    ),
    "max": (
        "Write a comprehensive technical specification for a distributed time-series "
        "database optimized for IoT sensor data at petabyte scale. Cover: "
        "1) Write path — LSM-tree storage engine with time-based partitioning, WAL with "
        "group commit, adaptive compaction. "
        "2) Query engine — time-range scans, downsampling (min/max/avg/p99 over configurable "
        "windows), cross-metric correlation. "
        "3) Distributed architecture — consistent hashing for shard placement, Raft for "
        "metadata consensus, hinted handoff for fault tolerance. "
        "4) Compression — delta-of-delta timestamps, XOR floating-point compression, "
        "dictionary encoding for string tags. "
        "5) Retention — automated tiered storage (NVMe→SSD→object store) with TTL eviction "
        "and downsampled rollups. "
        "6) API — gRPC for ingestion (protobuf), REST for admin/dashboard. "
        "7) Targets — 10M writes/sec, <10ms p99 query for 1h range, 99.999% durability. "
        "Provide pseudocode for write path (ingest→WAL→memtable→flush) and read path "
        "(bloom filter→block index→scan). Cover failure scenarios and recovery."
    ),
}


PASSES = [
    {
        "id": "pass_1_micro",
        "ctx": 1024,
        "gen_tokens": 128,
        "prompt": PROMPTS["micro"],
        "label": "Micro | ctx 1K | 128 tok | Fibonacci",
    },
    {
        "id": "pass_2_normal",
        "ctx": 16384,
        "gen_tokens": 1024,
        "prompt": PROMPTS["normal"],
        "label": "Normal | ctx 16K | 1K tok | API client",
    },
    {
        "id": "pass_3_high",
        "ctx": 65536,
        "gen_tokens": 2048,
        "prompt": PROMPTS["high"],
        "label": "High | ctx 64K budget | 2K out | Hard architecture prompt",
    },
    {
        "id": "pass_4_max",
        "ctx": -1,
        "gen_tokens": 4096,
        "prompt": PROMPTS["max"],
        "fill_context": True,
        "label": "Max | fill near ctx cap | 4K out | Long-context stress",
    },
]


def context_budget_for_pass(pass_cfg: dict, ctx_cap: int) -> int:
    """Return the context budget selected by a pass."""
    raw_ctx = int(pass_cfg["ctx"])
    return ctx_cap if raw_ctx < 0 else min(raw_ctx, ctx_cap)


def target_prompt_tokens_for_pass(pass_cfg: dict, ctx_used: int) -> int:
    """Return the intended prompt-token load for a pass.

    The raw llama-bench path uses ctx_used directly as native prompt length.
    Text-generation paths need a real prompt large enough to fill that budget.
    """
    if not pass_cfg.get("fill_context"):
        return 0
    gen_tokens = int(pass_cfg.get("gen_tokens", 0))
    return max(LONG_CONTEXT_MIN_PROMPT_TOKENS, ctx_used - gen_tokens - LONG_CONTEXT_MARGIN_TOKENS)


def build_prompt_for_pass(pass_cfg: dict, ctx_used: int) -> tuple[str, int]:
    """Build the prompt for a pass and return (prompt, target_prompt_tokens).

    For max-context tests, the filler is deterministic and synthetic so GitHub
    runs are reproducible. "context" is intentionally repetitive: common
    tokenizers usually encode it close to one token per word, and API usage
    fields later prove the actual consumed prompt token count.
    """
    prompt = str(pass_cfg["prompt"])
    target_tokens = target_prompt_tokens_for_pass(pass_cfg, ctx_used)
    if not target_tokens:
        return prompt, 0

    header = (
        "LONG CONTEXT STRESS SECTION\n"
        "Use the following repeated context as load-bearing context. "
        "It is deterministic benchmark filler, not source material.\n\n"
    )
    footer = (
        "\n\nEND LONG CONTEXT STRESS SECTION\n\n"
        "Now answer the actual task using the instructions below. "
        "Do not summarize the filler unless it is directly relevant.\n\n"
        f"{prompt}"
    )
    fixed_word_estimate = len((header + footer).split())
    filler_words = max(1, target_tokens - fixed_word_estimate)
    blocks = []
    block_words = 1024
    full_blocks, remainder = divmod(filler_words, block_words)
    filler_block = " ".join(["context"] * block_words)
    for index in range(full_blocks):
        blocks.append(f"[block {index + 1:04d}] {filler_block}")
    if remainder:
        blocks.append(f"[block {full_blocks + 1:04d}] " + " ".join(["context"] * remainder))
    return header + "\n".join(blocks) + footer, target_tokens
