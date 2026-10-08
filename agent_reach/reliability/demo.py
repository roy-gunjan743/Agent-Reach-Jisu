"""Autonomous Reliability Layer Demo for Agent Reach.

Demonstrates content-aware normalization, semantic evaluation via Gemma 4,
and automated multi-backend fallback.

Usage:
    python -m agent_reach.reliability.demo [--query "..."] [--offline-fixtures] [--fast] [--max-chars 4000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from agent_reach.reliability.backends import (
    DEMO_QUERY,
    build_real_router,
)
from agent_reach.reliability.router import AdaptiveRouter

# Ensure Windows console handles UTF-8 / symbols safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_demo_fixture(name: str) -> str:
    """Load sample fixture from reliability/demo_data/ directory."""
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
    path = os.path.join(demo_dir, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    raise FileNotFoundError(f"Fixture {name} not found in {demo_dir}")


def preflight_check() -> str | None:
    """Check for GEMINI_API_KEY environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "❌ GEMINI_API_KEY environment variable is missing.\n"
            "   The Gemma 4 judge requires a valid Gemini API key.\n\n"
            "To resolve this, set your key in the environment:\n"
            '   export GEMINI_API_KEY="your-gemini-api-key"   (Linux/macOS)\n'
            '   $env:GEMINI_API_KEY="your-gemini-api-key"     (PowerShell)\n'
            '   set GEMINI_API_KEY=your-gemini-api-key        (cmd)\n\n'
            "Get an API key at: https://aistudio.google.com/app/apikey"
        )
    return None


def run_demo(
    query: str = DEMO_QUERY,
    offline_fixtures: bool = False,
    fast: bool = False,
    max_chars: int = 4000,
    router: AdaptiveRouter | None = None,
) -> int:
    """Run the complete reliability fallback demonstration."""
    start_total_time = time.perf_counter()

    if offline_fixtures:
        print("===================================================================")
        print("OFFLINE FIXTURE MODE: results are NOT live and scores are NOT from Gemma")
        print("===================================================================")

    err = preflight_check()
    if err:
        print(err)
        return 1

    print("\n===================================================================")
    print("         AGENT REACH: ADAPTIVE RELIABILITY LAYER DEMO             ")
    print("===================================================================")
    print(f"Query: {query}")
    print("-------------------------------------------------------------------")

    if router is None:
        if offline_fixtures:
            def fixture_a(q: str) -> str:
                return _load_demo_fixture("backend_a_tron.txt")

            def fixture_b(q: str) -> str:
                return _load_demo_fixture("backend_b_mcp.txt")

            backends = {
                "direct-http-tron": fixture_a,
                "jina-mcp-wiki": fixture_b,
            }
            router = AdaptiveRouter(backends=backends)
        else:
            router = build_real_router()

    result = router.route(query)

    print("\n===================================================================")
    print("FINAL RESULT SUMMARY")
    print("===================================================================")

    total_pipeline_sec = time.perf_counter() - start_total_time

    trace = router.last_trace
    if not trace:
        print("❌ No backends attempted.")
        return 1

    trace_a = trace[0] if len(trace) > 0 else None
    trace_b = trace[1] if len(trace) > 1 else None

    if trace_a and trace_a["outcome"] == "fetch_error":
        print("Fetch error on Backend A (this is not a semantic rejection)")
        return 1

    if trace_a and trace_a["outcome"] == "judge_error":
        print(f"Judge unavailable: {trace_a.get('error')}")
        return 1

    if trace_a and trace_a["outcome"] == "accepted":
        print("Accepted on first backend (no fallback needed)")
        return 1

    if (
        trace_a
        and trace_a["outcome"] == "rejected"
        and trace_b
        and trace_b["outcome"] == "accepted"
        and result is not None
        and result.backend == "jina-mcp-wiki"
    ):
        print(f"✓ Selected backend: {result.backend}")
        print("✓ Adaptive fallback completed")
        print(
            f"[PASS] Semantic fallback successful: Backend A rejected -> Backend B accepted in {total_pipeline_sec:.2f}s."
        )
        return 0

    if result is None or (trace_b and trace_b["outcome"] != "accepted"):
        print("[FAIL] No backend passed the semantic quality threshold.")
        return 1

    print("[FAIL] Demo execution did not match expected fallback path.")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Reach Adaptive Router & Normalizer Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=DEMO_QUERY, help="Query to run")
    parser.add_argument(
        "--offline-fixtures",
        action="store_true",
        help="Use offline bundled fixtures instead of live web fetch (requires GEMINI_API_KEY)",
    )
    parser.add_argument("--fast", action="store_true", help="Disable pacing delays")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max characters for normalizer")

    args = parser.parse_args()

    exit_code = run_demo(
        query=args.query,
        offline_fixtures=args.offline_fixtures,
        fast=args.fast,
        max_chars=args.max_chars,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
