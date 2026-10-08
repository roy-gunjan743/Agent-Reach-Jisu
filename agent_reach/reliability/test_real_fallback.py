"""Real fallback end-to-end runner.

Usage::

    python -m agent_reach.reliability.test_real_fallback

Runs Backend A (Direct HTTP → Tron MCP) → judge REJECT → fallback →
Backend B (Jina Reader → Model Context Protocol) → judge ACCEPT.

Exit codes:
    0  — A rejected AND B accepted AND final backend is B.
    1  — Any other outcome, or missing GEMINI_API_KEY.

The judge is loaded from ``agent_reach.reliability.judge.judge_result``
(Gemma 4 via Gemini API).
If GEMINI_API_KEY is missing, prints a clean message and exits 1.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "❌ GEMINI_API_KEY environment variable is missing.\n"
            "   The Gemma 4 judge requires GEMINI_API_KEY to evaluate live results.\n"
            "   Please set GEMINI_API_KEY and rerun."
        )
        sys.exit(1)

    from agent_reach.reliability.backends import DEMO_QUERY, build_real_router

    print("=" * 60)
    print("  REAL FALLBACK TEST — Adaptive Router (Gemma Live)")
    print("=" * 60)
    print(f"\nQuery: {DEMO_QUERY}\n")

    router = build_real_router()  # uses real judge via lazy import
    result = router.route(DEMO_QUERY)

    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)

    if result is None:
        print("❌ No result returned at all.")
        router.print_stats()
        sys.exit(1)

    print(f"Backend:  {result.backend}")
    print(f"Score:    {result.score:.2f}")
    print(f"Decision: {result.evaluation.get('decision', 'unknown')}")
    print(f"Reason:   {result.evaluation.get('reason', '')}")
    print(f"\nResult excerpt ({len(result.result):,} chars):")
    print(result.result[:500])

    router.print_stats()

    # Verify expected outcome: A rejected, B accepted, final is B.
    stats = router.stats_summary()
    a_stats = stats.get("direct-http-tron", {})
    b_stats = stats.get("jina-mcp-wiki", {})

    a_rejected = a_stats.get("failures", 0) >= 1
    b_accepted = b_stats.get("successes", 0) >= 1
    final_is_b = result.backend == "jina-mcp-wiki"

    print("\n" + "=" * 60)
    print("  ASSERTIONS")
    print("=" * 60)
    print(f"  Backend A rejected:      {'✅' if a_rejected else '❌'}")
    print(f"  Backend B accepted:      {'✅' if b_accepted else '❌'}")
    print(f"  Final backend is B:      {'✅' if final_is_b else '❌'}")

    if a_rejected and b_accepted and final_is_b:
        print("\n✅ PASS — Real fallback worked as expected.")
        sys.exit(0)
    else:
        print("\n❌ FAIL — Unexpected outcome.")
        sys.exit(1)


if __name__ == "__main__":
    main()
