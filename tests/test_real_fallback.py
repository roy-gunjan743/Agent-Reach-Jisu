"""Offline and live tests for the adaptive router real fallback.

Offline tests (always run in CI):
    Use fake backends + stub judges to verify fallback mechanics, stats
    recording, error handling, deterministic ordering, and judge outages.

Live tests (opt-in):
    Marked ``@pytest.mark.live``, skipped unless ``RUN_LIVE=1`` is set.
    Fails if ``GEMINI_API_KEY`` is not set.

Run offline::

    pytest tests/test_real_fallback.py -q

Run live::

    RUN_LIVE=1 GEMINI_API_KEY=<key> pytest tests/test_real_fallback.py -q
"""

from __future__ import annotations

import os

import pytest

from agent_reach.reliability.router import AdaptiveRouter, BackendResult
from agent_reach.reliability.scorer import BackendStats

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _stub_judge_reject(query: str, result: str) -> dict:
    """Judge that always rejects."""
    return {
        "relevance": 0.2,
        "freshness": 0.3,
        "completeness": 0.1,
        "confidence": 0.2,
        "decision": "reject",
        "reason": "not relevant",
    }


def _stub_judge_accept(query: str, result: str) -> dict:
    """Judge that always accepts."""
    return {
        "relevance": 0.9,
        "freshness": 0.8,
        "completeness": 0.85,
        "confidence": 0.9,
        "decision": "accept",
        "reason": "good answer",
    }


def _stub_judge_selective(query: str, result: str) -> dict:
    """Reject if 'Tron' or 'game' in result, accept otherwise."""
    text = result.lower()
    if "tron" in text or "game" in text or "bad" in text:
        return {
            "relevance": 0.15,
            "freshness": 0.3,
            "completeness": 0.1,
            "confidence": 0.2,
            "decision": "reject",
            "reason": "wrong topic",
        }
    return {
        "relevance": 0.9,
        "freshness": 0.8,
        "completeness": 0.85,
        "confidence": 0.9,
        "decision": "accept",
        "reason": "matches query",
    }


QUERY = "What is MCP and how does it improve AI agent tool usage?"


# =========================================================================
# OFFLINE TESTS
# =========================================================================


class TestFallbackOffline:
    """Offline tests — no network, no API key."""

    def test_fallback_when_a_rejected(self) -> None:
        """Backend A rejected → fallback to B → B accepted."""
        def backend_a(q: str) -> str:
            return "MCP is a Tron game villain from 1982."

        def backend_b(q: str) -> str:
            return (
                "Model Context Protocol (MCP) is an open protocol for "
                "AI agents to connect to external tools."
            )

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_selective,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.backend == "b"
        assert result.evaluation["decision"] == "accept"

    def test_b_result_is_returned(self) -> None:
        """The actual content from backend B is in the returned result."""
        expected_text = "Model Context Protocol enables tool integration"

        def backend_a(q: str) -> str:
            return "bad irrelevant game content"

        def backend_b(q: str) -> str:
            return expected_text

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_selective,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.result == expected_text

    def test_stats_record_success_failure_quality(self) -> None:
        """A: failures==1 with quality recorded; B: successes==1 with quality."""
        def backend_a(q: str) -> str:
            return "Tron movie villain bad content"

        def backend_b(q: str) -> str:
            return "Model Context Protocol is a real protocol for AI."

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_selective,
        )
        router.route(QUERY)

        summary = router.stats_summary()

        assert summary["a"]["failures"] == 1
        assert summary["a"]["successes"] == 0
        assert summary["a"]["quality_score"] > 0  # quality was recorded

        assert summary["b"]["successes"] == 1
        assert summary["b"]["failures"] == 0
        assert summary["b"]["quality_score"] > 0

    def test_backend_exception_falls_back(self) -> None:
        """A backend raising an exception should fall back, not crash."""
        def crashing_backend(q: str) -> str:
            raise RuntimeError("connection refused")

        def good_backend(q: str) -> str:
            return "Model Context Protocol for AI agents."

        router = AdaptiveRouter(
            backends={"crash": crashing_backend, "good": good_backend},
            judge=_stub_judge_accept,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.backend == "good"

        summary = router.stats_summary()
        assert summary["crash"]["failures"] == 1

    def test_empty_result_falls_back(self) -> None:
        """A backend returning empty string should fall back."""
        def empty_backend(q: str) -> str:
            return ""

        def real_backend(q: str) -> str:
            return "Real MCP content."

        router = AdaptiveRouter(
            backends={"empty": empty_backend, "real": real_backend},
            judge=_stub_judge_accept,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.backend == "real"

        summary = router.stats_summary()
        assert summary["empty"]["failures"] == 1

    def test_whitespace_only_falls_back(self) -> None:
        """A backend returning only whitespace should fall back."""
        def ws_backend(q: str) -> str:
            return "   \n\t  "

        def real_backend(q: str) -> str:
            return "MCP content here."

        router = AdaptiveRouter(
            backends={"ws": ws_backend, "real": real_backend},
            judge=_stub_judge_accept,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.backend == "real"

    def test_all_rejected_returns_best(self) -> None:
        """If nothing passes threshold, the best attempt is returned."""
        def backend_a(q: str) -> str:
            return "Some content A"

        def backend_b(q: str) -> str:
            return "Some content B"

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_reject,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.score > 0
        assert len(router.last_trace) == 2
        assert all(t["outcome"] == "rejected" for t in router.last_trace)

    def test_deterministic_a_then_b_order(self) -> None:
        """With fresh stats (all 0), backends are tried in insertion order."""
        call_order: list[str] = []

        def backend_a(q: str) -> str:
            call_order.append("a")
            return "content a"

        def backend_b(q: str) -> str:
            call_order.append("b")
            return "content b"

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_accept,  # accept first one
        )
        router.route(QUERY)

        assert call_order == ["a"]

    def test_insertion_order_preserved_on_fallback(self) -> None:
        """On fallback, A is tried first, then B (insertion order)."""
        call_order: list[str] = []

        def backend_a(q: str) -> str:
            call_order.append("a")
            return "bad Tron game content"

        def backend_b(q: str) -> str:
            call_order.append("b")
            return "Model Context Protocol for AI."

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_selective,
        )
        router.route(QUERY)

        assert call_order == ["a", "b"]

    def test_injectable_judge(self) -> None:
        """The judge parameter overrides the default lazy import."""
        custom_called = False

        def custom_judge(query: str, result: str) -> dict:
            nonlocal custom_called
            custom_called = True
            return _stub_judge_accept(query, result)

        router = AdaptiveRouter(
            backends={"x": lambda q: "content"},
            judge=custom_judge,
        )
        router.route(QUERY)

        assert custom_called

    def test_stats_summary_structure(self) -> None:
        """stats_summary() returns the expected dict structure."""
        router = AdaptiveRouter(
            backends={"a": lambda q: "content"},
            judge=_stub_judge_accept,
        )
        router.route(QUERY)

        summary = router.stats_summary()
        assert "a" in summary
        entry = summary["a"]
        for key in (
            "successes",
            "failures",
            "quality_score",
            "success_rate",
            "reliability_score",
        ):
            assert key in entry

    def test_backend_result_fields(self) -> None:
        """BackendResult has all expected fields."""
        router = AdaptiveRouter(
            backends={"x": lambda q: "content"},
            judge=_stub_judge_accept,
        )
        result = router.route(QUERY)

        assert isinstance(result, BackendResult)
        assert isinstance(result.backend, str)
        assert isinstance(result.result, str)
        assert isinstance(result.evaluation, dict)
        assert isinstance(result.score, float)

    def test_build_real_router_with_stub(self) -> None:
        """build_real_router() accepts a stub judge and returns a router."""
        from agent_reach.reliability.backends import build_real_router

        router = build_real_router(judge=_stub_judge_selective)
        assert isinstance(router, AdaptiveRouter)
        assert "direct-http-tron" in router.backends
        assert "jina-mcp-wiki" in router.backends

    def test_quality_recorded_on_failure(self) -> None:
        """record_failure(quality=...) stores the quality score."""
        stat = BackendStats("test")
        stat.record_failure(quality=0.42)

        assert stat.failures == 1
        assert stat.quality_score == 0.42

    def test_record_failure_no_quality_backward_compat(self) -> None:
        """record_failure() with no args still works (backward compat)."""
        stat = BackendStats("test")
        original_quality = stat.quality_score
        stat.record_failure()

        assert stat.failures == 1
        assert stat.quality_score == original_quality  # unchanged

    def test_judge_error_does_not_count_as_failure(self) -> None:
        """Judge error must NOT record a failure on the backend."""
        def error_judge(q: str, r: str) -> dict:
            return {"judge_error": True, "reason": "API rate limit"}

        router = AdaptiveRouter(
            backends={"a": lambda q: "some text"},
            judge=error_judge,
        )
        result = router.route(QUERY)

        assert result is None
        assert router.stats["a"].failures == 0
        assert len(router.last_trace) == 1
        assert router.last_trace[0]["outcome"] == "judge_error"

    def test_judge_raising_exception_handles_as_judge_error(self) -> None:
        """Judge raising an exception is handled as judge_error without recording backend failure."""
        def raising_judge(q: str, r: str) -> dict:
            raise RuntimeError("Judge timeout")

        router = AdaptiveRouter(
            backends={"a": lambda q: "some text"},
            judge=raising_judge,
        )
        result = router.route(QUERY)

        assert result is None
        assert router.stats["a"].failures == 0
        assert len(router.last_trace) == 1
        assert router.last_trace[0]["outcome"] == "judge_error"

    def test_judge_decision_case_insensitive_accept(self) -> None:
        """Decision 'Accept' (capitalized) is normalized to 'accept' and accepted."""
        def accept_judge(q: str, r: str) -> dict:
            return {
                "decision": "Accept",
                "relevance": 0.9,
                "freshness": 0.9,
                "completeness": 0.9,
                "confidence": 0.9,
                "reason": "good",
            }

        router = AdaptiveRouter(
            backends={"a": lambda q: "content"},
            judge=accept_judge,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.evaluation["decision"] == "accept"
        assert router.last_trace[0]["outcome"] == "accepted"

    def test_score_coercion_percentage_scale(self) -> None:
        """Score of 90 on 100-scale is normalized to 0.90."""
        def scale_judge(q: str, r: str) -> dict:
            return {
                "decision": "accept",
                "score": 90,
                "reason": "high score",
            }

        router = AdaptiveRouter(
            backends={"a": lambda q: "content"},
            judge=scale_judge,
        )
        result = router.route(QUERY)

        assert result is not None
        assert abs(result.score - 0.90) < 1e-5

    def test_score_null_handling(self) -> None:
        """Null/None score values normalize safely to 0.0."""
        def null_score_judge(q: str, r: str) -> dict:
            return {
                "decision": "reject",
                "relevance": None,
                "freshness": None,
                "completeness": None,
                "confidence": None,
                "reason": "none scores",
            }

        router = AdaptiveRouter(
            backends={"a": lambda q: "content"},
            judge=null_score_judge,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.score == 0.0

    def test_fetch_error_on_a_accept_on_b_trace(self) -> None:
        """Fetch error on A + accept on B records outcome 'fetch_error' for A in trace."""
        def backend_a(q: str) -> str:
            raise RuntimeError("HTTP 500 error")

        def backend_b(q: str) -> str:
            return "Good MCP content"

        router = AdaptiveRouter(
            backends={"a": backend_a, "b": backend_b},
            judge=_stub_judge_accept,
        )
        result = router.route(QUERY)

        assert result is not None
        assert result.backend == "b"
        assert len(router.last_trace) == 2
        assert router.last_trace[0]["outcome"] == "fetch_error"
        assert router.last_trace[1]["outcome"] == "accepted"

    def test_quality_running_average(self) -> None:
        """Success 0.9 then reject 0.2 produces running average quality score 0.55."""
        stat = BackendStats("test_backend")
        stat.record_success(0.9)
        assert abs(stat.quality_score - 0.9) < 1e-5
        stat.record_failure(0.2)
        assert abs(stat.quality_score - 0.55) < 1e-5


# =========================================================================
# LIVE TESTS
# =========================================================================

_skip_live = pytest.mark.skipif(
    os.environ.get("RUN_LIVE", "0") != "1",
    reason="Live tests disabled (set RUN_LIVE=1 to enable)",
)


@pytest.mark.live
@_skip_live
class TestFallbackLive:
    """Live tests — requires network + GEMINI_API_KEY."""

    def test_real_fallback_a_reject_b_accept(self) -> None:
        """End-to-end: A (Tron) rejected → B (MCP) accepted."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.fail("GEMINI_API_KEY is missing (required when RUN_LIVE=1)")

        from agent_reach.reliability.backends import DEMO_QUERY, build_real_router

        router = build_real_router()  # uses real judge
        result = router.route(DEMO_QUERY)

        assert result is not None
        assert result.backend == "jina-mcp-wiki"
        assert result.evaluation.get("decision") == "accept"

        assert len(router.last_trace) == 2
        trace_a, trace_b = router.last_trace[0], router.last_trace[1]

        assert trace_a["backend"] == "direct-http-tron"
        if trace_a["outcome"] != "rejected":
            pytest.fail(f"Backend A outcome was '{trace_a['outcome']}': this is not a semantic rejection")

        assert trace_b["backend"] == "jina-mcp-wiki"
        assert trace_b["outcome"] == "accepted"
