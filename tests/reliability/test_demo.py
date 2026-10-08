"""Tests for the reliability demo runner."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from agent_reach.reliability.demo import main, run_demo
from agent_reach.reliability.router import AdaptiveRouter


def _stub_judge_reject_a_accept_b(query: str, result: str) -> dict:
    """Reject if Tron in result, accept otherwise."""
    if "tron" in result.lower():
        return {
            "relevance": 0.15,
            "freshness": 0.2,
            "completeness": 0.1,
            "confidence": 0.2,
            "decision": "reject",
            "reason": "wrong topic",
        }
    return {
        "relevance": 0.9,
        "freshness": 0.9,
        "completeness": 0.9,
        "confidence": 0.9,
        "decision": "accept",
        "reason": "good topic",
    }


def test_demo_requires_api_key() -> None:
    """Demo exits 1 with clear message if GEMINI_API_KEY is missing."""
    buf = io.StringIO()
    with patch.dict("os.environ", {}, clear=True), redirect_stdout(buf):
        exit_code = run_demo(fast=True)

    output = buf.getvalue()
    assert exit_code == 1
    assert "GEMINI_API_KEY environment variable is missing" in output


def test_demo_rejects_mock_flag() -> None:
    """The --mock flag no longer exists in CLI parser."""
    with patch("sys.argv", ["demo", "--mock"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0


def test_demo_offline_fixtures_banner() -> None:
    """--offline-fixtures mode prints prominent notice banner."""
    def backend_a(q: str) -> str:
        return "tron movie"

    def backend_b(q: str) -> str:
        return "model context protocol"

    router = AdaptiveRouter(
        backends={"direct-http-tron": backend_a, "jina-mcp-wiki": backend_b},
        judge=_stub_judge_reject_a_accept_b,
    )

    buf = io.StringIO()
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), redirect_stdout(buf):
        exit_code = run_demo(offline_fixtures=True, fast=True, router=router)

    output = buf.getvalue()
    assert exit_code == 0
    assert "OFFLINE FIXTURE MODE: results are NOT live and scores are NOT from Gemma" in output


def test_demo_pass_only_on_real_reject_then_accept() -> None:
    """Demo returns 0 and PASS only when A is rejected and B is accepted."""
    def backend_a(q: str) -> str:
        return "tron movie"

    def backend_b(q: str) -> str:
        return "model context protocol"

    router = AdaptiveRouter(
        backends={"direct-http-tron": backend_a, "jina-mcp-wiki": backend_b},
        judge=_stub_judge_reject_a_accept_b,
    )

    buf = io.StringIO()
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), redirect_stdout(buf):
        exit_code = run_demo(fast=True, router=router)

    output = buf.getvalue()
    assert exit_code == 0
    assert "✓ Selected backend: jina-mcp-wiki" in output
    assert "✓ Adaptive fallback completed" in output
    assert "[PASS] Semantic fallback successful" in output


def test_demo_fails_when_a_accepted_on_first_attempt() -> None:
    """If A is accepted on first attempt, demo prints notice and returns exit code 1."""
    def backend_a(q: str) -> str:
        return "good content"

    router = AdaptiveRouter(
        backends={"direct-http-tron": backend_a},
        judge=lambda q, r: {
            "decision": "accept",
            "relevance": 0.9,
            "freshness": 0.9,
            "completeness": 0.9,
            "confidence": 0.9,
            "reason": "good",
        },
    )

    buf = io.StringIO()
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), redirect_stdout(buf):
        exit_code = run_demo(fast=True, router=router)

    output = buf.getvalue()
    assert exit_code == 1
    assert "Accepted on first backend (no fallback needed)" in output


def test_demo_fails_on_fetch_error() -> None:
    """Fetch error on Backend A causes demo to report fetch error and exit 1."""
    def crashing_a(q: str) -> str:
        raise RuntimeError("network down")

    def backend_b(q: str) -> str:
        return "mcp info"

    router = AdaptiveRouter(
        backends={"direct-http-tron": crashing_a, "jina-mcp-wiki": backend_b},
        judge=lambda q, r: {"decision": "accept", "score": 0.9},
    )

    buf = io.StringIO()
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), redirect_stdout(buf):
        exit_code = run_demo(fast=True, router=router)

    output = buf.getvalue()
    assert exit_code == 1
    assert "Fetch error on Backend A (this is not a semantic rejection)" in output


def test_demo_fails_on_judge_error() -> None:
    """Judge error causes demo to print judge unavailable and exit 1."""
    router = AdaptiveRouter(
        backends={"direct-http-tron": lambda q: "content"},
        judge=lambda q, r: {"judge_error": True, "reason": "rate limit"},
    )

    buf = io.StringIO()
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), redirect_stdout(buf):
        exit_code = run_demo(fast=True, router=router)

    output = buf.getvalue()
    assert exit_code == 1
    assert "Judge unavailable: rate limit" in output
