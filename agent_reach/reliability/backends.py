"""Real backend wiring for the adaptive router.

Exposes ``build_real_router()`` so tests, demos, and teammates can
instantiate a router with two real backends and a reproducible query.

Backend A  — "direct-http-tron"
    Fetches https://en.wikipedia.org/wiki/MCP_(Tron) via ``DirectHTTPExecutor``
    (raw stdlib urllib, no Jina proxy).  The page is about the fictional Master
    Control Program from the 1982 film *Tron* — semantically wrong for an AI /
    protocol query, so the judge should reject it.

Backend B  — "jina-mcp-wiki"
    Fetches https://en.wikipedia.org/wiki/Model_Context_Protocol via
    ``JinaExecutor`` (Jina Reader → clean Markdown).  This is the canonical
    article about the Model Context Protocol, directly answering the demo query.
"""

from __future__ import annotations

from typing import Callable

from agent_reach.reliability.executor import DirectHTTPExecutor, JinaExecutor
from agent_reach.reliability.router import AdaptiveRouter

# ---------------------------------------------------------------------------
# Demo query
# ---------------------------------------------------------------------------
DEMO_QUERY = "What is MCP and how does it improve AI agent tool usage?"

# ---------------------------------------------------------------------------
# Pinned target URLs
# ---------------------------------------------------------------------------
BACKEND_A_URL = "https://en.wikipedia.org/wiki/MCP_(Tron)"
BACKEND_B_URL = "https://en.wikipedia.org/wiki/Model_Context_Protocol"

# ---------------------------------------------------------------------------
# Backend callables  (query: str) -> str
# ---------------------------------------------------------------------------
_jina = JinaExecutor()
_direct = DirectHTTPExecutor()


def backend_a(query: str) -> str:
    """Direct HTTP fetch of the Tron MCP article (expected: reject)."""
    return _direct.fetch(BACKEND_A_URL)


def backend_b(query: str) -> str:
    """Jina Reader fetch of the real MCP Wikipedia article (expected: accept)."""
    return _jina.fetch(BACKEND_B_URL)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def build_real_router(
    judge: Callable[[str, str], dict] | None = None,
) -> AdaptiveRouter:
    """Build a router with two real backends wired A-first, B-second.

    Parameters
    ----------
    judge:
        Optional judge callable ``(query, result) -> dict``.
        When *None* the router lazily imports ``judge_result`` from
        ``agent_reach.reliability.judge`` (i.e. the Gemma/Gemini judge).
    """
    # Dict insertion order controls deterministic A→B fallback priority.
    backends: dict[str, Callable[[str], str]] = {
        "direct-http-tron": backend_a,
        "jina-mcp-wiki": backend_b,
    }
    return AdaptiveRouter(backends=backends, judge=judge)
