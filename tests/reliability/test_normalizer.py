"""Tests for the content-aware normalizer module."""

from __future__ import annotations

import os

import pytest

from agent_reach.reliability.normalizer import NormalizedResult, normalize


@pytest.fixture
def sample_30k_jina_text() -> str:
    """Fixture with nav junk, real article content, and footer boilerplate."""
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "agent_reach",
        "reliability",
        "demo_data",
        "backend_b_mcp.txt",
    )
    if os.path.exists(fixture_path):
        with open(fixture_path, "r", encoding="utf-8") as f:
            return f.read()

    # Fallback synthetic generator if fixture is absent
    head = (
        "Title: Model Context Protocol - Wikipedia\n"
        "URL Source: https://en.wikipedia.org/wiki/Model_Context_Protocol\n"
        "Markdown Content:\n\n"
        "# Model Context Protocol\n\n"
        "The Model Context Protocol (MCP) connects AI models and agents to external tools.\n\n"
    )
    body = "\n\n".join(
        f"Section {i}: JSON-RPC 2.0 messages enable tools, resources, and prompt templates."
        for i in range(400)
    )
    foot = "\n\nWe use cookies. Subscribe to our newsletter. All rights reserved."
    return head + body + foot


def test_empty_and_whitespace_input():
    res = normalize("")
    assert res.text == ""
    assert res.original_chars == 0
    assert res.final_chars == 0
    assert res.reduction_pct == 0.0
    assert not res.truncated
    assert not res.metadata["looks_like_error"]

    res_ws = normalize("   \n\t  \n")
    assert res_ws.text == ""
    assert not res_ws.truncated


def test_non_string_input():
    res = normalize(None)  # type: ignore[arg-type]
    assert res.text == ""

    res_num = normalize(12345)  # type: ignore[arg-type]
    assert "12345" in res_num.text


def test_short_input_passed_through():
    short_text = "# Test Title\n\nThis is a short article that easily fits under the limit."
    res = normalize(short_text, max_chars=1000)
    assert not res.truncated
    assert "Test Title" in res.text
    assert "short article" in res.text
    assert res.final_chars <= 1000


def test_error_page_detection():
    error_403 = "403 Forbidden - You do not have permission to view this directory."
    res = normalize(error_403)
    assert res.metadata["looks_like_error"] is True

    cloudflare_block = "Attention Required! | Cloudflare - Please verify you are a human."
    res_cf = normalize(cloudflare_block)
    assert res_cf.metadata["looks_like_error"] is True

    normal_article = "# Artificial Intelligence\n\nAI is transforming software engineering."
    res_normal = normalize(normal_article)
    assert res_normal.metadata["looks_like_error"] is False


def test_reduction_and_relevance(sample_30k_jina_text: str):
    query = "What is MCP and how does it improve AI agent tool usage?"
    max_chars = 4000
    res = normalize(sample_30k_jina_text, query=query, max_chars=max_chars)

    assert isinstance(res, NormalizedResult)
    assert res.original_chars >= 25000
    assert res.final_chars <= max_chars
    assert res.truncated is True
    assert res.reduction_pct > 70.0

    # Ensure key query terms remain present
    lower_text = res.text.lower()
    assert "model context protocol" in lower_text or "mcp" in lower_text
    assert "tool" in lower_text
    assert "agent" in lower_text


def test_order_preserved_and_title_retained(sample_30k_jina_text: str):
    query = "MCP client server tools resources"
    res = normalize(sample_30k_jina_text, query=query, max_chars=3500)

    # Title retained
    assert "Model Context Protocol" in res.text
    assert res.metadata.get("title") is not None

    # First intro paragraph must appear before subsequent sections
    intro_marker = "open standard introduced by anthropic"
    arch_marker = "overview and architecture"

    lower = res.text.lower()
    assert intro_marker in lower
    if arch_marker in lower:
        assert lower.index(intro_marker) < lower.index(arch_marker)


def test_drops_wikipedia_front_matter_and_reference_blocks():
    raw = (
        "Title: Model Context Protocol\n"
        "URL Source: https://example.com/mcp\n"
        "Markdown Content:\n\n"
        "From Wikipedia, the free encyclopedia\n\n"
        "The Model Context Protocol is an open standard that connects AI agents "
        "to external tools and data sources.\n\n"
        "1. [Reference one](https://example.com/one). Retrieved 2025-01-01.\n"
        "2. [Reference two](https://example.com/two). Retrieved 2025-01-01.\n"
        "3. [Reference three](https://example.com/three). Retrieved 2025-01-01.\n"
    )

    result = normalize(raw, query="How does MCP connect AI agents to tools?", max_chars=250)

    assert "connects AI agents to external tools" in result.text
    assert "Reference one" not in result.text


def test_no_mid_word_cuts():
    long_words = " ".join([f"unbreakablewordidentifier{i}" for i in range(200)])
    text = f"# Long Document\n\n{long_words}"
    res = normalize(text, max_chars=250)

    assert res.final_chars <= 250
    # Must end with marker if cut
    if res.truncated:
        assert res.text.endswith(" […]")
        # Check that preceding token isn't a sliced partial fragment like 'unbreak'
        without_marker = res.text[:-4].strip()
        last_word = without_marker.split()[-1]
        assert last_word.startswith("unbreakablewordidentifier")
        assert last_word in long_words


def test_strips_html_images_and_boilerplate():
    raw = (
        "Title: Clean Tech Page\n"
        "URL Source: https://example.com/clean\n"
        "Markdown Content:\n"
        "<script>window.analytics.track();</script>\n"
        "<style>.banner { color: red; }</style>\n"
        "![Banner Image](https://example.com/banner.png)\n"
        "We use cookies on this site. Accept Cookies\n"
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA\n"
        "# Clean Heading\n\n"
        "This is valid content describing agent tools.\n\n"
        "Subscribe to our newsletter\n"
        "All rights reserved © 2026\n"
    )
    res = normalize(raw, max_chars=2000)
    assert "<script>" not in res.text
    assert "<style>" not in res.text
    assert "data:image" not in res.text
    assert "![Banner Image]" not in res.text
    assert "Accept Cookies" not in res.text
    assert "Subscribe to our newsletter" not in res.text
    assert "This is valid content describing agent tools." in res.text
