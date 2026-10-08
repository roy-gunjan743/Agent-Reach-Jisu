"""Tests for the reliability demo runner."""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout

from agent_reach.reliability.demo import run_demo


def test_demo_mock_fast_direct_call():
    """Verify demo returns 0 and produces expected reject -> fallback -> accept flow."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = run_demo(mock=True, fast=True)

    output = buf.getvalue()
    assert exit_code == 0
    assert "Backend A: direct-http-tron" in output
    assert "REJECT" in output
    assert "Fallback: Quality rejected" in output
    assert "Backend B: jina-mcp-wiki" in output
    assert "ACCEPT" in output
    assert "[PASS] Recovered via fallback" in output

    # Check order: REJECT appears before ACCEPT
    reject_idx = output.find("REJECT")
    accept_idx = output.find("ACCEPT")
    assert reject_idx != -1
    assert accept_idx != -1
    assert reject_idx < accept_idx


def test_demo_subprocess_execution():
    """Verify `python -m agent_reach.reliability.demo --mock --fast` runs cleanly as subprocess."""
    proc = subprocess.run(
        [sys.executable, "-m", "agent_reach.reliability.demo", "--mock", "--fast"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0
    output = proc.stdout
    assert "AGENT REACH: ADAPTIVE RELIABILITY LAYER DEMO" in output
    assert "Backend A: direct-http-tron" in output
    assert "REJECT" in output
    assert "Fallback:" in output
    assert "Backend B: jina-mcp-wiki" in output
    assert "ACCEPT" in output
    assert "jina-mcp-wiki" in output
    assert "Recovered via fallback in 2 attempts" in output
