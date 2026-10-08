"""Autonomous Reliability Layer Demo for Agent Reach.

Demonstrates content-aware normalization, semantic evaluation via Gemma,
and automated multi-backend fallback.

Usage:
    python -m agent_reach.reliability.demo [--mock] [--live] [--fast] [--query "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from typing import Any

from agent_reach.reliability.backends import (
    DEMO_QUERY,
    backend_a,
    backend_b,
)
from agent_reach.reliability.normalizer import NormalizedResult, normalize
from agent_reach.reliability.scorer import BackendStats

# Ensure Windows console handles UTF-8 / symbols safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Styling and ANSI formatting
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

C_RESET = "\033[0m" if _USE_COLOR else ""
C_BOLD = "\033[1m" if _USE_COLOR else ""
C_DIM = "\033[2m" if _USE_COLOR else ""
C_GREEN = "\033[1;32m" if _USE_COLOR else ""
C_RED = "\033[1;31m" if _USE_COLOR else ""
C_CYAN = "\033[1;36m" if _USE_COLOR else ""
C_YELLOW = "\033[1;33m" if _USE_COLOR else ""


# ---------------------------------------------------------------------------
# Preflight and Gemma / Ollama Client
# ---------------------------------------------------------------------------
DEFAULT_GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma2:2b")
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def preflight_check(model: str = DEFAULT_GEMMA_MODEL, timeout: float = 0.8) -> tuple[bool, str]:
    """Check if local Ollama server is running and inspect available models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            base_model = model.split(":")[0].lower()
            matching = [m for m in models if base_model in m.lower()]
            if matching:
                return True, f"Online ({matching[0]})"
            if models:
                return True, f"Online ({models[0]} fallback)"
            return True, "Online (Ollama ready)"
    except Exception:
        return False, "Ollama offline @ localhost:11434"


def _parse_gemma_json(raw_resp: str) -> dict[str, Any]:
    """Defensively parse JSON from Gemma LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_resp)
    cleaned = cleaned.replace("```", "").strip()

    # Attempt direct JSON decode
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract outermost JSON object via regex
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Fallback keyword extraction
    lower = cleaned.lower()
    if "accept" in lower and "reject" not in lower:
        return {"decision": "accept", "reason": "Keyword accept detected"}
    return {"decision": "reject", "reason": "Failed to parse JSON response"}


def evaluate_with_gemma(
    query: str,
    normalized: NormalizedResult,
    *,
    mock: bool = False,
    model: str = DEFAULT_GEMMA_MODEL,
    timeout: float = 12.0,
) -> tuple[dict[str, Any], float]:
    """Evaluate normalized content using Gemma or deterministic mock evaluator."""
    t0 = time.perf_counter()

    if mock:
        # Realistic mock evaluation: deterministic, fast, zero network dependencies
        text_lower = normalized.text.lower()
        if "tron" in text_lower or "fictional" in text_lower or "game" in text_lower or "villain" in text_lower:
            decision = "reject"
            reason = "Discusses fictional Tron 1982 film, not AI tool protocols"
            score = 0.15
        else:
            decision = "accept"
            reason = "Specifies Model Context Protocol architecture & agent tools"
            score = 0.92

        latency_ms = (time.perf_counter() - t0) * 1000 + 4.5
        eval_dict = {
            "decision": decision,
            "verdict": decision.upper(),
            "reason": reason,
            "score": score,
            "relevance": score,
        }
        return eval_dict, latency_ms

    # Live Ollama call
    prompt = f"""You are an AI quality evaluator for a web research agent.
Evaluate if the following web excerpt answers the query.

QUERY:
{query}

CONTENT:
{normalized.text[:2500]}

Respond ONLY with this JSON:
{{
  "decision": "accept" or "reject",
  "reason": "under 10 words reason"
}}
"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 120},
    }

    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            parsed = _parse_gemma_json(data.get("response", ""))
            decision = parsed.get("decision", "reject").lower()
            reason = parsed.get("reason", "Live evaluation completed")
            score = 0.90 if decision == "accept" else 0.20
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "decision": decision,
                "verdict": decision.upper(),
                "reason": reason,
                "score": score,
            }, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "decision": "reject",
            "verdict": "REJECT",
            "reason": f"Ollama error: {str(exc)[:40]}",
            "score": 0.0,
        }, latency_ms


# ---------------------------------------------------------------------------
# Fixture fallback loader
# ---------------------------------------------------------------------------
def _load_demo_fixture(name: str) -> str:
    """Load sample fixture from reliability/demo_data/ directory."""
    demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
    path = os.path.join(demo_dir, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    # Minimal inline fallback if file missing
    if "tron" in name:
        return "MCP Master Control Program fictional rogue AI from Tron 1982 film."
    return "# Model Context Protocol\n\nMCP is an open standard for AI agent tools by Anthropic."


# ---------------------------------------------------------------------------
# Demo Runner Pipeline
# ---------------------------------------------------------------------------
def run_demo(
    query: str = DEMO_QUERY,
    mock: bool | None = None,
    live_only: bool = False,
    fast: bool = False,
    max_chars: int = 4000,
) -> int:
    """Run the complete reliability fallback demonstration."""
    start_total_time = time.perf_counter()
    pause_sec = 0.0 if fast else 0.45

    print(f"\n{C_BOLD}{C_CYAN}==================================================================={C_RESET}")
    print(f"{C_BOLD}{C_CYAN}         AGENT REACH: ADAPTIVE RELIABILITY LAYER DEMO             {C_RESET}")
    print(f"{C_BOLD}{C_CYAN}==================================================================={C_RESET}")

    # Preflight check
    ollama_ok, ollama_status = preflight_check()
    effective_mock = mock

    if live_only and not ollama_ok:
        print(f"{C_RED}[!] --live specified but Gemma/Ollama is offline: {ollama_status}{C_RESET}")
        print("   Please start Ollama (`ollama serve`) and pull a model, or omit `--live`.")
        return 1

    if effective_mock is None:
        effective_mock = not ollama_ok

    status_str = (
        f"{C_GREEN}[Live]{C_RESET} ({ollama_status})"
        if not effective_mock
        else f"{C_YELLOW}[Mock Active]{C_RESET} (Deterministic evaluation)"
    )
    print(f"{C_DIM}Environment:{C_RESET} {status_str}")
    print(f"{C_BOLD}Query:{C_RESET}       {query}")
    print(f"{C_CYAN}-------------------------------------------------------------------{C_RESET}")

    time.sleep(pause_sec)

    backends_plan = [
        ("direct-http-tron", "Backend A", backend_a, "backend_a_tron.txt"),
        ("jina-mcp-wiki", "Backend B", backend_b, "backend_b_mcp.txt"),
    ]

    stats = {name: BackendStats(name) for name, _, _, _ in backends_plan}
    accepted_result: NormalizedResult | None = None
    accepted_backend: str | None = None
    attempts_count = 0

    for idx, (backend_id, label, callable_fn, fixture_file) in enumerate(backends_plan, 1):
        attempts_count += 1
        print(f"\n{C_BOLD}[{idx}/{len(backends_plan)}] Trying {label}: {backend_id}{C_RESET}")

        # 1. Fetch raw data
        raw_text = ""
        source_note = ""
        if effective_mock:
            raw_text = _load_demo_fixture(fixture_file)
            source_note = " (bundled fixture)"
        else:
            try:
                raw_text = callable_fn(query)
            except Exception as exc:
                raw_text = _load_demo_fixture(fixture_file)
                source_note = f" (fallback fixture: {str(exc)[:30]})"

        # 2. Content-Aware Normalizer
        t_norm_0 = time.perf_counter()
        normalized = normalize(raw_text, query=query, max_chars=max_chars, source=backend_id)
        norm_ms = (time.perf_counter() - t_norm_0) * 1000

        print(
            f"  {C_CYAN}|-> Result:{C_RESET} {normalized.original_chars:,} chars -> "
            f"{C_BOLD}{normalized.final_chars:,} chars{C_RESET} "
            f"({normalized.reduction_pct}% reduction in {norm_ms:.1f}ms){source_note}"
        )

        time.sleep(pause_sec)

        # 3. Gemma Semantic Evaluation
        eval_dict, eval_ms = evaluate_with_gemma(
            query=query,
            normalized=normalized,
            mock=effective_mock,
        )

        decision = eval_dict.get("decision", "reject").lower()
        reason = eval_dict.get("reason", "")
        score = eval_dict.get("score", 0.0)

        if decision == "accept":
            verdict_badge = f"{C_GREEN}ACCEPT{C_RESET}"
            stats[backend_id].record_success(score)
            print(f"  {C_CYAN}|-> Gemma Evaluation:{C_RESET} {verdict_badge} [Score: {score:.2f}]")
            print(f"      {C_DIM}Reason:{C_RESET} {reason} {C_DIM}({eval_ms:.0f}ms){C_RESET}")
            accepted_result = normalized
            accepted_backend = backend_id
            break
        else:
            verdict_badge = f"{C_RED}REJECT{C_RESET}"
            stats[backend_id].record_failure(score)
            print(f"  {C_CYAN}|-> Gemma Evaluation:{C_RESET} {verdict_badge} [Score: {score:.2f}]")
            print(f"      {C_DIM}Reason:{C_RESET} {reason} {C_DIM}({eval_ms:.0f}ms){C_RESET}")
            print(f"  {C_YELLOW}    Fallback: Quality rejected -> attempting next backend...{C_RESET}")
            time.sleep(pause_sec)

    # 4. Final summary block
    print(f"\n{C_BOLD}{C_CYAN}==================================================================={C_RESET}")
    print(f"{C_BOLD}FINAL RESULT{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}==================================================================={C_RESET}")

    total_pipeline_sec = time.perf_counter() - start_total_time

    if accepted_result and accepted_backend:
        preview = " ".join(accepted_result.text.split())[:360]
        title_str = accepted_result.metadata.get("title") or "Article Content"
        print(f"Accepted Backend: {C_BOLD}{accepted_backend}{C_RESET}")
        print(f"Document Title:   {title_str}")
        print(f"Normalized Text:  {preview}...")
        print(f"{C_CYAN}-------------------------------------------------------------------{C_RESET}")
        print(
            f"{C_GREEN}[PASS] Recovered via fallback in {attempts_count} attempts, "
            f"{accepted_result.original_chars:,} -> {accepted_result.final_chars:,} chars "
            f"({accepted_result.reduction_pct}% smaller).{C_RESET}"
        )
        print(f"{C_DIM}Pipeline execution completed in {total_pipeline_sec:.2f}s.{C_RESET}\n")
        return 0

    print(f"{C_RED}[FAIL] No backend result accepted after {attempts_count} attempts.{C_RESET}\n")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Reach Adaptive Router & Normalizer Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=DEMO_QUERY, help="Query to run")
    parser.add_argument("--mock", action="store_true", help="Force mock evaluator and offline fixtures")
    parser.add_argument("--live", action="store_true", help="Force live Gemma/Ollama path")
    parser.add_argument("--fast", action="store_true", help="Disable pacing delays between steps")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max characters for normalized output")

    args = parser.parse_args()

    mock_mode = True if args.mock else (False if args.live else None)
    exit_code = run_demo(
        query=args.query,
        mock=mock_mode,
        live_only=args.live,
        fast=args.fast,
        max_chars=args.max_chars,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
