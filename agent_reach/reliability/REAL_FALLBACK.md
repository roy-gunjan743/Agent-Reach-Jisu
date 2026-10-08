# Real Fallback — Adaptive Router

This document explains how to run the **real backend fallback** tests for the adaptive router.

## Architecture

```
Query → Backend A (Direct HTTP → Tron MCP article) → judge REJECT → fallback
      → Backend B (Jina Reader → Model Context Protocol article) → judge ACCEPT → final result
```

### Backend A — `direct-http-tron`
- **Executor**: `DirectHTTPExecutor` (stdlib `urllib`, no Jina proxy)
- **URL**: `https://en.wikipedia.org/wiki/MCP_(Tron)`
- **Why rejected**: The page is about the fictional Master Control Program from the 1982 film *Tron*. It is semantically irrelevant to the query about AI agent tool usage, so the judge rejects it.

### Backend B — `jina-mcp-wiki`
- **Executor**: `JinaExecutor` (Agent Reach's `WebChannel` → Jina Reader)
- **URL**: `https://en.wikipedia.org/wiki/Model_Context_Protocol`
- **Why accepted**: The page is the canonical Wikipedia article about the Model Context Protocol, directly answering the demo query.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | For live tests | API key for the Gemma judge via Gemini API |
| `RUN_LIVE` | For live pytest | Set to `1` to enable live tests in pytest |

If neither `GEMINI_API_KEY` nor a local Ollama instance is available, the live test will be skipped.

## Running Offline Tests (CI-safe)

No network or API key required. Uses fake backends and stub judges.

```bash
pytest tests/test_real_fallback.py -q
```

This runs all 14+ offline tests covering:
- Fallback when A is rejected
- B result content is returned
- Stats record success/failure/quality for both backends
- Backend exceptions fall back (no crash)
- Empty / whitespace results fall back
- If nothing passes threshold, best attempt is returned
- Deterministic A-then-B ordering
- Injectable judge overrides default
- `stats_summary()` structure
- `BackendResult` fields
- `build_real_router()` factory
- `record_failure(quality=...)` records quality
- `record_failure()` backward compatibility

## Running Live Tests (requires network + judge)

```bash
RUN_LIVE=1 GEMINI_API_KEY=<your-key> pytest tests/test_real_fallback.py -q
```

Or on Windows PowerShell:

```powershell
$env:RUN_LIVE="1"; $env:GEMINI_API_KEY="<your-key>"; uv run pytest tests/test_real_fallback.py -q
```

## Running the Live Script Directly

```bash
python -m agent_reach.reliability.test_real_fallback
```

Or with uv:

```bash
uv run python -m agent_reach.reliability.test_real_fallback
```

### Expected output (success)

```
============================================================
  REAL FALLBACK TEST — Adaptive Router
============================================================

Query: What is MCP and how does it improve AI agent tool usage?

🔎 Trying backend: direct-http-tron
   📄 Received ~100,000 characters
   Relevance:    0.15
   ...
   ⚠️ REJECTED: direct-http-tron
   ↳ Falling back to next backend...

🔎 Trying backend: jina-mcp-wiki
   📄 Received ~31,000 characters
   Relevance:    0.90
   ...
   ✅ ACCEPTED: jina-mcp-wiki

  Backend A rejected:      ✅
  Backend B accepted:      ✅
  Final backend is B:      ✅

✅ PASS — Real fallback worked as expected.
```

Exit code `0` on success, `1` on failure or missing judge.

## Old Simulated Test

The original simulated test still works:

```bash
python -m agent_reach.reliability.test
```
