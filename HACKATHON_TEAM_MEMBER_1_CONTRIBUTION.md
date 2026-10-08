# Agent Reach Hackathon Contribution

## Contributor Role

**Team Member 1 — Gemma semantic judge and reliability evaluation**

This document records only the work completed for my hackathon assignment.

## Assignment Scope

My assigned responsibility was to integrate Gemma 4 into Agent Reach as a
semantic judge for retrieved backend results. The judge evaluates whether a
backend result answers the user's query and returns a structured decision that
the reliability router can use.

The required public interface was preserved:

```python
judge_result(query, result)
```

## Work Completed

### 1. Replaced the previous judge implementation

Updated [`agent_reach/reliability/judge.py`](agent_reach/reliability/judge.py)
to use the official `google-genai` SDK and Gemma through the Gemini API.

The implementation:

- Uses `genai.Client(api_key=...)`.
- Uses `gemma-4-26b-a4b-it` by default.
- Loads local environment configuration safely.
- Sends the user query and backend result to Gemma.
- Requests structured JSON evaluation.
- Keeps the existing `judge_result(query, result)` interface.

### 2. Added reliability scoring

Gemma evaluates each backend result using:

- `relevance`
- `freshness`
- `completeness`
- `confidence`
- `decision`
- `reason`

Scores are validated to remain between `0` and `1`. The decision must be
either `accept` or `reject`.

### 3. Added robust failure handling

The judge now handles the following cases without crashing the router:

- Missing Gemini API key
- Invalid API key
- Gemini API errors
- Network and HTTP transport errors
- Request timeouts
- Empty API responses
- Malformed JSON
- JSON surrounded by Markdown fences or extra text
- Missing required evaluation fields
- Invalid score values
- Invalid decisions

Runtime failures are converted into a structured rejection so the adaptive
router can continue with fallback behavior.

### 4. Added configurable result size

The amount of backend content sent to Gemma can be configured with:

```text
AGENT_REACH_JUDGE_MAX_CHARS=12000
```

This prevents overly large web pages from being rejected only because the
judge received too little relevant context.

### 5. Added tests

Created [`tests/test_reliability_judge.py`](tests/test_reliability_judge.py)
with coverage for:

- Accepted evaluations
- Rejected evaluations
- Fenced JSON responses
- Malformed JSON
- Missing API key
- Gemini API errors
- HTTP transport errors
- Timeout handling

Run `python -m pytest -q` for current results.

### 6. Updated documentation and configuration templates

Updated:

- [`README.md`](README.md)
- [`.env.example`](.env.example)
- [`pyproject.toml`](pyproject.toml)
- [`agent_reach/reliability/test_real_router.py`](agent_reach/reliability/test_real_router.py)

The dependency manifest now includes:

```toml
"google-genai>=1.0",
```

The configuration template documents the Gemma API key, model, and result
length settings without containing a real secret.

### 7. Verified the real Gemma demo

The real judge-only demo was executed successfully using the local ignored
`.env` configuration.

The result was:

```text
Relevance:    1.00
Freshness:    1.00
Completeness: 1.00
Confidence:   1.00
Decision:     accept
Reason:       Directly answers both parts of the query.
```

This verified that the official SDK integration, Gemma model configuration,
prompt, response parsing, and evaluation output work together end to end.

## Integration With the Team Project

The Gemma judge is designed to work with the adaptive router:

```text
Backend result
    ↓
Gemma semantic evaluation
    ↓
accept → return result
reject → try fallback backend
```

The fallback architecture and backend implementations were synchronized from
the shared repository before the Gemma contribution was committed. My work
does not replace the backend or router ownership of other team members.

## Git and Repository Status

The completed contribution was committed on the team repository branch.

It was pushed to the team repository:

```text
https://github.com/roy-gunjan743/Agent-Reach-Jisu
```

The repository branch was verified to include the Gemma judge integration.

## Security Note

The actual Gemini API key is not included in this document, source code,
`.env.example`, or Git history. The local `.env` file is ignored by Git.

Any API key previously shared in chat or exposed publicly should be revoked
and replaced. The replacement key should remain only in the local ignored
`.env` file.

## Final Status

The Team Member 1 hackathon assignment is complete:

- Gemma 4 semantic judging is implemented.
- Reliability scores and decisions are validated.
- API and parsing failures are handled safely.
- Automated tests pass.
- The real Gemma demo succeeds.
- Documentation and configuration templates are updated.
- The completed work is committed and pushed to the team repository.
