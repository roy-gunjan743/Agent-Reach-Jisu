# Agent Reach — Run and Test Guide

This guide explains how to run the complete Agent Reach project and how to
reproduce the adaptive reliability issue solved for the hackathon.

The reliability demo uses:

```text
Backend A → normalization → Gemma semantic judge
                         ↓ reject
Backend B → normalization → Gemma semantic judge
                         ↓ accept
                       final result
```

## 1. Requirements

- Windows PowerShell, macOS, or Linux
- Python 3.10 or newer
- Internet access for live backend and Gemini API tests
- A Gemini API key for Gemma live evaluation

The project uses the official `google-genai` SDK and the model:

```text
gemma-4-26b-a4b-it
```

Never commit or paste the real API key into source code, documentation, or
Git. Use a local ignored `.env` file.

## 2. Create and activate a virtual environment

### Windows PowerShell

```powershell
py -3.11 -m venv agent_reach_venv
.\agent_reach_venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the interpreter directly as shown in the
commands below.

### macOS/Linux

```bash
python3 -m venv agent_reach_venv
source agent_reach_venv/bin/activate
```

## 3. Install the project

Run this from the repository root:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, if `python` is not the virtual-environment interpreter, use:

```powershell
& ".\agent_reach_venv\Scripts\python.exe" -m pip install -e ".[dev]"
```

The installation includes the project dependencies, `pytest`, `rich`,
`google-genai`, `httpx`, and the other packages declared in
[`pyproject.toml`](pyproject.toml).

## 4. Configure the Gemma judge

Copy the safe template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set the key privately:

```text
GEMINI_API_KEY=your-private-gemini-api-key
AGENT_REACH_JUDGE_MODEL=gemma-4-26b-a4b-it
AGENT_REACH_JUDGE_MAX_CHARS=12000
```

The `.env` file must remain local and ignored by Git. Do not print the key
while troubleshooting.

## 5. Run the complete hackathon demo

### Windows PowerShell

```powershell
$env:PYTHONIOENCODING = "utf-8"

& ".\agent_reach_venv\Scripts\python.exe" `
  -m agent_reach.reliability.demo --fast
```

### macOS/Linux

```bash
python -m agent_reach.reliability.demo --fast
```

The expected behavior is:

```text
direct-http-tron → Gemma rejects the irrelevant Tron result
                 → router falls back
jina-mcp-wiki    → Gemma accepts the correct MCP result

[PASS] Semantic fallback successful
```

The successful result should include:

```text
Selected backend: jina-mcp-wiki
Adaptive fallback completed
```

## 6. Run the explicit real fallback verification

This command prints backend statistics and assertions for both backends:

```powershell
$env:PYTHONIOENCODING = "utf-8"

& ".\agent_reach_venv\Scripts\python.exe" `
  -m agent_reach.reliability.test_real_fallback
```

Expected assertions:

```text
Backend A rejected: ✅
Backend B accepted: ✅
Final backend is B: ✅

✅ PASS — Real fallback worked as expected.
```

## 7. Run the normal judge-only demo

This demonstrates one real backend result and the Gemma evaluation:

```powershell
$env:PYTHONIOENCODING = "utf-8"

& ".\agent_reach_venv\Scripts\python.exe" `
  -m agent_reach.reliability.test_real_router
```

Expected output includes relevance, freshness, completeness, confidence,
decision, and reason.

## 8. Run an offline fixture demo

The fixture mode avoids live web fetching but still requires
`GEMINI_API_KEY` because Gemma evaluates the fixture content:

```powershell
& ".\agent_reach_venv\Scripts\python.exe" `
  -m agent_reach.reliability.demo --offline-fixtures --fast
```

This is useful when testing normalization and router behavior without relying
on the live web backends.

## 9. Run tests

### Full project test suite

```powershell
& ".\agent_reach_venv\Scripts\python.exe" -m pytest -q
```

### Team Member 1 Gemma judge tests

```powershell
& ".\agent_reach_venv\Scripts\python.exe" `
  -m pytest tests/test_reliability_judge.py -q
```

### Reliability and fallback tests

```powershell
& ".\agent_reach_venv\Scripts\python.exe" `
  -m pytest `
  tests/reliability/test_normalizer.py `
  tests/reliability/test_demo.py `
  tests/test_real_fallback.py `
  tests/test_reliability_judge.py -q
```

### Live pytest tests

Live tests are normally skipped unless explicitly enabled:

```powershell
$env:RUN_LIVE = "1"

& ".\agent_reach_venv\Scripts\python.exe" `
  -m pytest tests/test_real_fallback.py -q
```

## 10. Run the full Agent Reach CLI

After installing the project:

```powershell
& ".\agent_reach_venv\Scripts\agent-reach.exe" --help
```

Useful project commands include:

```powershell
& ".\agent_reach_venv\Scripts\agent-reach.exe" doctor
& ".\agent_reach_venv\Scripts\agent-reach.exe" configure
```

The CLI provides the broader Agent Reach functionality for web channels and
diagnostics. The Gemma reliability demo is run through the Python module
commands above.

## 11. What the solved issue demonstrates

The original reliability problem was that a backend could return a page that
matched the keyword `MCP` but did not answer the actual question.

For the demo query:

```text
What is MCP and how does it improve AI agent tool usage?
```

Backend A returns the fictional *Tron* MCP page. Gemma recognizes that it is
semantically wrong and rejects it.

The router then tries Backend B, which returns the Model Context Protocol
article. Gemma evaluates it as relevant and accepts it.

This prevents Agent Reach from blindly trusting the first backend result.

## 12. Main implementation files

- [`agent_reach/reliability/judge.py`](agent_reach/reliability/judge.py) —
  Gemma semantic judge and failure handling
- [`agent_reach/reliability/router.py`](agent_reach/reliability/router.py) —
  adaptive backend selection and fallback
- [`agent_reach/reliability/backends.py`](agent_reach/reliability/backends.py) —
  real demo backend definitions
- [`agent_reach/reliability/normalizer.py`](agent_reach/reliability/normalizer.py) —
  content cleanup and relevant excerpt selection
- [`agent_reach/reliability/demo.py`](agent_reach/reliability/demo.py) —
  primary live demo
- [`agent_reach/reliability/test_real_fallback.py`](agent_reach/reliability/test_real_fallback.py) —
  explicit live fallback verification
- [`tests/test_reliability_judge.py`](tests/test_reliability_judge.py) —
  Gemma judge tests
- [`HACKATHON_TEAM_MEMBER_1_CONTRIBUTION.md`](HACKATHON_TEAM_MEMBER_1_CONTRIBUTION.md) —
  Team Member 1 contribution record

## 13. Troubleshooting

### `ModuleNotFoundError`

Install the project into the active virtual environment:

```powershell
& ".\agent_reach_venv\Scripts\python.exe" -m pip install -e ".[dev]"
```

### `GEMINI_API_KEY environment variable is missing`

Check that `.env` exists in the repository root and contains a private key:

```text
GEMINI_API_KEY=your-private-gemini-api-key
```

### API key or Gemini API errors

- Confirm the key is valid in Google AI Studio.
- Confirm the key has access to the Gemini API.
- Do not include quotes or spaces around the value.
- Revoke any key that was exposed publicly and create a replacement.

### Windows console encoding errors

Run the demo with:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### Git pull from the team repository

The team remote is named `jisu`:

```powershell
git pull jisu main
```

This guide itself does not change the branch tracking configuration.
