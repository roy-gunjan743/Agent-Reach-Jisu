# Reliability Layer Demo — Judge Quickstart

Run the adaptive router fallback demo in one command:

```bash
python -m agent_reach.reliability.demo
```

### Environment Requirements

Set your Gemini API key in your `.env` file or environment before running:

```bash
export GEMINI_API_KEY="your-gemini-api-key"   # Linux/macOS
$env:GEMINI_API_KEY="your-gemini-api-key"     # PowerShell
set GEMINI_API_KEY=your-gemini-api-key        # Windows CMD
```

### Demo Execution Flags
- **Offline Fixtures**: `python -m agent_reach.reliability.demo --offline-fixtures` (uses static text fixtures, requires `GEMINI_API_KEY` for Gemma evaluation)
- **Fast Mode**: `python -m agent_reach.reliability.demo --fast` (disables pacing delays between pipeline steps)
- **Custom Query**: `python -m agent_reach.reliability.demo --query "What is MCP?"`
- **Custom Output Limit**: `python -m agent_reach.reliability.demo --max-chars 3000` (limits normalized excerpt length sent to judge)

### Real Evaluation Pipeline
The demo uses real web backends (Backend A: Direct HTTP Tron article, Backend B: Jina Reader MCP article), passes content through `agent_reach.reliability.normalizer`, and evaluates results using Gemma 4 via the Gemini API. All reported scores, metrics, decisions, and fallback decisions come directly from the live Gemma judge.
