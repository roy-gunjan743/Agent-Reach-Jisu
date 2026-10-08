# Reliability Layer Demo — Judge Quickstart

Run the self-contained fallback demo in one command:

```bash
python -m agent_reach.reliability.demo
```

### Pre-Demo Checklist (Optional — for live Gemma evaluation)
1. Start Ollama: `ollama serve`
2. Pull the model: `ollama pull gemma2:2b` (or set `GEMMA_MODEL=...`)

### Demo Execution Flags
- **Zero-Setup / Projector Mode**: `python -m agent_reach.reliability.demo --mock --fast`
- **Force Live LLM**: `python -m agent_reach.reliability.demo --live`
- **Custom Query**: `python -m agent_reach.reliability.demo --query "What is MCP?"`
- **Custom Output Limit**: `python -m agent_reach.reliability.demo --max-chars 3000`

### Resilience Guarantee
If Ollama is offline or network fails, the demo auto-switches to mock evaluation and bundled fixtures, guaranteeing a zero-crash presentation.
