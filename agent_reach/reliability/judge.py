import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"

MAX_RESULT_CHARS = 3000


def judge_result(query: str, result: str) -> dict:
    result_excerpt = result[:MAX_RESULT_CHARS]

    prompt = f"""You are evaluating a web result.

QUERY:
{query}

RESULT:
{result_excerpt}

Return ONLY this JSON object:

{{
  "relevance": 0.0,
  "freshness": 0.0,
  "completeness": 0.0,
  "confidence": 0.0,
  "decision": "accept",
  "reason": "short"
}}

Scores must be between 0 and 1.
Use "accept" only when the result answers the query.
Keep reason under 10 words.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 150,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read())

    raw = data["response"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("\n⚠️ Qwen returned invalid JSON:")
        print(raw)
        raise