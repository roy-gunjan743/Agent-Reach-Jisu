import json
import os
import time

import httpx
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


def load_env() -> None:
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)
    load_dotenv()


load_env()

GEMINI_API_URL = os.getenv(
    "AGENT_REACH_GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
MODEL = os.getenv("AGENT_REACH_JUDGE_MODEL", "gemma-4-26b-a4b-it")

MAX_RESULT_CHARS = int(os.getenv("AGENT_REACH_JUDGE_MAX_CHARS", "12000"))
SCORE_FIELDS = ("relevance", "freshness", "completeness", "confidence")


class JudgeError(RuntimeError):
    """Raised when the local semantic judge cannot return a valid evaluation."""


def _rejected_evaluation(reason: str) -> dict:
    return {
        "judge_error": True,
        "relevance": 0.0,
        "freshness": 0.0,
        "completeness": 0.0,
        "confidence": 0.0,
        "decision": "reject",
        "reason": reason,
    }


def _validate_evaluation(evaluation: object) -> dict:
    if not isinstance(evaluation, dict):
        raise JudgeError("Gemma returned a JSON value instead of an object")

    missing = [field for field in SCORE_FIELDS if field not in evaluation]
    if missing:
        raise JudgeError(f"Gemma evaluation is missing fields: {', '.join(missing)}")

    for field in SCORE_FIELDS:
        value = evaluation[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JudgeError(f"Gemma field '{field}' must be a number")
        if not 0 <= value <= 1:
            raise JudgeError(f"Gemma field '{field}' must be between 0 and 1")

    if evaluation.get("decision") not in {"accept", "reject"}:
        raise JudgeError("Gemma field 'decision' must be 'accept' or 'reject'")

    if not isinstance(evaluation.get("reason"), str):
        raise JudgeError("Gemma field 'reason' must be a string")

    return evaluation


def _parse_evaluation(raw: str) -> dict:
    candidates = [raw.strip()]
    if "```" in raw:
        candidates.append(
            raw.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
        )

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            evaluation = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return _validate_evaluation(evaluation)

    raise JudgeError("Gemma returned malformed JSON")


def judge_result(query: str, result: str) -> dict:
    load_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _rejected_evaluation("Gemini API key is missing")

    result_excerpt = result[:MAX_RESULT_CHARS]

    prompt = f"""You are evaluating a web result.

QUERY:
{query}

RESULT EXCERPT (the source may be longer; do not penalize it solely because
this excerpt is bounded):
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

    timeout_ms = int(os.getenv("AGENT_REACH_JUDGE_TIMEOUT_MS", "30000"))
    max_retries = int(os.getenv("AGENT_REACH_JUDGE_RETRIES", "1"))
    backoff_sec = float(os.getenv("AGENT_REACH_JUDGE_RETRY_BACKOFF_SEC", "2.0"))

    response = None
    for attempt in range(1 + max_retries):
        try:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            break
        except TimeoutError:
            if attempt < max_retries:
                time.sleep(backoff_sec)
                continue
            return _rejected_evaluation("Gemini API request timed out")
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            should_retry = code == 429 or (isinstance(code, int) and code >= 500)
            if should_retry and attempt < max_retries:
                time.sleep(backoff_sec)
                continue
            return _rejected_evaluation(f"Gemini API error: {exc}")
        except (httpx.HTTPError, OSError) as exc:
            if attempt < max_retries:
                time.sleep(backoff_sec)
                continue
            return _rejected_evaluation(f"Gemini API connection failed: {exc}")

    if response is None:
        return _rejected_evaluation("Gemma API returned no response")

    raw = response.text
    if not raw:
        return _rejected_evaluation("Gemma API response contained no generated text")

    try:
        return _parse_evaluation(raw)
    except JudgeError as exc:
        return _rejected_evaluation(str(exc))