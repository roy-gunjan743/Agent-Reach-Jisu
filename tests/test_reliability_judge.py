import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google.genai import errors as genai_errors

from agent_reach.reliability.judge import judge_result


def mock_client(response_text: str):
    return SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text=response_text)
        )
    )


def valid_evaluation() -> str:
    return json.dumps(
        {
            "relevance": 0.9,
            "freshness": 0.8,
            "completeness": 0.9,
            "confidence": 0.8,
            "decision": "accept",
            "reason": "Directly answers the query",
        }
    )


def test_judge_accepts_valid_gemma_evaluation():
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        return_value=mock_client(valid_evaluation()),
    ) as client_factory, patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("What is MCP?", "MCP connects agents to tools.")

    assert evaluation["decision"] == "accept"
    client_factory.assert_called_once_with(api_key="test-key")


def test_judge_accepts_fenced_json():
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        return_value=mock_client(f"```json\n{valid_evaluation()}\n```"),
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "accept"


def test_judge_rejects_bad_result():
    bad = json.dumps(
        {
            "relevance": 0.1,
            "freshness": 0.1,
            "completeness": 0.1,
            "confidence": 0.2,
            "decision": "reject",
            "reason": "Does not answer the query",
        }
    )

    with patch(
        "agent_reach.reliability.judge.genai.Client",
        return_value=mock_client(bad),
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("What is MCP?", "MCP is a multiplayer game.")

    assert evaluation["decision"] == "reject"


def test_judge_rejects_malformed_gemma_json():
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        return_value=mock_client('{"relevance": 0.9, malformed'),
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "reject"
    assert "malformed JSON" in evaluation["reason"]


def test_judge_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "reject"
    assert "missing" in evaluation["reason"]


def test_judge_handles_api_failure():
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        side_effect=OSError("connection refused"),
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "reject"
    assert "connection failed" in evaluation["reason"]


def test_judge_handles_gemini_api_error():
    api_error = genai_errors.APIError(
        400,
        {"error": {"message": "API key not valid"}},
    )
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        side_effect=api_error,
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "reject"
    assert "API key not valid" in evaluation["reason"]


def test_judge_handles_http_transport_error():
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        side_effect=httpx.RemoteProtocolError("server disconnected"),
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "reject"
    assert "connection failed" in evaluation["reason"]


def test_judge_handles_timeout():
    with patch(
        "agent_reach.reliability.judge.genai.Client",
        side_effect=TimeoutError,
    ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        evaluation = judge_result("query", "result")

    assert evaluation["decision"] == "reject"
    assert "timed out" in evaluation["reason"]
