from agent_reach.reliability.executor import JinaExecutor
from agent_reach.reliability.judge import judge_result


QUERY = "What is MCP and how does it improve AI agent tool usage?"

URL = "https://en.wikipedia.org/wiki/Model_Context_Protocol"


def main():
    executor = JinaExecutor()

    print("=" * 60)
    print("       AGENT REACH ADAPTIVE ROUTER")
    print("=" * 60)

    print(f"\nQuery:   {QUERY}")
    print(f"Backend: {executor.name}")
    print(f"URL:     {URL}")

    print("\n🔎 Fetching through Agent Reach...")

    result = executor.fetch(URL)

    print(f"✅ Received {len(result):,} characters")

    print("\n🧠 Running Qwen semantic judge...")

    evaluation = judge_result(QUERY, result)

    print("\n" + "=" * 60)
    print("              SEMANTIC EVALUATION")
    print("=" * 60)

    print(f"Relevance:    {evaluation.get('relevance', 0):.2f}")
    print(f"Freshness:    {evaluation.get('freshness', 0):.2f}")
    print(f"Completeness: {evaluation.get('completeness', 0):.2f}")
    print(f"Confidence:   {evaluation.get('confidence', 0):.2f}")
    print(f"Decision:     {evaluation.get('decision', 'unknown')}")
    print(f"Reason:       {evaluation.get('reason', '')}")

    print("\n" + "=" * 60)

    if evaluation.get("decision") == "accept":
        print("✅ RESULT ACCEPTED")
    else:
        print("❌ RESULT REJECTED")

    print("=" * 60)


if __name__ == "__main__":
    main()