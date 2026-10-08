from agent_reach.reliability.executor import WebExecutor


def main() -> None:
    executor = WebExecutor()

    url = "https://en.wikipedia.org/wiki/Model_Context_Protocol"

    print("=== AGENT REACH REAL BACKEND TEST ===")
    print(f"Backend: {executor.channel.backends[0]}")
    print(f"URL: {url}")
    print()

    result = executor.fetch(url)

    print("=== RESULT ===")
    print(result[:3000])


if __name__ == "__main__":
    main()