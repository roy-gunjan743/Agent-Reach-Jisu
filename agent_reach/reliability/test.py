from agent_reach.reliability.router import AdaptiveRouter


def bad_backend(query: str) -> str:
    return """
    MCP is a multiplayer game modification framework.
    Players can create custom maps and play together.
    """


def good_backend(query: str) -> str:
    return """
    Model Context Protocol (MCP) is an open protocol that allows
    AI applications to connect to external tools and data sources
    through standardized interfaces.

    It allows AI agents to discover and use external tools without
    requiring a separate custom integration for every tool.
    """


router = AdaptiveRouter(
    backends={
        "bad-backend": bad_backend,
        "good-backend": good_backend,
    }
)

query = "What is MCP and how does it improve AI agent tool usage?"

result = router.route(query)

print("\n==============================")
print("       FINAL ROUTER RESULT")
print("==============================")

if result:
    print(f"Backend: {result.backend}")
    print(f"Score:   {result.score:.2f}")
    print(f"Decision: {result.evaluation['decision']}")
    print("\nAnswer:")
    print(result.result)
else:
    print("No useful result found.")