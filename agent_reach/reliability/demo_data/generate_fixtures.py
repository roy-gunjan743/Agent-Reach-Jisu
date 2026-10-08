"""Generate realistic sample fixture data for demo."""
import os

demo_dir = os.path.dirname(os.path.abspath(__file__))
b_path = os.path.join(demo_dir, "backend_b_mcp.txt")

with open(b_path, "w", encoding="utf-8") as f:
    f.write("Title: Model Context Protocol - Wikipedia\n")
    f.write("URL Source: https://en.wikipedia.org/wiki/Model_Context_Protocol\n")
    f.write("Markdown Content:\n\n")
    f.write("# Model Context Protocol\n\n")
    f.write(
        "The **Model Context Protocol** (**MCP**) is an open standard introduced by Anthropic in November 2024 "
        "designed to standardize how artificial intelligence (AI) models, agents, and client applications "
        "securely communicate with external data sources, enterprise tools, and execution environments.\n\n"
    )
    f.write(
        "Before MCP, integrating an AI model or autonomous agent with external capabilities required bespoke "
        "API integrations, ad-hoc function calling schemas, and custom middleware for every data repository, "
        "database, or API service. MCP solves this interoperability fragmentation by providing a standardized, "
        "bidirectional communication protocol akin to the Language Server Protocol (LSP) used in modern IDEs.\n\n"
    )
    f.write("## Overview and Architecture\n\n")
    f.write(
        "MCP uses a client-server architecture built upon JSON-RPC 2.0 messages exchanged over standard transport "
        "mechanisms such as stdio (standard input/output) for local desktop tools and Server-Sent Events (SSE) "
        "over HTTP for remote cloud services.\n\n"
    )
    f.write("The core architecture comprises three fundamental components:\n")
    f.write("- **MCP Hosts**: The client applications or agent runtimes, such as Claude Desktop, IDE extensions, or custom autonomous agent frameworks, that initiate connections, manage user authorization, and maintain conversation context.\n")
    f.write("- **MCP Clients**: Protocol endpoints maintained within the host that establish 1:1 connections with servers and coordinate request-response cycles.\n")
    f.write("- **MCP Servers**: Lightweight service programs that expose specific capabilities, database queries, web scraping functions, or file system access to clients through standardized primitives.\n\n")
    f.write("## Core Primitives\n\n")
    f.write("MCP defines three primary structural primitives:\n")
    f.write("1. **Tools**: Executable functions that models can invoke to perform external side effects or computational actions, such as executing SQL queries, querying APIs, fetching web pages, or running terminal commands.\n")
    f.write("2. **Resources**: Read-only contextual data, documents, log files, or schema metadata that provide grounding information to the model.\n")
    f.write("3. **Prompts**: Parameterized prompt templates and interaction patterns that guide LLM behavior for domain-specific tasks.\n\n")
    f.write("## Benefits for AI Agents\n\n")
    f.write("1. **Interoperability**: A single MCP server implementation works seamlessly across multiple AI clients, tools, and developer platforms without rewriting integration code.\n")
    f.write("2. **Security and Isolation**: MCP servers operate in isolated processes with strict credential segregation. AI models interact through capability declarations rather than direct credential access.\n")
    f.write("3. **Modularity**: Developers can compose swarms of specialized servers (e.g., GitHub server, Postgres server, Web Reader server) to expand agent abilities dynamically.\n\n")

    topics = [
        ("JSON-RPC 2.0 Transport Details", "The protocol communication layer follows JSON-RPC 2.0. Messages include requests, responses, and unidirectional notifications. Request identifiers ensure asynchronous tracking across concurrent tool executions."),
        ("Transport Layer Mechanics", "Local agents communicate via stdin and stdout pipelines with newline-delimited JSON. Remote servers utilize HTTPS with Server-Sent Events (SSE) for server-to-client streaming and standard POST requests for client-to-server commands."),
        ("Security Model and Access Control", "Hosts maintain full user-in-the-loop permission granting before tools are executed. Sensitive operations require explicit confirmation. Server capability negotiation happens during the initialization handshake."),
        ("Ecosystem and Agent Adoption", "Since its open-source release, MCP has been adopted across leading developer tools, IDEs, code editors, and agent frameworks including Claude Desktop, Cursor, Continue, Zed, and open-source orchestrators."),
    ]
    
    for i in range(16):
        for title, desc in topics:
            f.write(f"### Section {i+1}: {title}\n\n{desc}\n\n")
            f.write(f"Detailed evaluation notes {i}: The protocol guarantees deterministic serialization and schema validation using JSON Schema definitions. Performance metrics show roundtrip latencies under 15ms.\n\n")

    f.write("## Extended Link Directory\n\n")
    for k in range(60):
        f.write(f"* [Reference link {k+1}: Technical Specification Documentation](https://spec.modelcontextprotocol.io/v1/details?utm_source=wiki&utm_medium=referral&id={k+1})\n")

    f.write("\n---\nWe use cookies to improve your browsing experience. Click here to accept cookies. Subscribe to our newsletter. All rights reserved. Terms of service apply. Share on Twitter, Facebook, LinkedIn.\n")

print("Generated backend_b_mcp.txt, size:", os.path.getsize(b_path))
