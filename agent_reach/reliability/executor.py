from __future__ import annotations

import urllib.request

from agent_reach.channels.web import WebChannel
from agent_reach.utils.url import normalize_public_http_url


class JinaExecutor:
    """
    Adapter between Agent Reach's WebChannel
    and the AdaptiveRouter.

    Agent Reach handles the actual web retrieval.
    The reliability layer evaluates the returned content.
    """

    name = "Jina Reader"

    def __init__(self) -> None:
        self.channel = WebChannel()

    def fetch(self, url: str) -> str:
        return self.channel.read(url)


class DirectHTTPExecutor:
    """
    Lightweight stdlib-only HTTP fetcher.

    Uses ``urllib`` directly (no Jina Reader proxy) so the router has a
    genuinely *different* fetch path from ``JinaExecutor``.  Reuses
    ``normalize_public_http_url`` for SSRF safety.

    The returned body is raw decoded text (not Markdown-rendered), which
    is intentionally less useful for structured content and helps the
    semantic judge distinguish good and bad results.
    """

    name = "Direct HTTP"

    _UA = (
        "Mozilla/5.0 (compatible; AgentReach/1.0; "
        "+https://github.com/Panniantong/agent-reach)"
    )
    _TIMEOUT = 30
    _MAX_BYTES = 5 * 1024 * 1024

    def fetch(self, url: str) -> str:
        url = normalize_public_http_url(url)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": self._UA, "Accept": "text/html, text/plain"},
        )
        with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
            body = resp.read(self._MAX_BYTES + 1)
        if len(body) > self._MAX_BYTES:
            raise ValueError(
                f"Response exceeds {self._MAX_BYTES} byte limit"
            )
        return body.decode("utf-8", errors="replace")


# Alias so that agent_reach/reliability/test_web.py can
# ``from agent_reach.reliability.executor import WebExecutor``.
WebExecutor = JinaExecutor