from agent_reach.channels.web import WebChannel


class JinaExecutor:
    """
    Adapter between Agent Reach's WebChannel
    and the AdaptiveRouter.

    Agent Reach handles the actual web retrieval.
    The reliability layer evaluates the returned content.
    """

    name = "Jina Reader"

    def __init__(self):
        self.channel = WebChannel()

    def fetch(self, url: str) -> str:
        return self.channel.read(url)