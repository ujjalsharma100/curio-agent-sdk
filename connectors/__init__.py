"""
Connector framework for external service integrations (GitHub, Slack, databases, etc.).

Connectors provide tools and optional context resources. Use with Agent:

    from curio_agent_sdk import Agent
    from curio_agent_sdk.connectors import Connector, ConnectorResource, ConnectorBridge

    class MyConnector(Connector):
        name = "myservice"
        async def connect(self, credentials=None): ...
        async def disconnect(self): ...
        def get_tools(self): return [my_tool]

    agent = Agent.builder() \\
        .connector(MyConnector(token="...")) \\
        .model("openai:gpt-4o") \\
        .build()
"""

from curio_agent_sdk.connectors.base import (
    Connector,
    ConnectorResource,
    resolve_credentials,
)
from curio_agent_sdk.connectors.bridge import ConnectorBridge

__all__ = [
    "Connector",
    "ConnectorResource",
    "ConnectorBridge",
    "resolve_credentials",
]
