"""
Connector framework — base class and resource type for external service integrations.

Connectors provide tools and optional context resources for agents, with
lifecycle (connect/disconnect) and credential management. Use ConnectorBridge
to register connector tools with an agent and run lifecycle at startup/shutdown.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from curio_agent_sdk.core.tools.tool import Tool


@dataclass
class ConnectorResource:
    """
    A resource exposed by a connector for context injection (e.g. API docs, schema).

    Content can be injected into the agent's system message at run start.
    """
    uri: str
    content: str
    mime_type: str = "text/plain"


def resolve_credentials(credentials: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve environment variable references in credential values.

    Values like "$VAR" or "${VAR}" are replaced with os.environ.get("VAR", "").
    Use this to avoid hardcoding secrets (e.g. tokens, API keys).
    """
    def resolve(val: Any) -> Any:
        if not isinstance(val, str):
            return val
        if not val or ("$" not in val):
            return val
        s = val.strip()
        if s.startswith("${") and s.endswith("}"):
            key = s[2:-1]
            return os.environ.get(key, "")
        if s.startswith("$") and re.match(r"^\$[A-Za-z_][A-Za-z0-9_]*$", s):
            key = s[1:]
            return os.environ.get(key, "")
        # Partial substitution ${VAR} inside string
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            return os.environ.get(key, "")
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, s)

    return {k: resolve(v) for k, v in credentials.items()}


class Connector(ABC):
    """
    Base class for external service integrations (GitHub, Slack, databases, etc.).

    Implementations provide:
    - name: identifier (used for tool name prefixing on clashes)
    - connect(credentials): establish connection; credentials can be passed at
      construction or here (e.g. from vault). Use resolve_credentials() to
      expand $VAR in dict values.
    - disconnect(): tear down connection
    - health_check(): whether the connector is operational
    - get_tools(): tools the agent can use for this service
    - get_resources(): optional context resources (e.g. API docs) for injection

    Lifecycle is driven by ConnectorBridge at agent startup/shutdown.
    """

    name: str = "connector"

    @abstractmethod
    async def connect(self, credentials: dict[str, Any] | None = None) -> None:
        """
        Establish connection to the service.

        Args:
            credentials: Optional override; if None, implementations may use
                credentials passed at construction. Use resolve_credentials()
                to expand $VAR / ${VAR} from environment.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the connection. Safe to call multiple times."""
        ...

    async def health_check(self) -> bool:
        """Return True if the connector is operational. Default: True."""
        return True

    @abstractmethod
    def get_tools(self) -> list["Tool"]:
        """Return tools this connector provides for the agent."""
        ...

    def get_resources(self) -> list[ConnectorResource]:
        """
        Return optional context resources (e.g. API docs) for injection into system prompt.

        Default: empty list.
        """
        return []
