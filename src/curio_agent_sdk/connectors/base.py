"""
Connector framework — base class and resource type for external service integrations.

Connectors provide tools and optional context resources for agents, with
lifecycle (connect/disconnect) and credential management. Use ConnectorBridge
to register connector tools with an agent and run lifecycle at startup/shutdown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from curio_agent_sdk.core.credentials import (
    CredentialResolver,
    EnvCredentialResolver,
    resolve_credential_mapping,
)

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


def resolve_credentials(credentials: Mapping[str, Any]) -> dict[str, Any]:
    """
    Legacy helper for resolving environment variable references in credential values.

    This now routes through the pluggable ``CredentialResolver`` abstraction,
    defaulting to ``EnvCredentialResolver`` so existing code keeps working:

    - Values like ``\"$VAR\"`` or ``\"${VAR}\"`` are replaced with the result of
      ``EnvCredentialResolver().resolve(\"VAR\")``.
    - Nested mappings and lists are supported.
    """
    return resolve_credential_mapping(credentials, EnvCredentialResolver())


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
