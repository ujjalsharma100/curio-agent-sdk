"""
Pluggable credential resolution for secrets (API keys, tokens, passwords).

This module introduces a small abstraction, ``CredentialResolver``, that can
fetch secrets from different backends (environment variables, Vault, AWS
Secrets Manager, etc.) and a helper for expanding ``$VAR`` / ``${VAR}``
style references in nested credential dictionaries.

The goal is to decouple secret lookup from individual components so that
multi-tenant deployments can centralize how credentials are stored and
audited.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, MutableMapping

logger = logging.getLogger(__name__)


class CredentialResolver(ABC):
    """
    Abstract base class for secret resolution.

    Implementations translate a logical key (e.g. ``"OPENAI_API_KEY"``) into
    a concrete secret value by looking it up in env vars, a vault, a cloud
    secret manager, etc.
    """

    @abstractmethod
    def resolve(self, key: str) -> str | None:
        """
        Resolve a single secret key to its value.

        Args:
            key: Logical secret key/name (e.g. ``\"OPENAI_API_KEY\"``).

        Returns:
            The resolved secret value, or None if not found.
        """
        raise NotImplementedError

    def resolve_optional(self, key: str, default: str | None = None) -> str | None:
        """Resolve a key, returning default if not found or empty."""
        value = self.resolve(key)
        if value is None or value == "":
            return default
        return value


class EnvCredentialResolver(CredentialResolver):
    """
    Resolve secrets from environment variables (default behavior).

    This is a thin wrapper over ``os.getenv`` but implements the
    ``CredentialResolver`` interface so that components don't depend on
    ``os.environ`` directly.
    """

    def __init__(self, prefix: str | None = None) -> None:
        """
        Args:
            prefix: Optional prefix to prepend to keys before lookup.
                For example, prefix=\"CURIO_\" will turn \"OPENAI_API_KEY\"
                into \"CURIO_OPENAI_API_KEY\" when resolving.
        """
        self.prefix = prefix

    def resolve(self, key: str) -> str | None:
        env_key = f"{self.prefix}{key}" if self.prefix else key
        return os.getenv(env_key)


class VaultCredentialResolver(CredentialResolver):
    """
    HashiCorp Vault-backed credential resolver.

    This implementation is intentionally minimal and only depends on the
    optional ``hvac`` library when it is actually used. If ``hvac`` is not
    installed, an informative ImportError is raised on first use.
    """

    def __init__(
        self,
        vault_url: str,
        token: str,
        *,
        mount_point: str = "secret",
        key_template: str = "{key}",
    ) -> None:
        """
        Args:
            vault_url: Base URL for the Vault server.
            token: Vault access token.
            mount_point: KV v2 mount point (default: ``\"secret\"``).
            key_template: Template for mapping logical keys to Vault paths,
                e.g. ``\"llm/{key}\"`` will look up ``llm/OPENAI_API_KEY``.
        """
        self.vault_url = vault_url
        self.token = token
        self.mount_point = mount_point
        self.key_template = key_template
        self._client = None

    def _get_client(self):
        try:
            import hvac  # type: ignore[import]
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "VaultCredentialResolver requires the 'hvac' package. "
                "Install with: pip install hvac"
            ) from e
        if self._client is None:
            self._client = hvac.Client(url=self.vault_url, token=self.token)
        return self._client

    def resolve(self, key: str) -> str | None:
        client = self._get_client()
        path = self.key_template.format(key=key)
        try:
            # KV v2 layout: data -> data -> fields
            response = client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.mount_point,
            )
            data = (response.get("data") or {}).get("data") or {}
            # Common conventions: {"value": "..."} or {KEY: "..."}
            if "value" in data and data["value"]:
                return str(data["value"])
            if key in data and data[key]:
                return str(data[key])
            # Fallback: first non-empty value
            for v in data.values():
                if v:
                    return str(v)
        except Exception as e:  # pragma: no cover - best-effort logging
            logger.warning("VaultCredentialResolver failed for %s: %s", key, e)
        return None


class AWSSecretsResolver(CredentialResolver):
    """
    AWS Secrets Manager-backed credential resolver.

    Uses ``boto3`` lazily when first used. If ``boto3`` is not installed,
    an informative ImportError is raised on first use.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        prefix: str | None = None,
    ) -> None:
        """
        Args:
            region_name: AWS region (falls back to standard AWS env/config
                resolution when omitted).
            prefix: Optional prefix to prepend to secret names.
        """
        self.region_name = region_name
        self.prefix = prefix
        self._client = None

    def _get_client(self):
        try:
            import boto3  # type: ignore[import]
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "AWSSecretsResolver requires the 'boto3' package. "
                "Install with: pip install boto3"
            ) from e
        if self._client is None:
            self._client = boto3.client("secretsmanager", region_name=self.region_name)
        return self._client

    def resolve(self, key: str) -> str | None:
        client = self._get_client()
        name = f"{self.prefix}{key}" if self.prefix else key
        try:
            resp = client.get_secret_value(SecretId=name)
            if "SecretString" in resp and resp["SecretString"]:
                secret = resp["SecretString"]
                # Try to parse JSON secrets for key lookup
                try:
                    data = json.loads(secret)
                    if isinstance(data, dict):
                        if "value" in data and data["value"]:
                            return str(data["value"])
                        if key in data and data[key]:
                            return str(data[key])
                except Exception:
                    pass
                return secret
            if "SecretBinary" in resp and resp["SecretBinary"]:
                # Binary secrets are returned as bytes; decode best-effort
                try:
                    return resp["SecretBinary"].decode("utf-8")
                except Exception:
                    return None
        except Exception as e:  # pragma: no cover - best-effort logging
            logger.warning("AWSSecretsResolver failed for %s: %s", name, e)
        return None


def resolve_credential_mapping(
    credentials: Mapping[str, Any],
    resolver: CredentialResolver,
) -> dict[str, Any]:
    """
    Resolve environment-style references in a credentials mapping.

    Values like ``\"$VAR\"`` or ``\"${VAR}\"`` are replaced with
    ``resolver.resolve(\"VAR\")``. For partial substitutions
    (e.g. ``\"Bearer ${TOKEN}\"``) only the referenced portion is replaced.

    Non-string values are passed through unchanged.
    """

    def _replace_env_refs(val: Any) -> Any:
        if not isinstance(val, str):
            return val
        if not val or ("$" not in val):
            return val

        s = val.strip()
        # Exact ${VAR}
        if s.startswith("${") and s.endswith("}"):
            key = s[2:-1]
            return resolver.resolve_optional(key, default="")
        # Exact $VAR
        if s.startswith("$") and re.match(r"^\$[A-Za-z_][A-Za-z0-9_]*$", s):
            key = s[1:]
            return resolver.resolve_optional(key, default="")

        # Partial substitution ${VAR} inside string
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            return resolver.resolve_optional(key, default="") or ""

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, s)

    out: dict[str, Any] = {}
    for k, v in credentials.items():
        if isinstance(v, Mapping):
            out[k] = resolve_credential_mapping(v, resolver)
        elif isinstance(v, list):
            out[k] = [_replace_env_refs(item) for item in v]
        else:
            out[k] = _replace_env_refs(v)
    return out


def resolve_credentials_with_env(credentials: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convenience helper that resolves credentials using environment variables.

    This mirrors the legacy ``resolve_credentials`` behavior from
    ``connectors.base`` but routes through the new ``CredentialResolver``
    abstraction so that callers can swap in a different resolver later.
    """
    return resolve_credential_mapping(credentials, EnvCredentialResolver())

