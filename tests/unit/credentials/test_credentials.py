"""
Unit tests for CredentialResolver and implementations (Phase 14 — Context & Credentials).
"""

import os
import pytest

from curio_agent_sdk.credentials.credentials import (
    EnvCredentialResolver,
    VaultCredentialResolver,
    AWSSecretsResolver,
    resolve_credential_mapping,
    resolve_credentials_with_env,
)


@pytest.mark.unit
def test_env_resolver_found(monkeypatch):
    """Resolve from env var."""
    monkeypatch.setenv("TEST_API_KEY", "secret123")
    resolver = EnvCredentialResolver()
    assert resolver.resolve("TEST_API_KEY") == "secret123"
    monkeypatch.delenv("TEST_API_KEY", raising=False)


@pytest.mark.unit
def test_env_resolver_not_found(monkeypatch):
    """Returns None when key not in env."""
    monkeypatch.delenv("MISSING_KEY", raising=False)
    resolver = EnvCredentialResolver()
    assert resolver.resolve("MISSING_KEY") is None


@pytest.mark.unit
def test_env_resolver_with_prefix(monkeypatch):
    """Prefix is prepended to key before lookup."""
    monkeypatch.setenv("CURIO_OPENAI_API_KEY", "sk-xxx")
    resolver = EnvCredentialResolver(prefix="CURIO_")
    assert resolver.resolve("OPENAI_API_KEY") == "sk-xxx"
    monkeypatch.delenv("CURIO_OPENAI_API_KEY", raising=False)


@pytest.mark.unit
def test_env_resolver_optional_default(monkeypatch):
    """resolve_optional() returns default when key not found."""
    monkeypatch.delenv("OPTIONAL_KEY", raising=False)
    resolver = EnvCredentialResolver()
    assert resolver.resolve_optional("OPTIONAL_KEY", default="fallback") == "fallback"
    monkeypatch.setenv("OPTIONAL_KEY", "set")
    assert resolver.resolve_optional("OPTIONAL_KEY", default="fallback") == "set"
    monkeypatch.delenv("OPTIONAL_KEY", raising=False)


@pytest.mark.unit
def test_vault_resolver_creation():
    """VaultCredentialResolver init (no hvac call)."""
    r = VaultCredentialResolver(
        vault_url="http://vault:8200",
        token="s.xxx",
        mount_point="secret",
        key_template="app/{key}",
    )
    assert r.vault_url == "http://vault:8200"
    assert r.token == "s.xxx"
    assert r.mount_point == "secret"
    assert r.key_template == "app/{key}"


@pytest.mark.unit
def test_aws_resolver_creation():
    """AWSSecretsResolver init."""
    r = AWSSecretsResolver(region_name="us-east-1", prefix="curio/")
    assert r.region_name == "us-east-1"
    assert r.prefix == "curio/"


@pytest.mark.unit
def test_resolve_credential_mapping_exact_and_partial(monkeypatch):
    """resolve_credential_mapping replaces $VAR, ${VAR}, and partial ${VAR} in strings."""
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("TOKEN", "t123")
    resolver = EnvCredentialResolver()
    credentials = {
        "api_key": "${API_KEY}",
        "token": "$TOKEN",
        "auth": "Bearer ${TOKEN}",
    }
    out = resolve_credential_mapping(credentials, resolver)
    assert out["api_key"] == "secret"
    assert out["token"] == "t123"
    assert out["auth"] == "Bearer t123"
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("TOKEN", raising=False)


@pytest.mark.unit
def test_resolve_credentials_with_env(monkeypatch):
    """resolve_credentials_with_env uses EnvCredentialResolver for $VAR substitution."""
    monkeypatch.setenv("MY_KEY", "env_value")
    out = resolve_credentials_with_env({"key": "${MY_KEY}"})
    assert out["key"] == "env_value"
    monkeypatch.delenv("MY_KEY", raising=False)
