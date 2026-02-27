"""
Tiered model routing with health tracking and round-robin key rotation.

The router selects which provider/model/key to use for each request based on
tier configuration, model health, and key availability.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from curio_agent_sdk.core.circuit_breaker import CircuitBreaker
from curio_agent_sdk.core.credentials import CredentialResolver, EnvCredentialResolver

logger = logging.getLogger(__name__)


class DegradationStrategy:
    """
    Strategy for what to do when all keys for a provider are unhealthy.

    Subclass and override ``handle`` to customize behavior.
    """

    def handle(self, router: "TieredRouter", provider: str) -> "ProviderKey | None":
        """
        Called when no healthy key is available for a provider.

        Args:
            router: The TieredRouter instance.
            provider: The provider name with all keys unhealthy.

        Returns:
            A ProviderKey to use (or None to skip this provider).
        """
        raise NotImplementedError


class ResetAndRetry(DegradationStrategy):
    """Reset all key health and return the first enabled key (original behavior)."""

    def handle(self, router: "TieredRouter", provider: str) -> "ProviderKey | None":
        router._reset_health(provider)
        prov = router.providers.get(provider)
        if not prov:
            return None
        enabled = [k for k in prov.keys if k.enabled]
        return enabled[0] if enabled else None


class FallbackToLowerTier(DegradationStrategy):
    """
    When all keys for a provider are down, try the next lower tier's providers.

    Falls back to ResetAndRetry if no lower tier is available.
    """

    def handle(self, router: "TieredRouter", provider: str) -> "ProviderKey | None":
        tier_order = ["tier3", "tier2", "tier1"]
        for tier_name in tier_order:
            tier = router.tiers.get(tier_name)
            if not tier or not tier.enabled:
                continue
            for mp in tier.model_priority:
                if mp.provider == provider:
                    continue
                if mp.provider not in router.providers:
                    continue
                prov = router.providers[mp.provider]
                if not prov.enabled:
                    continue
                key = router._get_healthy_key_no_degrade(mp.provider)
                if key:
                    return key
        # Fallback: reset and retry
        return ResetAndRetry().handle(router, provider)


class RaiseError(DegradationStrategy):
    """Return None so the caller gets a clear signal that no key is available."""

    def handle(self, router: "TieredRouter", provider: str) -> "ProviderKey | None":
        return None


@dataclass
class ProviderKey:
    """A single API key for a provider."""
    api_key: str
    name: str = "default"
    enabled: bool = True


@dataclass
class ModelPriority:
    """A model in the priority list."""
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class TierConfig:
    """Configuration for a routing tier."""
    name: str
    model_priority: list[ModelPriority] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ProviderConfig:
    """Provider configuration with keys and base URL."""
    name: str
    keys: list[ProviderKey] = field(default_factory=list)
    default_model: str = ""
    enabled: bool = True
    base_url: str | None = None


@dataclass
class KeyHealth:
    """
    Health tracking for a single API key.

    Delegates to CircuitBreaker for failure/recovery logic. The CircuitBreaker
    is a reusable utility that also works for connectors, MCP servers, etc.
    """
    _breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(
        max_failures=3,
        recovery_seconds=300.0,
        rate_limit_recovery_seconds=3600.0,
    ))
    last_used: datetime | None = None

    @property
    def success_count(self) -> int:
        return self._breaker.success_count

    @property
    def failure_count(self) -> int:
        return self._breaker.failure_count

    @property
    def consecutive_failures(self) -> int:
        return self._breaker.consecutive_failures

    @property
    def rate_limited(self) -> bool:
        return self._breaker.rate_limited

    @property
    def is_healthy(self) -> bool:
        return self._breaker.allows_request

    def record_success(self):
        self._breaker.record_success()
        self.last_used = datetime.now()

    def record_failure(self, is_rate_limit: bool = False):
        self._breaker.record_failure(is_rate_limit=is_rate_limit)


@dataclass
class RouteResult:
    """Result of routing: which provider/model/key to use."""
    provider: str
    model: str
    api_key: str
    key_name: str
    base_url: str | None = None


class TieredRouter:
    """
    Routes LLM requests to the best available provider/model/key.

    Features:
    - Three-tier system (tier1: fast/cheap, tier2: balanced, tier3: high quality)
    - Model list order defines priority
    - Round-robin key rotation
    - Health tracking with automatic recovery
    - Configurable via env vars or programmatic API
    - Retry/backoff configuration for rate limits
    """

    def __init__(
        self,
        tier1: list[str] | None = None,
        tier2: list[str] | None = None,
        tier3: list[str] | None = None,
        providers: dict[str, ProviderConfig] | None = None,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        retry_on_rate_limit: bool = True,
        degradation_strategy: DegradationStrategy | None = None,
        credential_resolver: CredentialResolver | None = None,
    ):
        """
        Initialize the router.

        Args:
            tier1: List of "provider:model" strings for tier1 (fast/cheap)
            tier2: List of "provider:model" strings for tier2 (balanced)
            tier3: List of "provider:model" strings for tier3 (high quality)
            providers: Pre-configured provider configs (otherwise loaded from env)
            retry_delay: Base delay (seconds) for rate-limit backoff.
            max_retry_delay: Maximum delay (seconds) for backoff.
            retry_on_rate_limit: Whether to sleep before failover on rate limits.
            degradation_strategy: Strategy when all keys are unhealthy.
                Defaults to ResetAndRetry (original behavior).
        """
        self.providers: dict[str, ProviderConfig] = providers or {}
        self.tiers: dict[str, TierConfig] = {}
        self.key_health: dict[str, dict[str, KeyHealth]] = {}  # provider -> key_name -> health
        self._key_index: dict[str, int] = {}  # provider -> round-robin index

        # Retry/backoff configuration used by LLMClient
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.retry_on_rate_limit = retry_on_rate_limit
        self.degradation_strategy = degradation_strategy or ResetAndRetry()
        # Pluggable secret resolution for API keys (env, Vault, AWS, etc.)
        self.credential_resolver: CredentialResolver | None = (
            credential_resolver or EnvCredentialResolver()
        )

        if not self.providers:
            self._load_providers_from_env()

        custom_tiers = {}
        if tier1:
            custom_tiers["tier1"] = tier1
        if tier2:
            custom_tiers["tier2"] = tier2
        if tier3:
            custom_tiers["tier3"] = tier3

        self._load_tiers(custom_tiers)

    def _get_secret(self, name: str) -> str | None:
        """
        Resolve a secret (API key) using the configured credential resolver.

        Falls back to os.getenv when no resolver is configured or when
        resolution fails.
        """
        try:
            if self.credential_resolver is not None:
                value = self.credential_resolver.resolve(name)
                if value:
                    return value
        except Exception as e:  # pragma: no cover - best-effort logging
            logger.warning("Credential resolver failed for %s: %s", name, e)
        return os.getenv(name) or None

    def _load_providers_from_env(self):
        """Load provider configurations from environment variables (and secrets)."""
        provider_defs = [
            ("openai", "OPENAI", "gpt-4o-mini"),
            ("anthropic", "ANTHROPIC", "claude-sonnet-4-6"),
            ("groq", "GROQ", "llama-3.1-8b-instant"),
        ]

        for name, prefix, default_model in provider_defs:
            keys = self._load_keys(prefix)
            if keys:
                self.providers[name] = ProviderConfig(
                    name=name,
                    keys=keys,
                    default_model=os.getenv(f"{prefix}_DEFAULT_MODEL", default_model),
                    enabled=os.getenv(f"{prefix}_ENABLED", "true").lower() == "true",
                    base_url=os.getenv(f"{prefix}_BASE_URL"),
                )

        # Ollama (no API key needed)
        ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
        ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
        if ollama_host or ollama_enabled:
            base_url = ollama_host or "http://localhost:11434"
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self.providers["ollama"] = ProviderConfig(
                name="ollama",
                keys=[ProviderKey(api_key="ollama", name="local")],
                default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b"),
                enabled=True,
                base_url=base_url,
            )

    def _load_keys(self, prefix: str) -> list[ProviderKey]:
        """Load API keys: single key + numbered keys via the credential resolver."""
        keys: list[ProviderKey] = []
        single = self._get_secret(f"{prefix}_API_KEY")
        if single:
            keys.append(ProviderKey(api_key=single, name="default"))

        i = 1
        while True:
            key_value = self._get_secret(f"{prefix}_API_KEY_{i}")
            if not key_value:
                break
            enabled_env = os.getenv(f"{prefix}_API_KEY_{i}_ENABLED", "true")
            enabled = enabled_env.lower() == "true"
            name = os.getenv(f"{prefix}_API_KEY_{i}_NAME", f"key{i}")
            keys.append(ProviderKey(api_key=key_value, name=name, enabled=enabled))
            i += 1

        return keys

    def _load_tiers(self, custom_tiers: dict[str, list[str]]):
        """Load tier configurations from env or custom config."""
        auto_tier_defaults = {
            "openai": {"tier1": "gpt-4o-mini", "tier2": "gpt-4o", "tier3": "gpt-4o"},
            "anthropic": {"tier1": "claude-haiku-4-5-20251001", "tier2": "claude-sonnet-4-6", "tier3": "claude-sonnet-4-6"},
            "groq": {"tier1": "llama-3.1-8b-instant", "tier2": "llama-3.3-70b-versatile", "tier3": "llama-3.3-70b-versatile"},
            "ollama": {"tier1": "llama3.1:8b", "tier2": "llama3.1:70b", "tier3": "llama3.1:70b"},
        }

        for tier_name in ["tier1", "tier2", "tier3"]:
            env_value = os.getenv(f"{tier_name.upper()}_MODELS", "")

            if tier_name in custom_tiers:
                models = self._parse_model_list(",".join(custom_tiers[tier_name]))
            elif env_value:
                models = self._parse_model_list(env_value)
            else:
                # Auto-detect from available providers
                models = []
                for prov_name in self.providers:
                    if prov_name in auto_tier_defaults and tier_name in auto_tier_defaults[prov_name]:
                        models.append(ModelPriority(
                            provider=prov_name,
                            model=auto_tier_defaults[prov_name][tier_name],
                        ))

            self.tiers[tier_name] = TierConfig(
                name=tier_name,
                model_priority=models,
                enabled=os.getenv(f"{tier_name.upper()}_ENABLED", "true").lower() == "true",
            )

    def _parse_model_list(self, models_str: str) -> list[ModelPriority]:
        result = []
        for item in models_str.split(","):
            item = item.strip()
            if ":" in item:
                provider, model = item.split(":", 1)
                result.append(ModelPriority(provider=provider.strip(), model=model.strip()))
        return result

    def route(
        self,
        tier: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        excluded_models: list[str] | None = None,
    ) -> RouteResult | None:
        """
        Route a request to a provider/model/key.

        Args:
            tier: Tier name for automatic routing.
            provider: Explicit provider name.
            model: Explicit model name.
            excluded_models: List of "provider:model" keys to skip.

        Returns:
            RouteResult or None if nothing available.
        """
        excluded = set(excluded_models or [])

        if tier:
            return self._route_by_tier(tier, excluded)
        elif provider:
            return self._route_by_provider(provider, model, excluded)
        else:
            # Use first available provider
            for prov_name, prov_config in self.providers.items():
                if prov_config.enabled:
                    key = self._get_healthy_key(prov_name)
                    if key:
                        return RouteResult(
                            provider=prov_name,
                            model=model or prov_config.default_model,
                            api_key=key.api_key,
                            key_name=key.name,
                            base_url=prov_config.base_url,
                        )
        return None

    def _route_by_tier(self, tier: str, excluded: set[str]) -> RouteResult | None:
        tier_config = self.tiers.get(tier)
        if not tier_config or not tier_config.enabled:
            return None

        for mp in tier_config.model_priority:
            if mp.key in excluded:
                continue
            if mp.provider not in self.providers:
                continue
            prov = self.providers[mp.provider]
            if not prov.enabled:
                continue

            key = self._get_healthy_key(mp.provider)
            if not key:
                continue

            return RouteResult(
                provider=mp.provider,
                model=mp.model,
                api_key=key.api_key,
                key_name=key.name,
                base_url=prov.base_url,
            )

        return None

    def _route_by_provider(self, provider: str, model: str | None, excluded: set[str]) -> RouteResult | None:
        prov = self.providers.get(provider)
        if not prov or not prov.enabled:
            return None

        key = self._get_healthy_key(provider)
        if not key:
            return None

        return RouteResult(
            provider=provider,
            model=model or prov.default_model,
            api_key=key.api_key,
            key_name=key.name,
            base_url=prov.base_url,
        )

    def _get_healthy_key(self, provider: str) -> ProviderKey | None:
        """Get next healthy key using round-robin, with degradation strategy on exhaustion."""
        prov = self.providers.get(provider)
        if not prov:
            return None

        # Initialize health tracking
        if provider not in self.key_health:
            self.key_health[provider] = {}
            for key in prov.keys:
                if key.enabled:
                    self.key_health[provider][key.name] = KeyHealth()

        # Collect healthy keys
        healthy = [k for k in prov.keys if k.enabled and self._is_key_healthy(provider, k.name)]

        if not healthy:
            # All keys unhealthy — delegate to degradation strategy
            return self.degradation_strategy.handle(self, provider)

        # Round-robin
        idx = self._key_index.get(provider, 0)
        selected = healthy[idx % len(healthy)]
        self._key_index[provider] = (idx + 1) % len(healthy)

        return selected

    def _get_healthy_key_no_degrade(self, provider: str) -> ProviderKey | None:
        """Get a healthy key without triggering degradation (for internal fallback use)."""
        prov = self.providers.get(provider)
        if not prov:
            return None

        if provider not in self.key_health:
            self.key_health[provider] = {}
            for key in prov.keys:
                if key.enabled:
                    self.key_health[provider][key.name] = KeyHealth()

        healthy = [k for k in prov.keys if k.enabled and self._is_key_healthy(provider, k.name)]
        if not healthy:
            return None

        idx = self._key_index.get(provider, 0)
        selected = healthy[idx % len(healthy)]
        self._key_index[provider] = (idx + 1) % len(healthy)
        return selected

    def _is_key_healthy(self, provider: str, key_name: str) -> bool:
        health = self.key_health.get(provider, {}).get(key_name)
        return health.is_healthy if health else True

    def _reset_health(self, provider: str):
        if provider in self.key_health:
            for health in self.key_health[provider].values():
                health._breaker.reset()

    def record_success(self, provider: str, key_name: str):
        health = self.key_health.get(provider, {}).get(key_name)
        if health:
            health.record_success()

    def record_failure(self, provider: str, key_name: str, is_rate_limit: bool = False):
        health = self.key_health.get(provider, {}).get(key_name)
        if health:
            health.record_failure(is_rate_limit)

    def register_provider(
        self,
        name: str,
        api_key: str = "",
        default_model: str = "",
        base_url: str | None = None,
        key_name: str = "default",
    ):
        """Register a custom provider at runtime."""
        self.providers[name] = ProviderConfig(
            name=name,
            keys=[ProviderKey(api_key=api_key, name=key_name)],
            default_model=default_model,
            enabled=True,
            base_url=base_url,
        )

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {"providers": {}, "tiers": {}}
        for name, prov in self.providers.items():
            prov_stats: dict[str, Any] = {
                "enabled": prov.enabled,
                "key_count": len(prov.keys),
                "keys": {},
            }
            for key_name, health in self.key_health.get(name, {}).items():
                prov_stats["keys"][key_name] = {
                    "success_count": health.success_count,
                    "failure_count": health.failure_count,
                    "rate_limited": health.rate_limited,
                    "consecutive_failures": health.consecutive_failures,
                }
            stats["providers"][name] = prov_stats

        for tier_name, tier in self.tiers.items():
            stats["tiers"][tier_name] = {
                "enabled": tier.enabled,
                "models": [mp.key for mp in tier.model_priority],
            }
        return stats
