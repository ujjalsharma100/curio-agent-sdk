"""
Unit tests for curio_agent_sdk.core.llm.router

Covers: TieredRouter — tier routing, round robin, health tracking,
degradation strategies, auto detection
"""

import os
from unittest.mock import patch

import pytest

from curio_agent_sdk.core.llm.router import (
    TieredRouter,
    RouteResult,
    ProviderConfig,
    ProviderKey,
    TierConfig,
    ModelPriority,
    KeyHealth,
    DegradationStrategy,
    ResetAndRetry,
    FallbackToLowerTier,
    RaiseError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(
    providers: dict[str, ProviderConfig] | None = None,
    tier1: list[str] | None = None,
    tier2: list[str] | None = None,
    tier3: list[str] | None = None,
    degradation_strategy: DegradationStrategy | None = None,
) -> TieredRouter:
    """Helper to create a TieredRouter with no env var side effects."""
    return TieredRouter(
        providers=providers or {},
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        degradation_strategy=degradation_strategy,
    )


def _provider(name: str, model: str = "m1", key: str = "sk-test") -> ProviderConfig:
    return ProviderConfig(
        name=name,
        keys=[ProviderKey(api_key=key, name="default")],
        default_model=model,
        enabled=True,
    )


# ===================================================================
# Tests
# ===================================================================


class TestTieredRouter:

    def test_router_route_tier1(self):
        """Routes to tier1 provider."""
        router = _make_router(
            providers={"openai": _provider("openai", "gpt-4o-mini")},
            tier1=["openai:gpt-4o-mini"],
        )
        result = router.route(tier="tier1")
        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"

    def test_router_route_tier2(self):
        """Routes to tier2 provider."""
        router = _make_router(
            providers={"openai": _provider("openai", "gpt-4o")},
            tier2=["openai:gpt-4o"],
        )
        result = router.route(tier="tier2")
        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    def test_router_route_tier3(self):
        """Routes to tier3 provider."""
        router = _make_router(
            providers={"anthropic": _provider("anthropic", "claude-sonnet-4-6")},
            tier3=["anthropic:claude-sonnet-4-6"],
        )
        result = router.route(tier="tier3")
        assert result is not None
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-6"

    def test_router_round_robin(self):
        """Rotates among keys in same provider."""
        router = _make_router(
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    keys=[
                        ProviderKey(api_key="key-A", name="keyA"),
                        ProviderKey(api_key="key-B", name="keyB"),
                    ],
                    default_model="gpt-4o-mini",
                ),
            },
            tier1=["openai:gpt-4o-mini"],
        )

        r1 = router.route(tier="tier1")
        r2 = router.route(tier="tier1")
        r3 = router.route(tier="tier1")

        keys_used = [r1.key_name, r2.key_name, r3.key_name]
        # Should rotate: keyA, keyB, keyA
        assert keys_used == ["keyA", "keyB", "keyA"]

    def test_router_unhealthy_provider_skip(self):
        """Skips unhealthy providers and routes to next."""
        router = _make_router(
            providers={
                "bad": _provider("bad", "bad-model", "bad-key"),
                "good": _provider("good", "good-model", "good-key"),
            },
            tier1=["bad:bad-model", "good:good-model"],
            degradation_strategy=RaiseError(),
        )

        # Trigger routing to initialize health tracking, then mark bad as unhealthy
        router.route(tier="tier1")  # initializes health for bad
        # Record enough failures to trip the circuit breaker (max_failures=3)
        for _ in range(5):
            router.record_failure("bad", "default")

        result = router.route(tier="tier1")
        assert result is not None
        assert result.provider == "good"

    def test_router_all_unhealthy(self):
        """All providers down — degradation strategy kicks in."""
        router = _make_router(
            providers={"only": _provider("only", "only-model")},
            tier1=["only:only-model"],
            degradation_strategy=ResetAndRetry(),
        )

        # Initialize health, then make unhealthy
        router.route(tier="tier1")
        for _ in range(5):
            router.record_failure("only", "default")

        # ResetAndRetry should reset health and return a key
        result = router.route(tier="tier1")
        assert result is not None
        assert result.provider == "only"

    def test_router_reset_and_retry_strategy(self):
        """ResetAndRetry resets health and returns first enabled key."""
        router = _make_router(
            providers={"p": _provider("p", "m1")},
            tier1=["p:m1"],
            degradation_strategy=ResetAndRetry(),
        )
        router.route(tier="tier1")
        for _ in range(5):
            router.record_failure("p", "default")

        strategy = ResetAndRetry()
        key = strategy.handle(router, "p")
        assert key is not None
        assert key.api_key == "sk-test"

    def test_router_fallback_to_lower_tier(self):
        """FallbackToLowerTier tries another tier's provider."""
        router = _make_router(
            providers={
                "primary": _provider("primary", "p-model"),
                "fallback": _provider("fallback", "f-model"),
            },
            tier1=["primary:p-model"],
            tier2=["fallback:f-model"],
            degradation_strategy=FallbackToLowerTier(),
        )

        # Initialize and break primary
        router.route(tier="tier1")
        for _ in range(5):
            router.record_failure("primary", "default")

        # FallbackToLowerTier should find fallback in tier2
        strategy = FallbackToLowerTier()
        key = strategy.handle(router, "primary")
        assert key is not None

    def test_router_raise_error_strategy(self):
        """RaiseError returns None clearly."""
        router = _make_router(
            providers={"p": _provider("p", "m1")},
            tier1=["p:m1"],
            degradation_strategy=RaiseError(),
        )
        router.route(tier="tier1")
        for _ in range(5):
            router.record_failure("p", "default")

        result = router.route(tier="tier1")
        assert result is None

    def test_router_empty_tier(self):
        """No providers in tier — returns None."""
        router = _make_router(
            providers={"p": _provider("p", "m1")},
            tier1=["p:m1"],
            # tier2 has no models
        )
        result = router.route(tier="tier2")
        assert result is None

    def test_router_provider_health_tracking(self):
        """Mark provider healthy/unhealthy through record_success/failure."""
        router = _make_router(
            providers={"p": _provider("p", "m1")},
            tier1=["p:m1"],
        )
        # Initialize health
        router.route(tier="tier1")

        router.record_success("p", "default")
        health = router.key_health["p"]["default"]
        assert health.success_count == 1
        assert health.is_healthy is True

        for _ in range(5):
            router.record_failure("p", "default")
        assert health.failure_count == 5
        assert health.consecutive_failures >= 3

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-test"}, clear=False)
    def test_router_auto_detection(self):
        """Auto-detect providers from environment variables."""
        router = TieredRouter()

        assert "openai" in router.providers
        assert router.providers["openai"].keys[0].api_key == "sk-env-test"
        # Should have auto-generated tier configs
        assert "tier1" in router.tiers


class TestRouteResult:
    def test_route_result_creation(self):
        r = RouteResult(provider="openai", model="gpt-4o", api_key="sk-x", key_name="default")
        assert r.provider == "openai"
        assert r.model == "gpt-4o"
        assert r.base_url is None

    def test_route_result_with_base_url(self):
        r = RouteResult(
            provider="openai", model="gpt-4o", api_key="sk-x",
            key_name="default", base_url="https://custom.api.com",
        )
        assert r.base_url == "https://custom.api.com"


class TestKeyHealth:
    def test_initial_state(self):
        h = KeyHealth()
        assert h.is_healthy is True
        assert h.success_count == 0
        assert h.failure_count == 0
        assert h.consecutive_failures == 0
        assert h.rate_limited is False

    def test_record_success(self):
        h = KeyHealth()
        h.record_success()
        assert h.success_count == 1
        assert h.last_used is not None

    def test_record_failure(self):
        h = KeyHealth()
        h.record_failure()
        assert h.failure_count == 1
        assert h.consecutive_failures == 1

    def test_rate_limit_failure(self):
        h = KeyHealth()
        h.record_failure(is_rate_limit=True)
        assert h.rate_limited is True
