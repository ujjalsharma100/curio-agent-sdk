"""
LLM Routing Configuration with Model Priority and Load Balancing.

This module provides tier-based routing where the model list order defines priority,
with round-robin load balancing for API keys and automatic health tracking.

Features:
- Three-tier system (tier1: fast/cheap, tier2: balanced, tier3: high quality)
- Model list order defines priority (first model tried first)
- Round-robin key rotation across multiple API keys
- Health tracking: success/failure counts, rate limit detection
- Automatic recovery after timeouts

Configuration:
    TIER1_MODELS=openai:gpt-4o-mini,groq:llama-3.1-8b,anthropic:claude-3-haiku

    The order in TIER1_MODELS IS the priority. First model is tried first,
    if it fails/rate-limits, next model is tried, etc.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderKey:
    """Represents a single API key for a provider."""
    api_key: str
    name: Optional[str] = None
    enabled: bool = True


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_name: str
    enabled: bool = True
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class ProviderConfig:
    """Configuration for a provider with multiple keys and models."""
    provider: str
    keys: List[ProviderKey]
    models: Dict[str, ModelConfig]  # Model name -> ModelConfig
    default_model: str
    enabled: bool = True
    base_url: Optional[str] = None


@dataclass
class ModelPriority:
    """A single model in the priority list."""
    provider: str
    model: str


@dataclass
class TierConfig:
    """Configuration for a tier with ordered model priority list."""
    tier_name: str
    model_priority: List[ModelPriority]  # Ordered list - first is highest priority
    enabled: bool = True


@dataclass
class KeyStatus:
    """Track status of an API key for health checking."""
    key: ProviderKey
    success_count: int = 0
    failure_count: int = 0
    rate_limit_hit: bool = False
    last_used: Optional[datetime] = None
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None


class LLMRoutingConfig:
    """
    Model-priority based routing configuration with load balancing.

    The model list order defines the priority. Example:
        TIER1_MODELS=openai:gpt-4o-mini,groq:llama-3.1-8b,openai:gpt-4o

    This means: try openai:gpt-4o-mini first, if it fails try groq:llama-3.1-8b,
    then openai:gpt-4o. You can mix providers in any order.

    Example:
        >>> routing = LLMRoutingConfig()
        >>> provider, model, key = routing.get_provider_and_model_for_tier("tier3")
        >>> print(f"Using {provider}/{model}")

    Configuration via Environment Variables:
        - TIER1_MODELS: Comma-separated model list (provider:model,provider:model)
        - TIER2_MODELS: Same format for tier2
        - TIER3_MODELS: Same format for tier3
        - OPENAI_API_KEY: API key for OpenAI
        - GROQ_API_KEY: API key for Groq
        - ANTHROPIC_API_KEY: API key for Anthropic
        - OLLAMA_HOST: Host for Ollama (no key needed)
    """

    def __init__(self, custom_tiers: Optional[Dict[str, List[str]]] = None):
        """
        Initialize routing configuration.

        Args:
            custom_tiers: Optional dict of tier_name -> list of "provider:model" strings
        """
        self.providers: Dict[str, ProviderConfig] = {}
        self.tiers: Dict[str, TierConfig] = {}
        self.key_statuses: Dict[str, Dict[str, KeyStatus]] = {}  # provider -> key_name -> status
        self._key_index: Dict[str, int] = {}  # provider -> current round-robin index

        self._load_from_env(custom_tiers)

    def _load_from_env(self, custom_tiers: Optional[Dict[str, List[str]]] = None):
        """Load configuration from environment variables."""

        # Load provider configurations FIRST (to know what's available)
        self._load_providers()

        # Load tier configurations
        for tier_name in ["tier1", "tier2", "tier3"]:
            env_var = f"{tier_name.upper()}_MODELS"
            env_value = os.getenv(env_var, "")

            if env_value:
                # Parse from environment
                model_priority = self._parse_model_list(env_value)
            elif custom_tiers and tier_name in custom_tiers:
                # Use custom config
                model_priority = self._parse_model_list(",".join(custom_tiers[tier_name]))
            else:
                # Auto-detect based on available providers
                model_priority = self._get_auto_models_for_tier(tier_name)

            self.tiers[tier_name] = TierConfig(
                tier_name=tier_name,
                model_priority=model_priority,
                enabled=os.getenv(f"{tier_name.upper()}_ENABLED", "true").lower() == "true",
            )

    def _parse_model_list(self, models_str: str) -> List[ModelPriority]:
        """
        Parse model priority list from string.
        Format: "provider:model,provider:model,provider:model"
        """
        result = []
        for item in models_str.split(","):
            item = item.strip()
            if ":" in item:
                provider, model = item.split(":", 1)
                result.append(ModelPriority(provider=provider.strip(), model=model.strip()))
            elif item:
                # If no provider specified, try to infer from model name
                logger.warning(f"Model '{item}' has no provider prefix, skipping")
        return result

    def _get_auto_models_for_tier(self, tier: str) -> List[ModelPriority]:
        """
        Auto-generate model priority for a tier based on available providers.
        """
        available = list(self.providers.keys())

        if not available:
            logger.warning("No providers configured. Set an API key (OPENAI_API_KEY, etc.)")
            return []

        # Provider-specific model defaults per tier
        defaults = {
            "openai": {"tier1": "gpt-4o-mini", "tier2": "gpt-4o", "tier3": "gpt-4o"},
            "anthropic": {"tier1": "claude-3-haiku-20240307", "tier2": "claude-3-5-sonnet-20241022", "tier3": "claude-3-5-sonnet-20241022"},
            "groq": {"tier1": "llama-3.1-8b-instant", "tier2": "llama-3.3-70b-versatile", "tier3": "llama-3.3-70b-versatile"},
            "ollama": {"tier1": "llama3.1:8b", "tier2": "llama3.1:70b", "tier3": "llama3.1:70b"},
        }

        result = []
        for provider in available:
            if provider in defaults and tier in defaults[provider]:
                result.append(ModelPriority(provider=provider, model=defaults[provider][tier]))

        logger.info(f"Auto-configured {tier}: {[f'{m.provider}:{m.model}' for m in result]}")
        return result

    def _load_provider_keys(self, provider_prefix: str) -> List[ProviderKey]:
        """Load multiple API keys for a provider."""
        keys = []

        # Try single key first (backward compatibility)
        single_key = os.getenv(f"{provider_prefix}_API_KEY")
        if single_key:
            keys.append(ProviderKey(api_key=single_key, name="default"))

        # Try multiple keys: PROVIDER_API_KEY_1, PROVIDER_API_KEY_2, etc.
        i = 1
        while True:
            key = os.getenv(f"{provider_prefix}_API_KEY_{i}")
            if not key:
                break
            enabled = os.getenv(f"{provider_prefix}_API_KEY_{i}_ENABLED", "true").lower() == "true"
            key_name = os.getenv(f"{provider_prefix}_API_KEY_{i}_NAME", f"key{i}")
            keys.append(ProviderKey(api_key=key, name=key_name, enabled=enabled))
            i += 1

        return keys

    def _load_providers(self):
        """Load provider configurations."""

        # Groq configuration
        groq_keys = self._load_provider_keys("GROQ")
        if groq_keys:
            self.providers["groq"] = ProviderConfig(
                provider="groq",
                keys=groq_keys,
                default_model=os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant"),
                models={},
                enabled=os.getenv("GROQ_ENABLED", "true").lower() == "true",
            )

        # OpenAI configuration
        openai_keys = self._load_provider_keys("OPENAI")
        if openai_keys:
            self.providers["openai"] = ProviderConfig(
                provider="openai",
                keys=openai_keys,
                default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                models={},
                enabled=os.getenv("OPENAI_ENABLED", "true").lower() == "true",
                base_url=os.getenv("OPENAI_BASE_URL"),  # Support custom base_url
            )

        # Anthropic configuration
        anthropic_keys = self._load_provider_keys("ANTHROPIC")
        if anthropic_keys:
            self.providers["anthropic"] = ProviderConfig(
                provider="anthropic",
                keys=anthropic_keys,
                default_model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-haiku-20240307"),
                models={},
                enabled=os.getenv("ANTHROPIC_ENABLED", "true").lower() == "true",
            )

        # Ollama configuration (enabled if OLLAMA_HOST is set or OLLAMA_ENABLED=true)
        ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
        ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
        if ollama_host or ollama_enabled:
            self.providers["ollama"] = ProviderConfig(
                provider="ollama",
                keys=[ProviderKey(api_key="", name="local")],
                default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b"),
                models={},
                enabled=True,
                base_url=ollama_host or "http://localhost:11434",
            )
    
    def register_custom_provider(
        self,
        provider_name: str,
        api_key: str,
        default_model: str,
        base_url: Optional[str] = None,
        key_name: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Register a custom provider programmatically.
        
        This is useful for on-prem LLM deployments or custom inference endpoints.
        
        Args:
            provider_name: Name of the provider (e.g., "onprem-llm", "company-llm")
            api_key: API key for authentication (can be empty string for no-auth endpoints)
            default_model: Default model name to use
            base_url: Base URL for the inference endpoint
            key_name: Optional name for the API key
            enabled: Whether the provider is enabled
        
        Example:
            >>> routing = LLMRoutingConfig()
            >>> routing.register_custom_provider(
            ...     provider_name="onprem-llm",
            ...     api_key="sk-custom-key",
            ...     default_model="llama-2-70b",
            ...     base_url="https://llm.company.com/v1",
            ... )
        """
        keys = [ProviderKey(api_key=api_key, name=key_name or "default")]
        
        self.providers[provider_name] = ProviderConfig(
            provider=provider_name,
            keys=keys,
            default_model=default_model,
            models={},
            enabled=enabled,
            base_url=base_url,
        )
        
        logger.info(f"Registered custom provider: {provider_name} with base_url={base_url}")

    def get_provider_and_model_for_tier(
        self,
        tier: str,
        excluded_models: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[ProviderKey]]:
        """
        Get provider, model, and API key for a tier using priority order.

        Models are tried in the order specified in TIER*_MODELS.
        If a model fails, the next one in the list is tried.

        Args:
            tier: The tier to get provider/model for ("tier1", "tier2", "tier3")
            excluded_models: List of "provider:model" strings to skip (e.g., rate-limited)

        Returns:
            Tuple of (provider_name, model_name, api_key) or (None, None, None) if unavailable
        """
        if tier not in self.tiers:
            logger.error(f"Unknown tier: {tier}")
            return None, None, None

        tier_config = self.tiers[tier]
        if not tier_config.enabled:
            logger.warning(f"Tier {tier} is disabled")
            return None, None, None

        if excluded_models is None:
            excluded_models = []

        # Try models in priority order
        for model_priority in tier_config.model_priority:
            provider_name = model_priority.provider
            model_name = model_priority.model
            model_key = f"{provider_name}:{model_name}"

            # Skip excluded models
            if model_key in excluded_models:
                logger.debug(f"Skipping excluded model {model_key}")
                continue

            # Check if provider is available
            if provider_name not in self.providers:
                logger.debug(f"Provider {provider_name} not configured, trying next")
                continue

            provider_config = self.providers[provider_name]
            if not provider_config.enabled:
                logger.debug(f"Provider {provider_name} is disabled, trying next")
                continue

            # Get next available API key (round-robin)
            key = self.get_next_healthy_key(provider_name)
            if not key:
                logger.warning(f"No healthy keys for {provider_name}, trying next model")
                continue

            logger.info(f"Selected {model_key} with key {key.name} for tier {tier}")
            return provider_name, model_name, key

        # No model available
        logger.error(f"No available model for tier {tier} (excluded: {excluded_models})")
        return None, None, None

    def get_next_healthy_key(self, provider: str) -> Optional[ProviderKey]:
        """
        Get next healthy API key using round-robin.
        Skips rate-limited or unhealthy keys.
        """
        if provider not in self.providers:
            return None

        provider_config = self.providers[provider]

        # Initialize key statuses if needed
        if provider not in self.key_statuses:
            self.key_statuses[provider] = {}
            for key in provider_config.keys:
                if key.enabled:
                    self.key_statuses[provider][key.name or "default"] = KeyStatus(key=key)

        # Get healthy keys
        healthy_keys = []
        for key in provider_config.keys:
            if not key.enabled:
                continue

            key_name = key.name or "default"
            status = self.key_statuses[provider].get(key_name)

            if status:
                # Skip if rate-limited (recover after 1 hour)
                if status.rate_limit_hit:
                    if status.last_failure_time:
                        time_since = (datetime.now() - status.last_failure_time).total_seconds()
                        if time_since > 3600:
                            logger.info(f"Key {key_name} for {provider} recovered from rate limit")
                            status.rate_limit_hit = False
                            status.consecutive_failures = 0
                        else:
                            continue
                    else:
                        continue

                # Skip if too many consecutive failures (recover after 5 minutes)
                if status.consecutive_failures >= 3:
                    if status.last_failure_time:
                        time_since = (datetime.now() - status.last_failure_time).total_seconds()
                        if time_since > 300:
                            logger.info(f"Key {key_name} for {provider} recovered from failures")
                            status.consecutive_failures = 0
                        else:
                            continue
                    else:
                        continue

            healthy_keys.append(key)

        if not healthy_keys:
            # All keys unhealthy, reset and return first key
            logger.warning(f"All keys unhealthy for {provider}, resetting")
            self._reset_key_statuses(provider)
            enabled_keys = [k for k in provider_config.keys if k.enabled]
            return enabled_keys[0] if enabled_keys else None

        # Round-robin through healthy keys
        if provider not in self._key_index:
            self._key_index[provider] = 0

        idx = self._key_index[provider]
        selected_key = healthy_keys[idx % len(healthy_keys)]
        self._key_index[provider] = (idx + 1) % len(healthy_keys)

        # Update status
        key_name = selected_key.name or "default"
        if provider in self.key_statuses and key_name in self.key_statuses[provider]:
            self.key_statuses[provider][key_name].last_used = datetime.now()

        return selected_key

    def _reset_key_statuses(self, provider: str):
        """Reset key statuses for a provider."""
        if provider in self.key_statuses:
            for status in self.key_statuses[provider].values():
                status.rate_limit_hit = False
                status.consecutive_failures = 0
                status.last_failure_time = None

    def record_success(self, provider: str, key_name: str):
        """Record successful API call."""
        if provider in self.key_statuses and key_name in self.key_statuses[provider]:
            status = self.key_statuses[provider][key_name]
            status.success_count += 1
            status.consecutive_failures = 0
            logger.debug(f"Recorded success for {provider}/{key_name}")

    def record_failure(self, provider: str, key_name: str, is_rate_limit: bool = False):
        """Record failed API call."""
        if provider in self.key_statuses and key_name in self.key_statuses[provider]:
            status = self.key_statuses[provider][key_name]
            status.failure_count += 1
            status.consecutive_failures += 1
            status.last_failure_time = datetime.now()
            if is_rate_limit:
                status.rate_limit_hit = True
                logger.warning(f"Rate limit hit for {provider}/{key_name}")
            else:
                logger.warning(f"Failure for {provider}/{key_name} (consecutive: {status.consecutive_failures})")

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {"providers": {}, "tiers": {}}

        for provider, config in self.providers.items():
            provider_stats = {
                "enabled": config.enabled,
                "key_count": len(config.keys),
                "keys": {},
            }
            if provider in self.key_statuses:
                for key_name, status in self.key_statuses[provider].items():
                    provider_stats["keys"][key_name] = {
                        "success_count": status.success_count,
                        "failure_count": status.failure_count,
                        "rate_limit_hit": status.rate_limit_hit,
                        "consecutive_failures": status.consecutive_failures,
                    }
            stats["providers"][provider] = provider_stats

        for tier_name, tier_config in self.tiers.items():
            stats["tiers"][tier_name] = {
                "enabled": tier_config.enabled,
                "model_priority": [f"{m.provider}:{m.model}" for m in tier_config.model_priority],
            }

        return stats
