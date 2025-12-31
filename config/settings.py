"""
Configuration settings for Curio Agent SDK.

This module provides configuration management through environment variables
and programmatic configuration.
"""

import os
from typing import Optional, Dict, Any, List, Type
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class DatabaseConfig:
    """
    Configuration for database persistence.

    Supports PostgreSQL and SQLite backends.

    Environment Variables:
        DB_TYPE: Database type ("postgres", "sqlite", "memory")
        DB_HOST: PostgreSQL host
        DB_PORT: PostgreSQL port
        DB_NAME: Database name
        DB_USER: Username
        DB_PASSWORD: Password
        DB_SCHEMA: Schema name
        DB_PATH: SQLite database file path
    """
    db_type: str = "sqlite"  # "postgres", "sqlite", "memory"
    host: str = "localhost"
    port: int = 5432
    database: str = "agent_sdk"
    user: str = "postgres"
    password: str = ""
    schema: str = "agent_sdk"
    sqlite_path: str = "agent_sdk.db"
    min_connections: int = 1
    max_connections: int = 10

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            db_type=os.getenv("DB_TYPE", "sqlite").lower(),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "agent_sdk"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            schema=os.getenv("DB_SCHEMA", "agent_sdk"),
            sqlite_path=os.getenv("DB_PATH", "agent_sdk.db"),
            min_connections=int(os.getenv("DB_MIN_CONNECTIONS", "1")),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10")),
        )

    def get_persistence(self):
        """Create and return the appropriate persistence instance."""
        from curio_agent_sdk.persistence import (
            PostgresPersistence,
            SQLitePersistence,
            InMemoryPersistence,
        )

        if self.db_type == "postgres":
            return PostgresPersistence(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                schema=self.schema,
                min_connections=self.min_connections,
                max_connections=self.max_connections,
            )
        elif self.db_type == "sqlite":
            return SQLitePersistence(db_path=self.sqlite_path)
        else:  # memory
            return InMemoryPersistence()


@dataclass
class LLMProviderConfig:
    """
    Configuration for a single LLM provider.

    Attributes:
        provider: Provider name (openai, anthropic, groq, ollama)
        api_keys: List of API keys for rotation
        default_model: Default model to use
        enabled: Whether this provider is enabled
        base_url: Custom base URL (for Ollama or custom endpoints)
    """
    provider: str
    api_keys: List[str] = field(default_factory=list)
    default_model: str = ""
    enabled: bool = True
    base_url: Optional[str] = None


@dataclass
class TierConfiguration:
    """
    Configuration for a model tier.

    Attributes:
        tier_name: Name of the tier (tier1, tier2, tier3)
        model_priority: Ordered list of "provider:model" strings (first = highest priority)
        enabled: Whether this tier is enabled
    """
    tier_name: str
    model_priority: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AgentConfig:
    """
    Main configuration for the Curio Agent SDK.

    This class holds all configuration needed to run agents including
    database settings, LLM provider settings, and tier configurations.

    Example:
        >>> # Load from environment
        >>> config = AgentConfig.from_env()
        >>>
        >>> # Programmatic configuration
        >>> config = AgentConfig(
        ...     database=DatabaseConfig(db_type="sqlite"),
        ...     default_tier="tier2",
        ... )
        >>>
        >>> # Get services
        >>> persistence = config.get_persistence()
        >>> llm_service = config.get_llm_service()

    Environment Variables:
        See DatabaseConfig and individual provider configurations.

    Provider Environment Variables:
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key
        GROQ_API_KEY: Groq API key
        OLLAMA_HOST: Ollama host URL

    Tier Environment Variables:
        TIER1_MODELS: Model priority list (provider:model,provider:model)
        TIER2_MODELS: Model priority list for tier2
        TIER3_MODELS: Model priority list for tier3
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    tiers: Dict[str, TierConfiguration] = field(default_factory=dict)
    default_tier: str = "tier2"
    log_level: str = "INFO"
    max_retries: int = 10
    request_timeout: int = 60

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """
        Load complete configuration from environment variables.

        This is the recommended way to configure the SDK in production.
        """
        config = cls(
            database=DatabaseConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_retries=int(os.getenv("MAX_RETRIES", "10")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            default_tier=os.getenv("DEFAULT_TIER", "tier2"),
        )

        # Load provider configurations
        config._load_providers_from_env()

        # Load tier configurations
        config._load_tiers_from_env()

        return config

    def _load_providers_from_env(self):
        """Load provider configurations from environment."""
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.providers["openai"] = LLMProviderConfig(
                provider="openai",
                api_keys=self._load_api_keys("OPENAI"),
                default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
                enabled=os.getenv("OPENAI_ENABLED", "true").lower() == "true",
            )

        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.providers["anthropic"] = LLMProviderConfig(
                provider="anthropic",
                api_keys=self._load_api_keys("ANTHROPIC"),
                default_model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-haiku-20240307"),
                enabled=os.getenv("ANTHROPIC_ENABLED", "true").lower() == "true",
            )

        # Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key or os.getenv("GROQ_API_KEY_1"):
            self.providers["groq"] = LLMProviderConfig(
                provider="groq",
                api_keys=self._load_api_keys("GROQ"),
                default_model=os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant"),
                enabled=os.getenv("GROQ_ENABLED", "true").lower() == "true",
            )

        # Ollama
        ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")
        ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
        if ollama_host or ollama_enabled:
            self.providers["ollama"] = LLMProviderConfig(
                provider="ollama",
                api_keys=[],
                default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b"),
                enabled=True,
                base_url=ollama_host or "http://localhost:11434",
            )

    def _load_api_keys(self, provider_prefix: str) -> List[str]:
        """Load multiple API keys for a provider."""
        keys = []

        # Single key
        single_key = os.getenv(f"{provider_prefix}_API_KEY")
        if single_key:
            keys.append(single_key)

        # Multiple keys
        i = 1
        while True:
            key = os.getenv(f"{provider_prefix}_API_KEY_{i}")
            if not key:
                break
            enabled = os.getenv(f"{provider_prefix}_API_KEY_{i}_ENABLED", "true").lower() == "true"
            if enabled:
                keys.append(key)
            i += 1

        return keys

    def _load_tiers_from_env(self):
        """Load tier configurations from environment."""
        for tier_name in ["tier1", "tier2", "tier3"]:
            models_env = os.getenv(f"{tier_name.upper()}_MODELS", "")

            # Parse model priority list: "provider:model,provider:model"
            model_priority = []
            if models_env:
                for item in models_env.split(","):
                    item = item.strip()
                    if ":" in item:
                        model_priority.append(item)

            self.tiers[tier_name] = TierConfiguration(
                tier_name=tier_name,
                model_priority=model_priority,
                enabled=os.getenv(f"{tier_name.upper()}_ENABLED", "true").lower() == "true",
            )

    def get_persistence(self):
        """Get the configured persistence instance."""
        return self.database.get_persistence()

    def get_llm_service(self, custom_providers: Optional[Dict[str, type]] = None):
        """
        Get the configured LLM service instance.
        
        Args:
            custom_providers: Optional dict of provider_name -> ProviderClass for custom providers
                             (e.g., on-prem LLM providers)
        
        Returns:
            Configured LLMService instance
        """
        from curio_agent_sdk.llm import LLMService, LLMRoutingConfig

        # Build custom tier config from our settings
        custom_tiers = {}
        for tier_name, tier_config in self.tiers.items():
            if tier_config.model_priority:
                custom_tiers[tier_name] = tier_config.model_priority

        routing_config = LLMRoutingConfig(custom_tiers=custom_tiers if custom_tiers else None)
        
        # Add custom providers from provider configs that have base_url set
        # (these are likely on-prem or custom endpoints)
        if custom_providers is None:
            custom_providers = {}
        
        # Check for providers with custom base_url (likely on-prem)
        for provider_name, provider_config in self.providers.items():
            if provider_config.base_url and provider_name not in ["ollama"]:
                # If it's not a standard provider, assume OpenAI-compatible API
                if provider_name not in ["openai", "anthropic", "groq", "ollama"]:
                    from curio_agent_sdk.llm.providers.openai import OpenAIProvider
                    custom_providers[provider_name] = OpenAIProvider

        return LLMService(
            config=self,
            persistence=self.get_persistence(),
            routing_config=routing_config,
            custom_providers=custom_providers if custom_providers else None,
        )

    def configure_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )


def load_config_from_env() -> AgentConfig:
    """
    Convenience function to load configuration from environment.

    Returns:
        AgentConfig loaded from environment variables
    """
    return AgentConfig.from_env()
