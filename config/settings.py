"""
Configuration for the Curio Agent SDK.

Simplified configuration that creates pre-configured components.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class DatabaseConfig:
    """Database configuration for persistence."""
    db_type: str = "sqlite"  # "postgres", "sqlite", "memory"
    host: str = "localhost"
    port: int = 5432
    database: str = "agent_sdk"
    user: str = "postgres"
    password: str = ""
    schema: str = "agent_sdk"
    sqlite_path: str = "agent_sdk.db"

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        return cls(
            db_type=os.getenv("DB_TYPE", "sqlite").lower(),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "agent_sdk"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            schema=os.getenv("DB_SCHEMA", "agent_sdk"),
            sqlite_path=os.getenv("DB_PATH", "agent_sdk.db"),
        )

    def get_persistence(self):
        """Create the appropriate persistence backend."""
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
            )
        elif self.db_type == "sqlite":
            return SQLitePersistence(db_path=self.sqlite_path)
        else:
            return InMemoryPersistence()


@dataclass
class AgentConfig:
    """
    Main SDK configuration.

    Example:
        config = AgentConfig.from_env()
        agent = Agent(
            llm=config.create_llm_client(),
            system_prompt="...",
            tools=[...],
        )
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    default_tier: str = "tier2"
    log_level: str = "INFO"
    max_retries: int = 10
    request_timeout: int = 60

    @classmethod
    def from_env(cls) -> AgentConfig:
        return cls(
            database=DatabaseConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_retries=int(os.getenv("MAX_RETRIES", "10")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            default_tier=os.getenv("DEFAULT_TIER", "tier2"),
        )

    def create_llm_client(self, **kwargs):
        """Create an LLMClient with routing configured from environment."""
        from curio_agent_sdk.llm.client import LLMClient
        from curio_agent_sdk.llm.router import TieredRouter

        router = TieredRouter(**kwargs) if kwargs else TieredRouter()
        return LLMClient(router=router)

    def get_persistence(self):
        """Get the configured persistence backend."""
        return self.database.get_persistence()

    def configure_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
