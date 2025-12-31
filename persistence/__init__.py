"""
Persistence module for Curio Agent SDK.

This module provides database abstraction for agent runs, events, and LLM usage tracking.
Supports multiple backends:
- PostgreSQL (production)
- SQLite (development/lightweight)
- In-memory (testing)
"""

from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.persistence.postgres import PostgresPersistence
from curio_agent_sdk.persistence.sqlite import SQLitePersistence
from curio_agent_sdk.persistence.memory import InMemoryPersistence

__all__ = [
    "BasePersistence",
    "PostgresPersistence",
    "SQLitePersistence",
    "InMemoryPersistence",
]
