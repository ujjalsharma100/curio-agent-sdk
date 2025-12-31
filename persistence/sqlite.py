"""
SQLite persistence implementation.

Lightweight database persistence suitable for development and
single-user deployments.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging
import os

from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.core.models import AgentRun, AgentRunEvent, AgentLLMUsage, AgentRunStatus

logger = logging.getLogger(__name__)


class SQLitePersistence(BasePersistence):
    """
    SQLite persistence implementation.

    Provides lightweight database persistence using SQLite.
    Suitable for development, testing, and single-user deployments.

    Example:
        >>> # Use default database file
        >>> persistence = SQLitePersistence()
        >>>
        >>> # Use custom database path
        >>> persistence = SQLitePersistence("./data/agent.db")
        >>>
        >>> # Initialize schema
        >>> persistence.initialize_schema()
        >>>
        >>> # Use with agent
        >>> agent = MyAgent("test-agent", persistence=persistence)
    """

    def __init__(self, db_path: str = "agent_sdk.db"):
        """
        Initialize SQLite persistence.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create directory if needed
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.initialize_schema()

    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Agent runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    run_id TEXT UNIQUE NOT NULL,
                    agent_name TEXT,
                    objective TEXT,
                    additional_context TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    total_iterations INTEGER DEFAULT 0,
                    final_synthesis_output TEXT,
                    execution_history TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Agent run events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    agent_name TEXT,
                    timestamp TEXT,
                    event_type TEXT,
                    data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
                )
            """)

            # LLM usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    run_id TEXT,
                    provider TEXT,
                    model TEXT,
                    prompt TEXT,
                    prompt_length INTEGER,
                    input_params TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    response_content TEXT,
                    response_length INTEGER,
                    usage_metrics TEXT,
                    status TEXT DEFAULT 'success',
                    error_message TEXT,
                    latency_ms INTEGER,
                    created_at TEXT
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_agent_id ON agent_runs(agent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON agent_run_events(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_agent_id ON agent_llm_usage(agent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_run_id ON agent_llm_usage(run_id)")

            logger.debug("SQLite schema initialized")

    # ==================== Agent Runs ====================

    def create_agent_run(self, run: AgentRun) -> None:
        """Create a new agent run record."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_runs (
                    agent_id, run_id, agent_name, objective, additional_context,
                    started_at, finished_at, total_iterations, final_synthesis_output,
                    execution_history, status, error_message, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.agent_id, run.run_id, run.agent_name, run.objective,
                run.additional_context,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.total_iterations, run.final_synthesis_output,
                run.execution_history, run.status, run.error_message,
                run.metadata, now, now
            ))
            run.id = cursor.lastrowid
            logger.debug(f"Created agent run: {run.run_id}")

    def update_agent_run(self, run_id: str, run: AgentRun) -> None:
        """Update an existing agent run record."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE agent_runs SET
                    agent_name = ?, objective = ?, additional_context = ?,
                    started_at = ?, finished_at = ?, total_iterations = ?,
                    final_synthesis_output = ?, execution_history = ?,
                    status = ?, error_message = ?, metadata = ?, updated_at = ?
                WHERE run_id = ?
            """, (
                run.agent_name, run.objective, run.additional_context,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.total_iterations, run.final_synthesis_output,
                run.execution_history, run.status, run.error_message,
                run.metadata, now, run_id
            ))
            logger.debug(f"Updated agent run: {run_id}")

    def get_agent_run(self, run_id: str) -> Optional[AgentRun]:
        """Get an agent run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_agent_run(row)
            return None

    def get_agent_runs(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[AgentRun]:
        """Get agent runs with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if agent_id:
                cursor.execute("""
                    SELECT * FROM agent_runs
                    WHERE agent_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (agent_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM agent_runs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            return [self._row_to_agent_run(row) for row in cursor.fetchall()]

    def delete_agent_run(self, run_id: str) -> bool:
        """Delete an agent run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agent_run_events WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted agent run: {run_id}")
            return deleted

    def _row_to_agent_run(self, row: sqlite3.Row) -> AgentRun:
        """Convert SQLite row to AgentRun."""
        return AgentRun(
            id=row["id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            agent_name=row["agent_name"],
            objective=row["objective"],
            additional_context=row["additional_context"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            total_iterations=row["total_iterations"],
            final_synthesis_output=row["final_synthesis_output"],
            execution_history=row["execution_history"],
            status=row["status"],
            error_message=row["error_message"],
            metadata=row["metadata"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    # ==================== Agent Run Events ====================

    def log_agent_run_event(self, event: AgentRunEvent) -> None:
        """Log an agent run event."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_run_events (
                    agent_id, run_id, agent_name, timestamp, event_type, data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.agent_id, event.run_id, event.agent_name,
                event.timestamp.isoformat() if event.timestamp else now,
                event.event_type, event.data, now
            ))
            event.id = cursor.lastrowid

    def get_agent_run_events(
        self,
        run_id: str,
        event_type: Optional[str] = None,
    ) -> List[AgentRunEvent]:
        """Get events for an agent run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if event_type:
                cursor.execute("""
                    SELECT * FROM agent_run_events
                    WHERE run_id = ? AND event_type = ?
                    ORDER BY timestamp
                """, (run_id, event_type))
            else:
                cursor.execute("""
                    SELECT * FROM agent_run_events
                    WHERE run_id = ?
                    ORDER BY timestamp
                """, (run_id,))
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def _row_to_event(self, row: sqlite3.Row) -> AgentRunEvent:
        """Convert SQLite row to AgentRunEvent."""
        return AgentRunEvent(
            id=row["id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            agent_name=row["agent_name"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
            event_type=row["event_type"],
            data=row["data"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # ==================== LLM Usage ====================

    def log_llm_usage(self, usage: AgentLLMUsage) -> None:
        """Log LLM usage for tracking."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_llm_usage (
                    agent_id, run_id, provider, model, prompt, prompt_length,
                    input_params, input_tokens, output_tokens, response_content,
                    response_length, usage_metrics, status, error_message,
                    latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                usage.agent_id, usage.run_id, usage.provider, usage.model,
                usage.prompt, usage.prompt_length, usage.input_params,
                usage.input_tokens, usage.output_tokens, usage.response_content,
                usage.response_length, usage.usage_metrics, usage.status,
                usage.error_message, usage.latency_ms, now
            ))
            usage.id = cursor.lastrowid

    def get_llm_usage(
        self,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentLLMUsage]:
        """Get LLM usage records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM agent_llm_usage WHERE 1=1"
            params = []

            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [self._row_to_llm_usage(row) for row in cursor.fetchall()]

    def _row_to_llm_usage(self, row: sqlite3.Row) -> AgentLLMUsage:
        """Convert SQLite row to AgentLLMUsage."""
        return AgentLLMUsage(
            id=row["id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            provider=row["provider"],
            model=row["model"],
            prompt=row["prompt"],
            prompt_length=row["prompt_length"],
            input_params=row["input_params"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            response_content=row["response_content"],
            response_length=row["response_length"],
            usage_metrics=row["usage_metrics"],
            status=row["status"],
            error_message=row["error_message"],
            latency_ms=row["latency_ms"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # ==================== Statistics ====================

    def get_agent_run_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for agent runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Base filter
            agent_filter = "WHERE agent_id = ?" if agent_id else ""
            params = [agent_id] if agent_id else []

            # Total runs
            cursor.execute(f"SELECT COUNT(*) FROM agent_runs {agent_filter}", params)
            total_runs = cursor.fetchone()[0]

            # Completed runs
            cursor.execute(f"""
                SELECT COUNT(*) FROM agent_runs {agent_filter}
                {"AND" if agent_id else "WHERE"} status = 'completed'
            """, params)
            completed_runs = cursor.fetchone()[0]

            # Error runs
            cursor.execute(f"""
                SELECT COUNT(*) FROM agent_runs {agent_filter}
                {"AND" if agent_id else "WHERE"} status = 'error'
            """, params)
            error_runs = cursor.fetchone()[0]

            # Average iterations
            cursor.execute(f"SELECT AVG(total_iterations) FROM agent_runs {agent_filter}", params)
            avg_iterations = cursor.fetchone()[0] or 0

            # LLM usage stats
            cursor.execute(f"""
                SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens)
                FROM agent_llm_usage {agent_filter}
            """, params)
            row = cursor.fetchone()
            total_llm_calls = row[0] or 0
            total_input_tokens = row[1] or 0
            total_output_tokens = row[2] or 0

            return {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "error_runs": error_runs,
                "avg_iterations": round(avg_iterations, 2),
                "total_llm_calls": total_llm_calls,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            }

    def close(self) -> None:
        """Close database connections (no-op for SQLite with per-call connections)."""
        pass

    def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            with self._get_connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return False
