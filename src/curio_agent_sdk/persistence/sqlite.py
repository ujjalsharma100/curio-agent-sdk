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
import hashlib

from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.models.agent import AgentRun, AgentRunEvent, AgentLLMUsage, AgentRunStatus

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

            # Audit log table (tamper-evident hash chain)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    run_id TEXT,
                    actor_type TEXT,
                    actor_id TEXT,
                    action TEXT,
                    resource TEXT,
                    resource_type TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    prev_hash TEXT,
                    hash TEXT,
                    created_at TEXT
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_run_id ON audit_logs(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_agent_id ON audit_logs(agent_id)")

            # Cost entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    agent_id TEXT,
                    model TEXT,
                    cost_usd REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cumulative_cost_usd REAL,
                    created_at TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_run_id ON cost_entries(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_model ON cost_entries(model)")

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
                run.total_iterations, run.final_output,
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
                run.total_iterations, run.final_output,
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
            final_output=row["final_synthesis_output"],
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
                getattr(usage, "prompt", None),
                getattr(usage, "prompt_length", None),
                getattr(usage, "input_params", None),
                usage.input_tokens, usage.output_tokens,
                getattr(usage, "response_content", None),
                getattr(usage, "response_length", None),
                getattr(usage, "usage_metrics", None),
                usage.status, usage.error_message, usage.latency_ms, now
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
            input_tokens=row["input_tokens"] or 0,
            output_tokens=row["output_tokens"] or 0,
            latency_ms=row["latency_ms"] or 0,
            status=row["status"] or "success",
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # ==================== Statistics ====================

    @staticmethod
    def _compute_percentiles(values: list[float], percentiles: list[int] | None = None) -> Dict[str, float]:
        """Compute percentile values from a sorted list of numbers."""
        if not values:
            return {}
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]
        values_sorted = sorted(values)
        n = len(values_sorted)
        result = {}
        for p in percentiles:
            idx = (p / 100) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            result[f"p{p}"] = round(values_sorted[lower] * (1 - weight) + values_sorted[upper] * weight, 2)
        return result

    def get_agent_run_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for agent runs, including extended analytics."""
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

            # Extended: tool metrics from agent_run_events
            tool_metrics: Dict[str, Any] = {}
            try:
                if agent_id:
                    cursor.execute("""
                        SELECT data FROM agent_run_events
                        WHERE agent_id = ? AND event_type = 'tool_call'
                    """, [agent_id])
                else:
                    cursor.execute("""
                        SELECT data FROM agent_run_events
                        WHERE event_type = 'tool_call'
                    """)
                for (data_str,) in cursor.fetchall():
                    if not data_str:
                        continue
                    try:
                        d = json.loads(data_str)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    tname = d.get("tool_name", "unknown")
                    if tname not in tool_metrics:
                        tool_metrics[tname] = {"call_count": 0, "total_latency_ms": 0, "error_count": 0}
                    tool_metrics[tname]["call_count"] += 1
                    tool_metrics[tname]["total_latency_ms"] += d.get("latency_ms", 0)
                    if d.get("error"):
                        tool_metrics[tname]["error_count"] += 1
                # Compute avg latency
                for tname in tool_metrics:
                    calls = tool_metrics[tname]["call_count"]
                    if calls > 0:
                        tool_metrics[tname]["avg_latency_ms"] = round(
                            tool_metrics[tname]["total_latency_ms"] / calls, 2
                        )
                    else:
                        tool_metrics[tname]["avg_latency_ms"] = 0
            except Exception:
                pass

            # Extended: latency percentiles from agent_llm_usage
            latency_values: list[float] = []
            input_tokens_list: list[int] = []
            output_tokens_list: list[int] = []
            try:
                cursor.execute(f"""
                    SELECT latency_ms, input_tokens, output_tokens
                    FROM agent_llm_usage {agent_filter}
                """, params)
                for row in cursor.fetchall():
                    if row[0] is not None:
                        latency_values.append(float(row[0]))
                    if row[1] is not None:
                        input_tokens_list.append(int(row[1]))
                    if row[2] is not None:
                        output_tokens_list.append(int(row[2]))
            except Exception:
                pass

            latency_percentiles = self._compute_percentiles(latency_values)

            # Token efficiency
            token_efficiency = (
                round(total_output_tokens / total_input_tokens, 4)
                if total_input_tokens > 0 else 0
            )
            avg_input_per_call = (
                round(total_input_tokens / total_llm_calls, 2)
                if total_llm_calls > 0 else 0
            )
            avg_output_per_call = (
                round(total_output_tokens / total_llm_calls, 2)
                if total_llm_calls > 0 else 0
            )

            return {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "error_runs": error_runs,
                "avg_iterations": round(avg_iterations, 2),
                "total_llm_calls": total_llm_calls,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                # Extended analytics
                "tool_metrics": tool_metrics,
                "latency_percentiles": latency_percentiles,
                "token_efficiency": token_efficiency,
                "avg_input_tokens_per_call": avg_input_per_call,
                "avg_output_tokens_per_call": avg_output_per_call,
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

    # ==================== Audit Logs ====================

    def _compute_audit_hash(self, payload: Dict[str, Any], prev_hash: str | None) -> str:
        data = {
            **payload,
            "prev_hash": prev_hash or "",
        }
        encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def log_audit_event(self, event: Any) -> None:
        """
        Log a structured audit event.

        Expected event shape (dict-like):
            {
                "agent_id": str | None,
                "run_id": str | None,
                "actor_type": "user" | "agent" | "system",
                "actor_id": str | None,
                "action": str,
                "resource": str | None,
                "resource_type": str | None,
                "metadata": dict | None,
                "timestamp": datetime | None,
            }
        """
        data = dict(event)
        agent_id = data.get("agent_id")
        run_id = data.get("run_id")
        actor_type = data.get("actor_type") or "agent"
        actor_id = data.get("actor_id")
        action = data.get("action") or ""
        resource = data.get("resource")
        resource_type = data.get("resource_type")
        metadata = data.get("metadata") or {}
        ts: datetime | None = data.get("timestamp")
        timestamp_str = (ts or datetime.now()).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT hash FROM audit_logs
                WHERE run_id = ? AND agent_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (run_id, agent_id),
            )
            row = cursor.fetchone()
            prev_hash = row["hash"] if row else None

            payload = {
                "agent_id": agent_id,
                "run_id": run_id,
                "actor_type": actor_type,
                "actor_id": actor_id,
                "action": action,
                "resource": resource,
                "resource_type": resource_type,
                "metadata": metadata,
                "timestamp": timestamp_str,
            }
            current_hash = self._compute_audit_hash(payload, prev_hash)

            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO audit_logs (
                    agent_id, run_id, actor_type, actor_id, action,
                    resource, resource_type, metadata, timestamp,
                    prev_hash, hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    run_id,
                    actor_type,
                    actor_id,
                    action,
                    resource,
                    resource_type,
                    json.dumps(metadata, default=str),
                    timestamp_str,
                    prev_hash,
                    current_hash,
                    now,
                ),
            )

    def get_audit_events(
        self,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve audit events, optionally filtered by run_id / agent_id."""
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params: list[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        events: List[Dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "id": row["id"],
                    "agent_id": row["agent_id"],
                    "run_id": row["run_id"],
                    "actor_type": row["actor_type"],
                    "actor_id": row["actor_id"],
                    "action": row["action"],
                    "resource": row["resource"],
                    "resource_type": row["resource_type"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "timestamp": row["timestamp"],
                    "prev_hash": row["prev_hash"],
                    "hash": row["hash"],
                    "created_at": row["created_at"],
                }
            )
        return events
