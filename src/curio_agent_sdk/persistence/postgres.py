"""
PostgreSQL persistence implementation.

Production-ready database persistence using PostgreSQL.
"""

import json
import math
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging
import hashlib

from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.models.agent import AgentRun, AgentRunEvent, AgentLLMUsage

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")


class PostgresPersistence(BasePersistence):
    """
    PostgreSQL persistence implementation.

    Production-ready database persistence using PostgreSQL with
    connection pooling.

    Example:
        >>> persistence = PostgresPersistence(
        ...     host="localhost",
        ...     port=5432,
        ...     database="agent_db",
        ...     user="postgres",
        ...     password="password",
        ...     schema="agent",
        ... )
        >>>
        >>> # Initialize schema
        >>> persistence.initialize_schema()
        >>>
        >>> # Use with agent
        >>> agent = MyAgent("prod-agent", persistence=persistence)

    Environment Variables:
        DB_HOST: PostgreSQL host
        DB_PORT: PostgreSQL port
        DB_NAME: Database name
        DB_USER: Username
        DB_PASSWORD: Password
        DB_SCHEMA: Schema name (default: agent)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "agent_sdk",
        user: str = "postgres",
        password: str = "",
        schema: str = "agent_sdk",
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """
        Initialize PostgreSQL persistence.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            schema: Schema name
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
        """
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema

        # Create connection pool
        self._pool = pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )

        logger.info(f"PostgreSQL connection pool created for {host}:{port}/{database}")

    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _table(self, name: str) -> str:
        """Get fully qualified table name."""
        return f"{self.schema}.{name}"

    def initialize_schema(self) -> None:
        """Create schema and tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create schema
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Agent runs table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table('agent_runs')} (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) UNIQUE NOT NULL,
                    agent_name VARCHAR(255),
                    objective TEXT,
                    additional_context TEXT,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    total_iterations INTEGER DEFAULT 0,
                    final_synthesis_output TEXT,
                    execution_history TEXT,
                    status VARCHAR(50) DEFAULT 'pending',
                    error_message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Agent run events table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table('agent_run_events')} (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL,
                    run_id VARCHAR(255) NOT NULL,
                    agent_name VARCHAR(255),
                    timestamp TIMESTAMP,
                    event_type VARCHAR(100),
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES {self._table('agent_runs')}(run_id)
                )
            """)

            # LLM usage table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table('agent_llm_usage')} (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255),
                    run_id VARCHAR(255),
                    provider VARCHAR(100),
                    model VARCHAR(255),
                    prompt TEXT,
                    prompt_length INTEGER,
                    input_params TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    response_content TEXT,
                    response_length INTEGER,
                    usage_metrics TEXT,
                    status VARCHAR(50) DEFAULT 'success',
                    error_message TEXT,
                    latency_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_runs_agent_id
                ON {self._table('agent_runs')}(agent_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_events_run_id
                ON {self._table('agent_run_events')}(run_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_usage_agent_id
                ON {self._table('agent_llm_usage')}(agent_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_usage_run_id
                ON {self._table('agent_llm_usage')}(run_id)
            """)

            # Audit log table (tamper-evident hash chain)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table('audit_logs')} (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255),
                    run_id VARCHAR(255),
                    actor_type VARCHAR(50),
                    actor_id VARCHAR(255),
                    action VARCHAR(255),
                    resource TEXT,
                    resource_type VARCHAR(100),
                    metadata JSONB,
                    timestamp TIMESTAMP,
                    prev_hash VARCHAR(128),
                    hash VARCHAR(128),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_run_id
                ON {self._table('audit_logs')}(run_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_agent_id
                ON {self._table('audit_logs')}(agent_id)
            """)

            # Cost entries table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table('cost_entries')} (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255),
                    agent_id VARCHAR(255),
                    model VARCHAR(255),
                    cost_usd DOUBLE PRECISION,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cumulative_cost_usd DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_cost_run_id
                ON {self._table('cost_entries')}(run_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_cost_model
                ON {self._table('cost_entries')}(model)
            """)

            logger.info(f"PostgreSQL schema initialized: {self.schema}")

    # ==================== Agent Runs ====================

    def create_agent_run(self, run: AgentRun) -> None:
        """Create a new agent run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f"""
                INSERT INTO {self._table('agent_runs')} (
                    agent_id, run_id, agent_name, objective, additional_context,
                    started_at, finished_at, total_iterations, final_synthesis_output,
                    execution_history, status, error_message, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                run.agent_id, run.run_id, run.agent_name, run.objective,
                run.additional_context, run.started_at, run.finished_at,
                run.total_iterations, run.final_synthesis_output,
                run.execution_history, run.status, run.error_message, run.metadata
            ))
            result = cursor.fetchone()
            run.id = result['id']
            logger.debug(f"Created agent run: {run.run_id}")

    def update_agent_run(self, run_id: str, run: AgentRun) -> None:
        """Update an existing agent run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE {self._table('agent_runs')} SET
                    agent_name = %s, objective = %s, additional_context = %s,
                    started_at = %s, finished_at = %s, total_iterations = %s,
                    final_synthesis_output = %s, execution_history = %s,
                    status = %s, error_message = %s, metadata = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s
            """, (
                run.agent_name, run.objective, run.additional_context,
                run.started_at, run.finished_at, run.total_iterations,
                run.final_synthesis_output, run.execution_history,
                run.status, run.error_message, run.metadata, run_id
            ))
            logger.debug(f"Updated agent run: {run_id}")

    def get_agent_run(self, run_id: str) -> Optional[AgentRun]:
        """Get an agent run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                f"SELECT * FROM {self._table('agent_runs')} WHERE run_id = %s",
                (run_id,)
            )
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
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if agent_id:
                cursor.execute(f"""
                    SELECT * FROM {self._table('agent_runs')}
                    WHERE agent_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (agent_id, limit, offset))
            else:
                cursor.execute(f"""
                    SELECT * FROM {self._table('agent_runs')}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
            return [self._row_to_agent_run(row) for row in cursor.fetchall()]

    def delete_agent_run(self, run_id: str) -> bool:
        """Delete an agent run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self._table('agent_run_events')} WHERE run_id = %s",
                (run_id,)
            )
            cursor.execute(
                f"DELETE FROM {self._table('agent_runs')} WHERE run_id = %s",
                (run_id,)
            )
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted agent run: {run_id}")
            return deleted

    def _row_to_agent_run(self, row: Dict) -> AgentRun:
        """Convert database row to AgentRun."""
        return AgentRun(
            id=row.get("id"),
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            agent_name=row.get("agent_name"),
            objective=row.get("objective"),
            additional_context=row.get("additional_context"),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            total_iterations=row.get("total_iterations", 0),
            final_synthesis_output=row.get("final_synthesis_output"),
            execution_history=row.get("execution_history"),
            status=row.get("status", "pending"),
            error_message=row.get("error_message"),
            metadata=row.get("metadata"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    # ==================== Agent Run Events ====================

    def log_agent_run_event(self, event: AgentRunEvent) -> None:
        """Log an agent run event."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f"""
                INSERT INTO {self._table('agent_run_events')} (
                    agent_id, run_id, agent_name, timestamp, event_type, data
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                event.agent_id, event.run_id, event.agent_name,
                event.timestamp or datetime.now(), event.event_type, event.data
            ))
            result = cursor.fetchone()
            event.id = result['id']

    def get_agent_run_events(
        self,
        run_id: str,
        event_type: Optional[str] = None,
    ) -> List[AgentRunEvent]:
        """Get events for an agent run."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if event_type:
                cursor.execute(f"""
                    SELECT * FROM {self._table('agent_run_events')}
                    WHERE run_id = %s AND event_type = %s
                    ORDER BY timestamp
                """, (run_id, event_type))
            else:
                cursor.execute(f"""
                    SELECT * FROM {self._table('agent_run_events')}
                    WHERE run_id = %s
                    ORDER BY timestamp
                """, (run_id,))
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def _row_to_event(self, row: Dict) -> AgentRunEvent:
        """Convert database row to AgentRunEvent."""
        return AgentRunEvent(
            id=row.get("id"),
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            agent_name=row.get("agent_name"),
            timestamp=row.get("timestamp"),
            event_type=row.get("event_type"),
            data=row.get("data"),
            created_at=row.get("created_at"),
        )

    # ==================== LLM Usage ====================

    def log_llm_usage(self, usage: AgentLLMUsage) -> None:
        """Log LLM usage for tracking."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f"""
                INSERT INTO {self._table('agent_llm_usage')} (
                    agent_id, run_id, provider, model, prompt, prompt_length,
                    input_params, input_tokens, output_tokens, response_content,
                    response_length, usage_metrics, status, error_message, latency_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                usage.agent_id, usage.run_id, usage.provider, usage.model,
                usage.prompt, usage.prompt_length, usage.input_params,
                usage.input_tokens, usage.output_tokens, usage.response_content,
                usage.response_length, usage.usage_metrics, usage.status,
                usage.error_message, usage.latency_ms
            ))
            result = cursor.fetchone()
            usage.id = result['id']

    def get_llm_usage(
        self,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentLLMUsage]:
        """Get LLM usage records."""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = f"SELECT * FROM {self._table('agent_llm_usage')} WHERE 1=1"
            params = []

            if agent_id:
                query += " AND agent_id = %s"
                params.append(agent_id)
            if run_id:
                query += " AND run_id = %s"
                params.append(run_id)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            return [self._row_to_llm_usage(row) for row in cursor.fetchall()]

    def _row_to_llm_usage(self, row: Dict) -> AgentLLMUsage:
        """Convert database row to AgentLLMUsage."""
        return AgentLLMUsage(
            id=row.get("id"),
            agent_id=row.get("agent_id"),
            run_id=row.get("run_id"),
            provider=row.get("provider", ""),
            model=row.get("model", ""),
            prompt=row.get("prompt", ""),
            prompt_length=row.get("prompt_length", 0),
            input_params=row.get("input_params"),
            input_tokens=row.get("input_tokens"),
            output_tokens=row.get("output_tokens"),
            response_content=row.get("response_content"),
            response_length=row.get("response_length"),
            usage_metrics=row.get("usage_metrics"),
            status=row.get("status", "success"),
            error_message=row.get("error_message"),
            latency_ms=row.get("latency_ms"),
            created_at=row.get("created_at"),
        )

    # ==================== Statistics ====================

    @staticmethod
    def _compute_percentiles(values: list[float], percentiles: list[int] | None = None) -> Dict[str, float]:
        """Compute percentile values from a list of numbers."""
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
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build filter
            agent_filter = "WHERE agent_id = %s" if agent_id else ""
            params = [agent_id] if agent_id else []

            # Total runs
            cursor.execute(
                f"SELECT COUNT(*) as count FROM {self._table('agent_runs')} {agent_filter}",
                params
            )
            total_runs = cursor.fetchone()['count']

            # Completed runs
            cursor.execute(f"""
                SELECT COUNT(*) as count FROM {self._table('agent_runs')} {agent_filter}
                {"AND" if agent_id else "WHERE"} status = 'completed'
            """, params)
            completed_runs = cursor.fetchone()['count']

            # Error runs
            cursor.execute(f"""
                SELECT COUNT(*) as count FROM {self._table('agent_runs')} {agent_filter}
                {"AND" if agent_id else "WHERE"} status = 'error'
            """, params)
            error_runs = cursor.fetchone()['count']

            # Average iterations
            cursor.execute(
                f"SELECT AVG(total_iterations) as avg FROM {self._table('agent_runs')} {agent_filter}",
                params
            )
            avg_iterations = cursor.fetchone()['avg'] or 0

            # LLM usage stats
            cursor.execute(f"""
                SELECT COUNT(*) as count,
                       COALESCE(SUM(input_tokens), 0) as input_tokens,
                       COALESCE(SUM(output_tokens), 0) as output_tokens
                FROM {self._table('agent_llm_usage')} {agent_filter}
            """, params)
            row = cursor.fetchone()
            total_llm_calls = row['count']
            total_input_tokens = row['input_tokens']
            total_output_tokens = row['output_tokens']

            # Extended: tool metrics
            tool_metrics: Dict[str, Any] = {}
            try:
                if agent_id:
                    cursor.execute(f"""
                        SELECT data FROM {self._table('agent_run_events')}
                        WHERE agent_id = %s AND event_type = 'tool_call'
                    """, [agent_id])
                else:
                    cursor.execute(f"""
                        SELECT data FROM {self._table('agent_run_events')}
                        WHERE event_type = 'tool_call'
                    """)
                for row in cursor.fetchall():
                    data_str = row.get("data")
                    if not data_str:
                        continue
                    try:
                        d = json.loads(data_str) if isinstance(data_str, str) else data_str
                    except (json.JSONDecodeError, TypeError):
                        continue
                    tname = d.get("tool_name", "unknown")
                    if tname not in tool_metrics:
                        tool_metrics[tname] = {"call_count": 0, "total_latency_ms": 0, "error_count": 0}
                    tool_metrics[tname]["call_count"] += 1
                    tool_metrics[tname]["total_latency_ms"] += d.get("latency_ms", 0)
                    if d.get("error"):
                        tool_metrics[tname]["error_count"] += 1
                for tname in tool_metrics:
                    calls = tool_metrics[tname]["call_count"]
                    tool_metrics[tname]["avg_latency_ms"] = (
                        round(tool_metrics[tname]["total_latency_ms"] / calls, 2) if calls else 0
                    )
            except Exception:
                pass

            # Extended: latency percentiles
            latency_values: list[float] = []
            try:
                cursor.execute(f"""
                    SELECT latency_ms FROM {self._table('agent_llm_usage')}
                    {agent_filter}
                """, params)
                for row in cursor.fetchall():
                    val = row.get("latency_ms")
                    if val is not None:
                        latency_values.append(float(val))
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
                "avg_iterations": round(float(avg_iterations), 2),
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
        """Close the connection pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
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
        timestamp = ts or datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                f"""
                SELECT hash FROM {self._table('audit_logs')}
                WHERE run_id = %s AND agent_id = %s
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
                "timestamp": timestamp.isoformat(),
            }
            current_hash = self._compute_audit_hash(payload, prev_hash)

            cursor.execute(
                f"""
                INSERT INTO {self._table('audit_logs')} (
                    agent_id, run_id, actor_type, actor_id, action,
                    resource, resource_type, metadata, timestamp,
                    prev_hash, hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    timestamp,
                    prev_hash,
                    current_hash,
                ),
            )

    def get_audit_events(
        self,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve audit events, optionally filtered by run_id / agent_id."""
        query = f"SELECT * FROM {self._table('audit_logs')} WHERE 1=1"
        params: list[Any] = []
        if run_id:
            query += " AND run_id = %s"
            params.append(run_id)
        if agent_id:
            query += " AND agent_id = %s"
            params.append(agent_id)
        query += " ORDER BY id DESC LIMIT %s"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()

        events: List[Dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "id": row.get("id"),
                    "agent_id": row.get("agent_id"),
                    "run_id": row.get("run_id"),
                    "actor_type": row.get("actor_type"),
                    "actor_id": row.get("actor_id"),
                    "action": row.get("action"),
                    "resource": row.get("resource"),
                    "resource_type": row.get("resource_type"),
                    "metadata": row.get("metadata") or {},
                    "timestamp": row.get("timestamp"),
                    "prev_hash": row.get("prev_hash"),
                    "hash": row.get("hash"),
                    "created_at": row.get("created_at"),
                }
            )
        return events
