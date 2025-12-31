"""
PostgreSQL persistence implementation.

Production-ready database persistence using PostgreSQL.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging

from curio_agent_sdk.persistence.base import BasePersistence
from curio_agent_sdk.core.models import AgentRun, AgentRunEvent, AgentLLMUsage

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

    def get_agent_run_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get statistics for agent runs."""
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

            return {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "error_runs": error_runs,
                "avg_iterations": round(float(avg_iterations), 2),
                "total_llm_calls": row['count'],
                "total_input_tokens": row['input_tokens'],
                "total_output_tokens": row['output_tokens'],
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
