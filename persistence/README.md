# Persistence Module Documentation

## Overview

The persistence module provides a unified database abstraction layer for storing and retrieving agent runs, events, and LLM usage data. It supports multiple backends (PostgreSQL, SQLite, in-memory) through a common interface, making it easy to switch between different storage solutions based on your needs.

## Architecture

### BasePersistence Interface

All persistence implementations inherit from `BasePersistence`, which defines the standard interface:

```python
class BasePersistence(ABC):
    # Agent Runs
    def create_agent_run(self, run: AgentRun) -> None
    def update_agent_run(self, run_id: str, run: AgentRun) -> None
    def get_agent_run(self, run_id: str) -> Optional[AgentRun]
    def get_agent_runs(self, agent_id: Optional[str] = None, limit: int = 10, offset: int = 0) -> List[AgentRun]
    def delete_agent_run(self, run_id: str) -> bool
    
    # Agent Run Events
    def log_agent_run_event(self, event: AgentRunEvent) -> None
    def get_agent_run_events(self, run_id: str, event_type: Optional[str] = None) -> List[AgentRunEvent]
    
    # LLM Usage
    def log_llm_usage(self, usage: AgentLLMUsage) -> None
    def get_llm_usage(self, agent_id: Optional[str] = None, run_id: Optional[str] = None, limit: int = 100) -> List[AgentLLMUsage]
    
    # Statistics
    def get_agent_run_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]
```

## Data Models and Schema

### 1. AgentRun

Represents a complete agent execution cycle from start to finish.

**Database Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER/SERIAL | Primary key (auto-increment) |
| `agent_id` | VARCHAR(255) | Unique identifier for the agent instance |
| `run_id` | VARCHAR(255) UNIQUE | Unique identifier for this specific run |
| `agent_name` | VARCHAR(255) | Human-readable name of the agent |
| `objective` | TEXT | The goal/objective for this run |
| `additional_context` | TEXT | Additional context provided (JSON string) |
| `started_at` | TIMESTAMP | When the run started |
| `finished_at` | TIMESTAMP | When the run finished (NULL if still running) |
| `total_iterations` | INTEGER | Number of plan-execute-critique iterations |
| `final_synthesis_output` | TEXT | The final synthesis summary |
| `execution_history` | TEXT | Full execution history (JSON string) |
| `status` | VARCHAR(50) | Status: `pending`, `running`, `completed`, `error`, `cancelled` |
| `error_message` | TEXT | Error message if status is `error` |
| `metadata` | TEXT | Additional metadata (JSON string) |
| `created_at` | TIMESTAMP | Record creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

**Python Model:**

```python
@dataclass
class AgentRun:
    agent_id: str
    run_id: str
    agent_name: str = ""
    objective: str = ""
    additional_context: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    total_iterations: int = 0
    final_synthesis_output: Optional[str] = None
    execution_history: Optional[str] = None
    status: str = AgentRunStatus.PENDING.value
    error_message: Optional[str] = None
    metadata: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

**Status Values:**
- `pending`: Run created but not started
- `running`: Run is currently executing
- `completed`: Run finished successfully
- `error`: Run failed with an error
- `cancelled`: Run was cancelled

### 2. AgentRunEvent

Represents individual events that occur during an agent run, providing detailed observability.

**Database Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER/SERIAL | Primary key (auto-increment) |
| `agent_id` | VARCHAR(255) | Agent identifier |
| `run_id` | VARCHAR(255) | Foreign key to `agent_runs.run_id` |
| `agent_name` | VARCHAR(255) | Agent name |
| `timestamp` | TIMESTAMP | When the event occurred |
| `event_type` | VARCHAR(100) | Type of event (see EventType enum) |
| `data` | TEXT | Event-specific data (JSON string) |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Python Model:**

```python
@dataclass
class AgentRunEvent:
    agent_id: str
    run_id: str
    agent_name: str = ""
    timestamp: Optional[datetime] = None
    event_type: str = ""
    data: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
```

**Event Types:**

| Event Type | Description |
|------------|-------------|
| `run_started` | Agent run has started |
| `run_completed` | Agent run completed successfully |
| `run_error` | Agent run encountered an error |
| `iteration_started` | New iteration of plan-execute-critique loop |
| `iteration_completed` | Iteration completed |
| `planning_started` | Planning phase started |
| `planning_completed` | Planning phase completed |
| `action_execution_started` | Tool/action execution started |
| `action_execution_completed` | Tool/action execution completed |
| `critique_started` | Critique phase started |
| `critique_completed` | Critique phase completed |
| `critique_result` | Critique result available |
| `synthesis_started` | Synthesis phase started |
| `synthesis_completed` | Synthesis phase completed |
| `synthesis_result` | Synthesis result available |
| `object_stored` | Object stored in identifier map |
| `object_not_found` | Object not found in identifier map |
| `tool_registered` | Tool registered with agent |
| `subagent_run_started` | Subagent run started |
| `subagent_run_completed` | Subagent run completed |
| `custom` | Custom event type |

**Event Data Structure:**

The `data` field contains JSON with event-specific information:

```json
{
  "action": "fetch_articles",
  "args": {"topic": "AI", "max_results": 5},
  "result": {"status": "ok", "result": "Fetched 5 articles"},
  "duration_ms": 1234
}
```

### 3. AgentLLMUsage

Tracks every LLM call made by agents for cost monitoring, debugging, and analytics.

**Database Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER/SERIAL | Primary key (auto-increment) |
| `agent_id` | VARCHAR(255) | Agent identifier (nullable) |
| `run_id` | VARCHAR(255) | Run identifier (nullable) |
| `provider` | VARCHAR(100) | LLM provider (openai, anthropic, groq, ollama) |
| `model` | VARCHAR(255) | Model name used |
| `prompt` | TEXT | The input prompt sent to the LLM |
| `prompt_length` | INTEGER | Character length of the prompt |
| `input_params` | TEXT | Input parameters (JSON: temperature, max_tokens, etc.) |
| `input_tokens` | INTEGER | Number of input tokens (if available) |
| `output_tokens` | INTEGER | Number of output tokens (if available) |
| `response_content` | TEXT | The response from the LLM |
| `response_length` | INTEGER | Character length of the response |
| `usage_metrics` | TEXT | Additional usage metrics (JSON) |
| `status` | VARCHAR(50) | `success` or `error` |
| `error_message` | TEXT | Error message if status is `error` |
| `latency_ms` | INTEGER | Response latency in milliseconds |
| `created_at` | TIMESTAMP | Record creation timestamp |

**Python Model:**

```python
@dataclass
class AgentLLMUsage:
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    provider: str = ""
    model: str = ""
    prompt: str = ""
    prompt_length: int = 0
    input_params: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_content: Optional[str] = None
    response_length: Optional[int] = None
    usage_metrics: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
```

**Usage Metrics Structure:**

The `usage_metrics` field contains provider-specific usage information:

```json
{
  "prompt_tokens": 150,
  "completion_tokens": 50,
  "total_tokens": 200,
  "cost_usd": 0.0003
}
```

## Database Relationships

```
agent_runs (1) ──< (many) agent_run_events
    │
    └──< (many) agent_llm_usage
```

**Relationship Details:**

1. **AgentRun → AgentRunEvent** (One-to-Many)
   - Each run can have multiple events
   - Foreign key: `agent_run_events.run_id` → `agent_runs.run_id`
   - Events are ordered by `timestamp`

2. **AgentRun → AgentLLMUsage** (One-to-Many)
   - Each run can have multiple LLM calls
   - Foreign key: `agent_llm_usage.run_id` → `agent_runs.run_id` (soft reference)
   - LLM usage can also exist without a run (standalone calls)

3. **Agent → AgentRun** (One-to-Many)
   - Each agent can have multiple runs
   - Linked via `agent_id` (no foreign key constraint for flexibility)

## Indexes

For optimal query performance, the following indexes are created:

1. **agent_runs**
   - `idx_runs_agent_id` on `agent_id` - Fast lookup of runs by agent

2. **agent_run_events**
   - `idx_events_run_id` on `run_id` - Fast lookup of events for a run

3. **agent_llm_usage**
   - `idx_usage_agent_id` on `agent_id` - Fast lookup of LLM usage by agent
   - `idx_usage_run_id` on `run_id` - Fast lookup of LLM usage by run

## Implementations

### SQLitePersistence

**Use Case:** Development, testing, single-user deployments

**Features:**
- File-based database (single file)
- No server required
- Automatic schema initialization
- Lightweight and portable

**Example:**

```python
from curio_agent_sdk.persistence import SQLitePersistence

# Use default database file
persistence = SQLitePersistence()

# Or specify custom path
persistence = SQLitePersistence("./data/agent.db")

# Initialize schema (creates tables if they don't exist)
persistence.initialize_schema()

# Use with agent
agent = MyAgent("agent-1", persistence=persistence)
```

**Database File Location:**
- Default: `agent_sdk.db` in current directory
- Custom: Any path you specify

### PostgresPersistence

**Use Case:** Production deployments, multi-user scenarios

**Features:**
- Connection pooling (configurable min/max connections)
- Schema support (can use custom schema)
- Production-ready with ACID guarantees
- Better performance for high concurrency

**Example:**

```python
from curio_agent_sdk.persistence import PostgresPersistence

persistence = PostgresPersistence(
    host="localhost",
    port=5432,
    database="agent_db",
    user="postgres",
    password="secret",
    schema="agent_sdk",  # Optional: use custom schema
    min_connections=1,
    max_connections=10,
)

# Initialize schema
persistence.initialize_schema()

# Use with agent
agent = MyAgent("agent-1", persistence=persistence)
```

**Environment Variables:**
- `DB_HOST`: PostgreSQL host
- `DB_PORT`: PostgreSQL port
- `DB_NAME`: Database name
- `DB_USER`: Username
- `DB_PASSWORD`: Password
- `DB_SCHEMA`: Schema name (default: `agent_sdk`)

**Connection Pooling:**
- Uses `psycopg2.pool.ThreadedConnectionPool`
- Automatically manages connections
- Reuses connections for better performance

### InMemoryPersistence

**Use Case:** Testing, development without persistence

**Features:**
- No database required
- Fast for testing
- Data lost on process exit
- Useful for unit tests

**Example:**

```python
from curio_agent_sdk.persistence import InMemoryPersistence

persistence = InMemoryPersistence()

# Use with agent
agent = MyAgent("agent-1", persistence=persistence)

# Get all data (for testing/debugging)
all_data = persistence.get_all_data()

# Clear all data
persistence.clear_all()
```

**Special Methods:**
- `clear_all()`: Clear all stored data
- `get_all_data()`: Get all data as dictionary (for debugging/export)

## Usage Patterns

### 1. Basic Usage with Agent

```python
from curio_agent_sdk import BaseAgent, AgentConfig
from curio_agent_sdk.persistence import SQLitePersistence

# Create persistence
persistence = SQLitePersistence("./agent.db")
persistence.initialize_schema()

# Create config with persistence
config = AgentConfig(database=DatabaseConfig(db_type="sqlite", sqlite_path="./agent.db"))

# Agent automatically uses persistence
agent = MyAgent("agent-1", config=config)
result = agent.run("Complete task")
```

### 2. Querying Run History

```python
# Get all runs for an agent
runs = persistence.get_agent_runs(agent_id="agent-1", limit=10, offset=0)

for run in runs:
    print(f"Run {run.run_id}: {run.status} ({run.total_iterations} iterations)")
    print(f"Objective: {run.objective}")
    print(f"Summary: {run.final_synthesis_output}")

# Get specific run
run = persistence.get_agent_run("run-123")
if run:
    print(f"Status: {run.status}")
    print(f"Error: {run.error_message}")
```

### 3. Querying Events

```python
# Get all events for a run
events = persistence.get_agent_run_events("run-123")

for event in events:
    print(f"{event.timestamp}: {event.event_type}")
    data = event.get_data_dict()
    print(f"Data: {data}")

# Get specific event type
action_events = persistence.get_agent_run_events(
    "run-123",
    event_type="action_execution_completed"
)
```

### 4. Querying LLM Usage

```python
# Get LLM usage for an agent
usage = persistence.get_llm_usage(agent_id="agent-1", limit=100)

total_tokens = sum(u.get_total_tokens() or 0 for u in usage)
total_cost = sum(u.get_cost_usd() or 0 for u in usage)

print(f"Total tokens: {total_tokens}")
print(f"Total cost: ${total_cost:.4f}")

# Get LLM usage for a specific run
run_usage = persistence.get_llm_usage(run_id="run-123")
```

### 5. Statistics

```python
# Get statistics for an agent
stats = persistence.get_agent_run_stats(agent_id="agent-1")

print(f"Total runs: {stats['total_runs']}")
print(f"Completed: {stats['completed_runs']}")
print(f"Errors: {stats['error_runs']}")
print(f"Avg iterations: {stats['avg_iterations']}")
print(f"Total LLM calls: {stats['total_llm_calls']}")
print(f"Total input tokens: {stats['total_input_tokens']}")
print(f"Total output tokens: {stats['total_output_tokens']}")
```

### 6. Health Checks

```python
# Check if persistence is healthy
if persistence.health_check():
    print("Database is accessible")
else:
    print("Database connection failed")
```

## Migration and Schema Management

### Automatic Schema Initialization

All persistence implementations automatically create tables on first use:

```python
persistence = SQLitePersistence("./agent.db")
persistence.initialize_schema()  # Creates tables if they don't exist
```

### Manual Schema Creation

For PostgreSQL, you can create the schema manually:

```sql
CREATE SCHEMA IF NOT EXISTS agent_sdk;

CREATE TABLE IF NOT EXISTS agent_sdk.agent_runs (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    -- ... other columns
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_runs_agent_id ON agent_sdk.agent_runs(agent_id);
```

## Best Practices

1. **Choose the Right Backend**
   - Development: SQLite
   - Production: PostgreSQL
   - Testing: InMemoryPersistence

2. **Connection Management**
   - PostgreSQL: Use connection pooling (configured automatically)
   - SQLite: Connections are per-operation (no pooling needed)
   - Always call `close()` when done (PostgreSQL)

3. **Error Handling**
   - Always check return values (None for not found)
   - Handle database connection errors gracefully
   - Use health checks before critical operations

4. **Performance**
   - Use indexes (created automatically)
   - Paginate large result sets (use `limit` and `offset`)
   - Batch operations when possible

5. **Data Retention**
   - Implement cleanup policies for old runs
   - Archive old data before deletion
   - Monitor database size

## Troubleshooting

### SQLite: Database Locked

**Problem:** Multiple processes accessing the same SQLite database

**Solution:** 
- Use separate database files per process
- Or use PostgreSQL for multi-process scenarios

### PostgreSQL: Connection Pool Exhausted

**Problem:** Too many concurrent connections

**Solution:**
- Increase `max_connections` in PostgresPersistence
- Or reduce concurrent agent instances

### Missing Data

**Problem:** Data not appearing in queries

**Solution:**
- Check that `initialize_schema()` was called
- Verify foreign key relationships
- Check transaction commits (automatic in implementations)

## API Reference

See the `BasePersistence` class in `persistence/base.py` for the complete API reference.

