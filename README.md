# Curio Agent SDK

A flexible, model-agnostic agentic framework for building autonomous AI agents.

**Repository**: [https://github.com/ujjalsharma100/curio-agent-sdk](https://github.com/ujjalsharma100/curio-agent-sdk)

## Overview

Curio Agent SDK provides a comprehensive toolkit for building autonomous agents with:

- **Model-agnostic LLM calling** - Unified interface for OpenAI, Anthropic, Groq, and Ollama
- **Tiered model routing** - Automatic selection based on task complexity with failover
- **Object identifier maps** - Context window optimization through object abstraction
- **Flexible tool registry** - Easy tool registration and management
- **Plan-critique-synthesize loop** - Production-ready agentic execution pattern
- **Database persistence** - PostgreSQL, SQLite, or in-memory for observability
- **Highly configurable** - Environment-based or programmatic configuration

## Installation

```bash
# From the project directory
pip install -e ./curio_agent_sdk

# Or install dependencies
pip install openai anthropic groq ollama psycopg2-binary python-dotenv
```

## Quick Start

### 1. Basic Agent

With the SDK, you only need to implement two methods:
- `get_agent_instructions()`: Define your agent's role and guidelines
- `initialize_tools()`: Register the tools your agent can use

The SDK **automatically handles** objective, tools, execution history, and the object identifier system.

```python
from curio_agent_sdk import BaseAgent, AgentConfig

class MyAgent(BaseAgent):
    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config=config)
        self.agent_name = "MyAgent"
        self.max_iterations = 5
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        # Only define your agent's role and guidelines!
        # Objective, tools, and history are automatically included.
        return """
You are a helpful assistant that greets users.

## GUIDELINES
- Be friendly and welcoming
- Use the greet tool to greet users
"""

    def initialize_tools(self):
        self.register_tool("greet", self.greet_tool)

    def greet_tool(self, args):
        """
        name: greet
        description: Greet a person by name
        parameters:
            name: The name of the person to greet
        """
        name = args.get("name", "World")
        return {"status": "ok", "result": f"Hello, {name}!"}

# Usage
config = AgentConfig.from_env()
agent = MyAgent("my-agent", config)
result = agent.run("Greet the user named Alice")
print(result.synthesis_summary)
```

### 2. Using LLM Service Directly

```python
from curio_agent_sdk import call_llm, initialize_llm_service

# Initialize with default config
initialize_llm_service()

# Simple call
response = call_llm("What is the capital of France?")
print(response)

# Tier-based call (automatic model selection)
response = call_llm(
    "Write a comprehensive essay about AI safety",
    tier="tier3"  # Use high-quality model
)

# Explicit provider/model
response = call_llm(
    "Summarize this text",
    provider="openai",
    model="gpt-4"
)
```

### 3. Object Identifier Map for Context Optimization

```python
from curio_agent_sdk import ObjectIdentifierMap

# Create object map
object_map = ObjectIdentifierMap()

# Store objects with short identifiers
articles = [
    {"title": "AI News", "content": "Long content here..."},
    {"title": "ML Update", "content": "Another long article..."},
]

for article in articles:
    identifier = object_map.store(article, "Article")
    print(f"Stored as: {identifier}")  # "Article1", "Article2"

# Use in prompts (saves tokens!)
prompt = f"Available articles: {', '.join(object_map.get_identifiers_by_type('Article'))}"

# Retrieve when needed
article = object_map.get("Article1")
```

## Configuration

### Quick Start (.env)

**Minimal** - just one API key (auto-configures models):

```bash
OPENAI_API_KEY=sk-your-key-here
```

**Explicit** - define your tier models:

```bash
OPENAI_API_KEY=sk-your-key-here

TIER1_MODELS=openai:gpt-4o-mini
TIER2_MODELS=openai:gpt-4o
TIER3_MODELS=openai:gpt-4o
```

The model list order IS the priority. First model is tried first; if it fails, next is tried.

### Multi-Provider Failover

Mix providers in any order for reliability:

```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# If Groq fails, try OpenAI
TIER1_MODELS=groq:llama-3.1-8b-instant,openai:gpt-4o-mini
TIER2_MODELS=groq:llama-3.3-70b-versatile,openai:gpt-4o
TIER3_MODELS=openai:gpt-4o,groq:llama-3.3-70b-versatile
```

### Local with Ollama

```bash
OLLAMA_HOST=http://localhost:11434

TIER1_MODELS=ollama:llama3.1:8b
TIER2_MODELS=ollama:llama3.1:70b
TIER3_MODELS=ollama:llama3.1:70b
```

### Auto-Detection Defaults

If no `TIER*_MODELS` set, SDK auto-configures based on available API keys:

| Provider | tier1 (fast) | tier2 (balanced) | tier3 (best) |
|----------|--------------|------------------|--------------|
| OpenAI | gpt-4o-mini | gpt-4o | gpt-4o |
| Anthropic | claude-3-haiku | claude-3.5-sonnet | claude-3.5-sonnet |
| Groq | llama-3.1-8b-instant | llama-3.3-70b | llama-3.3-70b |

### Database Configuration

```bash
# SQLite (default)
DB_TYPE=sqlite
DB_PATH=./agent.db

# PostgreSQL (production)
DB_TYPE=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agent_db
DB_USER=postgres
DB_PASSWORD=secret

# In-memory (testing)
DB_TYPE=memory
```

See [`examples/env_examples/`](examples/env_examples/) for complete configuration examples.

### Programmatic Configuration

```python
from curio_agent_sdk import AgentConfig, DatabaseConfig

config = AgentConfig(
    database=DatabaseConfig(
        db_type="sqlite",
        sqlite_path="./my_agent.db"
    ),
    default_tier="tier2",
    log_level="DEBUG",
)

# Get services
persistence = config.get_persistence()
llm_service = config.get_llm_service()
```

## Architecture

### Core Components

```
curio_agent_sdk/
├── core/
│   ├── base_agent.py      # BaseAgent abstract class
│   ├── object_identifier_map.py  # Context optimization
│   ├── tool_registry.py   # Tool management
│   ├── models.py          # Data models
│   └── README.md          # Comprehensive core module documentation
├── llm/
│   ├── providers/         # Provider implementations
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── groq.py
│   │   └── ollama.py
│   ├── service.py         # LLMService main class
│   ├── routing.py         # Tier-based routing
│   ├── models.py          # LLM data models
│   └── README.md          # Comprehensive LLM module documentation
├── persistence/
│   ├── postgres.py        # PostgreSQL backend
│   ├── sqlite.py          # SQLite backend
│   ├── memory.py          # In-memory backend
│   ├── base.py            # Base persistence interface
│   └── README.md          # Comprehensive persistence module documentation
└── config/
    └── settings.py        # Configuration management
```

### Module Documentation

For detailed documentation on each module:

- **[Core Module](./core/README.md)** - Agentic architecture, BaseAgent, plan-critique-synthesize loop, ObjectIdentifierMap, ToolRegistry
- **[LLM Module](./llm/README.md)** - LLM service architecture, routing, multi-provider support, standalone usage
- **[Persistence Module](./persistence/README.md)** - Database schemas, data models, relationships, implementations

### Agentic Loop

The SDK implements a plan-critique-synthesize loop:

```
1. Plan → LLM generates list of actions
2. Execute → Each action is executed
3. Critique → LLM evaluates progress (continue/done)
4. Repeat until done or max iterations
5. Synthesize → LLM summarizes results
```

### Tier System

Three tiers for different task complexities:

| Tier | Purpose | Default Models |
|------|---------|----------------|
| tier1 | Fast, simple tasks | llama-3.1-8b-instant |
| tier2 | Balanced quality/speed | llama-3.3-70b-versatile |
| tier3 | High-quality output | llama-4-maverick |

## Key Features

### 1. Automatic Prompt Assembly

The SDK automatically builds prompts with these sections (in order):

1. **Agent Instructions** - Your custom role/guidelines/preferences (from `get_agent_instructions()`)
2. **Objective** - The goal passed to `run()`
3. **Additional Context** - Any context passed to `run()` (auto-formatted as JSON)
4. **Available Tools** - Auto-generated from registered tools
5. **Execution History** - What has been done so far

You only define #1 - everything else is handled automatically. Include any custom sections you need (user preferences, stored objects, conversation history, etc.) directly in your `get_agent_instructions()` method.

### 2. Object Identifier Map (Context Optimization)

Reduces context window usage by storing objects locally and referencing with short identifiers:

```python
# Without: 500+ tokens per article in prompt
prompt = f"Articles: {json.dumps(articles)}"

# With: 2-3 tokens per reference
for article in articles:
    id = object_map.store(article, "Article", key=article['url'])
prompt = f"Articles: Article1, Article2, Article3"
```

### 3. Automatic Rate Limit Handling

```python
# Automatically retries with different models on 429 errors
response = call_llm(prompt, tier="tier3")
# If llama-4-maverick rate-limited → tries gpt-oss-120b → tries next...
```

### 4. Multi-Key Rotation

```python
# Configure multiple API keys
GROQ_API_KEY_1=key1
GROQ_API_KEY_2=key2
GROQ_API_KEY_3=key3

# SDK automatically rotates through healthy keys
```

### 5. Database Persistence

Track all agent runs and LLM usage:

```python
from curio_agent_sdk import SQLitePersistence

persistence = SQLitePersistence("./agent.db")

# After agent runs...
stats = persistence.get_agent_run_stats("my-agent")
print(f"Total runs: {stats['total_runs']}")
print(f"Total LLM calls: {stats['total_llm_calls']}")
print(f"Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}")
```

## Creating Custom Agents

### Step 1: Inherit from BaseAgent

```python
from curio_agent_sdk import BaseAgent

class ContentCuratorAgent(BaseAgent):
    def __init__(self, agent_id, config):
        super().__init__(
            agent_id,
            config=config,
            # Customize model tiers for each step (optional)
            plan_tier="tier3",      # Best model for planning
            critique_tier="tier3",  # Best model for critique
            synthesis_tier="tier1", # Fast/cheap for synthesis
            action_tier="tier2",    # Balanced for tool-related LLM calls
        )
        self.agent_name = "ContentCurator"
        self.max_iterations = 10
        self.initialize_tools()
```

**Default tier configuration:**
| Step | Default Tier | Purpose |
|------|--------------|---------|
| Planning | tier3 | Best quality for generating action plans |
| Critique | tier3 | Best quality for evaluating progress |
| Synthesis | tier1 | Fast/cheap for summarizing results |
| Action | tier2 | Balanced for tool-related LLM calls |

### Step 2: Implement Agent Instructions

Define your agent's role, guidelines, and any custom sections. The SDK automatically adds objective, additional context, tools, and execution history.

```python
def get_agent_instructions(self) -> str:
    # Include any dynamic content you need - pull from DB, services, etc.
    stored_articles = self.format_objects_for_prompt("Article")

    return f"""
You are a content curator that finds and analyzes articles.

## YOUR ROLE
- Fetch content based on user interests
- Analyze content for relevance and quality
- Provide concise summaries and recommendations

## GUIDELINES
- Always analyze articles before recommending them
- Use identifiers to reference stored content
- Be concise in summaries

## STORED ARTICLES
{stored_articles}
"""
```

### Step 3: Register Tools

You can register tools using **docstrings** or **decorators**:

#### Option A: Docstring-based (simple)

```python
def initialize_tools(self):
    self.register_tool("fetch_articles", self.fetch_articles)

def fetch_articles(self, args):
    """
    name: fetch_articles
    description: Fetch articles from RSS feed
    parameters:
        topic: The topic to search for
        max_results: Maximum articles to fetch
    required_parameters:
        - topic
    """
    topic = args.get("topic")
    max_results = args.get("max_results", 5)
    articles = self._fetch_from_rss(topic, max_results)

    identifiers = []
    for article in articles:
        id = self.store_object(article, "Article", key=article['url'])
        identifiers.append(id)

    return {"status": "ok", "result": f"Fetched {len(identifiers)} articles: {identifiers}"}
```

#### Option B: Decorator-based (explicit)

```python
from curio_agent_sdk import tool

class MyAgent(BaseAgent):
    def initialize_tools(self):
        # Register decorated methods
        self.tool_registry.register_from_method(self.fetch_articles)
        self.tool_registry.register_from_method(self.analyze_article)

    @tool(
        name="fetch_articles",
        description="Fetch articles from RSS feed",
        parameters={
            "topic": "The topic to search for",
            "max_results": "Maximum articles to fetch (default: 5)",
        },
        required_parameters=["topic"],
        response_format="List of article identifiers",
    )
    def fetch_articles(self, args):
        topic = args.get("topic")
        max_results = args.get("max_results", 5)
        # ... implementation
        return {"status": "ok", "result": f"Fetched {len(identifiers)} articles"}

    @tool(
        name="analyze_article",
        description="Analyze an article for relevance",
        parameters={
            "article_id": "The article identifier (e.g., 'Article1')",
        },
        required_parameters=["article_id"],
    )
    def analyze_article(self, args):
        article_id = args.get("article_id")
        article = self.get_object(article_id)

        if not article:
            return {"status": "error", "result": f"Article {article_id} not found"}

        # ... analysis implementation
        return {"status": "ok", "result": {"analysis_id": analysis_id}}
```

#### Option C: Standalone decorated functions

```python
from curio_agent_sdk import tool

# Define tools as standalone functions
@tool(
    name="format_json",
    description="Format JSON with indentation",
    parameters={"json_string": "The JSON to format"},
    required_parameters=["json_string"],
)
def format_json(args):
    import json
    data = json.loads(args["json_string"])
    return {"status": "ok", "result": json.dumps(data, indent=2)}

# Register in your agent
class MyAgent(BaseAgent):
    def initialize_tools(self):
        self.tool_registry.register_from_method(format_json)
```

### Step 4: Run the Agent

```python
config = AgentConfig.from_env()
agent = ContentCuratorAgent("curator-1", config)

result = agent.run(
    objective="Find and analyze recent AI news articles",
    additional_context={"preferences": "Focus on LLM developments"}
)

print(f"Status: {result.status}")
print(f"Summary: {result.synthesis_summary}")
print(f"Iterations: {result.total_iterations}")
```

## Subagent Pattern

Create hierarchical agent systems:

```python
class OrchestratorAgent(BaseAgent):
    def __init__(self, agent_id, config):
        super().__init__(agent_id, config=config)
        self.subagents = {
            "news": NewsAgent(f"{agent_id}-news", config),
            "research": ResearchAgent(f"{agent_id}-research", config),
        }
        self.initialize_tools()

    def initialize_tools(self):
        self.register_tool("delegate_to_news", self.delegate_news)
        self.register_tool("delegate_to_research", self.delegate_research)

    def delegate_news(self, args):
        """Delegate task to news agent"""
        result = self.subagents["news"].run(
            args.get("objective"),
            args.get("context", {})
        )
        return {"status": "ok", "result": result.synthesis_summary}
```

## Observability

The SDK provides comprehensive observability through database persistence. All agent runs, LLM calls, and events are automatically logged to your configured database (SQLite or PostgreSQL). A separate observability dashboard tool is available for visualizing this data.

### Enabling Persistence

All agent runs, events, and LLM calls are automatically tracked when you configure a database:

```bash
# In your .env file
DB_TYPE=sqlite
DB_PATH=./agent_sdk.db
```

Or use PostgreSQL for production:

```bash
DB_TYPE=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agent_sdk
DB_USER=postgres
DB_PASSWORD=your_password
```

### What Gets Tracked

| Data | Description |
|------|-------------|
| **Agent Runs** | Every `agent.run()` call with objective, status, iterations, timing |
| **Events** | Planning, critique, synthesis, action executions, errors |
| **LLM Calls** | Every LLM API call with full prompt, response, tokens, latency |
| **Execution History** | Complete step-by-step history of each run |

### Programmatic Access

```python
# Agent-level stats
stats = agent.get_run_stats()
print(f"Total runs: {stats['total_runs']}")
print(f"Success rate: {stats['completed_runs'] / stats['total_runs'] * 100}%")

# Run history
runs = agent.get_run_history(limit=10)
for run in runs:
    print(f"{run.run_id}: {run.status} ({run.total_iterations} iterations)")

# Run events
events = agent.get_run_events(run_id, event_type="action_execution_completed")
for event in events:
    print(f"{event.timestamp}: {event.get_data_dict()}")
```

### LLM Usage Tracking

```python
from curio_agent_sdk import SQLitePersistence

persistence = SQLitePersistence("./agent_sdk.db")
usage = persistence.get_llm_usage(agent_id="my-agent", limit=100)

total_tokens = sum(u.get_total_tokens() or 0 for u in usage)
print(f"Total tokens used: {total_tokens}")
```

---

## Observability Dashboard

The SDK provides comprehensive observability through database persistence. A separate **Curio Agent Observability** tool provides a full-featured web dashboard for exploring agent activity, debugging runs, and analyzing LLM usage.

> **Note**: The observability dashboard is a separate tool. See the [Curio Agent Observability repository](https://github.com/ujjalsharma100/curio-agent-observability) for installation and setup instructions. It reads from the same database that your agents use.

### Dashboard Features

- **Overview Dashboard** - Stats on runs, LLM calls, token usage at a glance
- **Runs Explorer** - Browse all runs with filtering by agent, status
- **Run Detail View** - Timeline of events and LLM calls, execution history
- **LLM Calls Browser** - Search prompts/responses, filter by provider/model
- **LLM Call Detail** - Full prompt and response text with copy buttons
- **Agent Analytics** - Per-agent statistics and run history
- **Model Analytics** - Usage breakdown by provider and model

### Quick Setup

**1. Navigate to the observability tool:**

Clone the observability repository:

```bash
git clone https://github.com/ujjalsharma100/curio-agent-observability.git
cd curio-agent-observability
```

**2. Install dependencies:**

```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

**3. Configure environment:**

The observability tool uses the same database configuration as your SDK. Set the same environment variables:

```bash
# Same database your agents use
DB_TYPE=sqlite
DB_PATH=/path/to/your/agent_sdk.db

# Or for PostgreSQL
# DB_TYPE=postgres
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=agent_sdk
# DB_USER=postgres
# DB_PASSWORD=your_password
```

**4. Start the dashboard:**

```bash
# Easy mode - runs both backend and frontend
./run.sh
```

Or manually:

```bash
# Terminal 1 - Backend API (Flask)
cd backend
python app.py  # Runs on port 5050

# Terminal 2 - Frontend UI (React)
cd frontend
npm start  # Runs on port 3000
```

**5. Open the dashboard:**

Navigate to `http://localhost:3000`

For detailed setup instructions, see the [Curio Agent Observability README](https://github.com/ujjalsharma100/curio-agent-observability/blob/main/README.md).

### Example Workflow

```python
# 1. Your agent code (uses persistence automatically)
from curio_agent_sdk import BaseAgent, AgentConfig

config = AgentConfig.from_env()  # Uses DB_TYPE, DB_PATH from .env

class MyAgent(BaseAgent):
    def get_agent_instructions(self):
        return "You are a helpful assistant."

    def initialize_tools(self):
        self.register_tool("greet", self.greet)

    def greet(self, args):
        """
        name: greet
        description: Greet someone
        parameters:
            name: Name to greet
        """
        return {"status": "ok", "result": f"Hello, {args.get('name')}!"}

# 2. Run your agent
agent = MyAgent("my-agent", config)
result = agent.run("Greet Alice")

# 3. Open the dashboard to see:
#    - The run with objective "Greet Alice"
#    - Timeline of events (planning, action execution, critique, synthesis)
#    - Each LLM call with full prompts and responses
#    - Token counts and latencies
```

### Dashboard API Endpoints

The backend exposes a REST API you can also use programmatically:

```bash
# Health check
curl http://localhost:5050/api/health

# Get overall stats
curl http://localhost:5050/api/stats

# List all runs
curl http://localhost:5050/api/runs

# Get specific run with events and LLM calls
curl http://localhost:5050/api/runs/{run_id}

# Get run timeline
curl http://localhost:5050/api/runs/{run_id}/timeline

# List LLM calls
curl http://localhost:5050/api/llm-calls

# Get specific LLM call with full prompt/response
curl http://localhost:5050/api/llm-calls/{call_id}

# Search across runs and LLM calls
curl http://localhost:5050/api/search?q=your_query

# Get model usage statistics
curl http://localhost:5050/api/llm-calls/models
```

### Troubleshooting

**No data showing?**
- Ensure `DB_PATH` in your dashboard `.env` points to the same database your agents use
- Run some agents first to populate data
- Check that the backend is running (`curl http://localhost:5050/api/health`)

**Backend won't start?**
- Verify port 5050 isn't in use
- Check database credentials are correct
- Ensure Flask dependencies are installed: `pip install -r requirements.txt`

**Frontend can't connect to backend?**
- The frontend proxies to port 5050 by default (configured in `package.json`)
- Make sure backend is running before starting frontend

## API Reference

### BaseAgent

**Required Methods (must implement):**

| Method | Description |
|--------|-------------|
| `get_agent_instructions()` | Define agent role, guidelines, and any custom sections |
| `initialize_tools()` | Register tools the agent can use |

**Optional Override Methods:**

| Method | Description |
|--------|-------------|
| `plan(objective, context, history)` | Custom planning logic |
| `critique(history, objective, context)` | Custom critique logic |
| `synthesis(history, objective, context)` | Custom synthesis logic |

**Core Methods:**

| Method | Description |
|--------|-------------|
| `run(objective, context, max_iterations)` | Run the agent |
| `register_tool(name, function)` | Register a tool |
| `store_object(obj, type, key)` | Store object in identifier map |
| `get_object(identifier)` | Get object from identifier map |
| `get_tools_description()` | Get formatted tool descriptions |
| `get_status()` | Get current agent status |
| `get_run_history(limit)` | Get run history |

### LLMService

| Method | Description |
|--------|-------------|
| `call_llm(prompt, tier, provider, model)` | Call LLM |
| `get_available_providers()` | List available providers |
| `get_routing_stats()` | Get routing statistics |

### ObjectIdentifierMap

| Method | Description |
|--------|-------------|
| `store(obj, type, key)` | Store object, return identifier |
| `get(identifier)` | Get object by identifier |
| `get_by_key(key)` | Get object by dedup key |
| `remove(identifier)` | Remove object |
| `clear(type)` | Clear objects |
| `format_for_prompt(type, formatter)` | Format for prompt |

### ToolRegistry

| Method | Description |
|--------|-------------|
| `register(name, function, description)` | Register tool |
| `execute(name, args)` | Execute tool |
| `get_descriptions()` | Get all descriptions |
| `enable(name)` / `disable(name)` | Toggle tool |

## Best Practices

1. **Use Object Maps** - Store large objects and reference with identifiers
2. **Choose Appropriate Tiers** - Use tier1 for simple tasks, tier3 for complex ones
3. **Configure Multiple Keys** - Avoid rate limits with key rotation
4. **Enable Persistence** - Track runs for debugging and analytics
5. **Handle Errors in Tools** - Return `{"status": "error", "result": message}`
6. **Keep Tools Focused** - One action per tool for better LLM understanding

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
