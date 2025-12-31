# Core Module Documentation

## Overview

The core module provides the foundational building blocks for building autonomous AI agents. It implements the agentic architecture, including the plan-critique-synthesize loop, object management for context optimization, tool registration, and event logging.

## Architecture

### Agentic Loop: Plan-Critique-Synthesize

The SDK implements a production-ready agentic execution pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Run Starts                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PLAN                                                      │
│    - LLM generates list of actions to take                 │
│    - Uses tier3 model (best quality) by default             │
│    - Returns: List of PlannedAction objects                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. EXECUTE                                                  │
│    - Each action is executed sequentially                   │
│    - Tools are invoked with provided arguments              │
│    - Results are collected                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. CRITIQUE                                                 │
│    - LLM evaluates progress toward objective               │
│    - Uses tier3 model (best quality) by default             │
│    - Returns: "continue" or "done"                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────┴────────┐
              │                 │
         "continue"         "done"
              │                 │
              ▼                 ▼
    ┌─────────────────┐  ┌─────────────────┐
    │  Repeat Loop     │  │  4. SYNTHESIZE  │
    │  (up to max      │  │  - LLM summarizes│
    │   iterations)    │  │    results       │
    └─────────────────┘  │  - Uses tier1     │
                         │    (fast/cheap)   │
                         └───────────────────┘
```

### Component Overview

```
BaseAgent
├── ObjectIdentifierMap    # Context window optimization
├── ToolRegistry           # Tool management
├── LLMService            # LLM calls with routing
├── Persistence            # Database operations
└── Event Logging          # Observability
```

## BaseAgent

The `BaseAgent` class is the abstract base class that all agents inherit from. It provides:

- **Automatic prompt assembly** - Combines agent instructions, objective, context, tools, and history
- **Plan-critique-synthesize loop** - Production-ready execution pattern
- **Object identifier map** - Context window optimization
- **Tool registry** - Flexible tool registration and execution
- **Event logging** - Detailed observability
- **Database persistence** - Automatic run tracking

### Required Methods

Subclasses must implement:

1. **`get_agent_instructions()`** - Define agent role, guidelines, and custom sections
2. **`initialize_tools()`** - Register tools the agent can use

### Optional Override Methods

You can customize these for advanced use cases:

- `plan(objective, context, history)` - Custom planning logic
- `critique(history, objective, context)` - Custom critique logic
- `synthesis(history, objective, context)` - Custom synthesis logic
- `execute_action(action_name, args)` - Custom action execution

### Tier Configuration

You can customize which model tier is used for each step:

```python
class MyAgent(BaseAgent):
    def __init__(self, agent_id, config):
        super().__init__(
            agent_id,
            config=config,
            plan_tier="tier3",      # Best model for planning
            critique_tier="tier3",  # Best model for critique
            synthesis_tier="tier1", # Fast/cheap for synthesis
            action_tier="tier2",    # Balanced for tool-related LLM calls
        )
```

**Default Tier Configuration:**

| Step | Default Tier | Purpose |
|------|--------------|---------|
| Planning | tier3 | Best quality for generating action plans |
| Critique | tier3 | Best quality for evaluating progress |
| Synthesis | tier1 | Fast/cheap for summarizing results |
| Action | tier2 | Balanced for tool-related LLM calls |

### Prompt Assembly

The SDK automatically builds prompts with these sections (in order):

1. **Agent Instructions** - Your custom role/guidelines/preferences (from `get_agent_instructions()`)
2. **Objective** - The goal passed to `run()`
3. **Additional Context** - Any context passed to `run()` (auto-formatted as JSON)
4. **Available Tools** - Auto-generated from registered tools
5. **Execution History** - What has been done so far

You only define #1 - everything else is handled automatically.

### Example Agent

```python
from curio_agent_sdk import BaseAgent, AgentConfig

class ContentCuratorAgent(BaseAgent):
    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config=config)
        self.agent_name = "ContentCurator"
        self.max_iterations = 10
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        # Pull dynamic data if needed
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

    def initialize_tools(self):
        self.register_tool("fetch_articles", self.fetch_articles)
        self.register_tool("analyze_article", self.analyze_article)

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

    def analyze_article(self, args):
        """
        name: analyze_article
        description: Analyze an article for relevance
        parameters:
            article_id: The article identifier (e.g., 'Article1')
        required_parameters:
            - article_id
        """
        article_id = args.get("article_id")
        article = self.get_object(article_id)
        
        if not article:
            return {"status": "error", "result": f"Article {article_id} not found"}
        
        # Analysis logic here
        return {"status": "ok", "result": {"relevance_score": 0.85}}
```

## ObjectIdentifierMap

The `ObjectIdentifierMap` is a key innovation for reducing LLM context window usage. Instead of passing full objects to the LLM, we store them locally and reference them with short identifiers.

### How It Works

**Without ObjectIdentifierMap:**
```python
# 500+ tokens per article in prompt
prompt = f"Articles: {json.dumps(articles)}"
```

**With ObjectIdentifierMap:**
```python
# 2-3 tokens per reference
for article in articles:
    id = object_map.store(article, "Article", key=article['url'])
prompt = f"Available articles: Article1, Article2, Article3"
```

### Key Features

1. **Automatic Identifier Generation** - Generates short identifiers like "Article1", "Article2"
2. **Duplicate Detection** - Uses custom keys to prevent duplicates
3. **Access Tracking** - Tracks how often objects are accessed
4. **Event Callbacks** - Hooks for logging/observability

### Usage

```python
# Store objects
article = {"title": "AI News", "content": "..."}
identifier = agent.store_object(article, "Article", key=article['url'])
# Returns: "Article1"

# Retrieve objects
article = agent.get_object("Article1")

# Format for prompts
formatted = agent.format_objects_for_prompt("Article")
# Returns: "- Article1: {'title': 'AI News'...}\n- Article2: ..."

# Get all identifiers of a type
ids = agent.object_map.get_identifiers_by_type("Article")
# Returns: ["Article1", "Article2", "Article3"]
```

### Deduplication

Use the `key` parameter to prevent storing duplicates:

```python
# First time
id1 = agent.store_object(article, "Article", key=article['url'])

# Second time (same URL)
id2 = agent.store_object(same_article, "Article", key=article['url'])
# Returns same identifier: id1 == id2
```

## ToolRegistry

The `ToolRegistry` manages all tools (actions) that an agent can execute.

### Tool Registration Methods

#### 1. Docstring-based (Simple)

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
    # Implementation
    return {"status": "ok", "result": f"Fetched {max_results} articles"}
```

#### 2. Decorator-based (Explicit)

```python
from curio_agent_sdk import tool

class MyAgent(BaseAgent):
    def initialize_tools(self):
        self.tool_registry.register_from_method(self.fetch_articles)

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
        # Implementation
        return {"status": "ok", "result": f"Fetched {max_results} articles"}
```

#### 3. Programmatic Registration

```python
def initialize_tools(self):
    self.tool_registry.register(
        name="fetch_articles",
        function=self.fetch_articles,
        description="Fetch articles from RSS feed",
        parameters={
            "topic": "The topic to search for",
            "max_results": "Maximum articles to fetch",
        },
        required_parameters=["topic"],
    )
```

### Tool Execution

Tools are executed automatically by the agent during the execution phase. They receive arguments as a dictionary and should return:

```python
{
    "status": "ok",      # or "error"
    "result": {...}      # The actual result
}
```

### Tool Management

```python
# Enable/disable tools
agent.tool_registry.enable("fetch_articles")
agent.tool_registry.disable("fetch_articles")

# Get tool descriptions (for LLM)
descriptions = agent.tool_registry.get_descriptions()

# Get execution history
history = agent.tool_registry.get_execution_history(limit=10)
```

## Data Models

### AgentRun

Represents a complete agent execution cycle. See [Persistence Documentation](./persistence/README.md) for full schema.

### AgentRunEvent

Represents individual events during a run. See [Persistence Documentation](./persistence/README.md) for full schema.

### PlanResult

Result of the planning phase:

```python
@dataclass
class PlanResult:
    plan: List[PlannedAction]  # List of actions to execute
    notes: str                  # Notes about the plan
    debug_info: str             # Debug information
```

### PlannedAction

A single action to execute:

```python
@dataclass
class PlannedAction:
    action: str                 # Tool name
    args: Dict[str, Any]        # Arguments for the tool
```

### CritiqueResult

Result of the critique phase:

```python
@dataclass
class CritiqueResult:
    status: str                 # "done" or "continue"
    critique_summary: str       # Summary of evaluation
    recommendations: str         # Recommendations for next steps
```

### SynthesisResult

Result of the synthesis phase:

```python
@dataclass
class SynthesisResult:
    synthesis_summary: str       # Final summary
```

### AgentRunResult

Complete result of an agent run:

```python
@dataclass
class AgentRunResult:
    status: str                 # "done", "error", etc.
    synthesis_summary: str      # Final summary
    total_iterations: int       # Number of iterations
    run_id: str                 # Run identifier
    error: Optional[str]        # Error message if failed
    execution_history: List[Dict[str, Any]]  # Full history
```

## Usage Patterns

### 1. Basic Agent

```python
from curio_agent_sdk import BaseAgent, AgentConfig

class SimpleAgent(BaseAgent):
    def __init__(self, agent_id, config):
        super().__init__(agent_id, config)
        self.agent_name = "SimpleAgent"
        self.max_iterations = 5
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        return "You are a helpful assistant."

    def initialize_tools(self):
        self.register_tool("greet", self.greet)

    def greet(self, args):
        name = args.get("name", "World")
        return {"status": "ok", "result": f"Hello, {name}!"}

# Usage
config = AgentConfig.from_env()
agent = SimpleAgent("agent-1", config)
result = agent.run("Greet the user named Alice")
print(result.synthesis_summary)
```

### 2. Agent with Object Storage

```python
class ContentAgent(BaseAgent):
    def initialize_tools(self):
        self.register_tool("fetch_and_store", self.fetch_and_store)

    def fetch_and_store(self, args):
        # Fetch content
        articles = fetch_articles(args.get("topic"))
        
        # Store with identifiers
        identifiers = []
        for article in articles:
            id = self.store_object(article, "Article", key=article['url'])
            identifiers.append(id)
        
        return {"status": "ok", "result": f"Stored {len(identifiers)} articles: {identifiers}"}
    
    def get_agent_instructions(self) -> str:
        # Include stored objects in prompt
        articles = self.format_objects_for_prompt("Article")
        return f"""
You are a content curator.

## STORED ARTICLES
{articles}
"""
```

### 3. Custom Planning Logic

```python
class CustomAgent(BaseAgent):
    def plan(self, objective: str, context: Dict[str, Any], history: List[Dict[str, Any]]) -> PlanResult:
        # Custom planning logic
        # You can use LLM or rule-based planning
        
        # Example: Use LLM with custom prompt
        prompt = f"""
        Objective: {objective}
        Context: {json.dumps(context)}
        
        Create a plan with specific actions.
        """
        
        response = self.llm_service.call_llm(prompt, tier="tier3")
        # Parse response into PlanResult
        # ...
        
        return PlanResult(plan=[...], notes="...")
```

### 4. Subagent Pattern

```python
class OrchestratorAgent(BaseAgent):
    def __init__(self, agent_id, config):
        super().__init__(agent_id, config)
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

### 5. Observability

```python
# Get run statistics
stats = agent.get_run_stats()
print(f"Total runs: {stats['total_runs']}")
print(f"Success rate: {stats['completed_runs'] / stats['total_runs'] * 100}%")

# Get run history
runs = agent.get_run_history(limit=10)
for run in runs:
    print(f"{run.run_id}: {run.status} ({run.total_iterations} iterations)")

# Get run events
events = agent.get_run_events(run_id, event_type="action_execution_completed")
for event in events:
    print(f"{event.timestamp}: {event.get_data_dict()}")
```

## Best Practices

1. **Use Object Maps** - Store large objects and reference with identifiers to save tokens
2. **Choose Appropriate Tiers** - Use tier1 for simple tasks, tier3 for complex ones
3. **Handle Errors in Tools** - Always return `{"status": "error", "result": message}` on errors
4. **Keep Tools Focused** - One action per tool for better LLM understanding
5. **Use Deduplication Keys** - Prevent storing duplicate objects
6. **Enable Persistence** - Track runs for debugging and analytics
7. **Customize Tiers** - Adjust tier configuration based on your needs

## API Reference

### BaseAgent Methods

**Core Methods:**
- `run(objective, context, max_iterations)` - Run the agent
- `register_tool(name, function)` - Register a tool
- `store_object(obj, type, key)` - Store object in identifier map
- `get_object(identifier)` - Get object from identifier map
- `format_objects_for_prompt(type)` - Format objects for prompt inclusion

**Observability:**
- `get_run_history(limit)` - Get run history
- `get_run_stats()` - Get statistics
- `get_run_events(run_id, event_type)` - Get events for a run
- `get_status()` - Get current agent status

**Abstract Methods (must implement):**
- `get_agent_instructions()` - Define agent role and guidelines
- `initialize_tools()` - Register tools

**Optional Override Methods:**
- `plan(objective, context, history)` - Custom planning
- `critique(history, objective, context)` - Custom critique
- `synthesis(history, objective, context)` - Custom synthesis

See the `BaseAgent` class in `core/base_agent.py` for the complete API reference.

