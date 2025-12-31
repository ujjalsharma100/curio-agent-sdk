"""
Decorator-Based Tool Registration Example

This example demonstrates different ways to define and register tools
using decorators instead of docstrings.

Three patterns are shown:
1. @tool decorator with explicit metadata
2. @self.tool_registry.tool decorator for direct registration
3. Mixed approach - some decorators, some docstrings

Custom Tier Configuration:
This example demonstrates a balanced tier configuration:
- plan_tier: tier2 (balanced model for planning)
- critique_tier: tier2 (balanced model for critique)
- synthesis_tier: tier1 (fast/cheap for result synthesis)
- action_tier: tier1 (fast/cheap for tool execution)

This shows how to balance quality and cost for moderate complexity tasks.
"""

import json
from typing import Dict, Any, Optional

from curio_agent_sdk import (
    BaseAgent,
    AgentConfig,
    InMemoryPersistence,
    LLMService,
    tool,  # Import the @tool decorator
)


class DecoratorToolsAgent(BaseAgent):
    """
    An agent demonstrating decorator-based tool registration.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[AgentConfig] = None,
        persistence: Optional[Any] = None,
        llm_service: Optional[LLMService] = None,
    ):
        # Configure custom tiers for different phases
        # For this example, we demonstrate a balanced approach
        # - plan_tier: tier2 (balanced - moderate complexity tasks)
        # - critique_tier: tier2 (balanced - standard critique needs)
        # - synthesis_tier: tier1 (fast/cheap - simple result synthesis)
        # - action_tier: tier1 (fast/cheap - straightforward tool execution)
        super().__init__(
            agent_id=agent_id,
            config=config,
            persistence=persistence,
            llm_service=llm_service,
            plan_tier="tier2",      # Balanced model for planning
            critique_tier="tier2",  # Balanced model for critique
            synthesis_tier="tier1", # Fast/cheap for result synthesis
            action_tier="tier1",    # Fast/cheap for tool execution
        )
        self.agent_name = "DecoratorToolsAgent"
        self.description = "Demonstrates decorator-based tool registration"
        self.max_iterations = 5
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        return """
You are a helpful assistant demonstrating different tool registration patterns.

## GUIDELINES
- Use the available tools to help the user
- Tools were registered using decorators for clean, explicit definitions
"""

    def initialize_tools(self) -> None:
        """
        Register tools using different patterns.
        """
        # Pattern 1: Register methods decorated with @tool
        self.tool_registry.register_from_method(self.search)
        self.tool_registry.register_from_method(self.calculate)
        self.tool_registry.register_from_method(self.store_data)

        # Pattern 2: Register directly using decorator on registry
        # (See standalone functions below the class)

        # Pattern 3: Traditional docstring-based (for comparison)
        self.register_tool("get_time", self.get_time)

    # ==================== Pattern 1: @tool decorator on methods ====================

    @tool(
        name="search",
        description="Search for information on a topic",
        parameters={
            "query": "The search query string",
            "max_results": "Maximum number of results to return (default: 5)",
        },
        required_parameters=["query"],
        response_format="List of search results with titles and snippets",
        examples=[
            'search({"query": "python decorators"})',
            'search({"query": "AI news", "max_results": 10})',
        ],
    )
    def search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search implementation."""
        query = args.get("query", "")
        max_results = args.get("max_results", 5)

        # Simulated search results
        results = [
            {"title": f"Result {i+1} for '{query}'", "snippet": f"This is about {query}..."}
            for i in range(min(max_results, 3))
        ]

        return {"status": "ok", "result": {"query": query, "results": results}}

    @tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "expression": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
        },
        required_parameters=["expression"],
        response_format="The calculated result",
    )
    def calculate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculator implementation."""
        expression = args.get("expression", "")

        # Safety check
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return {"status": "error", "result": "Invalid characters in expression"}

        try:
            result = eval(expression)
            return {"status": "ok", "result": f"{expression} = {result}"}
        except Exception as e:
            return {"status": "error", "result": str(e)}

    @tool(
        name="store_data",
        description="Store data with a key for later retrieval",
        parameters={
            "key": "Unique key to store the data under",
            "value": "The data to store (any JSON-serializable value)",
        },
        required_parameters=["key", "value"],
    )
    def store_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store data using the object identifier map."""
        key = args.get("key", "")
        value = args.get("value")

        identifier = self.store_object({"key": key, "value": value}, "StoredData", key=key)
        return {"status": "ok", "result": f"Stored as {identifier}"}

    # ==================== Pattern 3: Traditional docstring (for comparison) ====================

    def get_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: get_time
        description: Get the current time
        parameters:
            timezone: Optional timezone (default: UTC)
        response_format:
            Current time string
        examples:
            >>> get_time({})
            >>> get_time({"timezone": "US/Eastern"})
        """
        from datetime import datetime
        timezone = args.get("timezone", "UTC")
        return {"status": "ok", "result": f"Current time ({timezone}): {datetime.now().isoformat()}"}


# ==================== Pattern 2: Standalone decorated functions ====================
# These can be registered to any agent's tool registry

@tool(
    name="format_json",
    description="Format a JSON string with proper indentation",
    parameters={
        "json_string": "The JSON string to format",
        "indent": "Number of spaces for indentation (default: 2)",
    },
    required_parameters=["json_string"],
)
def format_json(args: Dict[str, Any]) -> Dict[str, Any]:
    """Format JSON string."""
    json_string = args.get("json_string", "{}")
    indent = args.get("indent", 2)

    try:
        parsed = json.loads(json_string)
        formatted = json.dumps(parsed, indent=indent)
        return {"status": "ok", "result": formatted}
    except json.JSONDecodeError as e:
        return {"status": "error", "result": f"Invalid JSON: {e}"}


@tool(
    name="word_count",
    description="Count words, characters, and lines in text",
    parameters={
        "text": "The text to analyze",
    },
    required_parameters=["text"],
)
def word_count(args: Dict[str, Any]) -> Dict[str, Any]:
    """Count words in text."""
    text = args.get("text", "")

    return {
        "status": "ok",
        "result": {
            "words": len(text.split()),
            "characters": len(text),
            "lines": len(text.splitlines()) or 1,
        }
    }


class ExtendedAgent(DecoratorToolsAgent):
    """
    Extended agent that adds standalone decorated functions as tools.
    """

    def initialize_tools(self) -> None:
        # Call parent to register base tools
        super().initialize_tools()

        # Add standalone decorated functions
        self.tool_registry.register_from_method(format_json)
        self.tool_registry.register_from_method(word_count)


def main():
    """Demonstrate decorator-based tool registration."""
    persistence = InMemoryPersistence()

    # Create agent
    agent = ExtendedAgent(
        agent_id="decorator-demo",
        persistence=persistence,
    )

    print("Decorator Tools Agent")
    print("=" * 50)
    print(f"\nRegistered tools: {agent.tool_registry.get_names()}")

    print("\n" + "=" * 50)
    print("Tool Descriptions (for LLM prompt):")
    print("=" * 50)
    print(agent.get_tools_description())

    print("\n" + "=" * 50)
    print("Testing tools:")
    print("=" * 50)

    # Test search
    result = agent.tool_registry.execute("search", {"query": "python decorators", "max_results": 2})
    print(f"\nsearch: {result}")

    # Test calculate
    result = agent.tool_registry.execute("calculate", {"expression": "10 * 5 + 2"})
    print(f"calculate: {result}")

    # Test format_json
    result = agent.tool_registry.execute("format_json", {"json_string": '{"name":"test","value":123}'})
    print(f"format_json: {result}")

    # Test word_count
    result = agent.tool_registry.execute("word_count", {"text": "Hello world, this is a test."})
    print(f"word_count: {result}")


if __name__ == "__main__":
    main()
