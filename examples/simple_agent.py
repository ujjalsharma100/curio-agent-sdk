"""
Simple Agent Example

This example demonstrates how to create a basic agent using the Curio Agent SDK.

With the SDK, you only need to implement:
- get_agent_instructions(): Define your agent's role and guidelines
- initialize_tools(): Register the tools your agent can use

The SDK automatically handles:
- Objective, tools, and execution history in the prompt
- Object identifier system disclaimer
- The plan-critique-synthesize loop

Custom Tier Configuration:
This example demonstrates custom tier configuration for different phases:
- plan_tier: tier2 (balanced model for planning simple tasks)
- critique_tier: tier2 (balanced model for critique)
- synthesis_tier: tier1 (fast/cheap for summarizing simple results)
- action_tier: tier1 (fast/cheap for simple tool execution)

You can customize these tiers based on your agent's needs:
- tier1: Fast/cheap models (good for simple tasks, synthesis)
- tier2: Balanced models (good for moderate complexity)
- tier3: High quality models (good for complex planning, analysis)
"""

import json
from typing import Dict, Any, List, Optional

from curio_agent_sdk import (
    BaseAgent,
    AgentConfig,
    InMemoryPersistence,
    LLMService,
)


class SimpleTaskAgent(BaseAgent):
    """
    A simple agent that can perform basic tasks like calculations,
    text manipulation, and answering questions.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[AgentConfig] = None,
        persistence: Optional[Any] = None,
        llm_service: Optional[LLMService] = None,
    ):
        # Configure custom tiers for different phases
        # For simple tasks, we can use faster/cheaper models for most steps
        # - plan_tier: tier2 (balanced - simple tasks don't need tier3)
        # - critique_tier: tier2 (balanced - sufficient for basic critique)
        # - synthesis_tier: tier1 (fast/cheap - just summarizing simple results)
        # - action_tier: tier1 (fast/cheap - simple tool calls)
        super().__init__(
            agent_id=agent_id,
            config=config,
            persistence=persistence,
            llm_service=llm_service,
            plan_tier="tier2",      # Balanced model for planning simple tasks
            critique_tier="tier2",  # Balanced model for critique
            synthesis_tier="tier1", # Fast/cheap for summarizing simple results
            action_tier="tier1",    # Fast/cheap for simple tool execution
        )
        self.agent_name = "SimpleTaskAgent"
        self.description = "A simple agent for basic tasks"
        self.max_iterations = 5
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        """
        Define the agent's role and guidelines.

        This is the ONLY prompt section you need to define!
        The SDK automatically adds: objective, tools, execution history,
        and the object identifier system disclaimer.
        """
        return """
You are a helpful assistant that can perform simple tasks.
You have access to tools for calculations, text manipulation, and answering questions.

## GUIDELINES
- Use the available tools to accomplish the objective
- Be efficient - don't use more tools than necessary
- If you can answer directly without tools, do so
- Always provide clear, concise responses
"""

    def initialize_tools(self) -> None:
        """Register tools for this agent."""
        self.register_tool("calculate", self.calculate_tool)
        self.register_tool("uppercase", self.uppercase_tool)
        self.register_tool("lowercase", self.lowercase_tool)
        self.register_tool("count_words", self.count_words_tool)
        self.register_tool("store_result", self.store_result_tool)
        self.register_tool("get_result", self.get_result_tool)
        self.register_tool("respond", self.respond_tool)

    # ==================== Tool Implementations ====================

    def calculate_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: calculate
        description: Perform a mathematical calculation
        parameters:
            expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
        required_parameters:
            - expression
        response_format:
            The result of the calculation
        examples:
            >>> calculate({"expression": "2 + 2"})
            >>> calculate({"expression": "(10 + 5) * 2"})
        """
        try:
            expression = args.get("expression", "")

            # Safety check - only allow basic math operations
            allowed_chars = set("0123456789+-*/().% ")
            if not all(c in allowed_chars for c in expression):
                return {
                    "status": "error",
                    "result": "Invalid characters in expression. Only numbers and +-*/().% allowed."
                }

            result = eval(expression)
            return {"status": "ok", "result": f"Result: {result}"}

        except Exception as e:
            return {"status": "error", "result": f"Calculation error: {str(e)}"}

    def uppercase_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: uppercase
        description: Convert text to uppercase
        parameters:
            text: The text to convert
        required_parameters:
            - text
        response_format:
            The uppercase text
        examples:
            >>> uppercase({"text": "hello world"})
        """
        text = args.get("text", "")
        return {"status": "ok", "result": text.upper()}

    def lowercase_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: lowercase
        description: Convert text to lowercase
        parameters:
            text: The text to convert
        required_parameters:
            - text
        response_format:
            The lowercase text
        examples:
            >>> lowercase({"text": "HELLO WORLD"})
        """
        text = args.get("text", "")
        return {"status": "ok", "result": text.lower()}

    def count_words_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: count_words
        description: Count the number of words in a text
        parameters:
            text: The text to count words in
        required_parameters:
            - text
        response_format:
            The word count
        examples:
            >>> count_words({"text": "Hello world, this is a test"})
        """
        text = args.get("text", "")
        words = text.split()
        return {"status": "ok", "result": f"Word count: {len(words)}"}

    def store_result_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: store_result
        description: Store a result for later retrieval
        parameters:
            value: The value to store
            label: Optional label for the stored value
        required_parameters:
            - value
        response_format:
            The identifier of the stored result
        examples:
            >>> store_result({"value": "important data", "label": "my_result"})
        """
        value = args.get("value")
        label = args.get("label", "Result")

        identifier = self.store_object(
            {"value": value, "label": label},
            "StoredResult"
        )

        return {"status": "ok", "result": f"Stored as {identifier}"}

    def get_result_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: get_result
        description: Retrieve a previously stored result
        parameters:
            identifier: The identifier of the stored result (e.g., "StoredResult1")
        required_parameters:
            - identifier
        response_format:
            The stored value
        examples:
            >>> get_result({"identifier": "StoredResult1"})
        """
        identifier = args.get("identifier", "")
        obj = self.get_object(identifier)

        if obj is None:
            return {"status": "error", "result": f"No result found with identifier: {identifier}"}

        return {"status": "ok", "result": obj}

    def respond_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: respond
        description: Provide a response to the user (use this for final answers)
        parameters:
            message: The response message
        required_parameters:
            - message
        response_format:
            Confirmation that the response was recorded
        examples:
            >>> respond({"message": "The answer to your question is 42"})
        """
        message = args.get("message", "")
        return {"status": "ok", "result": f"Response: {message}"}


def main():
    """Run a simple example."""
    # Create in-memory persistence for testing
    persistence = InMemoryPersistence()

    # Create config (uses environment variables if available)
    try:
        config = AgentConfig.from_env()
        llm_service = config.get_llm_service()
    except Exception as e:
        print(f"Warning: Could not load config from env: {e}")
        print("Running without LLM (will fail on actual run)")
        config = None
        llm_service = None

    # Create agent
    agent = SimpleTaskAgent(
        agent_id="simple-agent-1",
        config=config,
        persistence=persistence,
        llm_service=llm_service,
    )

    # Show agent status
    print("Agent Status:")
    status = agent.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # If we have LLM service, run the agent
    if llm_service:
        print("\nRunning agent...")
        result = agent.run(
            objective="Calculate 15 + 27, then store the result and tell me the answer",
            additional_context={"user": "test_user"},
        )

        print(f"\nResult:")
        print(f"  Status: {result.status}")
        print(f"  Iterations: {result.total_iterations}")
        print(f"  Summary: {result.synthesis_summary}")

        # Show run stats
        stats = persistence.get_agent_run_stats("simple-agent-1")
        print(f"\nRun Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\nSkipping agent run (no LLM service configured)")
        print("Set GROQ_API_KEY or other provider keys in environment")


if __name__ == "__main__":
    main()
