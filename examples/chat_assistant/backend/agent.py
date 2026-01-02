"""
Smart Calculator Chat Agent

A demonstration agent that can perform math operations and engage in
natural conversation while maintaining conversation history.
"""

import math
import json
import os
from typing import Any, Dict, List

from curio_agent_sdk import BaseAgent, AgentConfig


class ChatAssistantAgent(BaseAgent):
    """
    A conversational chat agent with math capabilities.

    This agent demonstrates:
    - Natural conversation handling
    - Math operations via tools
    - Conversation history management
    - Tiered model usage with Groq
    """

    def __init__(self, agent_id: str, config: AgentConfig, llm_service=None, persistence=None, conversation_history: List[Dict] = None):
        super().__init__(
            agent_id=agent_id,
            config=config,
            llm_service=llm_service,
            persistence=persistence,
            plan_tier="tier2",      # Use tier2 for planning (balanced)
            critique_tier="tier1",  # Use tier1 for critique (fast)
            synthesis_tier="tier1", # Use tier1 for synthesis (fast)
            action_tier="tier1",    # Use tier1 for actions (fast)
        )
        self.agent_name = "ChatAssistant"
        self.description = "A friendly chat assistant with math capabilities"
        self.max_iterations = 3
        self.conversation_history = conversation_history or []
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        """
        Define the agent's role, persona, and guidelines.
        Includes conversation history for context.
        """
        history_text = ""
        if self.conversation_history:
            history_text = "\n\n## CONVERSATION HISTORY\n"
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages for context
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"

        return f"""You are a friendly and helpful chat assistant named Curio.

## PERSONALITY
- You are warm, conversational, and helpful
- You explain things clearly and concisely
- You're enthusiastic about helping with math problems
- You remember previous conversation context

## CAPABILITIES
- General conversation and Q&A
- Mathematical calculations (basic arithmetic, advanced math)
- Unit conversions
- Percentage calculations

## GUIDELINES
1. For math questions, ALWAYS use the appropriate math tool to get accurate results
2. Show your work - explain how you arrived at answers
3. If the user's request is unclear, ask for clarification
4. Be conversational and natural in your responses
5. Reference previous conversation when relevant
6. Keep responses concise but informative
{history_text}

## RESPONSE FORMAT
Respond naturally as if having a conversation. When performing calculations,
explain what you're doing and show the result clearly."""

    def initialize_tools(self) -> None:
        """Register all available tools."""
        self.register_tool("calculate", self.calculate_tool)
        self.register_tool("advanced_math", self.advanced_math_tool)
        self.register_tool("unit_convert", self.unit_convert_tool)
        self.register_tool("percentage", self.percentage_tool)

    def calculate_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: calculate
        description: Perform basic arithmetic calculations. Supports +, -, *, /, and parentheses.
        parameters:
            expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5", "(3 + 4) * 2")
        required_parameters:
            - expression
        response_format:
            Dictionary with 'expression' and 'result' keys
        examples:
            >>> calculate({"expression": "2 + 2"})
            {"status": "ok", "result": {"expression": "2 + 2", "result": 4}}
        """
        expression = args.get("expression", "")

        if not expression:
            return {"status": "error", "result": "No expression provided"}

        try:
            # Sanitize: only allow safe characters
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {"status": "error", "result": "Invalid characters in expression. Only numbers and basic operators allowed."}

            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, {})

            return {
                "status": "ok",
                "result": {
                    "expression": expression,
                    "result": result
                }
            }
        except ZeroDivisionError:
            return {"status": "error", "result": "Cannot divide by zero"}
        except Exception as e:
            return {"status": "error", "result": f"Calculation error: {str(e)}"}

    def advanced_math_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: advanced_math
        description: Perform advanced mathematical operations like square root, power, logarithm, trigonometry, etc.
        parameters:
            operation: The math operation (sqrt, pow, log, log10, sin, cos, tan, factorial, abs, round, floor, ceil)
            value: The primary number to operate on
            second_value: Second value for operations that need two numbers (like pow)
        required_parameters:
            - operation
            - value
        response_format:
            Dictionary with operation details and result
        """
        operation = args.get("operation", "").lower()
        value = args.get("value")
        second_value = args.get("second_value")

        if value is None:
            return {"status": "error", "result": "No value provided"}

        try:
            value = float(value)

            operations = {
                "sqrt": lambda v, _: math.sqrt(v),
                "pow": lambda v, s: math.pow(v, float(s) if s else 2),
                "log": lambda v, _: math.log(v),
                "log10": lambda v, _: math.log10(v),
                "sin": lambda v, _: math.sin(math.radians(v)),
                "cos": lambda v, _: math.cos(math.radians(v)),
                "tan": lambda v, _: math.tan(math.radians(v)),
                "factorial": lambda v, _: math.factorial(int(v)),
                "abs": lambda v, _: abs(v),
                "round": lambda v, s: round(v, int(s) if s else 0),
                "floor": lambda v, _: math.floor(v),
                "ceil": lambda v, _: math.ceil(v),
            }

            if operation not in operations:
                return {
                    "status": "error",
                    "result": f"Unknown operation: {operation}. Available: {', '.join(operations.keys())}"
                }

            result = operations[operation](value, second_value)

            return {
                "status": "ok",
                "result": {
                    "operation": operation,
                    "value": value,
                    "second_value": second_value,
                    "result": result
                }
            }
        except ValueError as e:
            return {"status": "error", "result": f"Math error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "result": f"Calculation error: {str(e)}"}

    def unit_convert_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: unit_convert
        description: Convert between common units of measurement.
        parameters:
            value: The numeric value to convert
            from_unit: The unit to convert from (km, mi, m, ft, kg, lb, c, f, l, gal)
            to_unit: The unit to convert to
        required_parameters:
            - value
            - from_unit
            - to_unit
        """
        value = args.get("value")
        from_unit = args.get("from_unit", "").lower()
        to_unit = args.get("to_unit", "").lower()

        if value is None:
            return {"status": "error", "result": "No value provided"}

        try:
            value = float(value)

            # Conversion factors to base units
            conversions = {
                # Length (base: meters)
                ("km", "m"): lambda v: v * 1000,
                ("m", "km"): lambda v: v / 1000,
                ("mi", "km"): lambda v: v * 1.60934,
                ("km", "mi"): lambda v: v / 1.60934,
                ("m", "ft"): lambda v: v * 3.28084,
                ("ft", "m"): lambda v: v / 3.28084,
                ("mi", "m"): lambda v: v * 1609.34,
                ("m", "mi"): lambda v: v / 1609.34,

                # Weight (base: kg)
                ("kg", "lb"): lambda v: v * 2.20462,
                ("lb", "kg"): lambda v: v / 2.20462,
                ("kg", "g"): lambda v: v * 1000,
                ("g", "kg"): lambda v: v / 1000,

                # Temperature
                ("c", "f"): lambda v: (v * 9/5) + 32,
                ("f", "c"): lambda v: (v - 32) * 5/9,

                # Volume
                ("l", "gal"): lambda v: v * 0.264172,
                ("gal", "l"): lambda v: v / 0.264172,
                ("l", "ml"): lambda v: v * 1000,
                ("ml", "l"): lambda v: v / 1000,
            }

            key = (from_unit, to_unit)
            if key not in conversions:
                available = set()
                for k in conversions.keys():
                    available.add(k[0])
                    available.add(k[1])
                return {
                    "status": "error",
                    "result": f"Cannot convert from {from_unit} to {to_unit}. Available units: {', '.join(sorted(available))}"
                }

            result = conversions[key](value)

            return {
                "status": "ok",
                "result": {
                    "original": f"{value} {from_unit}",
                    "converted": f"{result:.4f} {to_unit}",
                    "value": result
                }
            }
        except Exception as e:
            return {"status": "error", "result": f"Conversion error: {str(e)}"}

    def percentage_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: percentage
        description: Calculate percentage operations - find percentage of a number, find what percent one number is of another, or calculate percentage change.
        parameters:
            operation: The percentage operation (of, is_what_percent, change)
            value1: First value
            value2: Second value
        required_parameters:
            - operation
            - value1
            - value2
        examples:
            - operation: "of" - What is 15% of 200? (value1=15, value2=200)
            - operation: "is_what_percent" - 30 is what percent of 150? (value1=30, value2=150)
            - operation: "change" - Percentage change from 100 to 150? (value1=100, value2=150)
        """
        operation = args.get("operation", "").lower()
        value1 = args.get("value1")
        value2 = args.get("value2")

        if value1 is None or value2 is None:
            return {"status": "error", "result": "Both value1 and value2 are required"}

        try:
            value1 = float(value1)
            value2 = float(value2)

            if operation == "of":
                # What is X% of Y?
                result = (value1 / 100) * value2
                return {
                    "status": "ok",
                    "result": {
                        "question": f"What is {value1}% of {value2}?",
                        "answer": result
                    }
                }
            elif operation == "is_what_percent":
                # X is what percent of Y?
                if value2 == 0:
                    return {"status": "error", "result": "Cannot calculate percentage of zero"}
                result = (value1 / value2) * 100
                return {
                    "status": "ok",
                    "result": {
                        "question": f"{value1} is what percent of {value2}?",
                        "answer": f"{result:.2f}%"
                    }
                }
            elif operation == "change":
                # Percentage change from X to Y
                if value1 == 0:
                    return {"status": "error", "result": "Cannot calculate percentage change from zero"}
                result = ((value2 - value1) / value1) * 100
                return {
                    "status": "ok",
                    "result": {
                        "question": f"Percentage change from {value1} to {value2}?",
                        "answer": f"{result:.2f}%",
                        "direction": "increase" if result > 0 else "decrease" if result < 0 else "no change"
                    }
                }
            else:
                return {
                    "status": "error",
                    "result": f"Unknown operation: {operation}. Use 'of', 'is_what_percent', or 'change'"
                }
        except Exception as e:
            return {"status": "error", "result": f"Calculation error: {str(e)}"}


def create_agent(conversation_history: List[Dict] = None) -> ChatAssistantAgent:
    """Factory function to create a configured agent instance."""
    config = AgentConfig.from_env()
    llm_service = config.get_llm_service()
    persistence = config.get_persistence()
    agent_id = f"chat-assistant-{os.getpid()}"
    return ChatAssistantAgent(
        agent_id=agent_id,
        config=config,
        llm_service=llm_service,
        persistence=persistence,
        conversation_history=conversation_history
    )
