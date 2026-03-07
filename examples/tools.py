"""Shared tool definitions for examples: calculator and search."""

from __future__ import annotations

import re

from curio_agent_sdk import tool


def _safe_eval(expression: str) -> float:
    """Evaluate a simple math expression; only allow digits and + - * / ( ) . and spaces."""
    allowed = re.sub(r"[^0-9+\-*/().\s]", "", expression)
    if allowed != expression.strip():
        raise ValueError("Expression may only contain numbers and + - * / ( )")
    try:
        return float(eval(allowed))
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}") from e


@tool(description="Evaluate a math expression. Only numbers and + - * / ( ) allowed.")
def calculator(expression: str) -> str:
    """
    Evaluate a simple math expression.

    Args:
        expression: A math expression, e.g. "2 + 3 * 4"
    """
    result = _safe_eval(expression)
    return str(result)


@tool(description="Search the web (placeholder). Returns a mock result.")
def search(query: str) -> str:
    """
    Search the web for the given query.

    Args:
        query: Search query string.
    """
    return f"[mock search result for: {query}]"
