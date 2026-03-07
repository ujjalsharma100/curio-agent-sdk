"""
Tool agent example using Ollama (local).

Requires Ollama running with a model that supports tools (e.g. llama3.1).
Sets OLLAMA_HOST so the router includes Ollama without requiring env.

  ollama run llama3.1
  pip install -e ".[ollama]"
  python examples/tool_agent_ollama.py

A timestamped .log file is written to the current directory with full run details.
"""

import asyncio
import os

# Ensure Ollama is in the router when using .model("ollama:...")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

from curio_agent_sdk import Agent, use_run_logger

from tools import calculator, search


async def main() -> None:
    builder = (
        Agent.builder()
        .model("ollama:qwen3.5:9b")
        .system_prompt("You are a helpful assistant. Use the calculator for math and search when asked.")
        .tools([calculator, search])
    )
    logger = use_run_logger(builder, base_name="tool-agent-ollama", output_dir=os.path.dirname(os.path.abspath(__file__)))
    agent = builder.build()

    result = await agent.arun("What is (12 + 34) * 2? Reply briefly.")
    print("Output:", result.output)
    path = logger.get_log_path()
    if path:
        print("Run log written to:", path)


if __name__ == "__main__":
    asyncio.run(main())
