"""
Tool agent example using Groq.

Run from repo root with Groq API key set:
  export GROQ_API_KEY=gsk_...
  pip install -e ".[groq]"
  python examples/tool_agent_groq.py

A timestamped .log file is written to the current directory with full run details.
"""

import asyncio
import os

from curio_agent_sdk import Agent, use_run_logger

from tools import calculator, search


async def main() -> None:
    builder = (
        Agent.builder()
        .model("groq:llama-3.3-70b-versatile")
        .system_prompt("You are a helpful assistant. Use the calculator for math and search when asked.")
        .tools([calculator, search])
    )
    logger = use_run_logger(builder, base_name="tool-agent-groq", output_dir=os.path.dirname(os.path.abspath(__file__)))
    agent = builder.build()

    result = await agent.arun("What is (12 + 34) * 2? Reply briefly.")
    print("Output:", result.output)
    path = logger.get_log_path()
    if path:
        print("Run log written to:", path)


if __name__ == "__main__":
    asyncio.run(main())
