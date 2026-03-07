"""
Tool agent example using OpenAI.

Run from repo root with OpenAI API key set:
  export OPENAI_API_KEY=sk-...
  pip install -e ".[openai]"
  python examples/tool_agent_openai.py

A timestamped .log file is written to the current directory with full run details.
"""

import asyncio
import os

from curio_agent_sdk import Agent, use_run_logger

from tools import calculator, search


async def main() -> None:
    builder = (
        Agent.builder()
        .model("openai:gpt-4.1-mini")
        .system_prompt("You are a helpful assistant. Use the calculator for math and search when asked.")
        .tools([calculator, search])
    )
    logger = use_run_logger(builder, base_name="tool-agent-openai", output_dir=os.path.dirname(os.path.abspath(__file__)))
    agent = builder.build()

    result = await agent.arun("What is (12 + 34) * 2? Reply briefly.")
    print("Output:", result.output)
    path = logger.get_log_path()
    if path:
        print("Run log written to:", path)


if __name__ == "__main__":
    asyncio.run(main())
