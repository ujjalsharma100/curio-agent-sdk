"""
BatchLLMClient - simple concurrent batch inference helper.

This client is provider-agnostic and built on top of LLMClient. It lets you
submit a list of LLMRequest objects and retrieve their results together,
executing the underlying LLM calls concurrently.

Example:

    batch_client = BatchLLMClient(llm_client)
    request_ids = await batch_client.submit_batch([req1, req2, req3])
    results = await batch_client.get_results(request_ids)
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from curio_agent_sdk.llm.client import LLMClient
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse


@dataclass
class BatchRequestHandle:
    """Handle for a submitted batch request."""

    id: str
    request: LLMRequest


class BatchLLMClient:
    """
    Lightweight batch inference client built on top of LLMClient.

    This implementation executes individual LLM calls concurrently using
    asyncio, so it works with all providers supported by LLMClient
    (OpenAI, Anthropic, Groq, Ollama, and custom providers).
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._tasks: Dict[str, asyncio.Task[LLMResponse]] = {}

    async def submit_batch(
        self,
        requests: List[LLMRequest],
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Submit a batch of LLMRequest objects for concurrent execution.

        Returns a list of request IDs that can be passed to get_results().
        """
        ids: List[str] = []
        for req in requests:
            rid = uuid.uuid4().hex
            task = asyncio.create_task(self._llm.call(req, run_id=run_id, agent_id=agent_id))
            self._tasks[rid] = task
            ids.append(rid)
        return ids

    async def get_results(self, request_ids: List[str]) -> List[LLMResponse]:
        """
        Await completion of the given request IDs and return their responses.

        Results are returned in the same order as request_ids.
        """
        results: List[LLMResponse] = []
        for rid in request_ids:
            task = self._tasks.pop(rid, None)
            if task is None:
                raise KeyError(f"No pending batch request with id={rid!r}")
            results.append(await task)
        return results

