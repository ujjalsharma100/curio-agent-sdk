"""
Unit tests for RateLimitMiddleware.
"""

import pytest

from curio_agent_sdk.middleware.rate_limit import RateLimitMiddleware
from curio_agent_sdk.models.llm import LLMRequest, Message


def _make_request(metadata=None):
    return LLMRequest(
        messages=[Message.user("hi")],
        model="gpt-4o",
        provider="openai",
        metadata=metadata or {},
    )


@pytest.mark.unit
class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_rate_limit_under(self):
        mw = RateLimitMiddleware(rate=100.0, burst=10)
        req = _make_request()
        out = await mw.before_llm_call(req)
        assert out is req

    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self):
        mw = RateLimitMiddleware(rate=100.0, burst=10, per_user=True)
        req1 = _make_request(metadata={"user_id": "u1"})
        req2 = _make_request(metadata={"user_id": "u2"})
        await mw.before_llm_call(req1)
        await mw.before_llm_call(req2)
        assert mw._bucket_key(req1) != mw._bucket_key(req2)

    @pytest.mark.asyncio
    async def test_rate_limit_per_agent(self):
        mw = RateLimitMiddleware(rate=100.0, burst=10, per_agent=True)
        req1 = _make_request(metadata={"agent_id": "a1"})
        req2 = _make_request(metadata={"agent_id": "a2"})
        assert "agent:a1" in mw._bucket_key(req1)
        assert "agent:a2" in mw._bucket_key(req2)

    @pytest.mark.asyncio
    async def test_rate_limit_window(self):
        mw = RateLimitMiddleware(rate=10.0, burst=2)
        req = _make_request()
        await mw.before_llm_call(req)
        await mw.before_llm_call(req)
        out = await mw.before_llm_call(req)
        assert out is req
