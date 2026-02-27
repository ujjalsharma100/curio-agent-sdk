"""
CircuitBreaker — reusable circuit breaker for any external dependency.

Extracted from KeyHealth (llm/router.py) into a standalone utility that
works for LLM providers, connectors, MCP servers, or any external service.

States:
- CLOSED: normal operation, calls pass through
- OPEN: too many failures, calls are blocked until recovery_seconds elapse
- HALF_OPEN: recovery period elapsed, next call is a probe (success → CLOSED, failure → OPEN)

Rate-limit handling:
- A rate-limit failure uses a separate (typically longer) recovery window
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Reusable circuit breaker for any external dependency.

    Example:
        cb = CircuitBreaker(max_failures=3, recovery_seconds=60)

        if cb.is_open:
            # skip this dependency, try fallback
            ...
        else:
            try:
                result = await call_external()
                cb.record_success()
            except RateLimitError:
                cb.record_failure(is_rate_limit=True)
            except Exception:
                cb.record_failure()
    """

    max_failures: int = 3
    recovery_seconds: float = 300.0
    rate_limit_recovery_seconds: float = 3600.0

    # Tracking state (not constructor args — managed internally)
    success_count: int = field(default=0, init=False)
    failure_count: int = field(default=0, init=False)
    consecutive_failures: int = field(default=0, init=False)
    rate_limited: bool = field(default=False, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    last_success_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        if self.consecutive_failures < self.max_failures and not self.rate_limited:
            return CircuitState.CLOSED

        if self.last_failure_time == 0.0:
            return CircuitState.CLOSED

        now = time.monotonic()
        recovery = (
            self.rate_limit_recovery_seconds
            if self.rate_limited
            else self.recovery_seconds
        )
        elapsed = now - self.last_failure_time

        if elapsed >= recovery:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    @property
    def is_open(self) -> bool:
        """True when the circuit is OPEN (blocking calls)."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """True when the circuit is CLOSED (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def allows_request(self) -> bool:
        """True when the circuit allows a request (CLOSED or HALF_OPEN)."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call — resets the circuit to CLOSED."""
        self.success_count += 1
        self.consecutive_failures = 0
        self.rate_limited = False
        self.last_success_time = time.monotonic()

    def record_failure(self, is_rate_limit: bool = False) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if is_rate_limit:
            self.rate_limited = True

    def reset(self) -> None:
        """Force-reset the circuit to CLOSED (e.g. for manual recovery)."""
        self.consecutive_failures = 0
        self.rate_limited = False
        self.last_failure_time = 0.0

    def get_stats(self) -> dict[str, object]:
        """Return a snapshot of circuit breaker statistics."""
        return {
            "state": self.state.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "rate_limited": self.rate_limited,
        }
