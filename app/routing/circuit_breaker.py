import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from app.providers.base import ProviderError


class CircuitOpen(ProviderError):
    def __init__(self, retry_after: int) -> None:
        super().__init__("Provider circuit is open")
        self.retry_after = retry_after


@dataclass(frozen=True)
class Permit:
    generation: int
    probe: bool


class CircuitBreaker:
    """Process-local provider breaker with exactly one recovery probe.

    Generations stop late completions closing a circuit opened by a newer failure.
    A short asyncio lock protects state and is never held during network I/O.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self.state: Literal["closed", "open", "half_open"] = "closed"
        self.failures = 0
        self.opened_at = 0.0
        self.generation = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Permit:
        async with self._lock:
            if self.state == "open":
                remaining = self.recovery_seconds - (self.clock() - self.opened_at)
                if remaining > 0:
                    raise CircuitOpen(max(1, math.ceil(remaining)))
                self.state = "half_open"
                return Permit(self.generation, True)
            if self.state == "half_open":
                raise CircuitOpen(1)
            return Permit(self.generation, False)

    async def success(self, permit: Permit) -> None:
        async with self._lock:
            if permit.generation != self.generation:
                return
            self.failures = 0
            if permit.probe:
                self.state = "closed"
                self.generation += 1

    async def failure(self, permit: Permit) -> None:
        async with self._lock:
            if permit.generation != self.generation:
                return
            self.failures += 1
            if permit.probe or self.failures >= self.failure_threshold:
                self._open()

    def _open(self) -> None:
        self.state = "open"
        self.opened_at = self.clock()
        self.generation += 1

    async def cancel(self, permit: Permit) -> None:
        async with self._lock:
            if permit.probe and permit.generation == self.generation:
                # Cancelled CPU/network work must not leave the exclusive probe occupied.
                self._open()
