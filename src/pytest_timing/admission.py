"""One host's CPU budget, fair waiting line, and pressure feedback.

Reservations are atomic. Backfilling may use pledged slots only when it finishes
before the oldest waiter could start. Feedback adjusts future admission without
interrupting running tests. No xdist or platform reads; see ARCHITECTURE.md."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from pytest_timing.model import CONTENDED_PRESSURE

PRESSURE_THRESHOLD = CONTENDED_PRESSURE
"""PSI ``some`` share above which a sample counts as contended."""
OWN_SHARE = 0.75
"""Pressure explained by this run's own CPU rate (this share of the limit or more) is
not held against it: full utilisation with matching throughput is healthy."""


@dataclass(slots=True)
class Waiter:
    slots: int  # the reservation the worker needs before it can go on
    since: float


@dataclass(slots=True)
class Reservation:
    slots: int
    until: float  # projected end of the work behind it
    busy: bool  # is any test granted (as opposed to fixtures merely kept alive)


class Admission:
    """One domain's slots: budget, the moving limit, reservations and the waiting line."""

    STEP_DOWN_AFTER = 3
    RECOVER_AFTER = 8
    COOLDOWN = 10.0

    def __init__(self, budget: int, name: str = "local") -> None:
        self.name = name
        self.budget = max(1, int(budget))
        self.limit = self.budget
        self.reserved: dict[Hashable, Reservation] = {}
        self.waiting: dict[Hashable, Waiter] = {}
        self.forced = 0  # reservations that had to exceed the limit
        self.clamped = 0  # requests above the limit, cut down to it
        self.lowered = 0  # limit reductions from pressure feedback
        self.lowest = self.budget
        self.contended = self.clean = 0
        self.moved_at = -math.inf
        self.last_throttled: int | None = None

    @property
    def used(self) -> int:
        return sum(r.slots for r in self.reserved.values())

    @property
    def free(self) -> int:
        return self.limit - self.used

    def held(self, worker: Hashable) -> int:
        reservation = self.reserved.get(worker)
        return reservation.slots if reservation is not None else 0

    def clamp(self, slots: int) -> int:
        """A request above the limit is cut down to it: it runs alone, not never."""
        if slots > self.limit:
            self.clamped += 1
            return self.limit
        return max(0, slots)

    def head(self) -> Hashable | None:
        """The oldest waiting worker, or ``None``."""
        return min(self.waiting, key=lambda worker: self.waiting[worker].since, default=None)

    def shadow(self, head: Hashable, shortfall: int, excluding: Hashable) -> float:
        """When the head could have its ``shortfall`` slots if nothing new is admitted."""
        available = self.free
        if available >= shortfall:
            return -math.inf
        releases = sorted(
            (r.until, r.slots)
            for w, r in self.reserved.items()
            if w not in (head, excluding) and r.busy
        )
        for until, slots in releases:
            available += slots
            if available >= shortfall:
                return until
        return math.inf

    def fits(self, worker: Hashable, slots: int, finish: float = math.inf) -> bool:
        """Could ``worker``'s reservation be raised to ``slots`` now?

        ``finish`` is when the work behind the request is projected to end; it only
        matters for backfilling past the head of the waiting line.
        """
        slots = min(slots, self.limit)
        delta = slots - self.held(worker)
        if delta <= 0:
            return True
        if delta > self.free:
            return False
        head = self.head()
        if head is None or head == worker:
            return True  # nothing is pledged past the head of the line
        shortfall = min(self.waiting[head].slots, self.limit) - self.held(head)
        if delta <= self.free - shortfall:
            return True
        return finish <= self.shadow(head, shortfall, excluding=worker)

    @property
    def idle(self) -> bool:
        """Is nothing running in this domain (only fixtures kept alive)?"""
        return not any(r.busy for r in self.reserved.values())

    def reserve(self, worker: Hashable, slots: int, until: float, finish: float = math.inf) -> bool:
        """Raise ``worker``'s reservation to ``slots`` if that fits; all or nothing."""
        slots = self.clamp(slots)
        if not self.fits(worker, slots, finish):
            return False
        self.reserved[worker] = Reservation(slots, until, busy=True)
        self.admitted(worker)
        return True

    def admitted(self, worker: Hashable) -> None:
        """``worker`` no longer waits (its request fit, or was already covered)."""
        self.waiting.pop(worker, None)

    def assign(self, worker: Hashable, slots: int, until: float, busy: bool) -> None:
        """Set ``worker``'s reservation without admission: releases, and what the
        scheduler decides when nothing can move, which may exceed the limit (counted)."""
        before = self.held(worker)
        slots = max(0, slots)
        if slots > before and slots - before > self.free:
            self.forced += 1
        self.reserved[worker] = Reservation(slots, until, busy)

    def release(self, worker: Hashable) -> None:
        self.reserved.pop(worker, None)
        self.waiting.pop(worker, None)

    def wait(self, worker: Hashable, slots: int, since: float) -> None:
        """Put ``worker`` in line (keeping its place if it already is)."""
        waiter = self.waiting.get(worker)
        if waiter is None:
            self.waiting[worker] = Waiter(slots, since)
        else:
            waiter.slots = slots

    def set_limit(self, limit: int) -> None:
        limit = max(1, min(self.budget, limit))
        if limit < self.limit:
            self.lowered += 1
        self.limit = limit
        self.lowest = min(self.lowest, limit)

    def observe(self, sample: PressureSample) -> int:
        """Move the limit (-1, 0 or +1) only on sustained external contention."""
        if sample.some is None and sample.throttled is None:
            return 0
        throttled = False
        if sample.throttled is not None:
            if self.last_throttled is not None:
                throttled = sample.throttled > self.last_throttled
            self.last_throttled = sample.throttled
        verdict = throttled or (
            sample.some is not None
            and sample.some >= PRESSURE_THRESHOLD
            and sample.rate < OWN_SHARE * self.limit
        )
        self.contended = self.contended + 1 if verdict else 0
        self.clean = 0 if verdict else self.clean + 1
        direction = -1 if verdict else 1
        count = self.contended if verdict else self.clean
        threshold = self.STEP_DOWN_AFTER if verdict else self.RECOVER_AFTER
        target = self.limit + direction
        if (
            count >= threshold
            and sample.at - self.moved_at >= self.COOLDOWN
            and 1 <= target <= self.budget
        ):
            self.set_limit(target)
            self.moved_at = sample.at
            self.contended = self.clean = 0
            return direction
        return 0

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "budget": self.budget,
            "limit": self.limit,
            "lowest": self.lowest,
            "lowered": self.lowered,
            "forced": self.forced,
            "clamped": self.clamped,
        }


@dataclass(slots=True)
class PressureSample:
    at: float
    some: float | None = None  # PSI: share of the last 10 s some task waited for a CPU
    throttled: int | None = None  # cumulative quota-throttled microseconds
    rate: float = 0.0  # this run's own measured CPU rate, in CPUs
