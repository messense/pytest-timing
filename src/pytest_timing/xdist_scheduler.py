"""Drive xdist with fixture-aware planning and CPU and memory admission.

A worker needs its next item before starting the current one. Dispatch therefore
admits the preceding item first; shutdown admits or withdraws the final item.
``load`` retains unsent plans; ``worksteal`` refunds recorded charges for stolen
queues. Projected fixture state and actual live holds remain separate.

Only this module imports the xdist scheduler. Import it lazily; see
ARCHITECTURE.md for lifecycle, admission, and runtime-request invariants."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pytest
from xdist.scheduler.load import LoadScheduling  # type: ignore[import-untyped]
from xdist.workermanage import WorkerController  # type: ignore[import-untyped]

from pytest_timing import xdist_compat
from pytest_timing.admission import Admission, PressureSample
from pytest_timing.demand import Declarations
from pytest_timing.fixtures import key_scope
from pytest_timing.schedule import (
    EPSILON,
    Charge,
    Costs,
    Estimates,
    Lane,
    pay,
    plan,
    projected_finish,
    transfer,
)
from pytest_timing.telemetry import Pressure

MIN_PENDING = 2
"""A worker runs a test only once it knows the following one (or that there is none)."""

MAX_QUEUE_SECONDS = 0.1
"""Upper bound on the estimated work dispatched per worker beyond the minimum count.

Larger batches save controller round trips on tiny tests; smaller ones leave more on
the controller to rebalance. The bound is scaled down for short suites (``_budget``).
"""

SAMPLE_INTERVAL = 1.0
"""Seconds between pressure samples used for admission feedback."""

AUTO_MEMORY_SHARE = 0.8
"""``--timing-memory auto``: this share of what the host offers is the budget. Memory
estimates are rises above each worker's footprint, and those footprints, the
controller, and whatever else the host runs are not in the estimates."""


@dataclass
class CpuSetup:
    """How the scheduler learns about CPU and memory: budgets, declarations, hosts."""

    # Slots per domain: a number, "auto" (detect per host and enforce it, even
    # below the worker count), or None (gate only when something declares CPU).
    cpus: int | Literal["auto"] | None = None
    # Bytes per domain: a number, "auto" (a share of what each host offers), or
    # None (no memory gate). Demand comes from recorded history, never declarations.
    memory: int | Literal["auto"] | None = None
    declarations: Callable[[Any], Declarations | None] = field(default=lambda node: None)
    domain_of: Callable[[Any], str] = field(default=lambda node: "local")
    pressure: Pressure | None = None  # the controller host's signals, for ``local``


@dataclass(frozen=True)
class AdmissionWait:
    worker: str
    index: int  # -1 for a worker that was parked and never ran what it waited for
    attempt: int
    seconds: float
    gates: frozenset[str] = frozenset()  # the gate kinds that refused: cpu, memory
    ended: float | None = None  # controller epoch, for placement on the report timeline

    def held_by(self, kind: str) -> bool:
        return kind in self.gates or not self.gates


@dataclass(frozen=True)
class FixtureRequest:
    index: int
    key: str
    setup: int
    hold: int
    attempt: int = 0
    request_id: int = 0
    cancelled: bool = False


class DurationScheduling(LoadScheduling):  # type: ignore[misc]  # xdist ships no types
    # Inherited state this class relies on, typed here because xdist is untyped.
    numnodes: int
    node2collection: dict[WorkerController, list[str]]
    node2pending: dict[WorkerController, list[int]]
    pending: list[int]
    collection: list[str] | None
    log: Any

    def __init__(
        self,
        config: pytest.Config,
        log: Any,
        estimates: Estimates,
        stealing: bool = False,
        cpu: CpuSetup | None = None,
    ) -> None:
        super().__init__(config, log)
        self.estimates = estimates
        self.stealing = stealing
        self.cpu = cpu
        self.costs: Costs | None = None
        self.lanes: dict[WorkerController, Lane] = {}
        self.dispatched: dict[WorkerController, dict[int, Charge]] = {}
        self.clock = time.monotonic
        self.epoch_clock = time.time
        self.budget = 0.0
        self.known = 0  # collected tests with a recorded duration, once scheduled
        # Work stealing: one request in flight at a time, and who is waiting for work.
        self.steal_requested_from_node: WorkerController | None = None
        self.starving: list[WorkerController] = []
        # CPU and memory admission: one budget per domain each, a reservation per worker.
        self.declarations: Declarations | None = None
        self.admissions: dict[str, Admission] = {}
        self.memory_admissions: dict[str, Admission] = {}
        self.domain: dict[WorkerController, str] = {}
        self.hosts: dict[str, dict[str, Any]] = {}  # what each domain's workers detected
        self.waiting: dict[WorkerController, float] = {}  # since when its head waits
        self.refused: dict[WorkerController, set[str]] = {}  # which gate kinds held it
        self.waits: list[AdmissionWait] = []
        self.parked: list[AdmissionWait] = []  # waits of workers that then ran nothing
        self.rates: dict[str, float] = {}  # last measured CPU rate per worker id
        self.last_sample = -math.inf
        self.cancelled: dict[WorkerController, set[int]] = {}  # withdrawn, never run
        self.withdrawn = 0  # tests withdrawn over the run, for the record
        # A blocked request is evidence that the protocol has already started,
        # even though it currently has no granted work.
        self.requests: dict[WorkerController, FixtureRequest] = {}
        self.live_holds: dict[WorkerController, int] = {}  # worker snapshots, not lane projections
        self.finished: set[WorkerController] = set()  # exited, possibly still registered by xdist

    @property
    def tests_finished(self) -> bool:
        if self.steal_requested_from_node is not None or self.waiting:
            return False
        return bool(super().tests_finished)

    def add_node(self, node: WorkerController) -> None:
        super().add_node(node)
        self.lanes[node] = Lane(free=self.clock())
        self.dispatched[node] = {}
        self.domain[node] = self.cpu.domain_of(node) if self.cpu is not None else "local"
        # Every shutdown, xdist's own included, first settles what the worker holds.
        xdist_compat.intercept_shutdown(node, lambda: self._withdraw(node))

    def schedule(self) -> None:
        assert self.collection_is_completed
        if self.collection is not None:
            # A node was added after the initial distribution (a restart). Work that
            # lost its lane while no worker was alive is planned onto the newcomer.
            assert self.costs is not None
            planned = {index for lane in self.lanes.values() for index in lane.plan}
            orphans = [index for index in self.pending if index not in planned]
            if orphans:
                self._replan(orphans)
            else:
                for node in self.nodes:
                    self.check_schedule(node)
            return
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return
        collection = next(iter(self.node2collection.values()))
        self.collection = collection
        self.declarations = self._declarations()
        self.costs = Costs(self.estimates, collection, self.declarations)
        self._setup_admission()
        self.pending[:] = range(len(collection))
        self.known = self.estimates.known(collection)
        self.budget = self._budget()
        if not collection:
            return
        now = self.clock()
        for lane in self.lanes.values():
            lane.free = now
        plan(self.costs, self.pending, self._live_lanes(), self._slots())
        # Heaviest heads first, so the slots go to the tests hardest to place, and
        # each worker's head is admitted before the next worker is looked at.
        ordered = sorted(self.nodes, key=lambda n: -self._head_demand(n))
        if self.stealing:
            for node in ordered:
                indices = self._fill(node, len(self.pending) + 1, borrow=False)
                if indices:
                    node.send_runtest_some(indices)
            for node in ordered:
                self._check_stealing(node)  # a worker left short steals or shuts down
        else:
            for node in ordered:
                self.check_schedule(node)  # fills, and shuts down what has nothing to gain

    def check_schedule(self, node: WorkerController, duration: float = 0) -> None:
        if node.shutting_down or node in self.finished:
            return
        if self.stealing:
            self._check_stealing(node)
            return
        queue = self.node2pending[node]
        limit = self.maxschedchunk or len(self.pending)
        indices = self._fill(node, max(limit, MIN_PENDING - len(queue)))
        if indices:
            node.send_runtest_some(indices)
        if self._parked(node):
            return  # it waits for holds elsewhere to be let go (``_resolve_stalls``)
        if not self.pending or len(queue) < MIN_PENDING:
            self._shutdown(node)  # nothing more will come, or it would only block
        self.log("num items waiting for node:", len(self.pending))

    def mark_test_complete(
        self, node: WorkerController, item_index: int, duration: float = 0
    ) -> None:
        request = self.requests.get(node)
        if request is not None and request.index == item_index:
            # Setup failed or timed out before a grant. Never answer that stale
            # request after this test has finished (or charge its wait to the tail).
            self._admitted(node)
            self.requests.pop(node)
        lane = self.lanes.get(node)
        if lane is not None:
            self.dispatched[node].pop(item_index, None)
            lane.free = self.clock() + sum(c.seconds for c in self.dispatched[node].values())
        # The protocol is complete, teardown included: the reservation shrinks first.
        self.node2pending[node].remove(item_index)
        self._reserve(node)
        self._sample()
        self.check_schedule(node, duration=duration)
        self._admit_waiting()

    def mark_test_pending(self, item: str) -> None:
        assert self.collection is not None and self.costs is not None
        index = self.collection.index(item)
        self.pending.insert(0, index)
        self._replan([index])
        self._admit_waiting()

    def remove_node(self, node: WorkerController) -> str | None:
        # A worker runs its head only once it knows the test after it, or that
        # there is none. A runtime request also proves the protocol started: its
        # temporary lack of a permit must not turn a crash into a silent replay.
        started = bool(self.node2pending[node]) and (
            node in self.requests or self.node2pending[node][0] in self._granted(node)
        )
        queued = self.node2pending.pop(node)
        lane = self.lanes.pop(node)
        self.dispatched.pop(node, None)
        self._release_runtime(node, idle=not queued)
        self.cancelled.pop(node, None)
        self.finished.discard(node)
        self.domain.pop(node, None)
        if node is self.steal_requested_from_node:
            self.steal_requested_from_node = None  # its reply will never come
        if node in self.starving:
            self.starving.remove(node)
        if not queued and not lane.plan:
            self._admit_waiting()
            return None
        assert self.collection is not None
        crashitem = self.collection[queued.pop(0)] if started else None
        # Its dispatched-but-unrun tests go back to the pool; its unsent plan was
        # never out of it. Both get planned again over the workers that are left.
        self.pending.extend(queued)
        self._replan(queued + lane.plan)
        self._admit_waiting()
        return crashitem

    def worker_finished(self, node: WorkerController) -> None:
        """Release an exited worker even when xdist skips ``remove_node``.

        On maxfail a worker reports ``shouldfail`` and xdist leaves its queued
        tests in the scheduler. Its processes and fixtures are nevertheless gone;
        their reservations must not keep another worker's run-time request blocked.
        Leave the queue intact for xdist's normal removal or crash bookkeeping.
        """
        if node not in self.node2pending:
            return
        self.finished.add(node)
        self._release_runtime(node, idle=not self.node2pending[node])
        self._admit_waiting()

    def _release_runtime(self, node: WorkerController, idle: bool = False) -> None:
        """Forget ``node``'s run-time state. ``idle`` says it held no test: a wait
        it was in is then a parked worker's, with no test to record it on."""
        self.requests.pop(node, None)
        self.live_holds.pop(node, None)
        since = self.waiting.pop(node, None)
        gates = frozenset(self.refused.pop(node, ()))
        if since is not None and idle and self.clock() - since > 0:
            self.parked.append(
                AdmissionWait(xdist_compat.node_id(node), -1, 0, self.clock() - since, gates)
            )
        self.rates.pop(xdist_compat.node_id(node), None)
        for admission in self._gates(node):
            admission.release(node)

    def _budget(self) -> float:
        """Estimated seconds to keep dispatched per worker; smaller for short suites."""
        assert self.costs is not None
        total = sum(self.costs.own)
        return min(MAX_QUEUE_SECONDS, total / (10 * max(1, self.numnodes)))

    def _live_lanes(self) -> list[Lane]:
        return [
            lane
            for node, lane in self.lanes.items()
            if not node.shutting_down and node not in self.finished
        ]

    def _refresh(self) -> None:
        """Bring the projected free times up to date with the clock.

        A worker projected to be done by now is late: it is assumed busy with all
        its dispatched work from now. One still projected to be busy cannot be free
        before it has run the tests it has not started yet (all but the first in
        its queue), however long the running one has already taken.
        """
        now = self.clock()
        for node, lane in self.lanes.items():
            charged = self.dispatched[node]
            if lane.free <= now:
                lane.free = now + sum(c.seconds for c in charged.values())
            else:
                unstarted = sum(charged[i].seconds for i in self.node2pending[node][1:])
                lane.free = max(lane.free, now + unstarted)

    def _low(self, node: WorkerController) -> bool:
        """Time to refill: below the minimum count, or below half the time budget."""
        queue = self.node2pending[node]
        charged = sum(c.seconds for c in self.dispatched[node].values())
        return len(queue) < MIN_PENDING or charged < self.budget / 2 - EPSILON

    def _wants_more(self, node: WorkerController) -> bool:
        """Keep filling up to the minimum count and the full time budget."""
        if self.stealing:
            return True  # the whole plan goes out; rebalancing is by stealing
        queue = self.node2pending[node]
        return (
            len(queue) < MIN_PENDING
            or sum(c.seconds for c in self.dispatched[node].values()) < self.budget - EPSILON
        )

    def _replan(self, indices: Iterable[int]) -> None:
        """Lay ``indices`` (in ``pending``, in no lane's plan) out over the live
        lanes and let every worker pick up what landed on it. With no live lane
        they stay orphans, planned when a worker next joins (``schedule``)."""
        indices = list(indices)
        if not indices:
            return
        assert self.costs is not None
        self._refresh()
        plan(self.costs, indices, self._live_lanes(), self._slots())
        for node in list(self.node2pending):
            self.check_schedule(node)

    def _fill(self, node: WorkerController, limit: int, borrow: bool = True) -> list[int]:
        """Dispatch from the node's plan, refilling the plan from others when it runs dry.

        Every test after the first lets the one before it start, so it goes out only
        once that one is admitted (``_grant``); a refusal ends the batch there. A
        test that cannot fit next to what the other workers hold is passed over: it
        stays in the plan until they let go, and a worker left with nothing else
        parks (``_parked``) rather than take it.
        """
        assert self.costs is not None
        costs = self.costs
        lane = self.lanes[node]
        queue = self.node2pending[node]
        out: list[int] = []
        if not self.stealing and not self._low(node):
            return out
        limit_slots = self._limit(node)
        elsewhere = self._held_elsewhere(node) if limit_slots and costs.holds else 0
        limit_bytes = self._memory_limit(node)
        kept_elsewhere = self._kept_elsewhere(node) if limit_bytes and costs.retained else 0
        while self.pending and len(out) < limit and self._wants_more(node):
            if not lane.plan:
                if not borrow:
                    break
                self._refresh()
                if not transfer(costs, self._live_lanes(), lane, self._slots()):
                    break
            position = charge = None
            blocked: set[str] = set()  # the gate kinds that passed tests over
            for i, index in enumerate(lane.plan):
                charge = costs.charge(index, lane.fixtures)
                # Could it never run here while the other workers keep what they
                # hold now? Only their exit would make room: pass it over.
                if elsewhere and min(charge.peak, limit_slots) + elsewhere > limit_slots:
                    blocked.add("cpu")
                elif (
                    kept_elsewhere
                    and min(charge.memory, limit_bytes) + kept_elsewhere > limit_bytes
                ):
                    blocked.add("memory")
                else:
                    position = i
                    break
            if position is None or charge is None:
                if not queue and not out:
                    self.waiting.setdefault(node, self.clock())  # parked
                    self.refused.setdefault(node, set()).update(blocked)
                break
            if queue and not self._grant(node):
                break
            index = lane.plan.pop(position)
            self.dispatched[node][index] = charge
            lane.dispatch(charge)
            self.pending.remove(index)
            queue.append(index)
            out.append(index)
        return out

    def _declarations(self) -> Declarations | None:
        if self.cpu is None:
            return None
        for node in self.nodes:
            declaration = self.cpu.declarations(node)
            if declaration is not None:
                return declaration
        return None

    def _setup_admission(self) -> None:
        """One budget per domain: the explicit one, or what its workers detected.

        Asked for (``auto``), the detected budget is enforced as it is, even below
        the number of workers: that is what a container's quota means. Turned on
        only by a declaration, it is never below the number of workers there.
        Heavy tests and held slots can still delay plain tests.
        """
        assert self.costs is not None
        cpu = self.cpu
        if cpu is None:
            return
        members: dict[str, list[WorkerController]] = {}
        for node in self.nodes:
            members.setdefault(self.domain[node], []).append(node)
            declaration = cpu.declarations(node)
            if declaration is not None and declaration.host:
                self.hosts.setdefault(self.domain[node], dict(declaration.host))
        if cpu.cpus is not None or self.costs.any_demand:  # else nobody asked: no gate
            for domain, nodes in members.items():
                if isinstance(cpu.cpus, int):
                    budget = cpu.cpus
                else:
                    detected = [
                        int(d.host["budget"])
                        for d in (cpu.declarations(n) for n in nodes)
                        if d is not None and d.host.get("budget")
                    ]
                    budget = min(detected) if detected else len(nodes)
                    if cpu.cpus is None:
                        budget = max(budget, len(nodes))
                self.admissions[domain] = Admission(budget, domain)
        if cpu.memory is None:
            return
        for domain, nodes in members.items():
            if isinstance(cpu.memory, int):
                size = cpu.memory
            else:
                offered = [
                    int(d.host["memory"])
                    for d in (cpu.declarations(n) for n in nodes)
                    if d is not None and d.host.get("memory")
                ]
                size = int(min(offered) * AUTO_MEMORY_SHARE) if offered else 0
            if size > 0:
                self.memory_admissions[domain] = Admission(size, domain, kind="memory")

    @property
    def gated(self) -> bool:
        return bool(self.admissions or self.memory_admissions)

    def _slots(self) -> int | None:
        """The CPU budget the planner should pack against: all domains' limits."""
        if not self.admissions:
            return None
        return sum(admission.limit for admission in self.admissions.values())

    def _admission(self, node: WorkerController) -> Admission | None:
        return self.admissions.get(self.domain.get(node, ""))

    def _memory_admission(self, node: WorkerController) -> Admission | None:
        return self.memory_admissions.get(self.domain.get(node, ""))

    def _gates(self, node: WorkerController) -> list[Admission]:
        """The admissions ``node`` goes through: CPU and memory, whichever exist."""
        gates = [self._admission(node), self._memory_admission(node)]
        return [gate for gate in gates if gate is not None]

    def _any_gate(self, domain: str) -> Admission | None:
        """One of a domain's admissions, for state they keep in step (idle, held)."""
        return self.admissions.get(domain) or self.memory_admissions.get(domain)

    def _memory_limit(self, node: WorkerController) -> int:
        admission = self._memory_admission(node)
        return admission.limit if admission is not None else 0

    def _kept_elsewhere(self, node: WorkerController) -> int:
        """Bytes the other live workers of ``node``'s domain keep for fixtures."""
        assert self.costs is not None
        domain = self.domain.get(node)
        kept = 0
        for other, lane in self.lanes.items():
            if (
                other is node
                or other.shutting_down
                or other in self.finished
                or self.domain.get(other) != domain
            ):
                continue
            kept += self.costs.kept_memory(lane.fixtures)
        return kept

    def _head_demand(self, node: WorkerController) -> int:
        assert self.costs is not None
        lane = self.lanes[node]
        if not lane.plan:
            return 0
        return self.costs.charge(lane.plan[0], lane.fixtures).peak

    def _limit(self, node: WorkerController) -> int:
        admission = self._admission(node)
        return admission.limit if admission is not None else 0

    def _held_elsewhere(self, node: WorkerController) -> int:
        """Slots the other live workers of ``node``'s domain hold for fixtures."""
        assert self.costs is not None
        domain = self.domain.get(node)
        held = 0
        for other in self.lanes:
            if (
                other is node
                or other.shutting_down
                or other in self.finished
                or self.domain.get(other) != domain
            ):
                continue
            # Parking handles holds with no scheduled release. Draining workers
            # will release theirs; the admission gate still accounts for them.
            held += self._holds(other)
        return held

    def _parked(self, node: WorkerController) -> bool:
        """Is ``node`` idle with a plan it cannot start next to what the other workers
        hold? It waits, undispatched (its plan can still move), is retried at every
        release and exit, and nothing is pledged to it, since no release can help."""
        return (
            node in self.waiting
            and not self.node2pending[node]
            and bool(self.lanes[node].plan)
            and node not in self.requests
        )

    def _granted(self, node: WorkerController) -> list[int]:
        """The tests ``node`` may run without another word: its whole queue once
        told to shut down, otherwise all but the tail; never a withdrawn test, and
        nothing while it is blocked in a run-time request."""
        if node in self.requests:
            return []
        queue = self.node2pending[node]
        granted = list(queue) if xdist_compat.shutdown_was_sent(node) else queue[:-1]
        withdrawn = self.cancelled.get(node)
        return [i for i in granted if i not in withdrawn] if withdrawn else granted

    def _need(self, node: WorkerController, tests: list[int]) -> int:
        """Slots ``node`` must hold to run ``tests``; idle, what its fixtures hold."""
        assert self.costs is not None
        withdrawn = self.cancelled.get(node)
        if withdrawn:
            tests = [i for i in tests if i not in withdrawn]
        if tests:
            demand = self.dispatched[node]
            live = self.live_holds.get(node, 0)
            return max(demand[i].with_live_holds(live) for i in tests)
        return self._holds(node)

    def _need_memory(self, node: WorkerController, tests: list[int]) -> int:
        """Bytes ``node`` must hold to run ``tests``; idle, what its fixtures keep.

        Fixture memory is projected through the queue: unlike CPU holds, workers
        report nothing about it, so the lane's fixture state stands in."""
        assert self.costs is not None
        withdrawn = self.cancelled.get(node)
        if withdrawn:
            tests = [i for i in tests if i not in withdrawn]
        if tests:
            demand = self.dispatched[node]
            return max(demand[i].memory for i in tests)
        return self.costs.kept_memory(self.lanes[node].fixtures)

    def _needs(self, node: WorkerController, tests: list[int]) -> list[tuple[Admission, int]]:
        """Each of ``node``'s gates with what ``tests`` need from it."""
        needs = []
        admission = self._admission(node)
        if admission is not None:
            needs.append((admission, self._need(node, tests)))
        memory = self._memory_admission(node)
        if memory is not None:
            needs.append((memory, self._need_memory(node, tests)))
        return needs

    def _raise(
        self, node: WorkerController, needs: list[tuple[Admission, int]], wait: bool = True
    ) -> bool:
        """Raise ``node``'s reservations to ``needs`` through the gates, all or none;
        ``wait`` puts a refused worker in every line, to be retried at the next
        release. A rise that fits one gate but not the other takes nothing."""
        assert needs
        lane = self.lanes[node]
        rising = [(gate, need) for gate, need in needs if need > gate.held(node)]
        refusing = [gate.kind for gate, need in rising if not gate.fits(node, need, lane.free)]
        if refusing:
            if wait:
                since = self.waiting.setdefault(node, self.clock())
                self.refused.setdefault(node, set()).update(refusing)
                for gate, need in needs:
                    gate.wait(node, need, since)
            return False
        for gate, need in needs:
            if need > gate.held(node):
                reserved = gate.reserve(node, need, lane.free, finish=lane.free)
                assert reserved  # it fit a moment ago, and nothing has changed since
            else:
                gate.assign(node, gate.held(node), lane.free, busy=True)
        self._admitted(node)
        return True

    def _grant(self, node: WorkerController) -> bool:
        """May the tail of ``node``'s queue start?

        True raises the reservation to cover the whole queue. A rise in demand while
        lighter tests are still ahead of the tail is deferred, not waited for: the
        worker is busy, and the slots would idle until it got there. A refusal at the
        head puts the worker in line.
        """
        queue = self.node2pending[node]
        if not self._gates(node) or not queue:
            return True
        needs = self._needs(node, queue)
        if len(queue) > 1 and any(need > gate.held(node) for gate, need in needs):
            return False
        return self._raise(node, needs)

    def _admitted(self, node: WorkerController) -> None:
        for gate in self._gates(node):
            gate.admitted(node)
        since = self.waiting.pop(node, None)
        gates = frozenset(self.refused.pop(node, ()))
        if since is not None:
            queue = self.node2pending[node]
            waited = self.clock() - since
            if queue and waited > 0:
                request = self.requests.get(node)
                index = request.index if request is not None else queue[0]
                attempt = request.attempt if request is not None else 0
                self.waits.append(
                    AdmissionWait(
                        xdist_compat.node_id(node),
                        index,
                        attempt,
                        waited,
                        gates,
                        self.epoch_clock(),
                    )
                )

    def _reserve(self, node: WorkerController) -> None:
        """Recompute ``node``'s reservation from what it may run without another word."""
        if not self._gates(node) or node not in self.node2pending or node in self.finished:
            return
        free = self.lanes[node].free
        if node.shutting_down and not self.node2pending[node]:
            for gate in self._gates(node):
                gate.assign(node, 0, free, busy=False)  # about to exit
            return
        granted = self._granted(node)
        for gate, need in self._needs(node, granted):
            gate.assign(node, need, free, bool(granted))

    def _shutdown(self, node: WorkerController) -> None:
        """Shut ``node`` down once everything it holds may run.

        Whatever is left in its plan (tests blocked by holds elsewhere) goes to the
        workers that stay, right away: a worker told to shut down takes nothing
        more, and its exit is what unblocks them.
        """
        if node.shutting_down:
            return
        if self.node2pending[node] and not self._grant(node):
            return  # deferred or waiting: it is checked again at the next release
        lane = self.lanes[node]
        stranded = list(lane.plan)
        if stranded and not [n for n in self.node2pending if n is not node and not n.shutting_down]:
            return  # the last worker standing keeps its plan
        node.shutdown()  # settles its reservation on the way (``_withdraw``)
        lane.plan.clear()
        self._replan(stranded)

    def _withdraw(self, node: WorkerController) -> None:
        """``node`` is being shut down by xdist: it will run everything it holds.

        What its reservation covers, or fits now, is granted. Otherwise the last
        test in its queue, the one it could not start without another word from
        the controller, is withdrawn (the worker skips it); a worker that cannot
        be told runs it anyway, counted as a forced admission.
        """
        queue = self.node2pending.get(node)
        if not self._gates(node) or not queue:
            return
        lane = self.lanes[node]
        if node in self.requests:
            # Shutdown also means "drain your queue" in xdist. The running test
            # still waits for its grant; completion or worker exit frees the slots.
            self._reserve(node)
            return
        needs = self._needs(node, queue)
        if self._raise(node, needs, wait=False):
            return
        tail = queue[-1]
        if xdist_compat.cancel_tests(node, [tail]):
            self.cancelled.setdefault(node, set()).add(tail)
            self.withdrawn += 1
            self.waiting.pop(node, None)  # it never ran: no wait to record
            self.refused.pop(node, None)
            for gate, need in self._needs(node, queue):
                gate.admitted(node)
                gate.assign(node, need, lane.free, busy=len(queue) > 1)
            return
        for gate, need in needs:
            gate.assign(node, need, lane.free, busy=True)  # over the limit: counted
        self._admitted(node)

    def _admit_waiting(self) -> None:
        """Retry the waiting workers, oldest first; then, if a domain is at a
        standstill, decide what gives."""
        for node in sorted(self.waiting, key=self.waiting.__getitem__):
            if node not in self.node2pending or node in self.finished:
                continue
            if node in self.requests:
                self._retry_request(node)
            elif not node.shutting_down:
                self.check_schedule(node)
        self._resolve_stalls()

    def _resolve_stalls(self) -> None:
        """A domain where nothing runs and nobody in line fits can only move by
        letting go of holds or by going over the limit. A parked worker is shut
        down first: its exit releases what it holds and its plan moves on. Failing
        that, the oldest head is admitted over the limit, counted as forced."""
        for domain in {*self.admissions, *self.memory_admissions}:
            admission = self._any_gate(domain)
            if admission is None or not admission.idle:
                continue
            candidates = sorted(
                (
                    n
                    for n in self.waiting
                    if self.domain.get(n) == domain
                    and n in self.node2pending
                    and n not in self.finished
                    and (not n.shutting_down or n in self.requests)
                ),
                key=self.waiting.__getitem__,
            )
            if not candidates:
                continue
            parked = next((n for n in candidates if self._parked(n)), None)
            if parked is not None:
                self._shutdown(parked)
            elif not any(
                n.shutting_down and n not in self.requests and admission.held(n)
                for n in self.node2pending
                if self.domain.get(n) == domain and n not in self.finished
            ):
                # A draining worker blocked inside a request cannot finish until
                # granted. Other draining workers can still release their holds
                # on exit: wait for that before resorting to a forced admission.
                self._force(candidates[0])

    def _force(self, node: WorkerController) -> None:
        """Admit ``node``'s head, or the request its running test is blocked on,
        over the limit: nothing else can move. Counted by ``assign``."""
        assert self.costs is not None
        lane = self.lanes[node]
        request = self.requests.get(node)
        if request is not None:
            for gate, need in self._request_needs(node, request):
                gate.assign(node, need, lane.free, busy=True)
            self._complete_request(node)
            return
        queue = self.node2pending[node]
        for gate, need in self._needs(node, queue):
            gate.assign(node, need, lane.free, busy=True)
        self._admitted(node)
        self.check_schedule(node)  # its next test, or its shutdown, may go out now

    def request(
        self,
        node: WorkerController,
        index: int,
        key: str,
        setup: int,
        hold: int,
        holds: int | None = None,
        attempt: int = 0,
        request_id: int = 0,
        cancelled: bool = False,
    ) -> None:
        """The test ``index`` running on ``node`` is about to set up the fixture
        behind ``key``, which it did not declare (``getfixturevalue``): the worker
        blocks until the slots are granted.

        While it waits the worker runs nothing, so its reservation drops to what
        its fixtures hold; the request is then a rise to the set-up's slots next to
        those holds, through the same gate as any other rise, and the fixture
        counts as alive on that worker afterwards, holds included. Two workers
        asking at once therefore take turns instead of deadlocking on the slots
        of the tests they are blocked in.
        """
        if node in self.finished:
            return
        if not self._gates(node) or node not in self.node2pending or self.costs is None:
            xdist_compat.grant_slots(node, key, request_id)  # nothing to gate
            return
        if holds is not None:
            self.live_holds[node] = holds
        if cancelled:
            request = self.requests.get(node)
            if request is None or request.request_id != request_id:
                return  # already granted: that grant still covers unwinding
            self._admitted(node)  # close the setup wait before requesting cleanup slots
            self.requests[node] = replace(request, setup=0, hold=0, cancelled=True)
        else:
            self.requests[node] = FixtureRequest(index, key, setup, hold, attempt, request_id)
        self._reserve(node)  # blocked, it runs nothing: down to what it holds
        if not self._retry_request(node):
            self._admit_waiting()  # the slots it let go may serve the head of the line

    def _holds(self, node: WorkerController) -> int:
        """Slots the fixtures alive on ``node`` hold."""
        assert self.costs is not None
        if node in self.live_holds:
            return self.live_holds[node]
        # Before the first worker snapshot, retain the conservative projection.
        # Runtime requests always carry a snapshot, including zero live holds.
        lane = self.lanes[node]
        return self.costs.hold(lane.fixtures)

    def update_holds(self, node: WorkerController, holds: int) -> None:
        """Keep fixture lifetimes separate from state projected through the queue.

        Events arrive on xdist's main loop, before the same worker's next request
        or protocol completion. Keep its execution reservation until completion;
        a fixture finalizer is still part of a running test.
        """
        if node in self.node2pending and node not in self.finished:
            self.live_holds[node] = holds

    def _request_charges(
        self, node: WorkerController, request: FixtureRequest
    ) -> dict[int, Charge]:
        """Preview the same resource reconciliation that a successful grant commits."""
        assert self.costs is not None
        demand = dict(self.dispatched[node])
        if request.cancelled:
            return demand
        shared = key_scope(request.key) != "function"
        kept = self.costs.retained.get(request.key, 0) if shared else 0
        live = self._holds(node)
        for index, charge in demand.items():
            current = index == request.index
            if current or shared:
                demand[index] = charge.adding_fixture(
                    request.key,
                    request.setup if current else 0,
                    request.hold,
                    live if current else 0,
                    kept,
                )
        return demand

    def _request_needs(
        self, node: WorkerController, request: FixtureRequest
    ) -> list[tuple[Admission, int]]:
        charges = self._request_charges(node, request)
        queue = self.node2pending[node]
        granted = list(queue) if xdist_compat.shutdown_was_sent(node) else queue[:-1]
        if request.index not in granted:
            granted.append(request.index)
        withdrawn = self.cancelled.get(node, set())
        granted = [i for i in granted if i not in withdrawn]
        needs = []
        admission = self._admission(node)
        if admission is not None:
            live = self._holds(node)
            needs.append(
                (
                    admission,
                    max(
                        request.setup + live,
                        max((charges[i].with_live_holds(live) for i in granted), default=live),
                    ),
                )
            )
        memory = self._memory_admission(node)
        if memory is not None:
            needs.append(
                (
                    memory,
                    max((charges[i].memory for i in granted), default=self._need_memory(node, [])),
                )
            )
        return needs

    def _retry_request(self, node: WorkerController) -> bool:
        request = self.requests[node]
        if not self._raise(node, self._request_needs(node, request)):
            return False
        self._complete_request(node)
        return True

    def _complete_request(self, node: WorkerController) -> None:
        request = self.requests[node]
        self._admitted(node)  # retain its execution identity until the wait is recorded
        self.requests.pop(node)
        if not request.cancelled:
            self._learn(node, request)
        xdist_compat.grant_slots(node, request.key, request.request_id)

    def _learn(self, node: WorkerController, request: FixtureRequest) -> None:
        """Commit the granted fixture's resource needs onto the queued work.

        Actual holds change only after the worker reports successful setup,
        not when a permit is issued."""
        assert self.costs is not None
        lane = self.lanes[node]
        key = request.key
        self.dispatched[node].update(self._request_charges(node, request))
        if key_scope(key) == "function":
            # Its hold dies at this test's teardown. It is never a warm shared
            # instance, and must not increase any later test's reservation.
            return
        self.costs.learn(key, request.setup, request.hold)
        queue = self.node2pending[node]
        if queue and key in self.costs.kept(queue[-1], {key}):
            pay(lane.fixtures, frozenset({key}))

    def observe_rate(self, worker_id: str, rate: float) -> None:
        """A worker's latest measured CPU rate (CPUs), from its test reports."""
        self.rates[worker_id] = rate

    def _sample(self) -> None:
        """Observe local pressure at most once per interval."""
        admission = self.admissions.get("local")
        if admission is None or self.cpu is None or self.cpu.pressure is None:
            return
        now = self.clock()
        if now - self.last_sample < SAMPLE_INTERVAL:
            return
        self.last_sample = now
        pressure = self.cpu.pressure
        # This run's own CPU rate: the last measured rate of every local worker
        # that is running something now; idle and departed workers count nothing.
        rate = 0.0
        for node in self.node2pending:
            reservation = admission.reserved.get(node)
            if self.domain.get(node) == "local" and reservation is not None and reservation.busy:
                rate += self.rates.get(xdist_compat.node_id(node), 0.0)
        sample = PressureSample(now, pressure.some(), pressure.throttled(), rate)
        if admission.observe(sample) > 0:
            self._admit_waiting()

    def cpu_summary(self) -> dict[str, Any] | None:
        """What happened on the CPU side, for the run's metadata and the summary."""
        if self.cpu is None:
            return None
        domains: dict[str, Any] = {}
        for domain, admission in self.admissions.items():
            domains[domain] = admission.summary()
        for domain, host in self.hosts.items():
            domains.setdefault(domain, {})["host"] = host
        heavy = 0
        if self.costs is not None:
            heavy = sum(1 for s in self.costs.slots if s > 1)
        return {
            "gated": self.gated,
            "domains": domains,
            "heavy_tests": heavy,
            **self._waits("cpu"),
            "cancelled": self.withdrawn,
        }

    def _waits(self, kind: str) -> dict[str, Any]:
        """How long tests waited at the gates of ``kind``, and how long workers sat
        parked at them without ever running what they waited for."""
        waits = [w for w in self.waits if w.held_by(kind)]
        parked = [w for w in self.parked if w.held_by(kind)]
        return {
            "waited": round(sum(w.seconds for w in waits), 6),
            "waited_tests": len({(w.worker, w.index, w.attempt) for w in waits}),
            "parked": round(sum(w.seconds for w in parked), 6),
            "parked_workers": len({w.worker for w in parked}),
        }

    def memory_summary(self) -> dict[str, Any] | None:
        """What happened on the memory side, for the run's metadata and the summary."""
        if self.cpu is None or self.cpu.memory is None:
            return None
        domains: dict[str, Any] = {}
        for domain, admission in self.memory_admissions.items():
            domains[domain] = admission.summary()
            offered = (self.hosts.get(domain) or {}).get("memory")
            if offered:
                domains[domain]["host"] = int(offered)
        known = largest = total = 0
        if self.costs is not None and self.collection is not None:
            total = len(self.collection)
            known = sum(1 for nodeid in self.collection if nodeid in self.estimates.memory)
            largest = max(self.costs.memory, default=0)
        return {
            "gated": bool(self.memory_admissions),
            "domains": domains,
            "tests": total,
            "known_tests": known,
            "largest": largest,
            **self._waits("memory"),
        }

    def _check_stealing(self, node: WorkerController) -> None:
        """Keep ``node`` two tests ahead: from a plan if it has one, else by stealing."""
        assert self.costs is not None
        queue = self.node2pending[node]
        if len(queue) >= MIN_PENDING:
            return
        indices = self._fill(node, len(self.pending) + 1)  # plans exist after crashes
        if indices:
            node.send_runtest_some(indices)
        if len(queue) >= MIN_PENDING or node in self.starving:
            return
        if node in self.waiting and not self._grant(node):
            return  # its head is not admitted: neither steal for it nor shut it down
        self.starving.append(node)
        self._steal_next()

    def _steal_next(self) -> None:
        """Request a chunk for the first waiting worker, one request in flight at a time."""
        if self.steal_requested_from_node is not None:
            return
        while self.starving:
            receiver = self.starving[0]
            if receiver.shutting_down or receiver not in self.node2pending:
                self.starving.pop(0)
                continue
            chosen = self._choose_steal(receiver)
            if chosen is None:
                self.starving.pop(0)
                self._shutdown(receiver)  # nothing worth taking: let it run its last test
                continue
            donor, chunk = chosen
            self.steal_requested_from_node = donor
            donor.send_steal(chunk)
            return

    def _choose_steal(
        self, receiver: WorkerController
    ) -> tuple[WorkerController, list[int]] | None:
        """The donor and tail chunk whose move gives the best predicted finish.

        Same candidates and same rule as moving unsent work: the tail run of one
        family, half of it, or its last test, and any move that does not lengthen the
        run is acceptable. The donor's running test, the one after it, and one more as
        a margin for reports still in flight are never asked for: the worker refuses
        a request unless it still holds every test in it.
        """
        assert self.costs is not None
        costs = self.costs
        self._refresh()
        states = {node: lane.project(costs) for node, lane in self.lanes.items()}
        finishes = {node: lane.free + states[node].seconds for node, lane in self.lanes.items()}
        budget = self._slots()
        areas = (
            {
                node: sum(d.work for d in self.dispatched[node].values()) + states[node].work
                for node in self.lanes
            }
            if budget
            else {}
        )
        origin = min(
            (
                lane.free - sum(c.seconds for c in self.dispatched[node].values())
                for node, lane in self.lanes.items()
            ),
            default=self.clock(),
        )
        before = projected_finish(finishes.values(), sum(areas.values()), budget, origin)
        best: tuple[float, float, WorkerController, list[int]] | None = None
        for donor, queue in self.node2pending.items():
            if donor is receiver or donor.shutting_down or len(queue) <= MIN_PENDING + 1:
                continue
            stealable = queue[MIN_PENDING + 1 :]
            tail_family = costs.family[stealable[-1]]
            run = 0
            for index in reversed(stealable):
                if costs.family[index] != tail_family:
                    break
                run += 1
            # The smallest chunk worth a round trip: a budget's worth of work, so a
            # balanced tail is not stolen one tiny test at a time.
            worth, seconds = 0, 0.0
            while worth < run and seconds < self.budget:
                seconds += costs.own[stealable[len(stealable) - 1 - worth]]
                worth += 1
            for count in sorted({worth, run // 2, run} - {0}):
                chunk = stealable[len(stealable) - count :]
                refund = sum(self.dispatched[donor][i].seconds for i in chunk)
                after_donor = finishes[donor] - refund
                added = costs.project(chunk, states[receiver])
                after_receiver = self.lanes[receiver].free + added.seconds
                rest = (f for n, f in finishes.items() if n not in (donor, receiver))
                total = sum(areas.values())
                if budget:
                    total -= sum(self.dispatched[donor][i].work for i in chunk)
                    total += added.work - states[receiver].work
                after = projected_finish(
                    [after_donor, after_receiver, *rest], total, budget, origin
                )
                # A chunk below the budget must shorten the run to be worth its round
                # trip; a bigger one is taken whenever it does not lengthen the run.
                # Best run first; then the pair itself as balanced as possible, so a
                # donor is not stripped bare only to steal everything back.
                acceptable = after < before - EPSILON or (
                    count >= worth and after <= before + EPSILON
                )
                rank = (after, max(after_donor, after_receiver))
                if acceptable and (best is None or rank < best[:2]):
                    best = (*rank, donor, chunk)
        if best is None:
            return None
        return best[2], best[3]

    def remove_pending_tests_from_node(self, node: WorkerController, indices: Any) -> None:
        """The donor answered a steal: it gave back all of ``indices`` or none of them.

        None means the donor had already started on the chunk (its completion reports
        were in flight when the request was made); the waiting worker stays waiting
        and a fresh choice is made right away, against the updated bookkeeping.
        Stolen tests go to the front of the taker's plan and out through the usual
        gate, so a taker never starts what it was not granted.
        """
        assert node is self.steal_requested_from_node
        assert self.costs is not None
        self.steal_requested_from_node = None
        stolen = [int(i) for i in indices]
        if stolen:
            taken = set(stolen)
            first = next(i for i in self.node2pending[node] if i in taken)
            checkpoint = self.dispatched[node][first].before
            self.node2pending[node] = [i for i in self.node2pending[node] if i not in taken]
            lane = self.lanes[node]
            for index in stolen:
                lane.free -= self.dispatched[node].pop(index).seconds
            lane.fixtures = set(checkpoint)
            self._reserve(node)
            self.pending.extend(stolen)  # back on the controller until sent again
            receiver = self.starving.pop(0) if self.starving else None
            if receiver is None or receiver.shutting_down or receiver not in self.node2pending:
                # Whoever waited is gone: keep the tests on the controller and plan them.
                self._replan(stolen)
            else:
                self.lanes[receiver].plan[0:0] = stolen
                # A single stolen test would leave it waiting for the one after: it
                # asks again, or is shut down so it can run what it has.
                self._check_stealing(receiver)
        self._steal_next()
        self._admit_waiting()
