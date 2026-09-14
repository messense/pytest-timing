"""Fixture-aware duration and CPU costs, planning, and queue transfers.

Pure computation, independent of xdist. ``Costs.charge`` owns each fixture
transition; ``Costs.project`` folds those charges for placement and balancing.
See ARCHITECTURE.md for the cost model and scheduling policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from statistics import fmean, median

from pytest_timing.demand import Declarations, key_base
from pytest_timing.model import Run

Family = frozenset[str]
NO_FIXTURES: Family = frozenset()
MIN_SETUP_SECONDS = 0.001
"""Fixtures cheaper than this are not worth steering tests around."""
EPSILON = 1e-9
SPLIT_GAIN = 0.01
"""A family is split over more workers only for at least this much predicted gain:
duplicated set-ups are real cost, and gains this small are within estimate noise."""


@cache
def _instance(key: str) -> tuple[str, str, str | None, dict[str, str]]:
    """(base, name, own parameter index, dependency parameter indices) of a key.

    The parameter part is ``<own index>`` and/or ``<dependency>=<index>`` entries,
    see ``_param_for`` in :mod:`pytest_timing.fixtures`. Cached: the planner
    replays keys many times over, and there are few distinct ones.
    """
    base = key_base(key)
    param = key[len(base) + 1 :]
    name = base.rpartition("::")[2]
    own: str | None = None
    deps: dict[str, str] = {}
    for part in param.rstrip("]").split(","):
        if not part:
            continue
        dep, eq, index = part.partition("=")
        if eq:
            deps[dep] = index
        else:
            own = part
    return base, name, own, deps


def pay(paid: set[str], family: Family) -> None:
    """Record fixture instances, replacing stale parameter instances.

    A parametrized fixture keeps one instance alive at a time: setting up one
    parameter tears the previous one down, so its sibling keys stop being paid, and
    so does every fixture built on the previous instance (pytest tears dependents
    down with what they depend on).
    """
    for key in family:
        base, name, own, _ = _instance(key)
        stale = []
        for other in paid:
            if other == key:
                continue
            other_base, _, _, other_deps = _instance(other)
            if other_base == base or (own is not None and other_deps.get(name, own) != own):
                stale.append(other)
        paid.difference_update(stale)
        paid.add(key)


@dataclass(slots=True)
class Estimates:
    durations: dict[str, float] = field(default_factory=dict)  # nodeid -> own seconds
    default: float = 0.0  # for nodeids without a recorded duration
    source: str = ""  # where the estimates came from, for the terminal summary
    families: dict[str, Family] = field(default_factory=dict)  # nodeid -> fixture keys
    setups: dict[str, float] = field(default_factory=dict)  # fixture key -> seconds
    memory: dict[str, int] = field(default_factory=dict)  # nodeid -> bytes it rose by
    retained: dict[str, int] = field(default_factory=dict)  # fixture key -> bytes kept

    @classmethod
    def from_run(cls, run: Run, source: str = "") -> Estimates:
        """Estimate each test's own time as the median of its most trustworthy attempts.

        Attempts are ranked clean before contended and passing before failing; only the
        best rank present is used. The median is robust to the odd slow attempt, and it
        does not grow with the number of attempts the way the longest one does, so a
        history merged from many runs stays comparable to a single run.

        Memory is the opposite: the largest rise any attempt showed, since the worst
        case is what an out-of-memory kill depends on. What an attempt kept resident
        after paying for shared set-ups is attributed to those fixtures, split evenly
        when it paid for several at once, again keeping the largest.
        """
        attempts: dict[str, dict[int, list[float]]] = {}  # nodeid -> rank -> own seconds
        families: dict[str, set[str]] = {}
        setups: dict[str, list[float]] = {}
        memory: dict[str, int] = {}
        retained: dict[str, int] = {}
        for test in run.tests:
            paid = []
            if test.fixtures:
                families.setdefault(test.nodeid, set()).update(test.fixtures)
                for key, seconds in test.fixtures.items():
                    if seconds:
                        setups.setdefault(key, []).append(seconds)
                        paid.append(key)
            if test.memory is not None:
                memory[test.nodeid] = max(memory.get(test.nodeid, 0), test.memory.rise)
                if paid and test.memory.retained:
                    share = test.memory.retained // len(paid)
                    for key in paid:
                        retained[key] = max(retained.get(key, 0), share)
            if test.outcome == "crashed":
                continue  # the recorded stop is the worker's death, not the test's
            own = test.duration - test.shared_setup
            contended = False
            if test.cpu is not None:
                own -= test.cpu.runtime_wait
                contended = test.cpu.contended
            if own <= 0:
                continue
            rank = (2 if contended else 0) + (1 if test.is_bad else 0)
            attempts.setdefault(test.nodeid, {}).setdefault(rank, []).append(own)
        durations = {nodeid: median(ranks[min(ranks)]) for nodeid, ranks in attempts.items()}
        default = fmean(durations.values()) if durations else 0.0
        return cls(
            durations,
            default,
            source,
            {nodeid: frozenset(keys) for nodeid, keys in families.items()},
            {key: median(seconds) for key, seconds in setups.items()},
            memory,
            retained,
        )

    @classmethod
    def load(cls, path: Path) -> Estimates:
        """Read a JSON run written by ``--timing-json`` (or ``pytest-timing merge``).

        Raises ``FileNotFoundError`` when there is no file and ``ValueError`` when the
        file is not a run.
        """
        text = path.read_text(encoding="utf-8")
        try:
            run = Run.from_json(text)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"not a pytest-timing run: {exc}") from exc
        return cls.from_run(run, str(path))

    def estimate(self, nodeid: str) -> float:
        return self.durations.get(nodeid, self.default)

    def family(self, nodeid: str) -> Family:
        return self.families.get(nodeid, NO_FIXTURES)

    def setup(self, key: str) -> float:
        return self.setups.get(key, 0.0)

    def known(self, nodeids: Sequence[str]) -> int:
        """How many of ``nodeids`` have a recorded duration."""
        return sum(1 for nodeid in nodeids if nodeid in self.durations)

    def rise(self, nodeid: str) -> int:
        """Bytes the test needed on top of its worker's footprint; unknown is zero."""
        return self.memory.get(nodeid, 0)

    def kept(self, key: str) -> int:
        """Bytes a shared fixture instance keeps resident while alive; unknown is zero."""
        return self.retained.get(key, 0)


class Costs:
    """The cost model over one collection: own durations, families, set-ups, slots,
    and memory.

    ``declarations`` (from the workers, see :mod:`pytest_timing.demand`) add the
    CPU side: slots per test, and per fixture definition the slots its set-up needs
    and the slots it holds while alive. A fixture with a declaration belongs to its
    tests' families even when no run has timed it yet; one a recorded run reports a
    test using without naming it (``getfixturevalue``) is matched by definition.

    Memory comes from history alone: the bytes each test rose by, and the bytes each
    shared fixture instance keeps resident. A fixture kept for its memory belongs to
    its tests' families like a declared one, so the planner keeps it alive on the
    lane that pays for it and the gate knows it is there.
    """

    def __init__(
        self,
        estimates: Estimates,
        collection: Sequence[str],
        declarations: Declarations | None = None,
    ) -> None:
        self.scope_prefixes = []
        for nodeid in collection:
            parts = nodeid.split("::")
            directories = parts[0].split("/")[:-1]
            self.scope_prefixes.append(
                tuple(
                    ["session:", "package::", f"module:{parts[0]}:"]
                    + [
                        "package:" + "/".join(directories[:i]) + ":"
                        for i in range(1, len(directories) + 1)
                    ]
                    + ["class:" + "::".join(parts[:i]) + ":" for i in range(2, len(parts) + 1)]
                )
            )
        self.own: list[float] = [estimates.estimate(nodeid) for nodeid in collection]
        decl = declarations or Declarations()
        self.declared = decl.any_demand
        self.function_holds = [decl.function_holds.get(i, 0) for i in range(len(collection))]
        self.slots: list[int] = [
            max(1, decl.tests[i]) if i < len(decl.tests) else 1 for i in range(len(collection))
        ]
        self.memory: list[int] = [estimates.rise(nodeid) for nodeid in collection]
        self.retained: dict[str, int] = {}  # fixture key -> bytes kept while alive
        self.family: list[Family] = []
        for index, nodeid in enumerate(collection):
            keys = [
                k
                for k in estimates.family(nodeid)
                if estimates.setup(k) >= MIN_SETUP_SECONDS or estimates.kept(k)
            ]
            keys += [k for k in decl.families.get(index, ()) if k not in keys]
            self.family.append(frozenset(keys) if keys else NO_FIXTURES)
            for key in keys:
                if estimates.kept(key):
                    self.retained[key] = estimates.kept(key)
        self.setup: Callable[[str], float] = estimates.setup
        self.shared = sorted({key for family in self.family for key in family})
        self.setup_slots: dict[str, int] = {}  # key base -> slots its set-up needs
        self.holds: dict[str, int] = {}  # key base -> slots held while alive
        for base in {key_base(key) for key in self.shared} | set(decl.fixtures):
            declared = decl.lookup(base)
            if declared is None:
                continue
            setup, hold = declared
            self.setup_slots[base] = setup
            if hold:
                self.holds[base] = hold

    def learn(self, key: str, setup: int, hold: int) -> None:
        """A fixture seen only at run time (``getfixturevalue``): its declaration."""
        base = key_base(key)
        self.setup_slots[base] = setup
        if hold:
            self.holds[base] = hold

    @property
    def any_demand(self) -> bool:
        """Does anything declare more than the one slot every test gets anyway?"""
        return (
            self.declared
            or any(s > 1 for s in self.slots)
            or any(s > 1 for s in self.setup_slots.values())
            or bool(self.holds)
        )

    def kept(self, index: int, fixtures: Iterable[str]) -> set[str]:
        """Instances still alive when this item starts, even if it does not use them.

        Scope owners are encoded before the definition in fixture keys. Matching
        the item's ancestor prefixes avoids parsing colon-bearing class node ids.
        An explicitly required key also supports history from custom collectors.
        """
        family = self.family[index]
        prefixes = self.scope_prefixes[index]
        return {key for key in fixtures if key in family or key.startswith(prefixes)}

    def hold(self, keys: Iterable[str]) -> int:
        return sum(self.holds.get(key_base(key), 0) for key in keys)

    def kept_memory(self, keys: Iterable[str]) -> int:
        """Bytes the fixture instances behind ``keys`` keep resident together."""
        return sum(self.retained.get(key, 0) for key in keys)

    @property
    def any_memory(self) -> bool:
        return any(self.memory) or bool(self.retained)

    def charge(self, index: int, fixtures: Iterable[str]) -> Charge:
        """One fixture transition, with its time, CPU work and peak reservation."""
        before = frozenset(fixtures)
        alive = self.kept(index, before)
        setup_seconds, peak = 0.0, self.slots[index]
        work = self.own[index] * peak
        for key in self.family[index]:
            if key in alive:
                continue
            setup, slots = self.setup(key), self.setup_slots.get(key_base(key), 1)
            setup_seconds += setup
            work += setup * slots
            peak = max(peak, slots)
        pay(alive, self.family[index])
        holds = self.hold(alive)
        return Charge(
            peak=peak + holds,
            holds=holds + self.function_holds[index],
            seconds=self.own[index] + setup_seconds,
            work=work,
            before=before,
            after=frozenset(alive),
            memory=self.memory[index] + self.kept_memory(alive),
        )

    def project(self, indices: Iterable[int], state: Projection | None = None) -> Projection:
        """Extend a projection without changing it or retaining intermediate charges."""
        state = state or Projection()
        seconds, work, fixtures = state.seconds, state.work, state.fixtures
        for index in indices:
            charge = self.charge(index, fixtures)
            seconds += charge.seconds
            work += charge.work
            fixtures = charge.after
        return Projection(seconds, work, fixtures)

    def cold(self, family: Family, members: Iterable[int]) -> float:
        return sum(self.own[i] for i in members) + sum(self.setup(k) for k in family)

    def area(self, family: Family, members: Iterable[int]) -> float:
        """Cold cost in slot-seconds: what the family takes out of the whole budget."""
        seconds = sum(self.own[i] * self.slots[i] for i in members)
        for key in family:
            seconds += self.setup(key) * self.setup_slots.get(key_base(key), 1)
        return seconds


@dataclass(frozen=True, slots=True)
class Charge:
    peak: int = 1
    holds: int = 0
    seconds: float = 0.0
    work: float = 0.0  # projected slot-seconds, refunded if this queued test is stolen
    before: Family = NO_FIXTURES  # fixture checkpoint before dispatch
    after: Family = NO_FIXTURES
    memory: int = 0  # bytes: the test's rise plus what the fixtures alive around it keep

    def with_live_holds(self, live: int) -> int:
        """Include unexpected live holds without counting declared holds twice."""
        return self.peak + max(0, live - self.holds)

    def adding_fixture(self, setup: int, hold: int, live: int) -> Charge:
        return replace(self, peak=max(self.peak + hold, setup + live), holds=self.holds + hold)


@dataclass(frozen=True, slots=True)
class Projection:
    seconds: float = 0.0
    work: float = 0.0
    fixtures: Family = NO_FIXTURES


@dataclass(slots=True)
class Lane:
    """A worker's projected finish time, fixture instances, and unsent plan."""

    free: float = 0.0
    fixtures: set[str] = field(default_factory=set)
    plan: list[int] = field(default_factory=list)

    def project(self, costs: Costs) -> Projection:
        return costs.project(self.plan, Projection(fixtures=frozenset(self.fixtures)))

    def finish(self, costs: Costs) -> float:
        return self.free + self.project(costs).seconds

    def without_tail(self, count: int) -> Lane:
        return Lane(self.free, self.fixtures, self.plan[: len(self.plan) - count])

    def dispatch(self, charge: Charge) -> None:
        self.fixtures = set(charge.after)
        self.free += charge.seconds


def _split(members: Sequence[int], parts: int, own: Sequence[float]) -> list[list[int]]:
    """Split members in their supplied order, balancing chunks by own duration."""
    bins: list[list[int]] = [[] for _ in range(parts)]
    loads = [0.0] * parts
    for index in members:
        lightest = min(range(parts), key=loads.__getitem__)
        bins[lightest].append(index)
        loads[lightest] += own[index]
    return bins


def plan(
    costs: Costs, indices: Iterable[int], lanes: Sequence[Lane], budget: int | None = None
) -> None:
    """Append ``indices`` to the lanes' plans, families whole, fixture-bound ones first.

    Each family is tried split over 1..n lanes. The split wins that promises the best
    run: the longer of the longest lane after the placement and the average lane once
    everything still unplaced is spread too, so an early family is not split just
    because the lanes are still empty; the fewer lanes the better on ties. With a CPU
    ``budget`` (slots) a third floor counts slot-seconds: a set-up that needs four
    slots, paid on every lane of a split, is four times the work, and no split can
    finish before the whole budget has done all of it. Finally single tests move off
    the tail of the longest lane while that shortens the run, and each lane's plan is
    ordered fixture-bound work first, then heaviest CPU demand first, so the cheapest
    tests sit at the tail, where other workers take from. Families are placed in
    order of slot-seconds, so the ones that take the most out of the CPU budget are
    placed while the lanes are emptiest.

    Lane free times are clock readings; the percentage tolerance is applied to time
    left from the earliest of them, never to the reading itself.
    """
    if not lanes:
        return
    by_family: dict[Family, list[int]] = {}
    for index in sorted(indices, key=lambda i: (-costs.slots[i], -costs.own[i])):
        by_family.setdefault(costs.family[index], []).append(index)
    ordered = sorted(by_family.items(), key=lambda kv: (not kv[0], -costs.area(kv[0], kv[1])))
    unplaced = sum(costs.cold(family, members) for family, members in ordered)
    unplaced_area = sum(costs.area(family, members) for family, members in ordered)
    origin = min(lane.free for lane in lanes)
    # Each lane's end-of-plan state, advanced as families are placed: nothing here
    # replays a plan from the start, so a family costs its own size per lane.
    states = [lane.project(costs) for lane in lanes]
    finishes = [lane.free + s.seconds - origin for lane, s in zip(lanes, states, strict=True)]
    slots = float(budget) if budget else 0.0
    for family, members in ordered:
        unplaced -= costs.cold(family, members)
        unplaced_area -= costs.area(family, members)
        best: tuple[float, float, list[list[int]], list[int]] | None = None
        for parts in range(1, min(len(lanes), len(members)) + 1):
            chunks = _split(members, parts, costs.own)
            # The lanes cheapest to run this family on, set-up owed included.
            ranked = sorted(
                range(len(lanes)),
                key=lambda j: finishes[j] + costs.charge(members[0], states[j].fixtures).seconds,
            )[:parts]
            ranked.sort(key=finishes.__getitem__)  # heaviest chunk to the lightest lane
            trial = list(finishes)
            area = sum(s.work for s in states)
            for j, chunk in zip(ranked, chunks, strict=True):
                after = costs.project(chunk, states[j])
                trial[j] = lanes[j].free + after.seconds - origin
                area += after.work - states[j].work
            predicted = max(max(trial), (sum(trial) + unplaced) / len(lanes))
            if slots:
                predicted = max(predicted, (area + unplaced_area) / slots)
            # Within the tolerance, a family with set-ups stays on fewer lanes; one
            # without is spread as evenly as possible, since that is free and leaves
            # less to rebalance when the estimates are off.
            spread = sum(f * f for f in trial) if not family else 0.0
            if best is None or predicted < best[0] * (1 - SPLIT_GAIN):
                best = (predicted, spread, chunks, ranked)
            elif not family and predicted <= best[0] * (1 + SPLIT_GAIN) and spread < best[1]:
                best = (predicted, spread, chunks, ranked)
        assert best is not None
        _, _, chunks, ranked = best
        for j, chunk in zip(ranked, chunks, strict=True):
            lanes[j].plan.extend(chunk)
            states[j] = costs.project(chunk, states[j])
            finishes[j] = lanes[j].free + states[j].seconds - origin
    _balance(costs, lanes, budget)
    for lane in lanes:
        # Heaviest first: a test waiting for slots should do so while the most work
        # is still left for the other workers, not at the tail.
        lane.plan.sort(key=lambda i: (not costs.family[i], -costs.slots[i], -costs.own[i]))
        lane.plan = _contiguous(costs, lane.plan)


def projected_finish(
    finishes: Iterable[float], area: float, budget: int | None, origin: float
) -> float:
    """One score for moving work: lane time bounded by total CPU work."""
    finish = max(finishes, default=origin)
    return max(finish, origin + area / budget) if budget else finish


def _contiguous(costs: Costs, order: list[int]) -> list[int]:
    """Stable grouping by family, families in order of first appearance."""
    groups: dict[Family, list[int]] = {}
    for index in order:
        groups.setdefault(costs.family[index], []).append(index)
    return [index for members in groups.values() for index in members]


def _balance(
    costs: Costs, lanes: Sequence[Lane], budget: int | None = None, limit: int = 2000
) -> None:
    """Move single tests from the tail of the longest lane while that clearly helps.

    With a budget, a move that adds a cold set-up is also judged by the slot-seconds
    floor, so the balance pass does not undo an area-aware split.
    """
    origin = min(lane.free for lane in lanes)
    slots = float(budget) if budget else 0.0
    states = [lane.project(costs) for lane in lanes]
    for _ in range(limit):
        finishes = [lane.free + s.seconds - origin for lane, s in zip(lanes, states, strict=True)]
        hi = max(range(len(lanes)), key=finishes.__getitem__)
        if not lanes[hi].plan:
            return
        index = lanes[hi].plan[-1]
        kept = lanes[hi].without_tail(1).project(costs)
        after_hi = lanes[hi].free + kept.seconds - origin
        best_lo, best_after = None, finishes[hi]
        total_work = sum(s.work for s in states)
        if slots:
            best_after = max(best_after, total_work / slots)
        best_state = None
        for lo in range(len(lanes)):
            if lo == hi:
                continue
            added = costs.project([index], states[lo])
            after_lo = lanes[lo].free + added.seconds - origin
            after = max(
                after_hi, after_lo, *(f for j, f in enumerate(finishes) if j not in (hi, lo))
            )
            if slots:
                total = total_work - states[hi].work + kept.work - states[lo].work + added.work
                after = max(after, total / slots)
            if after < best_after * (1 - SPLIT_GAIN):
                best_lo, best_after, best_state = lo, after, added
        if best_lo is None or best_state is None:
            return
        lanes[hi].plan.pop()
        lanes[best_lo].plan.append(index)
        states[hi], states[best_lo] = kept, best_state


def transfer(
    costs: Costs, lanes: Sequence[Lane], receiver: Lane, budget: int | None = None
) -> list[int]:
    """Move a tail chunk of another lane's plan to ``receiver`` unless that lengthens the run.

    Tries the whole tail run of one family, half of it, and its last test, from every
    other lane, and keeps the move with the best predicted finish, taking the most
    work on ties. A move that leaves the prediction unchanged is still taken: the
    receiver is idle, predictions are only estimates, and a worker running late looks
    to have little left, so waiting for a provable gain would leave workers idle
    exactly when the estimates are wrong. With a budget the slot-seconds floor applies
    too: a cold set-up that needs many slots is not paid again for a small gain.
    Returns the moved tests in run order (empty when every move would lengthen the run).
    """
    states = {id(lane): lane.project(costs) for lane in lanes}
    finishes = {id(lane): lane.free + states[id(lane)].seconds for lane in lanes}
    total_work = sum(s.work for s in states.values())
    origin = min((lane.free for lane in lanes), default=0.0)
    before = projected_finish(finishes.values(), total_work, budget, origin)
    best: tuple[float, float, Lane, int] | None = None
    for donor in lanes:
        if donor is receiver or not donor.plan:
            continue
        tail_family = costs.family[donor.plan[-1]]
        run = 0
        for index in reversed(donor.plan):
            if costs.family[index] != tail_family:
                break
            run += 1
        for count in sorted({1, run // 2, run} - {0}):
            chunk = donor.plan[len(donor.plan) - count :]
            kept = donor.without_tail(count).project(costs)
            added = costs.project(chunk, states[id(receiver)])
            after_donor = donor.free + kept.seconds
            after_receiver = receiver.free + added.seconds
            rest = (
                f for lane_id, f in finishes.items() if lane_id not in (id(donor), id(receiver))
            )
            total = total_work - states[id(donor)].work + kept.work
            total += added.work - states[id(receiver)].work
            after = projected_finish([after_donor, after_receiver, *rest], total, budget, origin)
            rank = (after, -after_receiver)
            if after <= before + EPSILON and (best is None or rank < best[:2]):
                best = (*rank, donor, count)
    if best is None:
        return []
    _, _, donor, count = best
    chunk = donor.plan[len(donor.plan) - count :]
    del donor.plan[len(donor.plan) - count :]
    receiver.plan.extend(chunk)
    return chunk
