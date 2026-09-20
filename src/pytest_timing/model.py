"""Data model for a recorded test run.

All times inside :class:`Worker` and :class:`TestSpan` are seconds relative to
``RunInfo.start`` (an epoch timestamp). The model is deliberately free of
pytest objects so renderers can be run out of process from saved JSON.

Semantics that every renderer must agree on live here, once:

* :data:`OUTCOME_REGISTRY` classifies outcomes (is it a failure?) and names them.
* :data:`TERMINATIONS` says how a session ended; ``complete`` is derived from it.
* Identity: a *worker* is a lane, a *test occurrence* is one collected item on a
  worker (duplicate selections are separate occurrences), and an *attempt* is one
  execution of an occurrence (retries add attempts).
* ``TestSpan.fixtures`` maps the shared (class / module / package / session) fixtures
  an attempt used to the seconds spent setting each one up for it, or ``None`` when it
  was already set up; see :mod:`pytest_timing.fixtures` for the key format.
* ``TestSpan.cpu`` is what the attempt cost in CPU terms (:class:`CpuRecord`): elapsed
  time, measured CPU work of the worker and its descendants, declared demand, how much
  of the process tree the measurement covered, and how long the controller held the
  test back for CPU slots. ``RunInfo.cpu`` describes the hosts and the budget.
* ``TestSpan.memory`` is the resident memory of the worker's process tree around the
  attempt (:class:`MemoryRecord`): before it, at its peak, and after it, in bytes.
  ``RunInfo.memory`` describes the memory budget and its admission, when one was set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from operator import attrgetter
from typing import Any

SCHEMA_VERSION = 1

PHASES: tuple[str, ...] = ("setup", "call", "teardown")


@dataclass(frozen=True, slots=True)
class OutcomeInfo:
    name: str
    label: str
    is_failure: bool

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "is_failure": self.is_failure}


OUTCOME_REGISTRY: dict[str, OutcomeInfo] = {
    o.name: o
    for o in (
        OutcomeInfo("passed", "passed", False),  # call phase passed
        OutcomeInfo("failed", "failed", True),  # call phase failed
        OutcomeInfo("error", "error", True),  # setup or teardown failed
        OutcomeInfo("skipped", "skipped", False),  # skipped during setup
        OutcomeInfo("xfailed", "xfailed", False),  # expected failure
        OutcomeInfo("xpassed", "xpassed", False),  # unexpectedly passed
        OutcomeInfo("crashed", "crashed", True),  # worker died while running it
        OutcomeInfo("rerun", "rerun", True),  # failed attempt that was retried
    )
}
BAD_OUTCOMES: frozenset[str] = frozenset(o.name for o in OUTCOME_REGISTRY.values() if o.is_failure)


@dataclass(frozen=True, slots=True)
class TerminationInfo:
    name: str
    label: str
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "complete": self.complete}


TERMINATION_REGISTRY: dict[str, TerminationInfo] = {
    t.name: t
    for t in (
        TerminationInfo("finished", "finished", True),  # the run loop completed
        TerminationInfo("collect_only", "collect only", True),  # nothing was meant to run
        # KeyboardInterrupt, pytest.exit(), -x / --maxfail, or an xdist stop
        TerminationInfo("interrupted", "interrupted", False),
        # xdist gave up early: crashed worker with restarts exhausted, collection mismatch
        TerminationInfo("aborted", "aborted", False),
        TerminationInfo("internal_error", "internal error", False),  # pytest raised
        TerminationInfo("unknown", "unknown", False),  # no evidence was recorded
    )
}
TERMINATIONS: tuple[str, ...] = tuple(TERMINATION_REGISTRY)
COMPLETE_TERMINATIONS: frozenset[str] = frozenset(
    t.name for t in TERMINATION_REGISTRY.values() if t.complete
)


def _round(value: float) -> float:
    return round(value, 6)


def _migrate_termination(run: dict[str, Any]) -> Any:
    """Termination for a run dict, migrating the early schema-1 shape.

    The first schema-1 reports carried only a ``complete`` flag. ``complete: true``
    can only have meant the run loop finished, so it maps to ``finished``; anything
    else is missing evidence and maps to ``unknown``. A present ``termination`` is
    returned untouched, valid or not, so bad values still fail validation.
    """
    if "termination" in run and run["termination"] is not None:
        return run["termination"]
    if run.get("complete") is True:
        return "finished"
    return "unknown"


def _opt_round(value: float | None) -> float | None:
    return None if value is None else _round(value)


def _opt_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return None if value is None else float(value)


def _opt_shift(value: float | None, seconds: float) -> float | None:
    return None if value is None else value + seconds


@dataclass(slots=True)
class Phase:
    """One runtest phase (setup / call / teardown) of an attempt."""

    start: float
    stop: float
    duration: float

    def to_list(self) -> list[float]:
        return [_round(self.start), _round(self.stop), _round(self.duration)]

    @classmethod
    def from_list(cls, data: list[float]) -> Phase:
        return cls(float(data[0]), float(data[1]), float(data[2]))

    def shifted(self, seconds: float) -> Phase:
        return Phase(self.start + seconds, self.stop + seconds, self.duration)


@dataclass(slots=True)
class CpuRecord:
    """CPU accounting of one attempt; see :mod:`pytest_timing.demand`.

    ``work`` and ``setup_work`` are CPU seconds; ``coverage`` says what the
    measurement saw (``tree``, ``reaped``, ``self`` or ``none``); ``pressure`` is the
    host's CPU stall share when the attempt ended and ``throttled`` whether a cgroup
    quota throttled it, both ``None`` when the platform does not tell.
    """

    elapsed: float = 0.0
    work: float = 0.0
    setup_work: float = 0.0  # inside shared fixture set-ups charged to this attempt
    demand: int = 1  # declared slots
    coverage: str = "none"
    pressure: float | None = None
    throttled: bool | None = None
    wait: float = 0.0  # total seconds held back by CPU admission
    runtime_wait: float = 0.0  # the part inside the test span; excluded from elapsed/work

    @property
    def contended(self) -> bool:
        """Was the host visibly short of CPU while this ran?"""
        return bool(self.throttled) or (
            self.pressure is not None and self.pressure >= CONTENDED_PRESSURE
        )

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "elapsed": _round(self.elapsed),
            "work": _round(self.work),
            "setup_work": _round(self.setup_work),
            "demand": self.demand,
            "coverage": self.coverage,
            "pressure": _opt_round(self.pressure),
            "throttled": self.throttled,
        }
        if self.wait:
            doc["wait"] = _round(self.wait)
        if self.runtime_wait:
            doc["runtime_wait"] = _round(self.runtime_wait)
        return doc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CpuRecord:
        throttled = data.get("throttled")
        return cls(
            elapsed=float(data.get("elapsed", 0.0)),
            work=float(data.get("work", 0.0)),
            setup_work=float(data.get("setup_work", 0.0)),
            demand=int(data.get("demand", 1)),
            coverage=str(data.get("coverage", "none")),
            pressure=_opt_float(data, "pressure"),
            throttled=None if throttled is None else bool(throttled),
            wait=float(data.get("wait", 0.0)),
            runtime_wait=float(data.get("runtime_wait", 0.0)),
        )


CONTENDED_PRESSURE = 0.25
"""PSI share above which an attempt is not a clean observation of its duration."""


@dataclass(slots=True)
class MemoryRecord:
    """Resident memory around one attempt, in bytes; see :mod:`pytest_timing.telemetry`.

    ``base`` is what the worker's process tree had resident when set-up began,
    ``peak`` the highest reading until the teardown report, ``after`` the reading at
    that report. ``coverage`` says whether descendants were counted (``tree``) or
    only the worker (``self``). A worker's heap rarely shrinks, so ``rise`` is
    what the attempt needed on top of what was already there, ``retained`` what
    stayed resident afterwards (a shared fixture it set up, or a leak), and
    ``transient`` the rest: what it needed beyond what stayed. ``wait`` is how long
    memory admission held the attempt back before it started, in seconds.
    """

    base: int = 0
    peak: int = 0
    after: int = 0
    coverage: str = "none"
    wait: float = 0.0

    @property
    def rise(self) -> int:
        return max(0, self.peak - self.base)

    @property
    def retained(self) -> int:
        return max(0, self.after - self.base)

    @property
    def transient(self) -> int:
        return max(0, self.peak - self.after)

    @property
    def measured(self) -> bool:
        """Did the platform give a reading? A live process is never resident at zero."""
        return self.peak > 0

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "base": self.base,
            "peak": self.peak,
            "after": self.after,
            "coverage": self.coverage,
        }
        if self.wait:
            doc["wait"] = _round(self.wait)
        return doc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        return cls(
            base=int(data.get("base", 0)),
            peak=int(data.get("peak", 0)),
            after=int(data.get("after", 0)),
            coverage=str(data.get("coverage", "none")),
            wait=float(data.get("wait", 0.0)),
        )


@dataclass(slots=True)
class WaitInterval:
    """A controller-observed admission interval, relative to the run's epoch."""

    start: float
    stop: float
    gates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"start": _round(self.start), "stop": _round(self.stop), "gates": list(self.gates)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitInterval:
        return cls(float(data["start"]), float(data["stop"]), tuple(data.get("gates", ())))


@dataclass(slots=True)
class TestSpan:
    """One attempt at running one test occurrence on one worker.

    ``occurrence`` numbers the collected items that share a nodeid on a worker
    (``--keep-duplicates``, ``--dist each`` across workers keeps 0). ``attempt``
    numbers the executions of that occurrence (retries). Reports only carry the
    nodeid, so the collector infers both from report order; see ``Collector``.
    """

    nodeid: str
    worker: str
    attempt: int
    outcome: str
    start: float
    stop: float
    phases: dict[str, Phase] = field(default_factory=dict)
    occurrence: int = 0
    fixtures: dict[str, float | None] = field(default_factory=dict)
    cpu: CpuRecord | None = None
    memory: MemoryRecord | None = None
    admission_waits: list[WaitInterval] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.stop - self.start

    @property
    def shared_setup(self) -> float:
        """Seconds of shared fixture set-up charged to this attempt."""
        return sum(v for v in self.fixtures.values() if v)

    @property
    def is_bad(self) -> bool:
        return self.outcome in BAD_OUTCOMES

    def shifted(self, seconds: float) -> TestSpan:
        return replace(
            self,
            start=self.start + seconds,
            stop=self.stop + seconds,
            phases={k: p.shifted(seconds) for k, p in self.phases.items()},
            admission_waits=[
                WaitInterval(w.start + seconds, w.stop + seconds, w.gates)
                for w in self.admission_waits
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "nodeid": self.nodeid,
            "worker": self.worker,
            "occurrence": self.occurrence,
            "attempt": self.attempt,
            "outcome": self.outcome,
            "start": _round(self.start),
            "stop": _round(self.stop),
            "phases": {name: phase.to_list() for name, phase in self.phases.items()},
        }
        if self.fixtures:
            doc["fixtures"] = {key: _opt_round(seconds) for key, seconds in self.fixtures.items()}
        if self.cpu is not None:
            doc["cpu"] = self.cpu.to_dict()
        if self.memory is not None:
            doc["memory"] = self.memory.to_dict()
        if self.admission_waits:
            doc["admission_waits"] = [w.to_dict() for w in self.admission_waits]
        return doc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestSpan:
        fixtures = data.get("fixtures") or {}
        cpu = data.get("cpu")
        memory = data.get("memory")
        return cls(
            nodeid=str(data["nodeid"]),
            worker=str(data["worker"]),
            occurrence=int(data.get("occurrence", 0)),
            attempt=int(data.get("attempt", 0)),
            outcome=str(data["outcome"]),
            start=float(data["start"]),
            stop=float(data["stop"]),
            phases={
                str(name): Phase.from_list(value)
                for name, value in dict(data.get("phases", {})).items()
            },
            fixtures={
                str(key): None if seconds is None else float(seconds)
                for key, seconds in dict(fixtures).items()
            },
            cpu=CpuRecord.from_dict(dict(cpu)) if isinstance(cpu, dict) else None,
            memory=MemoryRecord.from_dict(dict(memory)) if isinstance(memory, dict) else None,
            admission_waits=[WaitInterval.from_dict(w) for w in data.get("admission_waits", ())],
        )


@dataclass(slots=True)
class Worker:
    """Lifecycle of one xdist worker (or the single ``main`` lane without xdist).

    ``id`` is the worker's identity; merging may relabel it, but that is the merge
    operation's job (see ``cli.merge_runs``), never a renderer's.
    """

    id: str
    start: float | None = None  # when the controller launched it; None = not recorded
    ready: float | None = None
    collected: float | None = None
    items: int | None = None
    down: float | None = None
    error: str | None = None

    def shifted(self, seconds: float) -> Worker:
        return replace(
            self,
            start=_opt_shift(self.start, seconds),
            ready=_opt_shift(self.ready, seconds),
            collected=_opt_shift(self.collected, seconds),
            down=_opt_shift(self.down, seconds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": _opt_round(self.start),
            "ready": _opt_round(self.ready),
            "collected": _opt_round(self.collected),
            "items": self.items,
            "down": _opt_round(self.down),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Worker:
        items = data.get("items")
        return cls(
            id=str(data["id"]),
            start=_opt_float(data, "start"),
            ready=_opt_float(data, "ready"),
            collected=_opt_float(data, "collected"),
            items=None if items is None else int(items),
            down=_opt_float(data, "down"),
            error=None if data.get("error") is None else str(data["error"]),
        )


@dataclass(slots=True)
class RunInfo:
    """Metadata about the whole session, including how it ended."""

    start: float
    stop: float
    termination: str = "unknown"
    reason: str | None = None
    exit_status: int | None = None
    argv: list[str] = field(default_factory=list)
    rootdir: str = ""
    python: str = ""
    pytest: str = ""
    xdist: str | None = None
    dist: str | None = None
    numprocesses: int | None = None
    cpu: dict[str, Any] | None = None  # hosts, budget and admission summary
    memory: dict[str, Any] | None = None  # the memory budget and its admission summary

    @property
    def wall(self) -> float:
        return self.stop - self.start

    @property
    def complete(self) -> bool:
        """Derived from the recorded termination, never from test counts."""
        return self.termination in COMPLETE_TERMINATIONS

    @property
    def termination_label(self) -> str:
        return TERMINATION_REGISTRY[self.termination].label

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "stop": self.stop,
            "termination": self.termination,
            "reason": self.reason,
            "exit_status": self.exit_status,
            "complete": self.complete,
            "argv": list(self.argv),
            "rootdir": self.rootdir,
            "python": self.python,
            "pytest": self.pytest,
            "xdist": self.xdist,
            "dist": self.dist,
            "numprocesses": self.numprocesses,
            "cpu": self.cpu,
            "memory": self.memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunInfo:
        numprocesses = data.get("numprocesses")
        exit_status = data.get("exit_status")
        termination = _migrate_termination(data)
        if termination not in TERMINATION_REGISTRY:
            raise ValueError(
                f"run has no valid termination (got {termination!r}); "
                f"expected one of {', '.join(TERMINATIONS)}"
            )
        return cls(
            start=float(data["start"]),
            stop=float(data["stop"]),
            termination=str(termination),
            reason=None if data.get("reason") is None else str(data["reason"]),
            exit_status=None if exit_status is None else int(exit_status),
            argv=[str(a) for a in data.get("argv", [])],
            rootdir=str(data.get("rootdir", "")),
            python=str(data.get("python", "")),
            pytest=str(data.get("pytest", "")),
            xdist=None if data.get("xdist") is None else str(data["xdist"]),
            dist=None if data.get("dist") is None else str(data["dist"]),
            numprocesses=None if numprocesses is None else int(numprocesses),
            cpu=dict(data["cpu"]) if isinstance(data.get("cpu"), dict) else None,
            memory=dict(data["memory"]) if isinstance(data.get("memory"), dict) else None,
        )


@dataclass(slots=True)
class Lane:
    """One row of the timeline: a worker's window and its spans, sorted by start.

    ``ready``/``down`` fall back to the first/last span when the worker lifecycle was
    not recorded; ``boot`` and ``collect`` are only present when it was.
    """

    id: str
    worker: Worker | None
    spans: list[TestSpan]
    ready: float
    down: float
    last: float | None  # when the last span finished

    @property
    def busy(self) -> float:
        return sum(t.duration for t in self.spans)

    @property
    def boot(self) -> tuple[float, float] | None:
        w = self.worker
        if w is None or w.start is None or w.ready is None:
            return None
        return w.start, w.ready

    @property
    def collect(self) -> tuple[float, float] | None:
        w = self.worker
        if w is None or w.ready is None or w.collected is None:
            return None
        return w.ready, w.collected


_SPAN_ORDER = attrgetter("start", "stop")


def _utilisation(lanes: list[Lane]) -> float:
    available = sum(max(0.0, lane.down - lane.ready) for lane in lanes)
    return sum(lane.busy for lane in lanes) / available if available > 0 else 0.0


@dataclass(slots=True)
class Run:
    """A complete recorded run: session metadata, worker lanes, and test spans."""

    run: RunInfo
    workers: list[Worker] = field(default_factory=list)
    tests: list[TestSpan] = field(default_factory=list)

    @property
    def wall(self) -> float:
        """Total timeline length in seconds (never below the last recorded event)."""
        end = self.run.wall
        for worker in self.workers:
            for value in (worker.start, worker.ready, worker.collected, worker.down):
                if value is not None:
                    end = max(end, value)
        for test in self.tests:
            end = max(end, test.stop)
        return end

    def worker_ids(self) -> list[str]:
        """Lane ids: recorded workers first, then any worker only seen in tests."""
        ids = dict.fromkeys(w.id for w in self.workers)
        ids.update(dict.fromkeys(t.worker for t in self.tests))
        return list(ids)

    def lanes(self) -> list[Lane]:
        """Lane geometry, derived once for every renderer (see :class:`Lane`)."""
        spans_by_id: dict[str, list[TestSpan]] = {wid: [] for wid in self.worker_ids()}
        for test in self.tests:
            spans_by_id[test.worker].append(test)
        workers = {w.id: w for w in self.workers}
        wall = self.wall
        lanes = []
        for wid, spans in spans_by_id.items():
            spans.sort(key=_SPAN_ORDER)
            worker = workers.get(wid)
            last = max((t.stop for t in spans), default=None)
            ready = worker.ready if worker else None
            down = worker.down if worker else None
            lanes.append(
                Lane(
                    id=wid,
                    worker=worker,
                    spans=spans,
                    ready=ready if ready is not None else (spans[0].start if spans else 0.0),
                    down=down if down is not None else (last if last is not None else wall),
                    last=last,
                )
            )
        return lanes

    def tests_by_worker(self) -> dict[str, list[TestSpan]]:
        return {lane.id: lane.spans for lane in self.lanes()}

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for test in self.tests:
            counts[test.outcome] = counts.get(test.outcome, 0) + 1
        return counts

    def busy_seconds(self) -> float:
        return sum(t.duration for t in self.tests)

    def lane_window(self, worker_id: str) -> tuple[float, float]:
        """(ready, down) for a lane, falling back to the first/last test."""
        lane = next(lane for lane in self.lanes() if lane.id == worker_id)
        return lane.ready, lane.down

    def utilisation(self) -> float:
        """Busy seconds over the sum of every lane's (ready .. down) window."""
        return _utilisation(self.lanes())

    def summary(self) -> dict[str, Any]:
        """Derived figures every renderer shows; embedded so the HTML never recomputes."""
        lanes = self.lanes()
        return {
            "wall": _round(self.wall),
            "busy": _round(self.busy_seconds()),
            "utilisation": round(_utilisation(lanes), 4),
            "lanes": [
                {
                    "id": lane.id,
                    "ready": _round(lane.ready),
                    "down": _round(lane.down),
                    "busy": _round(lane.busy),
                    "last": _opt_round(lane.last),
                }
                for lane in lanes
            ],
        }

    def rebased(self, base_epoch: float) -> Run:
        """The same run expressed relative to ``base_epoch``.

        Every relative time (worker lifecycle, tests, phases) moves together, so
        durations and the gaps between events are preserved exactly.
        """
        seconds = self.run.start - base_epoch
        info = replace(self.run, start=base_epoch)
        return Run(
            run=info,
            workers=[w.shifted(seconds) for w in self.workers],
            tests=[t.shifted(seconds) for t in self.tests],
        )

    def relabelled(self, mapping: dict[str, str]) -> Run:
        """Rename workers; tests follow their worker. Missing keys keep their id."""
        return Run(
            run=replace(self.run),
            workers=[replace(w, id=mapping.get(w.id, w.id)) for w in self.workers],
            tests=[replace(t, worker=mapping.get(t.worker, t.worker)) for t in self.tests],
        )

    def to_dict(self) -> dict[str, Any]:
        from pytest_timing import __version__

        return {
            "schema": SCHEMA_VERSION,
            "pytest_timing_version": __version__,
            "outcomes": {name: info.to_dict() for name, info in OUTCOME_REGISTRY.items()},
            "terminations": {name: info.to_dict() for name, info in TERMINATION_REGISTRY.items()},
            "summary": self.summary(),
            "run": self.run.to_dict(),
            "workers": [w.to_dict() for w in self.workers],
            "tests": [t.to_dict() for t in self.tests],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        schema = int(data.get("schema", 0))
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported pytest-timing schema {schema}, expected {SCHEMA_VERSION}"
            )
        return cls(
            run=RunInfo.from_dict(data["run"]),
            workers=[Worker.from_dict(w) for w in data.get("workers", [])],
            tests=[TestSpan.from_dict(t) for t in data.get("tests", [])],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        separators = (",", ":") if indent is None else None
        return json.dumps(self.to_dict(), indent=indent, separators=separators)

    @classmethod
    def from_json(cls, text: str) -> Run:
        return cls.from_dict(json.loads(text))
