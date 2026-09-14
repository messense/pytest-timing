"""CPU declarations and per-test CPU and memory measurement where the tests run.

Declarations precede xdist collection reports. Dynamic fixture setup requests
reserve slots before execution; holds follow actual setup/finalization events.
Platform measurement lives in telemetry.py; protocol details in ARCHITECTURE.md."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from pytest_timing import xdist_compat
from pytest_timing.fixtures import FixtureTimer, _key_for, defined_at, fixture_key, fixturedefs
from pytest_timing.telemetry import MemorySampler, Pressure, ProcessTreeClock, host_cpu

MARKER = "timing_cpu"
FIXTURE_ATTR = "_pytest_timing_cpu"
REPORT_ATTR = "timing_cpu"
MEMORY_ATTR = "timing_memory"
"""Report attribute carrying the attempt's resident-memory window, on teardown."""
EXECUTION_ATTR = "timing_execution"
"""Private report identity: collection index and attempt, including retries."""
EVENT = "timing_cpu"
"""Name of the worker-to-controller event that carries the declarations."""
CANCEL = xdist_compat.CANCEL
"""Name of the controller-to-worker command that withdraws tests a worker holds."""
REQUEST_TIMEOUT = 300.0
"""Seconds before cancelling a run-time setup request. Failure unwinds only after
recovering the test's reservation; a timeout never authorizes unreserved execution."""


@dataclass(frozen=True, slots=True)
class Demand:
    setup: int  # slots while a fixture is set up, or a test runs
    hold: int = 0  # slots while a fixture instance stays alive


def _slots(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} expects a positive integer number of CPU slots, got {value!r}")
    if value < 1:
        raise ValueError(f"{what} expects at least one CPU slot, got {value}")
    return value


def _hold(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{what} hold= expects a non-negative integer, got {value!r}")
    return value


def cpu(setup: int, *, hold: int = 0) -> Callable[[Any], Any]:
    """Declare a fixture's CPU demand; works above or below ``@pytest.fixture``.

    ``setup`` slots are reserved while the fixture is set up, ``hold`` slots for as
    long as the instance is alive. Applied to a test function it means the same as
    ``@pytest.mark.timing_cpu(setup)``, but a marker on the test wins.
    """
    demand = Demand(_slots(setup, "pytest_timing.cpu"), _hold(hold, "pytest_timing.cpu"))

    def decorate(obj: Any) -> Any:
        # Applied above ``@pytest.fixture``, ``obj`` is what pytest made of the
        # function, and the fixture definition later points at the plain function
        # inside it: a definition object with ``_get_wrapped_function`` (pytest >=
        # 8.4) or a wrapper carrying ``__pytest_wrapped__`` (older). Mark the plain
        # function too, so either decorator order works. No pytest mark is added: a
        # mark on a function that then becomes a fixture is a collection error, and
        # ``item_demand`` reads the attribute anyway.
        wrapped = getattr(obj, "_get_wrapped_function", None)
        if callable(wrapped):
            setattr(wrapped(), FIXTURE_ATTR, demand)
        inner = getattr(getattr(obj, "__pytest_wrapped__", None), "obj", None)
        if inner is not None:
            setattr(inner, FIXTURE_ATTR, demand)
        setattr(obj, FIXTURE_ATTR, demand)
        return obj

    return decorate


def fixture_demand(fixturedef: Any) -> Demand | None:
    func = getattr(fixturedef, "func", None)
    demand = getattr(func, FIXTURE_ATTR, None)
    if demand is None:  # a wrapper pytest left around the function
        inner = getattr(getattr(func, "__pytest_wrapped__", None), "obj", None)
        demand = getattr(inner, FIXTURE_ATTR, None)
    return demand if isinstance(demand, Demand) else None


def definition_key(fixturedef: Any) -> str:
    """``<scope>:<defined at>::<name>``: a fixture definition, independent of any item."""
    scope = str(getattr(fixturedef, "scope", "function"))
    return f"{scope}:{defined_at(fixturedef)}::{fixturedef.argname}"


def item_demand(item: pytest.Item) -> tuple[int, str | None]:
    """(slots, error): the closest ``timing_cpu`` marker's weight, or 1.

    An invalid marker counts as one slot and comes back with the message the test
    should fail with, so a typo never silently changes the schedule.
    """
    marker = item.get_closest_marker(MARKER)
    if marker is None:
        demand = getattr(getattr(item, "obj", None), FIXTURE_ATTR, None)
        return (demand.setup if isinstance(demand, Demand) else 1), None
    if marker.args and "slots" in marker.kwargs or len(marker.args) > 1:
        return 1, f"{MARKER} takes one argument, the number of CPU slots"
    value = marker.args[0] if marker.args else marker.kwargs.get("slots")
    if value is None:
        return 1, f"{MARKER} needs the number of CPU slots, as in timing_cpu(4)"
    try:
        return _slots(value, MARKER), None
    except (TypeError, ValueError) as exc:
        return 1, str(exc)


def key_base(key: str) -> str:
    """A fixture key without its parameter index: the definition it points at."""
    return key.rpartition("[")[0]


@dataclass(slots=True)
class Declarations:
    """What one collection declares, in collection order, as the controller needs it.

    ``tests`` is the slots per item: its marker, raised to what a function-scoped
    fixture of its declares, since that set-up runs inside the test's own reservation.
    ``fixtures`` maps the shared fixtures the items are known to use (key base ->
    setup, hold); ``definitions`` maps every declared fixture definition the worker
    knows of (``<scope>:<defined at>::<name>``), so a fixture pulled in dynamically
    with ``getfixturevalue``, which no item's signature names but a recorded run
    reports, still has its demand.
    """

    tests: list[int] = field(default_factory=list)  # slots per item
    function_holds: dict[int, int] = field(default_factory=dict)  # already included in tests
    families: dict[int, list[str]] = field(default_factory=dict)  # item -> declared fixtures
    fixtures: dict[str, tuple[int, int]] = field(default_factory=dict)  # key base -> setup, hold
    definitions: dict[str, tuple[int, int]] = field(default_factory=dict)  # by definition
    errors: dict[int, str] = field(default_factory=dict)  # item -> invalid marker message
    host: dict[str, Any] = field(default_factory=dict)  # the worker's detected CPU environment

    @property
    def any_demand(self) -> bool:
        return any(s > 1 for s in self.tests) or any(
            setup > 1 or hold > 0
            for setup, hold in (*self.fixtures.values(), *self.definitions.values())
        )

    @classmethod
    def from_items(
        cls, items: list[pytest.Item], host: dict[str, Any], fixturemanager: Any = None
    ) -> Declarations:
        decl = cls(host=dict(host))
        for index, item in enumerate(items):
            slots, error = item_demand(item)
            if error:
                decl.errors[index] = error
            keys: list[str] = []
            inner: list[Demand] = []  # function-scoped fixtures: inside the test
            for fixturedef in fixturedefs(item).values():
                demand = fixture_demand(fixturedef)
                if demand is None:
                    continue
                key = _key_for(item, fixturedef)
                if key is None:
                    inner.append(demand)
                    continue
                keys.append(key)
                decl.fixtures[key_base(key)] = (demand.setup, demand.hold)
            # The test's stages: each function-scoped set-up while the others may
            # already be alive and holding, then the call with all of them alive.
            holds = sum(d.hold for d in inner)
            if holds:
                decl.function_holds[index] = holds
            stages = [d.setup + holds - d.hold for d in inner] + [slots + holds]
            decl.tests.append(max(stages))
            if keys:
                decl.families[index] = keys
        arg2fixturedefs: dict[str, Any] = getattr(fixturemanager, "_arg2fixturedefs", None) or {}
        for defs in arg2fixturedefs.values():
            for fixturedef in defs or ():
                demand = fixture_demand(fixturedef)
                if demand is not None:
                    decl.definitions[definition_key(fixturedef)] = (demand.setup, demand.hold)
        return decl

    def lookup(self, base: str) -> tuple[int, int] | None:
        """(setup, hold) declared for the fixture behind key base ``base``, if any."""
        found = self.fixtures.get(base)
        if found is not None or not self.definitions:
            return found
        # ``<scope>:<scope node id>:<defined at>::<name>`` against ``<scope>:<defined
        # at>::<name>``: the node id is anything, the rest must match exactly.
        scope, _, rest = base.partition(":")
        head, _, name = rest.rpartition("::")
        best: tuple[int, tuple[int, int]] | None = None
        for definition, value in self.definitions.items():
            d_scope, _, d_rest = definition.partition(":")
            defined_at, _, d_name = d_rest.rpartition("::")
            if d_scope != scope or d_name != name or not head.endswith(":" + defined_at):
                continue
            if best is None or len(defined_at) > best[0]:
                best = (len(defined_at), value)
        return best[1] if best is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tests": list(self.tests),
            "function_holds": {str(i): n for i, n in self.function_holds.items()},
            "families": {str(i): list(keys) for i, keys in self.families.items()},
            "fixtures": {key: list(value) for key, value in self.fixtures.items()},
            "definitions": {key: list(value) for key, value in self.definitions.items()},
            "errors": {str(i): msg for i, msg in self.errors.items()},
            "host": dict(self.host),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Declarations:
        def pairs(name: str) -> dict[str, tuple[int, int]]:
            return {str(k): (int(v[0]), int(v[1])) for k, v in dict(data.get(name, {})).items()}

        return cls(
            tests=[int(s) for s in data.get("tests", [])],
            function_holds={int(i): int(n) for i, n in data.get("function_holds", {}).items()},
            families={
                int(i): [str(k) for k in keys] for i, keys in data.get("families", {}).items()
            },
            fixtures=pairs("fixtures"),
            definitions=pairs("definitions"),
            errors={int(i): str(m) for i, m in dict(data.get("errors", {})).items()},
            host=dict(data.get("host", {})),
        )


class CpuMeter:
    """Measures each test's CPU work and memory and attaches them to the teardown report.

    Runs wherever tests run: in every xdist worker, or in the main process without
    xdist. After collection it also sends the :class:`Declarations` to the controller
    (``send``), before xdist reports the collection, so the scheduler has them when
    it lays out the run.

    It also takes the controller's ``timing_cancel`` command (see
    :mod:`pytest_timing.xdist_compat`): the tests it names are held by the worker
    but were never admitted, and when xdist shuts the worker down early they are
    skipped rather than run without slots.

    A fixture with a declaration that a test reaches only at run time
    (``getfixturevalue``) was not part of the test's reservation. Where the worker
    can take commands while a test runs, the meter asks the controller for the
    set-up's slots (``timing_request``) before the set-up starts and blocks until
    they are granted (``timing_grant``), so such a set-up goes through the same
    gate as a declared one.

    ``memory`` samples resident memory from set-up to the teardown report; the
    window (``timing_memory``) goes on that report next to the CPU record.
    """

    def __init__(
        self,
        timer: FixtureTimer,
        send: Callable[[str, dict[str, Any]], None] | None = None,
        clock: ProcessTreeClock | None = None,
        pressure: Pressure | None = None,
        memory: MemorySampler | None = None,
    ) -> None:
        self.timer = timer
        self.send = send
        self.clock = clock or ProcessTreeClock()
        self.pressure = pressure or Pressure()
        self.memory = memory
        self.declarations: Declarations | None = None
        self.cancelled: set[int] = set()  # item indices withdrawn by the controller
        self._live = False  # can the controller reach this worker while a test runs
        self._request: dict[str, Any] | None = None
        self._request_id = 0
        self._granted = False
        self._condition = threading.Condition()
        self._index: dict[int, int] = {}  # id(item) -> collection index
        self._fixture_holds: dict[int, int] = {}  # live fixture definitions -> held slots
        self._attempts: dict[int, int] = {}  # collection index -> current attempt
        self._started = 0.0
        self._work = 0.0
        self._throttled: int | None = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        items = session.items
        manager = getattr(session, "_fixturemanager", None)
        host = host_cpu(self.pressure).to_dict()
        self.declarations = Declarations.from_items(items, host, manager)
        self._index = {id(item): i for i, item in enumerate(items)}
        if self.send is not None:
            self.send(EVENT, self.declarations.to_dict())

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session: pytest.Session) -> None:
        if self.send is not None:  # in a worker: take the controller's commands
            xdist_compat.intercept_commands(session.config, CANCEL, self.cancel)
            self._live = xdist_compat.intercept_commands(
                session.config, xdist_compat.GRANT, self.grant
            )

    def cancel(self, payload: dict[str, Any]) -> None:
        """Runs on the channel's receiver thread: only record what was withdrawn."""
        self.cancelled.update(int(i) for i in payload.get("indices", ()))

    def grant(self, payload: dict[str, Any]) -> None:
        """Runs on the channel's receiver thread: wake the set-up waiting for this."""
        with self._condition:
            if self._request is not None and payload.get("request_id") == self._request_id:
                self._granted = True
                self._condition.notify_all()

    def _dynamic_request(self, item: pytest.Item | None, fixturedef: Any) -> str | None:
        """Ask the controller for the slots of a declared fixture the running test
        did not reserve for; the key to wait on, or ``None`` when nothing is owed."""
        if not self._live or self.send is None or item is None:
            return None
        demand = fixture_demand(fixturedef)
        if demand is None or (demand.setup <= 1 and demand.hold == 0):
            return None
        index = self._index.get(id(item))
        if index is None:
            return None
        name = str(fixturedef.argname)
        if fixturedefs(item).get(name) is fixturedef:
            return None  # named by the test: reserved with it
        key = _key_for(item, fixturedef) or fixture_key(
            "function", item.nodeid, defined_at(fixturedef), name, ""
        )
        with self._condition:
            self._request_id += 1
            self._granted = False
            self._request = {
                "request_id": self._request_id,
                "index": index,
                "key": key,
                "setup": demand.setup,
                "hold": demand.hold,
                "holds": sum(self._fixture_holds.values()),
                "attempt": self._attempts.get(index, 0),
            }
        self.send(xdist_compat.REQUEST, self._request)
        return key

    def _await(self, key: str) -> None:
        deadline = time.monotonic() + REQUEST_TIMEOUT
        started = time.perf_counter()
        work = self.clock.seconds()
        timed_out = False
        try:
            with self._condition:
                while not self._granted:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 and not timed_out:
                        timed_out = True
                        assert self.send is not None and self._request is not None
                        # Cancel the setup, then reacquire the original test's
                        # reservation before unwinding fixtures or starting a retry.
                        self.send(xdist_compat.REQUEST, {**self._request, "cancelled": True})
                    self._condition.wait(None if timed_out else remaining)
            if timed_out:
                pytest.fail(
                    f"pytest-timing: timed out waiting for CPU slots for {key}", pytrace=False
                )
        finally:
            with self._condition:
                self._request = None
                self._granted = False
            self.timer.record_wait(
                time.perf_counter() - started, max(0.0, self.clock.seconds() - work)
            )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_fixture_setup(self, fixturedef: Any, request: Any) -> Generator[None, Any, None]:
        key = self._dynamic_request(self.timer._item, fixturedef)
        if key is not None:
            try:
                self._await(key)  # before the timer inside starts counting the set-up
            except pytest.fail.Exception:
                # execute() may already have installed finalizers before calling
                # this hook. No fixture body ran to cache a result or a failure;
                # finish that incomplete instance so a retry can execute it.
                # Recent pytest versions make finish() a no-op without a cache
                # entry. A provisional empty result lets pytest unwind its own
                # finalizers, without depending on version-specific error tuples.
                fixturedef.cached_result = (None, fixturedef.cache_key(request), None)
                fixturedef.finish(request)
                raise
        outcome = yield
        demand = fixture_demand(fixturedef)
        if outcome.excinfo is None and demand is not None and demand.hold:
            self._fixture_holds[id(fixturedef)] = demand.hold
            self._send_holds()

    def pytest_fixture_post_finalizer(self, fixturedef: Any, request: Any) -> None:
        if self._fixture_holds.pop(id(fixturedef), None) is not None:
            self._send_holds()

    def _send_holds(self) -> None:
        if self.send is not None:
            self.send(xdist_compat.HOLDS, {"holds": sum(self._fixture_holds.values())})

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> Any:
        """A withdrawn test is not run at all: xdist's worker would otherwise run
        it on shutdown, and the controller has no slots reserved for it."""
        if self.cancelled and self._index.get(id(item)) in self.cancelled:
            return True
        return None

    def _declared(self, item: pytest.Item) -> tuple[int, str | None]:
        """(slots, invalid-marker message) declared for ``item`` at collection."""
        index = self._index.get(id(item))
        if index is None or self.declarations is None:
            return item_demand(item)
        return self.declarations.tests[index], self.declarations.errors.get(index)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        index = self._index.get(id(item))
        if index is not None:
            self._attempts[index] = self._attempts.get(index, -1) + 1
        message = self._declared(item)[1]
        if message:
            pytest.fail(f"pytest-timing: {message}", pytrace=False)
        self._started = time.perf_counter()
        self._work = self.clock.seconds()
        self._throttled = self.pressure.throttled()
        if self.memory is not None:
            self.memory.begin()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: Any) -> Generator[None, Any, None]:
        outcome = yield
        index = self._index.get(id(item))
        if index is not None:
            setattr(outcome.get_result(), EXECUTION_ATTR, (index, self._attempts.get(index, 0)))
        if call.when != "teardown" or not self._started:
            return
        window = self.memory.end() if self.memory is not None else None
        if window is not None:
            setattr(
                outcome.get_result(),
                MEMORY_ATTR,
                {
                    "base": window.base,
                    "peak": window.peak,
                    "after": window.after,
                    "coverage": window.coverage,
                },
            )
        elapsed = max(0.0, time.perf_counter() - self._started - self.timer.wait)
        work = self.clock.seconds() - self._work - self.timer.wait_work
        throttled_now = self.pressure.throttled()
        throttled = (
            None
            if throttled_now is None or self._throttled is None
            else throttled_now > self._throttled
        )
        record: dict[str, Any] = {
            "elapsed": elapsed,
            "work": max(0.0, work),
            "setup_work": self.timer.setup_work,
            "runtime_wait": self.timer.wait,
            "demand": self._declared(item)[0],
            "coverage": self.clock.coverage,
            "pressure": self.pressure.some(),
            "throttled": throttled,
        }
        setattr(outcome.get_result(), REPORT_ATTR, record)
        self._started = 0.0

    def pytest_unconfigure(self) -> None:
        if self.memory is not None:
            self.memory.close()
