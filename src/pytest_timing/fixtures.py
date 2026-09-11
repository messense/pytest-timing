"""Time shared fixture setup and carry dependency costs on pytest reports.

Keys identify scope, owner, definition and parameter dependencies. Dynamic fixture
requests also count for consumers of cached instances. Clocks exclude nested
setup and runtime admission waits. See ARCHITECTURE.md for the report contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from typing import Any

import pytest

REPORT_ATTR = "timing_fixtures"
SHARED_SCOPES: frozenset[str] = frozenset({"class", "module", "package", "session"})


def fixture_key(scope: str, node_id: str, defined_at: str, name: str, param: str) -> str:
    return f"{scope}:{node_id}:{defined_at}::{name}[{param}]"


def key_scope(key: str) -> str:
    return key.partition(":")[0]


def _scope_node_id(item: pytest.Item, scope: str, fixturedef: Any) -> str:
    if scope == "session":
        return ""
    if scope == "package":
        # Package node ids included __init__.py before pytest 8. A package-scoped
        # fixture with no matching Package collector belongs to the session.
        baseid = str(getattr(fixturedef, "baseid", "") or "")
        definition_node = getattr(fixturedef, "node", None)
        for package in reversed(item.listchain()):
            if not isinstance(package, pytest.Package):
                continue
            matches = (
                package is definition_node
                if definition_node is not None
                else package.nodeid in (baseid, baseid + "/__init__.py")
            )
            if matches:
                return package.nodeid.removesuffix("/__init__.py")
        return ""
    parent_type = pytest.Module if scope == "module" else pytest.Class
    parent = item.getparent(parent_type)
    # A class-scoped fixture on a function outside any class is per function.
    return parent.nodeid if parent is not None else item.nodeid


def fixturedefs(item: pytest.Item) -> dict[str, Any]:
    """The fixture definition each name resolves to for ``item`` (the innermost
    one, in closure order), from pytest's per-item fixture info."""
    info = getattr(item, "_fixtureinfo", None)
    name2fixturedefs: dict[str, Any] = getattr(info, "name2fixturedefs", None) or {}
    names = getattr(info, "names_closure", None) or list(name2fixturedefs)
    return {name: defs[-1] for name in names if (defs := name2fixturedefs.get(name))}


def _parametrized_dependencies(
    item: pytest.Item, fixturedef: Any, indices: dict[str, int]
) -> list[str]:
    """Names of the parametrized fixtures ``fixturedef`` depends on, transitively."""
    name2fixturedefs = fixturedefs(item)
    name = str(fixturedef.argname)
    found: list[str] = []
    seen = {name}
    todo = [str(a) for a in getattr(fixturedef, "argnames", ()) or ()]
    while todo:
        argname = todo.pop()
        if argname in seen:
            continue
        seen.add(argname)
        if argname in indices:
            found.append(argname)
        dependency = name2fixturedefs.get(argname)
        if dependency is not None:
            todo.extend(str(a) for a in getattr(dependency, "argnames", ()) or ())
    return sorted(found)


def _param_for(item: pytest.Item, fixturedef: Any) -> str:
    """The parameter part of a fixture key: which instance the item runs under.

    The item's callspec indexes every parameter it runs under, a fixture's own
    ``params`` and ``parametrize(..., indirect=True)`` alike. A fixture that depends
    on a parametrized one is set up once per parameter of that one too (pytest
    tears it down together with the instance it was built on), so those indices
    are part of its identity as well: ``derived[base=1]``.
    """
    indices = getattr(getattr(item, "callspec", None), "indices", None) or {}
    if not indices:
        return ""
    name = str(fixturedef.argname)
    parts = [str(indices[name])] if name in indices else []
    parts += [
        f"{dep}={indices[dep]}" for dep in _parametrized_dependencies(item, fixturedef, indices)
    ]
    return ",".join(parts)


def defined_at(fixturedef: Any) -> str:
    """Where a fixture definition lives: its ``baseid`` (the conftest directory or
    test module), or the qualified function for a plugin or root conftest fixture.
    Their baseids coincide on pytest 7, and a plugin class can live in conftest too.
    """
    baseid = str(getattr(fixturedef, "baseid", "") or "")
    if baseid not in ("", "."):
        return baseid
    func = getattr(fixturedef, "func", None)
    module = str(getattr(func, "__module__", "") or "")
    qualname = str(getattr(func, "__qualname__", "") or "")
    return f"{module}.{qualname}" if qualname else module


def _key_for(item: pytest.Item, fixturedef: Any) -> str | None:
    """The key of the fixture instance ``item`` uses; ``None`` for function scope."""
    scope = str(getattr(fixturedef, "scope", "function"))
    if scope not in SHARED_SCOPES:
        return None
    name = str(fixturedef.argname)
    where = defined_at(fixturedef)
    node_id = _scope_node_id(item, scope, fixturedef)
    return fixture_key(scope, node_id, where, name, _param_for(item, fixturedef))


def shared_fixtures(
    item: pytest.Item, dynamic: dict[str, frozenset[str]] | None = None
) -> dict[str, None]:
    """Keys of every shared fixture the item uses, direct, transitive or dynamic.

    The static closure names what the signatures ask for; the request's resolved
    fixture definitions add what ``getfixturevalue`` pulled in for this test, which
    the closure cannot know about; ``dynamic`` (key -> keys) adds what a shared
    fixture pulled in when it was set up, which a test served from the cache never
    sees requested.
    """
    definitions = fixturedefs(item)
    resolved = getattr(getattr(item, "_request", None), "_fixture_defs", None) or {}
    definitions.update(resolved)
    keys: dict[str, None] = {}
    todo = [key for fd in definitions.values() if (key := _key_for(item, fd)) is not None]
    while todo:
        key = todo.pop()
        if key in keys:
            continue
        keys[key] = None
        if dynamic:
            todo.extend(dynamic.get(key, ()))
    return keys


class FixtureTimer:
    """Records shared fixture set-ups during a test's setup phase onto its report.

    With a ``work`` clock (CPU seconds of the process tree, see
    :mod:`pytest_timing.telemetry`) it also sums the CPU time spent inside those
    set-ups into ``setup_work``, which stays readable until the next test starts.
    """

    def __init__(self, work: Callable[[], float] | None = None) -> None:
        self._item: pytest.Item | None = None
        self._setups: dict[str, float] = {}
        self._stack: list[float] = []  # seconds of nested set-ups to exclude, per level
        # What each shared fixture's set-up requested dynamically, by key: the cache
        # serves later tests without those requests, but they still depend on it.
        self._dynamic: dict[str, frozenset[str]] = {}
        self._work = work
        self.setup_work = 0.0
        self.wait = 0.0  # admission delay inside this test, including nested set-ups
        self.wait_work = 0.0  # process-tree CPU seconds during that delay

    def record_wait(self, seconds: float, work: float) -> None:
        """Pause all open fixture clocks while the worker waits for admission."""
        self.wait += seconds
        self.wait_work += work

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> Generator[None, None, None]:
        self._item = item
        self._setups = {}
        self.setup_work = 0.0
        self.wait = self.wait_work = 0.0
        yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef: Any, request: Any) -> Generator[None, None, None]:
        item = self._item
        key = _key_for(item, fixturedef) if item is not None else None
        if item is None or key is None:
            yield
            return
        # A fixture body may call ``getfixturevalue`` and nest another set-up inside
        # this interval; that one is timed on its own and excluded from this one.
        # The fixture's declared arguments were resolved before this hook, so what
        # the request resolves during it is what the body asked for dynamically.
        resolved: dict[str, Any] | None = getattr(request, "_fixture_defs", None)
        if resolved is None:
            resolved = {}  # (an empty map is the live one: never replace it)
        before = set(resolved)
        self._stack.append(0.0)
        started = time.perf_counter() - self.wait
        work_started = self._work() - self.wait_work if self._work is not None else 0.0
        yield
        elapsed = time.perf_counter() - self.wait - started
        nested = self._stack.pop()
        if self._stack:
            self._stack[-1] += elapsed
        self._setups[key] = elapsed - nested
        if self._work is not None and not self._stack:
            # Outermost set-up only: nested ones are inside this interval already.
            self.setup_work += max(0.0, self._work() - self.wait_work - work_started)
        pulled = [_key_for(item, fd) for name, fd in resolved.items() if name not in before]
        self._dynamic[key] = frozenset(k for k in pulled if k is not None)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: Any) -> Generator[None, Any, None]:
        outcome = yield
        if call.when == "teardown":
            self._item = None
            self._stack = []
            return
        # Setup and call: the test body may request (and set up) more fixtures, so
        # the call report carries everything known by then.
        fixtures: dict[str, float | None] = dict(shared_fixtures(item, self._dynamic))
        fixtures.update(self._setups)
        if fixtures:
            setattr(outcome.get_result(), REPORT_ATTR, fixtures)
