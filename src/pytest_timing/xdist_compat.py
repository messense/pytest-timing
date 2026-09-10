"""Everything that touches pytest-xdist internals, in one place.

xdist has no public API for "why did the controller stop?", so this module reads the
few attributes the controller session (``DSession``) exposes. Keep every such lookup
here so a change in xdist has exactly one place to break.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest


def is_distributed(config: pytest.Config) -> bool:
    """True on a controller that is actually distributing tests (``-n`` > 0)."""
    return config.pluginmanager.hasplugin("dsession")


def is_worker(config: pytest.Config) -> bool:
    return hasattr(config, "workerinput")


def worker_id(report: pytest.TestReport) -> str | None:
    """The gateway id (``gw0``) xdist attaches to reports on the controller."""
    node = getattr(report, "node", None)
    if node is None:
        return None
    gateway = getattr(node, "gateway", None)
    ident = getattr(gateway, "id", None)
    return str(ident) if ident is not None else None


def node_id(node: Any) -> str:
    return str(node.gateway.id)


def gateway_id(gateway: Any) -> str:
    return str(gateway.id)


_ABSENT = object()


def observe_gateway_creation(
    config: pytest.Config, on_created: Callable[[str, float], None]
) -> Callable[[], None] | None:
    """Time every gateway creation on the controller's execnet group.

    ``pytest_xdist_newgateway`` fires only after a gateway exists, and every hook-based
    marker before it sits ahead of some of xdist's own work (hook wrappers' post-yield
    code, preparing the remote module), so inferring a launch time from neighbouring
    events always leaks the previous worker's work into the next boot span. Instead,
    wrap the node manager's ``group.makegateway`` for the session: the timestamp is
    taken immediately before the call and associated with the gateway it returns.
    Covers initial workers and replacements alike. Returns a function that removes the
    observer, or ``None`` when the group is not reachable (then nothing is patched).
    """
    dsession = config.pluginmanager.getplugin("dsession")
    group = getattr(getattr(dsession, "nodemanager", None), "group", None)
    original = getattr(group, "makegateway", None)
    if group is None or original is None:
        return None
    # Another observer may already sit on the instance; keep it and put it back later.
    previous = group.__dict__.get("makegateway", _ABSENT)

    def makegateway(*args: Any, **kwargs: Any) -> Any:
        started = time.time()
        gateway = original(*args, **kwargs)
        on_created(gateway_id(gateway), started)
        return gateway

    group.makegateway = makegateway

    def restore() -> None:
        if group.__dict__.get("makegateway") is not makegateway:
            return  # someone else replaced it after us; not ours to touch
        if previous is _ABSENT:
            del group.__dict__["makegateway"]
        else:
            group.makegateway = previous

    return restore


class CollectionWatch:
    """Observe what each worker collected, to recognise a genuine mismatch abort.

    xdist's load-style schedulers abort when workers collect different node ids. They
    announce it with a failed ``CollectReport`` whose nodeid is the disagreeing worker's
    gateway id; but a worker can be given any id (``--tx popen//id=test_x.py``), so that
    shape alone is ambiguous with a forwarded collection error from a file of the same
    name. The abort is recognised only when this watch has itself seen two workers
    collect different ids.
    """

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}

    def collected(self, worker_id: str, ids: Any) -> None:
        import hashlib

        digest = hashlib.sha1("\0".join(str(i) for i in ids).encode("utf-8")).hexdigest()
        self._digests[worker_id] = digest

    @property
    def differs(self) -> bool:
        return len(set(self._digests.values())) > 1

    def mismatch_reason(self, report: Any) -> str | None:
        """The abort reason if ``report`` is the scheduler's mismatch announcement."""
        if not self.differs:
            return None
        if getattr(report, "outcome", None) != "failed":
            return None
        nodeid = getattr(report, "nodeid", None)
        if nodeid not in self._digests:
            return None
        longrepr = getattr(report, "longrepr", None)
        if not isinstance(longrepr, str):
            return None  # the scheduler's announcement is a plain message
        first = longrepr.strip().splitlines()[0] if longrepr.strip() else "collections differ"
        return f"{nodeid}: {first}"


def stop_reason(config: pytest.Config) -> str | None:
    """Why the controller decided to stop scheduling, if it did.

    Covers ``-x`` / ``--maxfail`` relayed from workers and a worker keyboard interrupt.
    A stop raises ``Interrupted`` out of the run loop, so it usually also surfaces as
    ``pytest_keyboard_interrupt``; this is the fallback when it does not.
    """
    dsession = config.pluginmanager.getplugin("dsession")
    value = getattr(dsession, "shouldstop", None)
    return str(value) if value else None


def abort_reason(config: pytest.Config) -> str | None:
    """Why the controller shut down early without raising, if it did.

    When a worker crashes and restarts are exhausted, ``DSession`` records the message
    it prints in the terminal summary and triggers a quiet shutdown. A crash that was
    recovered by restarting the worker leaves no such record, and the run completes.
    """
    dsession = config.pluginmanager.getplugin("dsession")
    value = getattr(dsession, "_summary_report", None)
    return str(value) if value else None


def version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("pytest-xdist")
    except importlib.metadata.PackageNotFoundError:
        return None
