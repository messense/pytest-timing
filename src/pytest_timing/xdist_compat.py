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
CANCEL = "timing_cancel"
"""The controller-to-worker command that withdraws tests a worker holds but was
never admitted to run; see ``cancel_tests``."""
GRANT = "timing_grant"
"""The controller-to-worker command that answers a ``REQUEST``: the slots are held."""
REQUEST = "timing_request"
"""The worker-to-controller event asking for slots for a fixture set-up the running
test did not declare statically (``getfixturevalue``); see ``forward_events``."""
HOLDS = "timing_holds"
"""A worker's live fixture holds after successful setup or finalization."""


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


def domain_of(node: Any) -> str:
    """The resource domain a worker runs in: ``local`` for popen workers, else the
    host named by its execnet spec (``ssh=host``, ``socket=host:port``)."""
    spec = getattr(getattr(node, "gateway", None), "spec", None)
    if spec is None or getattr(spec, "popen", None):
        return "local"
    for attr in ("ssh", "socket", "vagrant_ssh"):
        value = getattr(spec, attr, None)
        if value:
            return str(value)
    return str(getattr(spec, "id", "remote"))


Handler = Callable[[dict[str, Any]], None]


class _Router:
    """One layer in front of a message dispatcher that takes our ``(name, payload)``
    messages and passes everything else on; holds only what it needs."""

    __slots__ = ("handlers", "original")

    def __init__(self, original: Callable[[Any], Any]) -> None:
        self.original = original
        self.handlers: dict[str, Handler] = {}

    def take(self, message: Any) -> bool:
        """Handle ``message`` if it is one of ours; True when it was."""
        if not (isinstance(message, tuple) and len(message) == 2):
            return False
        handler = self.handlers.get(message[0])
        if handler is None:
            return False
        try:
            handler(message[1])
        except Exception:  # pragma: no cover - never break a receiver thread or run loop
            pass
        return True

    def __call__(self, message: Any) -> Any:
        if not self.take(message):
            return self.original(message)
        return None


def _router(owner: Any, attribute: str) -> _Router:
    """The router in front of ``owner.<attribute>``, installed on first use."""
    current = getattr(owner, attribute)
    if isinstance(current, _Router):
        return current
    router = _Router(current)
    setattr(owner, attribute, router)
    return router


def intercept_events(node: Any, name: str, handler: Handler) -> None:
    """Route the worker event ``name`` to ``handler`` instead of xdist's dispatcher.

    Called from ``pytest_configure_node``, before the controller registers the
    node's channel callback, so the router is what receives every event. xdist
    raises on event names it does not know, and the worker can send nothing else
    on its channel, so this is the one way a worker can hand the controller data
    ahead of its collection report. The handler runs on execnet's receiver thread:
    it must be quick and must not touch the scheduler.
    """
    _router(node, "process_from_remote").handlers[name] = handler


def intercept_shutdown(node: Any, before: Callable[[], None]) -> None:
    """Run ``before`` whenever ``node`` is told to shut down, by whoever tells it.

    xdist shuts workers down behind the scheduler's back (``-x``, ``--maxfail``, a
    crashed worker's restart budget) with a direct call on the node; wrapping the
    instance method is the only seam. Idempotent per node.
    """
    original = node.shutdown
    if getattr(original, "_pytest_timing_before", None) is not None:
        return

    def shutdown() -> None:
        before()
        original()

    shutdown._pytest_timing_before = before  # type: ignore[attr-defined]
    node.shutdown = shutdown


def shutdown_was_sent(node: Any) -> bool:
    """Was ``node`` told to shut down? Unlike ``shutting_down``, false for a worker
    that merely died, which xdist marks the same way."""
    return bool(getattr(node, "_shutdown_sent", node.shutting_down))


def _interactor(config: pytest.Config) -> Any | None:
    """xdist's ``WorkerInteractor`` in a worker process, or ``None``."""
    for plugin in config.pluginmanager.get_plugins():
        if type(plugin).__name__ == "WorkerInteractor" and hasattr(plugin, "sendevent"):
            return plugin
    return None


def send_event(config: pytest.Config, name: str, payload: dict[str, Any]) -> bool:
    """Send a custom event from a worker to the controller; False without a channel."""
    interactor = _interactor(config)
    if interactor is None:
        return False
    interactor.sendevent(name, **payload)
    return True


def intercept_commands(config: pytest.Config, name: str, handler: Handler) -> bool:
    """In a worker, route the controller command ``name`` to ``handler``; False
    outside a worker.

    Install before the interactor's run loop starts. The handler runs on execnet's
    receiver thread, possibly during a test; it must be quick and avoid pytest state.
    """
    interactor = _interactor(config)
    if interactor is None:
        return False
    _router(interactor, "handle_command").handlers[name] = handler
    return True


def send_command(node: Any, name: str, **payload: Any) -> bool:
    """Send a command of ours to ``node``'s worker; False when it cannot be sent (a
    stand-in without a channel, or a worker that is gone)."""
    send = getattr(node, "sendcommand", None)
    if send is None:
        return False
    try:
        send(name, **payload)
    except OSError:
        return False
    return True


def cancel_tests(node: Any, indices: list[int]) -> bool:
    """Tell ``node``'s worker not to run ``indices`` it holds."""
    return send_command(node, CANCEL, indices=list(indices))


def grant_slots(node: Any, key: str, request_id: int = 0) -> bool:
    """Answer ``node``'s request for the fixture behind ``key``: go ahead."""
    return send_command(node, GRANT, key=key, request_id=request_id)


def forward_events(node: Any, name: str) -> None:
    """Take the worker event ``name`` off ``node``'s channel and re-post it as an
    in-process event of the same name, so it reaches the controller's run loop on
    the main thread like xdist's own events do (``take_inproc_events``)."""

    def forward(payload: dict[str, Any]) -> None:
        node.notify_inproc(name, node=node, **payload)

    intercept_events(node, name, forward)


def take_inproc_events(config: pytest.Config, name: str, handler: Callable[..., None]) -> bool:
    """Have the controller's run loop call ``handler(node=..., **payload)`` for the
    in-process event ``name``: the loop looks the handler up as ``worker_<name>``
    on the session object, so it is set there."""
    dsession = config.pluginmanager.getplugin("dsession")
    if dsession is None:
        return False
    setattr(dsession, "worker_" + name, handler)
    return True


def version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("pytest-xdist")
    except importlib.metadata.PackageNotFoundError:
        return None
