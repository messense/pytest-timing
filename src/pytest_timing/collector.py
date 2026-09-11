"""Fold pytest phase reports and xdist worker events into a :class:`Run`.

The collector knows nothing about pytest objects; :mod:`pytest_timing.plugin`
extracts plain values from reports and feeds them in. That keeps this module
trivially unit-testable, and every operation here is O(1) per report.

Identity is inferred, and the inference is deliberately narrow: reports carry only
a nodeid and a worker, so when a ``setup`` report arrives for a nodeid whose previous
span on that worker is closed, it is a new *attempt* of the same occurrence if that
span was retried (outcome ``rerun``), and otherwise a new *occurrence* (a duplicate
selection). Optional worker execution tokens associate admission waits with the
correct span; they do not change that public occurrence/attempt numbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pytest_timing.model import PHASES, CpuRecord, Phase, Run, RunInfo, TestSpan, Worker

CRASH_WHEN = "???"  # xdist synthesises a report with this ``when`` for crashed items
RETRIED = "rerun"


@dataclass(slots=True)
class PhaseReport:
    """The subset of a pytest ``TestReport`` the collector needs."""

    nodeid: str
    when: str
    outcome: str  # passed / failed / skipped / rerun as pytest (or a plugin) reports it
    start: float  # epoch seconds, 0.0 when unknown
    stop: float
    duration: float
    worker: str
    wasxfail: bool = False
    received: float = 0.0  # epoch seconds when the controller saw the report
    fixtures: dict[str, float | None] | None = None  # shared fixtures, setup and call
    cpu: dict[str, Any] | None = None  # the worker's CPU record, on the teardown report
    execution: tuple[int, int] | None = None  # collection index, attempt on that worker


class Collector:
    def __init__(self, run: RunInfo) -> None:
        self.run = run
        self.workers: dict[str, Worker] = {}
        self.tests: list[TestSpan] = []
        self._open: dict[tuple[str, str], TestSpan] = {}
        self._last: dict[tuple[str, str], TestSpan] = {}
        self._last_stop: dict[str, float] = {}  # when each worker's latest span closed
        self._executions: dict[tuple[str, int, int], TestSpan] = {}

    def _worker(self, worker_id: str) -> Worker:
        worker = self.workers.get(worker_id)
        if worker is None:
            worker = self.workers[worker_id] = Worker(id=worker_id)
        return worker

    def _rel(self, epoch: float) -> float:
        return epoch - self.run.start

    def worker_started(self, worker_id: str, epoch: float) -> None:
        """The controller began launching this worker (also for replacements)."""
        self._worker(worker_id).start = self._rel(epoch)

    def worker_ready(self, worker_id: str, epoch: float) -> None:
        self._worker(worker_id).ready = self._rel(epoch)

    def worker_collected(self, worker_id: str, epoch: float, items: int) -> None:
        worker = self._worker(worker_id)
        worker.collected = self._rel(epoch)
        worker.items = items

    def add_wait(
        self, worker_id: str, nodeid: str, index: int, attempt: int, seconds: float
    ) -> None:
        """Add an admission delay to the exact execution that waited.

        A nodeid can have several selections and retries. Looking up its last span
        at session finish would put every delay on the final one instead.
        """
        span = self._executions.get((worker_id, index, attempt))
        if span is None:
            # A worker may die in setup before sending any report (and therefore
            # any execution token). Only its unreported crash can match here;
            # never fall back to another attempt that has already sent reports.
            span = self._last.get((worker_id, nodeid))
            if span is None or span.outcome != "crashed" or span.phases or span.attempt != attempt:
                return
        if span.cpu is None:
            span.cpu = CpuRecord(elapsed=span.duration)
        span.cpu.wait += seconds

    def worker_down(self, worker_id: str, epoch: float, error: str | None) -> None:
        worker = self._worker(worker_id)
        worker.down = self._rel(epoch)
        if error:
            worker.error = error
        # Anything still open on this worker will never get a teardown.
        for span in [s for s in self._open.values() if s.worker == worker_id]:
            span.stop = max(span.stop, worker.down)
            span.outcome = "crashed"
            self._close(span)

    def add_report(self, report: PhaseReport) -> None:
        key = (report.worker, report.nodeid)
        if report.worker not in self.workers:
            self._worker(report.worker)

        if report.when == CRASH_WHEN:
            self._crash(key, report)
            return

        if report.start > 0 and report.stop > 0:
            base = self.run.start
            start, stop = report.start - base, report.stop - base
        else:
            start, stop = self._times(report)
        span = self._open.get(key)
        if report.when == "setup" or span is None:
            if span is not None:
                self._close(span)
            span = self._new_span(key, start, stop)

        if report.execution is not None:
            self._executions[(report.worker, *report.execution)] = span

        span.phases[report.when] = Phase(start=start, stop=stop, duration=report.duration)
        if report.fixtures:
            span.fixtures.update(report.fixtures)
        if report.cpu:
            span.cpu = CpuRecord.from_dict(report.cpu)
        span.start = min(span.start, start)
        span.stop = max(span.stop, stop)
        self._apply_outcome(span, report)

        if report.when == "teardown":
            self._close(span)

    def _new_span(self, key: tuple[str, str], start: float, stop: float) -> TestSpan:
        worker, nodeid = key
        previous = self._last.get(key)
        if previous is None:
            occurrence, attempt = 0, 0
        elif previous.outcome == RETRIED:
            occurrence, attempt = previous.occurrence, previous.attempt + 1
        else:
            occurrence, attempt = previous.occurrence + 1, 0
        span = TestSpan(
            nodeid=nodeid,
            worker=worker,
            occurrence=occurrence,
            attempt=attempt,
            outcome="passed",
            start=start,
            stop=stop,
        )
        self._open[key] = span
        self._last[key] = span
        self.tests.append(span)
        return span

    def _times(self, report: PhaseReport) -> tuple[float, float]:
        if report.start > 0 and report.stop > 0:
            return self._rel(report.start), self._rel(report.stop)
        # pytest < 7.3 or a synthetic report: anchor on receipt time.
        stop = self._rel(report.received) if report.received > 0 else 0.0
        return max(0.0, stop - report.duration), stop

    def _crash(self, key: tuple[str, str], report: PhaseReport) -> None:
        now = self._rel(report.received) if report.received > 0 else None
        span = self._open.get(key)
        if span is None:
            last = self._last.get(key)
            if last is not None and last.outcome == "crashed":
                # worker_down already closed it; the crash report only refines the end.
                if now is not None:
                    last.stop = max(last.stop, now)
                return
            # No phase report ever arrived: the test began when this lane last went idle.
            start = self._last_stop.get(report.worker)
            if start is None:
                worker = self.workers[report.worker]
                start = worker.collected if worker.collected is not None else worker.ready
            span = self._new_span(key, start or 0.0, start or 0.0)
        if now is not None:
            span.stop = max(span.stop, now)
        span.outcome = "crashed"
        self._close(span)

    @staticmethod
    def _apply_outcome(span: TestSpan, report: PhaseReport) -> None:
        if span.outcome in ("crashed", RETRIED):
            return
        if report.outcome == RETRIED:
            # pytest-rerunfailures: this attempt failed and will be retried.
            span.outcome = RETRIED
        elif report.outcome == "failed":
            span.outcome = "failed" if report.when == "call" else "error"
        elif report.outcome == "skipped" and span.outcome == "passed":
            span.outcome = "xfailed" if report.wasxfail else "skipped"
        elif report.wasxfail and report.when == "call" and span.outcome == "passed":
            span.outcome = "xpassed"

    def _close(self, span: TestSpan) -> None:
        self._open.pop((span.worker, span.nodeid), None)
        previous = self._last_stop.get(span.worker, 0.0)
        self._last_stop[span.worker] = max(previous, span.stop)

    def finish(self, epoch: float, *, termination: str, reason: str | None = None) -> Run:
        self.run.stop = epoch
        self.run.termination = termination
        self.run.reason = reason
        complete = self.run.complete
        for span in list(self._open.values()):
            if not complete:
                # Still running when the session was interrupted.
                span.stop = max(span.stop, self._rel(epoch))
                if "teardown" not in span.phases:
                    span.outcome = "crashed"
            self._close(span)
        # Give every span a stable phase order for serialisation.
        for span in self.tests:
            span.phases = {p: span.phases[p] for p in PHASES if p in span.phases} | {
                p: v for p, v in span.phases.items() if p not in PHASES
            }
        workers = sorted(self.workers.values(), key=_worker_sort_key)
        return Run(run=self.run, workers=workers, tests=list(self.tests))


def _worker_sort_key(worker: Worker) -> tuple[int, int, str]:
    """Sort gw0, gw1, ... gw10 numerically, anything else after."""
    wid = worker.id
    if wid.startswith("gw") and wid[2:].isdigit():
        return (0, int(wid[2:]), wid)
    return (1, 0, wid)
