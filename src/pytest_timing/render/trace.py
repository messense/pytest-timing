"""Chrome Trace Event renderer, viewable in Perfetto (https://ui.perfetto.dev) or chrome://tracing."""

from __future__ import annotations

import json
from typing import Any

from pytest_timing import __version__
from pytest_timing.model import Run, _round

PID = 1


def trace_events(run: Run) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"ph": "M", "pid": PID, "tid": 0, "name": "process_name", "args": {"name": "pytest"}},
    ]
    tids = {wid: i + 1 for i, wid in enumerate(run.worker_ids())}
    for wid, tid in tids.items():
        events.append(
            {"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": wid}}
        )
        events.append(
            {
                "ph": "M",
                "pid": PID,
                "tid": tid,
                "name": "thread_sort_index",
                "args": {"sort_index": tid},
            }
        )

    def us(seconds: float) -> int:
        return int(round(seconds * 1_000_000))

    for worker in run.workers:
        tid = tids[worker.id]
        if worker.ready is not None:
            if worker.start is not None:
                events.append(
                    {
                        "ph": "X",
                        "pid": PID,
                        "tid": tid,
                        "name": "boot",
                        "cat": "worker",
                        "ts": us(worker.start),
                        "dur": us(worker.ready - worker.start),
                    }
                )
            if worker.collected is not None:
                events.append(
                    {
                        "ph": "X",
                        "pid": PID,
                        "tid": tid,
                        "name": "collect",
                        "cat": "worker",
                        "ts": us(worker.ready),
                        "dur": us(worker.collected - worker.ready),
                        "args": {"items": worker.items},
                    }
                )
        if worker.error:
            events.append(
                {
                    "ph": "i",
                    "pid": PID,
                    "tid": tid,
                    "name": "worker error",
                    "cat": "worker",
                    "ts": us(worker.down or run.wall),
                    "s": "t",
                    "args": {"error": worker.error},
                }
            )

    for test in run.tests:
        tid = tids[test.worker]
        args = {
            "nodeid": test.nodeid,
            "outcome": test.outcome,
            "worker": test.worker,
            "attempt": test.attempt,
            "duration": _round(test.duration),
        }
        events.append(
            {
                "ph": "X",
                "pid": PID,
                "tid": tid,
                "name": test.nodeid,
                "cat": "test",
                "ts": us(test.start),
                "dur": us(test.duration),
                "args": args,
            }
        )
        for name, phase in test.phases.items():
            events.append(
                {
                    "ph": "X",
                    "pid": PID,
                    "tid": tid,
                    "name": name,
                    "cat": "phase",
                    "ts": us(phase.start),
                    "dur": us(phase.stop - phase.start),
                    "args": {"nodeid": test.nodeid, "duration": _round(phase.duration)},
                }
            )
    return events


def render_trace(run: Run) -> str:
    return json.dumps(
        {
            "traceEvents": trace_events(run),
            "displayTimeUnit": "ms",
            "metadata": {
                "pytest-timing": __version__,
                "run": run.run.to_dict(),
            },
        },
        separators=(",", ":"),
    )
