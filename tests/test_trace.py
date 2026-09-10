from __future__ import annotations

import json

from pytest_timing.model import Run
from pytest_timing.render.trace import render_trace


def test_trace_events(sample_run: Run) -> None:
    doc = json.loads(render_trace(sample_run))
    events = doc["traceEvents"]
    names = {e["name"] for e in events if e["ph"] == "M"}
    assert names == {"process_name", "thread_name", "thread_sort_index"}
    threads = {e["args"]["name"] for e in events if e["name"] == "thread_name"}
    assert threads == {"gw0", "gw1"}
    tests = [e for e in events if e.get("cat") == "test"]
    assert len(tests) == 6
    two = next(e for e in tests if e["name"].endswith("test_two"))
    assert two["ts"] == 600_000
    assert two["dur"] == 1_400_000
    assert two["args"]["outcome"] == "passed"
    phases = [e for e in events if e.get("cat") == "phase"]
    assert {e["name"] for e in phases} == {"setup", "call", "teardown"}
    boot = next(e for e in events if e["name"] == "boot")
    assert boot["ts"] == 50_000 and boot["dur"] == 150_000
    assert any(e["name"] == "collect" for e in events)
    assert doc["metadata"]["run"]["dist"] == "load"
