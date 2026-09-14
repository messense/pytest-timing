"""Resident memory of each attempt: the sampler, the record, and real runs."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
from conftest import Collector, full_test, load_json, make_info, needs_xdist, run_timing

from pytest_timing.collector import PhaseReport
from pytest_timing.model import MemoryRecord, Run
from pytest_timing.telemetry import MemorySampler, ResidentMemory

MIB = 1024 * 1024
ALLOCATE = "b = bytearray({mib} * 1024 * 1024); b[::4096] = b'x' * len(b[::4096])"


def touch(mib: int) -> bytearray:
    """A buffer with every page written, so it is resident and not merely reserved."""
    big = bytearray(mib * MIB)
    big[::4096] = b"x" * len(big[::4096])
    return big


IN_PROCESS = """
import json, time
from pytest_timing.telemetry import MemorySampler
sampler = MemorySampler()
assert sampler.end() is None  # nothing open yet
assert sampler.begin()
big = bytearray(64 * 1024 * 1024); big[::4096] = b"x" * len(big[::4096])
time.sleep(0.05)
del big
window = sampler.end()
sampler.close()
print(json.dumps({"base": window.base, "peak": window.peak, "after": window.after,
                  "coverage": window.coverage}))
"""


def test_memory_sampler_sees_what_a_process_allocates() -> None:
    sampler = MemorySampler()
    assert sampler.coverage in ("tree", "self", "none")
    if sampler.coverage == "none":
        assert not sampler.begin() and sampler.end() is None
        sampler.close()
        pytest.skip("platform reports no resident memory")
    sampler.close()
    # In a fresh interpreter: this process has run half a test suite by now, and a
    # grown heap (or a loaded host compressing its pages) can absorb the allocation.
    output = subprocess.run(
        [sys.executable, "-c", IN_PROCESS], check=True, capture_output=True, text=True
    ).stdout
    window = json.loads(output)
    assert window["coverage"] == sampler.coverage
    assert window["peak"] - window["base"] >= 48 * MIB  # kept the peak, not the end
    assert window["after"] >= window["base"] > 0


def test_memory_sampler_counts_a_live_child_where_it_can() -> None:
    sampler = MemorySampler(tree_interval=0.02)
    if sampler.coverage != "tree":
        pytest.skip("no way to see descendants here")
    assert sampler.begin()  # before the child exists: its whole footprint is a rise
    child = subprocess.Popen(
        [sys.executable, "-c", ALLOCATE.format(mib=64) + "; import time; time.sleep(2)"]
    )
    try:
        deadline = time.monotonic() + 5.0
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
            if sampler._children >= 48 * MIB:
                break
        assert child.poll() is None and sampler.memory.descendants() >= 48 * MIB
        window = sampler.end()
    finally:
        sampler.close()
        child.kill()
        child.wait()
    assert window is not None and window.peak >= window.after > 0
    if sys.platform.startswith("linux"):
        # Elsewhere the allocator may hand back the previous test's buffer during
        # this window, so the worker's own drop can hide part of the child's rise.
        assert window.after - window.base >= 48 * MIB
        assert window.peak - window.base >= 48 * MIB


class FakeMemory(ResidentMemory):
    """Scripted readings: the sampler's arithmetic without a platform."""

    def __init__(self, own: list[int | None], children: list[int]) -> None:
        self.coverage = "tree"
        self._owns = own
        self._children = children
        self.tree_reads = 0

    def own(self) -> int | None:
        return self._owns.pop(0) if len(self._owns) > 1 else self._owns[0]

    def descendants(self) -> int:
        self.tree_reads += 1
        return self._children.pop(0) if len(self._children) > 1 else self._children[0]


def test_memory_sampler_folds_descendants_in_and_keeps_the_peak() -> None:
    memory = FakeMemory(own=[100, 150, 300, 110], children=[10, 5])
    sampler = MemorySampler(memory, interval=0.001, tree_interval=60.0)
    try:
        assert sampler.begin()  # own 100 + children 10 (a fresh tree read)
        assert (sampler._base, sampler._peak) == (110, 110)
        deadline = time.monotonic() + 5.0
        while sampler._peak < 310 and time.monotonic() < deadline:
            time.sleep(0.005)
        window = sampler.end()  # own 110 + children 5 (fresh again)
    finally:
        sampler.close()
    assert window is not None
    assert (window.base, window.peak, window.after) == (110, 310, 115)
    assert memory.tree_reads == 2  # begin and end only: the interval never elapsed


def test_memory_sampler_without_a_reading_records_nothing() -> None:
    sampler = MemorySampler(FakeMemory(own=[None], children=[0]))
    try:
        assert not sampler.begin()
        assert sampler.end() is None
    finally:
        sampler.close()
    assert sampler._thread is None


def test_memory_records_round_trip_and_derive_rise_and_retained() -> None:
    record = MemoryRecord(base=100, peak=350, after=120, coverage="tree")
    assert (record.rise, record.retained) == (250, 20)
    assert MemoryRecord.from_dict(record.to_dict()) == record
    assert MemoryRecord(base=200, peak=100, after=50).rise == 0
    c = Collector(make_info())
    full_test(c, "t.py::a", 0.0)
    c.add_report(
        PhaseReport(
            nodeid="t.py::b",
            when="teardown",
            outcome="passed",
            start=make_info().start + 2.0,
            stop=make_info().start + 2.1,
            duration=0.1,
            worker="gw0",
            memory=record.to_dict(),
        )
    )
    run = c.finish(make_info().start + 3, termination="finished")
    by_id = {t.nodeid: t for t in run.tests}
    assert by_id["t.py::a"].memory is None
    assert by_id["t.py::b"].memory == record
    again = Run.from_json(run.to_json())
    assert [t.memory for t in again.tests] == [None, record]
    assert "memory" not in run.to_dict()["tests"][0]


HUNGRY = """
import time

def test_hungry():
    big = bytearray(64 * 1024 * 1024)
    big[::4096] = b"x" * len(big[::4096])
    time.sleep(0.05)

def test_frugal():
    time.sleep(0.05)
"""


@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=needs_xdist)])
def test_every_attempt_records_its_memory(pytester: pytest.Pytester, workers: int) -> None:
    pytester.makepyfile(test_hungry=HUNGRY)
    result = run_timing(pytester, *(["-n", str(workers)] if workers else []))
    result.assert_outcomes(passed=2)
    doc = load_json(pytester.path)
    records = {t["nodeid"].split("::")[-1]: t.get("memory") for t in doc["tests"]}
    if ResidentMemory().coverage == "none":
        assert records == {"test_hungry": None, "test_frugal": None}
        pytest.skip("platform reports no resident memory")
    hungry, frugal = records["test_hungry"], records["test_frugal"]
    assert hungry is not None and frugal is not None
    for record in (hungry, frugal):
        assert record["coverage"] in ("tree", "self")
        assert 0 < record["base"] <= record["peak"]
        assert record["after"] > 0
    assert hungry["peak"] - hungry["base"] >= 48 * MIB
    assert frugal["peak"] - frugal["base"] < 16 * MIB
