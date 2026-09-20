"""Memory admission: estimates from history, the cost model, the gate, real runs."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import Collector, full_test, load_json, make_info, needs_xdist, run_timing
from test_schedule import FakeNode, make_scheduler, run_to_completion, running, start

from pytest_timing.demand import Declarations
from pytest_timing.model import MemoryRecord
from pytest_timing.plugin import format_bytes, parse_memory
from pytest_timing.schedule import Costs, Estimates

MIB = 1024 * 1024
SESSION = "session::/repo/conftest.py::model"
PACKAGE = "package:tests:/repo/tests/conftest.py::cache"
MODULE = "module:t.py:/repo/t.py::db"


def test_estimates_take_the_largest_need_and_attribute_retained_memory_to_fixtures() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::warm", 0.0)  # first on gw0: its window covers the warm-up
    c.tests[-1].memory = MemoryRecord(
        base=100 * MIB, peak=900 * MIB, after=800 * MIB, coverage="self"
    )
    full_test(c, "t.py::a", 1.0)
    c.tests[-1].memory = MemoryRecord(
        base=100 * MIB, peak=300 * MIB, after=110 * MIB, coverage="self"
    )
    full_test(c, "t.py::a", 2.0)  # a second attempt that needed more
    c.tests[-1].memory = MemoryRecord(
        base=110 * MIB, peak=360 * MIB, after=110 * MIB, coverage="self"
    )
    full_test(c, "t.py::b", 3.0)  # paid two set-ups and kept 400 MiB: 200 each
    c.tests[-1].fixtures = {SESSION: 0.5, MODULE: 0.2}
    c.tests[-1].memory = MemoryRecord(
        base=100 * MIB, peak=600 * MIB, after=500 * MIB, coverage="self"
    )
    full_test(c, "t.py::c", 4.0)  # no record at all
    full_test(c, "t.py::d", 5.0)  # a wait, but no reading
    c.tests[-1].memory = MemoryRecord(wait=0.5)
    run = c.finish(make_info().start + 6, termination="finished")
    est = Estimates.from_run(run)
    assert est.rise("t.py::a") == 250 * MIB
    # The payer needed its whole rise, but what stayed is the fixtures' (or the
    # worker's baseline); its own need is what it used beyond that.
    assert est.rise("t.py::b") == 100 * MIB
    assert est.rise("t.py::c") == 0 and "t.py::c" not in est.memory
    assert est.rise("t.py::d") == 0 and "t.py::d" not in est.memory
    assert est.kept(SESSION) == 0 and SESSION not in est.retained  # baseline, not a need
    assert est.kept(MODULE) == 200 * MIB
    assert est.kept("nothing") == 0
    # The warm-up window counts only while nothing better is known.
    assert est.rise("t.py::warm") == 800 * MIB
    full_test(c, "t.py::warm", 6.0, worker="gw1")  # not the first on its worker
    c.tests[-1].memory = MemoryRecord(
        base=100 * MIB, peak=150 * MIB, after=100 * MIB, coverage="self"
    )
    full_test(c, "t.py::e", 5.0, worker="gw1")  # gw1's warm-up: earlier, if reported later
    c.tests[-1].memory = MemoryRecord(
        base=100 * MIB, peak=700 * MIB, after=600 * MIB, coverage="self"
    )
    est = Estimates.from_run(c.finish(make_info().start + 8, termination="finished"))
    assert est.rise("t.py::warm") == 50 * MIB
    assert est.rise("t.py::e") == 600 * MIB


def test_session_and_package_fixture_memory_is_the_workers_baseline() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::a", 0.0)
    full_test(c, "t.py::b", 1.0)  # paid session, package and module set-ups: 300 each
    c.tests[-1].fixtures = {SESSION: 0.5, PACKAGE: 0.3, MODULE: 0.2}
    c.tests[-1].memory = MemoryRecord(
        base=100 * MIB, peak=1100 * MIB, after=1000 * MIB, coverage="self"
    )
    est = Estimates.from_run(c.finish(make_info().start + 2, termination="finished"))
    assert est.retained == {MODULE: 300 * MIB}
    assert est.rise("t.py::b") == 100 * MIB
    # Even hand-written estimates never charge them, and they do not make families.
    est = Estimates(
        {"t.py::a": 1.0, "t.py::b": 1.0},
        families={"t.py::a": frozenset({SESSION, PACKAGE})},
        setups={SESSION: 0.0, PACKAGE: 0.0},
        retained={SESSION: 4096 * MIB, PACKAGE: 1024 * MIB},
    )
    assert est.kept(SESSION) == 0 and est.kept(PACKAGE) == 0
    costs = Costs(est, ["t.py::a", "t.py::b"])
    assert not costs.retained and not costs.any_memory
    assert costs.family[0] == frozenset()
    assert costs.charge(0, set()).memory == 0


def test_costs_charge_a_test_with_its_need_and_the_fixtures_alive_around_it() -> None:
    ids = ["t.py::a", "t.py::b", "u.py::c"]
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        families={"t.py::a": frozenset({MODULE}), "t.py::b": frozenset({MODULE})},
        setups={MODULE: 0.0},  # too quick to matter for time, kept for its memory
        memory={"t.py::a": 50 * MIB, "u.py::c": 10 * MIB},
        retained={MODULE: 300 * MIB},
    )
    costs = Costs(est, ids, Declarations())
    assert costs.family[0] == frozenset({MODULE}) and costs.any_memory
    first = costs.charge(0, set())  # sets the fixture up: its need plus what it will keep
    assert first.memory == 350 * MIB and MODULE in first.after
    second = costs.charge(1, first.after)  # no need of its own, but the fixture is alive
    assert second.memory == 300 * MIB
    third = costs.charge(2, first.after)  # a module fixture is gone in another module
    assert third.memory == 10 * MIB and MODULE not in third.after
    assert costs.kept_memory(first.after) == 300 * MIB
    assert not Costs(Estimates(dict.fromkeys(ids, 1.0)), ids).any_memory


@pytest.mark.parametrize("declared", [False, True])
def test_runtime_fixture_resources_reconcile_without_recharging_history(declared: bool) -> None:
    ids = ["t.py::a"]
    key = MODULE + "[]"
    estimates = Estimates(
        {ids[0]: 1.0},
        families={ids[0]: frozenset({key})},
        setups={key: 0.1},
        retained={key: 300 * MIB},
    )
    declarations = Declarations(tests=[1], fixtures={MODULE: (2, 1)} if declared else {})
    charge = Costs(estimates, ids, declarations).charge(0, set())
    updated = charge.adding_fixture(key, setup=2, hold=1, live=0, memory=300 * MIB)
    assert updated.holds == 1  # even if history had timing/memory but no CPU declaration
    assert updated.peak == (3 if declared else 2)
    assert updated.memory == 300 * MIB
    assert updated.adding_fixture(key, 2, 1, 0, 300 * MIB) == updated
    assert (updated.seconds, updated.work, updated.before, updated.after) == (
        charge.seconds,
        charge.work,
        charge.before,
        charge.after,
    )
    # A newly discovered fixture remains separate even when its declaration matches.
    another = updated.adding_fixture(MODULE + "_other[]", 2, 1, 1, 100 * MIB)
    assert another.holds == 2 and another.memory == 400 * MIB


def memory_scheduler(
    est: Estimates,
    ids: list[str],
    workers: int = 2,
    budget: int | None = None,
    cpus: int | None = None,
    slots: dict[str, int] | None = None,
    host_memory: int | None = 16 * 1024 * MIB,
    auto: bool = False,
) -> Any:
    from pytest_timing.xdist_scheduler import CpuSetup

    decl = Declarations(
        tests=[(slots or {}).get(nodeid, 1) for nodeid in ids],
        host={"budget": 8, "cpus": 8, "memory": host_memory},
    )
    setup = CpuSetup(
        cpus=cpus,
        memory="auto" if auto else budget,
        declarations=lambda node: decl,
        domain_of=lambda node: "local",
    )
    return make_scheduler(est, workers, cpu=setup)


def run_checking_memory(sched: Any, nodes: list[FakeNode], actual: dict[int, float]) -> int:
    """Drive the run to completion; the most memory reserved at once for running tests."""
    peak = 0
    admission = sched.memory_admissions["local"]
    original = sched.mark_test_complete

    def complete(node: Any, index: int, duration: float = 0) -> None:
        nonlocal peak
        queues = {n: list(sched.node2pending[n]) for n in nodes if n in sched.node2pending}
        load = sum(
            min(sched.dispatched[n][queues[n][0]].memory, admission.limit)
            for n in running(sched, list(queues), queues)
        )
        peak = max(peak, load)
        assert load <= admission.limit, (load, admission.limit)
        original(node, index, duration)

    sched.mark_test_complete = complete
    run_to_completion(sched, nodes, actual)
    assert admission.forced == 0
    return peak


@needs_xdist
def test_hungry_tests_take_turns_under_a_memory_budget_alone() -> None:
    ids = ["h1", "h2", "l1", "l2", "l3", "l4"]
    est = Estimates(
        {"h1": 2.0, "h2": 2.0, **dict.fromkeys(ids[2:], 0.5)},
        memory={"h1": 700 * MIB, "h2": 700 * MIB, "l1": 10 * MIB},
    )
    sched = memory_scheduler(est, ids, budget=1000 * MIB)
    nodes = start(sched, ids)
    assert sched.gated and not sched.admissions and sched.memory_admissions["local"].budget
    assert sched.memory_summary()["gated"] and sched.memory_summary()["known_tests"] == 3
    assert sched.cpu_summary()["gated"]
    peak = run_checking_memory(sched, nodes, {i: est.estimate(ids[i]) for i in range(6)})
    assert peak <= 1000 * MIB and peak >= 700 * MIB
    assert sched.waits, "one hungry test had to wait for the other"
    assert all(w.gates == {"memory"} for w in sched.waits)  # and it was memory that held it
    assert sched.memory_summary()["waited_tests"] == 1
    assert sched.cpu_summary()["waited_tests"] == 0 and not sched.parked
    assert sched.now >= 4.0  # the hungry tests ran one after the other


@needs_xdist
def test_memory_and_cpu_gates_must_both_fit_before_a_test_starts() -> None:
    ids = ["cpu", "mem", "both", "l1", "l2"]
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        memory={"mem": 600 * MIB, "both": 600 * MIB},
    )
    sched = memory_scheduler(est, ids, budget=1000 * MIB, cpus=3, slots={"cpu": 2, "both": 2})
    nodes = start(sched, ids)
    cpu, mem = sched.admissions["local"], sched.memory_admissions["local"]
    queues = {n: list(n.queued) for n in nodes}
    ready = running(sched, nodes, queues)
    for node in ready:
        charge = sched.dispatched[node][queues[node][0]]
        assert cpu.held(node) >= charge.peak and mem.held(node) >= charge.memory
    run_checking_memory(sched, nodes, dict.fromkeys(range(5), 1.0))
    assert cpu.forced == 0 and cpu.used == 0 and mem.used == 0


@needs_xdist
def test_a_fixture_that_keeps_memory_counts_against_tests_on_other_workers() -> None:
    ids = ["f1", "f2", "f3", "big"]
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        families={f: frozenset({MODULE}) for f in ("f1", "f2", "f3")},
        setups={MODULE: 0.5},
        memory={"big": 600 * MIB},
        retained={MODULE: 500 * MIB},
    )
    sched = memory_scheduler(est, ids, budget=1000 * MIB)
    nodes = start(sched, ids)
    mem = sched.memory_admissions["local"]
    holder = next(n for n in nodes if MODULE in sched.lanes[n].fixtures)
    other = next(n for n in nodes if n is not holder)
    assert mem.held(holder) >= 500 * MIB
    # ``big`` cannot run next to the fixture: it is parked, or waits, until the
    # holder is done and gone, never admitted over the budget.
    assert ids.index("big") not in [i for n in [holder, other] for i in n.queued] or (
        other in sched.waiting
    )
    run_checking_memory(sched, nodes, dict.fromkeys(range(4), 1.0))
    assert all(w.gates == {"memory"} for w in sched.waits) and not sched.parked


@needs_xdist
def test_a_parked_worker_that_never_runs_keeps_its_wait_in_the_summary() -> None:
    ids = ["f1", "f2", "f3", "f4", "big"]
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        families={f: frozenset({MODULE}) for f in ids[:4]},
        setups={MODULE: 0.5},
        memory={"big": 600 * MIB},
        retained={MODULE: 500 * MIB},
    )
    sched = memory_scheduler(est, ids, workers=3, budget=1000 * MIB)
    nodes = start(sched, ids, workers=3)
    # The fixture family is split over two lanes; ``big`` cannot start next to what
    # they both keep, so the third worker sits parked with it, undispatched.
    parked = [n for n in nodes if n in sched.waiting and not sched.node2pending[n]]
    assert len(parked) == 1 and sched.refused[parked[0]] == {"memory"}
    assert sched.lanes[parked[0]].plan == [ids.index("big")]
    sched.now += 3.0
    sched.remove_node(parked[0])  # gone without a test: nothing to put the wait on
    assert [(w.index, w.seconds, w.gates) for w in sched.parked] == [
        (-1, 3.0, frozenset({"memory"}))
    ]
    summary = sched.memory_summary()
    assert summary["parked_workers"] == 1 and summary["parked"] == 3.0
    assert summary["waited_tests"] == 0
    assert sched.cpu_summary()["parked_workers"] == 0
    assert parked[0] not in sched.waiting and parked[0] not in sched.refused


def test_summary_lines_say_who_waited_and_who_sat_parked() -> None:
    from pytest_timing.plugin import _waited

    assert _waited({"waited_tests": 0, "parked_workers": 0}, "for memory") == ""
    assert (
        _waited({"waited_tests": 2, "waited": 1.5, "parked_workers": 1, "parked": 3}, "for memory")
        == "; 2 tests waited 1.50s for memory; 1 worker sat 3.00s parked for memory"
    )


@needs_xdist
def test_auto_budget_is_a_share_of_what_the_workers_report() -> None:
    ids = ["a", "b"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = memory_scheduler(est, ids, auto=True, host_memory=10 * 1024 * MIB)
    start(sched, ids)
    assert sched.memory_admissions["local"].budget == 8 * 1024 * MIB
    summary = sched.memory_summary()
    assert summary["domains"]["local"]["host"] == 10 * 1024 * MIB
    sched = memory_scheduler(est, ids, auto=True, host_memory=None)
    start(sched, ids)
    assert not sched.memory_admissions and sched.memory_summary()["gated"] is False


def test_memory_sizes_parse_with_binary_suffixes() -> None:
    assert parse_memory("auto") == "auto"
    assert parse_memory("1024") == 1024
    assert parse_memory("512M") == 512 * MIB
    assert parse_memory("8g") == 8 * 1024 * MIB
    assert parse_memory("2GiB") == 2 * 1024 * MIB
    assert parse_memory("1.5T") == int(1.5 * 2**40)
    assert parse_memory("64kB") == 64 * 1024
    for bad in ("", "0", "lots", "-1G", "1X"):
        with pytest.raises(pytest.UsageError):
            parse_memory(bad)
    assert format_bytes(300 * MIB) == "300.0 MiB"
    assert format_bytes(2**34) == "16.0 GiB"
    assert format_bytes(512) == "512 B"


HUNGRY = """
import time

def hog(mib, seconds):
    big = bytearray(mib * 1024 * 1024)
    big[::4096] = b"x" * len(big[::4096])
    time.sleep(seconds)

def test_hungry_a(): hog(200, 0.3)
def test_hungry_b(): hog(200, 0.3)
def test_light_1(): time.sleep(0.05)
def test_light_2(): time.sleep(0.05)
def test_light_3(): time.sleep(0.05)
"""


def _spans(doc: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {t["nodeid"].split("::")[-1]: (t["start"], t["stop"]) for t in doc["tests"]}


@needs_xdist
@pytest.mark.parametrize("dist", ["load", "worksteal"])
def test_the_second_run_keeps_hungry_tests_apart(pytester: pytest.Pytester, dist: str) -> None:
    from pytest_timing.telemetry import ResidentMemory

    if ResidentMemory().coverage == "none":
        pytest.skip("platform reports no resident memory")
    pytester.makepyfile(test_hungry=HUNGRY)
    args = ["-n", "2", "--dist", dist, "--timing-schedule", "pytest-timing.json"]
    args += ["--timing-memory", "300M"]
    first = run_timing(pytester, *args)
    first.assert_outcomes(passed=5)
    first.stdout.fnmatch_lines(
        ["memory: local: budget 300.0 MiB*; no recorded memory in the schedule history yet"]
    )
    doc = load_json(pytester.path)
    assert doc["run"]["memory"]["gated"] and doc["run"]["memory"]["known_tests"] == 0
    assert "cpu: " not in first.stdout.str()
    second = run_timing(pytester, *args)
    second.assert_outcomes(passed=5)
    second.stdout.fnmatch_lines(
        ["memory: local: budget 300.0 MiB*; 5 of 5 tests had recorded memory, largest 2*"]
    )
    doc = load_json(pytester.path)
    spans = _spans(doc)
    a, b = spans["test_hungry_a"], spans["test_hungry_b"]
    assert not (a[0] < b[1] and b[0] < a[1]), "the hungry tests overlapped"
    memory = doc["run"]["memory"]
    assert memory["domains"]["local"]["budget"] == 300 * MIB
    assert memory["domains"]["local"]["forced"] == 0
    assert memory["known_tests"] == 5 and memory["largest"] >= 190 * MIB
    assert memory["waited_tests"] >= 1 and memory["parked_workers"] == 0
    # The wait is on the test that was held, marked as memory's, and the chart says so.
    held = [t for t in doc["tests"] if t.get("memory", {}).get("wait")]
    assert [t["nodeid"].split("::")[-1] for t in held] in (["test_hungry_a"], ["test_hungry_b"])
    assert "wait" not in held[0].get("cpu", {})
    assert sum(t["memory"]["wait"] for t in held) == pytest.approx(memory["waited"], abs=1e-3)
    second.stdout.fnmatch_lines([f"*{held[0]['nodeid']}  waited * for memory"])


@needs_xdist
def test_auto_budget_and_settings_sources(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makepyfile(test_one="def test_one(): pass")
    result = run_timing(pytester, "-n", "1", "--timing-memory", "auto")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["memory: local: budget * of *"])
    pytester.makeini("[pytest]\ntiming_memory = 1G\n")
    result = run_timing(pytester, "-n", "1")
    result.stdout.fnmatch_lines(["memory: local: budget 1.0 GiB*"])
    monkeypatch.setenv("PYTEST_TIMING_MEMORY", "2G")
    result = run_timing(pytester, "-n", "1")
    result.stdout.fnmatch_lines(["memory: local: budget 2.0 GiB*"])  # env beats ini
    result = pytester.runpytest_subprocess("--timing-memory", "lots", "-p", "no:cacheprovider")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*timing_memory must be a size such as 512M or 8G*"])
    result = run_timing(pytester, "--timing-memory", "1G")  # without workers
    result.stdout.fnmatch_lines(["cpu: a budget needs pytest-xdist workers (-n); not applied"])
