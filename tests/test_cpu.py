"""CPU demand: declarations, measurement, and admission with real workers."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

import pytest
from conftest import (
    HAS_XDIST,
    T0,
    Collector,
    full_test,
    load_json,
    make_info,
    needs_xdist,
    run_timing,
)

import pytest_timing
from pytest_timing.demand import Declarations, fixture_demand, item_demand, key_base
from pytest_timing.model import CpuRecord
from pytest_timing.schedule import Estimates
from pytest_timing.telemetry import Pressure, ProcessTreeClock, host_cpu

BURN = "import time; t = time.process_time()\nwhile time.process_time() - t < {seconds}: pass"


MARKERS = """
import pytest

pytestmark = pytest.mark.timing_cpu(2)

def test_module_default(): pass

@pytest.mark.timing_cpu(4)
def test_function(): pass

class TestGroup:
    pytestmark = pytest.mark.timing_cpu(3)

    def test_class_default(self): pass

    @pytest.mark.timing_cpu(slots=5)
    def test_override(self): pass

@pytest.mark.timing_cpu("lots")
def test_bad_type(): pass

@pytest.mark.timing_cpu(0)
def test_bad_value(): pass

@pytest.mark.timing_cpu()
def test_missing(): pass

@pytest.mark.timing_cpu(True)
def test_bool(): pass
"""


def test_closest_marker_wins_and_invalid_ones_are_reported_not_applied(
    pytester: pytest.Pytester,
) -> None:
    items = {item.name: item for item in pytester.getitems(MARKERS)}
    assert item_demand(items["test_module_default"]) == (2, None)
    assert item_demand(items["test_function"]) == (4, None)
    assert item_demand(items["test_class_default"]) == (3, None)
    assert item_demand(items["test_override"]) == (5, None)
    for name in ("test_bad_type", "test_bad_value", "test_missing", "test_bool"):
        slots, error = item_demand(items[name])
        assert slots == 1 and error and "timing_cpu" in error


FIXTURES = """
import pytest
import pytest_timing

@pytest_timing.cpu(4, hold=1)
@pytest.fixture(scope="session")
def above():
    yield

@pytest.fixture(scope="module")
@pytest_timing.cpu(3)
def below():
    yield

@pytest.fixture(scope="session", params=[1, 2])
@pytest_timing.cpu(2)
def param(request):
    yield request.param

@pytest_timing.cpu(6)
def test_plain(above, below, param): pass

def test_other(): pass
"""


def test_fixture_decorator_works_in_either_order_and_marks_plain_functions(
    pytester: pytest.Pytester,
) -> None:
    items = pytester.getitems(FIXTURES)
    plain = next(i for i in items if i.name.startswith("test_plain"))
    defs = plain._fixtureinfo.name2fixturedefs  # type: ignore[attr-defined]
    assert fixture_demand(defs["above"][-1]) == pytest_timing.demand.Demand(4, 1)
    assert fixture_demand(defs["below"][-1]) == pytest_timing.demand.Demand(3, 0)
    assert fixture_demand(defs["param"][-1]) == pytest_timing.demand.Demand(2, 0)
    assert item_demand(plain) == (6, None)  # on a plain function: the marker
    with pytest.raises(TypeError):
        pytest_timing.cpu("4")
    with pytest.raises(ValueError):
        pytest_timing.cpu(0)
    with pytest.raises(TypeError):
        pytest_timing.cpu(2, hold=-1)
    decl = Declarations.from_items(items, host={"budget": 8})
    assert decl.tests == [6, 6, 1]  # two parametrized plain tests, one other
    assert decl.errors == {}
    keys = decl.families[0]
    assert {key_base(k) for k in keys} == {
        "session::test_fixture_decorator_works_in_either_order_and_marks_plain_functions.py::above",
        "module:test_fixture_decorator_works_in_either_order_and_marks_plain_functions.py:"
        "test_fixture_decorator_works_in_either_order_and_marks_plain_functions.py::below",
        "session::test_fixture_decorator_works_in_either_order_and_marks_plain_functions.py::param",
    }
    assert any(k.endswith("param[0]") for k in decl.families[0])
    assert any(k.endswith("param[1]") for k in decl.families[1])
    assert decl.fixtures[key_base(keys[0])] in {(4, 1), (3, 0), (2, 0)}
    again = Declarations.from_dict(json.loads(json.dumps(decl.to_dict())))
    assert again == decl


def test_the_cost_model_knows_whether_anything_declares_more_than_one_slot() -> None:
    from pytest_timing.schedule import Costs

    est = Estimates(dict.fromkeys(["a", "b"], 1.0))
    assert not Costs(est, ["a", "b"], Declarations(tests=[1, 1], fixtures={"k": (1, 0)})).any_demand
    assert Costs(est, ["a", "b"], Declarations(tests=[1, 2])).any_demand
    assert Costs(est, ["a", "b"], Declarations(tests=[1, 1], fixtures={"k": (1, 1)})).any_demand


def test_process_tree_clock_counts_children_the_process_reaped() -> None:
    clock = ProcessTreeClock()
    assert clock.coverage in ("tree", "reaped", "self")
    before = clock.seconds()
    subprocess.run([sys.executable, "-c", BURN.format(seconds=0.2)], check=True)
    after = clock.seconds()
    if clock.coverage == "self":
        pytest.skip("platform reports nothing about children")
    assert after - before >= 0.15


def test_process_tree_clock_counts_live_descendants_where_it_can() -> None:
    clock = ProcessTreeClock()
    if clock.coverage != "tree":
        pytest.skip("no way to see live descendants here")
    child = subprocess.Popen([sys.executable, "-c", BURN.format(seconds=0.4)])
    try:
        before = clock.seconds()
        deadline = time.monotonic() + 5.0
        during = 0.0
        while child.poll() is None and time.monotonic() < deadline:
            during = clock.seconds() - before
            if during >= 0.1:
                break
            time.sleep(0.02)
        assert child.poll() is None and during >= 0.1  # still alive: counted from /proc
    finally:
        child.wait()
    after = clock.seconds()
    assert after - before >= 0.3  # reaped now: moved to the reaped total, not doubled
    assert after - before < 0.8


def test_host_detection_is_sane_and_serialisable() -> None:
    host = host_cpu()
    assert host.budget >= 1
    if host.affinity is not None:
        assert host.budget <= host.affinity
    doc = host.to_dict()
    assert doc["budget"] == host.budget and doc["platform"] == sys.platform
    json.dumps(doc)


def test_pressure_readings_are_absent_or_in_range() -> None:
    pressure = Pressure()
    some = pressure.some()
    assert some is None or 0.0 <= some <= 1.0
    throttled = pressure.throttled()
    assert throttled is None or throttled >= 0


def test_estimates_prefer_attempts_recorded_without_contention() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::a", 0.0, setup=0.0, call=2.0, teardown=0.0)
    c.tests[-1].cpu = CpuRecord(elapsed=2.0, work=2.0, pressure=0.9)  # a loaded host
    full_test(c, "t.py::a", 3.0, setup=0.0, call=1.0, teardown=0.0)
    c.tests[-1].cpu = CpuRecord(elapsed=1.0, work=1.0, pressure=0.0)
    full_test(c, "t.py::b", 5.0, setup=0.0, call=3.0, teardown=0.0)
    c.tests[-1].cpu = CpuRecord(elapsed=3.0, work=3.0, throttled=True)  # only one
    run = c.finish(T0 + 9, termination="finished")
    est = Estimates.from_run(run)
    assert est.durations == pytest.approx({"t.py::a": 1.0, "t.py::b": 3.0})
    doc = run.to_dict()
    assert doc["tests"][0]["cpu"]["pressure"] == 0.9
    assert CpuRecord.from_dict(doc["tests"][2]["cpu"]).contended


HEAVY = """
import time
import pytest

@pytest.mark.timing_cpu(2)
def test_heavy_a(): time.sleep(0.3)

@pytest.mark.timing_cpu(2)
def test_heavy_b(): time.sleep(0.3)

def test_light_1(): time.sleep(0.05)
def test_light_2(): time.sleep(0.05)
def test_light_3(): time.sleep(0.05)
"""


def _spans(doc: dict[str, Any]) -> dict[str, tuple[float, float, str]]:
    return {t["nodeid"].split("::")[-1]: (t["start"], t["stop"], t["worker"]) for t in doc["tests"]}


def _overlap(a: tuple[float, float, str], b: tuple[float, float, str]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@needs_xdist
@pytest.mark.parametrize("dist", ["load", "worksteal"])
def test_heavy_tests_never_share_the_budget(pytester: pytest.Pytester, dist: str) -> None:
    pytester.makepyfile(test_heavy=HEAVY)
    result = run_timing(pytester, "-n", "2", "--dist", dist, "--timing-cpus", "2")
    result.assert_outcomes(passed=5)
    result.stdout.fnmatch_lines(["cpu: local: budget 2*; 2 tests over one slot*"])
    doc = load_json(pytester.path)
    spans = _spans(doc)
    assert not _overlap(spans["test_heavy_a"], spans["test_heavy_b"])
    for light in ("test_light_1", "test_light_2", "test_light_3"):
        for heavy in ("test_heavy_a", "test_heavy_b"):
            assert not _overlap(spans[light], spans[heavy])
    cpu = doc["run"]["cpu"]
    assert cpu["gated"] and cpu["domains"]["local"]["budget"] == 2
    assert cpu["domains"]["local"]["forced"] == 0
    records = {t["nodeid"].split("::")[-1]: t["cpu"] for t in doc["tests"]}
    assert records["test_heavy_a"]["demand"] == 2 and records["test_light_1"]["demand"] == 1
    assert all(r["coverage"] in ("tree", "reaped", "self") for r in records.values())
    assert any(r.get("wait", 0) > 0 for r in records.values())
    assert cpu["waited_tests"] >= 1 and cpu["waited"] > 0


COLD_FIXTURE = """
import time
import pytest
import pytest_timing

@pytest_timing.cpu(2)
@pytest.fixture(scope="session")
def build():
    time.sleep(0.2)
    yield

def test_1(build): time.sleep(0.05)
def test_2(build): time.sleep(0.05)
def test_3(build): time.sleep(0.05)
def test_4(build): time.sleep(0.05)
"""


@needs_xdist
def test_cold_fixture_setups_reserve_their_own_demand(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_cold=COLD_FIXTURE)
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2")
    result.assert_outcomes(passed=4)
    doc = load_json(pytester.path)
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for test in sorted(doc["tests"], key=lambda t: t["start"]):
        by_worker.setdefault(test["worker"], []).append(test)
    firsts = [tests[0] for tests in by_worker.values()]
    if len(firsts) == 2:  # both workers set the fixture up: never at the same time
        a, b = firsts
        assert not _overlap((a["start"], a["stop"], ""), (b["start"], b["stop"], ""))
        assert all(t["cpu"]["setup_work"] >= 0.0 for t in firsts)
    later = [t for tests in by_worker.values() for t in tests[1:]]
    assert len(later) + len(firsts) == 4
    result.stdout.fnmatch_lines(["cpu: local: budget 2*; 0 tests over one slot*"])


HOLD_FIXTURE = """
import time
import pytest
import pytest_timing

@pytest_timing.cpu(1, hold=1)
@pytest.fixture(scope="session")
def server():
    yield

def test_1(server): time.sleep(0.1)
def test_2(server): time.sleep(0.1)
def test_3(server): time.sleep(0.1)
def test_4(server): time.sleep(0.1)
"""


@needs_xdist
def test_held_slots_keep_counting_while_a_worker_waits(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_hold=HOLD_FIXTURE)
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2")
    result.assert_outcomes(passed=4)
    spans = list(_spans(load_json(pytester.path)).values())
    # Each worker holds one slot for the fixture and one for its test: two workers
    # can never run at the same time on a budget of two.
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            if a[2] != b[2]:
                assert not _overlap(a, b), (a, b)


OVERSIZED = """
import time
import pytest

@pytest.mark.timing_cpu(64)
def test_huge(): time.sleep(0.1)

def test_small_1(): time.sleep(0.05)
def test_small_2(): time.sleep(0.05)
"""


@needs_xdist
def test_a_demand_above_the_budget_runs_alone_instead_of_never(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_big=OVERSIZED)
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2")
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        ["cpu: local: budget 2*, 1 request over budget run alone; 1 test over one slot*"]
    )
    spans = _spans(load_json(pytester.path))
    for small in ("test_small_1", "test_small_2"):
        assert not _overlap(spans["test_huge"], spans[small])


@needs_xdist
def test_an_invalid_marker_fails_its_test_with_the_reason(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_bad="""
        import pytest

        @pytest.mark.timing_cpu("lots")
        def test_bad(): pass

        def test_good(): pass
        """
    )
    result = pytester.runpytest_subprocess(
        "-n", "2", "--timing-cpus", "2", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*pytest-timing: timing_cpu expects a positive integer*"])


def test_invalid_budget_is_a_usage_error(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_x="def test_x(): pass")
    for bad in ("0", "-2", "many"):
        result = pytester.runpytest_subprocess("--timing-cpus", bad, "-p", "no:cacheprovider")
        assert result.ret == 4
        result.stderr.fnmatch_lines(["*timing_cpus must be a positive integer or 'auto'*"])


SPAWNS = """
import subprocess, sys

BURN = "import time; t = time.process_time()\\nwhile time.process_time() - t < 0.4: pass"

def test_spawns():
    procs = [subprocess.Popen([sys.executable, "-c", BURN]) for _ in range(2)]
    for p in procs:
        p.wait()

def test_idle():
    import time
    time.sleep(0.05)
"""


@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=needs_xdist)])
def test_subprocess_cpu_work_is_measured(pytester: pytest.Pytester, workers: int) -> None:
    pytester.makepyfile(test_spawn=SPAWNS)
    args = ["-n", str(workers), "--timing-cpus", "2"] if workers else []
    result = run_timing(pytester, *args)
    result.assert_outcomes(passed=2)
    doc = load_json(pytester.path)
    records = {t["nodeid"].split("::")[-1]: t["cpu"] for t in doc["tests"]}
    spawns, idle = records["test_spawns"], records["test_idle"]
    if spawns["coverage"] == "self":
        pytest.skip("platform reports nothing about children")
    # Two concurrent 0.4 s burners: 0.8 s of CPU counted to the test, and with CPUs
    # to spare for them next to this test run, in well under 0.8 s of wall time.
    assert spawns["work"] >= 0.7
    if host_cpu().budget >= 4:
        assert spawns["work"] > spawns["elapsed"] * 1.2
    assert idle["work"] < 0.05
    assert spawns["demand"] == 1
    host = doc["run"]["cpu"]["domains"]["local"]["host"]
    assert host["budget"] >= 1 and host["platform"] == sys.platform


def test_ini_and_env_set_the_budget(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makepyfile(test_heavy=HEAVY)
    pytester.makeini("[pytest]\ntiming_cpus = 2\n")
    if not HAS_XDIST:
        result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
        result.assert_outcomes(passed=5)
        assert "timing report" in result.stdout.str()  # implies --timing
        return
    result = pytester.runpytest_subprocess("-n", "2", "-p", "no:cacheprovider")
    result.stdout.fnmatch_lines(["cpu: local: budget 2*; 2 tests over one slot*"])
    monkeypatch.setenv("PYTEST_TIMING_CPUS", "auto")
    result = pytester.runpytest_subprocess("-n", "2", "-p", "no:cacheprovider")
    result.stdout.fnmatch_lines(["cpu: local: budget * (* cpus*); 2 tests over one slot*"])
    budget = int(result.stdout.str().split("cpu: local: budget ")[1].split()[0])
    assert budget >= 2


@needs_xdist
def test_declared_demand_alone_turns_the_gate_on_with_the_schedule(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_heavy=HEAVY)
    args = ["-n", "2", "--timing-schedule", "pytest-timing.json"]
    run_timing(pytester, *args).assert_outcomes(passed=5)
    result = run_timing(pytester, *args)
    result.assert_outcomes(passed=5)
    result.stdout.fnmatch_lines(
        [
            "schedule: 5 of 5 tests had recorded durations in pytest-timing.json",
            "cpu: local: budget *; 2 tests over one slot*",
        ]
    )
    # Without a declaration and without --timing-cpus there is no gate at all.
    pytester.makepyfile(test_heavy=HEAVY.replace("@pytest.mark.timing_cpu(2)\n", ""))
    result = run_timing(pytester, *args)
    result.assert_outcomes(passed=5)
    assert "cpu:" not in result.stdout.str()


CRASH = """
import os, time
import pytest

@pytest.mark.timing_cpu(2)
def test_dies():
    os._exit(1)

@pytest.mark.timing_cpu(2)
def test_heavy(): time.sleep(0.2)

def test_light_1(): time.sleep(0.05)
def test_light_2(): time.sleep(0.05)
"""


@needs_xdist
def test_a_crash_under_the_gate_frees_its_slots_and_the_run_completes(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_crash=CRASH)
    result = run_timing(
        pytester, "-n", "2", "--timing-cpus", "2", "--max-worker-restart", "1", timeout=60
    )
    result.assert_outcomes(passed=3, failed=1)
    doc = load_json(pytester.path)
    assert doc["run"]["termination"] == "finished"
    assert doc["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
def test_maxfail_stops_a_gated_run_without_hanging(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_stop="""
        import time
        import pytest

        @pytest.mark.timing_cpu(2)
        def test_fails():
            time.sleep(0.1)
            assert False

        @pytest.mark.timing_cpu(2)
        def test_heavy(): time.sleep(0.3)

        def test_light_1(): time.sleep(0.05)
        def test_light_2(): time.sleep(0.05)
        def test_light_3(): time.sleep(0.05)
        """
    )
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2", "-x", timeout=60)
    assert result.ret != 0
    doc = load_json(pytester.path)
    assert doc["run"]["termination"] == "interrupted"
    assert len(doc["tests"]) < 5


INNER_FIXTURES = """
import pytest
import pytest_timing

@pytest_timing.cpu(3)
@pytest.fixture
def compiler():
    yield

@pytest_timing.cpu(2, hold=1)
@pytest.fixture
def server():
    yield

@pytest_timing.cpu(2)
@pytest.fixture(scope="session")
def warehouse():
    yield

def test_compiles(compiler): pass
def test_serves(server): pass
def test_both(compiler, server): pass
@pytest.mark.timing_cpu(4)
def test_marked(server): pass
def test_dynamic(request): request.getfixturevalue("warehouse")
"""


def test_function_scoped_fixtures_count_inside_the_test_and_definitions_are_declared(
    pytester: pytest.Pytester,
) -> None:
    items = pytester.getitems(INNER_FIXTURES)
    session = items[0].session
    decl = Declarations.from_items(items, {"budget": 8}, session._fixturemanager)
    by_name = dict(zip([i.name for i in items], decl.tests, strict=True))
    # A function-scoped set-up runs inside the test's own reservation: the test
    # weighs the set-up, or its own slots plus what such fixtures hold.
    assert by_name["test_compiles"] == 3
    assert by_name["test_serves"] == 2  # max(1 own + 1 held, 2 set-up)
    assert by_name["test_both"] == 4  # compiler's set-up while server holds a slot
    assert by_name["test_marked"] == 5  # 4 own + 1 held
    assert by_name["test_dynamic"] == 1  # nothing in its signature
    assert decl.families == {}  # no shared fixture is named by any signature
    # Every declared shared definition is known, so a recorded run that reports
    # ``warehouse`` under ``test_dynamic`` finds its slots by definition.
    (definition,) = [key for key in decl.definitions if key.startswith("session:")]
    assert definition.startswith("session:") and definition.endswith("::warehouse")
    assert decl.definitions[definition] == (2, 0)
    module = definition.partition(":")[2].rpartition("::")[0]
    assert decl.lookup(f"session::{module}::warehouse") == (2, 0)
    assert decl.lookup("session::elsewhere.py::warehouse") is None
    assert decl.lookup(f"module:{module}:{module}::warehouse") is None  # wrong scope
    again = Declarations.from_dict(json.loads(json.dumps(decl.to_dict())))
    assert again == decl
    # The same through the planner's cost model, from a recorded family.
    est = Estimates(
        {"test_dynamic": 1.0},
        1.0,
        "",
        {"test_dynamic": frozenset({f"session::{module}::warehouse[]"})},
        {f"session::{module}::warehouse[]": 0.5},
    )
    from pytest_timing.schedule import Costs

    costs = Costs(est, ["test_dynamic"], Declarations(tests=[1], definitions=decl.definitions))
    assert costs.charge(0, set()).peak == 2


DERIVED = """
import time
import pytest
import pytest_timing

@pytest.fixture(scope="session", params=[1, 2])
def base(request):
    return request.param

@pytest_timing.cpu(2)
@pytest.fixture(scope="session")
def derived(base):
    time.sleep(0.2)
    return base * 10

def test_a(derived): time.sleep(0.05)
def test_b(derived): time.sleep(0.05)
def test_c(derived): time.sleep(0.05)
"""


@needs_xdist
def test_a_fixture_built_on_a_parameter_is_set_up_again_under_its_own_reservation(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_derived=DERIVED)
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2")
    result.assert_outcomes(passed=6)
    doc = load_json(pytester.path)
    setups = [
        t
        for t in doc["tests"]
        if any("::derived[" in k and v and v > 0.1 for k, v in (t.get("fixtures") or {}).items())
    ]
    # Each parameter of base builds derived again: at least two set-ups, keyed by
    # the parameter they were built on, and each one alone on the budget.
    keys = {k for t in setups for k in t["fixtures"] if "::derived[" in k}
    assert {k.rpartition("[")[2] for k in keys} == {"base=0]", "base=1]"}
    assert len(setups) >= 2
    spans = _spans(doc)
    for test in setups:
        me = spans[test["nodeid"].split("::")[-1]]
        for name, other in spans.items():
            if other[2] != me[2]:
                assert not _overlap(me, other), (test["nodeid"], name)
    assert doc["run"]["cpu"]["domains"]["local"]["forced"] == 0


STOP = """
import pathlib, time
import pytest

@pytest.mark.timing_cpu(2)
def test_fails():
    time.sleep(0.1)
    assert False

@pytest.mark.timing_cpu(2)
def test_heavy():
    pathlib.Path("ran_heavy").write_text("yes")
    time.sleep(0.5)

def test_light_1(): time.sleep(0.05)
def test_light_2(): time.sleep(0.05)
def test_light_3(): time.sleep(0.05)
"""


@needs_xdist
def test_maxfail_withdraws_a_test_that_was_never_admitted(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_stop=STOP)
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2", "-x", timeout=60)
    assert result.ret != 0
    doc = load_json(pytester.path)
    assert doc["run"]["termination"] == "interrupted"
    ran = {t["nodeid"].split("::")[-1] for t in doc["tests"]}
    cpu = doc["run"]["cpu"]["domains"]["local"]
    if "test_heavy" in ran:
        # It went first and finished before the failure: nothing was waiting.
        assert (pytester.path / "ran_heavy").exists()
        return
    # It was waiting for the failing test's slots when xdist shut the workers
    # down: told to skip it, the worker never started it.
    assert cpu["forced"] == 0
    assert not (pytester.path / "ran_heavy").exists()
    assert doc["run"]["cpu"]["cancelled"] == 1
    result.stdout.fnmatch_lines(["cpu: local: budget 2*; 1 unadmitted test withdrawn at shutdown"])


def _writing_clears(path_type: type) -> Any:
    """``write_text`` that also forgets the cached cgroup layout: the test rewrites
    the process's membership file, which a real process never sees change."""
    from pytest_timing import telemetry

    original = path_type.write_text

    def write_text(self: Any, *args: Any, **kwargs: Any) -> Any:
        telemetry.cgroup_layout.cache_clear()
        return original(self, *args, **kwargs)

    return write_text


def test_cgroup_quota_follows_the_hierarchy_that_holds_the_cpu_controller(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pytest_timing import telemetry

    root = tmp_path / "cgroup"
    proc = tmp_path / "proc"
    cpu = root / "cpu,cpuacct"
    (cpu / "job" / "task").mkdir(parents=True)
    (root / "unified" / "job").mkdir(parents=True)
    (proc / "self").mkdir(parents=True)
    (cpu / "cpu.cfs_quota_us").write_text("-1\n")  # the mount root: unlimited
    (cpu / "cpu.cfs_period_us").write_text("100000\n")
    (cpu / "job" / "cpu.cfs_quota_us").write_text("400000\n")  # four CPUs
    (cpu / "job" / "cpu.cfs_period_us").write_text("100000\n")
    (cpu / "job" / "task" / "cpu.cfs_quota_us").write_text("200000\n")  # two, tighter
    (cpu / "job" / "task" / "cpu.cfs_period_us").write_text("100000\n")
    (cpu / "job" / "task" / "cpu.stat").write_text("nr_throttled 3\nthrottled_time 2500000\n")
    monkeypatch.setattr(telemetry, "CGROUP_ROOT", root)
    monkeypatch.setattr(telemetry, "PROC", proc)
    membership = proc / "self" / "cgroup"
    telemetry.cgroup_layout.cache_clear()  # the layout is read once per process
    monkeypatch.setattr(membership.__class__, "write_text", _writing_clears(membership.__class__))
    # Pure v1: the process's own group and its ancestors, tightest quota wins.
    membership.write_text("4:memory:/job/task\n3:cpu,cpuacct:/job/task\n")
    assert telemetry.cgroup_quota() == ("v1", 2.0)
    membership.write_text("3:cpu,cpuacct:/job\n")
    assert telemetry.cgroup_quota() == ("v1", 4.0)
    membership.write_text("3:cpu,cpuacct:/\n")
    assert telemetry.cgroup_quota() == ("v1", None)  # only the unlimited root applies
    # Hybrid: a v2 hierarchy exists too, but the cpu controller is on v1.
    membership.write_text("0::/job\n3:cpu,cpuacct:/job/task\n")
    assert telemetry.cgroup_quota() == ("v1", 2.0)
    monkeypatch.setattr(telemetry.sys, "platform", "linux")
    pressure = telemetry.Pressure()
    membership.write_text("3:cpu,cpuacct:/job/task\n")
    assert telemetry.Pressure().throttled() == 2500  # v1 counts nanoseconds
    del pressure
    # Pure v2: cpu.max, walked up to the root.
    (root / "job").mkdir()
    (root / "job" / "cpu.max").write_text("300000 100000\n")
    (root / "cpu.max").write_text("max 100000\n")
    (root / "job" / "cpu.stat").write_text("usage_usec 5\nthrottled_usec 7\n")
    membership.write_text("0::/job\n")
    monkeypatch.setattr(telemetry, "CGROUP_ROOT", root)
    import shutil

    shutil.rmtree(cpu)
    assert telemetry.cgroup_quota() == ("v2", 3.0)
    assert telemetry.Pressure().throttled() == 7
    membership.write_text("0::/elsewhere\n")  # a namespace seen from outside: the mount
    assert telemetry.cgroup_quota() == ("v2", None)


DYNAMIC_HEAVY = """
import pathlib, time
import pytest
import pytest_timing

@pytest_timing.cpu(2)
@pytest.fixture(scope="session")
def heavy():
    started = time.time()
    time.sleep(0.3)
    with open("setups.log", "a") as f:  # every worker builds its own instance
        f.write(f"{started} {time.time()}\\n")
    yield

def test_1(request):
    request.getfixturevalue("heavy")
    time.sleep(0.05)

def test_2(request):
    request.getfixturevalue("heavy")
    time.sleep(0.05)

def test_light_1(): time.sleep(0.05)
def test_light_2(): time.sleep(0.05)
"""


@needs_xdist
def test_a_fixture_reached_at_run_time_is_admitted_before_its_setup(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_dyn=DYNAMIC_HEAVY)
    result = run_timing(pytester, "-n", "2", "--timing-cpus", "2")
    result.assert_outcomes(passed=4)
    doc = load_json(pytester.path)
    setups = [
        tuple(float(x) for x in line.split())
        for line in (pytester.path / "setups.log").read_text().splitlines()
    ]
    assert setups  # somebody paid the set-up, and it was declared nowhere static
    records = {t["nodeid"].split("::")[-1]: t["cpu"] for t in doc["tests"]}
    assert records["test_1"]["demand"] == 1  # declared nothing: the request raised it
    # No two set-ups ran at the same time: each one asked for the budget's two
    # slots and got them only once the other worker had let go of its own.
    for i, (a_start, a_stop) in enumerate(setups):
        for b_start, b_stop in setups[i + 1 :]:
            assert a_stop <= b_start + 0.001 or b_stop <= a_start + 0.001, setups
    assert doc["run"]["cpu"]["domains"]["local"]["forced"] == 0
    if len(setups) == 2:
        assert doc["run"]["cpu"]["waited_tests"] >= 1
