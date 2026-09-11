"""Duration-aware xdist scheduling: estimates, the scheduler, and the plugin contract."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import (
    T0,
    Collector,
    full_test,
    load_json,
    make_info,
    needs_xdist,
    report,
)

from pytest_timing.schedule import Estimates

DB = "session::conftest.py::db[]"
M1 = "module:tests/test_m1.py:tests/test_m1.py::mod[]"


def _run_with(*tests: tuple[str, float, str]) -> Any:
    c = Collector(make_info())
    at = 0.0
    for nodeid, call, outcome in tests:
        full_test(c, nodeid, at, setup=0.0, call=call, teardown=0.0, outcome=outcome)
        at += call + 0.01
    return c.finish(T0 + at, termination="finished")


def test_estimate_is_longest_attempt_and_default_is_mean() -> None:
    run = _run_with(
        ("t.py::slow", 2.0, "passed"),
        ("t.py::flaky", 0.5, "failed"),
        ("t.py::flaky", 1.0, "passed"),  # a retry: keep the longer attempt
        ("t.py::fast", 0.1, "passed"),
    )
    est = Estimates.from_run(run)
    assert est.durations == pytest.approx(
        {"t.py::slow": 2.0, "t.py::flaky": 1.0, "t.py::fast": 0.1}, abs=1e-5
    )
    assert est.default == pytest.approx(3.1 / 3, abs=1e-5)
    assert est.estimate("t.py::new") == pytest.approx(3.1 / 3, abs=1e-5)
    assert est.known(["t.py::slow", "t.py::flaky", "t.py::fast", "t.py::new"]) == 3


def test_crashed_spans_carry_no_clock_and_are_unknown() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::ok", 0.0, call=0.3)
    c.add_report(report("t.py::victim", "setup", 0.5, 0.6, worker="gw1"))
    c.worker_down("gw1", T0 + 5.0, "crashed")  # victim's span is closed as crashed at 5.0
    run = c.finish(T0 + 5.0, termination="finished")
    est = Estimates.from_run(run)
    assert "t.py::victim" not in est.durations
    assert est.estimate("t.py::victim") == pytest.approx(0.5)  # ok's setup + call + teardown


def test_own_duration_excludes_shared_setup_and_records_families() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::first", 0.0, setup=2.1, call=0.5, teardown=0.1)
    c.tests[-1].fixtures = {DB: 2.0, M1: None}  # 2 s of the setup phase was db
    full_test(c, "t.py::second", 3.0, setup=0.1, call=0.5, teardown=0.1)
    c.tests[-1].fixtures = {DB: None, M1: 0.4}
    run = c.finish(T0 + 4, termination="finished")
    est = Estimates.from_run(run)
    assert est.durations == pytest.approx({"t.py::first": 0.7, "t.py::second": 0.3}, abs=1e-5)
    assert est.families == {"t.py::first": frozenset({DB, M1}), "t.py::second": frozenset({DB, M1})}
    assert est.setups == pytest.approx({DB: 2.0, M1: 0.4})
    assert est.family("t.py::unknown") == frozenset()


def test_load_rejects_non_runs(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        Estimates.load(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError):
        Estimates.load(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(ValueError):
        Estimates.load(tmp_path / "list.json")
    (tmp_path / "other.json").write_text(json.dumps({"schema": 1, "tests": []}))
    with pytest.raises(ValueError):
        Estimates.load(tmp_path / "other.json")


def test_load_reads_a_saved_run(tmp_path: pathlib.Path) -> None:
    run = _run_with(("t.py::a", 0.4, "passed"))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run.to_dict()))
    est = Estimates.load(path)
    assert est.durations == pytest.approx({"t.py::a": 0.4})
    assert est.source == str(path)


from pytest_timing.schedule import Costs, Lane, plan, transfer  # noqa: E402


def costs_for(
    tests: dict[str, tuple[float, list[str]]], setups: dict[str, float]
) -> tuple[Costs, list[str]]:
    """``tests`` maps nodeid to (own seconds, fixture keys)."""
    ids = list(tests)
    est = Estimates(
        {k: v[0] for k, v in tests.items()},
        1.0,
        "",
        {k: frozenset(v[1]) for k, v in tests.items()},
        dict(setups),
    )
    return Costs(est, ids), ids


def names(ids: list[str], indices: list[int]) -> list[str]:
    return [ids[i] for i in indices]


M2 = "module:tests/test_m2.py:tests/test_m2.py::mod[]"


def test_queue_cost_pays_sticky_once_and_module_per_run() -> None:
    costs, ids = costs_for(
        {"a": (1.0, [DB, M1]), "b": (1.0, [DB, M1]), "c": (1.0, [DB, M2]), "d": (1.0, [M1])},
        {DB: 5.0, M1: 1.0, M2: 1.0},
    )
    lane = Lane(plan=[0, 1, 2, 3])
    # db once; m1 for a (b continues it); m2 for c; m1 again for d after leaving it.
    assert lane.finish(costs) == pytest.approx(4 + 5 + 1 + 1 + 1)
    assert lane.without_tail(1).finish(costs) == pytest.approx(3 + 5 + 1 + 1)
    assert Lane(fixtures={DB, M1}, plan=[0]).finish(costs) == 1.0


def test_plan_keeps_module_families_whole() -> None:
    tests = {f"m{m}t{t}": (1.0, [f"module:m{m}:m{m}::mod[]"]) for m in range(4) for t in range(4)}
    setups = {f"module:m{m}:m{m}::mod[]": 1.0 for m in range(4)}
    costs, ids = costs_for(tests, setups)
    lanes = [Lane(), Lane()]
    plan(costs, range(len(ids)), lanes)
    for lane in lanes:
        modules = {ids[i][:2] for i in lane.plan}
        assert len(modules) == 2 and len(lane.plan) == 8
        assert lane.finish(costs) == pytest.approx(10.0)  # two set-ups, eight tests


def test_plan_splits_a_family_when_the_split_finishes_sooner() -> None:
    tests = {f"t{i}": (1.0, [DB]) for i in range(20)}
    costs, ids = costs_for(tests, {DB: 5.0})
    lanes = [Lane() for _ in range(4)]
    plan(costs, range(20), lanes)
    assert [len(lane.plan) for lane in lanes] == [5, 5, 5, 5]
    assert all(lane.finish(costs) == pytest.approx(10.0) for lane in lanes)


def test_plan_keeps_separate_expensive_families_apart() -> None:
    """Two families with different 100 s session fixtures, 20 x 1 s tests each."""
    tests = {f"a{i}": (1.0, ["session:::a::fa[]"]) for i in range(20)}
    tests |= {f"b{i}": (1.0, ["session:::b::fb[]"]) for i in range(20)}
    costs, ids = costs_for(tests, {"session:::a::fa[]": 100.0, "session:::b::fb[]": 100.0})
    lanes = [Lane(), Lane()]
    plan(costs, range(40), lanes)
    assert sorted(len(lane.plan) for lane in lanes) == [20, 20]
    assert all(lane.finish(costs) == pytest.approx(120.0) for lane in lanes)
    assert {ids[i][0] for i in lanes[0].plan} != {ids[i][0] for i in lanes[1].plan}
    # One such family alone is worth splitting: 110 s beats 120 s.
    lanes = [Lane(), Lane()]
    plan(costs, [i for i in range(40) if ids[i][0] == "a"], lanes)
    assert all(lane.finish(costs) == pytest.approx(110.0) for lane in lanes)


def test_plan_orders_fixture_bound_work_first_and_cheap_tests_last() -> None:
    tests = {f"free{i}": (0.1 * (i + 1), []) for i in range(6)}
    tests |= {f"m{i}": (0.5, [M1]) for i in range(3)}
    costs, ids = costs_for(tests, {M1: 2.0})
    lanes = [Lane(), Lane()]
    plan(costs, range(len(ids)), lanes)
    for lane in lanes:
        kinds = [ids[i][0] for i in lane.plan]
        assert kinds == sorted(kinds, key=lambda k: k != "m")  # module tests first
        free = [costs.own[i] for i in lane.plan if not costs.family[i]]
        assert free == sorted(free, reverse=True)  # cheapest at the tail


def test_queue_cost_pays_a_parametrized_sticky_fixture_again_after_a_switch() -> None:
    cfg0 = "session::conftest.py::cfg[0]"
    cfg1 = "session::conftest.py::cfg[1]"
    costs, ids = costs_for(
        {"a": (1.0, [cfg0]), "b": (1.0, [cfg1]), "c": (1.0, [cfg0]), "d": (1.0, [DB])},
        {cfg0: 5.0, cfg1: 5.0, DB: 2.0},
    )
    # cfg[0], then cfg[1] replaces it, then cfg[0] is set up once more; db is unrelated.
    assert Lane(plan=[0, 1, 2, 3]).finish(costs) == pytest.approx(4 + 5 + 5 + 5 + 2)
    assert Lane(plan=[0, 2, 1, 3]).finish(costs) == pytest.approx(4 + 5 + 5 + 2)
    lane = Lane()
    lane.dispatch(costs.charge(0, lane.fixtures))
    lane.dispatch(costs.charge(3, lane.fixtures))
    assert lane.fixtures == {cfg0, DB}
    lane.dispatch(costs.charge(1, lane.fixtures))
    assert lane.fixtures == {cfg1, DB}


@pytest.mark.parametrize("origin", [0.0, 1e5])
def test_plan_and_balance_judge_gains_on_time_left_not_on_the_clock(origin: float) -> None:
    # Three equal tests behind a cheap session fixture: split three ways whatever
    # the clock reads, then the same for the single-test moves of the balance pass.
    costs, ids = costs_for({f"t{i}": (8.0, [DB]) for i in range(3)}, {DB: 0.01})
    lanes = [Lane(free=origin) for _ in range(3)]
    plan(costs, range(3), lanes)
    assert [lane.finish(costs) - origin for lane in lanes] == pytest.approx([8.01] * 3)
    costs, ids = costs_for({f"t{i}": (float(9 - i), []) for i in range(9)}, {})
    lanes = [Lane(free=origin) for _ in range(3)]
    plan(costs, range(9), lanes)
    assert max(lane.finish(costs) - origin for lane in lanes) <= 16


def test_plan_splits_a_cpu_heavy_setup_less_under_a_budget() -> None:
    """A 0.5 s set-up needing 4 slots, 40 x 0.03 s tests, 8 lanes.

    By lane time alone the family is best spread over all 8 lanes (0.5 + 0.15 each);
    against an 8-slot budget those 8 set-ups are 16 slot-seconds of work that no
    lane time hides, so fewer lanes finish sooner.
    """
    from pytest_timing.demand import Declarations

    ids = [f"t{i}" for i in range(40)]
    est = Estimates(
        dict.fromkeys(ids, 0.03), 0.03, "", {i: frozenset({DB}) for i in ids}, {DB: 0.5}
    )
    decl = Declarations(tests=[1] * 40, fixtures={DB.rpartition("[")[0]: (4, 0)})
    costs = Costs(est, ids, decl)
    lanes = [Lane() for _ in range(8)]
    plan(costs, range(40), lanes)
    assert sum(1 for lane in lanes if lane.plan) == 8
    lanes = [Lane() for _ in range(8)]
    plan(costs, range(40), lanes, budget=8)
    used = sum(1 for lane in lanes if lane.plan)
    assert 2 <= used <= 4
    assert sorted(i for lane in lanes for i in lane.plan) == list(range(40))
    # An idle lane does not take a tail chunk either: the set-up it would pay is
    # four slots of work for a few hundredths of a second of tests.
    idle = next(lane for lane in lanes if not lane.plan)
    moved = transfer(costs, lanes, idle, budget=8)
    assert not moved or moved == idle.plan and sum(1 for lane in lanes if lane.plan) == used
    lanes = [Lane() for _ in range(8)]
    plan(costs, range(40), lanes, budget=8)
    idle = next(lane for lane in lanes if not lane.plan)
    moved = transfer(costs, lanes, idle)  # without the budget a split looks free
    assert moved and sum(1 for lane in lanes if lane.plan) == used + 1


def test_transfer_prefers_a_warm_destination_and_refuses_a_useless_cold_one() -> None:
    tests = {f"t{i}": (1.0, [DB]) for i in range(10)}
    costs, ids = costs_for(tests, {DB: 5.0})
    donor = Lane(fixtures={DB}, plan=list(range(10)))
    warm = Lane(fixtures={DB})
    cold = Lane()
    lanes = [donor, warm, cold]
    moved = transfer(costs, lanes, warm)
    assert len(moved) == 5 and warm.plan == moved and len(donor.plan) == 5
    # The cold lane would need 5 s of set-up to shorten a 5 s queue: nothing to gain.
    assert transfer(costs, lanes, cold) == []
    assert cold.plan == [] and len(donor.plan) == 5


def test_transfer_takes_the_best_chunk_and_accepts_a_non_worsening_move() -> None:
    tests = {f"t{i}": (1.0, []) for i in range(4)}
    costs, ids = costs_for(tests, {})
    busy = Lane(free=0.5, plan=[0, 1, 2, 3])  # finishes at 4.5
    idle = Lane(free=0.6)
    assert transfer(costs, [busy, idle], idle) == [2, 3]  # half: 2.5 and 2.6
    assert busy.plan == [0, 1]
    busy = Lane(free=0.5, plan=[0, 1, 2, 3])
    later = Lane(free=3.5)
    # One test makes it 3.5 and 4.5: no predicted gain, but nothing lost, so taken.
    assert transfer(costs, [busy, later], later) == [3]
    assert transfer(costs, [busy, later], later) == []  # a second would reach 5.5


class FakeNode:
    """The slice of ``WorkerController`` a scheduler touches."""

    def __init__(self, ident: str, *, live: bool = False) -> None:
        self.gateway = SimpleNamespace(id=ident)
        self.sent: list[list[int]] = []
        self.shutting_down = False
        self.steal_requests: list[list[int]] = []
        self.commands: list[tuple[str, dict[str, Any]]] = []
        if live:  # Otherwise simulate an unavailable command channel.
            self.sendcommand = lambda name, **payload: self.commands.append((name, payload))

    def send_steal(self, indices: list[int]) -> None:
        self.steal_requests.append(list(indices))

    def send_runtest_some(self, indices: list[int]) -> None:
        self.sent.append(list(indices))

    def shutdown(self) -> None:
        self.shutting_down = True

    @property
    def queued(self) -> list[int]:
        return [i for batch in self.sent for i in batch]


class FakeConfig:
    def __init__(self, workers: int, maxschedchunk: int | None = None) -> None:
        self._options = {"tx": [f"{workers}*popen"], "maxschedchunk": maxschedchunk}

    def getoption(self, name: str, default: Any = None) -> Any:
        return self._options.get(name, default)

    getvalue = getoption


def make_scheduler(
    estimates: Estimates, workers: int = 2, stealing: bool = False, cpu: Any = None, **kwargs: Any
) -> Any:
    from pytest_timing.xdist_scheduler import DurationScheduling

    sched = DurationScheduling(FakeConfig(workers, **kwargs), None, estimates, stealing, cpu)  # type: ignore[arg-type]
    sched.clock = lambda: sched.now
    sched.now = 0.0
    return sched


def start(
    sched: Any, collection: list[str], workers: int = 2, *, live: bool = False
) -> list[FakeNode]:
    nodes = [FakeNode(f"gw{i}", live=live) for i in range(workers)]
    for node in nodes:
        sched.add_node(node)
        sched.add_node_collection(node, collection)
    assert sched.collection_is_completed
    sched.schedule()
    return nodes


@needs_xdist
def test_initial_dispatch_follows_the_plan_and_keeps_two_queued() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    est = Estimates(dict(zip(ids, (6, 5, 4, 3, 2, 1), strict=True)))
    sched = make_scheduler(est)
    gw0, gw1 = start(sched, ids)
    # Planned longest-first onto the least loaded lane: [a, d, e] and [b, c, f].
    # Each worker gets its first two; the rest stays on the controller.
    assert [ids[i] for i in gw0.queued] == ["a", "d"]
    assert [ids[i] for i in gw1.queued] == ["b", "c"]
    assert [ids[i] for i in sched.pending] == ["e", "f"]
    assert sched.known == 6
    assert not gw0.shutting_down and not gw1.shutting_down


@needs_xdist
def test_worker_that_runs_dry_takes_from_another_plan() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    est = Estimates(dict(zip(ids, (6, 5, 4, 3, 2, 1), strict=True)))
    sched = make_scheduler(est)
    gw0, gw1 = start(sched, ids)
    sched.now = 1.0
    sched.mark_test_complete(gw1, ids.index("b"))  # early: b was estimated at 5
    assert [ids[i] for i in gw1.sent[-1]] == ["f"]  # its own plan first
    sched.now = 2.0
    sched.mark_test_complete(gw1, ids.index("c"))
    # gw1's plan is empty and gw0 is busy until 9: take e from gw0's plan.
    assert [ids[i] for i in gw1.sent[-1]] == ["e"]
    assert sched.pending == []
    sched.now = 3.0
    sched.mark_test_complete(gw1, ids.index("f"))
    assert gw1.shutting_down
    assert not gw0.shutting_down
    sched.now = 6.0
    sched.mark_test_complete(gw0, ids.index("a"))
    assert gw0.shutting_down


@needs_xdist
def test_module_family_stays_on_one_worker_for_the_whole_run() -> None:
    ids = [f"m{m}t{t}" for m in range(2) for t in range(4)]
    keys = {f"m{m}": f"module:m{m}:m{m}::mod[]" for m in range(2)}
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        1.0,
        "",
        {i: frozenset({keys[i[:2]]}) for i in ids},
        {k: 1.0 for k in keys.values()},
    )
    sched = make_scheduler(est)
    nodes = start(sched, ids)
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(8), 1.0))
    for order in ran.values():
        assert len({ids[i][:2] for i in order}) == 1  # one module per worker
        assert len(order) == 4


@needs_xdist
def test_idle_worker_shuts_down_rather_than_pay_a_useless_setup() -> None:
    ids = ["x", "y"]
    est = Estimates(
        {"x": 0.05, "y": 0.05}, 0.05, "", {"x": frozenset({DB}), "y": frozenset({DB})}, {DB: 5.0}
    )
    sched = make_scheduler(est)
    gw0, gw1 = start(sched, ids)
    assert sorted(ids[i] for i in gw0.queued) == ["x", "y"]
    assert gw1.queued == [] and gw1.shutting_down
    assert gw0.shutting_down  # everything is dispatched: nothing more to wait for


@needs_xdist
def test_tiny_tests_are_batched_with_hysteresis() -> None:
    ids = [f"t{i}" for i in range(400)]
    est = Estimates(dict.fromkeys(ids, 0.001))  # 0.4 s of work, 2 workers
    sched = make_scheduler(est)
    gw0, gw1 = start(sched, ids)
    # Budget is min(0.1, 0.4 / 20) = 0.02 s: 20 tests per worker up front.
    assert sched.budget == pytest.approx(0.02)
    assert len(gw0.queued) == 20 and len(gw1.queued) == 20
    for index in gw0.queued[:10]:
        sched.mark_test_complete(gw0, index)
    assert len(gw0.sent) == 1  # still at half the budget: no refill yet
    sched.mark_test_complete(gw0, gw0.queued[10])
    assert len(gw0.sent) == 2 and len(gw0.sent[-1]) == 11  # back up to the budget


@needs_xdist
def test_maxschedchunk_caps_a_refill_but_keeps_two_pending() -> None:
    ids = [f"t{i}" for i in range(50)]
    est = Estimates(dict.fromkeys(ids, 0.001))
    sched = make_scheduler(est, maxschedchunk=1)
    gw0, gw1 = start(sched, ids)
    for index in gw0.queued[:3]:
        sched.mark_test_complete(gw0, index)
    assert len(sched.node2pending[gw0]) >= 2
    assert all(len(batch) <= 2 for batch in gw0.sent[1:])


@needs_xdist
def test_crashed_worker_work_is_replanned_and_a_restart_joins() -> None:
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    est = Estimates(dict(zip(ids, (8, 7, 6, 5, 4, 3, 2, 1), strict=True)))
    sched = make_scheduler(est, workers=3)
    gw0, gw1, gw2 = start(sched, ids, workers=3)
    crashitem = sched.remove_node(gw0)
    assert crashitem == ids[gw0.queued[0]]  # the test it was running
    survivor = gw0.queued[1]
    assert survivor in sched.pending  # back in the pool, planned onto another lane
    assert survivor in sched.lanes[gw1].plan + sched.lanes[gw2].plan
    gw3 = FakeNode("gw3")
    sched.add_node(gw3)
    sched.add_node_collection(gw3, ids)
    sched.schedule()
    assert len(gw3.queued) >= 1  # took work from another lane's plan
    ran = run_to_completion(
        sched, [gw1, gw2, gw3], dict(zip(range(8), (8, 7, 6, 5, 4, 3, 2, 1), strict=True))
    )
    assert sorted(i for order in ran.values() for i in order) == sorted(
        set(range(8)) - {ids.index(crashitem)}
    )


@needs_xdist
def test_empty_collection_and_fewer_tests_than_workers() -> None:
    sched = make_scheduler(Estimates(), workers=3)
    gw0, gw1, gw2 = start(sched, [], workers=3)
    assert sched.pending == [] and sched.collection == []
    sched = make_scheduler(Estimates({"x": 1.0, "y": 1.0}), workers=3)
    gw0, gw1, gw2 = start(sched, ["x", "y"], workers=3)
    assert sorted(len(n.queued) for n in (gw0, gw1, gw2)) == [0, 1, 1]
    assert gw0.shutting_down and gw1.shutting_down and gw2.shutting_down


SLOW_LAST = """
import time

def test_fast_1(): time.sleep(0.01)
def test_fast_2(): time.sleep(0.01)
def test_fast_3(): time.sleep(0.01)
def test_fast_4(): time.sleep(0.01)
def test_fast_5(): time.sleep(0.01)
def test_fast_6(): time.sleep(0.01)
def test_slow(): time.sleep(0.3)
"""


def _starts(doc: dict[str, Any]) -> dict[str, float]:
    return {t["nodeid"].split("::")[1]: t["start"] for t in doc["tests"]}


def _first_on_its_worker(doc: dict[str, Any], name: str) -> bool:
    test = next(t for t in doc["tests"] if t["nodeid"].endswith("::" + name))
    same_worker = [t for t in doc["tests"] if t["worker"] == test["worker"]]
    return min(same_worker, key=lambda t: t["start"]) is test


@needs_xdist
def test_second_run_starts_the_slow_test_first(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_slow_last=SLOW_LAST)
    args = ["-n", "2", "--timing-json", "--timing-schedule", "pytest-timing.json"]
    args += ["-p", "no:cacheprovider"]
    first = pytester.runpytest_subprocess(*args)
    first.assert_outcomes(passed=7)
    first.stdout.fnmatch_lines(["schedule: no run recorded at pytest-timing.json yet; not applied"])
    doc = load_json(pytester.path)
    starts = _starts(doc)
    assert starts["test_slow"] == max(starts.values())  # collection order: it went last
    assert not _first_on_its_worker(doc, "test_slow")

    second = pytester.runpytest_subprocess(*args)
    second.assert_outcomes(passed=7)
    second.stdout.fnmatch_lines(
        ["schedule: 7 of 7 tests had recorded durations in pytest-timing.json"]
    )
    assert _first_on_its_worker(load_json(pytester.path), "test_slow")


@needs_xdist
def test_schedule_path_and_new_tests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_slow_last=SLOW_LAST)
    pytester.runpytest_subprocess(
        "-n", "2", "--timing-json-file", "runs/last.json", "-p", "no:cacheprovider"
    )
    pytester.makepyfile(test_extra="def test_new(): pass\n")
    result = pytester.runpytest_subprocess(
        "-n", "2", "--timing-schedule", "runs/last.json", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=8)
    result.stdout.fnmatch_lines(["schedule: 7 of 8 tests had recorded durations in runs/last.json"])
    assert "timing report" in result.stdout.str()  # implies --timing


@needs_xdist
def test_ini_and_env_enable_scheduling(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makepyfile(test_slow_last=SLOW_LAST)
    pytester.makeini("[pytest]\ntiming_json = true\ntiming_schedule = pytest-timing.json\n")
    pytester.runpytest_subprocess("-n", "2", "-p", "no:cacheprovider")
    result = pytester.runpytest_subprocess("-n", "2", "-p", "no:cacheprovider")
    result.stdout.fnmatch_lines(["schedule: 7 of 7 tests had recorded durations in *"])
    monkeypatch.setenv("PYTEST_TIMING_SCHEDULE", "elsewhere.json")
    result = pytester.runpytest_subprocess("-n", "2", "-p", "no:cacheprovider")
    result.stdout.fnmatch_lines(["schedule: no run recorded at elsewhere.json yet; not applied"])


@needs_xdist
def test_unreadable_file_and_unsupported_dist_fall_back(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_slow_last=SLOW_LAST)
    (pytester.path / "pytest-timing.json").write_text("{oops")
    result = pytester.runpytest_subprocess(
        "-n", "2", "--timing-schedule", "pytest-timing.json", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=7)
    result.stdout.fnmatch_lines(["schedule: could not read pytest-timing.json (*); not applied"])
    result = pytester.runpytest_subprocess(
        "-n", "2", "--dist", "loadscope", "--timing-schedule", "x.json", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=7)
    result.stdout.fnmatch_lines(["schedule: --dist=loadscope is not supported; not applied"])


def test_without_xdist_scheduling_is_a_noop_with_a_note(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_slow_last=SLOW_LAST)
    result = pytester.runpytest_subprocess("--timing-schedule", "x.json", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=7)
    result.stdout.fnmatch_lines(["schedule: needs pytest-xdist workers (-n); not applied"])
    assert not (pytester.path / "pytest-timing.json").exists()


@needs_xdist
def test_worksteal_dispatches_whole_plans_up_front() -> None:
    ids = [f"t{i}" for i in range(40)]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    assert len(gw0.sent) == 1 and len(gw1.sent) == 1  # one command each
    assert sorted(gw0.queued + gw1.queued) == list(range(40))
    assert sched.pending == [] and not gw0.shutting_down and not gw1.shutting_down


@needs_xdist
def test_worker_down_to_its_last_test_steals_a_tail_chunk() -> None:
    ids = [f"t{i}" for i in range(40)]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    # gw1 races through its share while gw0 is stuck on its first test, well past
    # the time it was projected to be done: it is late, and its queue is worth taking.
    for index in gw1.queued[:-2]:
        sched.now += 0.05
        sched.mark_test_complete(gw1, index)
    assert gw0.steal_requests == []  # gw1 is not low yet
    sched.now = 1.5
    sched.mark_test_complete(gw1, gw1.queued[-2])
    (request,) = gw0.steal_requests
    assert sched.steal_requested_from_node is gw0
    # Never the running test, the next one, or the one after that (in-flight margin).
    assert request == gw0.queued[len(gw0.queued) - len(request) :]
    assert len(request) <= len(gw0.queued) - 3
    assert len(request) >= 2  # at least a budget's worth, not one tiny test
    # The donor hands them back: they go straight to the waiting worker.
    sched.remove_pending_tests_from_node(gw0, request)
    assert gw1.sent[-1] == request
    assert sched.node2pending[gw1][-len(request) :] == request
    assert all(i not in sched.node2pending[gw0] for i in request)
    assert sched.steal_requested_from_node is None and sched.starving == []


@needs_xdist
def test_refused_steal_is_retried_and_a_hopeless_one_shuts_the_worker_down() -> None:
    ids = [f"t{i}" for i in range(12)]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    for index in gw1.queued[:-2]:
        sched.now += 0.05
        sched.mark_test_complete(gw1, index)
    sched.now = 2.0  # gw0 is late by now
    sched.mark_test_complete(gw1, gw1.queued[-2])
    first = gw0.steal_requests[-1]
    sched.remove_pending_tests_from_node(gw0, [])  # the donor had moved on: refused
    assert len(gw0.steal_requests) == 2  # asked again right away
    assert not gw1.shutting_down
    # Now the donor drains to its last three tests: nothing left to take.
    for index in gw0.queued[:-3]:
        sched.now += 0.05
        sched.mark_test_complete(gw0, index)
    sched.remove_pending_tests_from_node(gw0, [])
    assert gw1.shutting_down  # free to run its last test
    assert first  # (the first request was a real chunk)


@needs_xdist
def test_stealing_pays_the_fixture_only_when_worth_it() -> None:
    ids = [f"m{i}" for i in range(8)] + [f"f{i}" for i in range(8)]
    est = Estimates(
        dict.fromkeys(ids, 0.5), 0.5, "", {i: frozenset({DB}) for i in ids[:8]}, {DB: 10.0}
    )
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    holder = gw0 if any(ids[i].startswith("m") for i in gw0.queued) else gw1
    other = gw1 if holder is gw0 else gw0
    assert all(ids[i].startswith("m") for i in holder.queued)  # the family stays whole
    for index in other.queued[:-1]:
        sched.now += 0.5
        sched.mark_test_complete(other, index)
    # Taking db tests would cost the thief 10 s of set-up for at most 2.5 s of tests.
    assert holder.steal_requests == []
    assert other.shutting_down


@needs_xdist
def test_stealing_restores_the_fixture_state_before_the_returned_tail() -> None:
    ids = [f"tests/test_m1.py::test_{i}" for i in range(4)] + ["tests/test_m2.py::test_tail"]
    m2 = "module:tests/test_m2.py:tests/test_m2.py::mod[]"
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        families={
            **{nodeid: frozenset({DB, M1}) for nodeid in ids[:-1]},
            ids[-1]: frozenset({DB, m2}),
        },
        setups={DB: 0.1, M1: 2.0, m2: 0.5},
    )
    sched = make_scheduler(est, workers=1, stealing=True)
    (donor,) = start(sched, ids, workers=1)
    assert donor.queued == list(range(5))
    receiver = FakeNode("replacement")
    sched.add_node(receiver)
    sched.starving.append(receiver)
    sched.steal_requested_from_node = donor
    sched.remove_pending_tests_from_node(donor, [4])
    assert sched.lanes[donor].fixtures == {DB, M1}
    # The returned fixture was only projected, never initialized on the donor.
    assert sched.costs.charge(4, sched.lanes[donor].fixtures).seconds == 1.5
    assert sched.dispatched[receiver][4].seconds == 1.6  # its session fixture is cold too


@needs_xdist
def test_worksteal_is_not_finished_while_a_steal_is_in_flight() -> None:
    ids = [f"t{i}" for i in range(20)]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    for index in gw1.queued[:-2]:
        sched.now += 0.05
        sched.mark_test_complete(gw1, index)
    sched.now = 2.0  # gw0 is late by now
    sched.mark_test_complete(gw1, gw1.queued[-2])
    assert sched.steal_requested_from_node is gw0
    assert not sched.tests_finished
    crashitem = sched.remove_node(gw0)  # the donor dies mid-steal
    assert crashitem == ids[gw0.queued[0]]
    assert sched.steal_requested_from_node is None
    assert gw1.sent[-1]  # its work went to the survivor


@needs_xdist
def test_restart_after_the_last_worker_died_gets_the_remaining_work() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, workers=1)
    (gw0,) = start(sched, ids, workers=1)
    sched.mark_test_complete(gw0, gw0.queued[0])
    crashitem = sched.remove_node(gw0)
    assert crashitem == ids[gw0.queued[1]]
    assert sched.pending and sched.lanes == {}  # nowhere to plan it yet
    gw1 = FakeNode("gw1")
    sched.add_node(gw1)
    sched.add_node_collection(gw1, ids)
    sched.schedule()
    assert not gw1.shutting_down or sched.pending == []
    ran = run_to_completion(sched, [gw1], dict.fromkeys(range(6), 0.05))
    assert sorted(ran["gw1"]) == sorted(set(range(6)) - {gw0.queued[0], gw0.queued[1]})
    assert sched.pending == [] and sched.tests_finished


@needs_xdist
def test_worker_late_on_one_test_still_gives_up_its_unstarted_work() -> None:
    ids = [f"t{i}" for i in range(40)]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    # gw0 finishes one test on time and then hangs on the next; gw1 runs dry
    # shortly before gw0's whole queue was projected to be done. gw0 cannot be free
    # before now plus what it has not started, so its tail is worth taking.
    sched.mark_test_complete(gw0, gw0.queued[0], duration=0.05)
    for index in gw1.queued[:-2]:
        sched.now += 0.05
        sched.mark_test_complete(gw1, index)
    sched.now = 0.95
    sched.mark_test_complete(gw1, gw1.queued[-2])
    assert gw0.steal_requests, "the idle worker gave up instead of stealing"
    assert not gw1.shutting_down


@needs_xdist
def test_a_single_stolen_test_does_not_leave_its_taker_waiting() -> None:
    ids = [f"t{i}" for i in range(12)]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = make_scheduler(est, stealing=True)
    gw0, gw1 = start(sched, ids)
    for index in gw1.queued[:-1]:
        sched.now += 0.05
        sched.mark_test_complete(gw1, index)
    sched.now = 2.0
    sched.mark_test_complete(gw1, gw1.queued[-1])  # gw1 is empty now, gw0 is late
    (request,) = gw0.steal_requests
    sched.remove_pending_tests_from_node(gw0, request[-1:])  # hand over one test only
    assert sched.node2pending[gw1] == request[-1:]
    # It either asked for more right away or was released to run what it holds.
    assert len(gw0.steal_requests) == 2 or gw1.shutting_down


@needs_xdist
def test_worksteal_mode_schedules_and_notes_the_mode(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_slow_last=SLOW_LAST)
    args = ["-n", "2", "--dist", "worksteal", "--timing-json", "--timing-schedule"]
    args += ["pytest-timing.json", "-p", "no:cacheprovider"]
    pytester.runpytest_subprocess(*args).assert_outcomes(passed=7)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=7)
    result.stdout.fnmatch_lines(
        ["schedule (worksteal): 7 of 7 tests had recorded durations in pytest-timing.json"]
    )
    assert _first_on_its_worker(load_json(pytester.path), "test_slow")


from pytest_timing.demand import Declarations, key_base  # noqa: E402


def cpu_scheduler(
    est: Estimates,
    ids: list[str],
    workers: int = 2,
    budget: int | None = None,
    slots: dict[str, int] | None = None,
    fixtures: dict[str, tuple[int, int]] | None = None,
    families: dict[str, list[str]] | None = None,
    stealing: bool = False,
    domain_of: Any = None,
    pressure: Any = None,
    host_budget: int = 8,
    auto: bool = False,
) -> Any:
    from pytest_timing.xdist_scheduler import CpuSetup

    decl = Declarations(
        tests=[(slots or {}).get(nodeid, 1) for nodeid in ids],
        families={ids.index(n): list(keys) for n, keys in (families or {}).items()},
        fixtures=dict(fixtures or {}),
        host={"budget": host_budget, "cpus": host_budget},
    )
    cpu = CpuSetup(
        cpus="auto" if auto else budget,
        declarations=lambda node: decl,
        domain_of=domain_of or (lambda node: "local"),
        pressure=pressure,
    )
    return make_scheduler(est, workers, stealing, cpu=cpu)


def running(sched: Any, nodes: list[FakeNode], queues: dict[Any, list[int]]) -> list[FakeNode]:
    """Workers able to run their head: they know the next test, or are shut down."""
    return [n for n in nodes if queues[n] and (len(queues[n]) >= 2 or n.shutting_down)]


def run_to_completion(
    sched: Any, nodes: list[FakeNode], actual: dict[int, float]
) -> dict[str, list[int]]:
    """Drive recorded commands through completion and exit, checking every CPU domain."""
    queues = {node: list(sched.node2pending[node]) for node in nodes}  # what each holds now
    seen = {node: len(node.queued) for node in nodes}
    finishes: dict[FakeNode, float] = {}
    ran: dict[str, list[int]] = {node.gateway.id: [] for node in nodes}
    peak = 0
    alive = list(nodes)
    while True:
        # A worker told to shut down exits once its queue is empty, and its
        # fixtures (and what they held) go with it.
        for node in list(alive):
            if node.shutting_down and not queues[node] and node in sched.node2pending:
                assert sched.remove_node(node) is None
                alive.remove(node)
                for other in alive:  # an exit may send work to any worker
                    queues[other].extend(other.queued[seen[other] :])
                    seen[other] = len(other.queued)
        if not any(queues[n] for n in alive):
            left = list(sched.pending) + [i for n in alive for i in sched.lanes[n].plan]
            assert not left, f"work left but no worker can run it: {left}"
            break
        ready = running(sched, alive, queues)
        assert ready, "every worker is blocked waiting for a next test or a shutdown"
        for domain, admission in sched.admissions.items():
            need = {
                n: min(sched.dispatched[n][queues[n][0]].peak, admission.limit)
                for n in ready
                if sched.domain[n] == domain
            }
            load = sum(need.values())
            peak = max(peak, load)
            assert load <= admission.limit, (load, admission.limit)
            assert all(admission.held(n) >= slots for n, slots in need.items())
        for node in ready:
            finishes.setdefault(node, sched.now + actual[queues[node][0]])
        node = min(ready, key=finishes.__getitem__)
        index = queues[node].pop(0)
        ran[node.gateway.id].append(index)
        sched.now = finishes.pop(node)
        sched.mark_test_complete(node, index, duration=actual[index])
        for other in alive:  # a completion may send work to any worker
            queues[other].extend(other.queued[seen[other] :])
            seen[other] = len(other.queued)
    assert all(a.forced == 0 for a in sched.admissions.values())
    sched.peak = peak
    return ran


@needs_xdist
def test_gate_is_off_without_declarations_and_on_with_a_budget_or_a_heavy_test() -> None:
    ids = ["a", "b", "c", "d"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids)
    start(sched, ids)
    assert not sched.gated and sched.cpu_summary()["gated"] is False
    sched = cpu_scheduler(est, ids, budget=2)
    start(sched, ids)
    assert sched.gated and sched.admissions["local"].budget == 2
    sched = cpu_scheduler(est, ids, slots={"a": 2})
    start(sched, ids)
    assert sched.gated  # detected: never below the number of workers
    assert sched.admissions["local"].budget == 8
    sched = cpu_scheduler(est, ids, workers=3, slots={"a": 2}, host_budget=2)
    start(sched, ids, workers=3)
    assert sched.admissions["local"].budget == 3


@needs_xdist
def test_initial_admission_respects_the_budget_and_keeps_waiting_workers_alive() -> None:
    ids = ["h1", "h2", "l1", "l2", "l3", "l4", "l5", "l6"]
    est = Estimates({"h1": 5.0, "h2": 5.0, **dict.fromkeys(ids[2:], 1.0)})
    sched = cpu_scheduler(est, ids, workers=4, budget=4, slots={"h1": 4, "h2": 4})
    nodes = start(sched, ids, workers=4)
    admission = sched.admissions["local"]
    assert admission.used == 4  # one heavy test is running, nothing else fits
    ready = running(sched, nodes, {n: list(n.queued) for n in nodes})
    assert len(ready) == 1 and ids[ready[0].queued[0]].startswith("h")
    waiting = [n for n in nodes if n in sched.waiting]
    assert len(waiting) == 3
    assert all(len(n.queued) == 1 and not n.shutting_down for n in waiting)
    assert not sched.tests_finished
    ran = run_to_completion(sched, nodes, {i: est.estimate(ids[i]) for i in range(8)})
    assert sorted(i for order in ran.values() for i in order) == list(range(8))
    assert sched.peak == 4
    # Three initial waiters, plus the first heavy test's worker waiting for its tail.
    assert sorted(wait.seconds for wait in sched.waits) == [5.0, 5.0, 10.0, 10.0]
    assert sched.now == 12.0


@needs_xdist
def test_a_heavier_test_behind_lighter_ones_is_granted_only_when_it_is_the_head() -> None:
    # Tiny tests, so the batch would hold many: a module family, then the heavy
    # test, then hundreds of light ones. The batch stops at the heavy test.
    ids = ["m1", "m2", "m3", "m4", "h"] + [f"l{i}" for i in range(400)]
    families = {f"m{i}": frozenset({M1}) for i in range(1, 5)}
    est = Estimates(dict.fromkeys(ids, 0.001), 0.001, "", families, {M1: 0.001})
    sched = cpu_scheduler(est, ids, workers=1, budget=2, slots={"h": 2})
    (gw0,) = start(sched, ids, workers=1)
    admission = sched.admissions["local"]
    assert [ids[i] for i in gw0.queued] == ["m1", "m2", "m3", "m4", "h"]  # h is the tail
    assert admission.held(gw0) == 1
    for name in ("m1", "m2", "m3"):
        sched.mark_test_complete(gw0, ids.index(name))
        assert admission.held(gw0) == 1  # h is deferred, not granted
        assert gw0 not in sched.waiting and not gw0.shutting_down  # busy, not in line
        assert len(gw0.sent) == 1
    sched.mark_test_complete(gw0, ids.index("m4"))
    assert admission.held(gw0) == 2  # h is the head: granted, and the lights follow
    assert len(gw0.sent) == 2 and ids[gw0.sent[-1][0]] == "l0"
    sched.mark_test_complete(gw0, ids.index("h"))
    assert admission.held(gw0) == 1


@needs_xdist
def test_shutdown_waits_for_the_last_test_to_be_admitted() -> None:
    ids = ["h1", "h2"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=2, slots={"h1": 2, "h2": 2})
    nodes = start(sched, ids)
    admission = sched.admissions["local"]
    runner = next(n for n in nodes if n.shutting_down)
    waiter = next(n for n in nodes if not n.shutting_down)
    assert admission.held(runner) == 2 and waiter in sched.waiting
    assert sched.pending == [] and not sched.tests_finished
    sched.now = 1.0
    sched.mark_test_complete(runner, runner.queued[0])
    assert waiter.shutting_down and admission.held(waiter) == 2
    assert admission.held(runner) == 0
    assert waiter not in sched.waiting
    (record,) = sched.waits
    assert (record.worker, record.index, record.attempt, record.seconds) == (
        waiter.gateway.id,
        waiter.queued[0],
        0,
        1.0,
    )


@needs_xdist
def test_crash_releases_the_reservation_and_a_replacement_is_gated() -> None:
    ids = ["h1", "h2", "h3", "l1", "l2"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=3, budget=2, slots={"h1": 2, "h2": 2, "h3": 2})
    nodes = start(sched, ids, workers=3)
    admission = sched.admissions["local"]
    holder = next(n for n in nodes if admission.held(n) == 2)
    sched.remove_node(holder)
    assert holder not in admission.reserved and holder not in sched.waiting
    assert admission.used == 2  # the oldest waiter got the slots at once
    gw3 = FakeNode("gw3")
    sched.add_node(gw3)
    sched.add_node_collection(gw3, ids)
    sched.schedule()
    survivors = [n for n in nodes if n is not holder] + [gw3]
    actual = dict.fromkeys(range(5), 1.0)
    ran = run_to_completion(sched, survivors, actual)
    done = sorted(i for order in ran.values() for i in order)
    assert done == sorted(set(range(5)) - {holder.queued[0]})
    assert sched.peak <= 2


@needs_xdist
def test_fixture_setup_demand_is_paid_by_the_first_test_and_holds_stay() -> None:
    ids = ["a", "b", "c", "d"]
    est = Estimates(dict.fromkeys(ids, 1.0), 1.0, "", {i: frozenset({DB}) for i in ids}, {DB: 0.5})
    base = DB.rpartition("[")[0]
    sched = cpu_scheduler(est, ids, workers=2, budget=4, fixtures={base: (4, 0)})
    nodes = start(sched, ids)
    admission = sched.admissions["local"]
    assert sched.gated
    # The cheap set-up is paid on both workers; each first test reserves the 4 slots
    # of the cold set-up, each second one only its own slot. Two cold set-ups do not
    # fit the budget together: the second worker waits for the first to finish.
    demands = sorted([sched.dispatched[n][i].peak for i in n.queued] for n in nodes)
    assert demands == [[4], [4, 1]]  # the waiting worker was sent nothing more
    assert admission.used == 4 and len(sched.waiting) == 1
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(4), 1.0))
    assert sorted(i for order in ran.values() for i in order) == list(range(4))
    # With a hold, the instance keeps a slot busy for the rest of the worker's life.
    sched = cpu_scheduler(est, ids, workers=2, budget=8, fixtures={base: (2, 1)})
    nodes = start(sched, ids)
    holder = next(n for n in nodes if n.queued)
    assert sched.dispatched[holder][holder.queued[0]].peak == 3  # 2 for the set-up, 1 held
    assert sched.dispatched[holder][holder.queued[1]].peak == 2  # 1 own, 1 held
    sched.mark_test_complete(holder, holder.queued[0])
    assert sched.admissions["local"].held(holder) == 2
    sched.mark_test_complete(holder, holder.queued[1])
    assert sched.costs.hold(sched.lanes[holder].fixtures) == 1  # idle, it still holds one


@needs_xdist
def test_oversized_demand_is_clamped_to_the_limit_and_runs_alone() -> None:
    ids = ["huge", "l1", "l2", "l3"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=2, slots={"huge": 9})
    nodes = start(sched, ids)
    admission = sched.admissions["local"]
    assert admission.clamped >= 1
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(4), 1.0))
    assert sorted(i for order in ran.values() for i in order) == list(range(4))


@needs_xdist
def test_worksteal_dispatch_stops_before_a_rise_in_demand() -> None:
    ids = ["m1", "m2", "h", "l1", "l2"]
    est = Estimates(
        dict.fromkeys(ids, 1.0), 1.0, "", {"m1": frozenset({M1}), "m2": frozenset({M1})}, {M1: 0.5}
    )
    sched = cpu_scheduler(est, ids, workers=1, budget=2, slots={"h": 2}, stealing=True)
    (gw0,) = start(sched, ids, workers=1)
    admission = sched.admissions["local"]
    # The family, then h as the (ungranted) tail; the light tests stay in the plan.
    assert [ids[i] for i in gw0.queued] == ["m1", "m2", "h"]
    assert [ids[i] for i in sched.lanes[gw0].plan] == ["l1", "l2"]
    assert admission.held(gw0) == 1 and not gw0.shutting_down
    sched.mark_test_complete(gw0, ids.index("m1"))
    sched.mark_test_complete(gw0, ids.index("m2"))
    assert admission.held(gw0) == 2  # h granted once it was the head
    assert [ids[i] for i in gw0.queued[-2:]] == ["l1", "l2"]  # and the rest followed
    sched.mark_test_complete(gw0, ids.index("h"))
    assert admission.held(gw0) == 1
    sched.mark_test_complete(gw0, ids.index("l1"))
    assert gw0.shutting_down  # down to its last test, nothing to steal: released


@needs_xdist
def test_a_waiting_worker_neither_steals_nor_shuts_down_in_worksteal_mode() -> None:
    ids = ["h1", "h2", "l1", "l2", "l3", "l4"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=2, slots={"h1": 2, "h2": 2}, stealing=True)
    nodes = start(sched, ids)
    waiter = next(n for n in nodes if n in sched.waiting)
    runner = next(n for n in nodes if n is not waiter)
    assert waiter.steal_requests == [] and runner.steal_requests == []
    assert not waiter.shutting_down and sched.starving == []
    assert not sched.tests_finished
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(6), 1.0))
    assert sorted(i for order in ran.values() for i in order) == list(range(6))


@needs_xdist
def test_stolen_tests_reach_the_taker_through_the_gate() -> None:
    ids = [f"t{i}" for i in range(20)] + ["h"]
    est = Estimates(dict.fromkeys(ids, 0.05))
    sched = cpu_scheduler(est, ids, workers=2, budget=3, slots={"h": 2}, stealing=True)
    gw0, gw1 = start(sched, ids)
    admission = sched.admissions["local"]
    donor = gw0 if ids.index("h") in gw0.queued else gw1  # holds h: 2 of the 3 slots
    thief = gw1 if donor is gw0 else gw0
    assert admission.held(donor) == 2 and admission.held(thief) == 1
    done = list(thief.queued[:-1])
    for index in done[:-1]:
        sched.now += 0.05
        sched.mark_test_complete(thief, index)
    sched.now = 2.0  # the donor is late: its tail is worth taking
    sched.mark_test_complete(thief, done[-1])
    (request,) = donor.steal_requests
    sched.remove_pending_tests_from_node(donor, request)
    # The chunk went through the gate: the thief's head was admitted and the chunk
    # followed, or the thief is in line and the chunk sits in its plan.
    assert sched.node2pending[thief][-len(request) :] == request or thief in sched.waiting
    assert admission.used <= admission.limit
    ran = run_to_completion(sched, [gw0, gw1], dict.fromkeys(range(21), 0.05))
    assert sorted(done + [i for order in ran.values() for i in order]) == list(range(21))


class FakePressure:
    def __init__(self) -> None:
        self.some_value: float | None = None
        self.throttled_value: int | None = None

    def some(self) -> float | None:
        return self.some_value

    def throttled(self) -> int | None:
        return self.throttled_value


@needs_xdist
def test_governor_lowers_the_limit_under_pressure_and_waits_resume_on_recovery() -> None:
    ids = [f"t{i}" for i in range(40)]
    est = Estimates(dict.fromkeys(ids, 1.0))
    pressure = FakePressure()
    sched = cpu_scheduler(est, ids, workers=2, budget=2, pressure=pressure)
    nodes = start(sched, ids)
    admission = sched.admissions["local"]
    pressure.some_value = 0.9  # something else is eating the CPUs
    for _ in range(3):
        node = next(n for n in nodes if len(n.queued) >= 2 and not n.shutting_down)
        sched.now += 1.0
        sched.mark_test_complete(node, sched.node2pending[node][0])
    assert admission.limit == 1 and admission.lowered == 1
    # Running tests were not touched; the next admission had to wait for the slot.
    assert admission.used <= 2
    pressure.some_value = 0.0
    for _ in range(30):
        node = next(n for n in nodes if len(n.queued) >= 2 or n.shutting_down)
        sched.now += 1.0
        sched.mark_test_complete(node, sched.node2pending[node][0])
    assert admission.limit == 2
    assert admission.forced == 0


@needs_xdist
def test_each_domain_has_its_own_budget() -> None:
    ids = ["a", "b", "c", "d"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=1, domain_of=lambda node: node.gateway.id)
    nodes = start(sched, ids)
    assert set(sched.admissions) == {"gw0", "gw1"}
    assert all(sched.admissions[n.gateway.id].held(n) == 1 for n in nodes)
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(4), 1.0))
    assert sorted(i for order in ran.values() for i in order) == list(range(4))


@needs_xdist
def test_auto_enforces_the_detected_budget_even_below_the_worker_count() -> None:
    ids = [f"t{i}" for i in range(12)]
    est = Estimates(dict.fromkeys(ids, 1.0))
    # Nothing declares more than one slot, but the budget was asked for: a
    # container with two CPUs runs two of the four workers at a time.
    sched = cpu_scheduler(est, ids, workers=4, auto=True, host_budget=2)
    nodes = start(sched, ids, workers=4)
    assert sched.gated and sched.admissions["local"].budget == 2
    assert len(sched.waiting) == 2
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(12), 1.0))
    assert sorted(i for order in ran.values() for i in order) == list(range(12))
    assert sched.peak == 2
    # Turned on by a declaration only, the budget is never below the worker count.
    sched = cpu_scheduler(est, ids, workers=4, slots={"t0": 2}, host_budget=2)
    start(sched, ids, workers=4)
    assert sched.admissions["local"].budget == 4


@needs_xdist
def test_a_waiting_worker_that_dies_leaves_no_crashed_test_behind() -> None:
    ids = ["h1", "h2", "l1", "l2"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=2, slots={"h1": 2, "h2": 2})
    nodes = start(sched, ids)
    waiter = next(n for n in nodes if n in sched.waiting)
    runner = next(n for n in nodes if n is not waiter)
    head = waiter.queued[0]
    # It held one test it could not start; nothing of it ran. The test goes back
    # to the pool instead of being reported as crashed under a dead worker.
    assert sched.remove_node(waiter) is None
    assert head in sched.pending or head in sched.lanes[runner].plan or head in runner.queued
    ran = run_to_completion(sched, [runner], dict.fromkeys(range(4), 1.0))
    assert sorted(ran[runner.gateway.id]) == list(range(4))
    # A worker that was told to shut down had started its last test.
    sched = cpu_scheduler(est, ids, workers=2, budget=4, slots={"h1": 2, "h2": 2})
    nodes = start(sched, ids)
    for node in nodes:
        for index in list(sched.node2pending[node])[:-1]:
            sched.mark_test_complete(node, index)
    victim = next(n for n in nodes if n.shutting_down and len(sched.node2pending[n]) == 1)
    assert sched.remove_node(victim) == ids[victim.queued[-1]]


@needs_xdist
def test_an_external_shutdown_withdraws_the_unadmitted_head() -> None:
    from pytest_timing import xdist_compat

    ids = ["h1", "h2", "l1"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=2, slots={"h1": 2, "h2": 2})
    nodes = start(sched, ids, live=True)
    admission = sched.admissions["local"]
    waiter = next(n for n in nodes if n in sched.waiting)
    runner = next(n for n in nodes if n is not waiter)
    head = waiter.queued[0]
    # xdist shuts every worker down itself (-x): the waiter would run its head
    # without slots. It is told not to, and reserves nothing for it.
    for node in nodes:
        node.shutdown()
    assert admission.forced == 0
    assert waiter.commands == [(xdist_compat.CANCEL, {"indices": [head]})]
    assert sched.cancelled[waiter] == {head}
    assert admission.held(waiter) == 0 and waiter not in sched.waiting
    assert sched.cpu_summary()["cancelled"] == 1
    assert not sched.waits  # a withdrawn test never ran: no wait recorded
    assert admission.held(runner) == 2  # its own test was admitted before
    # Failed cancellation still requires a reservation when shutdown releases the test.
    sched = cpu_scheduler(est, ids, workers=2, budget=2, slots={"h1": 2, "h2": 2})
    nodes = start(sched, ids)
    waiter = next(n for n in nodes if n in sched.waiting)
    waiter.shutdown()
    assert sched.admissions["local"].forced == 1
    assert sched.admissions["local"].held(waiter) == 2


@needs_xdist
def test_idle_holds_do_not_force_a_head_over_the_budget_while_others_can_run() -> None:
    # Two workers keep a server fixture alive (one slot each) between tests that
    # need it; a third worker's three-slot tests must wait for a moment when the
    # budget of four has three slots free, not be forced over it.
    ids = [f"s{i}" for i in range(8)] + ["h1", "h2"]
    families = {f"s{i}": frozenset({DB}) for i in range(8)}
    est = Estimates(dict.fromkeys(ids, 1.0), 1.0, "", families, {DB: 0.5})
    base = DB.rpartition("[")[0]
    sched = cpu_scheduler(
        est, ids, workers=3, budget=4, slots={"h1": 3, "h2": 3}, fixtures={base: (1, 1)}
    )
    nodes = start(sched, ids, workers=3)
    ran = run_to_completion(sched, nodes, dict.fromkeys(range(10), 1.0))
    assert sorted(i for order in ran.values() for i in order) == list(range(10))
    assert sched.admissions["local"].forced == 0


@needs_xdist
def test_departed_and_idle_workers_contribute_no_cpu_rate_to_the_governor() -> None:
    ids = [f"t{i}" for i in range(40)]
    est = Estimates(dict.fromkeys(ids, 1.0))
    pressure = FakePressure()
    sched = cpu_scheduler(est, ids, workers=3, budget=4, pressure=pressure)
    nodes = start(sched, ids, workers=3)
    admission = sched.admissions["local"]
    gone = nodes[2]
    sched.observe_rate(gone.gateway.id, 4.0)  # a heavy worker's last rate
    sched.remove_node(gone)
    assert gone.gateway.id not in sched.rates
    sched.observe_rate(nodes[0].gateway.id, 0.5)
    sched.observe_rate(nodes[1].gateway.id, 0.5)
    pressure.some_value = 0.9
    for _ in range(3):
        node = next(n for n in nodes[:2] if len(n.queued) >= 2 and not n.shutting_down)
        sched.now += 1.0
        sched.mark_test_complete(node, sched.node2pending[node][0])
    # The stale 4.0 would have explained the pressure away; the live 1.0 does not.
    assert admission.limit == 3 and admission.lowered == 1


def test_a_fixture_built_on_a_parameter_is_paid_again_with_the_parameter() -> None:
    from pytest_timing.schedule import pay

    base0 = "session::conftest.py::base[0]"
    base1 = "session::conftest.py::base[1]"
    derived0 = "session::conftest.py::derived[base=0]"
    derived1 = "session::conftest.py::derived[base=1]"
    paid: set[str] = set()
    pay(paid, frozenset({base0, derived0}))
    assert paid == {base0, derived0}
    pay(paid, frozenset({base1}))  # tears base[0] down, and derived with it
    assert paid == {base1}
    pay(paid, frozenset({base1, derived1}))
    assert paid == {base1, derived1}
    pay(paid, frozenset({base0, derived0}))
    assert paid == {base0, derived0}
    # A worker in that state owes the set-up of derived[base=1], slots included.
    est = Estimates(
        dict.fromkeys(["a", "b"], 1.0),
        1.0,
        "",
        {"a": frozenset({base0, derived0}), "b": frozenset({base1, derived1})},
        {base0: 0.1, base1: 0.1, derived0: 0.5, derived1: 0.5},
    )
    decl = Declarations(tests=[1, 1], fixtures={key_base(derived0): (2, 0)})
    costs = Costs(est, ["a", "b"], decl)
    assert costs.charge(0, set()).peak == 2
    lane = Lane()
    lane.dispatch(costs.charge(0, lane.fixtures))
    assert costs.charge(1, lane.fixtures).peak == 2  # not warm: base changed
    assert costs.charge(1, lane.fixtures).seconds == pytest.approx(1.6)


@needs_xdist
def test_a_running_test_asks_for_the_slots_of_a_fixture_it_reaches_at_run_time() -> None:
    from pytest_timing import xdist_compat

    ids = ["a", "b", "c", "d"]
    est = Estimates(dict.fromkeys(ids, 1.0))
    sched = cpu_scheduler(est, ids, workers=2, budget=2)
    nodes = start(sched, ids, live=True)
    admission = sched.admissions["local"]
    gw0, gw1 = nodes
    assert admission.held(gw0) == 1 and admission.held(gw1) == 1
    key = "session::conftest.py::heavy[]"
    running = sched.node2pending[gw0][0]
    last = sched.node2pending[gw0][-1]
    # gw0's running test reaches a two-slot fixture with a hold. Blocked inside
    # its set-up the worker runs nothing, so it keeps only what its fixtures
    # hold (nothing yet) and waits for two slots; gw1 holds one for each of its
    # tests in turn (everything is dispatched, so it was shut down already).
    sched.request(gw0, running, key, setup=2, hold=1)
    assert gw0 in sched.requests and gw0 in sched.waiting and gw0.commands == []
    assert admission.held(gw0) == 0
    assert not sched.tests_finished
    sched.now = 1.0
    sched.mark_test_complete(gw1, sched.node2pending[gw1][0])
    assert gw0.commands == [] and admission.held(gw1) == 1
    sched.now = 2.0
    sched.mark_test_complete(gw1, sched.node2pending[gw1][0])
    assert gw0.commands == [(xdist_compat.GRANT, {"key": key, "request_id": 0})]
    assert admission.held(gw0) == 2 and gw0 not in sched.requests
    assert [(w.worker, w.index, w.attempt, w.seconds) for w in sched.waits] == [
        ("gw0", running, 0, 2.0)
    ]
    assert sched.dispatched[gw0][running].peak == 2
    assert sched.dispatched[gw0][last].peak == 2  # one of its own, one the instance holds
    sched.mark_test_complete(gw0, running)
    assert sched.costs.hold(sched.lanes[gw0].fixtures) == 1  # alive on gw0 from now on
    assert admission.held(gw0) == 2 and admission.used <= 2  # its last test, held slot
    assert admission.forced == 0
    # A draining shutdown never bypasses admission, including a repeated shutdown
    # from maxfail. A failed worker can exit with an unrun tail still in its queue.
    sched2 = cpu_scheduler(est, ids, workers=2, budget=2)
    nodes2 = start(sched2, ids, live=True)
    sched2.request(nodes2[0], sched2.node2pending[nodes2[0]][0], key, setup=2, hold=0)
    assert nodes2[0].commands == []
    for node in nodes2:
        node.shutdown()
        node.shutdown()
    assert nodes2[0].commands == []
    assert sched2.admissions["local"].forced == 0
    sched2.worker_finished(nodes2[1])
    assert (xdist_compat.GRANT, {"key": key, "request_id": 0}) in nodes2[0].commands
    assert sched2.admissions["local"].forced == 0


@needs_xdist
def test_forced_runtime_admission_keeps_attempt_identity_and_waits_for_completion() -> None:
    from pytest_timing import xdist_compat

    ids = ["a", "b"]
    sched = cpu_scheduler(Estimates(dict.fromkeys(ids, 1.0)), ids, budget=4, slots={"a": 2, "b": 2})
    nodes = start(sched, ids, live=True)
    gw0, gw1 = nodes
    assert all(node.shutting_down for node in nodes)  # draining, still running
    key = "function:test.py:conftest.py::heavy[]"
    sched.request(gw0, 0, key, setup=3, hold=0, holds=1, attempt=1)
    assert gw0 in sched.requests and not gw0.commands
    sched.now = 5
    sched.request(gw1, 1, key, setup=3, hold=0, holds=1)
    admission = sched.admissions["local"]
    assert admission.forced == 1
    assert gw0.commands == [(xdist_compat.GRANT, {"key": key, "request_id": 0})]
    assert gw1 in sched.requests and not gw1.commands
    (wait,) = sched.waits
    assert (wait.worker, wait.index, wait.attempt, wait.seconds) == ("gw0", 0, 1, 5)
    sched.update_holds(gw0, 0)  # finalizers ended, the protocol is still busy
    sched._admit_waiting()
    assert not gw1.commands and admission.forced == 1
    sched.mark_test_complete(gw0, 0)
    assert gw1.commands == [(xdist_compat.GRANT, {"key": key, "request_id": 0})]
    assert admission.forced == 1


def test_scope_transitions_keep_unused_holds_and_expire_only_departed_scopes() -> None:
    session = "session::conftest.py::session[]"
    package = "package:pkg:pkg::package[]"
    nested = "package:pkg/sub:pkg/sub::nested[]"
    module = "module:pkg/sub/test_a.py:pkg/sub/test_a.py::module[]"
    cls = "class:pkg/sub/test_a.py::TestA:pkg/sub/test_a.py::class[]"
    keys = [session, package, nested, module, cls]
    ids = [
        "pkg/sub/test_a.py::TestA::test_first",
        "pkg/sub/test_a.py::TestA::test_plain",
        "pkg/sub/test_a.py::TestB::test_plain",
        "pkg/sub/test_b.py::test_plain",
        "pkg/test_c.py::test_plain",
        "pkg_extra/test_d.py::test_plain",
        "pkg/sub/test_a.py::TestA::test_first_again",
    ]
    est = Estimates(
        dict.fromkeys(ids, 1.0),
        families={ids[0]: frozenset(keys), ids[-1]: frozenset(keys)},
        setups=dict.fromkeys(keys, 0.1),
    )
    decl = Declarations(tests=[1] * len(ids), fixtures={key_base(k): (1, 1) for k in keys})
    costs = Costs(est, ids, decl)
    lane = Lane()
    peaks = []
    charges = []
    for index in range(len(ids)):
        charge = costs.charge(index, lane.fixtures)
        lane.dispatch(charge)
        peaks.append(charge.peak)
        charges.append(charge.seconds)
    assert peaks == [6, 6, 5, 4, 3, 2, 6]
    assert charges == pytest.approx([1.5, 1, 1, 1, 1, 1, 1.4])


@needs_xdist
def test_worksteal_uses_the_cpu_work_floor_before_duplicating_a_cold_setup() -> None:
    ids = [f"t{i}" for i in range(20)]
    est = Estimates(
        dict.fromkeys(ids, 0.1), 0.1, "", dict.fromkeys(ids, frozenset({DB})), {DB: 1.0}
    )
    sched = cpu_scheduler(
        est, ids, workers=3, stealing=True, budget=4, fixtures={key_base(DB): (4, 0)}
    )
    start(sched, ids, workers=3)
    receiver = FakeNode("replacement")
    sched.add_node(receiver)
    assert sched._choose_steal(receiver) is None
    sched.admissions.clear()
    assert sched._choose_steal(receiver) is not None  # lane time alone would duplicate it
