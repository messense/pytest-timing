from __future__ import annotations

import pytest
from conftest import T0, crash_report, full_test, make_info, report

from pytest_timing.collector import Collector, PhaseReport
from pytest_timing.model import Run


def test_waits_use_execution_identity_and_accumulate_across_requests() -> None:
    collector = Collector(make_info())
    # Two selections with the same nodeid on gw0, the first retried without a
    # teardown report; the same collection index also appears on another worker.
    executions = [("gw0", 4, 0), ("gw0", 4, 1), ("gw0", 9, 0), ("gw1", 4, 0)]
    for number, (worker, index, attempt) in enumerate(executions):
        for when in ("setup", "call", "teardown"):
            if number == 0 and when == "teardown":
                continue
            phase = report("test.py::test_same", when, number, number + 0.1, worker=worker)
            phase.execution = (index, attempt)
            if number == 0 and when == "call":
                phase.outcome = "rerun"
            collector.add_report(phase)
    collector.add_wait("gw0", "test.py::test_same", 4, 0, 0.2)
    collector.add_wait("gw0", "test.py::test_same", 4, 0, 0.3)
    collector.add_wait("gw0", "test.py::test_same", 4, 1, 0.1)
    collector.add_wait("gw0", "test.py::test_same", 9, 0, 0.4)
    collector.add_wait("gw1", "test.py::test_same", 4, 0, 0.6)
    collector.add_wait("gw0", "test.py::test_same", 100, 0, 9)  # never guess another execution
    run = collector.finish(T0 + 5, termination="finished")
    assert [t.cpu.wait for t in run.tests if t.cpu] == pytest.approx([0.5, 0.1, 0.4, 0.6])
    assert [(t.occurrence, t.attempt) for t in run.tests] == [(0, 0), (0, 1), (1, 0), (0, 0)]


def test_phases_fold_into_one_span() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::test_x", 1.0, setup=0.1, call=0.5, teardown=0.2)
    run = c.finish(T0 + 5, termination="finished")

    assert len(run.tests) == 1
    span = run.tests[0]
    assert span.worker == "gw0"
    assert span.outcome == "passed"
    assert span.start == 1.0
    assert abs(span.stop - 1.8) < 1e-6
    assert list(span.phases) == ["setup", "call", "teardown"]
    assert abs(span.phases["call"].duration - 0.5) < 1e-6
    assert run.run.complete is True
    assert run.run.stop == T0 + 5


def test_outcomes() -> None:
    c = Collector(make_info())
    full_test(c, "t.py::fail", 0, outcome="failed")
    c.add_report(report("t.py::err", "setup", 1, 1.1, outcome="failed"))
    c.add_report(report("t.py::err", "teardown", 1.1, 1.2))
    c.add_report(report("t.py::skip", "setup", 2, 2.1, outcome="skipped"))
    c.add_report(report("t.py::skip", "teardown", 2.1, 2.2))
    c.add_report(report("t.py::xf", "setup", 3, 3.1))
    c.add_report(report("t.py::xf", "call", 3.1, 3.2, outcome="skipped", wasxfail=True))
    c.add_report(report("t.py::xf", "teardown", 3.2, 3.3))
    c.add_report(report("t.py::xp", "setup", 4, 4.1))
    c.add_report(report("t.py::xp", "call", 4.1, 4.2, outcome="passed", wasxfail=True))
    c.add_report(report("t.py::xp", "teardown", 4.2, 4.3))
    # failure in teardown after a passing call -> error
    c.add_report(report("t.py::td", "setup", 5, 5.1))
    c.add_report(report("t.py::td", "call", 5.1, 5.2))
    c.add_report(report("t.py::td", "teardown", 5.2, 5.3, outcome="failed"))
    run = c.finish(T0 + 6, termination="finished")

    outcomes = {t.nodeid: t.outcome for t in run.tests}
    assert outcomes == {
        "t.py::fail": "failed",
        "t.py::err": "error",
        "t.py::skip": "skipped",
        "t.py::xf": "xfailed",
        "t.py::xp": "xpassed",
        "t.py::td": "error",
    }
    assert run.outcome_counts()["failed"] == 1


def test_duplicate_selection_is_a_new_occurrence() -> None:
    """A repeated nodeid whose previous run was not retried is a separate collected item."""
    c = Collector(make_info())
    full_test(c, "t.py::dup", 0)
    full_test(c, "t.py::dup", 1, outcome="failed")
    run = c.finish(T0 + 3, termination="finished")
    assert [(t.occurrence, t.attempt, t.outcome) for t in run.tests] == [
        (0, 0, "passed"),
        (1, 0, "failed"),
    ]


def test_dist_each_runs_same_nodeid_on_every_worker() -> None:
    c = Collector(make_info(dist="each"))
    full_test(c, "t.py::x", 0, worker="gw0")
    full_test(c, "t.py::x", 0, worker="gw1")
    run = c.finish(T0 + 2, termination="finished")
    assert sorted((t.worker, t.attempt) for t in run.tests) == [("gw0", 0), ("gw1", 0)]


def test_crash_report_closes_open_span() -> None:
    c = Collector(make_info())
    c.worker_ready("gw0", T0 + 0.1)
    c.add_report(report("t.py::boom", "setup", 1.0, 1.1))
    c.add_report(report("t.py::boom", "call", 1.1, 1.2))  # call reported, then the process dies
    c.worker_down("gw0", T0 + 2.0, "worker crashed")
    c.add_report(crash_report("t.py::boom", 2.05))
    run = c.finish(T0 + 3, termination="finished")

    assert len(run.tests) == 1
    span = run.tests[0]
    assert span.outcome == "crashed"
    assert span.stop >= 2.0
    assert run.workers[0].error == "worker crashed"


def test_crash_without_any_phase_report_starts_after_previous_test() -> None:
    c = Collector(make_info())
    c.worker_ready("gw0", T0 + 0.1)
    full_test(c, "t.py::ok", 0.5)
    c.add_report(crash_report("t.py::boom", 4.0))
    run = c.finish(T0 + 5, termination="finished")
    boom = run.tests[-1]
    assert boom.outcome == "crashed"
    assert abs(boom.start - 1.2) < 1e-6
    assert abs(boom.stop - 4.0) < 1e-6


def test_missing_timestamps_fall_back_to_receipt_time() -> None:
    """pytest < 7.3 reports have no start/stop; anchor on when the controller saw them."""
    c = Collector(make_info())
    c.add_report(
        PhaseReport(
            nodeid="t.py::x",
            when="call",
            outcome="passed",
            start=0,
            stop=0,
            duration=0.5,
            worker="main",
            received=T0 + 2.0,
        )
    )
    run = c.finish(T0 + 3, termination="finished")
    span = run.tests[0]
    assert abs(span.start - 1.5) < 1e-6 and abs(span.stop - 2.0) < 1e-6


def test_interrupted_run_marks_open_spans() -> None:
    c = Collector(make_info())
    c.add_report(report("t.py::x", "setup", 0, 0.1))
    run = c.finish(T0 + 1, termination="interrupted", reason="KeyboardInterrupt")
    assert run.run.complete is False
    assert run.run.termination == "interrupted"
    assert run.run.reason == "KeyboardInterrupt"
    assert run.tests[0].outcome == "crashed"
    assert abs(run.tests[0].stop - 1.0) < 1e-6


def test_worker_sort_order() -> None:
    c = Collector(make_info())
    for wid in ("gw10", "gw2", "gw0", "main"):
        c.worker_ready(wid, T0)
    run = c.finish(T0 + 1, termination="finished")
    assert [w.id for w in run.workers] == ["gw0", "gw2", "gw10", "main"]


def test_json_round_trip(sample_run: Run) -> None:
    text = sample_run.to_json()
    again = Run.from_json(text)
    assert again.to_dict() == sample_run.to_dict()
    assert again.to_dict()["schema"] == 1
    assert again.tests[0].phases["call"].duration == sample_run.tests[0].phases["call"].duration


def test_derived_metrics(sample_run: Run) -> None:
    assert sample_run.worker_ids() == ["gw0", "gw1"]
    ready, down = sample_run.lane_window("gw0")
    assert abs(ready - 0.2) < 1e-6 and abs(down - 2.4) < 1e-6
    assert 0 < sample_run.utilisation() < 1
    assert abs(sample_run.wall - 3.0) < 1e-6
    lanes = sample_run.tests_by_worker()
    assert [t.nodeid for t in lanes["gw1"]][:2] == [
        "tests/test_a.py::test_two",
        "tests/test_b.py::test_four",
    ]


def test_rerun_outcome_marks_failed_attempt() -> None:
    """pytest-rerunfailures reports the failed attempt's call phase as 'rerun'."""
    c = Collector(make_info())
    c.add_report(report("t.py::flaky", "setup", 0, 0.1))
    c.add_report(report("t.py::flaky", "call", 0.1, 0.2, outcome="rerun"))
    c.add_report(report("t.py::flaky", "teardown", 0.2, 0.3))
    full_test(c, "t.py::flaky", 1)
    run = c.finish(T0 + 3, termination="finished")
    assert [(t.occurrence, t.attempt, t.outcome) for t in run.tests] == [
        (0, 0, "rerun"),
        (0, 1, "passed"),
    ]
    assert run.outcome_counts() == {"rerun": 1, "passed": 1}


def test_completeness_is_derived_from_termination() -> None:
    import pytest

    from pytest_timing.model import (
        COMPLETE_TERMINATIONS,
        TERMINATION_REGISTRY,
        TERMINATIONS,
        RunInfo,
    )

    for termination in TERMINATIONS:
        info = RunInfo(start=T0, stop=T0 + 1, termination=termination)
        assert info.complete is (termination in COMPLETE_TERMINATIONS)
        assert info.termination_label == TERMINATION_REGISTRY[termination].label
    assert TERMINATION_REGISTRY["unknown"].label == "unknown"
    # an invalid termination value is refused, never guessed
    with pytest.raises(ValueError, match="termination"):
        RunInfo.from_dict({"start": 1, "stop": 2, "termination": "weird"})
    # the early schema-1 shape (only a complete flag) migrates
    assert RunInfo.from_dict({"start": 1, "stop": 2, "complete": True}).termination == "finished"
    assert RunInfo.from_dict({"start": 1, "stop": 2, "complete": False}).termination == "unknown"
    assert RunInfo.from_dict({"start": 1, "stop": 2}).termination == "unknown"


def test_worker_start_is_an_explicit_event() -> None:
    c = Collector(make_info())
    c.worker_started("gw0", T0 + 0.3)
    c.worker_ready("gw0", T0 + 0.5)
    c.worker_ready("gw1", T0 + 0.5)  # never announced: launch time unknown
    run = c.finish(T0 + 1, termination="finished")
    gw0, gw1 = run.workers
    assert abs((gw0.start or 0) - 0.3) < 1e-6 and abs((gw0.ready or 0) - 0.5) < 1e-6
    assert gw1.start is None and gw1.ready is not None
    doc = run.to_dict()
    assert doc["workers"][1]["start"] is None
    assert Run.from_dict(doc).workers[1].start is None


def test_outcome_registry_is_the_single_source() -> None:
    from pytest_timing.model import BAD_OUTCOMES, OUTCOME_REGISTRY, Run

    assert BAD_OUTCOMES == {n for n, i in OUTCOME_REGISTRY.items() if i.is_failure}
    doc = Run(run=make_info()).to_dict()
    assert doc["outcomes"] == {
        n: {"label": i.label, "is_failure": i.is_failure} for n, i in OUTCOME_REGISTRY.items()
    }


def test_rebased_moves_everything_together(sample_run: Run) -> None:
    moved = sample_run.rebased(sample_run.run.start - 10)
    assert moved.run.start == sample_run.run.start - 10
    for before, after in zip(sample_run.tests, moved.tests, strict=True):
        assert abs(after.start - before.start - 10) < 1e-9
        assert abs(after.duration - before.duration) < 1e-9
        for name in before.phases:
            assert abs(after.phases[name].duration - before.phases[name].duration) < 1e-9
            assert abs(after.phases[name].start - before.phases[name].start - 10) < 1e-9
    for before_w, after_w in zip(sample_run.workers, moved.workers, strict=True):
        assert abs(after_w.start - before_w.start - 10) < 1e-9
        assert abs((after_w.ready or 0) - (before_w.ready or 0) - 10) < 1e-9


def test_gateway_observer_times_the_creation_call() -> None:
    """The observer stamps the moment before makegateway runs and restores itself."""
    import time
    from types import SimpleNamespace

    from pytest_timing.xdist_compat import observe_gateway_creation

    class Group:
        def makegateway(self, spec: str) -> SimpleNamespace:
            time.sleep(0.05)
            return SimpleNamespace(id=spec)

    group = Group()
    manager = SimpleNamespace(group=group)
    dsession = SimpleNamespace(nodemanager=manager)
    plugins = SimpleNamespace(getplugin=lambda name: dsession if name == "dsession" else None)
    config = SimpleNamespace(pluginmanager=plugins)
    seen: list[tuple[str, float]] = []
    restore = observe_gateway_creation(config, lambda wid, t: seen.append((wid, t)))  # type: ignore[arg-type]
    assert restore is not None
    before = time.time()
    gateway = group.makegateway("gw0")
    after = time.time()
    assert gateway.id == "gw0"
    assert seen and seen[0][0] == "gw0"
    assert before <= seen[0][1] <= after - 0.05  # stamped before the (slow) creation
    restore()
    assert "makegateway" not in group.__dict__  # the class method is back in charge
    group.makegateway("gw1")
    assert len(seen) == 1


def test_gateway_observer_coexists_with_another_instance_override() -> None:
    """A wrapper already installed on the instance keeps working and is restored."""
    from types import SimpleNamespace

    from pytest_timing.xdist_compat import observe_gateway_creation

    class Group:
        def makegateway(self, spec: str) -> SimpleNamespace:
            return SimpleNamespace(id=spec)

    group = Group()
    other_seen: list[str] = []
    class_method = group.makegateway

    def other_observer(spec: str) -> SimpleNamespace:
        other_seen.append(spec)
        return class_method(spec)

    group.makegateway = other_observer  # type: ignore[method-assign]
    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(
            getplugin=lambda name: SimpleNamespace(nodemanager=SimpleNamespace(group=group))
        )
    )
    ours: list[str] = []
    restore = observe_gateway_creation(config, lambda wid, t: ours.append(wid))  # type: ignore[arg-type]
    assert restore is not None
    group.makegateway("gw0")
    assert ours == ["gw0"] and other_seen == ["gw0"]  # both observers saw it
    restore()
    assert group.__dict__["makegateway"] is other_observer  # theirs is back, not deleted
    group.makegateway("gw1")
    assert ours == ["gw0"] and other_seen == ["gw0", "gw1"]


def test_gateway_observer_leaves_a_later_replacement_alone() -> None:
    from types import SimpleNamespace

    from pytest_timing.xdist_compat import observe_gateway_creation

    class Group:
        def makegateway(self, spec: str) -> SimpleNamespace:
            return SimpleNamespace(id=spec)

    group = Group()
    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(
            getplugin=lambda name: SimpleNamespace(nodemanager=SimpleNamespace(group=group))
        )
    )
    restore = observe_gateway_creation(config, lambda wid, t: None)  # type: ignore[arg-type]
    assert restore is not None
    later = lambda spec: SimpleNamespace(id=spec)  # noqa: E731
    group.makegateway = later  # type: ignore[method-assign]
    restore()
    assert group.__dict__["makegateway"] is later


def test_gateway_observer_needs_the_group() -> None:
    from types import SimpleNamespace

    from pytest_timing.xdist_compat import observe_gateway_creation

    config = SimpleNamespace(pluginmanager=SimpleNamespace(getplugin=lambda name: None))
    assert observe_gateway_creation(config, lambda wid, t: None) is None  # type: ignore[arg-type]


def test_collection_watch_needs_observed_difference() -> None:
    import pytest

    from pytest_timing.xdist_compat import CollectionWatch

    announce = pytest.CollectReport(
        nodeid="gw1",
        outcome="failed",
        longrepr="Different tests were collected between gw0 and gw1. The difference is: ...",
        result=[],
    )
    same = CollectionWatch()
    same.collected("gw0", ["t.py::a", "t.py::b"])
    same.collected("gw1", ["t.py::a", "t.py::b"])
    assert same.mismatch_reason(announce) is None

    differs = CollectionWatch()
    differs.collected("gw0", ["t.py::a"])
    differs.collected("gw1", ["t.py::b"])
    assert (
        differs.mismatch_reason(announce)
        == "gw1: Different tests were collected between gw0 and gw1. The difference is: ..."
    )
    # a forwarded collection error names a file, not a worker
    forwarded = pytest.CollectReport(
        nodeid="t.py", outcome="failed", longrepr="ImportError", result=[]
    )
    assert differs.mismatch_reason(forwarded) is None


def test_lanes_and_summary(sample_run: Run) -> None:
    lanes = {lane.id: lane for lane in sample_run.lanes()}
    assert list(lanes) == ["gw0", "gw1"]
    gw0 = lanes["gw0"]
    assert [t.nodeid for t in gw0.spans] == sorted(
        (t.nodeid for t in sample_run.tests if t.worker == "gw0"),
        key=lambda n: next(t.start for t in sample_run.tests if t.nodeid == n),
    )
    assert gw0.boot is not None and gw0.collect is not None
    assert abs(gw0.boot[1] - gw0.collect[0]) < 1e-9
    summary = sample_run.to_dict()["summary"]
    assert abs(summary["busy"] - sample_run.busy_seconds()) < 1e-6
    assert abs(summary["utilisation"] - sample_run.utilisation()) < 1e-3
    assert [lane["id"] for lane in summary["lanes"]] == ["gw0", "gw1"]
    assert abs(summary["lanes"][0]["last"] - max(t.stop for t in gw0.spans)) < 1e-6
