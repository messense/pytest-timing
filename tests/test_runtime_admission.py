"""Runtime fixture admission across real worker, fixture and report lifecycles."""

from __future__ import annotations

import importlib.util
import json
import os
import textwrap

import pytest
from conftest import (
    T0,
    Collector,
    event_peak,
    events,
    full_test,
    load_json,
    make_info,
    needs_xdist,
    run_timing,
    write_history,
)

from pytest_timing.model import CpuRecord, Run
from pytest_timing.schedule import Estimates

EVENTS = """
import json, os, time
from pathlib import Path
import pytest, pytest_timing

def event(kind, slots=0):
    # One file per process: concurrent appends to a shared file can clobber each
    # other on Windows, where O_APPEND is not atomic across handles.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    with open(f"events-{worker or 'main'}.jsonl", "a") as out:
        row = [time.time(), worker, kind, slots]
        out.write(json.dumps(row) + "\\n")

def wait_for(path):
    deadline = time.monotonic() + 5
    while not Path(path).exists():
        assert time.monotonic() < deadline, path
        time.sleep(.005)
"""


@needs_xdist
@pytest.mark.parametrize(
    "extra",
    [[], ["--timing-schedule", "missing.json"], ["--timing-cpus", "2", "--dist", "loadscope"]],
)
def test_dynamic_requests_work_when_the_scheduler_falls_back(
    pytester: pytest.Pytester, extra: list[str]
) -> None:
    pytester.makepyfile("""
        import pytest, pytest_timing
        @pytest.fixture(scope="session")
        @pytest_timing.cpu(2, hold=1)
        def heavy(): return 42
        def test_a(request): assert request.getfixturevalue("heavy") == 42
        def test_b(request): assert request.getfixturevalue("heavy") == 42
    """)
    result = run_timing(pytester, "-n2", *extra, timeout=15)
    result.assert_outcomes(passed=2)


@needs_xdist
def test_fixture_events_work_when_another_plugin_selects_the_scheduler(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest("""
        import pytest
        from xdist.scheduler.load import LoadScheduling
        class CustomScheduler:
            @pytest.hookimpl(tryfirst=True)
            def pytest_xdist_make_scheduler(self, config, log):
                return LoadScheduling(config, log)
        def pytest_sessionstart(session):
            if not hasattr(session.config, "workerinput"):
                session.config.pluginmanager.register(CustomScheduler())
    """)
    pytester.makepyfile("""
        import pytest, pytest_timing
        @pytest.fixture(scope="session")
        @pytest_timing.cpu(1, hold=1)
        def server(): yield
        @pytest.fixture
        @pytest_timing.cpu(2)
        def heavy(): pass
        def test_a(server, request): request.getfixturevalue("heavy")
        def test_b(server, request): request.getfixturevalue("heavy")
    """)
    result = run_timing(pytester, "-n2", timeout=15)
    result.assert_outcomes(passed=2)


@needs_xdist
@pytest.mark.parametrize("scope", ["function", "session"])
def test_runtime_setup_respects_live_function_fixture_holds(
    pytester: pytest.Pytester, scope: str
) -> None:
    pytester.makepyfile(
        EVENTS
        + f"""
@pytest.fixture
@pytest_timing.cpu(1, hold=1)
def server():
    event("server_start", 1)
    yield
    event("server_end", -1)

@pytest.fixture(scope={scope!r})
@pytest_timing.cpu(2)
def heavy():
    event("heavy_start", 2)
    time.sleep(.15)
    event("heavy_end", -2)

def test_a(server, request): request.getfixturevalue("heavy")
def test_b(server, request): request.getfixturevalue("heavy")
def test_c(server, request): request.getfixturevalue("heavy")
def test_d(server, request): request.getfixturevalue("heavy")
"""
    )
    result = run_timing(pytester, "-n2", "--timing-cpus", "4", timeout=15)
    result.assert_outcomes(passed=4)
    assert event_peak(pytester) <= 4
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
def test_runtime_hold_snapshots_follow_nested_setup_failure_and_parameter_replacement(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest("""
        import json
        from pytest_timing.xdist_scheduler import DurationScheduling
        original = DurationScheduling.request
        def request(
            self, node, index, key, setup, hold, holds=None, attempt=0,
            request_id=0, cancelled=False,
        ):
            with open("requests.jsonl", "a") as out:
                out.write(json.dumps([key.rpartition("::")[2], holds]) + "\\n")
            original(self, node, index, key, setup, hold, holds, attempt, request_id, cancelled)
        DurationScheduling.request = request
    """)
    pytester.makepyfile("""
        import pytest, pytest_timing
        @pytest.fixture(scope="session", params=[0, 1])
        @pytest_timing.cpu(1, hold=1)
        def base(request): yield request.param
        @pytest.fixture
        @pytest_timing.cpu(2, hold=2)
        def leaf(): yield
        @pytest.fixture
        @pytest_timing.cpu(2, hold=1)
        def outer(request):
            request.getfixturevalue("leaf")
            yield
        @pytest.fixture
        @pytest_timing.cpu(1, hold=8)
        def broken(): raise ValueError("setup failed")
        @pytest.fixture
        @pytest_timing.cpu(2)
        def heavy(): pass
        def test_run(base, request):
            request.getfixturevalue("outer")
            with pytest.raises(ValueError, match="setup failed"):
                request.getfixturevalue("broken")
            request.getfixturevalue("heavy")
    """)
    result = run_timing(pytester, "-n1", "--timing-cpus", "16", timeout=15)
    result.assert_outcomes(passed=2)
    requests = [
        json.loads(line) for line in (pytester.path / "requests.jsonl").read_text().splitlines()
    ]
    assert requests == [["outer[]", 1], ["leaf[]", 1], ["broken[]", 4], ["heavy[]", 4]] * 2
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
def test_maxfail_releases_exited_workers_for_runtime_requests(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        EVENTS
        + """
@pytest.fixture
def cleanup():
    yield
    event("cleanup_start")
    time.sleep(.15)
    event("cleanup_end")

@pytest.fixture(scope="session")
@pytest_timing.cpu(2)
def heavy():
    event("heavy_start")
    time.sleep(.05)
    event("heavy_end")

def test_a_fails(cleanup):
    wait_for("requested")
    time.sleep(.05)
    assert False

def test_b_dynamic(request):
    Path("requested").touch()
    request.getfixturevalue("heavy")

def test_c(): pass
def test_d(): pass
"""
    )
    result = run_timing(pytester, "-n2", "-x", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(failed=1, passed=2)
    times = {kind: at for at, _, kind, _ in events(pytester)}
    assert times["heavy_start"] >= times["cleanup_end"]
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
def test_timeout_fails_without_starting_the_fixture(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("""
        import pytest_timing.demand
        pytest_timing.demand.REQUEST_TIMEOUT = .05
    """)
    pytester.makepyfile(
        EVENTS
        + """
@pytest.fixture(scope="session")
@pytest_timing.cpu(2)
def heavy(): event("unreserved_setup")

def test_a_competing():
    Path("running").touch()
    time.sleep(.25)

def test_b_dynamic(request):
    wait_for("running")
    request.getfixturevalue("heavy")
"""
    )
    result = run_timing(pytester, "-n2", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*timed out waiting for CPU slots*"])
    assert not list(pytester.path.glob("events-*.jsonl"))
    doc = load_json(pytester.path)
    assert doc["run"]["cpu"]["domains"]["local"]["forced"] == 0
    failed = next(t for t in doc["tests"] if t["outcome"] == "failed")
    assert failed["cpu"]["runtime_wait"] >= 0.04


@needs_xdist
@pytest.mark.parametrize("nested", [False, True])
def test_runtime_wait_is_excluded_from_test_and_fixture_estimates(
    pytester: pytest.Pytester, nested: bool
) -> None:
    pytester.makepyfile(
        textwrap.dedent("""
        import time, pytest, pytest_timing
        @pytest.fixture(scope="session")
        @pytest_timing.cpu(2)
        def heavy(): time.sleep(.4)
        @pytest.fixture(scope="session")
        def outer(request):
            time.sleep(.02)
            request.getfixturevalue("heavy")
            time.sleep(.02)
    """)
        + (
            "\ndef test_a(outer): pass\ndef test_b(outer): pass\n"
            if nested
            else '\ndef test_a(request): request.getfixturevalue("heavy")\n'
            'def test_b(request): request.getfixturevalue("heavy")\n'
        )
    )
    result = run_timing(pytester, "-n2", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=2)
    run = Run.from_dict(load_json(pytester.path))
    estimates = Estimates.from_run(run)
    assert max(t.cpu.runtime_wait for t in run.tests if t.cpu) >= 0.2
    assert all(duration < 0.15 for duration in estimates.durations.values())
    for test in run.tests:
        assert test.cpu is not None
        assert test.cpu.elapsed == pytest.approx(test.duration - test.cpu.runtime_wait, abs=0.02)
    if nested:
        # ``outer`` keeps only its own 40ms: not the nested 0.4s set-up, nor the
        # wait. A loaded runner stretches those 40ms to ~0.2s, hence the ceiling.
        assert all(
            0.03 <= seconds < 0.3 for key, seconds in estimates.setups.items() if "::outer[" in key
        )


def test_only_wait_inside_execution_is_subtracted_from_saved_estimates() -> None:
    collector = Collector(make_info())
    full_test(collector, "prestart", 0, setup=0, call=1, teardown=0)
    collector.tests[-1].cpu = CpuRecord(elapsed=1, work=0.5, wait=5)
    full_test(collector, "runtime", 2, setup=0, call=4, teardown=0)
    collector.tests[-1].cpu = CpuRecord(elapsed=1, work=0.5, wait=8, runtime_wait=3)
    run = collector.finish(T0 + 7, termination="finished")
    restored = Run.from_dict(run.to_dict())
    assert Estimates.from_run(restored).durations == pytest.approx({"prestart": 1, "runtime": 1})
    assert restored.tests[1].cpu == collector.tests[1].cpu


@needs_xdist
@pytest.mark.skipif(
    importlib.util.find_spec("pytest_rerunfailures") is None, reason="rerun plugin not installed"
)
def test_admission_wait_stays_with_the_attempt_that_waited(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("""
        import time, pytest, pytest_timing
        @pytest.fixture(scope="session")
        @pytest_timing.cpu(2)
        def heavy(): time.sleep(.2)
        @pytest.mark.flaky(reruns=1)
        def test_a(request):
            request.getfixturevalue("heavy")
            assert request.node.execution_count > 1
        @pytest.mark.flaky(reruns=1)
        def test_b(request):
            request.getfixturevalue("heavy")
            assert request.node.execution_count > 1
    """)
    result = run_timing(pytester, "-p", "rerunfailures", "-n2", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=2)
    assert result.parseoutcomes()["rerun"] == 2
    run = Run.from_dict(load_json(pytester.path))
    waited = [t for t in run.tests if t.cpu and t.cpu.wait >= 0.15]
    assert waited
    assert all(t.attempt == 0 for t in waited)
    for test in run.tests:
        if test.cpu is None:
            # Older rerun plugins omit failed-attempt teardown reports. A span
            # without a controller wait then has no CPU record at all.
            assert test.attempt == 0
            continue
        if test.cpu.runtime_wait:
            assert test.cpu.wait == pytest.approx(test.cpu.runtime_wait, abs=0.05)
        if test.attempt == 1:
            assert test.cpu.wait == 0


@needs_xdist
@pytest.mark.skipif(os.name == "nt", reason="kills a worker with SIGKILL")
def test_crash_during_runtime_admission_is_reported_without_replaying_the_test(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest("""
        import os, signal
        from pathlib import Path
        from pytest_timing.xdist_scheduler import DurationScheduling
        original = DurationScheduling.request
        def request(self, node, *args, **kwargs):
            original(self, node, *args, **kwargs)
            if node in self.requests and not Path("killed").exists():
                Path("killed").touch()
                pid = int(Path("pid-" + node.gateway.id).read_text())
                os.kill(pid, signal.SIGKILL)
        DurationScheduling.request = request
        def pytest_sessionstart(session):
            worker = os.environ.get("PYTEST_XDIST_WORKER")
            if worker:
                Path("pid-" + worker).write_text(str(os.getpid()))
    """)
    pytester.makepyfile(
        EVENTS
        + """
@pytest.fixture(scope="session")
@pytest_timing.cpu(2)
def heavy(): pass
def test_a_competing():
    Path("running").touch()
    time.sleep(.3)
def test_b_dynamic(request):
    wait_for("running")
    event("body_started")
    request.getfixturevalue("heavy")
"""
    )
    result = run_timing(pytester, "-n2", "--max-worker-restart=1", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=1, failed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert (pytester.path / "killed").exists()
    assert len([e for e in events(pytester) if e[2] == "body_started"]) == 1


@needs_xdist
def test_draining_runtime_waiters_escape_a_holds_deadlock(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("""
        import pytest_timing.demand
        pytest_timing.demand.REQUEST_TIMEOUT = .5
    """)
    pytester.makepyfile(
        EVENTS
        + """
@pytest.fixture
@pytest_timing.cpu(1, hold=1)
def server():
    Path(os.environ["PYTEST_XDIST_WORKER"] + "-holding").touch()
    yield
@pytest.fixture
@pytest_timing.cpu(3)
def heavy(): time.sleep(.05)
def run(request):
    wait_for("gw0-holding")
    wait_for("gw1-holding")
    request.getfixturevalue("heavy")
def test_a(server, request): run(request)
def test_b(server, request): run(request)
"""
    )
    result = run_timing(pytester, "-n2", "--timing-cpus", "4", timeout=15)
    result.assert_outcomes(passed=2)
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 1


@needs_xdist
def test_admission_wait_survives_a_crash_before_any_phase_report(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("""
        import os, time, pytest
        @pytest.fixture
        def crash(): os._exit(17)
        @pytest.mark.timing_cpu(2)
        def test_a(): time.sleep(.25)
        @pytest.mark.timing_cpu(2)
        def test_b(crash): pass
    """)
    result = run_timing(pytester, "-n2", "--max-worker-restart=0", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=1, failed=1)
    run = Run.from_dict(load_json(pytester.path))
    crashed = next(test for test in run.tests if test.outcome == "crashed")
    assert not crashed.phases
    assert crashed.cpu is not None and crashed.cpu.wait >= 0.15
    assert run.run.cpu is not None
    assert run.run.cpu["waited"] == pytest.approx(crashed.cpu.wait)


@needs_xdist
@pytest.mark.parametrize("dist", ["load", "worksteal"])
def test_runtime_setup_keeps_module_holds_despite_prefetched_context_changes(
    pytester: pytest.Pytester, dist: str
) -> None:
    pytester.makeconftest(
        EVENTS
        + """
@pytest.fixture(scope="module")
@pytest_timing.cpu(1, hold=1)
def server():
    event("server_start", 1)
    yield
    event("server_end", -1)
@pytest.fixture(scope="session")
@pytest_timing.cpu(2)
def heavy():
    event("heavy_start", 2)
    time.sleep(.15)
    event("heavy_end", -2)
"""
    )
    body = 'def test_run(server, request): request.getfixturevalue("heavy")'
    pytester.makepyfile(
        test_a=body,
        test_b=body,
        test_z="""
            import time
            def test_plain1(): time.sleep(.05)
            def test_plain2(): time.sleep(.05)
        """,
    )
    ids = [
        "test_a.py::test_run",
        "test_b.py::test_run",
        "test_z.py::test_plain1",
        "test_z.py::test_plain2",
    ]
    write_history(pytester.path / "history.json", dict.fromkeys(ids, 1), stop=4)
    result = run_timing(
        pytester,
        "-n2",
        "--dist",
        dist,
        "--timing-cpus",
        "4",
        "--timing-schedule",
        "history.json",
        timeout=15,
    )
    result.assert_outcomes(passed=4)
    assert event_peak(pytester) <= 4
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
@pytest.mark.parametrize("dist", ["load", "worksteal"])
def test_static_reservations_keep_holds_until_the_scope_ends(
    pytester: pytest.Pytester, dist: str
) -> None:
    pytester.makeconftest(EVENTS)
    pytester.makepyfile(
        test_a="""
        import time, pytest, pytest_timing
        from conftest import event
        @pytest.fixture(scope="module")
        @pytest_timing.cpu(2, hold=1)
        def server():
            event("hold_start", 1)
            yield
            event("hold_end", -1)
        def test_owner(server): time.sleep(.05)
        def test_next():
            event("next_start", 1)
            time.sleep(.15)
            event("next_end", -1)
    """,
        test_b="""
        import time, pytest
        from conftest import event
        @pytest.mark.timing_cpu(2)
        def test_heavy():
            event("heavy_start", 2)
            time.sleep(.15)
            event("heavy_end", -2)
    """,
    )
    write_history(
        pytester.path / "history.json",
        {
            "test_a.py::test_owner": 1,
            "test_a.py::test_next": 1,
            "test_b.py::test_heavy": 3,
        },
        stop=10,
    )
    result = run_timing(
        pytester,
        "-n2",
        "--dist",
        dist,
        "--timing-cpus",
        "3",
        "--timing-schedule",
        "history.json",
        timeout=15,
    )
    result.assert_outcomes(passed=3)
    assert event_peak(pytester) <= 3
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
@pytest.mark.parametrize("scope", ["session", "function"])
def test_dynamic_only_declarations_activate_admission(
    pytester: pytest.Pytester, scope: str
) -> None:
    pytester.makepyfile(
        EVENTS
        + f"""
@pytest.fixture(scope="{scope}")
@pytest_timing.cpu(1000000)
def heavy():
    event("start", 1)
    time.sleep(.1)
    event("end", -1)
def test_a(request): request.getfixturevalue("heavy")
def test_b(request): request.getfixturevalue("heavy")
"""
    )
    write_history(
        pytester.path / "history.json",
        {f"{pytester.path.name}.py::{name}": 1 for name in ["test_a", "test_b"]},
        stop=2,
    )
    result = run_timing(pytester, "-n2", "--timing-schedule", "history.json", timeout=15)
    result.assert_outcomes(passed=2)
    assert load_json(pytester.path)["run"]["cpu"]["gated"]
    assert event_peak(pytester) == 1


@needs_xdist
@pytest.mark.skipif(
    importlib.util.find_spec("pytest_rerunfailures") is None, reason="rerun plugin not installed"
)
def test_timed_out_fixture_can_be_retried_after_capacity_returns(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("""
        import pytest_timing.demand
        pytest_timing.demand.REQUEST_TIMEOUT = .04
    """)
    pytester.makepyfile(
        EVENTS
        + """
@pytest.fixture(scope="session")
@pytest_timing.cpu(2)
def heavy():
    event("setup")
    return 42
def test_a_competing():
    Path("running").touch()
    time.sleep(.15)
@pytest.mark.flaky(reruns=1, reruns_delay=.2)
def test_b_dynamic(request):
    wait_for("running")
    assert request.getfixturevalue("heavy") == 42
"""
    )
    result = run_timing(pytester, "-p", "rerunfailures", "-n2", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=2)
    assert result.parseoutcomes()["rerun"] == 1
    assert len(events(pytester)) == 1


@needs_xdist
def test_package_reentry_reserves_the_repeated_setup(pytester: pytest.Pytester) -> None:
    for package in ("pkg_a", "pkg_b"):
        (pytester.mkdir(package) / "__init__.py").write_text("")
    pytester.makepyfile(eventlog=EVENTS)
    files = {
        "pkg_a/conftest": """
            import time, pytest, pytest_timing
            from eventlog import event
            @pytest.fixture(scope="package")
            @pytest_timing.cpu(2)
            def costly():
                event("package_start", 2)
                time.sleep(.15)
                event("package_end", -2)
                yield
        """,
        "pkg_a/test_a": """
            import time, pytest
            @pytest.fixture(scope="module")
            def local(): time.sleep(.005)
            def test_a(costly, local): time.sleep(.06)
        """,
        "pkg_b/test_b": """
            import time, pytest
            @pytest.fixture(scope="module")
            def local(): time.sleep(.005)
            def test_b(local): time.sleep(.04)
        """,
        "pkg_a/test_c": """
            import time, pytest
            @pytest.fixture(scope="module")
            def local(): time.sleep(.005)
            def test_c(costly, local): time.sleep(.02)
        """,
        "pkg_b/test_long": """
            import time, pytest
            from eventlog import event
            @pytest.fixture(scope="module")
            def local(): time.sleep(.005)
            def test_long(local):
                event("long_start", 1)
                time.sleep(1.)
                event("long_end", -1)
        """,
    }
    pytester.makepyfile(**files)
    pytester.makeconftest("""
        def pytest_collection_modifyitems(items):
            order = {"test_a": 0, "test_b": 1, "test_c": 2, "test_long": 3}
            items.sort(key=lambda item: order[item.name])
    """)
    first = run_timing(pytester, timeout=15)
    first.assert_outcomes(passed=4)
    (pytester.path / "pytest-timing.json").rename(pytester.path / "history.json")
    for path in pytester.path.glob("events-*.jsonl"):
        path.unlink()
    result = run_timing(
        pytester, "-n2", "--timing-cpus", "2", "--timing-schedule", "history.json", timeout=15
    )
    result.assert_outcomes(passed=4)
    assert event_peak(pytester) <= 2
    assert load_json(pytester.path)["run"]["cpu"]["domains"]["local"]["forced"] == 0


@needs_xdist
@pytest.mark.skipif(
    importlib.util.find_spec("pytest_rerunfailures") is None, reason="rerun plugin not installed"
)
def test_repeated_timeouts_keep_waits_with_the_attempt_that_waited(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest("""
        import pytest_timing.demand
        pytest_timing.demand.REQUEST_TIMEOUT = .05
    """)
    pytester.makepyfile(
        EVENTS
        + """
@pytest.fixture(scope="session")
@pytest_timing.cpu(2)
def heavy(): return 42
def test_a_competing():
    # Hold the slot until the other test has asked for the fixture, then long
    # enough for that request to time out at least once, whatever the host's pace.
    Path("running").touch()
    wait_for("requested")
    time.sleep(.15)
@pytest.mark.flaky(reruns=4)
def test_b_dynamic(request):
    wait_for("running")
    Path("requested").touch()
    assert request.getfixturevalue("heavy") == 42
"""
    )
    result = run_timing(pytester, "-p", "rerunfailures", "-n2", "--timing-cpus", "2", timeout=15)
    result.assert_outcomes(passed=2)
    assert result.parseoutcomes()["rerun"] >= 1
    run = Run.from_dict(load_json(pytester.path))
    attempts = [t for t in run.tests if t.nodeid.endswith("::test_b_dynamic")]
    assert all(t.cpu and t.cpu.wait > 0.02 for t in attempts if t.outcome == "rerun")
    for attempt in attempts:
        if attempt.cpu and attempt.cpu.runtime_wait:
            assert attempt.cpu.wait == pytest.approx(attempt.cpu.runtime_wait, abs=0.03)
