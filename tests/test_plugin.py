"""End-to-end contracts through pytester: option handling, lifecycle, xdist."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import pytest
from conftest import load_json, needs_xdist

HAS_RERUNS = importlib.util.find_spec("pytest_rerunfailures") is not None
needs_reruns = pytest.mark.skipif(not HAS_RERUNS, reason="pytest-rerunfailures not installed")
PYTEST_VERSION = tuple(int(x) for x in pytest.__version__.split(".")[:2])
needs_argfiles = pytest.mark.skipif(PYTEST_VERSION < (8, 2), reason="@file needs pytest 8.2")

SUITE = """
import time, pytest

@pytest.fixture
def slow_fixture():
    time.sleep(0.01)
    yield
    time.sleep(0.005)

@pytest.mark.parametrize("i", range(4))
def test_sleep(i, slow_fixture):
    time.sleep(0.01)

def test_fail():
    assert 0

@pytest.mark.skip
def test_skip():
    pass

@pytest.mark.xfail
def test_xfail():
    assert 0
"""


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makepyfile(test_suite=SUITE)
    pytester.makepyfile(test_other="def test_other():\n    pass\n")
    return pytester


def test_disabled_by_default(suite: pytest.Pytester) -> None:
    result = suite.runpytest("-p", "no:cacheprovider")
    assert "timing report" not in result.stdout.str()
    assert not (suite.path / "pytest-timing.html").exists()


def test_single_process_ascii(suite: pytest.Pytester) -> None:
    result = suite.runpytest(
        "--timing", "--timing-width=80", "-p", "no:cacheprovider", "test_suite.py"
    )
    result.assert_outcomes(passed=4, failed=1, skipped=1, xfailed=1)
    out = result.stdout.str()
    assert "timing report" in out
    assert re.search(r"pytest-timing: 7 tests \(1 failed\), 1 worker, wall", out)
    assert re.search(r"^main\s+\S", out, re.M)
    assert "slowest" in out
    assert "test_suite.py::test_sleep[" in out


def test_ini_and_env_enable(suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    suite.makeini("[pytest]\ntiming_top = 2\ntiming_ascii_style = ascii\n")
    monkeypatch.setenv("PYTEST_TIMING", "1")
    result = suite.runpytest("-p", "no:cacheprovider", "--timing-width=70")
    out = result.stdout.str()
    assert "timing report" in out
    assert "slowest 2 tests" in out
    assert out[out.index("timing report") :].isascii()


def _assert_selection_preserved(suite: pytest.Pytester, result: pytest.RunResult) -> None:
    """Only test_suite.py was selected: test_other.py must not have run."""
    assert result.ret != pytest.ExitCode.USAGE_ERROR, result.stderr.str()
    out = result.stdout.str()
    assert "test_other" not in out
    assert "7 tests" in out


@pytest.mark.parametrize("kind", ["json", "html", "trace"])
def test_bare_flag_keeps_selection_on_cli(suite: pytest.Pytester, kind: str) -> None:
    target = suite.path / "test_suite.py"
    before = target.read_bytes()
    result = suite.runpytest(f"--timing-{kind}", "test_suite.py", "-p", "no:cacheprovider")
    _assert_selection_preserved(suite, result)
    assert target.read_bytes() == before
    default = {
        "json": "pytest-timing.json",
        "html": "pytest-timing.html",
        "trace": "pytest-timing.trace.json",
    }[kind]
    assert (suite.path / default).exists()


@pytest.mark.parametrize("source", ["cli", "env", "ini"])
@pytest.mark.parametrize("value", ["light", "Light"])
def test_light_capture_omits_optional_metrics(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, source: str, value: str
) -> None:
    pytester.makeconftest("""
        import pytest_timing.plugin, pytest_timing.demand
        def unexpected(*args, **kwargs):
            raise AssertionError("light capture constructed an optional telemetry reader")
        pytest_timing.plugin.ProcessTreeClock = unexpected
        pytest_timing.demand.ProcessTreeClock = unexpected
        pytest_timing.plugin.MemorySampler = unexpected
    """)
    pytester.makepyfile("""
        import pytest
        @pytest.fixture(scope="module")
        def shared(): return 42
        def test_one(shared): assert shared == 42
    """)
    pytester.makeini("[pytest]\ntiming_capture = " + (value if source == "ini" else "full"))
    if source in ("cli", "env"):
        monkeypatch.setenv("PYTEST_TIMING_CAPTURE", "full" if source == "cli" else value)
    args = ["--timing-capture=" + value] if source == "cli" else []
    result = pytester.runpytest_subprocess(*args, "--timing-json", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    (test,) = load_json(pytester.path)["tests"]
    assert "cpu" not in test and "memory" not in test
    assert test["fixtures"] and test["phases"]["call"][2] >= 0


def test_trace_only_plugin_does_not_build_a_json_document(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("""
        from pytest_timing.model import Run
        def unexpected(self): raise AssertionError("unused document")
        Run.to_dict = unexpected
    """)
    pytester.makepyfile("def test_one(): pass")
    result = pytester.runpytest_subprocess("--timing-trace", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    assert json.loads((pytester.path / "pytest-timing.trace.json").read_text())["traceEvents"]


def test_file_option_takes_explicit_path(suite: pytest.Pytester) -> None:
    result = suite.runpytest(
        "--timing-json-file",
        "out/run.json",
        "--timing-html-file=out/r.html",
        "test_suite.py",
        "-p",
        "no:cacheprovider",
    )
    _assert_selection_preserved(suite, result)
    assert (suite.path / "out" / "run.json").exists()
    assert (suite.path / "out" / "r.html").exists()


def test_file_option_requires_a_value(suite: pytest.Pytester) -> None:
    target = suite.path / "test_suite.py"
    before = target.read_bytes()
    result = suite.runpytest("--timing-json-file", "-p", "no:cacheprovider", "test_suite.py")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert target.read_bytes() == before


def test_bare_flag_via_ini_addopts(suite: pytest.Pytester) -> None:
    suite.makeini("[pytest]\naddopts = --timing-json\n")
    result = suite.runpytest("test_suite.py", "-p", "no:cacheprovider")
    _assert_selection_preserved(suite, result)
    assert (suite.path / "pytest-timing.json").exists()


def test_paths_via_ini_and_env(suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    suite.makeini("[pytest]\ntiming_json = from-ini.json\ntiming_html = true\n")
    monkeypatch.setenv("PYTEST_TIMING_TRACE", "from-env.json")
    result = suite.runpytest("test_suite.py", "-p", "no:cacheprovider")
    _assert_selection_preserved(suite, result)
    assert (suite.path / "from-ini.json").exists()
    assert (suite.path / "pytest-timing.html").exists()
    assert (suite.path / "from-env.json").exists()


def test_bare_flag_via_pytest_addopts_env(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--timing-trace")
    result = suite.runpytest("test_suite.py", "-p", "no:cacheprovider")
    _assert_selection_preserved(suite, result)
    assert (suite.path / "pytest-timing.trace.json").exists()


@needs_argfiles
def test_bare_flag_inside_argument_file(suite: pytest.Pytester) -> None:
    (suite.path / "args.txt").write_text("--timing-json\ntest_suite.py\n-p\nno:cacheprovider\n")
    target = suite.path / "test_suite.py"
    before = target.read_bytes()
    result = suite.runpytest("@args.txt")
    _assert_selection_preserved(suite, result)
    assert target.read_bytes() == before


def test_bare_flag_with_late_plugin_loading(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loaded from conftest with autoload off, the options behave identically."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    suite.makeconftest('pytest_plugins = ["pytest_timing.plugin"]')
    target = suite.path / "test_suite.py"
    before = target.read_bytes()
    result = suite.runpytest("--timing-json", "test_suite.py")
    _assert_selection_preserved(suite, result)
    assert target.read_bytes() == before
    assert (suite.path / "pytest-timing.json").exists()


def test_finished_run(suite: pytest.Pytester) -> None:
    suite.runpytest("--timing-json", "test_suite.py", "-p", "no:cacheprovider")
    doc = load_json(suite.path)
    assert doc["run"]["termination"] == "finished"
    assert doc["run"]["complete"] is True
    assert doc["run"]["exit_status"] == int(pytest.ExitCode.TESTS_FAILED)


def test_stop_on_first_failure_is_interrupted(suite: pytest.Pytester) -> None:
    result = suite.runpytest("-x", "--timing-json", "test_suite.py", "-p", "no:cacheprovider")
    doc = load_json(suite.path)
    assert doc["run"]["termination"] == "interrupted"
    assert "stopping after 1 failures" in (doc["run"]["reason"] or "")
    assert doc["run"]["complete"] is False
    assert "ended: interrupted (stopping after 1 failures)" in result.stdout.str()


def test_collect_only_is_complete(suite: pytest.Pytester) -> None:
    suite.runpytest("--collect-only", "--timing-json", "-p", "no:cacheprovider")
    doc = load_json(suite.path)
    assert doc["tests"] == []
    assert doc["run"]["termination"] == "collect_only"
    assert doc["run"]["complete"] is True


def test_keep_duplicates_are_separate_occurrences(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_repeat="def test_one():\n    pass\n")
    pytester.runpytest(
        "--keep-duplicates",
        "test_repeat.py",
        "test_repeat.py",
        "--timing-json",
        "-p",
        "no:cacheprovider",
    )
    doc = load_json(pytester.path)
    assert [(t["occurrence"], t["attempt"]) for t in doc["tests"]] == [(0, 0), (1, 0)]
    assert doc["run"]["complete"] is True


EXIT_SUITE = """
import pytest
state = {}
@pytest.mark.flaky(reruns=1)
def test_flaky():
    state["n"] = state.get("n", 0) + 1
    assert state["n"] > 1
def test_exit():
    pytest.exit("stopping", returncode=0)
def test_never_runs():
    pass
"""


@needs_reruns
@pytest.mark.parametrize(
    "extra, expected",
    [
        # pytest.exit() reaches pytest_keyboard_interrupt in a single process
        ([], "interrupted"),
        # under xdist a worker exiting mid-run trips an assertion inside xdist's
        # controller (with or without retries) and pytest reports an internal error
        pytest.param(["-n", "1"], "internal_error", marks=needs_xdist),
    ],
)
def test_exit_is_recorded_even_with_retries_and_zero_status(
    pytester: pytest.Pytester, extra: list[str], expected: str
) -> None:
    pytester.makepyfile(test_exit=EXIT_SUITE)
    pytester.runpytest_subprocess("--timing-json", "-p", "no:cacheprovider", *extra)
    doc = load_json(pytester.path)
    spans = [
        (t["nodeid"].split("::")[1], t["occurrence"], t["attempt"], t["outcome"])
        for t in doc["tests"]
    ]
    assert ("test_flaky", 0, 0, "rerun") in spans
    assert ("test_flaky", 0, 1, "passed") in spans
    assert not any(s[0] == "test_never_runs" for s in spans)
    assert doc["run"]["termination"] == expected
    assert doc["run"]["complete"] is False


def test_keyboard_interrupt_is_recorded(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_kb="""
        def test_first():
            raise KeyboardInterrupt()
        def test_second():
            pass
        """
    )
    pytester.runpytest_subprocess("--timing-json", "-p", "no:cacheprovider")
    doc = load_json(pytester.path)
    assert doc["run"]["termination"] == "interrupted"
    assert "KeyboardInterrupt" in doc["run"]["reason"]
    assert doc["run"]["complete"] is False


@needs_xdist
def test_xdist_records_every_worker(suite: pytest.Pytester) -> None:
    result = suite.runpytest(
        "-n",
        "2",
        "--timing",
        "--timing-json",
        "--timing-width=100",
        "-p",
        "no:cacheprovider",
        "test_suite.py",
    )
    result.assert_outcomes(passed=4, failed=1, skipped=1, xfailed=1)
    out = result.stdout.str()
    assert "2 workers" in out
    assert "JSON written to pytest-timing.json" in out

    doc = load_json(suite.path)
    assert doc["schema"] == 1
    assert doc["run"]["dist"] == "load"
    assert doc["run"]["numprocesses"] == 2
    assert doc["run"]["xdist"]
    assert doc["run"]["termination"] == "finished"
    assert [w["id"] for w in doc["workers"]] == ["gw0", "gw1"]
    for worker in doc["workers"]:
        assert worker["ready"] is not None
        assert worker["collected"] is not None
        assert worker["items"] == 7
        assert worker["down"] is not None
        assert worker["error"] is None
        assert worker["ready"] <= worker["collected"] <= worker["down"]
    tests = doc["tests"]
    assert len(tests) == 7
    assert {t["worker"] for t in tests} == {"gw0", "gw1"}
    outcomes = {t["nodeid"].split("::")[1]: t["outcome"] for t in tests}
    assert outcomes["test_fail"] == "failed"
    assert outcomes["test_skip"] == "skipped"
    assert outcomes["test_xfail"] == "xfailed"
    sleep = next(t for t in tests if t["nodeid"].endswith("test_sleep[0]"))
    assert set(sleep["phases"]) == {"setup", "call", "teardown"}
    assert sleep["phases"]["call"][2] >= 0.01
    assert sleep["phases"]["setup"][2] >= 0.01
    for t in tests:
        assert 0 <= t["start"] <= t["stop"] <= doc["run"]["stop"] - doc["run"]["start"] + 0.01


@needs_xdist
def test_all_outputs_and_paths(suite: pytest.Pytester) -> None:
    out_dir = suite.path / "reports"
    result = suite.runpytest(
        "-n",
        "2",
        f"--timing-html-file={out_dir / 'timing.html'}",
        f"--timing-trace-file={out_dir / 'timing.trace.json'}",
        "--timing-top=0",
        "-p",
        "no:cacheprovider",
    )
    text = result.stdout.str()
    assert "timing report" in text
    assert "slowest" not in text
    assert "HTML report written to" in text
    assert "Chrome trace written to" in text
    html = (out_dir / "timing.html").read_text()
    assert "pytest-timing-data" in html
    assert "test_suite.py::test_fail" in html
    trace = json.loads((out_dir / "timing.trace.json").read_text())
    assert any(e.get("cat") == "test" for e in trace["traceEvents"])


CRASH_SUITE = """
import os, time
def test_ok():
    time.sleep(0.05)
def test_boom():
    os._exit(1)
def test_after():
    time.sleep(0.05)
"""


@needs_xdist
def test_worker_crash_with_restarts_exhausted_is_aborted(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_crash=CRASH_SUITE)
    result = pytester.runpytest_subprocess(
        "-n", "1", "--timing", "--timing-json", "--max-worker-restart=0", "-p", "no:cacheprovider"
    )
    doc = load_json(pytester.path)
    outcomes = {t["nodeid"].split("::")[1]: t["outcome"] for t in doc["tests"]}
    assert outcomes["test_boom"] == "crashed"
    assert "test_after" not in outcomes
    assert doc["workers"][0]["error"]
    assert doc["run"]["termination"] == "aborted"
    assert "restarting disabled" in doc["run"]["reason"]
    assert doc["run"]["complete"] is False
    assert "ended: aborted" in result.stdout.str()


@needs_xdist
def test_worker_crash_with_recovery_is_finished(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_crash=CRASH_SUITE)
    pytester.runpytest_subprocess(
        "-n", "1", "--timing-json", "--max-worker-restart=1", "-p", "no:cacheprovider"
    )
    doc = load_json(pytester.path)
    outcomes = {t["nodeid"].split("::")[1]: t["outcome"] for t in doc["tests"]}
    assert outcomes["test_boom"] == "crashed"
    assert outcomes["test_after"] == "passed"
    assert [w["id"] for w in doc["workers"]] == ["gw0", "gw1"]  # the replacement worker
    assert doc["workers"][0]["error"]
    assert doc["run"]["termination"] == "finished"
    assert doc["run"]["complete"] is True


MISMATCH_SUITE = """
import os, pytest
@pytest.mark.parametrize("i", [os.getpid()])
def test_differs(i):
    pass
def test_same():
    pass
"""


@needs_xdist
def test_collection_mismatch_is_aborted(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_mm=MISMATCH_SUITE)
    result = pytester.runpytest_subprocess("-n", "2", "--timing-json", "-p", "no:cacheprovider")
    assert result.ret != 0
    doc = load_json(pytester.path)
    assert doc["run"]["termination"] == "aborted"
    assert doc["run"]["complete"] is False
    assert "Different tests were collected" in doc["run"]["reason"]
    assert doc["run"]["reason"].startswith("gw")
    assert doc["tests"] == []


def test_ordinary_failure_after_normal_execution_is_finished(suite: pytest.Pytester) -> None:
    result = suite.runpytest("--timing-json", "test_suite.py", "-p", "no:cacheprovider")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    doc = load_json(suite.path)
    assert doc["run"]["termination"] == "finished"
    assert doc["run"]["complete"] is True


@needs_xdist
def test_all_workers_record_a_launch_time(suite: pytest.Pytester) -> None:
    suite.runpytest_subprocess(
        "-n", "2", "--timing-json", "test_suite.py", "-p", "no:cacheprovider"
    )
    doc = load_json(suite.path)
    for worker in doc["workers"]:
        assert worker["start"] is not None
        assert 0 <= worker["start"] < worker["ready"]


@needs_xdist
def test_replacement_worker_boot_starts_at_its_launch(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_crash="""
        import os, time
        def test_slow():
            time.sleep(0.4)
        def test_boom():
            os._exit(1)
        def test_after():
            pass
        """
    )
    pytester.runpytest_subprocess(
        "-n", "1", "--timing-json", "--max-worker-restart=1", "-p", "no:cacheprovider"
    )
    doc = load_json(pytester.path)
    gw0, gw1 = doc["workers"]
    assert gw1["id"] == "gw1" and gw1["error"] is None
    slow = next(t for t in doc["tests"] if t["nodeid"].endswith("test_slow"))
    # launched after gw0 died, so its boot span excludes gw0's execution
    assert gw1["start"] >= gw0["down"] >= slow["stop"]
    assert gw1["start"] > slow["start"] + 0.4
    assert gw1["ready"] - gw1["start"] > 0


posix_only = pytest.mark.skipif(os.name != "posix", reason="uses a shell wrapper script")


@needs_xdist
@posix_only
def test_boot_time_includes_interpreter_startup(suite: pytest.Pytester) -> None:
    """A worker interpreter that takes 400 ms to start must show that in its boot span."""
    wrapper = suite.path / "slowpython"
    wrapper.write_text(f'#!/bin/sh\nsleep 0.4\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    tx = f"popen//python={wrapper}"
    suite.runpytest_subprocess(
        "--dist=load",
        "--tx",
        tx,
        "--tx",
        tx,
        "--timing-json",
        "test_other.py",
        "-p",
        "no:cacheprovider",
    )
    doc = load_json(suite.path)
    gw0, gw1 = doc["workers"]
    for worker in (gw0, gw1):
        assert worker["ready"] - worker["start"] >= 0.4, worker
    # gateways are created one after another: gw1's launch begins once gw0 exists
    assert gw1["start"] >= gw0["start"] + 0.4
    assert doc["run"]["termination"] == "finished"


@needs_xdist
def test_worker_id_colliding_with_a_failing_file_is_not_a_mismatch(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(test_bad="import nosuchmodule_for_pytest_timing\n")
    pytester.makepyfile(test_ok="def test_ok():\n    pass\n")
    result = pytester.runpytest_subprocess(
        "--dist=load", "--tx", "popen//id=test_bad.py", "--timing-json", "-p", "no:cacheprovider"
    )
    assert result.ret != 0  # the collection error is still an error
    doc = load_json(pytester.path)
    assert [w["id"] for w in doc["workers"]] == ["test_bad.py"]
    assert len(doc["tests"]) == 1
    assert doc["run"]["termination"] == "finished"
    assert doc["run"]["complete"] is True


@pytest.mark.parametrize("alias", ["yes", "true", "1", "on"])
def test_explicit_file_paths_are_never_boolean_aliases(suite: pytest.Pytester, alias: str) -> None:
    result = suite.runpytest(
        f"--timing-json-file={alias}",
        "--timing-html-file",
        alias + ".d/r",
        "--timing-trace-file",
        alias,
        "test_suite.py",
        "-p",
        "no:cacheprovider",
    )
    _assert_selection_preserved(suite, result)
    assert (suite.path / alias).exists()  # json, then overwritten by the trace: same name
    assert (suite.path / f"{alias}.d" / "r").exists()
    assert not (suite.path / "pytest-timing.json").exists()
    assert not (suite.path / "pytest-timing.html").exists()


def test_env_and_ini_boolean_aliases_mean_the_default_file(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite.makeini("[pytest]\ntiming_json = on\n")
    monkeypatch.setenv("PYTEST_TIMING_TRACE", "yes")
    suite.runpytest("test_suite.py", "-p", "no:cacheprovider")
    assert (suite.path / "pytest-timing.json").exists()
    assert (suite.path / "pytest-timing.trace.json").exists()


def _boot(worker: dict) -> float:  # type: ignore[type-arg]
    return float(worker["ready"] - worker["start"])


SLOW_PREVIOUS_WORKER = {
    "configure_node": """
        import time
        def pytest_configure_node(node):
            if node.gateway.id == "gw0":
                time.sleep(0.4)
        """,
    "configure_node_post_yield": """
        import time, pytest
        @pytest.hookimpl(hookwrapper=True)
        def pytest_configure_node(node):
            yield
            if node.gateway.id == "gw0":
                time.sleep(0.4)
        """,
    "getremotemodule": """
        import time, pytest
        calls = []
        @pytest.hookimpl(tryfirst=True)
        def pytest_xdist_getremotemodule():
            calls.append(1)
            if len(calls) == 1:
                time.sleep(0.4)
            return None  # let xdist supply the module
        """,
}


@needs_xdist
@pytest.mark.parametrize("stage", sorted(SLOW_PREVIOUS_WORKER))
def test_boot_time_excludes_work_done_for_the_previous_worker(
    suite: pytest.Pytester, stage: str
) -> None:
    """400 ms spent on gw0 after its creation must not appear in gw1's boot span."""
    suite.makeconftest(SLOW_PREVIOUS_WORKER[stage])
    suite.runpytest_subprocess(
        "-n", "2", "--timing-json", "test_other.py", "-p", "no:cacheprovider"
    )
    doc = load_json(suite.path)
    gw0, gw1 = doc["workers"]
    assert gw1["start"] >= gw0["start"] + 0.4  # gw1 was created after that work
    # gw0's span includes the delay (its readiness is only observed afterwards);
    # gw1's must be the plain spawn-and-bootstrap time, well below it.
    assert _boot(gw1) < _boot(gw0) - 0.3


@needs_xdist
def test_observer_is_removed_when_worker_startup_fails(suite: pytest.Pytester) -> None:
    """An invalid worker executable aborts before sessionfinish; cleanup must still run."""
    suite.makeconftest(
        """
        import pytest
        state = {}

        def pytest_configure(config):
            # Cleanups run last-in first-out, so this one runs after the plugin's.
            def check():
                group = state.get("group")
                verdict = "no-group" if group is None else (
                    "dirty" if "makegateway" in group.__dict__ else "clean")
                config.rootpath.joinpath("observer.txt").write_text(verdict)
            config.add_cleanup(check)

        def pytest_xdist_setupnodes(config, specs):
            state["group"] = config.pluginmanager.getplugin("dsession").nodemanager.group
        """
    )
    result = suite.runpytest_subprocess(
        "--dist=load",
        "--tx",
        "popen//python=/nonexistent/python",
        "--timing-json",
        "test_other.py",
        "-p",
        "no:cacheprovider",
    )
    assert result.ret != 0
    assert not (suite.path / "pytest-timing.json").exists()  # sessionfinish never ran
    assert (suite.path / "observer.txt").read_text() == "clean"
