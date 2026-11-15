"""Worker-only capture exercises pytest's lifecycle without per-test telemetry."""

from __future__ import annotations

import json

import pytest
from conftest import load_json, needs_xdist

from pytest_timing.cli import main, merge_runs
from pytest_timing.compare import compare_runs
from pytest_timing.model import Run


@needs_xdist
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("source", ["cli", "env", "ini"])
def test_workers_capture_skips_test_and_fixture_instrumentation(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, workers: int, source: str
) -> None:
    pytester.makeconftest("""
        import pytest_timing.plugin as plugin
        from pytest_timing.collector import Collector
        def unexpected(*args, **kwargs):
            raise AssertionError('workers capture installed per-test instrumentation')
        for name in ('FixtureTimer', 'CpuMeter', 'ProcessTreeClock', 'MemorySampler', 'host_cpu'):
            setattr(plugin, name, unexpected)
        Collector.add_report = unexpected
    """)
    pytester.makepyfile("""
        import pytest
        @pytest.fixture(scope='session')
        def shared():
            yield 42
        @pytest.mark.parametrize('i', range(100))
        def test_value(i, shared): assert shared == 42
    """)
    args = ["--timing-capture=Workers"] if source == "cli" else []
    if source == "env":
        monkeypatch.setenv("PYTEST_TIMING_CAPTURE", "Workers")
    if source == "ini":
        pytester.makeini("[pytest]\ntiming_capture = Workers\n")
    result = pytester.runpytest_subprocess(
        *args, "-n", str(workers), "--dist=worksteal", "--timing-json", timeout=60
    )
    result.assert_outcomes(passed=100)
    doc = load_json(pytester.path)
    assert doc["tests"] == []
    assert doc["run"]["workers_only"] is True
    assert doc["run"]["termination"] == "finished"
    assert doc["run"]["exit_status"] == 0
    assert doc["run"]["cpu"] is None and doc["run"]["memory"] is None
    assert len(doc["workers"]) == (workers or 1)
    if workers:
        assert doc["run"]["dist"] == "worksteal"
    for worker in doc["workers"]:
        assert 0 <= worker["start"] <= worker["ready"] <= worker["collected"] <= worker["down"]
        assert worker["items"] == 100 and worker["error"] is None
    out = result.stdout.str()
    assert "worker lifecycle capture" in out
    assert "busy%" not in out and "slowest" not in out
    assert len((pytester.path / "pytest-timing.json").read_bytes()) < 4096


@needs_xdist
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("scenario", ["failure", "empty", "crash"])
def test_workers_capture_preserves_outcomes(
    pytester: pytest.Pytester, workers: int, scenario: str
) -> None:
    source = {
        "failure": "def test_fail(): assert False",
        "empty": "",
        "crash": "def test_crash():\n    import os\n    os._exit(3)",
    }[scenario]
    if scenario == "crash" and workers == 0:
        pytest.skip("an in-process os._exit cannot produce a session summary")
    pytester.makepyfile(test_sample=source)
    result = pytester.runpytest_subprocess(
        "--timing-capture=workers",
        "--timing-json",
        "-n",
        str(workers),
        "--max-worker-restart=0",
        timeout=60,
    )
    expected = (
        pytest.ExitCode.NO_TESTS_COLLECTED if scenario == "empty" else pytest.ExitCode.TESTS_FAILED
    )
    assert result.ret == expected
    doc = load_json(pytester.path)
    assert doc["tests"] == []
    assert doc["run"]["exit_status"] == int(expected)
    assert doc["run"]["termination"] == ("aborted" if scenario == "crash" else "finished")
    assert any(worker["error"] is not None for worker in doc["workers"]) == (scenario == "crash")


@pytest.mark.parametrize(
    "option,value",
    [
        ("cpus", "2"),
        ("memory", "1G"),
        ("schedule", "history.json"),
        ("html", "true"),
        ("trace", "true"),
    ],
)
@pytest.mark.parametrize("source", ["cli", "env", "ini"])
def test_workers_capture_rejects_incompatible_options(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    value: str,
    source: str,
) -> None:
    pytester.makepyfile("def test_pass(): pass")
    args = []
    if source == "cli":
        args = [f"--timing-{option}" + ("" if option in ("html", "trace") else f"={value}")]
    elif source == "env":
        monkeypatch.setenv(f"PYTEST_TIMING_{option.upper()}", value)
    else:
        pytester.makeini(f"[pytest]\ntiming_{option} = {value}\n")
    result = pytester.runpytest_subprocess("--timing-capture=workers", *args)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "workers capture" in result.stderr.str()


def test_saved_worker_runs_keep_their_capture_contract(
    pytester: pytest.Pytester, capsys: pytest.CaptureFixture[str]
) -> None:
    pytester.makepyfile("def test_pass(): pass")
    result = pytester.runpytest_subprocess("--timing-capture=workers", "--timing-json")
    result.assert_outcomes(passed=1)
    path = pytester.path / "pytest-timing.json"
    run = Run.from_json(path.read_text())
    assert run.run.workers_only
    merged = merge_runs([run, run], ["a", "b"])
    assert merged.run.workers_only and merged.tests == []
    merged_path = pytester.path / "merged.json"
    merged_path.write_text(merged.to_json())
    assert main(["render", str(merged_path)]) == 0
    out = capsys.readouterr().out
    assert "worker lifecycle capture, 2 workers" in out
    assert "busy%" not in out
    with pytest.raises(ValueError, match="per-test comparisons"):
        compare_runs(run, run)
    for kind in ("html", "trace"):
        with pytest.raises(ValueError, match="JSON output only"):
            main(["render", str(path), f"--{kind}", str(pytester.path / f"out.{kind}")])
        assert not (pytester.path / f"out.{kind}").exists()
    plain_doc = json.loads(path.read_text())
    del plain_doc["run"]["workers_only"]
    plain = Run.from_dict(plain_doc)
    with pytest.raises(ValueError, match="cannot merge"):
        merge_runs([run, plain], ["a", "b"])
