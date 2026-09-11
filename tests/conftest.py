from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import Any

import pytest

from pytest_timing.collector import Collector, PhaseReport
from pytest_timing.model import Run, RunInfo

pytest_plugins = ["pytester"]

HAS_XDIST = importlib.util.find_spec("xdist") is not None
needs_xdist = pytest.mark.skipif(not HAS_XDIST, reason="pytest-xdist not installed")

T0 = 1_700_000_000.0


@pytest.fixture(autouse=True)
def utf8_subprocess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pytester decodes subprocess output as UTF-8; make the child encode it that way.

    Without this, a Windows child writes its terminal report in the locale code page
    (the report's "…" becomes 0x85) and ``runpytest_subprocess`` fails to decode it.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")


def make_info(**overrides: object) -> RunInfo:
    base: dict[str, object] = dict(
        start=T0,
        stop=T0,
        argv=["pytest", "-n", "2"],
        rootdir="/repo",
        python="3.12.0",
        pytest="8.0.0",
        xdist="3.6.0",
        dist="load",
        numprocesses=2,
    )
    base.update(overrides)
    return RunInfo(**base)  # type: ignore[arg-type]


def report(
    nodeid: str,
    when: str,
    start: float,
    stop: float,
    *,
    worker: str = "gw0",
    outcome: str = "passed",
    wasxfail: bool = False,
) -> PhaseReport:
    """A phase report with times given relative to T0."""
    return PhaseReport(
        nodeid=nodeid,
        when=when,
        outcome=outcome,
        start=T0 + start,
        stop=T0 + stop,
        duration=stop - start,
        worker=worker,
        wasxfail=wasxfail,
        received=T0 + stop,
    )


def full_test(
    collector: Collector,
    nodeid: str,
    start: float,
    *,
    worker: str = "gw0",
    setup: float = 0.1,
    call: float = 0.5,
    teardown: float = 0.1,
    outcome: str = "passed",
) -> None:
    t = start
    collector.add_report(report(nodeid, "setup", t, t + setup, worker=worker))
    t += setup
    collector.add_report(report(nodeid, "call", t, t + call, worker=worker, outcome=outcome))
    t += call
    collector.add_report(report(nodeid, "teardown", t, t + teardown, worker=worker))


def crash_report(nodeid: str, received: float, *, worker: str = "gw0") -> PhaseReport:
    """xdist's synthetic report for an item that was pending when its worker died."""
    from pytest_timing.collector import CRASH_WHEN

    return PhaseReport(
        nodeid=nodeid,
        when=CRASH_WHEN,
        outcome="failed",
        start=0,
        stop=0,
        duration=0,
        worker=worker,
        received=T0 + received,
    )


def empty_run(termination: str = "finished", reason: str | None = None) -> Run:
    return Collector(make_info()).finish(T0 + 1, termination=termination, reason=reason)


def shifted_copy(run: Run, shift: float, rename: dict[str, str] | None = None) -> Run:
    """A copy of ``run`` that started ``shift`` seconds later, optionally with renamed workers."""
    doc = run.to_dict()
    doc["run"]["start"] += shift
    doc["run"]["stop"] += shift
    for w in doc["workers"]:
        w["id"] = (rename or {}).get(w["id"], w["id"])
    for t in doc["tests"]:
        t["worker"] = (rename or {}).get(t["worker"], t["worker"])
    return Run.from_dict(doc)


def chart_columns(line: str, run: Run) -> str:
    """The chart part of a rendered lane row, using the renderer's own widths."""
    label_width = max([len(lane.id) for lane in run.lanes()] + [6])
    return line[label_width + 1 : -8]  # "<label> <chart>  busy%"


def load_json(path: pathlib.Path, name: str = "pytest-timing.json") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((path / name).read_text())
    return data


def run_timing(
    pytester: pytest.Pytester, *args: str, timeout: float | None = None
) -> pytest.RunResult:
    """Run a behavior scenario with JSON timing and no pytest cache."""
    return pytester.runpytest_subprocess(
        *args, "--timing-json", "-p", "no:cacheprovider", timeout=timeout
    )


def write_history(path: pathlib.Path, durations: dict[str, float], *, stop: float) -> None:
    """Write a duration-only run; insertion order determines collection order."""
    collector = Collector(make_info())
    for index, (nodeid, duration) in enumerate(durations.items()):
        full_test(collector, nodeid, index, setup=0, call=duration, teardown=0)
    path.write_text(collector.finish(T0 + stop, termination="finished").to_json())


def events(pytester: pytest.Pytester) -> list[Any]:
    """Every recorded event across all worker files, in time order."""
    return sorted(
        json.loads(line)
        for path in pytester.path.glob("events-*.jsonl")
        for line in path.read_text().splitlines()
    )


def event_peak(pytester: pytest.Pytester) -> int:
    """Find the peak declared slots and require every recorded hold to be released."""
    used = peak = 0
    for _, _, _, delta in events(pytester):
        used += delta
        peak = max(peak, used)
    assert used == 0
    return peak


@pytest.fixture
def sample_run() -> Run:
    """Two workers, six tests, one failure, one skip, realistic lifecycle."""
    c = Collector(make_info())
    for wid, ready in (("gw0", 0.2), ("gw1", 0.25)):
        c.worker_started(wid, T0 + 0.05)
        c.worker_ready(wid, T0 + ready)
        c.worker_collected(wid, T0 + ready + 0.3, 6)
    full_test(c, "tests/test_a.py::test_one", 0.6, worker="gw0")
    full_test(c, "tests/test_a.py::test_two", 0.6, worker="gw1", call=1.2)
    full_test(c, "tests/test_a.py::test_three", 1.3, worker="gw0", outcome="failed")
    full_test(c, "tests/test_b.py::test_four", 2.0, worker="gw1", call=0.05)
    # skipped in setup: no call phase
    c.add_report(
        report("tests/test_b.py::test_skip", "setup", 2.1, 2.15, worker="gw0", outcome="skipped")
    )
    c.add_report(report("tests/test_b.py::test_skip", "teardown", 2.15, 2.16, worker="gw0"))
    full_test(c, "tests/test_b.py::test_five", 2.2, worker="gw1", call=0.3)
    c.worker_down("gw0", T0 + 2.4, None)
    c.worker_down("gw1", T0 + 2.9, None)
    return c.finish(T0 + 3.0, termination="finished")


def chrome_binary() -> str | None:
    """A headless-capable Chrome/Chromium, if one is installed."""
    import shutil

    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    return None


def render_in_chrome(html_path: pathlib.Path) -> str:
    """Load a report in headless Chrome and return the rendered DOM."""
    import subprocess

    binary = chrome_binary()
    assert binary is not None
    proc = subprocess.run(
        [binary, "--headless", "--disable-gpu", "--no-sandbox", "--dump-dom", html_path.as_uri()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout
