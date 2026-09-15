from __future__ import annotations

import re

import pytest
from conftest import chart_columns, empty_run, make_info

from pytest_timing.collector import Collector
from pytest_timing.model import Run
from pytest_timing.render.ascii import format_seconds, render_ascii

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_header_and_lanes(sample_run: Run) -> None:
    out = render_ascii(sample_run, width=80, top=3)
    lines = out.splitlines()
    assert lines[0].startswith("pytest-timing: 6 tests (1 failed), 2 workers, wall 3.00s, busy ")
    assert lines[1].startswith("worker |")
    assert lines[1].endswith("busy%")
    gw0 = next(line for line in lines if line.startswith("gw0 "))
    gw1 = next(line for line in lines if line.startswith("gw1 "))
    assert "X" in gw0 and "X" not in gw1  # the failure is on gw0
    assert "█" in gw1
    assert "░" in gw0  # boot band before ready
    assert gw0.rstrip().endswith("%")
    assert "legend:" in out
    assert "slowest 3 tests" in out
    assert "tests/test_a.py::test_two" in out  # the longest test
    # every chart line fits the requested width
    for line in lines:
        assert len(line) <= 80, line


def test_lane_rows_align_with_axis(sample_run: Run) -> None:
    out = render_ascii(sample_run, width=72, top=0)
    lines = out.splitlines()
    axis = lines[1]
    chart_start = axis.index("|")
    chart_end = axis.rindex("-") + 1
    for line in lines[2:4]:
        row = line[chart_start:chart_end]
        assert len(row) == chart_end - chart_start
    assert "slowest" not in out


def test_ascii_style_has_no_unicode(sample_run: Run) -> None:
    out = render_ascii(sample_run, width=80, style="ascii")
    assert out.isascii()
    assert "#" in out and "X" in out


def test_min_duration_filters_slowest(sample_run: Run) -> None:
    out = render_ascii(sample_run, width=100, top=10, min_duration=1.0)
    assert "slowest 1 tests" in out
    assert "test_two" in out and "test_four" not in out


def test_slowest_rows_say_how_long_admission_held_a_test(sample_run: Run) -> None:
    from pytest_timing.model import CpuRecord, MemoryRecord

    two = next(t for t in sample_run.tests if t.nodeid.endswith("test_two"))
    two.memory = MemoryRecord(base=1, peak=2, after=1, coverage="self", wait=0.3)
    two.cpu = CpuRecord(elapsed=1.0, wait=1.25)
    out = render_ascii(sample_run, width=100, top=2)
    assert "tests/test_a.py::test_two  waited 1.25s for cpu, waited 0.30s for memory" in out
    for line in out.splitlines():
        assert len(line) <= 100, line
    assert "waited" not in render_ascii(sample_run, width=100, top=0)


def test_color_only_wraps_glyphs(sample_run: Run) -> None:
    plain = render_ascii(sample_run, width=80)
    colored = render_ascii(sample_run, width=80, color=True)
    assert "\x1b[" in colored
    assert ANSI.sub("", colored) == plain


def test_empty_run() -> None:
    out = render_ascii(empty_run(), width=60)
    assert "0 tests" in out
    assert "no tests recorded" in out


def test_narrow_width_still_renders(sample_run: Run) -> None:
    out = render_ascii(sample_run, width=20)
    assert "gw0" in out and "gw1" in out


@pytest.mark.parametrize("style, mark", [("unicode", "…"), ("ascii", "...")])
def test_long_nodeid_is_truncated_from_the_left(sample_run: Run, style: str, mark: str) -> None:
    sample_run.tests[1].nodeid = "tests/" + "very_long_module_name_" * 6 + "py::test_two"
    out = render_ascii(sample_run, width=80, top=1, style=style)
    bar, last = out.splitlines()[-2:]
    assert len(bar) <= 80 and len(last) <= 80
    assert bar.startswith("gw1 ") and bar.rstrip().endswith("1.40s")
    assert last.endswith("py::test_two") and last.lstrip().startswith(mark)
    if style == "ascii":
        out.encode("ascii")  # must not raise


def test_format_seconds() -> None:
    assert format_seconds(0.5) == "0.50s"
    assert format_seconds(12.34) == "12.3s"
    assert format_seconds(75) == "1m15s"
    assert format_seconds(3725) == "1h02m"


def test_overlaps_keep_the_tail_of_an_interval() -> None:
    """0.8s-1.4s over 1s columns must credit 0.2s to column 0 and 0.4s to column 1."""
    from pytest_timing.render.ascii import Scale

    scale = Scale(total=4.0, columns=4)
    buckets = [0.0] * 4
    scale.overlaps(0.8, 1.4, buckets)
    assert [round(b, 6) for b in buckets] == [0.2, 0.4, 0.0, 0.0]
    buckets = [0.0] * 4
    scale.overlaps(0.0, 4.0, buckets)
    assert [round(b, 6) for b in buckets] == [1.0, 1.0, 1.0, 1.0]


def test_boot_band_starts_at_worker_start(sample_run: Run) -> None:
    # launched at 0.3s, ready at 0.5s, before its first test at 0.6s
    sample_run.workers[0].start = 0.3
    sample_run.workers[0].ready = 0.5
    out = render_ascii(sample_run, width=80, top=0, style="ascii")
    gw0 = next(line for line in out.splitlines() if line.startswith("gw0 "))
    chart = chart_columns(gw0, sample_run)
    assert chart[0] == " "  # nothing before the worker started
    first_boot = chart.index(".")
    assert 0 < first_boot < chart.index("#")


def test_rerun_counts_as_failure_in_chart() -> None:
    from conftest import T0, full_test, report

    c = Collector(make_info())
    c.worker_ready("gw0", T0)
    c.add_report(report("t.py::flaky", "setup", 0, 0.1))
    c.add_report(report("t.py::flaky", "call", 0.1, 1.0, outcome="rerun"))
    c.add_report(report("t.py::flaky", "teardown", 1.0, 1.1))
    full_test(c, "t.py::flaky", 1.2)
    c.worker_down("gw0", T0 + 2.0, None)
    run = c.finish(T0 + 2.0, termination="finished")
    out = render_ascii(run, width=60, top=0)
    assert "(1 rerun)" in out.splitlines()[0]
    assert "X" in out.splitlines()[2]


def test_overlaps_sum_to_clipped_duration() -> None:
    """Contract: bucket overlaps add up to exactly the part of the interval on the axis."""
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from pytest_timing.render.ascii import Scale

    @settings(max_examples=200, deadline=None)
    @given(
        total=st.floats(0.001, 5000),
        columns=st.integers(1, 300),
        a=st.floats(-10, 6000),
        b=st.floats(-10, 6000),
    )
    def check(total: float, columns: int, a: float, b: float) -> None:
        start, stop = min(a, b), max(a, b)
        scale = Scale(total, columns)
        buckets = [0.0] * columns
        scale.overlaps(start, stop, buckets)
        clipped = max(0.0, min(stop, total) - max(start, 0.0))
        assert abs(sum(buckets) - clipped) <= 1e-6 * max(1.0, clipped)
        assert all(0 <= v <= scale.per_column + 1e-9 for v in buckets)

    check()


def test_unknown_launch_time_draws_no_boot_band(sample_run: Run) -> None:
    sample_run.workers[0].start = None
    out = render_ascii(sample_run, width=80, top=0, style="ascii")
    gw0 = next(line for line in out.splitlines() if line.startswith("gw0 "))
    gw1 = next(line for line in out.splitlines() if line.startswith("gw1 "))
    assert "." not in chart_columns(gw0, sample_run)
    assert "." in chart_columns(gw1, sample_run)


def test_incomplete_header_names_the_termination() -> None:
    run = empty_run("aborted", "gw1: differs")
    assert "ended: aborted (gw1: differs)" in render_ascii(run, width=80).splitlines()[0]
