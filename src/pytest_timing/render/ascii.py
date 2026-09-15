"""Terminal renderer: worker lanes plus a slowest-tests Gantt on a shared axis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby

from pytest_timing.model import BAD_OUTCOMES, Lane, Run, TestSpan

MIN_CHART = 20
TICK_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600)


@dataclass(frozen=True)
class Glyphs:
    boot: str
    collect: str
    idle: str
    test: str
    partial: str
    bad: str
    setup: str
    call: str
    teardown: str
    ellipsis: str


STYLES: dict[str, Glyphs] = {
    "unicode": Glyphs(
        boot="░",
        collect="▒",
        idle=" ",
        test="█",
        partial="▄",
        bad="X",
        setup="░",
        call="█",
        teardown="▒",
        ellipsis="…",
    ),
    "ascii": Glyphs(
        boot=".",
        collect=":",
        idle=" ",
        test="#",
        partial="+",
        bad="X",
        setup="-",
        call="#",
        teardown="=",
        ellipsis="...",
    ),
}

_RED = "\x1b[31m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if seconds < 3600:
        return f"{int(minutes)}m{rest:02.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def render_ascii(
    run: Run,
    *,
    width: int = 80,
    top: int = 10,
    min_duration: float = 0.0,
    style: str = "unicode",
    color: bool = False,
) -> str:
    glyphs = STYLES.get(style, STYLES["unicode"])
    lanes = run.lanes()
    label_width = max([len(lane.id) for lane in lanes] + [6])
    suffix_width = 8  # "  99.9%"
    chart = max(MIN_CHART, width - label_width - 1 - suffix_width)
    total = max(run.wall, 1e-9)
    scale = Scale(total, chart)

    lines: list[str] = [_header(run, lanes, color)]
    if not run.tests and not run.workers:
        lines.append("no tests recorded")
        return "\n".join(lines)

    axis, labels = _axis(scale)
    lines.append(f"{'worker':<{label_width}} {axis}  busy%")
    for lane in lanes:
        row, busy = _lane_row(lane, scale, glyphs)
        lines.append(f"{lane.id:<{label_width}} {_colorize(row, glyphs, color)}  {busy:5.1f}%")
    lines.append(f"{'':<{label_width}} {labels}")
    lines.append(
        f"{'':<{label_width}} legend: {glyphs.boot} boot  {glyphs.collect} collect  "
        f"{glyphs.test} tests  {glyphs.partial} <50% busy  {glyphs.bad} failure"
    )

    if top:
        candidates = [t for t in run.tests if t.duration >= min_duration]
        candidates.sort(key=lambda t: t.duration, reverse=True)
        slowest = candidates[:top]
        if slowest:
            lines.append("")
            lines.append(
                f"slowest {len(slowest)} tests (setup {glyphs.setup} / call {glyphs.call} / "
                f"teardown {glyphs.teardown}):"
            )
            for test in slowest:
                bar = _test_bar(test, scale, glyphs)
                if color and test.is_bad:
                    bar = f"{_RED}{bar}{_RESET}"
                duration = f"{format_seconds(test.duration):>7}"
                lines.append(f"{test.worker:<{label_width}} {bar} {duration}")
                held = _held(test)
                room = width - label_width - 1 - (len(held) + 2 if held else 0)
                nodeid = _fit_nodeid(test.nodeid, room, glyphs.ellipsis)
                lines.append(f"{'':<{label_width}} {nodeid}{'  ' + held if held else ''}")
    return "\n".join(lines)


class Scale:
    def __init__(self, total: float, columns: int) -> None:
        self.total = total
        self.columns = columns
        self.per_column = total / columns

    def col(self, seconds: float) -> int:
        value = int(seconds / self.total * self.columns)
        return min(self.columns - 1, max(0, value))

    def span(self, start: float, stop: float) -> tuple[int, int]:
        """Half-open column range that a [start, stop] interval covers, at least one wide."""
        first = self.col(start)
        last = max(first + 1, min(self.columns, int(round(stop / self.total * self.columns))))
        return first, last

    def overlaps(self, start: float, stop: float, *targets: list[float]) -> None:
        """Add the seconds that [start, stop] overlaps each column into every target."""
        if stop <= start:
            return
        first = self.col(start)
        last = min(self.columns, max(first + 1, math.ceil(stop / self.per_column)))
        for column in range(first, last):
            lo = column * self.per_column
            hi = lo + self.per_column
            overlap = max(0.0, min(stop, hi) - max(start, lo))
            for target in targets:
                target[column] += overlap


def _header(run: Run, lanes: list[Lane], color: bool) -> str:
    counts = run.outcome_counts()
    bad = {k: v for k, v in counts.items() if k in BAD_OUTCOMES}
    tests = f"{len(run.tests)} tests"
    if bad:
        detail = ", ".join(f"{n} {k}" for k, n in sorted(bad.items()))
        detail = f"{_RED}{detail}{_RESET}" if color else detail
        tests += f" ({detail})"
    from pytest_timing.model import _utilisation

    workers = len(lanes)
    lane = "worker" if workers == 1 else "workers"
    parts = [
        f"pytest-timing: {tests}",
        f"{workers} {lane}",
        f"wall {format_seconds(run.wall)}",
        f"busy {_utilisation(lanes) * 100:.1f}%",
    ]
    if not run.run.complete:
        ended = f"ended: {run.run.termination_label}"
        if run.run.reason:
            ended += f" ({run.run.reason})"
        parts.append(f"{_RED}{ended}{_RESET}" if color else ended)
    return ", ".join(parts)


def _axis(scale: Scale) -> tuple[str, str]:
    step = next((s for s in TICK_STEPS if s / scale.per_column >= 8), TICK_STEPS[-1])
    axis = ["-"] * scale.columns
    labels = [" "] * scale.columns
    tick = 0.0
    while tick < scale.total:
        column = int(tick / scale.per_column)
        if column >= scale.columns:
            break
        axis[column] = "|"
        text = format_seconds(tick) if tick else "0s"
        if column + len(text) <= scale.columns and all(
            labels[i] == " " for i in range(column, column + len(text))
        ):
            for i, ch in enumerate(text):
                labels[column + i] = ch
        tick += step
    end = format_seconds(scale.total)
    start = scale.columns - len(end)
    if start > 0 and all(labels[i] == " " for i in range(start - 1, scale.columns)):
        for i, ch in enumerate(end):
            labels[start + i] = ch
    return "".join(axis), "".join(labels).rstrip()


def _lane_row(lane: Lane, scale: Scale, glyphs: Glyphs) -> tuple[str, float]:
    columns = scale.columns
    test_time = [0.0] * columns
    bad_time = [0.0] * columns
    for span in lane.spans:
        if span.is_bad:
            scale.overlaps(span.start, span.stop, test_time, bad_time)
        else:
            scale.overlaps(span.start, span.stop, test_time)

    boot_time = [0.0] * columns
    collect_time = [0.0] * columns
    if lane.boot:
        scale.overlaps(*lane.boot, boot_time)
    if lane.collect:
        scale.overlaps(*lane.collect, collect_time)

    row: list[str] = []
    for i in range(columns):
        lo = i * scale.per_column
        hi = lo + scale.per_column
        in_window = max(0.0, min(lane.down, hi) - max(0.0, lo)) > 0
        if bad_time[i] > 0:
            row.append(glyphs.bad)
        elif test_time[i] >= scale.per_column * 0.5:
            row.append(glyphs.test)
        elif test_time[i] > 0:
            row.append(glyphs.partial)
        elif boot_time[i] > 0 and boot_time[i] >= collect_time[i]:
            row.append(glyphs.boot)
        elif collect_time[i] > 0:
            row.append(glyphs.collect)
        elif in_window:
            row.append(glyphs.idle)
        else:
            row.append(" ")
    window = max(lane.down - lane.ready, 1e-9)
    return "".join(row), min(100.0, lane.busy / window * 100)


def _test_bar(test: TestSpan, scale: Scale, glyphs: Glyphs) -> str:
    row = [" "] * scale.columns
    phase_glyphs = (("setup", glyphs.setup), ("call", glyphs.call), ("teardown", glyphs.teardown))
    for name, glyph in phase_glyphs:
        phase = test.phases.get(name)
        if phase is None:
            continue
        first, last = scale.span(phase.start, phase.stop)
        for i in range(first, last):
            row[i] = glyph
    if not test.phases:
        first, last = scale.span(test.start, test.stop)
        for i in range(first, last):
            row[i] = glyphs.bad if test.is_bad else glyphs.call
    return "".join(row)


def _held(test: TestSpan) -> str:
    """How long admission held the test back before it started, and at which gate."""
    parts = []
    if test.cpu is not None and test.cpu.wait:
        parts.append(f"waited {format_seconds(test.cpu.wait)} for cpu")
    if test.memory is not None and test.memory.wait:
        parts.append(f"waited {format_seconds(test.memory.wait)} for memory")
    return ", ".join(parts)


def _fit_nodeid(nodeid: str, room: int, ellipsis: str) -> str:
    if room >= len(nodeid) or room < 8 + len(ellipsis):
        return nodeid
    return ellipsis + nodeid[-(room - len(ellipsis)) :]


def _colorize(row: str, glyphs: Glyphs, color: bool) -> str:
    if not color:
        return row
    styles = {glyphs.bad: f"{_RED}{_BOLD}", glyphs.boot: _DIM, glyphs.collect: _DIM}
    out: list[str] = []
    for ch, group in groupby(row):
        run = "".join(group)
        style = styles.get(ch)
        out.append(f"{style}{run}{_RESET}" if style else run)
    return "".join(out)
