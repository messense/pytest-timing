"""pytest hooks: capture timings on the controller process and emit reports.

Two rules keep this adapter honest:

* Options are unambiguous. Boolean flags turn an output on; ``*-file`` options take a
  required path. Nothing has an optional value, so argparse can never mistake a test
  path for a report path however the plugin is loaded.
* Termination is recorded as it happens (``pytest_keyboard_interrupt``,
  ``pytest_internalerror``, xdist's stop and abort reasons) and ``complete`` is derived
  from that record, never from counting tests.
"""

from __future__ import annotations

import os
import platform
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest

from pytest_timing import xdist_compat
from pytest_timing.collector import Collector, PhaseReport
from pytest_timing.demand import EVENT, EXECUTION_ATTR, MARKER, CpuMeter, Declarations
from pytest_timing.demand import REPORT_ATTR as CPU_ATTR
from pytest_timing.fixtures import REPORT_ATTR, FixtureTimer
from pytest_timing.model import Run, RunInfo
from pytest_timing.outputs import OUTPUTS, write_output
from pytest_timing.render.ascii import render_ascii
from pytest_timing.schedule import Estimates
from pytest_timing.telemetry import Pressure, ProcessTreeClock, host_cpu

MAIN_LANE = "main"
EQUAL_ESTIMATE = 0.001  # seconds per test when a run has no recorded durations
DEFAULT_TOP = 10
DEFAULT_MIN = 0.0
ASCII_STYLES = ("unicode", "ascii")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("timing", "test timing reports (pytest-timing)")
    group.addoption(
        "--timing",
        action="store_true",
        default=False,
        help="Record test timings and print an ASCII Gantt chart in the terminal summary.",
    )
    for kind, output in OUTPUTS.items():
        group.addoption(
            f"--timing-{kind}",
            action="store_true",
            default=False,
            help=f"Write the {output.label} to {output.default}. Implies --timing.",
        )
        group.addoption(
            f"--timing-{kind}-file",
            metavar="PATH",
            default=None,
            help=f"Write the {output.label} to PATH. Implies --timing-{kind}.",
        )
    group.addoption(
        "--timing-schedule",
        metavar="PATH",
        default=None,
        help="In xdist load/worksteal mode, balance tests using durations and fixture costs "
        "from the JSON run at PATH (written by --timing-json in an earlier session), "
        "plus declared CPU demand. Implies --timing.",
    )
    group.addoption(
        "--timing-cpus",
        metavar="N|auto",
        default=None,
        help="In xdist load/worksteal mode, admit tests against a budget of N CPU slots per "
        "host ('auto' detects CPUs, capped by affinity and cgroup quota). Tests declare their "
        "demand with @pytest.mark.timing_cpu(N), fixtures with @pytest_timing.cpu(N). "
        "Implies --timing.",
    )
    group.addoption(
        "--timing-top",
        type=int,
        default=None,
        metavar="N",
        help=f"Rows in the slowest-tests section of the ASCII chart (default: {DEFAULT_TOP}, "
        "0 hides the section).",
    )
    group.addoption(
        "--timing-min",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Hide tests shorter than this in the slowest-tests section (default: 0).",
    )
    group.addoption(
        "--timing-ascii-style",
        choices=ASCII_STYLES,
        default=None,
        help="Characters used for the ASCII chart: unicode (default) or ascii.",
    )
    group.addoption(
        "--timing-width",
        type=int,
        default=None,
        metavar="COLUMNS",
        help="Override the terminal width used for the ASCII chart.",
    )
    parser.addini("timing", "Enable pytest-timing (same as --timing).", type="bool", default=False)
    for kind, output in OUTPUTS.items():
        parser.addini(
            f"timing_{kind}",
            f"Path for the {output.label} (implies timing); 'true' uses {output.default}.",
            default="",
        )
    parser.addini(
        "timing_schedule",
        "Path of a JSON run to schedule xdist workers from (implies timing).",
        default="",
    )
    parser.addini(
        "timing_cpus",
        "CPU slots per host for xdist admission, or 'auto' (implies timing).",
        default="",
    )
    parser.addini("timing_top", "Rows in the slowest-tests section.", default="")
    parser.addini("timing_min", "Minimum duration for the slowest-tests section.", default="")
    parser.addini("timing_ascii_style", "unicode or ascii.", default="")


_TRUE = ("1", "true", "yes", "on")


class Settings:
    """Resolve values by CLI, environment, then ini; enable timing if any source asks."""

    def __init__(self, config: pytest.Config) -> None:
        self.outputs: dict[str, Path] = {}
        for kind, output in OUTPUTS.items():
            path = self._output_path(config, kind, output.default)
            if path is not None:
                self.outputs[kind] = path
        schedule = self._value(config, "timing_schedule")
        self.schedule = self._path(config, schedule) if schedule else None
        cpus = self._value(config, "timing_cpus")
        self.cpus: int | Literal["auto"] | None = None  # the CPU budget asked for
        if cpus is not None:
            text = str(cpus).strip().lower()
            if text == "auto":
                self.cpus = "auto"
            elif text.isdigit() and int(text) > 0:
                self.cpus = int(text)
            else:
                raise pytest.UsageError(
                    f"timing_cpus must be a positive integer or 'auto', not {cpus!r}"
                )
        top = self._value(config, "timing_top")
        self.top = int(top) if top is not None else DEFAULT_TOP
        minimum = self._value(config, "timing_min")
        self.min_duration = float(minimum) if minimum is not None else DEFAULT_MIN
        style = self._value(config, "timing_ascii_style")
        if style is not None and style not in ASCII_STYLES:
            raise pytest.UsageError(
                f"timing_ascii_style must be one of {', '.join(ASCII_STYLES)}, not {style!r}"
            )
        self.ascii_style = str(style) if style is not None else "unicode"
        width = config.getoption("timing_width", None)
        self.width = int(width) if width else None
        env = os.environ.get("PYTEST_TIMING", "").lower()
        self.enabled = bool(
            config.getoption("timing", False)
            or env in _TRUE
            or config.getini("timing")
            or self.outputs
            or self.schedule is not None
            or self.cpus is not None
        )

    @staticmethod
    def _value(config: pytest.Config, name: str) -> Any:
        """Resolve a value: option, then environment, then ini."""
        value = config.getoption(name, None)
        if value is not None:
            return value
        env = os.environ.get("PYTEST_" + name.upper())
        if env:
            return env
        value = config.getini(name)
        return value if value not in ("", None) else None

    @classmethod
    def _output_path(cls, config: pytest.Config, kind: str, default: str) -> Path | None:
        """``--timing-<kind>-file`` wins, then the flag, then env / ini values.

        Env and ini values are paths, or ``true`` for the default file name.
        """
        explicit = config.getoption(f"timing_{kind}_file", None)
        if explicit:
            value = str(explicit)  # a path, verbatim: never a boolean alias
        elif config.getoption(f"timing_{kind}", False):
            value = default
        else:
            env = os.environ.get(f"PYTEST_TIMING_{kind.upper()}") or config.getini(f"timing_{kind}")
            if not env:
                return None
            value = default if str(env).lower() in _TRUE else str(env)
        return cls._path(config, value)

    @staticmethod
    def _path(config: pytest.Config, value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else Path(config.rootpath, path)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}(slots): CPU slots this test's workload needs, subprocesses included "
        "(pytest-timing schedules it against the host's budget; default 1).",
    )
    settings = Settings(config)
    if not settings.enabled:
        return
    # Shared fixture set-up is timed, and CPU work measured, wherever tests run: in
    # every xdist worker, or here.
    clock = ProcessTreeClock()
    timer = FixtureTimer(work=clock.seconds)
    config.pluginmanager.register(timer, "pytest_timing_fixtures")
    worker = xdist_compat.is_worker(config)
    send: Callable[[str, dict[str, Any]], None] | None = None
    if worker:

        def send(name: str, payload: dict[str, Any]) -> None:
            xdist_compat.send_event(config, name, payload)

    config.pluginmanager.register(CpuMeter(timer, send, clock, Pressure()), "pytest_timing_cpu")
    if worker:
        return  # everything else happens on the controller
    config.pluginmanager.register(TimingPlugin(config, settings), "pytest_timing")


class TimingPlugin:
    def __init__(self, config: pytest.Config, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        self.collector = Collector(self._run_info())
        self.result: Run | None = None
        self.written: list[str] = []
        self.errors: list[str] = []
        self.scheduler: Any = None  # the DurationScheduling in use, if any
        self.schedule_note: str | None = None  # why scheduling was not used
        self.declarations: dict[str, Declarations] = {}  # per worker id, from the workers
        self._declarations_lock = threading.Lock()
        self.termination: str | None = None
        self.reason: str | None = None
        self._collections = xdist_compat.CollectionWatch()
        self._stop_observing: Callable[[], None] | None = None

    @property
    def distributed(self) -> bool:
        return xdist_compat.is_distributed(self.config)

    def _run_info(self) -> RunInfo:
        config = self.config
        return RunInfo(
            start=time.time(),
            stop=0.0,
            argv=list(config.invocation_params.args),
            rootdir=str(config.rootpath),
            python=platform.python_version(),
            pytest=pytest.__version__,
            xdist=xdist_compat.version() if config.pluginmanager.hasplugin("xdist") else None,
        )

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self) -> None:
        """Runs before xdist launches its workers, so their start events have a home."""
        config = self.config
        if self.distributed:
            dist = config.getoption("dist", None)
            numprocesses = config.getoption("numprocesses", None)
            self.collector.run.dist = str(dist) if dist else None
            self.collector.run.numprocesses = (
                int(numprocesses) if isinstance(numprocesses, int) else None
            )
        else:
            if self.settings.schedule is not None:
                self.schedule_note = "schedule: needs pytest-xdist workers (-n); not applied"
            elif self.settings.cpus is not None:
                self.schedule_note = "cpu: a budget needs pytest-xdist workers (-n); not applied"
            now = time.time()
            self.collector.worker_started(MAIN_LANE, self.collector.run.start)
            self.collector.worker_ready(MAIN_LANE, now)

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        if not self.distributed:
            self.collector.worker_collected(MAIN_LANE, time.time(), len(session.items))

    def pytest_collectreport(self, report: Any) -> None:
        if self.distributed:
            reason = self._collections.mismatch_reason(report)
            if reason:
                self._record("aborted", reason)

    def pytest_keyboard_interrupt(self, excinfo: Any) -> None:
        """Fired for KeyboardInterrupt, ``pytest.exit()`` (any return code) and xdist's
        ``Interrupted``; the run loop did not finish on its own."""
        self._record("interrupted", _describe(excinfo))

    def pytest_internalerror(self, excrepr: Any) -> None:
        lines = str(excrepr).strip().splitlines()
        self._record("internal_error", lines[-1] if lines else None)

    def _record(self, termination: str, reason: Any) -> None:
        if self.termination is None:
            self.termination = termination
            self.reason = reason if isinstance(reason, str) or reason is None else str(reason)

    def _termination(self, session: pytest.Session) -> tuple[str, str | None]:
        if self.termination is not None:
            return self.termination, self.reason
        if self.config.getoption("collectonly", False):
            return "collect_only", None
        for attr in ("shouldfail", "shouldstop"):
            value = getattr(session, attr, False)
            if value:
                return "interrupted", str(value)
        if self.distributed:
            reason = xdist_compat.stop_reason(self.config)
            if reason:
                return "interrupted", reason
            reason = xdist_compat.abort_reason(self.config)
            if reason:
                return "aborted", reason
        return "finished", None

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_setupnodes(self) -> None:
        """Before any gateway exists: time each creation as it happens."""
        self._stop_observing = xdist_compat.observe_gateway_creation(
            self.config, self.collector.worker_started
        )
        if self._stop_observing is not None:
            # Also at unconfigure: a failed worker start-up never reaches sessionfinish.
            self.config.add_cleanup(self._stop_observing_now)

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        """Before the worker starts: take its CPU declarations off its channel."""
        worker = xdist_compat.node_id(node)
        # Register before forwarding anything. Another plugin can select the
        # scheduler without ever calling our first-result make_scheduler hook.
        xdist_compat.take_inproc_events(self.config, xdist_compat.REQUEST, self._on_request)
        xdist_compat.take_inproc_events(self.config, xdist_compat.HOLDS, self._on_holds)

        def receive(payload: dict[str, Any]) -> None:
            declarations = Declarations.from_dict(payload)
            with self._declarations_lock:
                self.declarations[worker] = declarations

        xdist_compat.intercept_events(node, EVENT, receive)
        xdist_compat.forward_events(node, xdist_compat.REQUEST)
        xdist_compat.forward_events(node, xdist_compat.HOLDS)

    def _on_request(
        self,
        node: Any,
        index: int,
        key: str,
        setup: int,
        hold: int,
        holds: int = 0,
        attempt: int = 0,
        request_id: int = 0,
        cancelled: bool = False,
    ) -> None:
        """A worker asks for slots for a fixture its running test reaches at run time."""
        if self.scheduler is None:
            xdist_compat.grant_slots(node, key, request_id)
            return
        self.scheduler.request(
            node,
            int(index),
            str(key),
            int(setup),
            int(hold),
            int(holds),
            int(attempt),
            int(request_id),
            bool(cancelled),
        )

    def _on_holds(self, node: Any, holds: int) -> None:
        if self.scheduler is not None:
            self.scheduler.update_holds(node, int(holds))

    def _declarations_for(self, node: Any) -> Declarations | None:
        with self._declarations_lock:
            return self.declarations.get(xdist_compat.node_id(node))

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_newgateway(self, gateway: Any) -> None:
        """Fallback when the creation call could not be observed: the gateway exists
        now, and nothing earlier is known."""
        if self._stop_observing is None:
            self.collector.worker_started(xdist_compat.gateway_id(gateway), time.time())

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodeready(self, node: Any) -> None:
        self.collector.worker_ready(xdist_compat.node_id(node), time.time())

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: Any) -> None:
        worker = xdist_compat.node_id(node)
        self._collections.collected(worker, ids)
        self.collector.worker_collected(worker, time.time(), len(ids))

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        message = None if error is None else str(error)
        self.collector.worker_down(xdist_compat.node_id(node), time.time(), message)
        if error is None and self.scheduler is not None:
            self.scheduler.worker_finished(node)

    @pytest.hookimpl(optionalhook=True, tryfirst=True)
    def pytest_xdist_make_scheduler(self, config: pytest.Config, log: Any) -> Any:
        """Replace xdist's ``load`` scheduler with the duration-aware one when asked.

        Runs on the controller right before the run loop, so the file from the last
        session is read before this session overwrites it.
        """
        path = self.settings.schedule
        admission = self.settings.cpus is not None
        if path is None and not admission:
            return None
        dist = config.getoption("dist", None)
        if dist not in ("load", "worksteal"):
            self.schedule_note = f"schedule: --dist={dist} is not supported; not applied"
            return None
        estimates: Estimates | None = None
        if path is not None:
            shown = self._relative(path)
            try:
                estimates = Estimates.load(path)
            except FileNotFoundError:
                self.schedule_note = f"schedule: no run recorded at {shown} yet; not applied"
            except (OSError, ValueError) as exc:
                self.schedule_note = f"schedule: could not read {shown} ({exc}); not applied"
            if estimates is None and not admission:
                return None
        if estimates is None:
            # No durations, but a CPU budget to enforce: every test counts as equal,
            # and small, so batches of them are still sent in one command.
            estimates = Estimates(default=EQUAL_ESTIMATE)
        from pytest_timing.xdist_scheduler import CpuSetup, DurationScheduling

        cpu = CpuSetup(
            cpus=self.settings.cpus,
            declarations=self._declarations_for,
            domain_of=xdist_compat.domain_of,
            pressure=Pressure(),
        )
        self.scheduler = DurationScheduling(
            config, log, estimates, stealing=dist == "worksteal", cpu=cpu
        )
        return self.scheduler

    def _relative(self, path: Path | str) -> str:
        return str(path).replace(str(self.config.rootpath) + os.sep, "", 1)

    def _schedule_summary(self) -> list[str]:
        lines: list[str] = []
        if self.schedule_note is not None:
            lines.append(self.schedule_note)
        scheduler = self.scheduler
        if scheduler is None or scheduler.collection is None:
            return lines
        total = len(scheduler.collection)
        shared = len(scheduler.costs.shared) if scheduler.costs is not None else 0
        fixtures = f", {shared} shared fixture{'s' if shared != 1 else ''}" if shared else ""
        mode = " (worksteal)" if scheduler.stealing else ""
        if scheduler.estimates.source:
            lines.append(
                f"schedule{mode}: {scheduler.known} of {total} tests had recorded durations"
                f"{fixtures} in {self._relative(Path(scheduler.estimates.source))}"
            )
        else:
            lines.append(f"schedule{mode}: no recorded durations, {total} tests taken as equal")
        cpu = self._cpu_summary()
        if cpu is not None:
            lines.append(cpu)
        return lines

    def _cpu_summary(self) -> str | None:
        scheduler = self.scheduler
        summary = scheduler.cpu_summary() if scheduler is not None else None
        if summary is None:
            return None
        heavy = summary["heavy_tests"]
        if not summary["gated"]:
            return None
        parts: list[str] = []
        for name, domain in summary["domains"].items():
            budget = domain.get("budget")
            if budget is None:
                continue
            host = domain.get("host") or {}
            detail = []
            if host.get("cpus"):
                detail.append(f"{host['cpus']} cpus")
            if host.get("quota"):
                detail.append(f"quota {host['quota']:g}")
            where = f" ({', '.join(detail)})" if detail else ""
            text = f"{name}: budget {budget}{where}"
            if domain.get("lowest", budget) < budget:
                text += f", lowered to {domain['lowest']} under pressure"
            clamped = domain.get("clamped")
            if clamped:
                text += f", {clamped} request{'s' if clamped != 1 else ''} over budget run alone"
            parts.append(text)
        line = "cpu: " + "; ".join(parts)
        line += f"; {heavy} test{'s' if heavy != 1 else ''} over one slot"
        if summary["waited_tests"]:
            n = summary["waited_tests"]
            line += f"; {n} test{'s' if n != 1 else ''} waited {summary['waited']:.2f}s for slots"
        if summary.get("cancelled"):
            n = summary["cancelled"]
            line += f"; {n} unadmitted test{'s' if n != 1 else ''} withdrawn at shutdown"
        return line

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # Called three times per test: keep it to attribute reads and one object.
        cpu = getattr(report, CPU_ATTR, None)
        start = getattr(report, "start", 0.0)
        stop = getattr(report, "stop", 0.0)
        self.collector.add_report(
            PhaseReport(
                nodeid=report.nodeid,
                when=report.when,
                outcome=report.outcome,
                start=start,
                stop=stop,
                duration=report.duration,
                worker=xdist_compat.worker_id(report) or MAIN_LANE,
                wasxfail=hasattr(report, "wasxfail"),
                # Receipt time is only needed when the report carries no clock of its own.
                received=time.time() if not (start and stop) else 0.0,
                fixtures=getattr(report, REPORT_ATTR, None),
                cpu=cpu,
                execution=getattr(report, EXECUTION_ATTR, None),
            )
        )
        if cpu and self.scheduler is not None and cpu.get("elapsed"):
            worker = xdist_compat.worker_id(report) or MAIN_LANE
            self.scheduler.observe_rate(worker, float(cpu["work"]) / float(cpu["elapsed"]))

    def _stop_observing_now(self) -> None:
        if self._stop_observing is not None:
            self._stop_observing()
            self._stop_observing = None

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._stop_observing_now()
        now = time.time()
        if not self.distributed:
            self.collector.worker_down(MAIN_LANE, now, None)
        termination, reason = self._termination(session)
        self.collector.run.exit_status = int(exitstatus)
        self._record_cpu()
        self.result = self.collector.finish(now, termination=termination, reason=reason)
        self._write_outputs(self.result)

    def _record_cpu(self) -> None:
        """Admission waits onto their spans, and the CPU environment onto the run."""
        scheduler = self.scheduler
        if scheduler is not None and scheduler.collection is not None:
            for wait in scheduler.waits:
                self.collector.add_wait(
                    wait.worker,
                    scheduler.collection[wait.index],
                    wait.index,
                    wait.attempt,
                    wait.seconds,
                )
            self.collector.run.cpu = scheduler.cpu_summary()
        elif not self.distributed:
            self.collector.run.cpu = {
                "gated": False,
                "domains": {"local": {"host": host_cpu().to_dict()}},
            }

    def _write_outputs(self, run: Run) -> None:
        if not self.settings.outputs:
            return
        doc = run.to_dict()  # serialised once, shared by every output
        for kind, path in self.settings.outputs.items():
            output = OUTPUTS[kind]
            try:
                self.written.append(write_output(output, path, run, doc))
            except Exception as exc:  # pragma: no cover - reported, never fatal
                self.errors.append(f"could not write {output.label} to {path}: {exc}")

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        run = self.result
        if run is None:
            return
        tr = terminalreporter
        width = self.settings.width or _terminal_width(tr)
        tr.write_sep("=", "timing report")
        text = render_ascii(
            run,
            width=width,
            top=self.settings.top,
            min_duration=self.settings.min_duration,
            style=self.settings.ascii_style,
            color=bool(getattr(tr, "hasmarkup", False)),
        )
        for line in text.splitlines():
            tr.write_line(line)
        for line in self._schedule_summary():
            tr.write_line(line)
        for line in self.written:
            tr.write_line(self._relative(line))
        for error in self.errors:
            tr.write_line(error, red=True)


def _describe(excinfo: Any) -> str:
    value = getattr(excinfo, "value", excinfo)
    text = str(getattr(value, "msg", "") or value or "").strip()
    name = type(value).__name__
    return f"{name}: {text}" if text else name


def _terminal_width(terminalreporter: Any) -> int:
    writer = getattr(terminalreporter, "_tw", None)
    width = getattr(writer, "fullwidth", None)
    if isinstance(width, int) and width > 0:
        return width
    return shutil.get_terminal_size().columns
