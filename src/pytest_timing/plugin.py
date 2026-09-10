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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pytest_timing import xdist_compat
from pytest_timing.collector import Collector, PhaseReport
from pytest_timing.model import Run, RunInfo
from pytest_timing.outputs import OUTPUTS, write_output
from pytest_timing.render.ascii import render_ascii

MAIN_LANE = "main"
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
    parser.addini("timing_top", "Rows in the slowest-tests section.", default="")
    parser.addini("timing_min", "Minimum duration for the slowest-tests section.", default="")
    parser.addini("timing_ascii_style", "unicode or ascii.", default="")


_TRUE = ("1", "true", "yes", "on")


class Settings:
    """Resolved configuration: command line, then environment, then ini."""

    def __init__(self, config: pytest.Config) -> None:
        self.outputs: dict[str, Path] = {}
        for kind, output in OUTPUTS.items():
            path = self._output_path(config, kind, output.default)
            if path is not None:
                self.outputs[kind] = path
        top = self._value(config, "timing_top", "PYTEST_TIMING_TOP")
        self.top = int(top) if top is not None else DEFAULT_TOP
        minimum = self._value(config, "timing_min", "PYTEST_TIMING_MIN")
        self.min_duration = float(minimum) if minimum is not None else DEFAULT_MIN
        style = self._value(config, "timing_ascii_style", "PYTEST_TIMING_ASCII_STYLE")
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
        )

    @staticmethod
    def _value(config: pytest.Config, name: str, env: str, ini: str | None = None) -> Any:
        """One precedence rule for every setting: option, then environment, then ini."""
        value = config.getoption(name, None)
        if value is not None:
            return value
        if os.environ.get(env):
            return os.environ[env]
        value = config.getini(ini or name)
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
        path = Path(value)
        return path if path.is_absolute() else Path(config.rootpath, path)


def pytest_configure(config: pytest.Config) -> None:
    if xdist_compat.is_worker(config):
        return  # xdist worker: everything happens on the controller
    settings = Settings(config)
    if settings.enabled:
        config.pluginmanager.register(TimingPlugin(config, settings), "pytest_timing")


class TimingPlugin:
    def __init__(self, config: pytest.Config, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        self.collector = Collector(self._run_info())
        self.result: Run | None = None
        self.written: list[str] = []
        self.errors: list[str] = []
        # Lifecycle evidence, recorded as it happens.
        self.termination: str | None = None
        self.reason: str | None = None
        self._collections = xdist_compat.CollectionWatch()
        self._stop_observing: Callable[[], None] | None = None

    @property
    def distributed(self) -> bool:
        return xdist_compat.is_distributed(self.config)

    # ---- session -------------------------------------------------------------------

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

    # ---- termination evidence ------------------------------------------------------

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

    # ---- xdist worker lifecycle (optionalhook: xdist may not be installed) -----------

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

    # ---- reports -------------------------------------------------------------------

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # Called three times per test: keep it to attribute reads and one object.
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
            )
        )

    # ---- finish and output ---------------------------------------------------------

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
        self.result = self.collector.finish(now, termination=termination, reason=reason)
        self._write_outputs(self.result)

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
        for line in self.written:
            tr.write_line(line.replace(str(self.config.rootpath) + os.sep, "", 1))
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
