"""Platform telemetry: the host's CPU budget, CPU time of a process tree, pressure.

Everything that reads ``/proc``, ``/sys/fs/cgroup`` or the process table lives here,
behind small objects the rest of the plugin can replace in tests. Every reading
degrades explicitly: a value is ``None`` when the platform cannot provide it, and
a CPU measurement says how much of the process tree it covered.

Budget
    :func:`host_cpu` combines the CPUs the process may run on (its affinity mask,
    which also reflects a cpuset), and a cgroup CPU quota when one applies. The
    budget is the smaller of the two, rounded down, never below one.

CPU time of a test
    ``process_time()`` sees only the current process. :class:`ProcessTreeClock`
    adds the children the process has already reaped (``os.times``) and, where the
    platform lists them, the CPU time its live descendants have accumulated so far.
    A waited-for child moves from the live total to the reaped total. Discovery is
    a snapshot: exits during traversal and descendants that outlive or detach from
    their parents can leave gaps in the measurement.

Pressure
    Linux pressure stall information (``/proc/pressure/cpu`` or the cgroup's
    ``cpu.pressure``) says what share of the recent past some runnable task spent
    waiting for a CPU; the cgroup's ``cpu.stat`` counts quota throttling. Both are
    absent on other platforms. The load average is deliberately not used: it lags
    by a minute and counts what this run contributes.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC = Path("/proc")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


@dataclass(frozen=True, slots=True)
class HostCpu:
    cpus: int | None  # what the OS reports
    affinity: int | None  # CPUs this process may run on (cpuset included)
    quota: float | None  # cgroup CPU quota in CPUs, when one applies
    cgroup: str | None  # "v2", "v1" or None
    pressure: bool  # is pressure stall information readable
    platform: str

    @property
    def budget(self) -> int:
        """Slots this host can offer: affinity capped by the quota, at least one."""
        candidates = [c for c in (self.affinity, self.cpus) if c]
        slots = float(min(candidates)) if candidates else 1.0
        if self.quota is not None:
            slots = min(slots, self.quota)
        return max(1, int(slots))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpus": self.cpus,
            "affinity": self.affinity,
            "quota": self.quota,
            "cgroup": self.cgroup,
            "pressure": self.pressure,
            "platform": self.platform,
            "budget": self.budget,
        }


def _affinity() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return len(getter(0)) or None
    except OSError:
        return None


V1_CPU_MOUNTS = ("cpu", "cpu,cpuacct", "cpuacct,cpu")


@lru_cache(maxsize=1)
def cgroup_layout() -> tuple[str, Path, Path] | None:
    """(version, own group, mount) of the hierarchy that holds the ``cpu`` controller.
    Cached for the process: its membership does not change during a run.

    ``/proc/self/cgroup`` lists one line per hierarchy: ``0::<path>`` for the v2
    hierarchy and ``<id>:<controllers>:<path>`` for each v1 one. On a hybrid
    system both exist, and the CPU quota lives with whichever holds ``cpu``, so a v1
    line naming it wins over the v2 line. A path that does not exist under the
    mount (a cgroup namespace seen from outside) falls back to the mount itself.
    """
    text = _read(PROC / "self" / "cgroup") or ""
    v2_path: str | None = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if "cpu" in controllers.split(","):
            for name in V1_CPU_MOUNTS:
                mount = CGROUP_ROOT / name
                if mount.is_dir():
                    own = mount / path.lstrip("/")
                    return "v1", (own if own.is_dir() else mount), mount
        elif hierarchy == "0" and controllers == "" and v2_path is None:
            v2_path = path
    if v2_path is not None and CGROUP_ROOT.is_dir():
        own = CGROUP_ROOT / v2_path.lstrip("/")
        return "v2", (own if own.is_dir() else CGROUP_ROOT), CGROUP_ROOT
    for name in V1_CPU_MOUNTS:  # no readable membership, but a v1 mount
        mount = CGROUP_ROOT / name
        if mount.is_dir():
            return "v1", mount, mount
    return None


def _cpu_max(directory: Path) -> float | None:
    """A v2 group's own quota in CPUs (``cpu.max``), or ``None`` when unlimited."""
    parts = (_read(directory / "cpu.max") or "").split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    try:
        value = int(parts[0]) / int(parts[1])
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _cfs_quota(directory: Path) -> float | None:
    """A v1 group's own quota in CPUs (``cpu.cfs_quota_us``), or ``None`` when unlimited."""
    quota_text = _read(directory / "cpu.cfs_quota_us")
    period_text = _read(directory / "cpu.cfs_period_us")
    if quota_text is None or period_text is None:
        return None
    try:
        quota, period = int(quota_text), int(period_text)
    except ValueError:
        return None
    return quota / period if quota > 0 and period > 0 else None


def _tightest(own: Path, mount: Path, read: Callable[[Path], float | None]) -> float | None:
    """The smallest quota ``read`` reports from ``own`` up to ``mount``: a group is
    bound by its ancestors' limits as much as by its own."""
    quota: float | None = None
    current = own
    while True:
        value = read(current)
        if value is not None:
            quota = value if quota is None else min(quota, value)
        if current == mount or current.parent == current:
            break
        current = current.parent
        if not str(current).startswith(str(mount)):
            break
    return quota


def cgroup_quota() -> tuple[str | None, float | None]:
    """(cgroup version, CPU quota in CPUs) for this process, from its own group."""
    layout = cgroup_layout()
    if layout is None:
        return None, None
    version, own, mount = layout
    return version, _tightest(own, mount, _cpu_max if version == "v2" else _cfs_quota)


def host_cpu(pressure: Pressure | None = None) -> HostCpu:
    """Detect the CPU resources available to this process."""
    cpus = os.cpu_count()
    affinity = _affinity()
    quota: float | None = None
    cgroup: str | None = None
    pressure_readable = False
    if sys.platform.startswith("linux"):
        cgroup, quota = cgroup_quota()
        pressure_readable = (pressure or Pressure()).some() is not None
    return HostCpu(cpus, affinity, quota, cgroup, pressure_readable, sys.platform)


class ProcessTreeClock:
    """CPU seconds used by this process, its reaped children and its live descendants.

    ``coverage`` names what a reading includes: ``tree`` when live descendants are
    counted (Linux with ``/proc``, or anywhere with psutil installed), ``reaped``
    when only children that have already been waited for are (``os.times``), and
    ``self`` where the platform reports nothing about children at all.
    """

    def __init__(self) -> None:
        self._psutil: Any = None
        self._tick = 100.0
        self.coverage = "self" if sys.platform == "win32" else "reaped"
        if sys.platform.startswith("linux") and (PROC / "self" / "task").is_dir():
            try:
                self._tick = float(os.sysconf("SC_CLK_TCK"))
            except (ValueError, OSError, AttributeError):
                pass
            self._live = self._live_proc
            self.coverage = "tree"
        else:
            try:
                import psutil  # type: ignore[import-not-found,import-untyped,unused-ignore]

                self._psutil = psutil
                self._live = self._live_psutil
                self.coverage = "tree"
            except ImportError:
                self._live = self._live_none

    def seconds(self) -> float:
        times = os.times()
        return (
            times.user + times.system + times.children_user + times.children_system + self._live()
        )

    @staticmethod
    def _live_none() -> float:
        return 0.0

    def _live_psutil(self) -> float:
        psutil = self._psutil
        total = 0.0
        try:
            children = psutil.Process().children(recursive=True)
        except psutil.Error:
            return 0.0
        for child in children:
            try:
                times = child.cpu_times()
            except psutil.Error:
                continue
            total += times.user + times.system
            total += getattr(times, "children_user", 0.0) + getattr(times, "children_system", 0.0)
        return total

    def _live_proc(self) -> float:
        total = 0.0
        todo = [os.getpid()]
        seen: set[int] = set()
        while todo:
            pid = todo.pop()
            if pid in seen:
                continue
            seen.add(pid)
            task_dir = PROC / str(pid) / "task"
            try:
                tids = os.listdir(task_dir)
            except OSError:
                continue
            for tid in tids:
                text = _read(task_dir / tid / "children")
                if text:
                    todo.extend(int(c) for c in text.split() if c.isdigit())
            if pid != os.getpid():
                total += self._proc_cpu(pid)
        return total

    def _proc_cpu(self, pid: int) -> float:
        text = _read(PROC / str(pid) / "stat")
        if not text:
            return 0.0
        # The command name may contain spaces and parentheses; fields follow the last ')'.
        rest = text.rpartition(")")[2].split()
        try:
            utime, stime, cutime, cstime = (int(v) for v in rest[11:15])
        except (ValueError, IndexError):
            return 0.0
        return (utime + stime + cutime + cstime) / self._tick


PSI_TTL = 1.0
"""Seconds a PSI reading is reused: the kernel refreshes ``avg10`` every two."""


class Pressure:
    """CPU pressure and quota throttling on this host, when it reports them."""

    def __init__(self) -> None:
        self._psi: Path | None = None
        self._stat: Path | None = None
        self._some: tuple[float, float | None] = (-1.0, None)  # (read at, value)
        if sys.platform.startswith("linux"):
            layout = cgroup_layout()
            own = layout[1] if layout is not None else None
            v2 = own if layout is not None and layout[0] == "v2" else None
            if v2 is not None and _read(v2 / "cpu.pressure"):
                self._psi = v2 / "cpu.pressure"
            elif _read(PROC / "pressure" / "cpu"):
                self._psi = PROC / "pressure" / "cpu"
            if own is not None and _read(own / "cpu.stat"):
                self._stat = own / "cpu.stat"  # v2: throttled_usec; v1: throttled_time (ns)

    def some(self) -> float | None:
        """Share (0..1) of the last ten seconds some task spent waiting for a CPU."""
        if self._psi is None:
            return None
        now = time.monotonic()
        if now - self._some[0] < PSI_TTL:
            return self._some[1]
        value = self._read_some()
        self._some = (now, value)
        return value

    def _read_some(self) -> float | None:
        text = _read(self._psi) if self._psi is not None else None
        if not text:
            return None
        for line in text.splitlines():
            if line.startswith("some"):
                for field_ in line.split()[1:]:
                    name, _, value = field_.partition("=")
                    if name == "avg10":
                        try:
                            return float(value) / 100.0
                        except ValueError:
                            return None
        return None

    def throttled(self) -> int | None:
        """Cumulative microseconds the cgroup was throttled by its quota."""
        text = _read(self._stat) if self._stat is not None else None
        if not text:
            return None
        for line in text.splitlines():
            name, _, value = line.partition(" ")
            if name in ("throttled_usec", "throttled_time"):
                try:
                    number = int(value)
                except ValueError:
                    return None
                return number if name == "throttled_usec" else number // 1000
        return None
