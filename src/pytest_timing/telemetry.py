"""Platform telemetry: the host's CPU budget, CPU time and resident memory of a
process tree, pressure.

Everything that reads ``/proc``, ``/sys/fs/cgroup``, the process table or the
platform's process-information API lives here, behind small objects the rest of the
plugin can replace in tests. Every reading degrades explicitly: a value is ``None``
when the platform cannot provide it, and a CPU or memory measurement says how much
of the process tree it covered.

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

Resident memory of a test
    :class:`ResidentMemory` reads the resident set size of the current process
    (``/proc/self/statm`` on Linux, ``libproc`` on macOS, ``psapi`` on Windows, or
    psutil anywhere) and, where descendants can be listed, adds theirs.
    :class:`MemorySampler` polls it from a thread while a test runs and keeps the
    peak, since a high-water mark like ``ru_maxrss`` never comes back down and would
    attribute a whole worker's history to whichever test happened to raise it.

Pressure
    Linux pressure stall information (``/proc/pressure/cpu`` or the cgroup's
    ``cpu.pressure``) says what share of the recent past some runnable task spent
    waiting for a CPU; the cgroup's ``cpu.stat`` counts quota throttling. Both are
    absent on other platforms. The load average is deliberately not used: it lags
    by a minute and counts what this run contributes.
"""

from __future__ import annotations

import math
import os
import sys
import threading
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


def proc_descendants() -> list[int]:
    """Live descendants of this process, through ``/proc/<pid>/task/*/children``.

    A snapshot: a process that exits during the walk is skipped, and one that
    detached from its parent is not found.
    """
    found: list[int] = []
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
            found.append(pid)
    return found


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
        return sum(self._proc_cpu(pid) for pid in proc_descendants())

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


SAMPLE_INTERVAL = 0.02
"""Seconds between resident-memory readings of the worker while a test runs."""
TREE_INTERVAL = 0.1
"""Seconds between readings of the worker's descendants: listing them costs more."""


def _linux_own_rss() -> int | None:
    text = _read(PROC / "self" / "statm")
    if not text:
        return None
    try:
        return int(text.split()[1]) * _PAGE_SIZE
    except (IndexError, ValueError):
        return None


def _linux_tree_rss() -> int:
    total = 0
    for pid in proc_descendants():
        text = _read(PROC / str(pid) / "statm")
        if text:
            try:
                total += int(text.split()[1]) * _PAGE_SIZE
            except (IndexError, ValueError):
                pass
    return total


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 4096


_PAGE_SIZE = _page_size()


def _darwin_own_rss_reader() -> Callable[[], int | None] | None:
    """``proc_pidinfo(PROC_PIDTASKINFO)``: ``pti_resident_size`` is the second field."""
    import ctypes
    import ctypes.util

    try:
        name = ctypes.util.find_library("proc")
        lib = ctypes.CDLL(name or "libproc.dylib")
        pidinfo = lib.proc_pidinfo
    except (OSError, AttributeError):
        return None
    pidinfo.restype = ctypes.c_int
    pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    buffer = ctypes.create_string_buffer(512)
    pid = os.getpid()

    def own() -> int | None:
        size = pidinfo(pid, 4, 0, buffer, len(buffer))  # PROC_PIDTASKINFO
        if size < 16:
            return None
        return int.from_bytes(buffer.raw[8:16], sys.byteorder)

    return own if own() else None


def _windows_own_rss_reader() -> Callable[[], int | None] | None:
    """``GetProcessMemoryInfo``: the working set, the process's resident pages."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        windll = ctypes.windll  # type: ignore[attr-defined,unused-ignore]
        psapi, kernel32 = windll.psapi, windll.kernel32
        info = psapi.GetProcessMemoryInfo
        handle = kernel32.GetCurrentProcess()
    except (OSError, AttributeError):
        return None
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)

    def own() -> int | None:
        if not info(handle, ctypes.byref(counters), ctypes.sizeof(Counters)):
            return None
        return int(counters.WorkingSetSize)

    return own if own() else None


class ResidentMemory:
    """Resident set size of this process and, where they can be listed, its descendants.

    ``coverage`` names what a reading includes: ``tree`` when live descendants are
    counted, ``self`` when only this process is, ``none`` when the platform gives no
    reading at all (then :meth:`own` is ``None``). Descendants are summed, so pages
    they share with each other count more than once: a conservative total.
    """

    def __init__(self) -> None:
        self.coverage = "none"
        self._own: Callable[[], int | None] = lambda: None
        self._tree: Callable[[], int] | None = None
        if sys.platform.startswith("linux") and _linux_own_rss():
            self._own, self._tree = _linux_own_rss, _linux_tree_rss
            self.coverage = "tree"
            return
        try:
            import psutil  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:
            psutil = None
        if psutil is not None:
            process = psutil.Process()

            def own() -> int | None:
                try:
                    return int(process.memory_info().rss)
                except psutil.Error:
                    return None

            def tree() -> int:
                total = 0
                try:
                    children = process.children(recursive=True)
                except psutil.Error:
                    return 0
                for child in children:
                    try:
                        total += int(child.memory_info().rss)
                    except psutil.Error:
                        continue
                return total

            if own():
                self._own, self._tree = own, tree
                self.coverage = "tree"
                return
        reader = None
        try:
            if sys.platform == "darwin":
                reader = _darwin_own_rss_reader()
            elif sys.platform == "win32":
                reader = _windows_own_rss_reader()
        except Exception:  # ctypes lookups fail in platform-specific ways
            reader = None
        if reader is not None:
            self._own = reader
            self.coverage = "self"

    def own(self) -> int | None:
        """Resident bytes of this process, or ``None`` when unreadable."""
        return self._own()

    def descendants(self) -> int:
        """Resident bytes of live descendants; zero when they cannot be listed."""
        return self._tree() if self._tree is not None else 0


@dataclass(frozen=True, slots=True)
class MemoryWindow:
    """What one sampling window saw, in bytes."""

    base: int  # resident when the window opened
    peak: int  # the highest reading inside it
    after: int  # resident when it closed
    coverage: str


class MemorySampler:
    """Polls resident memory from a thread while a window is open and keeps the peak.

    The thread reads the process every ``interval`` seconds and its descendants
    every ``tree_interval``, adding the latest descendant total to each reading.
    Between windows it sleeps. ``begin`` and ``end`` are called from the thread
    running the tests; ``close`` stops the sampler for good.
    """

    def __init__(
        self,
        memory: ResidentMemory | None = None,
        interval: float = SAMPLE_INTERVAL,
        tree_interval: float = TREE_INTERVAL,
    ) -> None:
        self.memory = memory or ResidentMemory()
        self.interval = interval
        self.tree_interval = tree_interval
        self._lock = threading.Lock()
        self._open = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._base = self._peak = 0
        self._children = 0
        self._children_at = -math.inf

    @property
    def coverage(self) -> str:
        return self.memory.coverage

    def _reading(self, fresh: bool) -> int | None:
        own = self.memory.own()
        if own is None:
            return None
        now = time.monotonic()
        if fresh or now - self._children_at >= self.tree_interval:
            self._children = self.memory.descendants()
            self._children_at = now
        return own + self._children

    def begin(self) -> bool:
        """Open a window; ``False`` when the platform gives no reading."""
        with self._lock:
            reading = self._reading(fresh=True)
            if reading is None:
                return False
            self._base = self._peak = reading
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(
                target=self._run, name="pytest-timing-memory", daemon=True
            )
            self._thread.start()
        self._open.set()
        return True

    def end(self) -> MemoryWindow | None:
        """Close the window and report it; ``None`` when none was open."""
        if not self._open.is_set():
            return None
        self._open.clear()
        with self._lock:
            after = self._reading(fresh=True)
            if after is None:
                after = self._peak
            peak = max(self._peak, after)
            return MemoryWindow(self._base, peak, after, self.memory.coverage)

    def _sample(self) -> None:
        with self._lock:
            reading = self._reading(fresh=False)
            if reading is not None and reading > self._peak:
                self._peak = reading

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._open.wait(0.5):
                continue
            self._sample()
            self._stop.wait(self.interval)

    def close(self) -> None:
        self._stop.set()
        self._open.clear()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None
