"""Benchmarks for CPU-aware scheduling. Timing-sensitive by nature: nothing here is a
test, and every number depends on the machine, its load and its CPU quota.

    uv run python benchmarks/cpu_bench.py --workers 8 --cpus 8
    docker run --rm -v "$PWD":/src -w /src --cpus=4 python:3.12-slim sh -c \\
        "pip install -q -e . pytest-xdist && python benchmarks/cpu_bench.py --workers 4"

Each scenario writes a suite to a temporary directory and compares timing and CPU
admission configurations under pytest-xdist. Reported per run: whole-process wall
time including pytest startup and reporting, the slowest heavy test, failures
(deadline misses in the ``deadline`` scenario), shared fixture set-ups paid, and,
where available, cgroup throttling and CPU pressure. JSON includes session time from
the timing report when present, otherwise process wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

BURN = "import time; t = time.process_time()\\nwhile time.process_time() - t < {seconds}: pass"

HEAVY = """
import subprocess, sys, time
import pytest

BURN = "{burn}"

def burn(n, seconds):
    code = BURN.format(seconds=seconds)
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(n)]
    for p in procs:
        p.wait()

@pytest.mark.parametrize("i", range({heavy}))
@pytest.mark.timing_cpu({slots})
def test_heavy(i):
    started = time.perf_counter()
    burn({slots}, {seconds})
    elapsed = time.perf_counter() - started
    if {deadline} and elapsed > {seconds} * {deadline}:
        raise AssertionError(f"took {{elapsed:.2f}}s, deadline {{{seconds} * {deadline}:.2f}}s")

@pytest.mark.parametrize("i", range({light}))
def test_light(i):
    t = time.process_time()
    while time.process_time() - t < {light_seconds}:
        pass
"""

MIXED = """
import subprocess, sys, time
import pytest

BURN = "{burn}"

def burn(n, seconds):
    code = BURN.format(seconds=seconds)
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(n)]
    for p in procs:
        p.wait()

@pytest.mark.parametrize("i", range(2))
@pytest.mark.timing_cpu(4)
def test_four(i):
    burn(4, 0.4)

@pytest.mark.parametrize("i", range(4))
@pytest.mark.timing_cpu(2)
def test_two(i):
    burn(2, 0.4)

@pytest.mark.parametrize("i", range(8))
def test_one(i):
    burn(1, 0.2)

@pytest.mark.parametrize("i", range(24))
def test_light(i):
    t = time.process_time()
    while time.process_time() - t < 0.02:
        pass
"""

FIXTURE = """
import subprocess, sys, time
import pytest
import pytest_timing

BURN = "{burn}"

@pytest_timing.cpu(4)
@pytest.fixture(scope="session")
def build():
    procs = [subprocess.Popen([sys.executable, "-c", BURN.format(seconds=0.5)]) for _ in range(4)]
    for p in procs:
        p.wait()
    yield

@pytest.mark.parametrize("i", range(40))
def test_uses_build(build, i):
    t = time.process_time()
    while time.process_time() - t < 0.03:
        pass
"""

SINGLE = """
import time
import pytest

@pytest.mark.parametrize("i", range({count}))
def test_cpu(i):
    t = time.process_time()
    while time.process_time() - t < {seconds}:
        pass
"""

TINY = "\n".join(f"def test_{i}(): pass" for i in range(3000)) + "\n"


def run_pytest(suite: Path, args: list[str], timeout: float = 900.0) -> dict[str, Any]:
    json_path = suite / "pytest-timing.json"
    # Keep the previous run readable while the scheduler starts. A distinct
    # output also prevents a failed invocation from masquerading as a fresh run.
    output_path = suite / "pytest-timing-output.json"
    output_path.unlink(missing_ok=True)
    if "--timing-json" in args:
        args = [*args, "--timing-json-file", str(output_path)]
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - started
    if "--timing-schedule" in args and "tests had recorded durations" not in proc.stdout:
        raise RuntimeError(f"scheduled benchmark did not load its history:\n{proc.stdout}")
    result: dict[str, Any] = {"wall": wall, "ret": proc.returncode, "stdout": proc.stdout}
    if output_path.exists():
        output_path.replace(json_path)
        doc = json.loads(json_path.read_text())
        tests = doc["tests"]
        heavy = [t for t in tests if "test_heavy" in t["nodeid"] or "test_four" in t["nodeid"]]
        result["session"] = doc["summary"]["wall"]
        result["failed"] = sum(1 for t in tests if t["outcome"] in ("failed", "error"))
        result["heavy_max"] = max((t["stop"] - t["start"] for t in heavy), default=0.0)
        result["heavy_mean"] = (
            statistics.fmean(t["stop"] - t["start"] for t in heavy) if heavy else 0.0
        )
        result["setups"] = sum(1 for t in tests for v in (t.get("fixtures") or {}).values() if v)
        records = [t.get("cpu") for t in tests if t.get("cpu")]
        result["throttled"] = sum(1 for r in records if r.get("throttled"))
        pressures = [r["pressure"] for r in records if r.get("pressure") is not None]
        result["pressure"] = max(pressures) if pressures else None
        result["waited"] = (doc["run"].get("cpu") or {}).get("waited", 0.0)
        result["work"] = sum(r.get("work", 0.0) for r in records)
    else:
        result["session"] = wall
        result["failed"] = -1
    for key in ("heavy_max", "heavy_mean", "setups", "throttled", "pressure", "waited", "work"):
        result.setdefault(key, None)
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


class Bench:
    def __init__(self, workers: int, cpus: int, root: Path, repeat: int) -> None:
        self.workers = workers
        self.cpus = cpus
        self.root = root
        self.repeat = repeat
        self.rows: list[dict[str, Any]] = []

    def suite(self, name: str, files: dict[str, str]) -> Path:
        path = self.root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        for filename, text in files.items():
            (path / filename).write_text(textwrap.dedent(text))
        return path

    def measure(self, scenario: str, config: str, suite: Path, args: list[str]) -> dict[str, Any]:
        best: dict[str, Any] | None = None
        for _ in range(self.repeat):
            result = run_pytest(suite, args)
            if best is None or result["wall"] < best["wall"]:
                best = result
        assert best is not None
        row = {"scenario": scenario, "config": config, **best}
        self.rows.append(row)
        print(
            f"  {scenario:<10} {config:<26} wall {best['wall']:6.2f}s"
            f"  heavy max {fmt(best['heavy_max']):>6}s"
            f"  failed {fmt(best['failed']):>3}  setups {fmt(best['setups']):>3}"
            f"  waited {fmt(best['waited']):>6}s  throttled {fmt(best['throttled'])}"
            f"  pressure {fmt(best['pressure'])}",
            flush=True,
        )
        return row

    def plain(self) -> list[str]:
        return ["-n", str(self.workers), "--timing-json"]

    def gated(self, budget: int | None = None) -> list[str]:
        return [*self.plain(), "--timing-cpus", str(budget or self.cpus)]

    def oversubscribed(self) -> tuple[int, int]:
        """Slots per heavy test and heavy test count such that, ungated, the heavy tests
        xdist starts together keep about four times the budget busy."""
        heavy = max(2, self.workers)
        slots = max(2, min(self.cpus, 4 * self.cpus // heavy))
        return slots, heavy

    def tiny(self) -> None:
        suite = self.suite("tiny", {"test_tiny.py": TINY})
        n = ["-n", str(self.workers)]
        self.measure("tiny", "plugin off", suite, [*n, "-p", "no:timing"])
        self.measure("tiny", "--timing", suite, [*n, "--timing"])
        self.measure("tiny", "--timing-json", suite, self.plain())
        self.measure("tiny", f"--timing-cpus {self.cpus}", suite, self.gated())

    def single(self) -> None:
        suite = self.suite("single", {"test_single.py": SINGLE.format(count=64, seconds=0.1)})
        self.measure("single", "xdist", suite, self.plain())
        self.measure("single", f"--timing-cpus {self.cpus}", suite, self.gated())

    def parallel(self, deadline: float = 0.0, name: str = "parallel") -> None:
        slots, heavy = self.oversubscribed()
        text = HEAVY.format(
            burn=BURN,
            heavy=heavy,
            slots=slots,
            seconds=0.4,
            light=48,
            light_seconds=0.02,
            deadline=deadline,
        )
        suite = self.suite(name, {"test_parallel.py": text})
        self.measure(name, "xdist", suite, self.plain())
        self.measure(name, f"--timing-cpus {self.cpus}", suite, self.gated())
        self.measure(
            name,
            "gate + schedule",
            suite,
            [*self.gated(), "--timing-schedule", "pytest-timing.json"],
        )

    def mixed(self) -> None:
        suite = self.suite("mixed", {"test_mixed.py": MIXED.format(burn=BURN)})
        self.measure("mixed", "xdist", suite, self.plain())
        self.measure("mixed", f"--timing-cpus {self.cpus}", suite, self.gated())

    def fixture(self) -> None:
        suite = self.suite("fixture", {"test_fixture.py": FIXTURE.format(burn=BURN)})
        self.measure("fixture", "xdist", suite, self.plain())
        self.measure("fixture", f"--timing-cpus {self.cpus}", suite, self.gated())
        self.measure(
            "fixture",
            "gate + schedule",
            suite,
            [*self.gated(), "--timing-schedule", "pytest-timing.json"],
        )

    def background(self) -> None:
        """The ``parallel`` suite while half the budget is busy with someone else's work."""
        load = max(1, self.cpus // 2)
        code = BURN.replace("\\n", "\n").format(seconds=600)
        burners = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(load)]
        try:
            slots, heavy = self.oversubscribed()
            text = HEAVY.format(
                burn=BURN,
                heavy=heavy,
                slots=slots,
                seconds=0.4,
                light=48,
                light_seconds=0.02,
                deadline=0.0,
            )
            suite = self.suite("background", {"test_parallel.py": text})
            self.measure("background", "xdist", suite, self.plain())
            self.measure("background", f"--timing-cpus {self.cpus}", suite, self.gated())
            self.measure(
                "background",
                f"--timing-cpus {self.cpus - load}",
                suite,
                self.gated(self.cpus - load),
            )
        finally:
            for proc in burners:
                proc.kill()
            for proc in burners:
                proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="budget for gated runs (default: greater of detected budget and worker count)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs per configuration; the fastest counts"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["tiny", "single", "parallel", "deadline", "mixed", "fixture", "background"],
    )
    parser.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    args = parser.parse_args()
    from pytest_timing.telemetry import host_cpu

    host = host_cpu()
    cpus = args.cpus or max(host.budget, args.workers)
    print(f"host: {host.to_dict()}")
    print(f"workers {args.workers}, budget {cpus}, python {sys.version.split()[0]}")
    root = Path(tempfile.mkdtemp(prefix="pytest-timing-bench-"))
    bench = Bench(args.workers, cpus, root, args.repeat)
    scenarios = args.scenario or [
        "tiny",
        "single",
        "parallel",
        "deadline",
        "mixed",
        "fixture",
        "background",
    ]
    for name in scenarios:
        print(f"[{name}]")
        if name == "deadline":
            bench.parallel(deadline=1.5, name="deadline")
        else:
            getattr(bench, name)()
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "host": host.to_dict(),
                    "workers": args.workers,
                    "budget": cpus,
                    "rows": [{k: v for k, v in row.items() if k != "stdout"} for row in bench.rows],
                },
                indent=1,
            )
        )
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
