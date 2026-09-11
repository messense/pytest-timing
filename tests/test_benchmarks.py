"""Verify the benchmark exercises the scheduling mode its rows claim."""

import importlib.util
from pathlib import Path

import pytest
from conftest import needs_xdist


@needs_xdist
def test_benchmark_preserves_history_for_scheduled_runs(pytester: pytest.Pytester) -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "cpu_bench.py"
    spec = importlib.util.spec_from_file_location("cpu_bench", path)
    assert spec and spec.loader
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    pytester.makepyfile("def test_a(): pass\ndef test_b(): pass\n")
    args = ["-n2", "--timing-json", "--timing-cpus", "2"]
    assert bench.run_pytest(pytester.path, args)["ret"] == 0
    result = bench.run_pytest(pytester.path, [*args, "--timing-schedule", "pytest-timing.json"])
    assert result["ret"] == 0
    assert "2 of 2 tests had recorded durations" in result["stdout"]
