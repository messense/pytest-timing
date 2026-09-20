from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import empty_run

from pytest_timing.cli import main
from pytest_timing.compare import compare_runs, format_comparison, parse_budget
from pytest_timing.model import CpuRecord, MemoryRecord, Phase, Run
from pytest_timing.model import TestSpan as Span


def observation(duration: float = 1, **changes: object) -> Span:
    span = Span(
        "test_a.py::test_a",
        "gw0",
        0,
        "passed",
        0,
        duration,
        phases={
            "setup": Phase(0, 0.2, 0.2),
            "call": Phase(0.2, duration, duration - 0.2),
            "teardown": Phase(duration, duration, 0),
        },
        cpu=CpuRecord(elapsed=duration, work=duration, coverage="tree"),
        memory=MemoryRecord(base=10, peak=110, after=10, coverage="self"),
    )
    return replace(span, **changes)  # type: ignore[arg-type]


def run(*tests: Span) -> Run:
    result = empty_run()
    result.tests = list(tests)
    return result


def test_comparison_matches_identity_and_keeps_added_removed_unknown() -> None:
    before = run(observation(), observation(nodeid="removed"))
    after = run(observation(2, worker="gw8", cpu=None), observation(nodeid="added"))
    result = compare_runs(before, after)
    assert result["counts"] == {"common": 1, "added": 1, "removed": 1}
    row = next(r for r in result["tests"] if r["status"] == "common")
    assert row["changes"]["duration"] == {"delta": 1, "percent": 100}
    assert row["changes"]["cpu"] == {"delta": None, "percent": None}
    assert row["after"]["metrics"]["setup"] == 0.2
    assert row["after"]["metrics"]["call"] == 1.8
    assert row["after"]["metrics"]["memory"] == 100
    assert "+100.0%" in format_comparison(result)


def test_comparison_uses_final_retries_and_median_occurrences() -> None:
    before = run(observation(2))
    after = run(
        observation(100, outcome="rerun"),
        observation(2, attempt=1),
        observation(4, occurrence=1),
        observation(9, worker="gw1"),
    )
    result = compare_runs(before, after)
    row = result["tests"][0]
    assert row["after"]["metrics"]["duration"] == 4
    assert row["after"]["attempts"] == 4
    assert row["after"]["occurrences"] == 3
    assert row["after"]["retries"] == 1
    assert "retries: 0 -> 1" in format_comparison(result)


@pytest.mark.parametrize(
    "budget,limit,relative",
    [
        ("duration=10%", 10, True),
        ("call=50ms", 0.05, False),
        ("cpu=2", 2, False),
        ("memory=2MiB", 2 * 1024**2, False),
        ("teardown=0s", 0, False),
        ("memory=10%", 10, True),
    ],
)
def test_budget_units(budget: str, limit: float, relative: bool) -> None:
    parsed = parse_budget(budget)
    assert (parsed.limit, parsed.relative) == (limit, relative)


@pytest.mark.parametrize(
    "budget",
    ["duration=-1", "unknown=5", "cpu=nan", "cpu=1MiB", "memory=1s", "cpu=inf", "cpu=" + "9" * 400],
)
def test_invalid_budgets(budget: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_budget(budget)


@pytest.mark.parametrize("budget", ["duration=10%", "duration=100ms"])
def test_budget_boundary_and_regression(budget: str) -> None:
    before = run(observation())
    assert compare_runs(before, run(observation(1.1)), [parse_budget(budget)])["exit_code"] == 0
    result = compare_runs(before, run(observation(1.10001)), [parse_budget(budget)])
    assert result["exit_code"] == 1
    assert result["violations"][0]["nodeid"] == "test_a.py::test_a"
    assert "Budget exceeded" in format_comparison(result, top=0)


def test_zero_baseline_does_not_invent_a_percentage() -> None:
    before = run(observation(0.2))
    after = run(observation(0.3))
    result = compare_runs(before, after, [parse_budget("call=10%")])
    assert result["tests"][0]["changes"]["call"]["percent"] is None
    assert result["exit_code"] == 1


@pytest.mark.parametrize("metric", ["cpu", "memory", "call"])
def test_partial_measurements_make_budget_unavailable(metric: str) -> None:
    unknown = observation(
        occurrence=1, cpu=CpuRecord(work=100, coverage="none"), memory=MemoryRecord(), phases={}
    )
    result = compare_runs(
        run(observation()), run(observation(), unknown), [parse_budget(f"{metric}=0%")]
    )
    assert result["exit_code"] == 2
    assert result["tests"][0]["after"]["metrics"][metric] is None
    assert result["unavailable"] == [{"nodeid": unknown.nodeid, "metric": metric}]


def test_incomplete_failed_and_disjoint_runs_do_not_pass_budgets() -> None:
    before = run(observation())
    incomplete = run(observation())
    incomplete.run.termination = "interrupted"
    budget = [parse_budget("duration=10%")]
    for after in (incomplete, run(observation(outcome="failed")), run(observation(nodeid="new"))):
        assert compare_runs(before, after, budget)["exit_code"] == 2
        assert compare_runs(before, after)["exit_code"] == 0
    assert compare_runs(before, incomplete)["warnings"] == ["one or both runs are incomplete"]
    with pytest.raises(ValueError, match="one budget"):
        compare_runs(before, before, budget * 2)


def test_compare_cli_json_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before, after, output = (
        tmp_path / name for name in ("before.json", "after.json", "delta.json")
    )
    before.write_text(run(observation()).to_json())
    after.write_text(run(observation(2)).to_json())
    args = ["compare", str(before), str(after)]
    assert main(args) == 0
    assert main([*args, "--budget", "duration=10%", "--json", str(output)]) == 1
    assert json.loads(output.read_text())["exit_code"] == 1
    assert "Budget exceeded" in capsys.readouterr().out
    malformed = run(observation()).to_dict()
    malformed["tests"][0]["phases"]["call"] = [0, 1]
    for contents in ("{broken", "null", "[]", json.dumps(malformed)):
        after.write_text(contents)
        assert main(args) == 2
        assert "pytest-timing compare:" in capsys.readouterr().err
    after.unlink()
    assert main(args) == 2


@pytest.mark.parametrize("stage", ["decode", "compare", "format"])
@pytest.mark.parametrize("error", [AttributeError, TypeError])
def test_programming_errors_are_not_reported_as_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, error: type[Exception]
) -> None:
    from pytest_timing import cli

    saved = tmp_path / "run.json"
    saved.write_text(run(observation()).to_json())

    def broken(*args: object, **kwargs: object) -> None:
        raise error("programming error")

    if stage == "decode":
        monkeypatch.setattr(Run, "from_dict", broken)
    else:
        monkeypatch.setattr(
            cli, "compare_runs" if stage == "compare" else "format_comparison", broken
        )
    with pytest.raises(error, match="programming error"):
        main(["compare", str(saved), str(saved)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("phases", []),
        ("cpu", {"work": []}),
        ("memory", {"peak": None}),
        ("admission_waits", [None]),
        ("start", None),
        ("stop", float("inf")),
    ],
)
def test_compare_rejects_invalid_nested_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, value: object
) -> None:
    doc = run(observation()).to_dict()
    doc["tests"][0][field] = value
    saved = tmp_path / "invalid.json"
    saved.write_text(json.dumps(doc))
    assert main(["compare", str(saved), str(saved)]) == 2
    assert f"tests[0].{field}" in capsys.readouterr().err
