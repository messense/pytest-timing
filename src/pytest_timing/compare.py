"""Compare saved runs by test identity, with explicit missing data and CI budgets."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Any

from pytest_timing.model import Run, TestSpan

METRICS = ("duration", "setup", "call", "teardown", "cpu", "memory")


@dataclass(frozen=True)
class Budget:
    metric: str
    limit: float
    relative: bool


def parse_budget(text: str) -> Budget:
    metric, separator, value = text.partition("=")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(%|ms|s|B|KiB|MiB|GiB)?", value)
    if not separator or metric not in METRICS or match is None:
        raise argparse.ArgumentTypeError(
            "budget must be METRIC=LIMIT, e.g. duration=10%, call=50ms or memory=64MiB; "
            f"metrics: {', '.join(METRICS)}"
        )
    amount, unit = float(match[1]), match[2] or ("B" if metric == "memory" else "s")
    units = (
        {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
        if metric == "memory"
        else {
            "s": 1,
            "ms": 0.001,
        }
    )
    if unit != "%" and unit not in units:
        raise argparse.ArgumentTypeError(f"unit {unit!r} does not apply to {metric}")
    limit = amount if unit == "%" else amount * units[unit]
    if not math.isfinite(limit):
        raise argparse.ArgumentTypeError("budget must be finite")
    return Budget(metric, limit, unit == "%")


def _measurements(test: TestSpan) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "duration": test.duration,
        **{p: test.phases[p].duration if p in test.phases else None for p in METRICS[1:4]},
        "cpu": test.cpu.work if test.cpu is not None and test.cpu.coverage != "none" else None,
        "memory": test.memory.rise if test.memory is not None and test.memory.measured else None,
    }
    return {
        k: v if v is not None and math.isfinite(v) and v >= 0 else None for k, v in values.items()
    }


def _tests(run: Run) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[TestSpan]] = {}
    for test in run.tests:
        grouped.setdefault(test.nodeid, []).append(test)
    result = {}
    for nodeid, attempts in grouped.items():
        final: dict[tuple[str, int], TestSpan] = {}
        for test in attempts:
            key = (test.worker, test.occurrence)
            previous = final.get(key)
            if previous is None or (test.attempt, test.stop) > (previous.attempt, previous.stop):
                final[key] = test
        observations = [_measurements(t) for t in final.values()]
        metrics = {}
        for metric in METRICS:
            values = [value for row in observations if (value := row[metric]) is not None]
            # Partial coverage must not silently change the population being compared.
            metrics[metric] = median(values) if len(values) == len(observations) else None
        result[nodeid] = {
            "metrics": metrics,
            "occurrences": len(final),
            "attempts": len(attempts),
            "retries": len(attempts) - len(final),
            "successful": all(t.outcome in ("passed", "xpassed") for t in final.values()),
        }
    return result


def compare_runs(before: Run, after: Run, budgets: list[Budget] | None = None) -> dict[str, Any]:
    budgets = budgets or []
    if len({b.metric for b in budgets}) != len(budgets):
        raise ValueError("specify at most one budget per metric")
    left, right = _tests(before), _tests(after)
    rows, violations, unavailable = [], [], []
    for nodeid in sorted(left.keys() | right.keys()):
        a, b = left.get(nodeid), right.get(nodeid)
        status = "added" if a is None else "removed" if b is None else "common"
        changes = {}
        if a is not None and b is not None:
            for metric in METRICS:
                old, new = a["metrics"][metric], b["metrics"][metric]
                changes[metric] = {
                    "delta": None if old is None or new is None else new - old,
                    "percent": None
                    if old is None or new is None or old == 0
                    else (new - old) / old * 100,
                }
            for budget in budgets:
                old, new = a["metrics"][budget.metric], b["metrics"][budget.metric]
                if old is None or new is None or not (a["successful"] and b["successful"]):
                    unavailable.append({"nodeid": nodeid, "metric": budget.metric})
                    continue
                maximum = old + (old * budget.limit / 100 if budget.relative else budget.limit)
                if new > maximum and not math.isclose(new, maximum, rel_tol=1e-12, abs_tol=1e-9):
                    violations.append(
                        {
                            "nodeid": nodeid,
                            "metric": budget.metric,
                            "before": old,
                            "after": new,
                            "limit": budget.limit,
                            "relative": budget.relative,
                        }
                    )
        rows.append(
            {"nodeid": nodeid, "status": status, "before": a, "after": b, "changes": changes}
        )
    errors = []
    if budgets:
        if not before.run.complete or not after.run.complete:
            errors.append("budgets require two complete runs")
        if not left.keys() & right.keys():
            errors.append("budgets require tests present in both runs")
        if unavailable:
            errors.append("some budgeted metrics lack complete successful observations")
    warnings = []
    for field in ("python", "pytest", "numprocesses", "dist"):
        if getattr(before.run, field) != getattr(after.run, field):
            warnings.append(f"{field} differs between runs")
    if not before.run.complete or not after.run.complete:
        warnings.append("one or both runs are incomplete")
    return {
        "schema": 1,
        "aggregation": "median of final attempts per worker and occurrence; missing stays unknown",
        "counts": {
            status: sum(r["status"] == status for r in rows)
            for status in ("common", "added", "removed")
        },
        "tests": rows,
        "violations": violations,
        "unavailable": unavailable,
        "errors": errors,
        "warnings": warnings,
        "exit_code": 2 if errors else 1 if violations else 0,
    }


def format_comparison(result: dict[str, Any], top: int = 20) -> str:
    counts = result["counts"]
    lines = [
        f"Comparison: {counts['common']} common, {counts['added']} added, "
        f"{counts['removed']} removed"
    ]
    lines.extend("Warning: " + w for w in result["warnings"])
    rows = sorted(
        result["tests"], key=lambda r: -abs(r["changes"].get("duration", {}).get("delta") or 0)
    )

    def fmt(value: float | None, metric: str) -> str:
        if value is None:
            return "unavailable"
        return f"{value / 1024**2:.3g} MiB" if metric == "memory" else f"{value:.6g}s"

    for row in rows[: max(0, top)]:
        if row["status"] != "common":
            lines.append(f"{row['status']}: {row['nodeid']}")
            continue
        parts = []
        for metric in METRICS:
            a, b = row["before"]["metrics"][metric], row["after"]["metrics"][metric]
            percent = row["changes"][metric]["percent"]
            suffix = "" if percent is None else f" ({percent:+.1f}%)"
            parts.append(f"{metric} {fmt(a, metric)} -> {fmt(b, metric)}{suffix}")
        lines.append(row["nodeid"] + ": " + "; ".join(parts))
        if row["before"]["retries"] or row["after"]["retries"]:
            lines.append(f"  retries: {row['before']['retries']} -> {row['after']['retries']}")
    if len(rows) > top:
        lines.append(f"Showing {max(0, top)} of {len(rows)} tests; --json writes every comparison.")
    lines.append(
        f"Budget violations: {len(result['violations'])}; "
        f"unavailable checks: {len(result['unavailable'])}"
    )
    for violation in result["violations"]:
        metric = violation["metric"]
        limit = (
            f"{violation['limit']:g}%" if violation["relative"] else fmt(violation["limit"], metric)
        )
        lines.append(f"Budget exceeded: {violation['nodeid']} ({metric}, allowed increase {limit})")
    lines.extend("Error: " + e for e in result["errors"])
    return "\n".join(lines)
