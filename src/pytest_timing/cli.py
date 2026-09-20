"""``pytest-timing`` command line: render, merge or compare saved JSON runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from pytest_timing.compare import compare_runs, format_comparison, parse_budget
from pytest_timing.model import Run, RunInfo, TestSpan, Worker
from pytest_timing.outputs import OUTPUTS, write_output
from pytest_timing.render.ascii import render_ascii


def _number(value: Any, where: str) -> None:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{where} must be a number")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{where} must be finite")


def _record(
    value: Any, where: str, *, required: str = "", numbers: str = "", optional: str = ""
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    for key in required.split():
        if key not in value:
            raise ValueError(f"{where}.{key} is required")
    for key in numbers.split():
        if key in value:
            _number(value[key], f"{where}.{key}")
    for key in optional.split():
        if value.get(key) is not None:
            _number(value[key], f"{where}.{key}")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _load(path: str) -> Run:
    # Check external input before model conversion. Programming errors in the
    # model, comparison or formatter must propagate rather than becoming exit 2.
    doc = _record(
        json.loads(Path(path).read_text("utf-8")), "report", required="run", numbers="schema"
    )
    info = _record(
        doc["run"],
        "run",
        required="start stop",
        numbers="start stop",
        optional="numprocesses exit_status",
    )
    if info.get("termination") is not None and not isinstance(info["termination"], str):
        raise ValueError("run.termination must be a string")
    _array(info.get("argv", []), "run.argv")
    for i, value in enumerate(_array(doc.get("workers", []), "workers")):
        _record(value, f"workers[{i}]", required="id", optional="start ready collected down items")
    for i, value in enumerate(_array(doc.get("tests", []), "tests")):
        where = f"tests[{i}]"
        test = _record(
            value,
            where,
            required="nodeid worker outcome start stop",
            numbers="start stop attempt occurrence",
        )
        for name, phase in _record(test.get("phases", {}), f"{where}.phases").items():
            at = f"{where}.phases.{name}"
            if len(_array(phase, at)) != 3:
                raise ValueError(f"{at} must contain start, stop and duration")
            for number in phase:
                _number(number, at)
        for key, seconds in _record(test.get("fixtures") or {}, f"{where}.fixtures").items():
            if seconds is not None:
                _number(seconds, f"{where}.fixtures.{key}")
        if test.get("cpu") is not None:
            _record(
                test["cpu"],
                f"{where}.cpu",
                numbers="elapsed work setup_work demand wait runtime_wait",
                optional="pressure",
            )
        if test.get("memory") is not None:
            _record(test["memory"], f"{where}.memory", numbers="base peak after wait")
        for j, value in enumerate(
            _array(test.get("admission_waits", []), f"{where}.admission_waits")
        ):
            at = f"{where}.admission_waits[{j}]"
            wait = _record(value, at, required="start stop", numbers="start stop")
            if not all(isinstance(g, str) for g in _array(wait.get("gates", []), f"{at}.gates")):
                raise ValueError(f"{at}.gates must contain strings")
    return Run.from_dict(doc)


def cmd_render(args: argparse.Namespace) -> int:
    run = _load(args.run)
    wrote = False
    doc = None
    for kind, output in OUTPUTS.items():
        path = getattr(args, kind)
        if path:
            if doc is None and output.needs_doc:
                doc = run.to_dict()
            print(write_output(output, Path(path), run, doc))
            wrote = True
    if args.ascii or not wrote:
        print(
            render_ascii(
                run,
                width=args.width,
                top=args.top,
                min_duration=args.min,
                style=args.style,
                color=sys.stdout.isatty() and not args.no_color,
            )
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_runs(_load(args.before), _load(args.after), args.budget)
        if args.json:
            Path(args.json).write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
        print(format_comparison(result, args.top))
        return int(result["exit_code"])
    except (OSError, ValueError) as exc:
        print(f"pytest-timing compare: {exc}", file=sys.stderr)
        return 2


def _uniquify(candidate: str, used: set[str]) -> str:
    """``candidate``, or ``candidate#2``, ``#3``... until it is not in ``used``."""
    name, n = candidate, 1
    while name in used:
        n += 1
        name = f"{candidate}#{n}"
    used.add(name)
    return name


def merge_runs(runs: list[Run], names: list[str]) -> Run:
    """Place several runs on one shared axis.

    Each run is rebased to the earliest start (``Run.rebased`` moves every time
    together, so durations survive), and workers are relabelled so every input worker
    stays a distinct lane. This function is the single owner of worker relabelling.
    """
    if not runs:
        raise ValueError("nothing to merge")
    if len(names) != len(runs):
        raise ValueError("one name per run is required")
    base = min(r.run.start for r in runs)
    first = runs[0].run
    info = RunInfo(
        start=base,
        stop=max(r.run.stop for r in runs),
        termination=_merged_termination(runs),
        reason=None,
        exit_status=max((r.run.exit_status or 0) for r in runs),
        argv=list(first.argv),
        rootdir=first.rootdir,
        python=first.python,
        pytest=first.pytest,
        xdist=first.xdist,
        dist=first.dist,
        numprocesses=sum(r.run.numprocesses or 0 for r in runs) or None,
    )
    distinct = {w for r in runs for w in r.worker_ids()}
    collide = len(distinct) < sum(len(r.worker_ids()) for r in runs)
    used: set[str] = set()
    workers: list[Worker] = []
    tests: list[TestSpan] = []
    for run, name in zip(runs, names, strict=True):
        mapping = {
            wid: _uniquify(f"{name}/{wid}" if collide else wid, used) for wid in run.worker_ids()
        }
        placed = run.rebased(base).relabelled(mapping)
        workers.extend(placed.workers)
        tests.extend(placed.tests)
    return Run(run=info, workers=workers, tests=tests)


def _merged_termination(runs: list[Run]) -> str:
    if all(r.run.complete for r in runs):
        return "finished"
    for candidate in ("internal_error", "aborted", "interrupted", "unknown"):
        if any(r.run.termination == candidate for r in runs):
            return candidate
    return "unknown"


def unique_names(paths: list[str]) -> list[str]:
    """Short, distinct labels for input files: stems, then paths, then a counter."""
    names = [Path(p).stem for p in paths]
    if len(set(names)) == len(names):
        return names
    used: set[str] = set()
    return [_uniquify(Path(p).with_suffix("").as_posix().lstrip("./"), used) for p in paths]


def cmd_merge(args: argparse.Namespace) -> int:
    runs = [_load(p) for p in args.runs]
    merged = merge_runs(runs, unique_names(args.runs))
    Path(args.output).write_text(merged.to_json(), encoding="utf-8")
    print(f"merged {len(runs)} runs into {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pytest-timing", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render a saved run (JSON) as ASCII, HTML or trace")
    render.add_argument("run", help="pytest-timing JSON file")
    for kind, output in OUTPUTS.items():
        render.add_argument(f"--{kind}", metavar="PATH", help=f"write the {output.label}")
    render.add_argument("--ascii", action="store_true", help="print the ASCII chart")
    render.add_argument("--width", type=int, default=100)
    render.add_argument("--top", type=int, default=10)
    render.add_argument("--min", type=float, default=0.0, help="min duration for slowest tests")
    render.add_argument("--style", choices=("unicode", "ascii"), default="unicode")
    render.add_argument("--no-color", action="store_true")
    render.set_defaults(func=cmd_render)

    merge = sub.add_parser("merge", help="merge several runs (e.g. CI shards) onto one axis")
    merge.add_argument("runs", nargs="+", help="pytest-timing JSON files")
    merge.add_argument("-o", "--output", required=True, metavar="PATH")
    merge.set_defaults(func=cmd_merge)
    compare = sub.add_parser(
        "compare", help="compare two saved runs and check per-test regression budgets"
    )
    compare.add_argument("before", help="baseline JSON run")
    compare.add_argument("after", help="current JSON run")
    compare.add_argument(
        "--budget",
        action="append",
        type=parse_budget,
        default=[],
        metavar="METRIC=LIMIT",
        help="maximum increase per common test, e.g. duration=10%%, cpu=50ms, memory=64MiB",
    )
    compare.add_argument("--json", metavar="PATH", help="write all differences and budget results")
    compare.add_argument("--top", type=int, default=20, help="test details to print (default: 20)")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
