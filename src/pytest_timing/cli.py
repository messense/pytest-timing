"""``pytest-timing`` command line: re-render or merge saved JSON runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pytest_timing.model import Run, RunInfo, TestSpan, Worker
from pytest_timing.outputs import OUTPUTS, write_output
from pytest_timing.render.ascii import render_ascii


def _load(path: str) -> Run:
    return Run.from_json(Path(path).read_text("utf-8"))


def cmd_render(args: argparse.Namespace) -> int:
    run = _load(args.run)
    wrote = False
    doc = None
    for kind, output in OUTPUTS.items():
        path = getattr(args, kind)
        if path:
            doc = doc or run.to_dict()
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
