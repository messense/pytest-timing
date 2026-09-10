"""The report outputs, in one table shared by the pytest plugin and the CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pytest_timing.model import Run
from pytest_timing.render.html import render_html_doc
from pytest_timing.render.trace import render_trace

Renderer = Callable[[Run, dict[str, Any]], str]
"""Renders a run; the second argument is ``run.to_dict()``, built once and shared."""


def _render_json(run: Run, doc: dict[str, Any]) -> str:
    return json.dumps(doc, separators=(",", ":"))


def _render_trace(run: Run, doc: dict[str, Any]) -> str:
    return render_trace(run)


@dataclass(frozen=True, slots=True)
class Output:
    kind: str
    default: str  # default file name
    label: str  # how the terminal refers to it
    render: Renderer


OUTPUTS: dict[str, Output] = {
    o.kind: o
    for o in (
        Output("html", "pytest-timing.html", "HTML report", render_html_doc),
        Output("json", "pytest-timing.json", "JSON", _render_json),
        Output("trace", "pytest-timing.trace.json", "Chrome trace", _render_trace),
    )
}


def write_output(output: Output, path: Path, run: Run, doc: dict[str, Any] | None = None) -> str:
    """Render and write one output; returns the line to show the user.

    Pass ``doc`` (``run.to_dict()``) when writing several outputs so it is built once.
    """
    if doc is None:
        doc = run.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output.render(run, doc), encoding="utf-8")
    return f"{output.label} written to {path}"
