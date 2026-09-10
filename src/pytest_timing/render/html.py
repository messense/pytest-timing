"""Self-contained HTML report: the static template with the run JSON embedded."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from pytest_timing.model import Run

PLACEHOLDER = "__PYTEST_TIMING_DATA__"


def load_template() -> str:
    return resources.files("pytest_timing.static").joinpath("report.html").read_text("utf-8")


def embed_json(doc: dict[str, Any]) -> str:
    """Compact JSON that is safe inside a ``<script type="application/json">`` element."""
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html_doc(run: Run, doc: dict[str, Any]) -> str:
    template = load_template()
    if PLACEHOLDER not in template:
        raise RuntimeError("report template is missing the data placeholder")
    return template.replace(PLACEHOLDER, embed_json(doc), 1)


def render_html(run: Run) -> str:
    return render_html_doc(run, run.to_dict())
