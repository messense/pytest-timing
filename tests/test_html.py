from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import chrome_binary, empty_run, make_info, report

from pytest_timing.collector import Collector
from pytest_timing.model import Run
from pytest_timing.render.html import PLACEHOLDER, embed_json, load_template, render_html


def test_template_is_self_contained() -> None:
    template = load_template()
    assert template.count(PLACEHOLDER) == 1
    assert template.lower().startswith("<!doctype html>")
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=[\"']https?://", template)
    assert "prefers-color-scheme" in template
    assert len(template.encode()) < 60_000


def test_render_embeds_data(sample_run: Run) -> None:
    html = render_html(sample_run)
    assert PLACEHOLDER not in html
    assert 'id="pytest-timing-data"' in html
    assert "tests/test_a.py::test_two" in html
    assert '"schema":1' in html


def test_embedded_json_cannot_break_out_of_script() -> None:
    run = empty_run()
    run.run.argv = ["pytest", "</script><b>x</b>&"]
    text = embed_json(run.to_dict())
    assert "</script>" not in text and "<" not in text and ">" not in text and "&" not in text
    assert "\\u003c/script\\u003e" in text


def test_empty_run_renders() -> None:
    html = render_html(empty_run())
    assert '"tests":[]' in html


def _chrome_dom(run: Run, tmp_path: Path) -> str:
    from conftest import render_in_chrome

    path = tmp_path / "report.html"
    path.write_text(render_html(run), encoding="utf-8")
    dom = render_in_chrome(path)
    marker = '<pre id="render-error"'
    assert marker not in dom, dom[dom.find(marker) :][:2000]
    return dom


needs_chrome = pytest.mark.skipif(chrome_binary() is None, reason="no headless Chrome found")


@needs_chrome
def test_report_executes_in_a_browser(sample_run: Run, tmp_path: Path) -> None:
    dom = _chrome_dom(sample_run, tmp_path)
    assert dom.count("<rect") >= len(sample_run.tests)
    assert "tests/test_a.py::test_two" in dom
    assert "Ended:" in dom and "finished" in dom


@needs_chrome
def test_large_report_executes_in_a_browser(tmp_path: Path) -> None:
    """150k tests on one lane: no argument-limit errors, merged bars, capped table."""
    c = Collector(make_info())
    c.worker_ready("gw0", make_info().start + 0.1)
    t = 0.2
    for i in range(150_000):
        for when in ("setup", "call", "teardown"):
            c.add_report(report(f"t.py::test[{i}]", when, t, t + 0.0001))
            t += 0.0001
    c.worker_down("gw0", make_info().start + t, None)
    run = c.finish(make_info().start + t + 0.1, termination="finished")
    dom = _chrome_dom(run, tmp_path)
    assert dom.count("<rect") > 0
    assert "150000" in dom


@needs_chrome
def test_empty_and_incomplete_reports_execute(tmp_path: Path) -> None:
    dom = _chrome_dom(empty_run("interrupted"), tmp_path)
    assert "incomplete run" in dom
