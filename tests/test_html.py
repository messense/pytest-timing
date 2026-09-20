from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import chrome_binary, empty_run, make_info, report

from pytest_timing.collector import Collector
from pytest_timing.model import CpuRecord, Run
from pytest_timing.model import TestSpan as Span
from pytest_timing.render.html import PLACEHOLDER, embed_json, load_template, render_html


def test_template_is_self_contained() -> None:
    template = load_template()
    assert template.count(PLACEHOLDER) == 1
    assert template.lower().startswith("<!doctype html>")
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=[\"']https?://", template)
    assert "prefers-color-scheme" in template
    assert len(template.encode()) < 80_000


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


@pytest.mark.parametrize("runtime_wait", [0, 1])
def test_cpu_timeline_preserves_recorded_work(runtime_wait: int) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    run = empty_run()
    run.tests = [
        Span(
            "a",
            "gw0",
            0,
            "passed",
            0,
            2,
            cpu=CpuRecord(
                elapsed=2 - runtime_wait, work=1, coverage="tree", runtime_wait=runtime_wait
            ),
        ),
        Span("b", "gw1", 0, "passed", 1, 3, cpu=CpuRecord(elapsed=2, work=2, coverage="self")),
        Span(
            "unknown", "gw2", 0, "passed", 0, 3, cpu=CpuRecord(elapsed=3, work=90, coverage="none")
        ),
        Span("empty", "gw3", 0, "passed", 4, 4, cpu=CpuRecord(coverage="self")),
    ]
    # Execute the actual template's data preparation, stopping before DOM rendering.
    # Only the embedded JSON element is needed; no browser or DOM emulation runs.
    script = re.search(r"<script>\s*(.*?)</script>", render_html(run), re.S)
    assert script is not None
    prepare, separator, _ = script[1].partition("  var MEM_EVENTS = ")
    assert separator
    result = subprocess.run(
        [
            node,
            "-e",
            """
            var document = {getElementById: function () {
                return {textContent: require('fs').readFileSync(0, 'utf8')};
            }};
        """
            + prepare
            + """
            process.stdout.write(JSON.stringify({events: CPU_EVENTS,
                rates: TESTS.map(function (t) { return t.cpuRate; })}));
            } catch (error) { throw error; }
        })();
        """,
        ],
        input=run.to_json(),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    data = json.loads(result.stdout)
    # Sum the piecewise-constant curve, including its overlap, in CPU seconds.
    rate = area = previous = 0.0
    for at, delta in data["events"]:
        area += rate * (at - previous)
        rate += delta
        previous = at
    assert area == pytest.approx(3)
    assert rate == pytest.approx(0)
    assert data["rates"] == [1 / (2 - runtime_wait), 1, None, None]


def run_js_kernel(names: list[str], script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js not installed")
    functions = []
    for name in names:
        found = re.search(r"  function " + name + r"\(.*?\n  }", load_template(), re.S)
        assert found is not None, name
        functions.append(found[0])
    subprocess.run([node, "-e", "\n".join(functions) + "\n" + script], check=True, timeout=10)


def test_zoom_sampling_and_axes_have_a_fixed_budget() -> None:
    run_js_kernel(
        ["sample", "stepPath", "niceStep", "tickLabel", "axisSvg"],
        """
        const assert = require('node:assert/strict');
        var SAMPLE_CAP = 4096, WALL = 3600, TOP_PAD = 24, BOTTOM_PAD = 22;
        var width = WALL * 2000;
        var s = sample([[0, 1], [1800, -1], [3599, 9], [3600, -9]], width, 2000);
        assert.equal(s.v.length, SAMPLE_CAP);
        assert.equal(s.v.length * s.stride, width);
        assert.equal(s.peak, 9);
        assert.equal(s.v[s.v.length - 1], 9);
        assert.ok(stepPath(s, 10, 8, 100).includes(' L7200008 '));
        var axis = axisSvg(8, width, 160, 2000);
        assert.ok((axis.match(/class="grid"/g) || []).length <= 1025);
        var small = sample([[0, 1], [2, -1]], 4, 2);
        assert.deepEqual(small.v, [1, 1, 1, 1]);
        assert.equal(small.stride, 1);
    """,
    )


def test_large_cluster_caches_top_ten_and_hover_details() -> None:
    run_js_kernel(
        ["addToCluster", "clusterLane", "tipForCluster"],
        """
        const assert = require('node:assert/strict');
        var BAD = {}, fmt = String, esc = String;
        var tests = [], at = 0, total = 0;
        for (var i = 0; i < 150000; i++) {
            var dur = .00001 + (i % 17) * .0000001;
            tests.push({nodeid: 'test' + i, start: at, stop: at + dur, dur: dur});
            at += dur; total += dur;
        }
        var clusters = clusterLane(tests, .1);
        assert.equal(clusters.length, 1);
        var c = clusters[0];
        assert.equal(c.total, total);
        assert.equal(c.tests.length, 150000);
        assert.equal(c.top.length, 10);
        assert.ok(c.top.every(t => t.dur === .00001 + 16 * .0000001));
        var first = tipForCluster(c);
        c.top.map = function () { throw Error('recomputed hover'); };
        assert.equal(tipForCluster(c), first);
    """,
    )


def test_visible_window_resolves_narrow_peaks_and_range_totals() -> None:
    run_js_kernel(
        ["sample", "eventStats"],
        """
        const assert = require('node:assert/strict');
        var SAMPLE_CAP = 4096;
        var ev = [[0, 1], [1800.02, 8], [1800.04, -8], [3600, -1]];
        var overview = sample(ev, 800, 800 / 3600);
        var detail = sample(ev, 800, 8000, 1800);
        assert.ok(detail.v.filter(v => v === 9).length >= 159);
        assert.ok(detail.v.filter(v => v === 9).length <= 162);
        assert.equal(detail.v[0], 1);
        assert.equal(detail.v[799], 1);
        assert.equal(overview.v.filter(v => v === 9).length, 1);
        assert.ok(Math.abs(eventStats(ev, 1800, 1800.1).area - .26) < 1e-9);
        assert.equal(eventStats(ev, 1800, 1800.1).peak, 9);
        assert.equal(eventStats([[1, 1], [2, -1], [2, 1], [3, -1]], 1, 3).peak, 1);
        assert.deepEqual(eventStats(ev, 3600, 3601), {area: 0, peak: 0});
    """,
    )


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
def test_report_zoom_scroll_range_and_fixture_controls(tmp_path: Path) -> None:
    from conftest import render_in_chrome

    from pytest_timing.model import MemoryRecord, WaitInterval

    run = empty_run()
    run.run.stop = run.run.start + 3600
    run.tests = [
        Span("early", "gw0", 0, "passed", 1, 2, fixtures={"module:database": 1}),
        Span(
            "long",
            "gw0",
            0,
            "passed",
            1799,
            1801,
            fixtures={"module:database": 0.5},
            cpu=CpuRecord(elapsed=2, work=2, coverage="self"),
            memory=MemoryRecord(base=0, peak=10, after=0, coverage="self"),
            admission_waits=[WaitInterval(1800.01, 1800.03, ("cpu", "memory"))],
        ),
        Span(
            "burst",
            "gw1",
            0,
            "passed",
            1800.02,
            1800.04,
            fixtures={"module:database": None},
            cpu=CpuRecord(elapsed=0.02, work=0.16, coverage="tree"),
            memory=MemoryRecord(base=0, peak=80, after=0, coverage="self"),
        ),
        Span("late", "gw0", 0, "passed", 3500, 3501),
    ]
    exercise = """
    <script>
    (async function () {
      var status = document.createElement('pre'); status.id = 'interaction-result';
      document.body.appendChild(status);
      function check(ok, message) { if (!ok) throw new Error(message); }
      function byId(id) { return document.getElementById(id); }
      function pause() { return new Promise(resolve => setTimeout(resolve, 120)); }
      var ids = ['pipeline-container', 'timing-container', 'cpu-container',
                 'mem-container', 'wait-container'];
      function linked(start) {
        ids.forEach(function (id) {
          var el = byId(id), svg = el.querySelector('svg');
          check(Math.abs(+el.dataset.rangeStart - start) < .01, id + ' not linked');
          check(svg.width.baseVal.value <= el.clientWidth + 1, id + ' oversized svg');
          check(Math.abs(svg.getBoundingClientRect().left - el.getBoundingClientRect().left) < 2,
                id + ' svg scrolled offscreen');
        });
      }
      try {
        check(!byId('render-error'), 'initial render failed');
        check(document.querySelectorAll('.wait-bar').length === 1, 'wait not on lane');
        check(document.querySelector('td[title="20.0ms for cpu, 20.0ms for memory"]')
              .textContent === '20.0ms', 'Held column double-counted a joint wait');
        var first = document.querySelector('#fixture-table tbody tr');
        check(first.cells[2].textContent === '2' && first.cells[3].textContent === '1.50s',
              'fixture repetition missing');
        first.querySelector('button').click();
        check(byId('table-note').textContent === '(3 of 4)', 'fixture filter');
        check(document.activeElement === byId('clear-fixture'), 'fixture focus lost');
        byId('clear-fixture').click();
        check(byId('table-note').textContent === '(4 of 4)', 'clear fixture');
        byId('scale').value = 1000;
        byId('scale').dispatchEvent(new Event('input', {bubbles: true})); await pause();
        var pc = byId('pipeline-container'); pc.scrollLeft = 1800 * 2000;
        pc.dispatchEvent(new Event('scroll')); await pause();
        linked(1800);
        var cpu = byId('cpu-container'), rect = cpu.querySelector('svg').getBoundingClientRect();
        cpu.dispatchEvent(new MouseEvent('mousemove', {clientX: rect.left + 150,
                          clientY: rect.top + 50, bubbles: true}));
        check(byId('tooltip').textContent.includes('30m'), 'hover did not use scrolled time');
        check(byId('tooltip').textContent.includes('9.00 CPUs'), 'zoom lost the narrow CPU peak');
        byId('range-start').value = 1800; byId('range-stop').value = 1800.1;
        byId('range-form').dispatchEvent(new Event('submit', {cancelable: true, bubbles: true}));
        await pause();
        linked(1800);
        ids.forEach(id => check(Math.abs(+byId(id).dataset.rangeStop - 1800.1) < .001,
                                id + ' wrong selection end'));
        check(byId('range-summary').textContent.includes('260.0ms measured CPU time'),
              'CPU range integral');
        check(byId('range-summary').textContent.includes('20.0ms located admission wait'),
              'wait counted twice');
        check(byId('table-note').textContent === '(2 of 4)', 'range filter');
        check(byId('fixture-note').textContent.includes('not clipped'), 'fixture precision notice');
        byId('view-unit').click();
        check(byId('pipeline-title').textContent.includes('Slowest'), 'unit view');
        byId('view-lane').click();
        byId('range-start').value = 10; byId('range-stop').value = 5;
        byId('range-form').dispatchEvent(new Event('submit', {cancelable: true}));
        check(byId('range-error').textContent.includes('start before'), 'invalid range accepted');
        byId('clear-range').click(); await pause(); linked(0);
        check(byId('table-note').textContent === '(4 of 4)', 'clear selection');
        check(byId('range-summary').textContent.startsWith('Whole run:'), 'summary not reset');
        check(!byId('range-error').textContent, 'range error not cleared');
        check(!byId('render-error'), 'interaction render failed');
        status.textContent = 'passed';
      } catch (error) { status.textContent = error.stack; }
    })();
    </script>
    """
    path = tmp_path / "interactive.html"
    path.write_text(render_html(run).replace("</body>", exercise + "</body>"), encoding="utf-8")
    dom = render_in_chrome(path, virtual_time=3000)
    result = re.search(r'<pre id="interaction-result">(.*?)</pre>', dom, re.S)
    assert result is not None, dom[-2000:]
    assert result[1] == "passed", result[1]


@needs_chrome
def test_slowest_labels_fit_the_visible_chart(tmp_path: Path) -> None:
    from conftest import render_in_chrome

    run = empty_run()
    run.run.stop = run.run.start + 20
    long_name = "tests/test_long.py::test_parameters[" + "long_parameter_" * 30 + "]"
    run.tests = [
        Span("early", "gw0", 0, "passed", 1, 2),
        Span(long_name, "gw1", 0, "passed", 3, 20),
        Span("tests/test_last.py::test_finishes_at_end", "gw0", 0, "passed", 19, 20),
    ]
    exercise = """
    <script>
    (function () {
      var status = document.createElement('pre'); status.id = 'label-result';
      document.body.appendChild(status);
      function check(ok, message) { if (!ok) throw Error(message); }
      function verify(expected) {
        var svg = document.querySelector('#pipeline-container svg');
        var viewport = svg.getBoundingClientRect();
        var labels = Array.from(svg.querySelectorAll('text'))
          .filter(el => el.textContent.includes(' (gw'));
        check(labels.length === expected, 'missing labels');
        labels.forEach(function (label) {
          var box = label.getBoundingClientRect();
          check(box.width > 0 && box.height > 0, 'empty label');
          check(box.left >= viewport.left && box.right <= viewport.right,
                'label outside viewport: ' + label.textContent);
          check(box.top >= viewport.top && box.bottom <= viewport.bottom,
                'label outside vertical viewport');
        });
        var name = JSON.parse(document.getElementById('pytest-timing-data').textContent)
          .tests[1].nodeid;
        check(labels.some(el => el.textContent === name + ' (gw1): 17.00s'),
              'long label text was truncated');
      }
      try {
        document.getElementById('view-unit').click(); verify(3);
        document.getElementById('range-start').value = 18;
        document.getElementById('range-stop').value = 20;
        document.getElementById('range-form')
          .dispatchEvent(new Event('submit', {cancelable: true}));
        verify(2);
        document.getElementById('clear-range').click(); verify(3);
        status.textContent = 'passed';
      } catch (error) { status.textContent = error.stack; }
    })();
    </script>
    """
    path = tmp_path / "labels.html"
    path.write_text(render_html(run).replace("</body>", exercise + "</body>"), encoding="utf-8")
    dom = render_in_chrome(path)
    result = re.search(r'<pre id="label-result">(.*?)</pre>', dom, re.S)
    assert result is not None, dom[-2000:]
    assert result[1] == "passed", result[1]


@needs_chrome
def test_report_executes_in_a_browser(sample_run: Run, tmp_path: Path) -> None:
    dom = _chrome_dom(sample_run, tmp_path)
    assert dom.count("<rect") >= len(sample_run.tests)
    assert "tests/test_a.py::test_two" in dom
    assert "Ended:" in dom and "finished" in dom


@needs_chrome
def test_report_shows_how_long_admission_held_a_test(sample_run: Run, tmp_path: Path) -> None:
    from pytest_timing.model import CpuRecord, MemoryRecord

    dom = _chrome_dom(sample_run, tmp_path)
    assert "Held</th>" not in dom  # nothing waited: no column
    two = next(t for t in sample_run.tests if t.nodeid.endswith("test_two"))
    two.memory = MemoryRecord(base=1, peak=2, after=1, coverage="self", wait=0.3)
    two.cpu = CpuRecord(elapsed=1.0, wait=1.25)
    dom = _chrome_dom(sample_run, tmp_path)
    assert "Held</th>" in dom
    assert 'title="1.25s for cpu, 300.0ms for memory">1.55s</td>' in dom
    assert dom.count('title="">-</td>') == len(sample_run.tests) - 1


@needs_chrome
def test_report_shows_cpu_and_memory_usage(sample_run: Run, tmp_path: Path) -> None:
    from pytest_timing.model import CpuRecord, MemoryRecord

    dom = _chrome_dom(sample_run, tmp_path)
    assert "CPU time</th>" not in dom and "Memory</th>" not in dom  # nothing measured
    assert '<div id="cpu-usage" hidden' in dom and '<div id="mem-usage" hidden' in dom

    mib = 1024 * 1024
    two = next(t for t in sample_run.tests if t.nodeid.endswith("test_two"))
    two.cpu = CpuRecord(elapsed=2.0, work=3.0, demand=4, coverage="tree")
    two.memory = MemoryRecord(base=100 * mib, peak=612 * mib, after=110 * mib, coverage="tree")
    four = next(t for t in sample_run.tests if t.nodeid.endswith("test_four"))
    four.cpu = CpuRecord(elapsed=four.duration, wait=0.5)  # held back, but nothing measured
    sample_run.run.cpu = {
        "gated": True,
        "domains": {"local": {"budget": 4, "lowest": 4, "host": {"cpus": 8}}},
        "waited": 0.5,
        "waited_tests": 1,
    }
    dom = _chrome_dom(sample_run, tmp_path)
    for heading in ("CPU time</th>", "CPUs</th>", "Memory</th>"):
        assert heading in dom
    assert '>3.00s</td><td class="num">1.50 <span class="muted">/ 4</span></td>' in dom
    assert "100.0 MiB before, 612.0 MiB at peak, 110.0 MiB after (+10.0 MiB kept)" in dom
    assert ">+512.0 MiB</td>" in dom
    unmeasured = len(sample_run.tests) - 1
    assert dom.count('<td class="num">-</td><td class="num">-</td><td class="num">-</td>') == (
        unmeasured
    )
    assert "3.00s of CPU time, 1.00 CPUs busy on average, up to 2.14" in dom
    assert "8 cpus, budget 4 slots; 1 test waited 500.0ms for slots" in dom
    assert "up to 612.0 MiB resident across workers" in dom
    assert '<div id="cpu-usage">' in dom and '<div id="mem-usage">' in dom
    for kind in ("cpu", "mem"):
        assert re.search(rf'id="{kind}-container"[^>]*><div class="chart-space"[^>]*><svg', dom)


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
