from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import shifted_copy

from pytest_timing.cli import main, merge_runs
from pytest_timing.model import Run


@pytest.fixture
def run_file(sample_run: Run, tmp_path: Path) -> Path:
    path = tmp_path / "run.json"
    path.write_text(sample_run.to_json())
    return path


def test_render_ascii_default(run_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["render", str(run_file), "--width", "80"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("pytest-timing: 6 tests")


def test_render_outputs(run_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    html = tmp_path / "r.html"
    trace = tmp_path / "r.trace.json"
    assert main(["render", str(run_file), "--html", str(html), "--trace", str(trace)]) == 0
    assert html.exists() and "<html" in html.read_text().lower()
    assert json.loads(trace.read_text())["traceEvents"]
    out = capsys.readouterr().out
    assert "HTML report written" in out and "pytest-timing:" not in out


def test_merge_offsets_by_start_time(sample_run: Run, tmp_path: Path) -> None:
    later = shifted_copy(sample_run, 10)
    merged = merge_runs([sample_run, later], ["a", "b"])
    assert merged.run.start == sample_run.run.start
    assert merged.run.stop == later.run.stop
    assert merged.run.complete is True
    # colliding worker ids get prefixed with the run name
    assert sorted(merged.worker_ids()) == ["a/gw0", "a/gw1", "b/gw0", "b/gw1"]
    b_tests = [t for t in merged.tests if t.worker.startswith("b/")]
    assert min(t.start for t in b_tests) == 0.6 + 10
    assert merged.wall >= 13.0


def test_merge_cli(sample_run: Run, run_file: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(shifted_copy(sample_run, 5, {"gw0": "xgw0", "gw1": "xgw1"}).to_json())
    out = tmp_path / "merged.json"
    assert main(["merge", str(run_file), str(other), "-o", str(out)]) == 0
    merged = Run.from_json(out.read_text())
    assert sorted(merged.worker_ids()) == ["gw0", "gw1", "xgw0", "xgw1"]
    assert len(merged.tests) == 12


def test_unique_names_for_same_named_shard_files(tmp_path: Path) -> None:
    from pytest_timing.cli import unique_names

    a = tmp_path / "shard1" / "pytest-timing.json"
    b = tmp_path / "shard2" / "pytest-timing.json"
    names = unique_names([str(a), str(b)])
    assert len(set(names)) == 2
    assert names[0].endswith("shard1/pytest-timing")
    assert unique_names([str(a), str(a)]) != [names[0], names[0]]
    assert unique_names(["x.json", "y.json"]) == ["x", "y"]


def test_merge_same_named_shards_keeps_workers_apart(sample_run: Run, tmp_path: Path) -> None:
    files = []
    for i, name in enumerate(("shard1", "shard2")):
        d = tmp_path / name
        d.mkdir()
        path = d / "pytest-timing.json"
        path.write_text(shifted_copy(sample_run, 10 * i).to_json())
        files.append(str(path))
    out = tmp_path / "merged.json"
    assert main(["merge", *files, "-o", str(out)]) == 0
    merged = Run.from_json(out.read_text())
    assert len(merged.worker_ids()) == 4
    assert len(set(merged.worker_ids())) == 4


def test_merge_preserves_boot_duration(sample_run: Run) -> None:
    later = shifted_copy(sample_run, 10)
    merged = merge_runs([sample_run, later], ["a", "b"])
    original = sample_run.workers[0]
    assert original.start is not None and original.ready is not None
    boot = original.ready - original.start
    b_gw0 = next(w for w in merged.workers if w.id == "b/gw0")
    assert b_gw0.start is not None and abs(b_gw0.start - (original.start + 10)) < 1e-6
    assert abs((b_gw0.ready or 0) - b_gw0.start - boot) < 1e-6
    from pytest_timing.render.trace import trace_events

    boots = [e for e in trace_events(merged) if e["name"] == "boot"]
    assert any(
        e["ts"] == round(b_gw0.start * 1_000_000) and abs(e["dur"] - boot * 1_000_000) <= 1
        for e in boots
    )


def test_merge_keeps_every_worker_and_association(sample_run: Run) -> None:
    """Contract: every input worker stays distinct and every test keeps its worker."""
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=60, deadline=None)
    @given(
        names=st.lists(st.text(alphabet="ax/", min_size=1, max_size=4), min_size=1, max_size=4),
        shifts=st.lists(st.floats(0, 1000), min_size=4, max_size=4),
        worker_names=st.lists(
            st.sampled_from(["gw0", "gw1", "x/gw0", "a/x/gw0"]),
            min_size=2,
            max_size=2,
            unique=True,
        ),
    )
    def check(names: list[str], shifts: list[float], worker_names: list[str]) -> None:
        rename = dict(zip(["gw0", "gw1"], worker_names, strict=True))
        runs = [shifted_copy(sample_run, shifts[i], rename) for i in range(len(names))]
        merged = merge_runs(runs, names)
        expected_lanes = sum(len(r.worker_ids()) for r in runs)
        assert len(merged.workers) == expected_lanes
        assert len(set(merged.worker_ids())) == expected_lanes
        assert {t.worker for t in merged.tests} <= set(merged.worker_ids())
        assert len(merged.tests) == sum(len(r.tests) for r in runs)
        # durations survive the time shift
        for r in runs:
            for t in r.tests:
                match = [
                    m
                    for m in merged.tests
                    if m.nodeid == t.nodeid and abs(m.duration - t.duration) < 1e-6
                ]
                assert match

    check()


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "name, termination, complete",
    [
        ("legacy_complete_true", "finished", True),
        ("legacy_complete_false", "unknown", False),
        ("legacy_no_flag", "unknown", False),
    ],
)
def test_early_schema1_reports_load_render_merge_and_round_trip(
    name: str, termination: str, complete: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = FIXTURES / f"{name}.json"
    run = Run.from_json(src.read_text())
    assert run.run.termination == termination
    assert run.run.complete is complete
    assert run.run.reason is None
    # CLI rendering
    html = tmp_path / "r.html"
    assert main(["render", str(src), "--html", str(html), "--ascii", "--width", "80"]) == 0
    out = capsys.readouterr().out
    assert "pytest-timing: 2 tests" in out
    assert ("ended: " in out) is (not complete)
    assert "pytest-timing-data" in html.read_text()
    # merging with itself, shifted
    merged = merge_runs([run, run.rebased(run.run.start - 10)], ["a", "b"])
    assert len(merged.worker_ids()) == 4
    assert merged.run.complete is complete
    # saved again, the run carries an explicit termination
    saved = Run.from_json(run.to_json())
    assert saved.to_dict()["run"]["termination"] == termination
    assert saved.to_dict() == run.to_dict()


def test_invalid_termination_is_still_rejected() -> None:
    doc = json.loads((FIXTURES / "legacy_complete_true.json").read_text())
    doc["run"]["termination"] = "weird"
    with pytest.raises(ValueError, match="termination"):
        Run.from_dict(doc)
