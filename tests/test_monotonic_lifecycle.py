"""Controller clock adjustments must not distort elapsed worker lifecycles."""

from __future__ import annotations

import pytest
from conftest import load_json, needs_xdist


@needs_xdist
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("step", [-3600, 3600])
def test_controller_wall_clock_step(pytester: pytest.Pytester, workers: int, step: int) -> None:
    pytester.makeconftest(
        f"""
        from types import SimpleNamespace
        import time
        import pytest
        import pytest_timing.plugin as timing
        import pytest_timing.xdist_compat as compat

        clock = SimpleNamespace(shift=0.0, released=0)
        timing.time = SimpleNamespace(
            time=lambda: time.time() + clock.shift,
            perf_counter=time.perf_counter,
        )
        compat.time = timing.time

        @pytest.hookimpl(tryfirst=True)
        def pytest_sessionstart():
            clock.shift = {step}

        @pytest.hookimpl(trylast=True)
        def pytest_collection_finish(session):
            if not session.config.getoption('numprocesses', 0):
                clock.shift += {step}

        @pytest.hookimpl(trylast=True)
        def pytest_testnodedown(node, error):
            clock.released += 1
            if clock.released == 1:
                clock.shift += {step}
        """
    )
    pytester.makepyfile(
        """
        def test_pass(): pass
        def test_fail(): assert False
        """
    )
    result = pytester.runpytest_subprocess(
        "-n",
        str(workers),
        "--dist=worksteal",
        "--timing-capture=light",
        "--timing-json",
        "-p",
        "no:cacheprovider",
        timeout=60,
    )
    result.assert_outcomes(passed=1, failed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    run = load_json(pytester.path)
    duration = run["run"]["stop"] - run["run"]["start"]
    # The subprocess timeout bounds real elapsed time far below the injected step.
    assert 0 <= duration < abs(step)
    assert len(run["workers"]) == (workers or 1)
    if workers:
        assert run["run"]["dist"] == "worksteal"
    for worker in run["workers"]:
        assert 0 <= worker["start"] <= worker["ready"] <= worker["collected"]
        assert worker["collected"] <= worker["down"] <= duration + 1e-5
    for test in run["tests"]:
        # Pytest's wall-clock phase stamps may be coarser than the controller clock.
        assert 0 <= test["start"] <= test["stop"] <= duration + 0.1
