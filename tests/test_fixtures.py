"""Shared fixture set-up timing, recorded where the tests run and carried to the JSON."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import load_json, needs_xdist

CONFTEST = """
import time, pytest

@pytest.fixture(scope="session")
def db():
    time.sleep(0.1)
    yield

@pytest.fixture(scope="session", params=[1, 2])
def cfg(request):
    time.sleep(0.02)
    return request.param

@pytest.fixture(scope="session")
def inner():
    time.sleep(0.1)

@pytest.fixture(scope="session")
def outer(request):
    time.sleep(0.03)
    request.getfixturevalue("inner")  # nested dynamic set-up: timed on its own
"""

MODULE = """
import time, pytest

@pytest.fixture(scope="module")
def conn(db):
    time.sleep(0.05)
    yield

@pytest.fixture
def client(conn):
    yield  # function-scoped: not shared, but pulls conn and db in

def test_client(client):
    pass

def test_plain():
    pass

class TestC:
    @pytest.fixture(scope="class")
    @staticmethod
    def cfix():
        time.sleep(0.02)

    def test_class(self, cfix, db):
        pass

def test_param(cfg):
    pass

def test_nested(outer):
    pass
"""


def by_name(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["nodeid"].split("::", 1)[1]: t for t in doc["tests"]}


def fixtures_named(test: dict[str, Any], name: str) -> dict[str, float | None]:
    return {k: v for k, v in (test.get("fixtures") or {}).items() if k.endswith(f"::{name}[]")}


def shapes(test: dict[str, Any]) -> set[tuple[str, str]]:
    """(scope, name) of every shared fixture key on a test; the node ids in between
    depend on the pytest version's idea of the root conftest's location."""
    return {(k.partition(":")[0], k.rpartition("::")[2]) for k in test.get("fixtures") or {}}


def test_package_fixture_without_a_package_collector_has_session_lifetime(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        **{
            "plain/conftest": """
                import time, pytest
                @pytest.fixture(scope="package")
                def service():
                    time.sleep(.01)
                    yield
            """,
            "plain/test_first": "def test_first(service): pass",
            "test_outside": "def test_outside(): pass",
            "plain/test_last": "def test_last(service): pass",
        }
    )
    pytester.makeconftest("""
        def pytest_collection_modifyitems(items):
            order = {"test_first": 0, "test_outside": 1, "test_last": 2}
            items.sort(key=lambda item: order[item.name])
    """)
    result = pytester.runpytest_subprocess(
        "--timing-json",
        "-p",
        "no:cacheprovider",
    )
    result.assert_outcomes(passed=3)
    tests = by_name(load_json(pytester.path))
    key = "package::plain::service[]"
    first = fixtures_named(tests["test_first"], "service")
    assert set(first) == {key}
    assert first[key] is not None
    assert fixtures_named(tests["test_last"], "service") == {key: None}


@pytest.mark.parametrize("extra", [[], pytest.param(["-n", "2"], marks=needs_xdist)])
def test_shared_fixtures_are_recorded_with_scope_keys(
    pytester: pytest.Pytester, extra: list[str]
) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_fx=MODULE)
    result = pytester.runpytest_subprocess("--timing-json", "-p", "no:cacheprovider", *extra)
    result.assert_outcomes(passed=6)
    tests = by_name(load_json(pytester.path))


    assert shapes(tests["test_client"]) == {("session", "db[]"), ("module", "conn[]")}
    assert "module:test_fx.py:test_fx.py::conn[]" in tests["test_client"]["fixtures"]
    assert "fixtures" not in tests["test_plain"]
    assert shapes(tests["TestC::test_class"]) == {("session", "db[]"), ("class", "cfix[]")}
    assert (
        "class:test_fx.py::TestC:test_fx.py::TestC::cfix[]"
        in tests["TestC::test_class"]["fixtures"]
    )
    params = {
        k.rpartition("::")[2]
        for t in tests.values()
        for k in t.get("fixtures") or {}
        if "::cfg[" in k
    }
    assert params == {"cfg[0]", "cfg[1]"}

    # Every set-up was charged to exactly one test per worker, at about its real cost.
    db_setups = [v for t in tests.values() for k, v in fixtures_named(t, "db").items() if v]
    workers = {t["worker"] for t in tests.values() if fixtures_named(t, "db")}
    assert len(db_setups) == len(workers)
    assert all(0.09 < v < 0.5 for v in db_setups)
    for name in ("conn", "cfix"):
        seconds = [v for t in tests.values() for v in fixtures_named(t, name).values() if v]
        assert seconds and all(0.015 < v < 0.5 for v in seconds), name

    # The nested set-up is not double counted.
    nested = tests["test_nested"]
    (outer,) = fixtures_named(nested, "outer").values()
    (inner,) = fixtures_named(nested, "inner").values()
    assert outer is not None and 0.02 < outer < 0.08
    assert inner is not None and 0.09 < inner < 0.5


MODULES_WITH_FIXTURES = {
    f"test_mod{m}": (
        "import time, pytest\n"
        "@pytest.fixture(scope='module')\n"
        "def heavy():\n"
        "    time.sleep(0.3)\n"
        + "".join(f"def test_{i}(heavy):\n    time.sleep(0.02)\n" for i in range(4))
    )
    for m in range(2)
}


@needs_xdist
def test_scheduled_run_keeps_each_module_fixture_on_one_worker(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(**MODULES_WITH_FIXTURES)
    args = ["-n", "2", "--timing-json", "--timing-schedule", "pytest-timing.json"]
    args += ["-p", "no:cacheprovider"]
    pytester.runpytest_subprocess(*args).assert_outcomes(passed=8)
    result = pytester.runpytest_subprocess(*args)
    result.assert_outcomes(passed=8)
    result.stdout.fnmatch_lines(
        ["schedule: 8 of 8 tests had recorded durations, 2 shared fixtures in pytest-timing.json"]
    )
    doc = load_json(pytester.path)
    for m in range(2):
        module = [t for t in doc["tests"] if t["nodeid"].startswith(f"test_mod{m}.py")]
        assert len({t["worker"] for t in module}) == 1, "module split across workers"
        setups = [v for t in module for v in t["fixtures"].values() if v]
        assert len(setups) == 1, "module fixture set up more than once"


DYNAMIC_CONFTEST = """
import time, pytest

@pytest.fixture(scope="session")
def db():
    time.sleep(0.05)

@pytest.fixture(scope="session")
def api(request):
    request.getfixturevalue("db")  # only ever asked for dynamically

@pytest.fixture(scope="session", params=["a", "b"])
def backend(request):
    time.sleep(0.02)
    return request.param
"""

DYNAMIC_MODULE = """
import pytest

def test_via_fixture_1(api): pass
def test_via_fixture_2(api): pass

def test_in_body_1(request): request.getfixturevalue("db")
def test_in_body_2(request): request.getfixturevalue("db")

@pytest.mark.parametrize("backend", ["x", "y"], indirect=True)
def test_indirect(backend): pass

def test_direct(backend): pass
"""


def test_dynamic_and_indirect_fixtures_get_their_own_keys(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(DYNAMIC_CONFTEST)
    pytester.makepyfile(test_dyn=DYNAMIC_MODULE)
    result = pytester.runpytest_subprocess("--timing-json", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=8)
    tests = by_name(load_json(pytester.path))
    # db reached through getfixturevalue counts for every test that uses it, not only
    # for the one whose setup happened to create it.
    for name in ("test_via_fixture_1", "test_via_fixture_2", "test_in_body_1", "test_in_body_2"):
        assert ("session", "db[]") in shapes(tests[name]), name
    db_setups = [v for t in tests.values() for v in fixtures_named(t, "db").values() if v]
    assert len(db_setups) == 1
    # Indirect and direct parametrization each get an index per parameter.
    params = {
        name: {k.rpartition("::")[2] for k in t.get("fixtures") or {} if "::backend[" in k}
        for name, t in tests.items()
        if "backend" in name or "indirect" in name or "direct" in name
    }
    assert params["test_indirect[x]"] == {"backend[0]"}
    assert params["test_indirect[y]"] == {"backend[1]"}
    assert params["test_direct[a]"] == {"backend[0]"}
    assert params["test_direct[b]"] == {"backend[1]"}


DEPENDENT_CONFTEST = """
import pytest

@pytest.fixture(scope="session", params=["a", "b"])
def base(request):
    return request.param

@pytest.fixture(scope="session")
def derived(base):
    return base * 2

@pytest.fixture(scope="session")
def plain():
    return 1
"""

DEPENDENT_MODULE = """
def test_1(derived, plain): pass
def test_2(derived): pass
"""


def test_a_fixture_depending_on_a_parametrized_one_is_keyed_by_that_parameter(
    pytester: pytest.Pytester,
) -> None:
    # pytest tears ``derived`` down with the ``base`` instance it was built on and
    # builds it again for the next parameter: two instances, two keys.
    pytester.makeconftest(DEPENDENT_CONFTEST)
    pytester.makepyfile(test_dep=DEPENDENT_MODULE)
    result = pytester.runpytest_subprocess("--timing-json", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=4)
    tests = by_name(load_json(pytester.path))
    names = {
        name: {k.rpartition("::")[2] for k in t.get("fixtures") or {}} for name, t in tests.items()
    }
    assert names["test_1[a]"] == {"base[0]", "derived[base=0]", "plain[]"}
    assert names["test_2[a]"] == {"base[0]", "derived[base=0]"}
    assert names["test_1[b]"] == {"base[1]", "derived[base=1]", "plain[]"}
    assert names["test_2[b]"] == {"base[1]", "derived[base=1]"}
    setups = [
        k
        for t in tests.values()
        for k, v in (t.get("fixtures") or {}).items()
        if v and "derived" in k
    ]
    assert len(setups) == 2 and {k.rpartition("[")[2] for k in setups} == {"base=0]", "base=1]"}


PLUGIN_FIXTURE = """
import time
import pytest

@pytest.fixture(scope="session")
def heavy():
    time.sleep(0.05)
    return "plugin"
"""

OVERRIDE_CONFTEST = """
import pytest

@pytest.fixture(scope="session")
def heavy(heavy):
    return heavy + "+conftest"  # wraps the plugin's: both are set up
"""


def test_a_plugin_fixture_and_its_root_conftest_override_have_distinct_keys(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(myplug=PLUGIN_FIXTURE)
    pytester.makeconftest(OVERRIDE_CONFTEST)
    pytester.makepyfile(test_o="def test_o(heavy): assert heavy == 'plugin+conftest'")
    pytester.syspathinsert()
    result = pytester.runpytest_subprocess(
        "-p", "myplug", "--timing-json", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=1)
    (test,) = by_name(load_json(pytester.path)).values()
    keys = {k: v for k, v in test["fixtures"].items() if k.endswith("::heavy[]")}
    # A plugin's fixture has no directory; on pytest 7 neither has the root
    # conftest's, so the definitions are told apart by their module instead.
    defined = sorted(k.rpartition("::")[0].rpartition(":")[2] for k in keys)
    assert defined == ["conftest.heavy", "myplug.heavy"]
    plugin = next(v for k, v in keys.items() if "myplug" in k)
    assert plugin is not None and plugin >= 0.04  # the slow one kept its own time


def test_a_plugin_class_in_conftest_keeps_its_own_fixture_identity(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest("""
        import time, pytest
        class BasePlugin:
            @pytest.fixture(scope="session")
            def service(self):
                time.sleep(0.05)
                return "base"
        def pytest_configure(config):
            config.pluginmanager.register(BasePlugin(), "base-service")
        @pytest.fixture(scope="session")
        def service(service):
            return service + "-wrapped"
    """)
    pytester.makepyfile('def test_service(service): assert service == "base-wrapped"')
    result = pytester.runpytest_subprocess("--timing-json", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    (test,) = load_json(pytester.path)["tests"]
    setups = fixtures_named(test, "service")
    assert len(setups) == 2
    base = next(value for key, value in setups.items() if "BasePlugin.service" in key)
    assert base is not None and base >= 0.04
