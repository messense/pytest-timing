"""Small demo suite for trying pytest-timing. Run it with:

    uv run pytest examples -n 3 --timing --timing-html
"""

import time

import pytest


@pytest.fixture(scope="module")
def database():
    time.sleep(0.3)  # expensive module fixture: shows up as setup on the first test per worker
    yield


@pytest.fixture
def client(database):
    time.sleep(0.05)
    yield
    time.sleep(0.02)


@pytest.mark.parametrize("i", range(12))
def test_api(client, i):
    time.sleep(0.02 * (i % 4 + 1))


@pytest.mark.parametrize("i", range(3))
def test_slow(i):
    time.sleep(0.2 + 0.1 * i)


def test_fails():
    time.sleep(0.05)
    assert 1 == 2


@pytest.mark.skip(reason="demo")
def test_skipped():
    pass
