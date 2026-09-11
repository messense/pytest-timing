"""The CPU admission model on its own: reservations, the waiting line, the governor."""

from __future__ import annotations

import math

import pytest

from pytest_timing.admission import Admission, PressureSample


def test_reservations_are_atomic_and_never_exceed_the_limit() -> None:
    adm = Admission(4)
    assert adm.reserve("a", 3, until=10.0)
    assert adm.used == 3 and adm.free == 1
    assert not adm.reserve("b", 2, until=10.0)  # would need 5
    assert adm.held("b") == 0  # nothing partial
    assert adm.reserve("b", 1, until=10.0)
    assert adm.free == 0
    # Raising a reservation only needs the difference.
    assert not adm.reserve("a", 4, until=10.0)
    adm.assign("b", 0, until=10.0, busy=False)
    assert adm.reserve("a", 4, until=10.0)
    assert adm.used == 4 and adm.forced == 0


def test_lowering_is_always_allowed_and_release_forgets() -> None:
    adm = Admission(2)
    assert adm.reserve("a", 2, until=1.0)
    adm.assign("a", 1, until=1.0, busy=True)
    assert adm.free == 1
    adm.release("a")
    assert adm.reserved == {} and adm.free == 2


def test_oversized_requests_are_clamped_to_the_limit_and_run_alone() -> None:
    adm = Admission(4)
    assert adm.reserve("a", 1, until=5.0)
    assert not adm.reserve("big", 9, until=9.0)  # counts as 4: needs everyone gone
    adm.assign("a", 0, until=5.0, busy=False)
    assert adm.reserve("big", 9, until=9.0)
    assert adm.held("big") == 4 and adm.clamped == 2


def test_the_head_of_the_line_is_pledged_to_and_others_only_get_the_rest() -> None:
    adm = Admission(4)
    for worker in "abcd":
        assert adm.reserve(worker, 1, until=10.0)
    adm.wait("big", 3, since=0.0)
    # a and b finish; the head is short by 3, and there are only 2 free.
    adm.assign("a", 0, until=10.0, busy=False)
    adm.assign("b", 0, until=10.0, busy=False)
    assert adm.free == 2
    assert not adm.reserve("big", 3, until=20.0)
    assert not adm.reserve("a", 1, until=11.0, finish=11.0)  # pledged, and a finishes late
    adm.assign("c", 0, until=10.0, busy=False)
    assert adm.free == 3
    assert not adm.reserve("a", 1, until=11.0, finish=11.0)  # still: 3 free, 3 pledged
    assert adm.reserve("big", 3, until=20.0)
    assert "big" not in adm.waiting and adm.used == 4


def test_backfilling_lets_short_work_use_pledged_slots() -> None:
    adm = Admission(4)
    assert adm.reserve("a", 2, until=10.0)  # releases at 10
    assert adm.reserve("b", 2, until=30.0)  # releases at 30
    adm.wait("big", 3, since=0.0)
    adm.assign("a", 0, until=10.0, busy=False)  # a is done: 2 free, 3 pledged
    # The head gets its 3rd slot when b releases at 30: work done by then may go.
    assert adm.fits("a", 2, finish=25.0)
    assert not adm.fits("a", 2, finish=31.0)
    assert adm.reserve("a", 1, until=25.0, finish=25.0)
    # Nothing is pledged past the head itself.
    adm.wait("late", 1, since=5.0)
    assert adm.fits("big", 3, finish=50.0) is False  # only 1 free now
    adm.assign("a", 0, until=25.0, busy=False)
    adm.assign("b", 0, until=30.0, busy=False)
    assert adm.reserve("big", 3, until=50.0)


def test_shadow_time_is_when_enough_reservations_release() -> None:
    adm = Admission(4)
    adm.reserve("a", 1, until=10.0)
    adm.reserve("b", 1, until=20.0)
    adm.reserve("c", 2, until=30.0)
    adm.wait("big", 4, since=0.0)
    assert adm.shadow("big", 4, excluding=None) == 30.0
    assert adm.shadow("big", 2, excluding=None) == 20.0
    assert adm.shadow("big", 2, excluding="a") == 30.0  # a's release does not count
    assert adm.shadow("big", 1, excluding="a") == 20.0
    adm.wait("x", 1, since=1.0)
    assert adm.shadow("big", 9, excluding=None) == math.inf


def test_admission_never_exceeds_the_limit_and_a_forced_assignment_is_counted() -> None:
    adm = Admission(4)
    for worker in "abc":
        adm.assign(worker, 1, until=0.0, busy=False)  # fixtures kept alive, nothing running
    adm.wait("big", 4, since=0.0)
    assert adm.idle and not adm.fits("big", 4)
    assert not adm.reserve("big", 4, until=9.0)  # admission itself never goes over
    # Whoever decides that nothing else can move sets it, and that is counted.
    adm.assign("big", 4, until=9.0, busy=True)
    assert adm.forced == 1 and adm.used == 7 and not adm.idle


def test_waiters_keep_their_place_and_are_ordered_by_arrival() -> None:
    adm = Admission(1)
    adm.wait("late", 1, since=5.0)
    adm.wait("early", 1, since=1.0)
    adm.wait("late", 2, since=99.0)  # updates the request, keeps the place
    assert sorted(adm.waiting, key=lambda w: adm.waiting[w].since) == ["early", "late"]
    assert adm.waiting["late"].since == 5.0 and adm.waiting["late"].slots == 2
    assert adm.head() == "early"
    adm.admitted("early")
    assert adm.head() == "late"


def test_limit_moves_within_one_and_the_budget_and_is_counted() -> None:
    adm = Admission(3)
    adm.set_limit(0)
    assert adm.limit == 1 and adm.lowered == 1 and adm.lowest == 1
    adm.set_limit(9)
    assert adm.limit == 3 and adm.lowered == 1
    assert adm.summary()["lowest"] == 1


def samples(
    n: int, *, some: float | None, rate: float = 0.0, start: float = 0.0, throttled: bool = False
) -> list[PressureSample]:
    out = []
    usec = 0
    for i in range(n):
        if throttled:
            usec += 1000
        out.append(
            PressureSample(start + i, some=some, throttled=usec if throttled else None, rate=rate)
        )
    return out


def test_without_reliable_signals_the_limit_never_moves() -> None:
    adm = Admission(8)
    for sample in samples(50, some=None):
        assert adm.observe(sample) == 0
    assert adm.limit == 8


def test_persistent_external_pressure_lowers_the_limit_with_hysteresis_and_recovery() -> None:
    adm = Admission(8)
    # Two contended samples are not enough; the third moves the limit.
    assert [adm.observe(s) for s in samples(3, some=0.6)] == [0, 0, -1]
    assert adm.limit == 7
    # Within the cooldown nothing moves however contended it stays.
    assert all(adm.observe(s) == 0 for s in samples(9, some=0.6, start=3.0))
    assert adm.observe(PressureSample(13.0, some=0.6)) == -1
    assert adm.limit == 6
    # A clean stretch recovers one slot at a time, after the cooldown.
    changes = [adm.observe(s) for s in samples(30, some=0.0, start=14.0)]
    assert changes.count(1) >= 2 and adm.limit == 8
    assert adm.summary()["lowered"] == 2 and adm.summary()["lowest"] == 6


def test_pressure_from_our_own_full_utilisation_is_not_held_against_us() -> None:
    adm = Admission(8)
    # High stall share, but the run's own measured rate accounts for the load.
    for sample in samples(10, some=0.9, rate=7.5):
        assert adm.observe(sample) == 0
    assert adm.limit == 8
    # Same pressure with the run mostly idle: something else is eating the CPUs.
    assert [adm.observe(s) for s in samples(3, some=0.9, rate=1.0, start=20.0)] == [0, 0, -1]


def test_quota_throttling_counts_as_contention_from_the_second_sample() -> None:
    adm = Admission(4)
    seen = [adm.observe(s) for s in samples(5, some=None, throttled=True)]
    assert seen == [0, 0, 0, -1, 0]  # the first sample only sets the baseline
    assert adm.limit == 3
    # A frozen counter is a clean sample.
    frozen = [PressureSample(10.0 + i, some=None, throttled=5000) for i in range(20)]
    assert 1 in [adm.observe(s) for s in frozen]


def test_the_limit_never_drops_below_one() -> None:
    adm = Admission(1)
    assert all(adm.observe(s) == 0 for s in samples(10, some=0.9))
    assert adm.limit == 1


@pytest.mark.parametrize("some", [0.24, 0.25])
def test_the_pressure_threshold_is_inclusive(some: float) -> None:
    adm = Admission(4)
    changes = [adm.observe(s) for s in samples(3, some=some)]
    assert changes == [0, 0, -1 if some >= 0.25 else 0]
