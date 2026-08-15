"""Tests for ``shared/queue.py`` — SPEC §7.

The load-bearing tests are :class:`TestIngestIsNeverStarved` and
:class:`TestSustainedLoadSimulation`. They exist because the obvious implementation of
this module — strict priority — is wrong: under sustained interactive load ingest never
reaches the head of the queue, captions stop arriving, and M5 goes blind with no error
anywhere. The simulation runs the naive policy first and asserts the bug is real, then
runs ours and asserts it is gone.

No test sleeps. The policy takes an injected clock and the threaded tests synchronise on
events, so both the ordering and the timing are assertions rather than races.

Stdlib ``unittest``:  ``python3 -m unittest discover -s tests -t . -v``
"""

from __future__ import annotations

import logging
import threading
import unittest
from dataclasses import dataclass
from typing import Any

from shared import config
from shared.queue import (
    Job,
    JobState,
    Priority,
    PriorityPolicy,
    QueueClosed,
    QueueTimeout,
    VLMQueue,
)

ORDER = [Priority.INTERACTIVE, Priority.VERIFICATION, Priority.INGEST]
CAP = float(config.get("vlm.queue.max_ingest_pause_seconds"))

_QUIET = logging.getLogger("test.queue.quiet")
_QUIET.addHandler(logging.NullHandler())
_QUIET.propagate = False


class ManualClock:
    """A clock we move by hand. Nothing here waits on real time."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass(eq=False)
class Item:
    """Minimal thing the policy can order: a lane and a timestamp."""

    priority: Priority
    enqueued_at: float
    name: str = ""


def policy(cap: float | None = CAP, clock: ManualClock | None = None) -> PriorityPolicy[Item]:
    return PriorityPolicy(ORDER, {Priority.INGEST: cap}, clock=clock or ManualClock())


def fill(p: PriorityPolicy[Item], *specs: tuple[Priority, float, str]) -> None:
    for priority, at, name in specs:
        p.enqueue(Item(priority, at, name))


def order_of(p: PriorityPolicy[Item], now: float) -> list[str]:
    out: list[str] = []
    while (item := p.select(now)) is not None:
        out.append(item.name)
    return out


# ======================================================================================
# The ordering rule, as arithmetic
# ======================================================================================


class TestStrictPriorityWhenNothingIsOverdue(unittest.TestCase):
    def test_priority_order_comes_from_config(self) -> None:
        self.assertEqual(
            [p.value for p in ORDER],
            list(config.get("vlm.queue.priorities")),
            "SPEC §7: interactive > verification > ingest",
        )

    def test_highest_priority_lane_wins(self) -> None:
        p = policy()
        fill(
            p,
            (Priority.INGEST, 0.0, "ingest"),
            (Priority.VERIFICATION, 0.0, "verify"),
            (Priority.INTERACTIVE, 0.0, "user"),
        )
        self.assertEqual(order_of(p, now=0.0), ["user", "verify", "ingest"])

    def test_fifo_within_a_lane(self) -> None:
        p = policy()
        fill(
            p,
            (Priority.INGEST, 0.0, "a"),
            (Priority.INGEST, 1.0, "b"),
            (Priority.INGEST, 2.0, "c"),
        )
        self.assertEqual(order_of(p, now=2.0), ["a", "b", "c"])

    def test_verification_outranks_ingest_but_not_interactive(self) -> None:
        p = policy()
        fill(p, (Priority.VERIFICATION, 0.0, "verify"), (Priority.INTERACTIVE, 1.0, "user"))
        self.assertEqual(order_of(p, now=1.0), ["user", "verify"])

    def test_empty_policy_selects_nothing(self) -> None:
        self.assertIsNone(policy().select(now=99.0))

    def test_priorities_list_must_be_complete(self) -> None:
        with self.assertRaises(config.ConfigError):
            PriorityPolicy([Priority.INTERACTIVE, Priority.INGEST], {})

    def test_remove_withdraws_a_queued_item(self) -> None:
        p = policy()
        item = Item(Priority.INGEST, 0.0, "a")
        p.enqueue(item)
        self.assertTrue(p.remove(item))
        self.assertFalse(p.remove(item))
        self.assertEqual(p.depth(), 0)

    def test_depth_reports_per_lane_and_total(self) -> None:
        p = policy()
        fill(
            p,
            (Priority.INGEST, 0.0, "a"),
            (Priority.INGEST, 0.0, "b"),
            (Priority.INTERACTIVE, 0.0, "u"),
        )
        self.assertEqual(p.depth(Priority.INGEST), 2)
        self.assertEqual(p.depth(Priority.INTERACTIVE), 1)
        self.assertEqual(p.depth(), 3)


class TestIngestIsNeverStarved(unittest.TestCase):
    """SPEC §7: ingest may be paused, never starved. The pause is capped at 5 s."""

    def test_ingest_waits_while_it_is_within_its_pause_budget(self) -> None:
        p = policy(cap=CAP)
        fill(p, (Priority.INGEST, 0.0, "ingest"), (Priority.INTERACTIVE, 0.0, "user"))
        # 4.9 s of pause is allowed — that is the "may be paused" half of the rule.
        self.assertEqual(p.select(now=CAP - 0.1).name, "user")

    def test_ingest_wins_outright_once_the_pause_cap_is_exhausted(self) -> None:
        p = policy(cap=CAP)
        fill(p, (Priority.INGEST, 0.0, "ingest"), (Priority.INTERACTIVE, 0.0, "user"))
        self.assertEqual(p.select(now=CAP).name, "ingest")

    def test_a_flood_of_interactive_work_cannot_hold_ingest_off(self) -> None:
        p = policy(cap=CAP)
        p.enqueue(Item(Priority.INGEST, 0.0, "ingest"))
        for i in range(100):  # sustained interactive load, all fresher than the cap
            p.enqueue(Item(Priority.INTERACTIVE, float(i) * 0.05, f"user{i}"))
        self.assertEqual(p.select(now=CAP + 0.5).name, "ingest")

    def test_ingest_catches_up_then_yields(self) -> None:
        """Promotion is per item, not a hand-over of the whole queue."""
        p = policy(cap=CAP)
        # Three ingest chunks backed up at 4 s stride, plus interactive demand.
        fill(
            p,
            (Priority.INGEST, 0.0, "i0"),
            (Priority.INGEST, 4.0, "i1"),
            (Priority.INGEST, 8.0, "i2"),
            (Priority.INTERACTIVE, 0.0, "u0"),
            (Priority.INTERACTIVE, 0.0, "u1"),
        )
        now = 10.0  # i0 (10 s old) and i1 (6 s old) are overdue; i2 (2 s old) is not
        self.assertEqual(p.select(now).name, "i0")
        self.assertEqual(p.select(now).name, "i1")
        # Backlog drained down to a fresh head — ingest yields and interactive resumes.
        self.assertEqual(p.select(now).name, "u0")
        self.assertEqual(p.select(now).name, "u1")
        self.assertEqual(p.select(now).name, "i2")

    def test_worst_starved_lane_goes_first_when_several_are_overdue(self) -> None:
        p = PriorityPolicy[Item](
            ORDER,
            {Priority.INGEST: 5.0, Priority.VERIFICATION: 5.0},
            clock=ManualClock(),
        )
        fill(p, (Priority.VERIFICATION, 3.0, "verify"), (Priority.INGEST, 0.0, "ingest"))
        # Both are past the cap; ingest has waited longer, so ingest goes first even
        # though verification outranks it.
        self.assertEqual(p.select(now=10.0).name, "ingest")
        self.assertEqual(p.select(now=10.0).name, "verify")

    def test_lanes_without_a_cap_are_never_promoted(self) -> None:
        p = policy(cap=CAP)
        fill(p, (Priority.VERIFICATION, 0.0, "verify"), (Priority.INTERACTIVE, 0.0, "user"))
        self.assertEqual(p.select(now=10_000.0).name, "user")

    def test_overdue_reports_the_brake_state(self) -> None:
        p = policy(cap=CAP)
        p.enqueue(Item(Priority.INGEST, 0.0, "ingest"))
        self.assertEqual(p.overdue(now=CAP - 0.1), [])
        self.assertEqual(p.overdue(now=CAP), [Priority.INGEST])

    def test_policy_uses_its_injected_clock_when_now_is_omitted(self) -> None:
        clock = ManualClock()
        p = policy(cap=CAP, clock=clock)
        fill(p, (Priority.INGEST, 0.0, "ingest"), (Priority.INTERACTIVE, 0.0, "user"))
        self.assertEqual(p.select().name, "user")
        clock.advance(CAP)
        self.assertEqual(p.select().name, "ingest")


class TestSustainedLoadSimulation(unittest.TestCase):
    """The regression test for the actual bug, run in virtual time.

    Interactive requests arrive faster than the single in-flight slot can serve them —
    a user hammering the Ask pane while M4 escalations run. That is precisely the
    condition under which a strict-priority queue stops captioning.
    """

    HORIZON = 300.0
    INTERACTIVE_EVERY = 1.0  # arrivals
    INTERACTIVE_COST = 2.0  # ... at twice the rate the box can serve them
    INGEST_EVERY = 4.0  # SPEC §2.2 stride
    INGEST_COST = 2.0  # SPEC §2.4 target, ~2 s per captioned chunk

    def _simulate(self, cap: float | None) -> dict[str, Any]:
        p = policy(cap=cap)
        arrivals: list[tuple[float, Priority]] = []
        t = 0.0
        while t < self.HORIZON:
            arrivals.append((t, Priority.INTERACTIVE))
            t += self.INTERACTIVE_EVERY
        t = 0.0
        while t < self.HORIZON:
            arrivals.append((t, Priority.INGEST))
            t += self.INGEST_EVERY
        arrivals.sort(key=lambda a: (a[0], a[1].value))

        cost = {Priority.INTERACTIVE: self.INTERACTIVE_COST, Priority.INGEST: self.INGEST_COST}
        served: dict[Priority, list[float]] = {Priority.INTERACTIVE: [], Priority.INGEST: []}
        now, i = 0.0, 0
        while now < self.HORIZON:
            while i < len(arrivals) and arrivals[i][0] <= now:
                at, priority = arrivals[i]
                p.enqueue(Item(priority, at))
                i += 1
            item = p.select(now)
            if item is None:
                if i >= len(arrivals):
                    break
                now = arrivals[i][0]
                continue
            served[item.priority].append(now - item.enqueued_at)
            now += cost[item.priority]
        return {
            "served": {k: len(v) for k, v in served.items()},
            "max_wait": {k: (max(v) if v else float("inf")) for k, v in served.items()},
        }

    def test_a_naive_strict_priority_queue_starves_ingest(self) -> None:
        """The bug, demonstrated. If this ever passes with a non-zero count, the load
        model got too gentle and the test below stopped proving anything."""
        result = self._simulate(cap=None)
        self.assertEqual(
            result["served"][Priority.INGEST],
            0,
            "strict priority is expected to caption nothing under this load",
        )

    def test_the_pause_cap_keeps_ingest_flowing(self) -> None:
        result = self._simulate(cap=CAP)
        captioned = result["served"][Priority.INGEST]
        self.assertGreater(captioned, 0)
        # Ingest gets in roughly once per (cap + its own cost): ~5 min / 7 s.
        self.assertGreater(captioned, self.HORIZON / (CAP + self.INGEST_COST) * 0.8)
        # And interactive still dominates — this is a bounded pause, not round-robin.
        self.assertGreater(result["served"][Priority.INTERACTIVE], captioned)

    def test_ingest_wait_is_bounded_by_the_cap_plus_one_request(self) -> None:
        """The +1 request is irreducible: max_inflight is 1 and work is not preemptible."""
        result = self._simulate(cap=CAP)
        longest_request = max(self.INTERACTIVE_COST, self.INGEST_COST)
        self.assertLessEqual(result["max_wait"][Priority.INGEST], CAP + longest_request)


# ======================================================================================
# The queue itself
# ======================================================================================


class Recorder:
    """Records execution order and observed concurrency across worker threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.order: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    def task(self, name: str, gate: threading.Event | None = None,
             running: threading.Event | None = None) -> Any:
        def run() -> str:
            with self.lock:
                self.order.append(name)
                self.concurrent += 1
                self.max_concurrent = max(self.max_concurrent, self.concurrent)
            if running is not None:
                running.set()
            if gate is not None:
                gate.wait(timeout=5.0)
            with self.lock:
                self.concurrent -= 1
            return name

        return run


class QueueTestCase(unittest.TestCase):
    def make(self, **kwargs: Any) -> VLMQueue:
        kwargs.setdefault("logger", _QUIET)
        queue = VLMQueue(**kwargs)
        self.addCleanup(queue.stop, drain=False, timeout=5.0)
        queue.start()
        return queue


class TestQueueBasics(QueueTestCase):
    def test_defaults_come_from_config(self) -> None:
        queue = self.make()
        stats = queue.stats()
        self.assertEqual(list(stats["depth"]), list(config.get("vlm.queue.priorities")))
        self.assertEqual(stats["max_ingest_pause_seconds"], CAP)

    def test_result_returns_the_callables_value(self) -> None:
        queue = self.make()
        job = queue.submit(Priority.INGEST, lambda: "caption")
        self.assertEqual(job.result(timeout=5.0), "caption")
        self.assertIs(job.state, JobState.DONE)

    def test_priority_accepts_the_string_form(self) -> None:
        queue = self.make()
        job = queue.submit("interactive", lambda: 1)
        self.assertEqual(job.result(timeout=5.0), 1)
        self.assertIs(job.priority, Priority.INTERACTIVE)

    def test_unknown_priority_is_rejected(self) -> None:
        queue = self.make()
        with self.assertRaises(ValueError):
            queue.submit("urgent", lambda: 1)

    def test_exceptions_propagate_to_the_caller(self) -> None:
        queue = self.make()
        job: Job[Any] = queue.submit(Priority.INTERACTIVE, _boom)
        with self.assertRaises(RuntimeError):
            job.result(timeout=5.0)
        self.assertIs(job.state, JobState.FAILED)

    def test_a_failed_job_does_not_wedge_the_queue(self) -> None:
        queue = self.make()
        queue.submit(Priority.INTERACTIVE, _boom).wait(timeout=5.0)
        self.assertEqual(queue.submit(Priority.INGEST, lambda: "ok").result(timeout=5.0), "ok")

    def test_result_times_out_rather_than_hanging(self) -> None:
        queue = self.make()
        gate = threading.Event()
        self.addCleanup(gate.set)
        job = queue.submit(Priority.INGEST, lambda: gate.wait(5.0))
        with self.assertRaises(QueueTimeout):
            job.result(timeout=0.01)

    def test_submitting_after_stop_is_refused(self) -> None:
        queue = self.make()
        queue.stop(drain=False)
        with self.assertRaises(QueueClosed):
            queue.submit(Priority.INGEST, lambda: 1)

    def test_stop_cancels_queued_work(self) -> None:
        queue = self.make()
        gate, running = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        recorder = Recorder()
        queue.submit(Priority.INGEST, recorder.task("blocker", gate=gate, running=running))
        running.wait(timeout=5.0)  # the single slot is occupied; the next job stays queued
        pending = queue.submit(Priority.INGEST, recorder.task("pending"))
        queue.stop(drain=False, timeout=0.1)
        self.assertIs(pending.state, JobState.CANCELLED)
        with self.assertRaises(QueueClosed):
            pending.result(timeout=1.0)

    def test_stop_with_drain_finishes_queued_work(self) -> None:
        queue = self.make()
        jobs = [queue.submit(Priority.INGEST, lambda i=i: i) for i in range(5)]
        queue.stop(drain=True, timeout=5.0)
        self.assertEqual([j.result(timeout=5.0) for j in jobs], [0, 1, 2, 3, 4])

    def test_cancel_withdraws_a_queued_job(self) -> None:
        queue = self.make()
        gate, running = threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        recorder = Recorder()
        queue.submit(Priority.INGEST, recorder.task("blocker", gate=gate, running=running))
        running.wait(timeout=5.0)
        later = queue.submit(Priority.INGEST, recorder.task("later"))
        self.assertTrue(queue.cancel(later))
        self.assertIs(later.state, JobState.CANCELLED)
        gate.set()
        self.assertNotIn("later", recorder.order)


class TestOneRequestInFlight(QueueTestCase):
    """``vlm.queue.max_inflight`` is 1: one camera means one request at a time."""

    def test_config_says_one(self) -> None:
        self.assertEqual(config.get("vlm.queue.max_inflight"), 1)

    def test_work_never_overlaps(self) -> None:
        queue = self.make()
        recorder = Recorder()
        jobs = [queue.submit(Priority.INGEST, recorder.task(f"j{i}")) for i in range(20)]
        for job in jobs:
            job.result(timeout=5.0)
        self.assertEqual(recorder.max_concurrent, 1)
        self.assertEqual(queue.stats()["inflight"], 0)


class TestQueueOrdering(QueueTestCase):
    """End-to-end ordering, with the worker held busy so submission order is fixed."""

    def _run(self, clock: ManualClock, advance_to: float) -> list[str]:
        queue = self.make(
            max_inflight=1,
            max_ingest_pause_seconds=CAP,
            clock=clock,
        )
        recorder = Recorder()
        gate, running = threading.Event(), threading.Event()
        self.addCleanup(gate.set)

        blocker = queue.submit(
            Priority.INTERACTIVE, recorder.task("blocker", gate=gate, running=running)
        )
        running.wait(timeout=5.0)  # the single slot is now occupied

        jobs = [
            queue.submit(Priority.INGEST, recorder.task("ingest"), label="ingest"),
            queue.submit(Priority.INTERACTIVE, recorder.task("user1")),
            queue.submit(Priority.INTERACTIVE, recorder.task("user2")),
        ]
        clock.t = advance_to  # time passes while the slot is busy
        gate.set()
        blocker.result(timeout=5.0)
        for job in jobs:
            job.result(timeout=5.0)
        self.stats = queue.stats()
        return recorder.order

    def test_interactive_jumps_the_queue_while_ingest_is_within_its_budget(self) -> None:
        order = self._run(ManualClock(0.0), advance_to=CAP - 0.1)
        self.assertEqual(order, ["blocker", "user1", "user2", "ingest"])
        self.assertEqual(self.stats["ingest_promotions"], 0)

    def test_ingest_is_promoted_once_it_has_waited_out_the_cap(self) -> None:
        order = self._run(ManualClock(0.0), advance_to=CAP + 1.0)
        self.assertEqual(
            order,
            ["blocker", "ingest", "user1", "user2"],
            "ingest was paused for its full budget and then jumped ahead",
        )
        self.assertEqual(self.stats["ingest_promotions"], 1)
        self.assertGreaterEqual(self.stats["max_wait_s"]["ingest"], CAP)


class TestQueueConfigValidation(unittest.TestCase):
    def test_max_inflight_must_be_at_least_one(self) -> None:
        with self.assertRaises(config.ConfigError):
            VLMQueue(max_inflight=0, logger=_QUIET)

    def test_incomplete_priority_list_is_rejected(self) -> None:
        with self.assertRaises(config.ConfigError):
            VLMQueue(priorities=[Priority.INTERACTIVE], logger=_QUIET)


def _boom() -> None:
    raise RuntimeError("the VLM fell over")


if __name__ == "__main__":
    unittest.main()
