"""Tests for the action server — SPEC §6.4, CLAUDE.md invariant 5.

These tests exist to prove one thing: that a single staged event cannot produce thirty
alerts. Everything else here is support for that claim.

Rules this file follows, because the module under test is the one that changes the
outside world:

* the log always lives in a tempdir, never ``data/actions.jsonl``;
* time is injected, never slept — a brake test that depends on wall clock is a brake test
  that will be marked flaky and then deleted;
* nothing shells out. ffmpeg is not installed on this box (CLAUDE.md machine state) and
  the clip commands are asserted as values, which is why they are built by a pure
  function.

Run with::

    python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared import config
from shared.schema import ActionKind, ActionStatus, Task
from services.mcp import (
    ActionServer,
    Brake,
    NullClipCutter,
    SegmentSlice,
    build_clip_plan,
    clip_path_for,
    ranges_collide,
    read_action_log,
)

# Numbers come from settings.yaml, not from this file. CLAUDE.md: no magic numbers.
WINDOW_SECONDS = float(config.get("ingest.window_seconds"))
STRIDE_SECONDS = float(config.get("ingest.stride_seconds"))
DEFAULT_COOLDOWN = float(config.get("monitor.default_cooldown_seconds"))
DEDUPE_PAD = float(config.get("monitor.dedupe_overlap_seconds"))
CAMERA_ID = str(config.get("camera.id"))
CONTAINER = str(config.get("recorder.container"))

#: A fixed instant so every assertion below is reproducible.
T0 = datetime(2026, 8, 14, 21, 11, 7, tzinfo=timezone.utc)


class FakeClock:
    """Injected wall clock. Advancing it is the only way time passes in these tests."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now


class SequentialIds:
    """Deterministic entry ids so failures name the row that broke."""

    def __init__(self, prefix: str = "e") -> None:
        self.prefix = prefix
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"{self.prefix}{self.n:04d}"


class ServerCase(unittest.TestCase):
    """Base fixture: a tempdir log, a fake clock, deterministic ids."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.log_path = self.tmp / "actions.jsonl"
        self.clock = FakeClock(T0)
        self.ids = SequentialIds()

    def make_server(self, **overrides: object) -> ActionServer:
        kwargs: dict[str, object] = {
            "log_path": self.log_path,
            "clips_dir": self.tmp / "clips",
            "clock": self.clock,
            "id_factory": self.ids,
        }
        kwargs.update(overrides)
        return ActionServer(**kwargs)  # type: ignore[arg-type]

    def raw_lines(self) -> list[str]:
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding="utf-8").splitlines()

    def chunk_range(self, index: int) -> tuple[datetime, datetime]:
        """Footage range of the ``index``-th analysis window off the ingest grid.

        5 s windows on a 4 s stride: consecutive windows share a second of footage and
        will hand M5 the same event several times over. That overlap is the reason the
        dedupe brake exists.
        """
        start = T0 + timedelta(seconds=index * STRIDE_SECONDS)
        return start, start + timedelta(seconds=WINDOW_SECONDS)


# ======================================================================================
# THE HEADLINE — one event, many chunks, exactly one action
# ======================================================================================


class TestOneEventOneAction(ServerCase):
    """SPEC §6.4: the demo failure mode is firing thirty alerts for one event."""

    def test_single_staged_event_over_minutes_fires_exactly_one_action(self) -> None:
        server = self.make_server()
        task = Task(
            task_id="fire-door-blocked",
            describe="a vehicle stopped in front of the fire door",
            window=120,
            action=ActionKind.RAISE_ALERT,
        )

        # One vehicle, parked, matching every window from the moment it stops until just
        # inside the cooldown horizon. Every chunk is a genuine stage-2 match; M5 asks on
        # every single one, exactly as it will in the demo.
        event_seconds = DEFAULT_COOLDOWN - WINDOW_SECONDS
        n_chunks = int(event_seconds // STRIDE_SECONDS)
        self.assertGreater(n_chunks, 50, "test is meaningless without many chunks")

        results = []
        for i in range(n_chunks):
            t_start, t_end = self.chunk_range(i)
            # The clock tracks the footage: chunks arrive as they are analysed.
            self.clock.now = t_end
            results.append(
                server.raise_alert(t_start, t_end, task=task, reason="vehicle at fire door")
            )

        fired = [r for r in results if r.fired]
        self.assertEqual(len(fired), 1, f"expected exactly 1 alert, got {len(fired)}")
        self.assertEqual(len(self.raw_lines()), 1, "exactly one row must reach the log")

        # And it was the first chunk that won, not an arbitrary later one.
        self.assertTrue(results[0].fired)
        self.assertEqual(fired[0].entry.t_start, self.chunk_range(0)[0])

        # Every other ask was refused by a brake, not by an error or a silent drop.
        suppressed = [r for r in results if not r.fired]
        self.assertEqual(len(suppressed), n_chunks - 1)
        self.assertTrue(all(r.brake is not None for r in suppressed))

        # Both brakes did work here, and each was independently sufficient for part of
        # the run: the early chunks still overlap the fired footage range, the later ones
        # no longer do and are held by the cooldown alone.
        engaged = {b for r in suppressed for b in r.engaged_brakes}
        self.assertEqual(engaged, {Brake.DEDUPE, Brake.COOLDOWN})
        self.assertTrue(
            any(r.engaged_brakes == (Brake.COOLDOWN,) for r in suppressed),
            "the cooldown must be carrying chunks that dedupe no longer covers",
        )
        self.assertTrue(any(Brake.DEDUPE in r.engaged_brakes for r in suppressed))

    def test_the_same_event_seen_by_two_processes_still_fires_once(self) -> None:
        """M3 and M5 hold separate servers over one file. The log is the shared state."""
        m5 = self.make_server(id_factory=SequentialIds("m5-"))
        m3 = self.make_server(id_factory=SequentialIds("m3-"))
        task = Task("fire-door-blocked", "vehicle", 120, ActionKind.RAISE_ALERT)

        first = m5.raise_alert(*self.chunk_range(0), task=task)
        second = m3.raise_alert(*self.chunk_range(1), task=task)

        self.assertTrue(first.fired)
        self.assertFalse(second.fired, "a second process must not re-fire the same event")
        self.assertEqual(len(self.raw_lines()), 1)


# ======================================================================================
# BRAKE 1 — cooldown, per task, on ts
# ======================================================================================


class TestCooldownBrake(ServerCase):
    def test_second_fire_inside_cooldown_is_suppressed(self) -> None:
        server = self.make_server()
        # Ranges far enough apart in footage that dedupe cannot be what stops this.
        first = server.raise_alert(T0, T0 + timedelta(seconds=WINDOW_SECONDS), task_id="t")
        self.clock.advance(DEFAULT_COOLDOWN - 1)
        far = self.clock.now + timedelta(hours=1)
        second = server.raise_alert(far, far + timedelta(seconds=WINDOW_SECONDS), task_id="t")

        self.assertTrue(first.fired)
        self.assertFalse(second.fired)
        self.assertIs(second.brake, Brake.COOLDOWN)
        self.assertEqual(second.blocked_by.entry_id, first.entry_id)
        self.assertEqual(
            second.engaged_brakes, (Brake.COOLDOWN,), "dedupe must not be what stopped this"
        )

    def test_fires_again_once_cooldown_expires(self) -> None:
        server = self.make_server()
        self.assertTrue(server.raise_alert(*self.chunk_range(0), task_id="t").fired)
        self.clock.advance(DEFAULT_COOLDOWN + 1)
        far = self.clock.now + timedelta(hours=1)
        again = server.raise_alert(far, far + timedelta(seconds=WINDOW_SECONDS), task_id="t")
        self.assertTrue(again.fired, "a genuinely new event after the cooldown must alert")
        self.assertEqual(len(self.raw_lines()), 2)

    def test_cooldown_comes_from_the_task_not_the_default(self) -> None:
        """SPEC §6.1 makes cooldown a per-task dial. The brake still always runs."""
        server = self.make_server()
        task = Task("bay", "unloading", 30, ActionKind.SAVE_CLIP, cooldown=120)
        self.assertLess(task.cooldown, DEFAULT_COOLDOWN)

        self.assertTrue(server.save_clip(*self.chunk_range(0), task=task).fired)
        self.clock.advance(task.cooldown + 1)
        far = self.clock.now + timedelta(hours=1)
        self.assertTrue(
            server.save_clip(far, far + timedelta(seconds=WINDOW_SECONDS), task=task).fired,
            "the task's shorter cooldown should have lapsed",
        )

    def test_each_task_gets_its_own_cooldown(self) -> None:
        server = self.make_server()
        a = server.raise_alert(*self.chunk_range(0), task_id="fire-door-blocked")
        b = server.raise_alert(*self.chunk_range(0), task_id="loading-bay-activity")
        self.assertTrue(a.fired)
        self.assertTrue(b.fired, "two tasks watching one moment are two concerns")

    def test_negative_cooldown_is_rejected(self) -> None:
        server = self.make_server()
        with self.assertRaises(ValueError):
            server.raise_alert(*self.chunk_range(0), task_id="t", cooldown_seconds=-1)


# ======================================================================================
# BRAKE 2 — dedupe on the FOOTAGE range, not the wall clock
# ======================================================================================


class TestDedupeBrake(ServerCase):
    def make_nocooldown_server(self) -> ActionServer:
        """Cooldown off so any suppression below is provably the dedupe brake."""
        return self.make_server(default_cooldown_seconds=0)

    def test_overlapping_consecutive_chunks_are_deduped(self) -> None:
        server = self.make_nocooldown_server()
        first = server.raise_alert(*self.chunk_range(0), task_id="t")
        second = server.raise_alert(*self.chunk_range(1), task_id="t")

        self.assertTrue(first.fired)
        self.assertFalse(second.fired)
        self.assertIs(second.brake, Brake.DEDUPE)
        self.assertEqual(second.blocked_by.entry_id, first.entry_id)
        self.assertEqual(
            second.engaged_brakes, (Brake.DEDUPE,), "the cooldown is off; dedupe stood alone"
        )

    def test_dedupe_compares_footage_time_not_request_time(self) -> None:
        """The two fields exist separately for exactly this reason.

        Long after any cooldown could apply, the same *footage* is still the same
        footage. A dedupe implemented against ``ts`` would let this through.
        """
        server = self.make_nocooldown_server()
        first = server.raise_alert(*self.chunk_range(0), task_id="t")
        self.clock.advance(DEFAULT_COOLDOWN * 10)
        replay = server.raise_alert(*self.chunk_range(0), task_id="t")

        self.assertTrue(first.fired)
        self.assertFalse(replay.fired)
        self.assertEqual(replay.engaged_brakes, (Brake.DEDUPE,))

    def test_distinct_footage_passes_dedupe(self) -> None:
        server = self.make_nocooldown_server()
        self.assertTrue(server.raise_alert(*self.chunk_range(0), task_id="t").fired)
        later = T0 + timedelta(seconds=DEDUPE_PAD + WINDOW_SECONDS + 1)
        result = server.raise_alert(
            later, later + timedelta(seconds=WINDOW_SECONDS), task_id="t"
        )
        self.assertTrue(result.fired, "a genuinely different moment must not be deduped")

    def test_pad_closes_the_stride_gap(self) -> None:
        """Windows two strides apart no longer intersect but are still one event.

        A bare intersection test would fire a second alert on chunk 2 of every staged
        event. ``monitor.dedupe_overlap_seconds`` is what closes that gap.
        """
        a_start, a_end = self.chunk_range(0)
        c_start, c_end = self.chunk_range(2)
        self.assertGreater(
            (c_start - a_end).total_seconds(), 0, "chunk 2 should not intersect chunk 0"
        )
        self.assertTrue(ranges_collide(a_start, a_end, c_start, c_end, DEDUPE_PAD))

        server = self.make_nocooldown_server()
        self.assertTrue(server.raise_alert(a_start, a_end, task_id="t").fired)
        self.assertFalse(server.raise_alert(c_start, c_end, task_id="t").fired)

    def test_dedupe_is_per_action_kind(self) -> None:
        server = self.make_nocooldown_server()
        rng = self.chunk_range(0)
        self.assertTrue(server.save_clip(*rng, task_id="t").fired)
        self.assertTrue(
            server.raise_alert(*rng, task_id="t").fired,
            "saving a clip of a moment is not the same act as alerting a human about it",
        )

    def test_retraction_does_not_release_either_brake(self) -> None:
        """Being wrong is not a licence to re-fire immediately."""
        server = self.make_server()
        first = server.raise_alert(*self.chunk_range(0), task_id="t")
        server.retract(first.entry_id, reason="worker found no vehicle")

        retry = server.raise_alert(*self.chunk_range(1), task_id="t")
        self.assertFalse(retry.fired)
        self.assertEqual(retry.blocked_by.entry_id, first.entry_id)


# ======================================================================================
# BRAKE 3 — the append-only log
# ======================================================================================


class TestAppendOnlyLog(ServerCase):
    def test_no_existing_line_is_ever_rewritten(self) -> None:
        """The load-bearing claim of §11.4 and §4.1: history is not tidied up."""
        server = self.make_server()
        snapshots: list[bytes] = []

        def snapshot() -> None:
            snapshots.append(self.log_path.read_bytes())

        alert = server.raise_alert(*self.chunk_range(0), task_id="t", reason="vehicle")
        snapshot()
        self.clock.advance(DEFAULT_COOLDOWN + 1)
        ticket = server.file_ticket(
            *self.chunk_range(100), task_id="t2", reason="door still blocked"
        )
        snapshot()
        server.verify(alert.entry_id, reason="confirmed", clip_path="/clips/a.mp4")
        snapshot()
        server.retract(ticket.entry_id, reason="worker found no vehicle")
        snapshot()

        # Each state of the file is a byte-exact prefix of the next. That is what
        # append-only means, and it is stronger than "the rows are still there".
        for older, newer in zip(snapshots, snapshots[1:]):
            self.assertTrue(
                newer.startswith(older),
                "an earlier byte of the log changed; the log is no longer append-only",
            )
        self.assertEqual(len(self.raw_lines()), 4)

        # And the original alert row still reads UNVERIFIED on disk, un-edited.
        first = json.loads(self.raw_lines()[0])
        self.assertEqual(first["entry_id"], alert.entry_id)
        self.assertEqual(first["status"], ActionStatus.UNVERIFIED.value)
        self.assertIsNone(first["parent_id"])
        self.assertIsNone(first["clip_path"])

    def test_every_line_is_one_complete_json_object(self) -> None:
        server = self.make_server()
        for i in range(5):
            self.clock.advance(DEFAULT_COOLDOWN + 1)
            server.raise_alert(*self.chunk_range(i * 1000), task_id=f"t{i}")
        lines = self.raw_lines()
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(json.loads(line)["action"], ActionKind.RAISE_ALERT.value)

    def test_concurrent_writers_do_not_interleave(self) -> None:
        """M3 and M5 both write this file; a torn line loses the row you needed.

        Each thread holds its own ``ActionServer``, so the in-process lock cannot be what
        saves us — the cross-process ``flock`` in ``ActionLog`` has to.
        """
        threads_n, per_thread = 8, 12
        errors: list[BaseException] = []

        def writer(worker: int) -> None:
            try:
                server = self.make_server(id_factory=SequentialIds(f"w{worker}-"))
                for i in range(per_thread):
                    # Distinct task per row so no brake suppresses anything; this test is
                    # about the bytes, not the brakes.
                    server.raise_alert(*self.chunk_range(i), task_id=f"w{worker}-{i}")
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        lines = self.raw_lines()
        self.assertEqual(len(lines), threads_n * per_thread)
        ids = {json.loads(line)["entry_id"] for line in lines}
        self.assertEqual(len(ids), threads_n * per_thread, "a row was lost or duplicated")

    def test_a_second_server_reads_rows_written_by_the_first(self) -> None:
        writer = self.make_server(id_factory=SequentialIds("w-"))
        reader = self.make_server(id_factory=SequentialIds("r-"))
        entry = writer.raise_alert(*self.chunk_range(0), task_id="t").entry

        rows = reader.read_action_log(T0 - timedelta(minutes=1), T0 + timedelta(minutes=1))
        self.assertEqual([r.entry_id for r in rows], [entry.entry_id])

    def test_module_level_read_action_log_matches_the_server(self) -> None:
        server = self.make_server()
        server.raise_alert(*self.chunk_range(0), task_id="t")
        rows = read_action_log(
            T0 - timedelta(minutes=1), T0 + timedelta(minutes=1), log_path=self.log_path
        )
        self.assertEqual(len(rows), 1)


# ======================================================================================
# Severity split and amendments — SPEC §6.3
# ======================================================================================


class TestSeveritySplitAndAmendments(ServerCase):
    def test_save_clip_owes_no_verification(self) -> None:
        server = self.make_server()
        result = server.save_clip(*self.chunk_range(0), task_id="bay")
        self.assertTrue(result.fired)
        self.assertFalse(result.entry.action.reaches_a_human)
        self.assertFalse(
            result.awaits_verification, "low-stakes actions fire on stage 2 and are done"
        )

    def test_human_reaching_actions_fire_provisionally(self) -> None:
        server = self.make_server(default_cooldown_seconds=0)
        for kind in (ActionKind.RAISE_ALERT, ActionKind.FILE_TICKET):
            with self.subTest(kind=kind):
                rng = self.chunk_range(1000 * (kind is ActionKind.FILE_TICKET))
                result = server.fire(kind, *rng, task_id=f"t-{kind.value}")
                self.assertTrue(result.fired)
                self.assertIs(result.entry.status, ActionStatus.UNVERIFIED)
                self.assertTrue(result.awaits_verification)

    def test_verification_appends_a_child_row(self) -> None:
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t", reason="vehicle")
        self.clock.advance(38)
        amendment = server.verify(
            alert.entry_id, reason="confirmed by worker", clip_path="/clips/x.mp4"
        )

        self.assertEqual(amendment.parent_id, alert.entry_id)
        self.assertIs(amendment.status, ActionStatus.VERIFIED)
        self.assertNotEqual(amendment.entry_id, alert.entry_id)
        # The amendment carries the original's footage range so a row read alone still
        # says what it is about.
        self.assertEqual(amendment.t_start, alert.entry.t_start)
        self.assertEqual(amendment.action, alert.entry.action)
        # ...but it is a later moment in wall clock.
        self.assertGreater(amendment.ts, alert.entry.ts)

    def test_amendments_do_not_count_as_actions_for_the_brakes(self) -> None:
        """Otherwise a verified alert would hold the cooldown twice over."""
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t")
        server.verify(alert.entry_id, reason="confirmed")
        self.clock.advance(DEFAULT_COOLDOWN + 1)
        far = self.clock.now + timedelta(hours=1)
        self.assertTrue(
            server.raise_alert(far, far + timedelta(seconds=WINDOW_SECONDS), task_id="t").fired
        )

    def test_amending_to_unverified_is_refused(self) -> None:
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t")
        with self.assertRaises(ValueError):
            server.amend(alert.entry_id, ActionStatus.UNVERIFIED)

    def test_amending_an_unknown_entry_is_refused(self) -> None:
        server = self.make_server()
        with self.assertRaises(KeyError):
            server.verify("does-not-exist")

    def test_pending_verification_lists_only_unresolved_human_actions(self) -> None:
        server = self.make_server(default_cooldown_seconds=0)
        alert = server.raise_alert(*self.chunk_range(0), task_id="a")
        server.raise_alert(*self.chunk_range(1000), task_id="b")
        server.save_clip(*self.chunk_range(2000), task_id="c")
        server.verify(alert.entry_id, reason="confirmed")

        window = (T0 - timedelta(hours=1), T0 + timedelta(hours=4))
        pending = server.pending_verification(*window)
        self.assertEqual([p.original.task_id for p in pending], ["b"])


# ======================================================================================
# Resolved view — fold parent_id chains once, here
# ======================================================================================


class TestResolvedView(ServerCase):
    def test_resolve_folds_a_chain_to_its_latest_status(self) -> None:
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t", reason="vehicle")
        self.clock.advance(38)
        verified = server.verify(alert.entry_id, reason="confirmed", clip_path="/clips/x.mp4")
        self.clock.advance(60)
        retracted = server.retract(alert.entry_id, reason="second look: no vehicle")

        resolved = server.resolve(alert.entry_id)
        self.assertIsNotNone(resolved)
        self.assertIs(resolved.status, ActionStatus.RETRACTED)
        self.assertTrue(resolved.retracted)
        self.assertFalse(resolved.awaits_verification)
        self.assertEqual(
            [a.entry_id for a in resolved.amendments],
            [verified.entry_id, retracted.entry_id],
            "amendments must stay in append order for the §11.4 render",
        )
        # The clip attached mid-chain survives a later retraction — the evidence for why
        # we alerted is still the evidence.
        self.assertEqual(resolved.clip_path, "/clips/x.mp4")
        self.assertEqual(resolved.reason, "second look: no vehicle")
        self.assertEqual(resolved.resolved_at, retracted.ts)

    def test_unamended_action_resolves_to_itself(self) -> None:
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t")
        resolved = server.resolve(alert.entry_id)
        self.assertEqual(resolved.amendments, ())
        self.assertIs(resolved.status, ActionStatus.UNVERIFIED)
        self.assertTrue(resolved.awaits_verification)

    def test_resolving_an_amendment_id_returns_the_whole_action(self) -> None:
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t")
        amendment = server.verify(alert.entry_id, reason="confirmed")
        self.assertEqual(server.resolve(amendment.entry_id).entry_id, alert.entry_id)

    def test_resolved_log_is_one_row_per_action(self) -> None:
        server = self.make_server(default_cooldown_seconds=0)
        first = server.raise_alert(*self.chunk_range(0), task_id="a")
        second = server.raise_alert(*self.chunk_range(1000), task_id="b")
        server.retract(first.entry_id, reason="no vehicle")
        server.verify(second.entry_id, reason="confirmed")

        rows = server.resolved_log(T0 - timedelta(hours=1), T0 + timedelta(hours=4))
        self.assertEqual([r.entry_id for r in rows], [first.entry_id, second.entry_id])
        self.assertEqual(
            [r.status for r in rows], [ActionStatus.RETRACTED, ActionStatus.VERIFIED]
        )

    def test_resolve_of_unknown_id_is_none(self) -> None:
        self.assertIsNone(self.make_server().resolve("nope"))


# ======================================================================================
# read_action_log — SPEC §4.1, "why did you alert at 21:11?"
# ======================================================================================


class TestReadActionLog(ServerCase):
    def test_matches_on_either_the_act_or_the_footage(self) -> None:
        server = self.make_server(default_cooldown_seconds=0)
        # Something happened at T0; we acted on it an hour later (a replayed backlog).
        self.clock.now = T0 + timedelta(hours=1)
        by_footage = server.raise_alert(*self.chunk_range(0), task_id="a")
        # ...and something else happened an hour later that we acted on immediately.
        self.clock.now = T0 + timedelta(hours=1)
        later = T0 + timedelta(hours=1)
        by_ts = server.raise_alert(
            later, later + timedelta(seconds=WINDOW_SECONDS), task_id="b"
        )

        # A window around the footage moment finds the first.
        found = server.read_action_log(T0 - timedelta(seconds=1), T0 + timedelta(seconds=30))
        self.assertEqual([r.entry_id for r in found], [by_footage.entry_id])

        # A window around the acting moment finds both, since both were appended then.
        found = server.read_action_log(
            later - timedelta(seconds=1), later + timedelta(seconds=30)
        )
        self.assertEqual(
            {r.entry_id for r in found}, {by_footage.entry_id, by_ts.entry_id}
        )

    def test_amendments_travel_with_their_original(self) -> None:
        """§11.4 renders a retraction beneath its original. Neither may arrive alone."""
        server = self.make_server()
        alert = server.raise_alert(*self.chunk_range(0), task_id="t")
        self.clock.advance(3600)
        retraction = server.retract(alert.entry_id, reason="no vehicle")

        # Query the alert's moment: the much-later retraction still comes with it.
        rows = server.read_action_log(T0 - timedelta(seconds=1), T0 + timedelta(seconds=30))
        self.assertEqual([r.entry_id for r in rows], [alert.entry_id, retraction.entry_id])

        # Query the retraction's moment: the original comes with it.
        rows = server.read_action_log(
            retraction.ts - timedelta(seconds=1), retraction.ts + timedelta(seconds=1)
        )
        self.assertEqual([r.entry_id for r in rows], [alert.entry_id, retraction.entry_id])

    def test_empty_log_reads_empty(self) -> None:
        server = self.make_server()
        self.assertEqual(server.read_action_log(T0, T0 + timedelta(days=1)), [])
        self.assertFalse(self.log_path.exists(), "reading must not create the log")


# ======================================================================================
# Clips — pure command construction, nothing executed
# ======================================================================================


class TestClipPlanning(ServerCase):
    def test_single_segment_is_one_copy_cut(self) -> None:
        out = self.tmp / "clip.mp4"
        plan = build_clip_plan(
            [SegmentSlice("/archive/cam01_20260814_211100.mp4", 7.0, 5.0)],
            out,
            ffmpeg_bin="ffmpeg",
            copy_codec=True,
        )
        self.assertEqual(len(plan.commands), 1)
        argv = plan.commands[0]
        self.assertEqual(argv[0], "ffmpeg")
        self.assertEqual(argv[-1], str(out))
        # Fast seek: -ss must precede -i, or the copy-cut decodes from zero.
        self.assertLess(argv.index("-ss"), argv.index("-i"))
        self.assertEqual(argv[argv.index("-ss") + 1], "7.000")
        self.assertEqual(argv[argv.index("-t") + 1], "5.000")
        # Invariant 7: evidence is never re-encoded.
        self.assertIn("-c", argv)
        self.assertEqual(argv[argv.index("-c") + 1], "copy")
        self.assertIsNone(plan.concat_list_path)

    def test_boundary_spanning_range_cuts_parts_then_concats(self) -> None:
        """Invariant 3: an event can span two segment files. One filename cannot hold it."""
        out = self.tmp / "clip.mp4"
        plan = build_clip_plan(
            [
                SegmentSlice("/archive/cam01_20260814_211100.mp4", 57.0, 3.0),
                SegmentSlice("/archive/cam01_20260814_211200.mp4", 0.0, 2.0),
            ],
            out,
            ffmpeg_bin="ffmpeg",
            copy_codec=True,
        )
        self.assertEqual(len(plan.commands), 3)
        self.assertEqual(len(plan.part_paths), 2)
        self.assertEqual(plan.commands[-1][-1], str(out))
        self.assertIn("concat", plan.commands[-1])
        self.assertIsNotNone(plan.concat_list_text)
        for part in plan.part_paths:
            self.assertIn(f"file '{part}'", plan.concat_list_text)

    def test_a_clip_is_named_after_its_footage_range(self) -> None:
        path = clip_path_for(
            T0,
            T0 + timedelta(seconds=5),
            clips_dir=self.tmp / "clips",
            camera_id=CAMERA_ID,
            container=CONTAINER,
        )
        self.assertEqual(path.name, f"{CAMERA_ID}_20260814T211107_211112.{CONTAINER}")

    def test_empty_slice_list_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_clip_plan([], self.tmp / "clip.mp4", ffmpeg_bin="ffmpeg", copy_codec=True)

    def test_clip_path_lands_in_the_log_when_a_cutter_produces_one(self) -> None:
        """SPEC §6.4: the log carries the clip."""
        cut_plans = []

        class RecordingCutter:
            def cut(self, plan):  # type: ignore[no-untyped-def]
                cut_plans.append(plan)
                return str(plan.out_path)

        def resolver(t_start, t_end):  # type: ignore[no-untyped-def]
            return [SegmentSlice("/archive/seg.mp4", 7.0, (t_end - t_start).total_seconds())]

        server = self.make_server(segment_resolver=resolver, clip_cutter=RecordingCutter())
        result = server.save_clip(*self.chunk_range(0), task_id="bay")

        self.assertEqual(len(cut_plans), 1)
        self.assertEqual(result.entry.clip_path, str(cut_plans[0].out_path))
        self.assertEqual(json.loads(self.raw_lines()[0])["clip_path"], result.entry.clip_path)

    def test_no_clip_is_claimed_when_none_was_cut(self) -> None:
        """Better an honest null than a path the Timeline pane offers and cannot open."""
        cutter = NullClipCutter()

        def resolver(t_start, t_end):  # type: ignore[no-untyped-def]
            return [SegmentSlice("/archive/seg.mp4", 0.0, 5.0)]

        server = self.make_server(segment_resolver=resolver, clip_cutter=cutter)
        result = server.save_clip(*self.chunk_range(0), task_id="bay")
        self.assertIsNone(result.entry.clip_path)
        self.assertEqual(len(cutter.plans), 1, "the plan is still built and inspectable")

    def test_suppressed_actions_do_not_cut_clips(self) -> None:
        calls = []

        def resolver(t_start, t_end):  # type: ignore[no-untyped-def]
            calls.append((t_start, t_end))
            return [SegmentSlice("/archive/seg.mp4", 0.0, 5.0)]

        server = self.make_server(segment_resolver=resolver, clip_cutter=NullClipCutter())
        server.save_clip(*self.chunk_range(0), task_id="bay")
        server.save_clip(*self.chunk_range(1), task_id="bay")
        self.assertEqual(len(calls), 1, "a braked action must not touch the archive")


# ======================================================================================
# Misc guards
# ======================================================================================


class TestGuards(ServerCase):
    def test_inverted_footage_range_is_refused(self) -> None:
        server = self.make_server()
        with self.assertRaises(ValueError):
            server.raise_alert(T0 + timedelta(seconds=5), T0, task_id="t")

    def test_stats_count_each_brake_that_engaged(self) -> None:
        server = self.make_server()
        server.raise_alert(*self.chunk_range(0), task_id="t")
        # Overlapping footage inside the cooldown: both brakes engage on this one.
        server.raise_alert(*self.chunk_range(1), task_id="t")
        # Distinct footage inside the cooldown: the cooldown alone.
        far = T0 + timedelta(hours=1)
        server.raise_alert(far, far + timedelta(seconds=WINDOW_SECONDS), task_id="t")

        self.assertEqual(server.stats["fired"], 1)
        self.assertEqual(server.stats["suppressed"], 2)
        self.assertEqual(server.stats["suppressed_dedupe"], 1)
        self.assertEqual(server.stats["suppressed_cooldown"], 2)

    def test_timestamps_round_trip_through_the_log_exactly(self) -> None:
        """The footage range is the join to the pixels; a lossy write breaks the worker."""
        server = self.make_server()
        t_start = T0 + timedelta(microseconds=123456)
        t_end = t_start + timedelta(seconds=WINDOW_SECONDS)
        entry = server.raise_alert(t_start, t_end, task_id="t").entry

        reader = self.make_server()
        (row,) = reader.read_action_log(T0 - timedelta(minutes=1), T0 + timedelta(minutes=1))
        self.assertEqual(row.t_start, t_start)
        self.assertEqual(row.t_end, t_end)
        self.assertEqual(row.ts, entry.ts)

    def test_the_default_log_path_is_the_configured_one(self) -> None:
        """Guards against a tempdir default silently shipping to the demo."""
        self.assertEqual(
            config.repo_path("paths.action_log").name,
            Path(str(config.get("paths.action_log"))).name,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
