"""Tests for ``services/retention.py`` — the one operation that destroys evidence.

Everything else in this system appends, so everything else can be re-run after a bad
test. This cannot, which is why the assertions below are mostly about what the sweep
*refuses* to touch: the segment the recorder still has open, footage on the newer side of
the cutoff, and anything at all when the requested age is inside the live window.

The archive is a tempdir of small placeholder files. Like ``tests/test_timecode.py``,
nothing here decodes video — the sweep reads names, sizes and mtimes.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services import retention
from shared import config, timecode as tc
from shared.schema import ChunkRecord

CAM = "cam01"
UTC = timezone.utc
NOW = datetime(2026, 8, 15, 21, 0, 0, tzinfo=UTC)


class FakeIndex:
    """The two methods :mod:`services.retention` needs, over a dict."""

    def __init__(self, records: list[ChunkRecord] | None = None) -> None:
        self.records = {r.chunk_id: r for r in (records or [])}
        self.deleted_calls: list[list[str]] = []

    def select_before(self, cutoff: datetime) -> list[str]:
        return sorted(r.chunk_id for r in self.records.values() if r.t_end <= cutoff)

    def delete(self, chunk_ids: list[str]) -> int:
        self.deleted_calls.append(list(chunk_ids))
        removed = 0
        for chunk_id in chunk_ids:
            if self.records.pop(chunk_id, None) is not None:
                removed += 1
        return removed


def chunk(t_start: datetime, seconds: float = 5.0) -> ChunkRecord:
    t_end = t_start + timedelta(seconds=seconds)
    return ChunkRecord(
        chunk_id=f"{CAM}-{int(t_start.timestamp())}",
        camera_id=CAM,
        t_start=t_start,
        t_end=t_end,
        segment=tc.segment_name_for(CAM, t_start.replace(second=0, microsecond=0)),
        pts_offset=float(t_start.second),
        caption="a person walks past the door",
    )


class RetentionCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.archive = Path(self._tmp.name)
        self.segment_seconds = float(config.get("recorder.segment_seconds"))
        self.settings = retention.RetentionSettings(
            max_age_seconds=10800.0,
            min_age_seconds=900.0,
            live_guard_seconds=self.segment_seconds * 2,
            archive_dir=self.archive,
            camera_id=CAM,
        )

    def write_segment(self, start: datetime, *, size: int = 1024, mtime: datetime | None = None) -> Path:
        """One placeholder segment. ``mtime`` defaults to the moment it stops covering."""
        path = self.archive / tc.segment_name_for(CAM, start)
        path.write_bytes(b"\0" * size)
        stamp = (mtime or start + timedelta(seconds=self.segment_seconds)).timestamp()
        os.utime(path, (stamp, stamp))
        return path

    def hours_ago(self, hours: float) -> datetime:
        return NOW - timedelta(hours=hours)

    def plan(self, index: FakeIndex, **kwargs):
        kwargs.setdefault("now", NOW)
        kwargs.setdefault("settings", self.settings)
        return retention.plan_retention(index, **kwargs)


# --------------------------------------------------------------------------------------
# The age itself
# --------------------------------------------------------------------------------------


class TestAgeIsBounded(RetentionCase):
    def test_none_means_the_configured_default(self) -> None:
        self.assertEqual(retention.resolve_age(None, self.settings), 10800.0)

    def test_an_age_inside_the_live_window_is_refused(self) -> None:
        """CLAUDE.md invariant 4 and SPEC §2 both live in the last few minutes: the
        analysis window, ingest's backlog and any in-flight deep job. A sweep reaching
        into them deletes the footage the next question is about."""
        with self.assertRaises(ValueError) as caught:
            retention.resolve_age(30.0, self.settings)
        self.assertIn("min_age_seconds", str(caught.exception))

    def test_the_floor_is_not_silently_clamped(self) -> None:
        """Clamping would delete something adjacent to what was asked for, which is how a
        guard turns into a story about the tool deleting the wrong thing."""
        with self.assertRaises(ValueError):
            retention.resolve_age(self.settings.min_age_seconds - 1, self.settings)
        # Exactly at the floor is allowed — it is a floor, not an exclusive bound.
        self.assertEqual(
            retention.resolve_age(self.settings.min_age_seconds, self.settings),
            self.settings.min_age_seconds,
        )

    def test_nonsense_ages_are_refused(self) -> None:
        for bad in (float("nan"), float("inf")):
            with self.subTest(age=bad), self.assertRaises(ValueError):
                retention.resolve_age(bad, self.settings)


# --------------------------------------------------------------------------------------
# What the plan selects
# --------------------------------------------------------------------------------------


class TestPlanSelectsOnlyTheOldSide(RetentionCase):
    def test_a_segment_ending_after_the_cutoff_survives_whole(self) -> None:
        """The archive cannot delete half a file, so a straddling segment is kept —
        the newer half of it is footage the operator asked to keep."""
        old = self.write_segment(self.hours_ago(5))
        straddling = self.write_segment(self.hours_ago(3) - timedelta(seconds=30))
        self.write_segment(self.hours_ago(0.1))

        plan = self.plan(FakeIndex())
        names = [s.path.name for s in plan.segments]
        self.assertIn(old.name, names)
        self.assertNotIn(straddling.name, names)

    def test_the_newest_segment_is_never_deleted(self) -> None:
        """It is the one ffmpeg has open. Unlinking it takes the mp4 moov atom with it,
        and every analysis window overlapping the result fails to decode (CLAUDE.md)."""
        self.write_segment(self.hours_ago(9))
        newest = self.write_segment(self.hours_ago(8))  # still older than the cutoff

        plan = self.plan(FakeIndex())
        self.assertNotIn(newest.name, [s.path.name for s in plan.segments])
        self.assertEqual(plan.kept_live_segments, [newest.name])

    def test_a_recently_written_segment_is_kept_even_if_its_name_is_old(self) -> None:
        """A name is a claim about when recording started; mtime is evidence about now.
        A recorder replaying an old file writes old-looking names this second."""
        stale_name_fresh_file = self.write_segment(self.hours_ago(9), mtime=NOW)
        self.write_segment(self.hours_ago(4))
        self.write_segment(self.hours_ago(0.1))

        plan = self.plan(FakeIndex())
        self.assertNotIn(stale_name_fresh_file.name, [s.path.name for s in plan.segments])

    def test_bytes_are_counted_from_the_files_themselves(self) -> None:
        self.write_segment(self.hours_ago(6), size=4096)
        self.write_segment(self.hours_ago(5), size=2048)
        self.write_segment(self.hours_ago(0.1), size=999)

        self.assertEqual(self.plan(FakeIndex()).bytes_to_free, 6144)

    def test_captions_follow_the_same_cutoff_as_the_footage(self) -> None:
        old = chunk(self.hours_ago(6))
        recent = chunk(self.hours_ago(0.5))
        plan = self.plan(FakeIndex([old, recent]))
        self.assertEqual(plan.chunk_ids, [old.chunk_id])

    def test_a_caption_straddling_the_cutoff_survives(self) -> None:
        """``t_end <= cutoff``, deliberately narrower than the overlap rule retrieval
        uses: search widens at a boundary, deletion narrows at it."""
        straddler = chunk(self.hours_ago(3) - timedelta(seconds=2), seconds=5)
        self.assertEqual(self.plan(FakeIndex([straddler])).chunk_ids, [])

    def test_a_missing_archive_is_reported_not_treated_as_clean(self) -> None:
        settings = retention.RetentionSettings(
            max_age_seconds=10800.0,
            min_age_seconds=900.0,
            live_guard_seconds=120.0,
            archive_dir=self.archive / "nope",
            camera_id=CAM,
        )
        old = chunk(self.hours_ago(6))
        plan = self.plan(FakeIndex([old]), settings=settings)
        self.assertTrue(plan.archive_missing)
        self.assertEqual(plan.segments, [])
        self.assertEqual(plan.chunk_ids, [old.chunk_id])  # rows still go

    def test_an_empty_plan_says_so(self) -> None:
        self.write_segment(self.hours_ago(0.2))
        plan = self.plan(FakeIndex([chunk(self.hours_ago(0.1))]))
        self.assertTrue(plan.is_empty)

    def test_planning_deletes_nothing(self) -> None:
        path = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        index = FakeIndex([chunk(self.hours_ago(6))])

        self.plan(index)

        self.assertTrue(path.is_file())
        self.assertEqual(index.deleted_calls, [])
        self.assertEqual(len(index.records), 1)


# --------------------------------------------------------------------------------------
# Applying it
# --------------------------------------------------------------------------------------


class TestApply(RetentionCase):
    def test_files_and_rows_both_go(self) -> None:
        doomed = self.write_segment(self.hours_ago(6), size=2048)
        kept = self.write_segment(self.hours_ago(0.1), size=512)
        old_chunk = chunk(self.hours_ago(6))
        index = FakeIndex([old_chunk, chunk(self.hours_ago(0.1))])

        result = retention.apply_retention(index, self.plan(index))

        self.assertFalse(doomed.exists())
        self.assertTrue(kept.is_file())
        self.assertEqual(result.segments_deleted, 1)
        self.assertEqual(result.bytes_freed, 2048)
        self.assertEqual(result.chunks_deleted, 1)
        self.assertNotIn(old_chunk.chunk_id, index.records)
        self.assertEqual(result.errors, [])

    def test_index_rows_go_before_the_files(self) -> None:
        """Both orders can be interrupted; only this one is safe when it is. A caption
        outliving its footage is cited, answered from, and escalated on — and the deep
        worker then finds nothing to re-watch. Orphaned footage is merely invisible."""
        self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        order: list[str] = []

        index = FakeIndex([chunk(self.hours_ago(6))])
        plan = self.plan(index)

        real_delete = index.delete

        def watched_delete(ids: list[str]) -> int:
            order.append("index")
            return real_delete(ids)

        index.delete = watched_delete  # type: ignore[method-assign]
        original_unlink = Path.unlink

        def watched_unlink(self_path: Path, *args, **kwargs):  # noqa: ANN002, ANN003
            order.append("file")
            return original_unlink(self_path, *args, **kwargs)

        Path.unlink = watched_unlink  # type: ignore[method-assign]
        self.addCleanup(setattr, Path, "unlink", original_unlink)

        retention.apply_retention(index, plan)
        self.assertEqual(order, ["index", "file"])

    def test_a_file_that_vanished_is_not_an_error(self) -> None:
        """The button can be clicked twice, and a plan can outlive the disk it read."""
        doomed = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        index = FakeIndex()
        plan = self.plan(index)
        doomed.unlink()

        result = retention.apply_retention(index, plan)
        self.assertEqual(result.segments_deleted, 0)
        self.assertEqual(result.errors, [])

    def test_one_unremovable_file_does_not_abort_the_sweep(self) -> None:
        """Half a swept archive is a state the system already handles — timecode reports
        holes. Half a swept index is not: aborting after the rows are gone would leave
        footage nothing can find."""
        first = self.write_segment(self.hours_ago(7), size=100)
        second = self.write_segment(self.hours_ago(6), size=100)
        self.write_segment(self.hours_ago(0.1))
        index = FakeIndex()
        plan = self.plan(index)

        original_unlink = Path.unlink

        def flaky_unlink(self_path: Path, *args, **kwargs):  # noqa: ANN002, ANN003
            if self_path.name == first.name:
                raise PermissionError(13, "Permission denied")
            return original_unlink(self_path, *args, **kwargs)

        Path.unlink = flaky_unlink  # type: ignore[method-assign]
        self.addCleanup(setattr, Path, "unlink", original_unlink)

        result = retention.apply_retention(index, plan)
        self.assertEqual(result.segments_deleted, 1)
        self.assertFalse(second.exists())
        self.assertEqual(len(result.errors), 1)
        self.assertIn(first.name, result.errors[0])


# --------------------------------------------------------------------------------------
# The routes
#
# Only ``index`` and ``jobs`` are exercised, so the rest of AgentApp is left unbuilt
# rather than mocked into existence — a fake ask agent here would assert nothing about
# these two routes and would rot the moment M3's constructor changes. `retention` is
# INJECTED: without it these tests would plan against the real data/archive.
# --------------------------------------------------------------------------------------


class FakeJobs:
    def __init__(self, jobs: dict | None = None) -> None:
        self._jobs = jobs or {}

    def jobs(self) -> dict:
        return self._jobs


class TestRoutes(RetentionCase):
    def build(self, index: FakeIndex, jobs: dict | None = None):
        from services.agent.server import AgentApp

        return AgentApp(
            agent=None,  # type: ignore[arg-type]
            index=index,  # type: ignore[arg-type]
            actions=None,  # type: ignore[arg-type]
            jobs=FakeJobs(jobs),  # type: ignore[arg-type]
            chat_log=None,  # type: ignore[arg-type]
            tasks=None,  # type: ignore[arg-type]
            hub=None,  # type: ignore[arg-type]
            settings=None,  # type: ignore[arg-type]
            clip_cutter=None,  # type: ignore[arg-type]
            retention=self.settings,
        )

    def test_get_plans_without_deleting(self) -> None:
        path = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        app = self.build(FakeIndex([chunk(self.hours_ago(6))]))

        status, payload = app.get_retention()

        self.assertEqual(int(status), 200)
        self.assertEqual(payload["segment_count"], 1)
        self.assertEqual(payload["chunk_count"], 1)
        self.assertEqual(payload["defaults"]["min_age_seconds"], 900.0)
        self.assertTrue(path.is_file())

    def test_get_rejects_an_age_inside_the_live_window(self) -> None:
        app = self.build(FakeIndex())
        status, payload = app.get_retention(older_than_seconds=10.0)
        self.assertEqual(int(status), 400)
        self.assertIn("min_age_seconds", payload["detail"])

    def test_post_without_confirm_deletes_nothing(self) -> None:
        """A route that deletes an afternoon of footage must not be reachable by a stray
        POST, a retried request or a page that reloaded oddly."""
        path = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        app = self.build(FakeIndex())

        status, payload = app.post_retention({})

        self.assertEqual(int(status), 400)
        self.assertIn("confirm", payload["detail"])
        self.assertTrue(path.is_file())

    def test_post_with_confirm_deletes(self) -> None:
        path = self.write_segment(self.hours_ago(6), size=64)
        self.write_segment(self.hours_ago(0.1))
        old = chunk(self.hours_ago(6))
        index = FakeIndex([old])
        app = self.build(index)

        status, payload = app.post_retention({"confirm": True})

        self.assertEqual(int(status), 200)
        self.assertEqual(payload["segments_deleted"], 1)
        self.assertEqual(payload["chunks_deleted"], 1)
        self.assertFalse(path.exists())

    def test_a_non_numeric_age_is_rejected_rather_than_defaulted(self) -> None:
        """The read-only planner falls back to the configured age on garbage; the route
        that deletes does not — silently sweeping three hours because a field was
        misspelled is the failure this whole module is shaped around."""
        path = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        app = self.build(FakeIndex())

        status, _ = app.post_retention({"confirm": True, "older_than_seconds": "yesterday"})

        self.assertEqual(int(status), 400)
        self.assertTrue(path.is_file())

    def test_an_in_flight_deep_job_over_the_range_blocks_the_sweep(self) -> None:
        """M4 re-reads the archive by time range. Deleting the footage under a running
        job turns a 90 s escalation into a decode error, on stage, with nothing on screen
        to distinguish the two."""
        from shared.schema import DeepJob, JobState

        path = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        job = DeepJob(
            job_id="job-1",
            t_start=self.hours_ago(6),
            t_end=self.hours_ago(6) + timedelta(seconds=30),
            question="what happened at the door?",
            state=JobState.RUNNING,
            requested_at=NOW,
        )
        app = self.build(FakeIndex(), jobs={job.job_id: job})

        status, payload = app.post_retention({"confirm": True})

        self.assertEqual(int(status), 409)
        self.assertEqual(payload["jobs"], ["job-1"])
        self.assertTrue(path.is_file())

    def test_a_finished_job_over_the_range_does_not_block(self) -> None:
        from shared.schema import DeepJob, JobState

        path = self.write_segment(self.hours_ago(6))
        self.write_segment(self.hours_ago(0.1))
        job = DeepJob(
            job_id="job-1",
            t_start=self.hours_ago(6),
            t_end=self.hours_ago(6) + timedelta(seconds=30),
            question="what happened at the door?",
            state=JobState.DONE,
            requested_at=NOW,
            completed_at=NOW,
        )
        app = self.build(FakeIndex(), jobs={job.job_id: job})

        status, _ = app.post_retention({"confirm": True})

        self.assertEqual(int(status), 200)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
