"""Tests for ``shared/timecode.py``.

CLAUDE.md names four cases this module has to survive — boundary-spanning ranges, segment
gaps, clock drift, DST-adjacent local conversions — because each of them fails *silently*
in production: a stitch that quietly drops the second file, a hole reported as a shorter
range, a drifted segment whose seek lands 1.5 s off, a timeline that jumps an hour once a
year. Round-tripping is here too, since every other test rests on it.

No real video is needed. The archive is a tempdir of empty placeholder files: this module
only ever reads *names*, and pretending otherwise would make the tests slow and the
fixtures a liability.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from shared import config, timecode as tc

CAM = "cam01"
UTC = timezone.utc


def utc(y: int, mo: int, d: int, h: int, mi: int, s: int = 0, us: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, s, us, tzinfo=UTC)


def has_zone(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError):
        return False
    return True


class ArchiveFixture(unittest.TestCase):
    """Builds a fake archive from a list of segment start times."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.archive = Path(self._tmp.name)
        self.nominal = float(config.get("recorder.segment_seconds"))
        # Read the same way the module reads it, so retuning the setting retunes the
        # tests rather than breaking them. Filenames carry whole seconds only, so the
        # drift fixtures below work in integral steps.
        self.drift = float(
            config.get("recorder.max_drift_seconds", tc._FALLBACK_MAX_DRIFT_SECONDS)
        )

    def write_segments(self, *starts: datetime) -> list[str]:
        names = []
        for start in starts:
            name = tc.segment_name_for(CAM, start)
            (self.archive / name).write_bytes(b"")
            names.append(name)
        return names

    def resolve(self, t0: datetime, t1: datetime) -> list[tc.SegmentSpan]:
        return tc.resolve_range(t0, t1, archive_dir=self.archive)

    def assert_tiles(self, spans: list[tc.SegmentSpan], t0: datetime, t1: datetime) -> None:
        """The contract every caller relies on: spans cover [t0, t1) exactly, in order."""
        self.assertTrue(spans, "resolve_range must never return an empty list")
        self.assertEqual(spans[0].t_start, t0)
        self.assertEqual(spans[-1].t_end, t1)
        for a, b in zip(spans, spans[1:]):
            self.assertEqual(a.t_end, b.t_start, "spans must be contiguous, no implicit holes")
            self.assertLess(a.t_start, a.t_end, "spans must be non-empty")


# --------------------------------------------------------------------------------------
# Filename <-> start time
# --------------------------------------------------------------------------------------


class TestNameRoundTrip(unittest.TestCase):
    def test_spec_example_parses(self) -> None:
        # SPEC §2.1 spells this filename out; if it ever stops parsing, the spec moved.
        self.assertEqual(
            tc.segment_start_from_name("cam01_20260814_211100.mp4"),
            utc(2026, 8, 14, 21, 11, 0),
        )

    def test_round_trip_over_a_day(self) -> None:
        start = utc(2026, 8, 14, 0, 0, 0)
        for i in range(0, 1440, 7):  # every 7th minute of a day, incl. midnight rollover
            dt = start + timedelta(minutes=i)
            self.assertEqual(tc.segment_start_from_name(tc.segment_name_for(CAM, dt)), dt)

    def test_round_trip_across_year_boundary(self) -> None:
        dt = utc(2026, 12, 31, 23, 59, 0)
        self.assertEqual(tc.segment_start_from_name(tc.segment_name_for(CAM, dt)), dt)

    def test_leap_day_is_not_invented(self) -> None:
        dt = utc(2028, 2, 29, 12, 0, 0)
        self.assertEqual(tc.segment_start_from_name(tc.segment_name_for(CAM, dt)), dt)

    def test_full_path_accepted(self) -> None:
        self.assertEqual(
            tc.segment_start_from_name("/data/archive/cam01_20260814_211100.mp4"),
            utc(2026, 8, 14, 21, 11, 0),
        )

    def test_returns_aware_utc(self) -> None:
        self.assertEqual(tc.segment_start_from_name("cam01_20260814_211100.mp4").tzinfo, UTC)

    def test_non_matching_name_raises(self) -> None:
        for bad in ("notes.txt", "cam01_2026814_2111.mp4", "cam01_20260814_211100.mkv", ""):
            with self.assertRaises(tc.TimecodeError):
                tc.segment_start_from_name(bad)

    def test_camera_id_with_underscores_survives(self) -> None:
        # `.+?` for the camera must still let the trailing digit runs bind correctly.
        dt = utc(2026, 8, 14, 21, 11, 0)
        name = tc.segment_name_for("dock_cam_02", dt)
        self.assertEqual(tc.segment_start_from_name(name), dt)

    def test_naive_datetime_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tc.segment_name_for(CAM, datetime(2026, 8, 14, 21, 11, 0))

    def test_non_utc_input_is_normalised(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        self.assertEqual(
            tc.segment_name_for(CAM, utc(2026, 8, 14, 21, 11, 0).astimezone(ist)),
            "cam01_20260814_211100.mp4",
        )


class TestPtsOffset(unittest.TestCase):
    def test_spec_example(self) -> None:
        # SPEC §3.1: t_start 21:11:07 in cam01_..._211100.mp4 -> pts_offset 7.00
        self.assertEqual(
            tc.pts_offset_for(utc(2026, 8, 14, 21, 11, 7), "cam01_20260814_211100.mp4"), 7.0
        )

    def test_sub_second_precision_survives(self) -> None:
        offset = tc.pts_offset_for(
            utc(2026, 8, 14, 21, 11, 7, 250_000), "cam01_20260814_211100.mp4"
        )
        self.assertAlmostEqual(offset, 7.25, places=6)

    def test_instant_before_segment_raises(self) -> None:
        with self.assertRaises(tc.TimecodeError):
            tc.pts_offset_for(utc(2026, 8, 14, 21, 10, 59), "cam01_20260814_211100.mp4")

    def test_instant_far_past_segment_raises(self) -> None:
        # 21:13:00 is two files later; a seek to pts=120 would land somewhere plausible
        # and wrong, which is exactly what this guard exists to prevent.
        with self.assertRaises(tc.TimecodeError):
            tc.pts_offset_for(utc(2026, 8, 14, 21, 13, 0), "cam01_20260814_211100.mp4")

    def test_naive_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tc.pts_offset_for(datetime(2026, 8, 14, 21, 11, 7), "cam01_20260814_211100.mp4")


# --------------------------------------------------------------------------------------
# Boundary-spanning ranges — CLAUDE.md invariant 3 / SPEC §3.1
# --------------------------------------------------------------------------------------


class TestBoundarySpanning(ArchiveFixture):
    def test_event_spanning_two_files(self) -> None:
        # SPEC §3.1's own example: 21:11:58 running 12 s.
        self.write_segments(
            utc(2026, 8, 14, 21, 10, 0), utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 12, 0)
        )
        t0, t1 = utc(2026, 8, 14, 21, 11, 58), utc(2026, 8, 14, 21, 12, 10)
        spans = self.resolve(t0, t1)

        self.assertEqual(len(spans), 2)
        self.assert_tiles(spans, t0, t1)
        self.assertFalse(any(s.is_gap for s in spans))

        first, second = spans
        self.assertEqual(first.segment, "cam01_20260814_211100.mp4")
        self.assertAlmostEqual(first.pts_in, 58.0)
        self.assertAlmostEqual(first.pts_out, 60.0)
        self.assertEqual(second.segment, "cam01_20260814_211200.mp4")
        self.assertAlmostEqual(second.pts_in, 0.0)
        self.assertAlmostEqual(second.pts_out, 10.0)
        self.assertAlmostEqual(tc.covered_seconds(spans), 12.0)

    def test_range_inside_one_file(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 12, 0))
        t0, t1 = utc(2026, 8, 14, 21, 11, 7), utc(2026, 8, 14, 21, 11, 12)
        spans = self.resolve(t0, t1)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].segment, "cam01_20260814_211100.mp4")
        self.assertAlmostEqual(spans[0].pts_in, 7.0)
        self.assertAlmostEqual(spans[0].pts_out, 12.0)

    def test_range_spanning_three_files(self) -> None:
        starts = [utc(2026, 8, 14, 21, 10 + i, 0) for i in range(4)]
        self.write_segments(*starts)
        t0, t1 = utc(2026, 8, 14, 21, 10, 30), utc(2026, 8, 14, 21, 12, 30)
        spans = self.resolve(t0, t1)
        expected = [tc.segment_name_for(CAM, s) for s in starts[:3]]
        self.assertEqual([s.segment for s in spans], expected)
        self.assert_tiles(spans, t0, t1)
        self.assertAlmostEqual(tc.covered_seconds(spans), 120.0)

    def test_range_exactly_on_a_boundary_does_not_open_an_empty_file(self) -> None:
        # [21:11:00, 21:12:00) is entirely the 211100 file. An off-by-one here would open
        # 211200 and ask it for pts 60..60 — a zero-frame read that looks like "no event".
        self.write_segments(
            utc(2026, 8, 14, 21, 10, 0), utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 12, 0)
        )
        spans = self.resolve(utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 12, 0))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].segment, "cam01_20260814_211100.mp4")
        self.assertAlmostEqual(spans[0].pts_in, 0.0)
        self.assertAlmostEqual(spans[0].pts_out, 60.0)

    def test_pts_and_wall_clock_agree(self) -> None:
        # The whole point of the record: both coordinates name the same moment.
        self.write_segments(utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 12, 0))
        for span in self.resolve(utc(2026, 8, 14, 21, 11, 58), utc(2026, 8, 14, 21, 12, 10)):
            self.assertEqual(span.segment_start, tc.segment_start_from_name(span.segment))
            self.assertAlmostEqual(span.pts_in, tc.pts_offset_for(span.t_start, span.segment))
            self.assertAlmostEqual(span.pts_out - span.pts_in, span.duration)

    def test_inverted_and_empty_ranges_rejected(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        t = utc(2026, 8, 14, 21, 11, 10)
        with self.assertRaises(ValueError):
            self.resolve(t, t)
        with self.assertRaises(ValueError):
            self.resolve(t, t - timedelta(seconds=5))

    def test_naive_range_rejected(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        with self.assertRaises(ValueError):
            self.resolve(datetime(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 11, 5))

    def test_non_segment_files_ignored(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        (self.archive / "cam01_20260814_211100.mp4.tmp").write_bytes(b"")
        (self.archive / "README").write_bytes(b"")
        (self.archive / "nested").mkdir()
        spans = self.resolve(utc(2026, 8, 14, 21, 11, 5), utc(2026, 8, 14, 21, 11, 10))
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].is_gap)

    def test_mixed_cameras_raise_unless_disambiguated(self) -> None:
        start = utc(2026, 8, 14, 21, 11, 0)
        (self.archive / tc.segment_name_for(CAM, start)).write_bytes(b"")
        (self.archive / tc.segment_name_for("cam02", start)).write_bytes(b"")
        with self.assertRaises(tc.TimecodeError):
            self.resolve(start, start + timedelta(seconds=5))
        spans = tc.resolve_range(
            start, start + timedelta(seconds=5), archive_dir=self.archive, camera_id=CAM
        )
        self.assertEqual(spans[0].segment, tc.segment_name_for(CAM, start))

    def test_missing_archive_directory_raises(self) -> None:
        with self.assertRaises(tc.TimecodeError):
            tc.resolve_range(
                utc(2026, 8, 14, 21, 11, 0),
                utc(2026, 8, 14, 21, 11, 5),
                archive_dir=self.archive / "does-not-exist",
            )


# --------------------------------------------------------------------------------------
# Gaps — a hole must be reported, never silently shortened
# --------------------------------------------------------------------------------------


class TestGaps(ArchiveFixture):
    def test_hole_between_segments_is_reported(self) -> None:
        # Recorder restarted: 21:12 and 21:13 were never written.
        self.write_segments(
            utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 14, 0), utc(2026, 8, 14, 21, 15, 0)
        )
        t0, t1 = utc(2026, 8, 14, 21, 11, 30), utc(2026, 8, 14, 21, 14, 30)
        spans = self.resolve(t0, t1)

        self.assert_tiles(spans, t0, t1)
        holes = tc.gaps(spans)
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0].t_start, utc(2026, 8, 14, 21, 12, 0))
        self.assertEqual(holes[0].t_end, utc(2026, 8, 14, 21, 14, 0))
        self.assertIsNone(holes[0].path)
        self.assertEqual(holes[0].segment, "")
        # 30 s before the hole + 30 s after it; the missing 120 s is not quietly dropped.
        self.assertAlmostEqual(tc.covered_seconds(spans), 60.0)

    def test_require_complete_raises_and_names_the_hole(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 14, 0))
        spans = self.resolve(utc(2026, 8, 14, 21, 11, 30), utc(2026, 8, 14, 21, 14, 30))
        with self.assertRaises(tc.MissingFootageError) as ctx:
            tc.require_complete(spans)
        self.assertEqual(len(ctx.exception.holes), 1)
        self.assertIn("21:12:00", str(ctx.exception))

    def test_require_complete_passes_a_covered_range(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 12, 0))
        spans = self.resolve(utc(2026, 8, 14, 21, 11, 58), utc(2026, 8, 14, 21, 12, 10))
        self.assertIs(tc.require_complete(spans), spans)

    def test_empty_archive_is_one_gap_not_an_error(self) -> None:
        t0, t1 = utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 11, 30)
        spans = self.resolve(t0, t1)
        self.assertEqual(len(spans), 1)
        self.assertTrue(spans[0].is_gap)
        self.assert_tiles(spans, t0, t1)
        self.assertEqual(tc.covered_seconds(spans), 0.0)

    def test_range_before_the_archive_starts(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        t0, t1 = utc(2026, 8, 14, 20, 0, 0), utc(2026, 8, 14, 20, 0, 30)
        spans = self.resolve(t0, t1)
        self.assertEqual([s.is_gap for s in spans], [True])
        self.assert_tiles(spans, t0, t1)

    def test_leading_and_trailing_gaps_around_footage(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        t0, t1 = utc(2026, 8, 14, 21, 10, 30), utc(2026, 8, 14, 21, 12, 30)
        spans = self.resolve(t0, t1)
        self.assertEqual([s.is_gap for s in spans], [True, False, True])
        self.assert_tiles(spans, t0, t1)
        self.assertAlmostEqual(tc.covered_seconds(spans), 60.0)

    def test_tail_beyond_the_last_segment_is_a_gap(self) -> None:
        # The last file has no successor, so it is credited the nominal length only. An
        # under-claim surfaces as a reported hole; an over-claim would be footage we
        # promise and cannot deliver.
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        spans = self.resolve(utc(2026, 8, 14, 21, 11, 30), utc(2026, 8, 14, 21, 12, 30))
        self.assertEqual([s.is_gap for s in spans], [False, True])
        self.assertEqual(spans[1].t_start, utc(2026, 8, 14, 21, 12, 0))

    def test_segment_and_offset_refuses_a_hole(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0), utc(2026, 8, 14, 21, 14, 0))
        name, offset = tc.segment_and_offset(utc(2026, 8, 14, 21, 11, 7), archive_dir=self.archive)
        self.assertEqual(name, "cam01_20260814_211100.mp4")
        self.assertAlmostEqual(offset, 7.0)
        with self.assertRaises(tc.MissingFootageError):
            tc.segment_and_offset(utc(2026, 8, 14, 21, 12, 30), archive_dir=self.archive)


# --------------------------------------------------------------------------------------
# Clock drift — segments are never exactly the nominal length
# --------------------------------------------------------------------------------------


class TestClockDrift(ArchiveFixture):
    def test_short_segment_does_not_leave_a_phantom_gap(self) -> None:
        # 59 s segments. Believing the nominal 60 s would put a 1 s hole at every boundary
        # and a 1 s error into every seek after it.
        short = self.nominal - 1
        starts = [utc(2026, 8, 14, 21, 11, 0) + timedelta(seconds=short * i) for i in range(3)]
        self.write_segments(*starts)
        t0, t1 = starts[1] - timedelta(seconds=9), starts[1] + timedelta(seconds=11)
        spans = self.resolve(t0, t1)

        self.assertEqual([s.is_gap for s in spans], [False, False])
        self.assert_tiles(spans, t0, t1)
        self.assertAlmostEqual(spans[0].pts_out, short, msg="the file really is 59 s long")
        self.assertAlmostEqual(spans[1].pts_in, 0.0)
        self.assertAlmostEqual(spans[1].pts_out, 11.0)
        self.assertAlmostEqual(tc.covered_seconds(spans), 20.0)

    def test_long_segment_within_tolerance_is_believed(self) -> None:
        overshoot = math.floor(self.drift)
        if overshoot < 1:
            self.skipTest("drift tolerance is below the 1 s resolution of a filename")
        start = utc(2026, 8, 14, 21, 11, 0)
        nxt = start + timedelta(seconds=self.nominal + overshoot)
        self.write_segments(start, nxt)
        spans = self.resolve(nxt - timedelta(seconds=5), nxt + timedelta(seconds=5))
        self.assertEqual([s.is_gap for s in spans], [False, False])
        self.assertAlmostEqual(spans[0].pts_out, self.nominal + overshoot)
        self.assertAlmostEqual(spans[1].pts_in, 0.0)

    def test_drift_beyond_tolerance_reads_as_a_restart(self) -> None:
        # A jump much larger than nominal + drift is a recorder restart, not a long file.
        # The interval past nominal belongs to nobody and must be reported as a hole.
        start = utc(2026, 8, 14, 21, 11, 0)
        gap_start = start + timedelta(seconds=self.nominal)
        next_start = gap_start + timedelta(seconds=math.ceil(self.drift) + 30)
        self.write_segments(start, next_start)
        spans = self.resolve(gap_start - timedelta(seconds=10), next_start + timedelta(seconds=5))
        self.assertEqual([s.is_gap for s in spans], [False, True, False])
        self.assertEqual(spans[1].t_start, gap_start)
        self.assertEqual(spans[1].t_end, next_start)

    def test_drift_accumulates_without_desyncing_the_stitch(self) -> None:
        # A second per minute for ten minutes: by the end the archive is 10 s behind
        # nominal. Filenames stay the source of truth, so nothing drifts out of alignment.
        starts, t = [], utc(2026, 8, 14, 21, 0, 0)
        for _ in range(10):
            starts.append(t)
            t += timedelta(seconds=self.nominal - 1)
        self.write_segments(*starts)
        t0, t1 = starts[0] + timedelta(seconds=5), starts[-1] + timedelta(seconds=5)
        spans = self.resolve(t0, t1)
        self.assertFalse(any(s.is_gap for s in spans))
        self.assert_tiles(spans, t0, t1)
        self.assertAlmostEqual(tc.covered_seconds(spans), (t1 - t0).total_seconds())
        for span in spans:
            self.assertAlmostEqual(span.pts_in, tc.pts_offset_for(span.t_start, span.segment))

    def test_list_segments_marks_assumed_ends(self) -> None:
        short = self.nominal - 1
        start = utc(2026, 8, 14, 21, 11, 0)
        self.write_segments(start, start + timedelta(seconds=short))
        segs = tc.list_segments(self.archive)
        self.assertEqual([s.end_is_nominal for s in segs], [False, True])
        self.assertAlmostEqual(segs[0].duration, short, msg="measured from the next filename")
        self.assertAlmostEqual(segs[1].duration, self.nominal, msg="assumed; no successor")

    def test_out_of_order_directory_listing_is_sorted(self) -> None:
        # iterdir() order is arbitrary; a resolution that depended on it would be a
        # heisenbug that only shows up on someone else's filesystem.
        self.write_segments(
            utc(2026, 8, 14, 21, 12, 0), utc(2026, 8, 14, 21, 10, 0), utc(2026, 8, 14, 21, 11, 0)
        )
        segs = tc.list_segments(self.archive)
        self.assertEqual([s.start for s in segs], sorted(s.start for s in segs))


# --------------------------------------------------------------------------------------
# Local time — SPEC §11.5, the one conversion, including across DST
# --------------------------------------------------------------------------------------


class TestLocalTime(unittest.TestCase):
    def test_uses_configured_timezone_and_format(self) -> None:
        # Derived from config rather than hardcoded to one city's offset. What this test
        # is for is that `to_local` reads `ui.display_timezone` at all — pinning the
        # offset here only asserts where the box happens to sit, and makes moving the
        # camera to another city look like a timecode regression. The DST cases below
        # pin their own zones, because those are about the arithmetic, not the config.
        dt = utc(2026, 8, 14, 21, 11, 7)
        zone = ZoneInfo(str(config.get("ui.display_timezone")))
        expected = dt.astimezone(zone)
        self.assertEqual(tc.to_local(dt).utcoffset(), expected.utcoffset())
        self.assertEqual(
            tc.format_local(dt), expected.strftime(str(config.get("ui.time_format")))
        )
        if expected.utcoffset() != timedelta(0):
            self.assertNotEqual(
                tc.format_local(dt),
                tc.format_local(dt, tz="UTC"),
                "a non-UTC display zone must render differently from UTC, or the "
                "conversion is not happening",
            )

    def test_conversion_preserves_the_instant(self) -> None:
        dt = utc(2026, 8, 14, 21, 11, 7)
        self.assertEqual(tc.to_local(dt).astimezone(UTC), dt)

    def test_naive_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tc.to_local(datetime(2026, 8, 14, 21, 11, 7))

    def test_unknown_timezone_raises_rather_than_defaulting_to_utc(self) -> None:
        with self.assertRaises(tc.TimecodeError):
            tc.to_local(utc(2026, 8, 14, 21, 11, 7), tz="Mars/Olympus_Mons")

    @unittest.skipUnless(has_zone("America/New_York"), "tzdata for America/New_York missing")
    def test_fall_back_hour_is_unambiguous(self) -> None:
        # 2026-11-01: 02:00 EDT -> 01:00 EST. Local 01:30 happens twice; the two UTC
        # instants either side must land on the same wall clock with different offsets,
        # and neither may throw.
        zone = "America/New_York"
        before = utc(2026, 11, 1, 5, 30)  # 01:30 EDT (-4)
        after = utc(2026, 11, 1, 6, 30)  # 01:30 EST (-5)
        self.assertEqual(tc.format_local(before, tz=zone), "01:30:00")
        self.assertEqual(tc.format_local(after, tz=zone), "01:30:00")
        self.assertEqual(tc.to_local(before, tz=zone).utcoffset(), timedelta(hours=-4))
        self.assertEqual(tc.to_local(after, tz=zone).utcoffset(), timedelta(hours=-5))
        self.assertEqual(tc.to_local(before, tz=zone).fold, 0)
        self.assertEqual(tc.to_local(after, tz=zone).fold, 1)
        # No information is lost: the instants stay distinct underneath.
        self.assertNotEqual(
            tc.to_local(before, tz=zone).astimezone(UTC),
            tc.to_local(after, tz=zone).astimezone(UTC),
        )
        # ...but PEP 495 compares two aware datetimes in the *same* zone by wall clock and
        # ignores `fold`, so these two distinct instants test equal. This is why sorting
        # or deduping must happen in UTC and only rendering happens in local time.
        self.assertEqual(tc.to_local(before, tz=zone), tc.to_local(after, tz=zone))

    @unittest.skipUnless(has_zone("America/New_York"), "tzdata for America/New_York missing")
    def test_spring_forward_hour_does_not_shift(self) -> None:
        # 2026-03-08: 02:00 EST -> 03:00 EDT. Local 02:30 never happens; converting
        # instants either side must stay monotonic and skip the hour cleanly.
        zone = "America/New_York"
        before = utc(2026, 3, 8, 6, 59)  # 01:59 EST
        after = utc(2026, 3, 8, 7, 1)  # 03:01 EDT
        self.assertEqual(tc.format_local(before, tz=zone), "01:59:00")
        self.assertEqual(tc.format_local(after, tz=zone), "03:01:00")
        self.assertLess(tc.to_local(before, tz=zone), tc.to_local(after, tz=zone))

    @unittest.skipUnless(has_zone("America/New_York"), "tzdata for America/New_York missing")
    def test_every_minute_across_a_dst_transition_converts(self) -> None:
        zone = ZoneInfo("America/New_York")
        dt = utc(2026, 11, 1, 4, 0)
        seen = []
        for _ in range(180):
            seen.append(tc.to_local(dt, tz=zone))
            dt += timedelta(minutes=1)
        # Nothing throws across the fold, and every instant stays distinct and ordered —
        # in UTC, which is the only ordering that survives a repeated wall-clock hour.
        instants = [d.astimezone(UTC) for d in seen]
        self.assertEqual(instants, sorted(instants))
        self.assertEqual(len(set(instants)), 180)
        # The wall clock, by contrast, replays 01:00-01:59. Rendering it is fine; relying
        # on it for ordering is not.
        self.assertEqual(len({d.replace(tzinfo=None) for d in seen}), 120)

    def test_zoneinfo_object_accepted(self) -> None:
        zone = ZoneInfo("UTC")
        self.assertEqual(tc.format_local(utc(2026, 8, 14, 21, 11, 7), tz=zone), "21:11:07")

    def test_explicit_format_override(self) -> None:
        self.assertEqual(
            tc.format_local(utc(2026, 8, 14, 21, 11, 7), fmt="%Y-%m-%d %H:%M", tz="UTC"),
            "2026-08-14 21:11",
        )


# --------------------------------------------------------------------------------------
# Serialization — spans get logged and pushed over the WebSocket
# --------------------------------------------------------------------------------------


class TestSpanSerialization(ArchiveFixture):
    def test_to_dict_is_utc_iso_and_flags_gaps(self) -> None:
        self.write_segments(utc(2026, 8, 14, 21, 11, 0))
        spans = self.resolve(utc(2026, 8, 14, 21, 11, 58), utc(2026, 8, 14, 21, 12, 10))
        footage, hole = spans[0].to_dict(), spans[1].to_dict()
        self.assertEqual(footage["t_start"], "2026-08-14T21:11:58Z")
        self.assertEqual(footage["segment"], "cam01_20260814_211100.mp4")
        self.assertFalse(footage["is_gap"])
        self.assertTrue(hole["is_gap"])
        self.assertIsNone(hole["path"])
        self.assertIsNone(hole["segment_start"])


if __name__ == "__main__":
    unittest.main()
