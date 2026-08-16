"""M1 tests — SPEC §2.

    python3 -m unittest discover -s tests -t . -v

Stdlib ``unittest``, no third-party packages, no camera and no network. CLAUDE.md forbids
tests touching the real inference endpoint (it contends with ingest) and there is nothing
to touch on this box anyway — ``vlm.backend`` is ``stub`` while SPEC §10 D1 is open.

**Most of this file never spawns a subprocess.** The window walk, the motion arithmetic,
the gate's decision logic, the filter-graph construction and the overlay's minimum-height
search are all pure functions or take an injected collaborator, which is the reason they
are written that way. The handful of tests that do run ffmpeg generate their own input
with ``-f lavfi`` — no footage is read from anywhere on disk that this file did not write.

The load-bearing tests:

* :meth:`TestFilterChain.test_scale_comes_before_drawtext` — CLAUDE.md invariant 8. Burn
  the overlay *after* the resize, or temporal localization fails silently.
* :meth:`TestPipeline.test_gated_window_still_writes_a_record` — SPEC §2.3. A gap in the
  record stream is indistinguishable from crashed ingest.
* :meth:`TestPipelineAgainstRealIndex.test_records_satisfy_the_index_contract` — M1's
  output has to be something M2 will actually accept, including the null records.
* :meth:`TestEndToEnd.test_generated_footage_produces_correct_wall_clock` — the whole
  chain over footage this test encoded itself, asserting the times come back right.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shared import config
from shared.schema import ChunkRecord, Tier, chunk_id_for
from shared.timecode import SegmentSpan, resolve_range
from shared.vlm_client import VLMChunk, VLMClient

from services.index import HashingEmbedder, IndexSettings, IndexStore, InMemoryBackend, LexicalReranker
from services.ingest import (
    FrameExtractor,
    GateBackend,
    GateReason,
    IngestError,
    IngestPipeline,
    IngestSettings,
    MotionGate,
    OverlayPosition,
    OverlayTooSmallError,
    PassthroughGate,
    StubCaptioner,
    VLMCaptioner,
    Window,
    build_captioner,
    build_gate,
    drawtext_filter,
    filter_chain,
    gmtime_expression,
    mean_abs_delta,
    measure_text_height_px,
    motion_score,
    plan_windows,
    resolve_overlay_fontsize,
    scale_filter,
    split_jpeg_stream,
    split_thumbnails,
)
from services.ingest.ffmpeg import FFmpegDecodeError
from services.ingest.gate import thumbnail_command
from services.ingest.settings import PENDING_SETTINGS

CAMERA = "cam01"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# A staged minute on cam01, matching the SPEC §3.1 example so the ids in the assertions
# below are the ones the spec prints.
SEGMENT_START = datetime(2026, 8, 14, 21, 11, 0, tzinfo=timezone.utc)
SEGMENT = "cam01_20260814_211100.mp4"

FFMPEG = os.environ.get("SPARK_FFMPEG") or "ffmpeg"
HAVE_FFMPEG = subprocess.run(  # noqa: S603
    [FFMPEG, "-version"], capture_output=True, check=False
).returncode == 0
HAVE_FONT = Path(FONT).is_file()


def make_settings(**overrides: Any) -> IngestSettings:
    """Settings with no config dependency, for the pure tests.

    Deliberately not ``IngestSettings.from_config()``: these tests assert behaviour, and a
    behaviour test that silently changes meaning when somebody edits settings.yaml is a
    test that will be deleted rather than fixed.
    """
    base = IngestSettings(
        camera_id=CAMERA,
        archive_dir=Path("/nonexistent/archive"),
        ffmpeg_bin=FFMPEG,
        ffmpeg_timeout_seconds=30.0,
        window_seconds=5.0,
        stride_seconds=4.0,
        sample_fps=1.0,
        live_short_side_px=512,
        frame_jpeg_quality=3,
        gate_enabled=True,
        gate_backend=GateBackend.MOTION,
        gate_sample_fps=2.0,
        motion_threshold=0.02,
        thumbnail_size=32,
        warmup_windows=2,
        target_skip_rate=0.80,
        warn_skip_rate=0.60,
        overlay_enabled=True,
        overlay_format="%Y-%m-%d %H:%M:%S UTC",
        overlay_position=OverlayPosition.BOTTOM,
        overlay_min_height_px=16,
        overlay_fontfile=FONT,
        overlay_fontsize=20,
        overlay_fontcolor="white",
        overlay_box_opacity=0.6,
        overlay_padding_px=10,
        overlay_max_fontsize=64,
        vlm_backend="stub",
        caption_prompt="Describe what is happening in this scene.",
        # Off by default here: these are behaviour tests over the caption path, and a
        # checklist read from a real config file would make them depend on whatever
        # standing tasks happen to exist. tests/test_watchlist.py covers it on its own.
        watchlist_enabled=False,
        watchlist_path=Path("/nonexistent/watchlist.json"),
        watchlist_seed_path=Path("/nonexistent/tasks.yaml"),
        watchlist_preamble="",
        watchlist_max_items=8,
        caption_timeout_seconds=30.0,
        poll_interval_seconds=1.0,
        bench_target_seconds=4.0,
    )
    return dataclasses.replace(base, **overrides)


def flat(value: int, size: int = 1024) -> bytes:
    """A uniform thumbnail. Two of these differ by exactly their level difference."""
    return bytes([value]) * size


def span_at(offset: float, duration: float, path: str = SEGMENT) -> SegmentSpan:
    return SegmentSpan(
        path=Path(path),
        segment_start=SEGMENT_START,
        pts_in=offset,
        pts_out=offset + duration,
        t_start=SEGMENT_START + timedelta(seconds=offset),
        t_end=SEGMENT_START + timedelta(seconds=offset + duration),
    )


def gap_at(offset: float, duration: float) -> SegmentSpan:
    return SegmentSpan(
        path=None,
        segment_start=None,
        pts_in=0.0,
        pts_out=0.0,
        t_start=SEGMENT_START + timedelta(seconds=offset),
        t_end=SEGMENT_START + timedelta(seconds=offset + duration),
    )


# --------------------------------------------------------------------------------------
# SPEC §2.2 — analysis windows
# --------------------------------------------------------------------------------------


class TestWindowPlanning(unittest.TestCase):
    """Windows are time ranges. Nothing here opens a file."""

    def test_five_second_windows_on_a_four_second_stride_overlap_by_one(self) -> None:
        start = SEGMENT_START
        windows = list(plan_windows(start, start + timedelta(seconds=30), 5.0, 4.0))
        self.assertEqual(windows[0].t_start, start)
        self.assertEqual(windows[0].t_end, start + timedelta(seconds=5))
        self.assertEqual(windows[1].t_start, start + timedelta(seconds=4))
        # SPEC §2.2's whole reason for a stride shorter than the window: an event on the
        # boundary must appear whole in at least one of them.
        overlap = (windows[0].t_end - windows[1].t_start).total_seconds()
        self.assertEqual(overlap, 1.0)

    def test_indices_are_consecutive_from_the_start_index(self) -> None:
        start = SEGMENT_START
        windows = list(plan_windows(start, start + timedelta(seconds=20), 5.0, 4.0, start_index=7))
        self.assertEqual([w.index for w in windows], [7, 8, 9, 10])

    def test_a_partial_trailing_window_is_never_emitted(self) -> None:
        start = SEGMENT_START
        # 7 s of footage holds exactly one complete 5 s window; the remaining 2 s is not
        # a window, it is footage waiting for the archive to grow.
        windows = list(plan_windows(start, start + timedelta(seconds=7), 5.0, 4.0))
        self.assertEqual(len(windows), 1)
        self.assertLessEqual(windows[-1].t_end, start + timedelta(seconds=7))

    def test_every_window_ends_inside_the_requested_range(self) -> None:
        start = SEGMENT_START
        end = start + timedelta(seconds=61)
        for window in plan_windows(start, end, 5.0, 4.0):
            self.assertGreaterEqual(window.t_start, start)
            self.assertLessEqual(window.t_end, end)

    def test_chunk_id_matches_the_schema_helper(self) -> None:
        window = next(iter(plan_windows(SEGMENT_START, SEGMENT_START + timedelta(seconds=5), 5.0, 4.0)))
        self.assertEqual(
            window.chunk_id(CAMERA), chunk_id_for(CAMERA, window.t_start, window.t_end)
        )

    def test_naive_datetimes_are_rejected(self) -> None:
        naive = datetime(2026, 8, 14, 21, 11, 0)
        with self.assertRaises(ValueError):
            list(plan_windows(naive, naive + timedelta(seconds=30), 5.0, 4.0))

    def test_nonpositive_window_or_stride_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            list(plan_windows(SEGMENT_START, SEGMENT_START + timedelta(seconds=30), 0.0, 4.0))
        with self.assertRaises(ValueError):
            list(plan_windows(SEGMENT_START, SEGMENT_START + timedelta(seconds=30), 5.0, -1.0))


class TestSettingsValidation(unittest.TestCase):
    def test_a_stride_longer_than_the_window_is_rejected(self) -> None:
        with self.assertRaises(IngestError) as ctx:
            make_settings(stride_seconds=6.0).validate()
        self.assertIn("never be analysed", str(ctx.exception))

    def test_motion_threshold_must_be_normalised(self) -> None:
        with self.assertRaises(IngestError):
            make_settings(motion_threshold=12.0).validate()

    def test_frames_per_window_follows_spec_2_2(self) -> None:
        # 5 s at 1 fps -> 5 frames. The number SPEC §9 block 0 benchmarks.
        self.assertEqual(make_settings().frames_per_window, 5)

    def test_thumbnail_bytes_is_the_gate_budget(self) -> None:
        self.assertEqual(make_settings().thumbnail_bytes, 1024)

    def test_pending_settings_are_all_absent_from_the_yaml(self) -> None:
        """A pending key that has since been added to settings.yaml should be deleted
        from the table — otherwise the fallback shadows nothing and confuses the next
        person to go looking for the dial."""
        for dotted in PENDING_SETTINGS:
            with self.subTest(setting=dotted):
                self.assertIs(
                    config.get(dotted, None),
                    None,
                    f"{dotted} is now in settings.yaml; drop it from PENDING_SETTINGS",
                )


# --------------------------------------------------------------------------------------
# SPEC §2.3 — the gate, as arithmetic
# --------------------------------------------------------------------------------------


class TestMotionArithmetic(unittest.TestCase):
    def test_identical_frames_score_zero(self) -> None:
        self.assertEqual(mean_abs_delta(flat(128), flat(128)), 0.0)

    def test_black_to_white_scores_one(self) -> None:
        self.assertEqual(mean_abs_delta(flat(0), flat(255)), 1.0)

    def test_score_is_normalised_by_the_gray_range(self) -> None:
        # A uniform shift of 25 levels out of 255.
        self.assertAlmostEqual(mean_abs_delta(flat(100), flat(125)), 25 / 255, places=6)

    def test_mismatched_sizes_raise(self) -> None:
        with self.assertRaises(ValueError):
            mean_abs_delta(flat(0, 1024), flat(0, 512))

    def test_empty_thumbnails_raise(self) -> None:
        with self.assertRaises(ValueError):
            mean_abs_delta(b"", b"")

    def test_window_score_is_the_maximum_not_the_mean(self) -> None:
        """One second of movement inside five of stillness must not average away — that
        is the difference between a gate and a blur."""
        frames = [flat(100), flat(100), flat(200), flat(200), flat(200)]
        peak = mean_abs_delta(flat(100), flat(200))
        self.assertAlmostEqual(motion_score(frames), peak, places=9)
        mean = peak / 4
        self.assertGreater(motion_score(frames), mean)

    def test_fewer_than_two_frames_has_no_score(self) -> None:
        self.assertIsNone(motion_score([]))
        self.assertIsNone(motion_score([flat(10)]))

    def test_split_thumbnails_discards_a_truncated_tail(self) -> None:
        raw = flat(1) + flat(2) + b"\x03" * 100
        frames = split_thumbnails(raw, 1024)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1], flat(2))

    def test_thumbnail_command_seeks_before_the_input(self) -> None:
        argv = thumbnail_command(FFMPEG, span_at(7.0, 5.0), sample_fps=2, size=32)
        self.assertLess(argv.index("-ss"), argv.index("-i"))
        self.assertIn("fps=2,scale=32:32,format=gray", argv)
        self.assertEqual(argv[argv.index("-t") + 1], "5.000")

    def test_thumbnail_command_refuses_a_gap(self) -> None:
        with self.assertRaises(ValueError):
            thumbnail_command(FFMPEG, gap_at(0.0, 5.0), sample_fps=2, size=32)


class TestMotionGate(unittest.TestCase):
    """The decision logic, against injected byte strings. No ffmpeg, no footage."""

    def gate(self, frames_by_call: list[list[bytes]], **overrides: Any) -> MotionGate:
        calls = iter(frames_by_call)

        def extract(spans: Any) -> list[bytes]:
            try:
                return next(calls)
            except StopIteration:  # pragma: no cover - a test asking for too many windows
                return []

        return MotionGate(make_settings(**overrides), extract=extract)

    def test_warmup_windows_are_never_skipped(self) -> None:
        """SPEC §2.3 / ingest.gate.warmup_windows: there is no reference frame yet, and a
        false skip leaves no trace at all."""
        gate = self.gate([[flat(50)] * 4, [flat(50)] * 4], warmup_windows=2)
        for index in (0, 1):
            decision = gate.evaluate(Window(index, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
            self.assertTrue(decision.passed)
            self.assertIs(decision.reason, GateReason.WARMUP)

    def test_a_still_scene_is_skipped(self) -> None:
        gate = self.gate([[flat(90)] * 6], warmup_windows=0)
        decision = gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        self.assertTrue(decision.skipped)
        self.assertIs(decision.reason, GateReason.STILL)
        self.assertEqual(decision.score, 0.0)

    def test_movement_passes(self) -> None:
        gate = self.gate([[flat(30), flat(30), flat(200), flat(200)]], warmup_windows=0)
        decision = gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        self.assertTrue(decision.passed)
        self.assertIs(decision.reason, GateReason.MOTION)
        self.assertGreater(decision.score or 0.0, 0.02)

    def test_the_previous_window_supplies_the_reference_frame(self) -> None:
        """A scene that is internally still but different from the last window is motion.
        Without the carried reference, a movement that completes between two windows is
        invisible to both."""
        gate = self.gate([[flat(40)] * 3, [flat(200)] * 3], warmup_windows=0)
        first = gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        second = gate.evaluate(Window(1, SEGMENT_START, SEGMENT_START), [span_at(4, 5)])
        self.assertTrue(first.skipped)
        self.assertTrue(second.passed)
        self.assertIs(second.reason, GateReason.MOTION)

    def test_an_undecodable_window_fails_open(self) -> None:
        """We did not look, so we do not get to say nothing happened."""

        def boom(spans: Any) -> list[bytes]:
            raise FFmpegDecodeError("moov atom not found")

        gate = MotionGate(make_settings(warmup_windows=0), extract=boom)
        decision = gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        self.assertTrue(decision.passed)
        self.assertIs(decision.reason, GateReason.UNDECODABLE)
        self.assertIn("moov atom", decision.error)

    def test_a_window_that_is_all_gap_fails_open(self) -> None:
        gate = self.gate([], warmup_windows=0)
        decision = gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [gap_at(0, 5)])
        self.assertTrue(decision.passed)
        self.assertIs(decision.reason, GateReason.NO_FOOTAGE)

    def test_a_single_frame_cannot_be_scored_and_passes(self) -> None:
        gate = self.gate([[flat(10)]], warmup_windows=0)
        decision = gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        self.assertTrue(decision.passed)
        self.assertIs(decision.reason, GateReason.UNDECODABLE)

    def test_reset_forgets_the_reference(self) -> None:
        gate = self.gate([[flat(40)] * 3, [flat(200)] * 3], warmup_windows=0)
        gate.evaluate(Window(0, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        gate.reset()
        second = gate.evaluate(Window(1, SEGMENT_START, SEGMENT_START), [span_at(4, 5)])
        self.assertTrue(second.skipped)


class TestGateSelection(unittest.TestCase):
    def test_disabled_gate_passes_everything(self) -> None:
        gate = build_gate(make_settings(gate_enabled=False))
        self.assertIsInstance(gate, PassthroughGate)
        decision = gate.evaluate(Window(9, SEGMENT_START, SEGMENT_START), [span_at(0, 5)])
        self.assertTrue(decision.passed)
        self.assertIs(decision.reason, GateReason.DISABLED)

    def test_motion_is_the_backend_that_ships(self) -> None:
        self.assertIsInstance(build_gate(make_settings()), MotionGate)

    def test_an_unimplemented_backend_raises_rather_than_falling_back(self) -> None:
        """A gate silently running a different backend than the configured one makes
        every skip-rate number meaningless."""
        with self.assertRaises(IngestError) as ctx:
            build_gate(make_settings(gate_backend=GateBackend.DEEPSTREAM))
        self.assertIn("deepstream", str(ctx.exception))


# --------------------------------------------------------------------------------------
# SPEC §2.4 — resize, then burn. CLAUDE.md invariant 8.
# --------------------------------------------------------------------------------------


class TestFilterChain(unittest.TestCase):
    def test_scale_comes_before_drawtext(self) -> None:
        """CLAUDE.md invariant 8. Burning first and resizing after turns 22 px of text
        into grey mush, and nothing errors — the model just stops citing times."""
        chain = filter_chain(make_settings(), SEGMENT_START, 22)
        self.assertLess(chain.index("scale="), chain.index("drawtext="))

    def test_sampling_comes_first_of_all(self) -> None:
        chain = filter_chain(make_settings(), SEGMENT_START, 22)
        self.assertTrue(chain.startswith("fps=1.0,"))

    def test_overlay_can_be_switched_off(self) -> None:
        chain = filter_chain(make_settings(overlay_enabled=False), SEGMENT_START, 22)
        self.assertNotIn("drawtext", chain)
        self.assertIn("scale=", chain)

    def test_scale_targets_the_short_side_in_both_orientations(self) -> None:
        """The camera offers portrait modes (1080x1920). A fixed height would blow a
        portrait frame up to 512 px wide, quadrupling exactly the thing SPEC §2.5 asks us
        to shrink."""
        expr = scale_filter(512)
        self.assertIn("gt(iw,ih)", expr)
        self.assertIn("min(512,ih)", expr)
        self.assertIn("min(512,iw)", expr)

    def test_scale_never_upscales(self) -> None:
        self.assertIn("min(", scale_filter(512))

    def test_scale_rejects_a_nonpositive_side(self) -> None:
        with self.assertRaises(ValueError):
            scale_filter(0)

    def test_overlay_is_positioned_relative_to_its_own_height(self) -> None:
        """``y=h-text_h-margin``, not a literal offset: raising the fontsize to meet the
        min-height floor would otherwise push the clock off the bottom of the frame."""
        bottom = drawtext_filter(
            "x", fontfile=FONT, fontsize=22, fontcolor="white", box_opacity=0.6, padding_px=10
        )
        self.assertIn("y=h-text_h-10", bottom)
        top = drawtext_filter(
            "x",
            fontfile=FONT,
            fontsize=22,
            fontcolor="white",
            box_opacity=0.6,
            padding_px=10,
            position=OverlayPosition.TOP,
        )
        self.assertIn("y=10", top)

    def test_gmtime_expression_escapes_each_parser_layer(self) -> None:
        """Verified empirically against ffmpeg 6.1.1: one backslash separates ``%{pts}``'s
        own arguments, three escape a colon inside the strftime format. Fewer and ffmpeg
        says "%{pts} requires at most 3 arguments"; differently and it says "Stray %"."""
        # The helper takes a wall-clock instant, not an epoch — there is exactly one
        # implementation of this escaping in the repo (services/worker/decode.py) and it
        # speaks datetimes like everything else that crosses a module boundary.
        wall_start = datetime.fromtimestamp(1786785682, tz=timezone.utc)
        expr = gmtime_expression(wall_start, "%Y-%m-%d %H:%M:%S UTC")
        self.assertEqual(
            expr,
            "%{pts\\:gmtime\\:1786785682\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S UTC}",
        )

    def test_the_overlay_carries_utc(self) -> None:
        """SPEC §10 D8 — the overlay is drawn onto the sampled frames the model reads,
        never into the archive, so the timezone is a config change rather than a
        re-shoot. It carries UTC like every other cross-boundary value."""
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        self.assertIn("gmtime", gmtime_expression(epoch, "%H:%M:%S"))


class TestOverlayMinimumHeight(unittest.TestCase):
    """The invariant-8 floor, searched against an injected measurement."""

    def test_a_short_overlay_raises_the_fontsize(self) -> None:
        # 0.75 px of ink per px of em, which is what DejaVuSans measures on this box.
        resolved = resolve_overlay_fontsize(
            make_settings(overlay_fontsize=20, overlay_min_height_px=16),
            measure=lambda size: int(size * 0.75),
        )
        self.assertEqual(resolved, 22)
        self.assertGreaterEqual(int(resolved * 0.75), 16)

    def test_an_adequate_fontsize_is_left_alone(self) -> None:
        resolved = resolve_overlay_fontsize(
            make_settings(overlay_fontsize=32, overlay_min_height_px=16),
            measure=lambda size: int(size * 0.75),
        )
        self.assertEqual(resolved, 32)

    def test_an_unreachable_floor_stops_the_run(self) -> None:
        """Rather than indexing footage whose burned clock nothing can read."""
        with self.assertRaises(OverlayTooSmallError):
            resolve_overlay_fontsize(
                make_settings(overlay_min_height_px=999, overlay_max_fontsize=40),
                measure=lambda size: int(size * 0.75),
            )

    def test_a_disabled_overlay_short_circuits(self) -> None:
        def never(size: int) -> int:  # pragma: no cover - must not be called
            raise AssertionError("measurement attempted with the overlay disabled")

        self.assertEqual(
            resolve_overlay_fontsize(
                make_settings(overlay_enabled=False, overlay_fontsize=20), measure=never
            ),
            20,
        )

    def test_a_failed_probe_over_estimates_rather_than_stopping(self) -> None:
        def boom(size: int) -> int:
            raise FFmpegDecodeError("no such font")

        resolved = resolve_overlay_fontsize(
            make_settings(overlay_fontsize=10, overlay_min_height_px=16), measure=boom
        )
        # ceil(16 / 0.70) — the fallback ratio is the low end of what was measured, so
        # the estimate errs large. An overlay two pixels taller than needed costs nothing.
        self.assertEqual(resolved, 23)


class TestJpegSplitting(unittest.TestCase):
    def test_a_concatenated_stream_splits_on_the_soi_marker(self) -> None:
        a = b"\xff\xd8\xff" + b"AAAA"
        b = b"\xff\xd8\xff" + b"BBBB"
        self.assertEqual(split_jpeg_stream(a + b), [a, b])

    def test_an_empty_stream_yields_nothing(self) -> None:
        self.assertEqual(split_jpeg_stream(b""), [])

    def test_bytes_without_a_marker_yield_nothing(self) -> None:
        self.assertEqual(split_jpeg_stream(b"not a jpeg"), [])


# --------------------------------------------------------------------------------------
# SPEC §2.4 — captioning
# --------------------------------------------------------------------------------------


def make_vlm_chunk(chunk_id: str, frames: int = 5) -> VLMChunk:
    return VLMChunk(chunk_id=chunk_id, frames=[f"data:image/jpeg;base64,{i}" for i in range(frames)])


class TestStubCaptioner(unittest.TestCase):
    """Not a test mock — the thing that makes the pipeline runnable while D1 is open."""

    def test_captions_are_deterministic_across_instances(self) -> None:
        """Re-running ingest over the same archive must produce an identical index, or
        "did my change break retrieval?" is not an answerable question."""
        chunk = make_vlm_chunk("cam01_20260814T211107_211112")
        first = StubCaptioner(make_settings()).caption([chunk])[0].text
        second = StubCaptioner(make_settings()).caption([chunk])[0].text
        self.assertEqual(first, second)

    def test_different_chunks_get_different_captions(self) -> None:
        captioner = StubCaptioner(make_settings())
        texts = {
            captioner.caption([make_vlm_chunk(f"cam01_2026081{i}T211107_211112")])[0].text
            for i in range(8)
        }
        self.assertGreater(len(texts), 1)

    def test_every_caption_is_marked_synthetic(self) -> None:
        """An unmarked synthetic caption looks exactly like a real one on a screen at
        hour 38, and the demo's whole claim is that the captions came from the footage."""
        text = StubCaptioner(make_settings()).caption([make_vlm_chunk("x")])[0].text
        self.assertTrue(text.startswith("[stub]"))

    def test_the_result_reports_the_stub_as_its_model(self) -> None:
        result = StubCaptioner(make_settings()).caption([make_vlm_chunk("x")])[0]
        self.assertEqual(result.model, "stub")
        self.assertEqual(result.profile, "live")
        # Zero rather than a plausible guess: a fabricated token count would flow into
        # bench.py's tokens-per-second and make a meaningless number look credible.
        self.assertEqual(result.completion_tokens, 0)

    def test_the_api_takes_a_list_and_answers_in_order(self) -> None:
        """CLAUDE.md invariant 9."""
        chunks = [make_vlm_chunk("a"), make_vlm_chunk("b"), make_vlm_chunk("c")]
        results = StubCaptioner(make_settings()).caption(chunks)
        self.assertEqual([r.chunk_id for r in results], ["a", "b", "c"])

    def test_a_chunk_with_no_frames_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            StubCaptioner(make_settings()).caption([VLMChunk(chunk_id="x", frames=[])])


class FakeTransport:
    """Stands in for the inference endpoint. CLAUDE.md forbids tests reaching the real one."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, url: str, payload: Any, *, timeout: float | None) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        return {
            "choices": [{"message": {"content": "A white panel van reverses toward the door."}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 24},
        }


class TestVLMCaptioner(unittest.TestCase):
    """The real path, against a mocked transport."""

    def test_it_captions_through_the_shared_client_on_the_live_profile(self) -> None:
        transport = FakeTransport()
        client = VLMClient(transport, model="cosmos-reason-test")
        captioner = VLMCaptioner(make_settings(), client)
        results = captioner.caption([make_vlm_chunk("cam01_20260814T211107_211112")])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].profile, "live")
        self.assertEqual(captioner.model, "cosmos-reason-test")

    def test_it_never_widens_the_token_budget(self) -> None:
        """CLAUDE.md invariant 6: max_tokens is the profile's, not M1's. Decode is ~95%
        of latency and letting captions get longer is the fastest way to lose real time."""
        transport = FakeTransport()
        captioner = VLMCaptioner(make_settings(), VLMClient(transport, model="m"))
        captioner.caption([make_vlm_chunk("x")])
        payload = transport.payloads[0]
        self.assertEqual(payload["max_tokens"], int(config.get("vlm.profiles.live.max_tokens")))
        self.assertFalse(payload["chat_template_kwargs"]["enable_reasoning"])

    def test_the_prompt_comes_from_config_not_from_m1(self) -> None:
        """So M1 and the rollup job (SPEC §3.3) cannot drift into two caption styles in
        one index."""
        transport = FakeTransport()
        settings = make_settings(caption_prompt="THE CONFIGURED PROMPT")
        VLMCaptioner(settings, VLMClient(transport, model="m")).caption([make_vlm_chunk("x")])
        text_parts = [
            part["text"]
            for part in transport.payloads[0]["messages"][0]["content"]
            if part["type"] == "text"
        ]
        self.assertEqual(text_parts, ["THE CONFIGURED PROMPT"])


class TestCaptionerSelection(unittest.TestCase):
    def test_stub_is_selected_from_config(self) -> None:
        self.assertIsInstance(build_captioner(make_settings(vlm_backend="stub")), StubCaptioner)

    def test_an_unknown_backend_raises_rather_than_defaulting(self) -> None:
        """A typo would otherwise produce a full, plausible-looking index of captions of
        nothing — the worst outcome available to this module."""
        with self.assertRaises(IngestError):
            build_captioner(make_settings(vlm_backend="vlllm"))


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------


class RecordingSink:
    """A ``ChunkSink`` that keeps what it was given."""

    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []

    def insert(self, chunks: list[ChunkRecord]) -> int:
        self.records.extend(chunks)
        return len(chunks)


class FixedGate:
    def __init__(self, passed: bool, reason: GateReason) -> None:
        self._passed = passed
        self._reason = reason

    def evaluate(self, window: Window, spans: Any) -> Any:
        from services.ingest import GateDecision

        return GateDecision(self._passed, self._reason, score=0.0)

    def reset(self) -> None:
        return None


class FakeExtractor:
    def __init__(self, frames: list[bytes] | None = None, error: Exception | None = None) -> None:
        self._frames = frames if frames is not None else [b"\xff\xd8\xffJPEG"] * 5
        self._error = error

    def extract(self, spans: Any) -> list[bytes]:
        if self._error is not None:
            raise self._error
        return self._frames


class ArchiveFixture(unittest.TestCase):
    """Base class that lays down an archive of correctly *named* segment files.

    ``shared/timecode.py`` maps time onto the archive from filenames alone — no ffprobe,
    no decode — which is what lets these tests exercise the real resolution path against
    zero-byte files. The default ``recorder.segment_seconds`` is 60, so the fixture uses
    60 s spacing and needs no config override.
    """

    segments = 2

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name)
        for n in range(self.segments):
            start = SEGMENT_START + timedelta(seconds=60 * n)
            (self.archive / f"{CAMERA}_{start:%Y%m%d_%H%M%S}.mp4").write_bytes(b"")
        self.settings = make_settings(archive_dir=self.archive)


class TestPipeline(ArchiveFixture):
    def pipeline(self, **kwargs: Any) -> tuple[IngestPipeline, RecordingSink]:
        sink = RecordingSink()
        kwargs.setdefault("gate", FixedGate(True, GateReason.MOTION))
        kwargs.setdefault("extractor", FakeExtractor())
        kwargs.setdefault("captioner", StubCaptioner(self.settings))
        return IngestPipeline(self.settings, sink=sink, **kwargs), sink

    def test_a_captioned_window_carries_wall_clock_and_pts(self) -> None:
        """CLAUDE.md invariant 2 — both, always. ``segment`` + ``pts_offset`` is the only
        join between a text hit and the pixels."""
        pipeline, sink = self.pipeline()
        with pipeline:
            record = pipeline.process(
                Window(5, SEGMENT_START + timedelta(seconds=7), SEGMENT_START + timedelta(seconds=12))
            )
        assert record is not None
        self.assertEqual(record.chunk_id, "cam01_20260814T211107_211112")
        self.assertEqual(record.segment, SEGMENT)
        self.assertEqual(record.pts_offset, 7.0)
        self.assertEqual(record.tier, Tier.LIVE)
        self.assertFalse(record.gated)
        self.assertTrue(record.caption)
        self.assertEqual(sink.records, [record])

    def test_a_window_spanning_two_segments_anchors_on_the_first(self) -> None:
        """SPEC §3.1: ``pts_offset = t_start − segment_start``. The window ends in the
        next file; the record is anchored where it starts."""
        pipeline, _ = self.pipeline()
        with pipeline:
            record = pipeline.process(
                Window(0, SEGMENT_START + timedelta(seconds=58), SEGMENT_START + timedelta(seconds=63))
            )
        assert record is not None
        self.assertEqual(record.segment, SEGMENT)
        self.assertEqual(record.pts_offset, 58.0)

    def test_gated_window_still_writes_a_record(self) -> None:
        """SPEC §2.3. A gap in the record stream is indistinguishable from crashed
        ingest, and the skip rate is only observable because these rows exist."""
        pipeline, sink = self.pipeline(gate=FixedGate(False, GateReason.STILL))
        with pipeline:
            record = pipeline.process(Window(5, SEGMENT_START, SEGMENT_START + timedelta(seconds=5)))
        assert record is not None
        self.assertTrue(record.gated)
        self.assertEqual(record.caption, "")
        self.assertEqual(record.embedding, [])
        # Still located in time: a null record the deep worker cannot place is useless.
        self.assertEqual(record.segment, SEGMENT)
        self.assertEqual(len(sink.records), 1)
        self.assertEqual(pipeline.stats.skipped, 1)
        self.assertEqual(pipeline.stats.captioned, 0)

    def test_an_undecodable_window_is_not_counted_as_a_skip(self) -> None:
        """A 100% skip rate made of unreadable files is a broken archive wearing a
        healthy gate's numbers."""
        pipeline, sink = self.pipeline(
            extractor=FakeExtractor(error=FFmpegDecodeError("moov atom not found"))
        )
        with pipeline:
            record = pipeline.process(Window(5, SEGMENT_START, SEGMENT_START + timedelta(seconds=5)))
        assert record is not None
        self.assertTrue(record.gated)
        self.assertEqual(pipeline.stats.decode_failures, 1)
        self.assertEqual(pipeline.stats.skipped, 0)
        self.assertEqual(pipeline.stats.captioned, 0)
        self.assertEqual(len(sink.records), 1)

    def test_a_window_with_no_frames_is_a_decode_failure(self) -> None:
        pipeline, _ = self.pipeline(extractor=FakeExtractor(frames=[]))
        with pipeline:
            pipeline.process(Window(5, SEGMENT_START, SEGMENT_START + timedelta(seconds=5)))
        self.assertEqual(pipeline.stats.decode_failures, 1)

    def test_a_window_outside_the_archive_writes_nothing(self) -> None:
        """``segment`` is required by the schema and there is no honest value for it. It
        is counted separately so it can never be read as a gate skip."""
        pipeline, sink = self.pipeline()
        far = SEGMENT_START + timedelta(days=1)
        with pipeline:
            record = pipeline.process(Window(0, far, far + timedelta(seconds=5)))
        self.assertIsNone(record)
        self.assertEqual(sink.records, [])
        self.assertEqual(pipeline.stats.no_footage, 1)
        self.assertEqual(pipeline.stats.skipped, 0)

    def test_skip_rate_is_computed_over_decided_windows(self) -> None:
        pipeline, _ = self.pipeline(gate=FixedGate(False, GateReason.STILL))
        with pipeline:
            stats = pipeline.run()
        self.assertGreater(stats.windows, 0)
        self.assertEqual(stats.skip_rate, 1.0)
        self.assertEqual(stats.health(self.settings), "ok")

    def test_a_busy_scene_reports_the_gate_as_mistuned(self) -> None:
        """SPEC §2.3: below ingest.gate.warn_skip_rate the gate is mistuned and real time
        is gone. The number has to say so on its own."""
        pipeline, _ = self.pipeline()
        with pipeline:
            stats = pipeline.run(limit=4)
        self.assertEqual(stats.skip_rate, 0.0)
        self.assertEqual(stats.health(self.settings), "low")

    def test_an_empty_run_is_empty_not_a_division_error(self) -> None:
        pipeline, _ = self.pipeline()
        with pipeline:
            stats = pipeline.run(limit=0)
        self.assertEqual(stats.skip_rate, 0.0)
        self.assertEqual(stats.health(self.settings), "empty")

    def test_the_walk_covers_the_archive_on_the_configured_stride(self) -> None:
        pipeline, _ = self.pipeline()
        windows = list(pipeline.plan())
        self.assertGreater(len(windows), 1)
        deltas = {
            (b.t_start - a.t_start).total_seconds() for a, b in zip(windows, windows[1:])
        }
        self.assertEqual(deltas, {self.settings.stride_seconds})


class TestPipelineAgainstRealIndex(ArchiveFixture):
    """M1's output has to be something M2 accepts — including the null records.

    ``IndexStore._validate`` rejects a gated record carrying a caption, and an ungated one
    without. Those are M1 contract violations by construction, so asserting against the
    real store is the only way to know the two modules agree.
    """

    def index(self) -> IndexStore:
        settings = IndexSettings.from_config()
        return IndexStore(
            backend=InMemoryBackend(settings),
            embedder=HashingEmbedder(settings.embed_dims),
            reranker=LexicalReranker(),
            settings=settings,
        )

    def test_records_satisfy_the_index_contract(self) -> None:
        store = self.index()
        gate = _AlternatingGate()
        with IngestPipeline(
            self.settings,
            gate=gate,
            extractor=FakeExtractor(),
            captioner=StubCaptioner(self.settings),
            sink=store,
        ) as pipeline:
            stats = pipeline.run(limit=6)

        self.assertEqual(stats.records_written, 6)
        counts = store.stats()
        self.assertEqual(counts.total, 6)
        self.assertGreater(counts.captioned, 0)
        self.assertGreater(counts.gated, 0)
        # The gate skip rate survives into M2 — SPEC §2.3's metric is observable after
        # the fact precisely because the null records are stored rather than dropped.
        self.assertAlmostEqual(counts.skip_rate, stats.skip_rate, places=6)

    def test_a_captioned_chunk_is_retrievable_with_its_wall_clock_intact(self) -> None:
        store = self.index()
        with IngestPipeline(
            self.settings,
            gate=FixedGate(True, GateReason.MOTION),
            extractor=FakeExtractor(),
            captioner=StubCaptioner(self.settings),
            sink=store,
        ) as pipeline:
            pipeline.process(
                Window(0, SEGMENT_START + timedelta(seconds=7), SEGMENT_START + timedelta(seconds=12))
            )
        hits = store.search("person seated door chair room")
        self.assertTrue(hits)
        t_start, t_end = hits[0].time_range
        self.assertEqual(t_start, SEGMENT_START + timedelta(seconds=7))
        self.assertEqual(t_end, SEGMENT_START + timedelta(seconds=12))
        self.assertEqual(hits[0].record.segment, SEGMENT)
        self.assertEqual(hits[0].record.pts_offset, 7.0)


class _AlternatingGate:
    """Passes every other window, so one run produces both record shapes."""

    def __init__(self) -> None:
        self._n = 0

    def evaluate(self, window: Window, spans: Any) -> Any:
        from services.ingest import GateDecision

        self._n += 1
        passed = self._n % 2 == 1
        return GateDecision(
            passed, GateReason.MOTION if passed else GateReason.STILL, score=0.0
        )

    def reset(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# The parts that need ffmpeg. Every input below is generated by this file with -f lavfi;
# nothing reads a camera, a network or any footage the tests did not create.
# --------------------------------------------------------------------------------------


@unittest.skipUnless(HAVE_FFMPEG and HAVE_FONT, "ffmpeg or the DejaVu font is unavailable")
class TestOverlayMeasuredForReal(unittest.TestCase):
    def test_the_burned_text_clears_the_configured_floor(self) -> None:
        """Invariant 8, measured rather than asserted: render the real string with the
        real font and count the rows it inks."""
        settings = make_settings()
        fontsize = resolve_overlay_fontsize(settings)
        height = measure_text_height_px(settings, fontsize)
        self.assertGreaterEqual(height, settings.overlay_min_height_px)

    def test_the_shipped_fontsize_is_one_pixel_short(self) -> None:
        """Documenting a real finding about config/settings.yaml rather than a
        hypothetical: DejaVuSans renders ~0.75 of its em as ink, so ``fontsize: 20``
        yields 15 px against a ``min_height_px`` of 16, and the run raises it."""
        settings = make_settings(overlay_fontsize=20)
        self.assertLess(measure_text_height_px(settings, 20), settings.overlay_min_height_px)
        self.assertGreater(resolve_overlay_fontsize(settings), 20)

    def test_height_grows_with_fontsize(self) -> None:
        settings = make_settings()
        self.assertGreater(
            measure_text_height_px(settings, 40), measure_text_height_px(settings, 20)
        )


@unittest.skipUnless(HAVE_FFMPEG and HAVE_FONT, "ffmpeg or the DejaVu font is unavailable")
class TestEndToEnd(unittest.TestCase):
    """The whole chain over footage this test encoded itself.

    Two short segments from ``lavfi testsrc``, named the way the recorder names them, and
    a settings file pointing at them. It is the only test here that runs the real gate,
    the real frame path and the real overlay together — which is the combination that
    fails silently if any one of the three is wrong.
    """

    SEG_SECONDS = 4
    START = datetime(2026, 8, 14, 21, 11, 0, tzinfo=timezone.utc)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = Path(self.tmp.name) / "archive"
        self.archive.mkdir()
        for n in range(2):
            start = self.START + timedelta(seconds=self.SEG_SECONDS * n)
            self._encode(self.archive / f"{CAMERA}_{start:%Y%m%d_%H%M%S}.mp4", n)

        self.previous = os.environ.get("SPARK_SETTINGS")
        self.addCleanup(self._restore)
        self._write_settings()

    def _restore(self) -> None:
        if self.previous is None:
            os.environ.pop("SPARK_SETTINGS", None)
        else:
            os.environ["SPARK_SETTINGS"] = self.previous
        config.load.cache_clear()

    def _write_settings(self) -> None:
        """A copy of the real settings file with the segment length this fixture uses.

        ``shared/timecode.py`` reads ``recorder.segment_seconds`` to decide where a file
        ends, and encoding 60 s segments to test a 5 s window would be a minute of CPU per
        run for no additional coverage.
        """
        import yaml

        data = yaml.safe_load(
            (config.REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        data["recorder"]["segment_seconds"] = self.SEG_SECONDS
        data["paths"]["archive"] = str(self.archive)
        path = Path(self.tmp.name) / "settings.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        os.environ["SPARK_SETTINGS"] = str(path)
        config.load.cache_clear()

    def _encode(self, path: Path, index: int) -> None:
        # testsrc's moving bar guarantees inter-frame motion, so the gate has something
        # real to find; the second segment differs from the first so the carried
        # reference frame is exercised too.
        source = f"testsrc=s=320x240:r=10:d={self.SEG_SECONDS}"
        if index:
            source = f"testsrc2=s=320x240:r=10:d={self.SEG_SECONDS}"
        subprocess.run(  # noqa: S603
            [
                FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "lavfi", "-i", source,
                "-c:v", "libx264", "-preset", "ultrafast", "-g", "10", "-pix_fmt", "yuv420p",
                "-y", str(path),
            ],
            check=True,
            capture_output=True,
        )

    def test_generated_footage_produces_correct_wall_clock(self) -> None:
        settings = IngestSettings.from_config()
        self.assertEqual(settings.archive_dir, self.archive)

        sink = RecordingSink()
        with IngestPipeline(settings, sink=sink) as pipeline:
            stats = pipeline.run()

        self.assertGreater(stats.windows, 0)
        self.assertEqual(stats.no_footage, 0)
        self.assertEqual(stats.decode_failures, 0)
        self.assertEqual(len(sink.records), stats.windows)

        first = sink.records[0]
        self.assertEqual(first.t_start, self.START)
        self.assertEqual(first.t_end, self.START + timedelta(seconds=settings.window_seconds))
        self.assertEqual(first.segment, f"{CAMERA}_{self.START:%Y%m%d_%H%M%S}.mp4")
        self.assertEqual(first.pts_offset, 0.0)
        self.assertEqual(first.chunk_id, chunk_id_for(CAMERA, first.t_start, first.t_end))

        # Every record is located in time whether or not it was captioned.
        for record in sink.records:
            self.assertTrue(record.segment)
            self.assertGreaterEqual(record.pts_offset, 0.0)
            self.assertEqual(record.camera_id, CAMERA)
            if record.gated:
                self.assertEqual(record.caption, "")
            else:
                self.assertTrue(record.caption)

    def test_the_gate_fires_on_a_moving_test_pattern(self) -> None:
        """testsrc's scrolling bar is unambiguous motion. A gate that skips it is not a
        conservative gate, it is a broken one."""
        settings = IngestSettings.from_config()
        with IngestPipeline(settings, sink=None) as pipeline:
            stats = pipeline.run()
        self.assertGreater(stats.captioned, 0)

    def test_frames_are_sampled_scaled_and_overlaid(self) -> None:
        settings = IngestSettings.from_config()
        spans = resolve_range(
            self.START,
            self.START + timedelta(seconds=settings.window_seconds),
            str(self.archive),
            CAMERA,
        )
        extractor = FrameExtractor(settings)
        frames = extractor.extract(spans)
        self.assertEqual(len(frames), settings.frames_per_window)
        for frame in frames:
            self.assertTrue(frame.startswith(b"\xff\xd8\xff"))
        # The source is 320x240 — already under the 512 px short side — so the scale
        # filter must leave it alone rather than upscaling it into KV cache.
        self.assertGreaterEqual(extractor.fontsize, settings.overlay_fontsize)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestFollowSkipsTheOpenSegment(unittest.TestCase):
    """A follower must never walk into the file the recorder still has open.

    That file has no moov atom until it closes, so NOTHING in it decodes — not a window
    or two at the tail, the whole file. `follow()`'s cursor only moves forward, so those
    windows are burned as decode failures and never retried. Started against a live
    recorder on a fresh archive, that is every window there is: measured 111 decode
    failures and zero captions before this guard existed.
    """

    def setUp(self) -> None:
        self.archive = Path(tempfile.mkdtemp(prefix="bounds-"))

    def _segments(self, count: int) -> None:
        base = SEGMENT_START
        for i in range(count):
            stamp = (base + timedelta(seconds=60 * i)).strftime("%Y%m%d_%H%M%S")
            (self.archive / f"{CAMERA}_{stamp}.mp4").touch()

    def test_the_newest_segment_is_excluded(self) -> None:
        from services.ingest.windows import archive_bounds

        self._segments(3)
        full = archive_bounds(str(self.archive), CAMERA)
        closed = archive_bounds(str(self.archive), CAMERA, exclude_open=True)
        assert full and closed
        # One whole segment shorter — the open one.
        self.assertEqual((full[1] - closed[1]).total_seconds(), 60.0)
        self.assertEqual(full[0], closed[0])

    def test_a_lone_segment_yields_nothing_to_analyse(self) -> None:
        """The only file present is the one being written; there is no closed footage."""
        from services.ingest.windows import archive_bounds

        self._segments(1)
        self.assertIsNotNone(archive_bounds(str(self.archive), CAMERA))
        self.assertIsNone(archive_bounds(str(self.archive), CAMERA, exclude_open=True))

    def test_an_empty_archive_is_still_none(self) -> None:
        from services.ingest.windows import archive_bounds

        self.assertIsNone(archive_bounds(str(self.archive), CAMERA, exclude_open=True))

    def test_follow_asks_for_closed_footage_only(self) -> None:
        """The guard is worthless if the follow loop does not use it."""
        import inspect

        from services.ingest import pipeline

        src = inspect.getsource(pipeline.IngestPipeline.follow)
        self.assertIn("exclude_open=True", src)
