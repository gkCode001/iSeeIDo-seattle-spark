"""Tests for M4, the deep worker — SPEC §5, with the backstops of SPEC §4.3.

The claims this file exists to prove, in order of how badly they fail on stage:

* a user turn never blocks on ``deep_analyze`` (CLAUDE.md invariant 4) — ``submit``
  returns a ``QUEUED`` job before the archive is touched;
* an impatient double-click gets the *same* job, not two;
* one deep job runs at a time;
* the 90 s budget surfaces as ``JobState.TIMEOUT`` with a sentence, never as an exception
  escaping into a chat turn;
* a range that spans a segment boundary is stitched, and a hole in the archive is stated
  rather than swallowed (invariant 3);
* the decode never carries a scale filter (invariant 7) and always carries the wall-clock
  overlay (invariant 8);
* a stub answer is unmistakably marked as one.

Rules this file follows:

* **nothing shells out.** The decode plan and the clip plan are values, asserted as argv;
  the extractor and the cutter are injected fakes. ffmpeg is installed on this box now,
  but a test suite that decodes 1080p is a test suite nobody runs.
* **time is injected, never slept.** Concurrency is coordinated with ``threading.Event``,
  which is deterministic; a timeout test that sleeps is a flaky test with extra steps.
* the archive is a tempdir of empty placeholder files — ``shared/timecode.py`` reads only
  names, and real video would make these tests slow and the fixtures a liability.

Run with::

    python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared import config, timecode as tc
from shared.queue import Priority, VLMQueue
from shared.schema import DeepJob, JobState
from services.worker import (
    STUB_MARKER,
    AnalysisRequest,
    AnalysisResult,
    DeepWorker,
    StubAnalysisBackend,
    WorkerSettings,
    archive_resolver,
    build_decode_plan,
    derive_confidence,
    detect_hedge,
    drawtext_expansion,
    video_filter,
)
from services.worker.settings import PENDING_SETTINGS

# Numbers come from settings.yaml (or the worker's pending table), never from this file.
SEGMENT_SECONDS = float(config.get("recorder.segment_seconds"))
SAMPLE_FPS = float(config.get("vlm.profiles.deep.sample_fps"))
TIMEOUT_SECONDS = float(config.get("agent.deep.timeout_seconds"))
MAX_INFLIGHT = int(config.get("agent.deep.max_inflight"))
MAX_RANGE_SECONDS = float(
    config.get("agent.deep.max_range_seconds", PENDING_SETTINGS["agent.deep.max_range_seconds"])
)
HEDGED_FACTOR = float(
    config.get(
        "agent.deep.hedged_confidence_factor",
        PENDING_SETTINGS["agent.deep.hedged_confidence_factor"],
    )
)
CAMERA_ID = str(config.get("camera.id"))
OVERLAY_FORMAT = str(config.get("ingest.overlay.format"))

UTC = timezone.utc

#: A fixed instant, on a segment boundary, so every assertion below is reproducible.
T0 = datetime(2026, 8, 14, 21, 11, 0, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------------------


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
    """Deterministic job ids so a failure names the job that broke."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"job{self.n:03d}"


class RecordingExtractor:
    """Writes placeholder frame files instead of decoding, and keeps the plans it saw."""

    def __init__(self, *, yield_ratio: float = 1.0) -> None:
        self.plans: list[object] = []
        self.yield_ratio = yield_ratio

    def extract(self, plan):  # type: ignore[no-untyped-def]
        self.plans.append(plan)
        plan.out_dir.mkdir(parents=True, exist_ok=True)
        made: list[Path] = []
        for step in plan.steps:
            wanted = int(round(step.expected_frames * self.yield_ratio))
            for n in range(1, wanted + 1):
                path = plan.out_dir / f"s{step.span_index:03d}_{n:05d}.jpg"
                path.write_bytes(b"\xff\xd8\xff")  # a JPEG SOI marker, and nothing more
                made.append(path)
        return sorted(made)


class RecordingCutter:
    """Records the clip plan and returns the path it would have written. Cuts nothing."""

    def __init__(self) -> None:
        self.plans: list[object] = []

    def cut(self, plan):  # type: ignore[no-untyped-def]
        self.plans.append(plan)
        return str(plan.out_path)


class ScriptedBackend:
    """An analysis backend under the test's control.

    ``gate`` blocks the call until the test sets it, which is how "still running" is
    asserted without sleeping.
    """

    def __init__(
        self,
        *,
        answer: str = "A white van reverses toward the door at 21:11:19Z.",
        hedged: bool = False,
        is_stub: bool = False,
        error: BaseException | None = None,
    ) -> None:
        self.answer = answer
        self.hedged = hedged
        self._is_stub = is_stub
        self.error = error
        self.calls: list[AnalysisRequest] = []
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def is_stub(self) -> bool:
        return self._is_stub

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        self.calls.append(request)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(30)
        if self.error is not None:
            raise self.error
        return AnalysisResult(
            answer=self.answer,
            reasoning="scripted trace",
            hedged=self.hedged,
            is_stub=self._is_stub,
            model="scripted",
        )


class RecordingQueue(VLMQueue):
    """A real :class:`VLMQueue` that remembers the priority each job was submitted at."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.submitted: list[tuple[Priority, str]] = []

    def submit(self, priority, fn, *, label: str = ""):  # type: ignore[no-untyped-def]
        self.submitted.append((Priority(priority), label))
        return super().submit(priority, fn, label=label)


# --------------------------------------------------------------------------------------
# Base fixture
# --------------------------------------------------------------------------------------


class WorkerFixture(unittest.TestCase):
    """A tempdir archive of empty segment files, and a worker wired to fakes."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.archive = self.root / "archive"
        self.archive.mkdir()
        self.clips = self.root / "clips"
        self.frames_root = self.root / "frames"
        self.frames_root.mkdir()
        self.clock = FakeClock(T0)

    def write_segments(self, *starts: datetime) -> list[str]:
        names = []
        for start in starts:
            name = tc.segment_name_for(CAMERA_ID, start)
            (self.archive / name).write_bytes(b"")
            names.append(name)
        return names

    def settings(self, **overrides: object) -> WorkerSettings:
        base = WorkerSettings.from_config(archive_dir=self.archive)
        merged = {**base.__dict__, "clips_dir": self.clips, **overrides}
        return WorkerSettings(**merged)  # type: ignore[arg-type]

    def make_worker(
        self,
        *,
        backend: object | None = None,
        extractor: object | None = None,
        cutter: object | None = None,
        queue: VLMQueue | None = None,
        **setting_overrides: object,
    ) -> DeepWorker:
        self.backend = backend or ScriptedBackend()
        self.extractor = extractor or RecordingExtractor()
        self.cutter = cutter or RecordingCutter()
        self.queue = queue or RecordingQueue()
        self.queue.start()
        worker = DeepWorker(
            settings=self.settings(**setting_overrides),
            queue=self.queue,
            backend=self.backend,  # type: ignore[arg-type]
            extractor=self.extractor,  # type: ignore[arg-type]
            clip_cutter=self.cutter,  # type: ignore[arg-type]
            segment_resolver=archive_resolver(self.archive, CAMERA_ID),
            clock=self.clock,
            id_factory=SequentialIds(),
            frames_root=self.frames_root,
        )
        self.addCleanup(self.queue.stop)
        self.addCleanup(worker.shutdown, timeout=5.0)
        return worker

    def finished(self, worker: DeepWorker, job: DeepJob, timeout: float = 5.0) -> DeepJob:
        """Wait on real time for the worker thread, then apply the injected deadline."""
        event = worker._events[job.job_id]
        self.assertTrue(event.wait(timeout), f"job {job.job_id} never finished")
        return worker.poll(job)


# --------------------------------------------------------------------------------------
# The happy path — SPEC §5's contract
# --------------------------------------------------------------------------------------


class DeepAnalyzeTests(WorkerFixture):
    def test_returns_answer_clip_and_confidence(self) -> None:
        """``deep_analyze`` -> {answer, evidence_clip, confidence}, on a ``DeepJob``."""
        self.write_segments(T0)
        worker = self.make_worker()
        job = worker.analyze(T0 + timedelta(seconds=5), T0 + timedelta(seconds=15), "what?")

        self.assertIs(job.state, JobState.DONE)
        self.assertIn("white van", job.answer)
        self.assertEqual(job.reasoning, "scripted trace")
        self.assertIsNotNone(job.evidence_clip)
        self.assertEqual(job.confidence, 1.0)
        self.assertIsNone(job.error)
        self.assertIsNotNone(job.completed_at)

    def test_frames_are_decoded_at_the_deep_profile_rate(self) -> None:
        """4 fps over 10 s is 40 frames — SPEC §5, and the number comes from config."""
        self.write_segments(T0)
        worker = self.make_worker()
        job = worker.analyze(T0 + timedelta(seconds=5), T0 + timedelta(seconds=15), "what?")

        report = worker.report(job)
        assert report is not None
        self.assertEqual(report.sample_fps, SAMPLE_FPS)
        self.assertEqual(report.expected_frames, int(10 * SAMPLE_FPS))
        self.assertEqual(report.frames_decoded, int(10 * SAMPLE_FPS))
        self.assertEqual(len(self.backend.calls[0].frames), int(10 * SAMPLE_FPS))

    def test_native_resolution_is_not_negotiable(self) -> None:
        """Invariant 7: the deep path must refuse to be configured into a downscale."""
        root = config.load()
        deep = root["vlm"]["profiles"]["deep"]
        saved = deep["native_resolution"]
        deep["native_resolution"] = False
        try:
            with self.assertRaises(config.ConfigError):
                WorkerSettings.from_config(archive_dir=self.archive)
        finally:
            deep["native_resolution"] = saved

    def test_naive_datetimes_are_rejected(self) -> None:
        self.write_segments(T0)
        worker = self.make_worker()
        with self.assertRaises(ValueError):
            worker.submit(datetime(2026, 8, 14, 21, 11, 5), T0 + timedelta(seconds=15), "q")
        with self.assertRaises(ValueError):
            worker.submit(T0 + timedelta(seconds=15), T0 + timedelta(seconds=5), "q")
        with self.assertRaises(ValueError):
            worker.submit(T0, T0 + timedelta(seconds=5), "   ")

    def test_backend_failure_becomes_a_state_not_a_traceback(self) -> None:
        """An exception must never escape into a chat turn."""
        self.write_segments(T0)
        worker = self.make_worker(backend=ScriptedBackend(error=RuntimeError("vLLM died")))
        # Captured so the deliberate traceback does not land in the test output — the
        # worker logging it is the correct behaviour, and is asserted here.
        with self.assertLogs("services.worker", level="ERROR"):
            job = worker.analyze(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        self.assertIs(job.state, JobState.FAILED)
        self.assertIn("vLLM died", job.error or "")


# --------------------------------------------------------------------------------------
# Invariant 4 — a user turn never blocks
# --------------------------------------------------------------------------------------


class NonBlockingTests(WorkerFixture):
    def test_submit_returns_queued_immediately(self) -> None:
        self.write_segments(T0)
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend)
        self.addCleanup(gate.set)

        job = worker.submit(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        # The assertion that matters: the caller got a job id and nothing was awaited.
        self.assertIsInstance(job, DeepJob)
        self.assertIn(job.state, (JobState.QUEUED, JobState.RUNNING))
        self.assertEqual(job.answer, "")
        self.assertIsNone(job.confidence)

        self.assertTrue(backend.entered.wait(5))
        gate.set()
        done = self.finished(worker, job)
        self.assertIs(done.state, JobState.DONE)
        # The same object the caller kept: the refinement lands in place.
        self.assertIs(done, job)

    def test_state_is_written_after_the_payload(self) -> None:
        """Observing DONE implies answer/confidence/clip are already visible."""
        self.write_segments(T0)
        worker = self.make_worker()
        job = worker.analyze(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        self.assertIs(job.state, JobState.DONE)
        self.assertTrue(job.answer)
        self.assertIsNotNone(job.confidence)


# --------------------------------------------------------------------------------------
# Backstop 1 — one deep job in flight
# --------------------------------------------------------------------------------------


class InflightTests(WorkerFixture):
    def test_second_job_waits_for_the_only_slot(self) -> None:
        self.write_segments(T0)
        self.assertEqual(MAX_INFLIGHT, 1, "this test asserts the shipped configuration")
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend)
        self.addCleanup(gate.set)

        first = worker.submit(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q1")
        self.assertTrue(backend.entered.wait(5))

        # A different range, so dedupe cannot be what holds it back.
        second = worker.submit(T0 + timedelta(seconds=20), T0 + timedelta(seconds=25), "q2")
        self.assertIs(second.state, JobState.QUEUED)
        self.assertEqual(len(backend.calls), 1, "two deep jobs ran at once")

        gate.set()
        self.assertIs(self.finished(worker, first).state, JobState.DONE)
        self.assertIs(self.finished(worker, second).state, JobState.DONE)
        self.assertEqual(len(backend.calls), 2)


# --------------------------------------------------------------------------------------
# Backstop 2 — dedupe identical ranges
# --------------------------------------------------------------------------------------


class DedupeTests(WorkerFixture):
    def _blocked_worker(self):  # type: ignore[no-untyped-def]
        self.write_segments(T0)
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend)
        self.addCleanup(gate.set)
        return worker, backend, gate

    def test_double_click_returns_the_same_job(self) -> None:
        worker, backend, gate = self._blocked_worker()
        t0, t1 = T0 + timedelta(seconds=1), T0 + timedelta(seconds=6)

        first = worker.request(t0, t1, "was the door open?")
        self.assertTrue(backend.entered.wait(5))
        second = worker.request(t0, t1, "  Was  the DOOR open? ")

        self.assertTrue(second.deduped)
        self.assertIs(second.job, first.job)
        self.assertIn(first.job.job_id, second.detail)

        gate.set()
        self.finished(worker, first.job)
        self.assertEqual(len(backend.calls), 1, "the work was queued twice")
        report = worker.report(first.job)
        assert report is not None
        self.assertEqual(report.dedupe_hits, 1)

    def test_a_different_question_is_different_work(self) -> None:
        """Same range, different question. Handing back the first answer would be wrong."""
        worker, backend, gate = self._blocked_worker()
        t0, t1 = T0 + timedelta(seconds=1), T0 + timedelta(seconds=6)

        first = worker.request(t0, t1, "was the door open?")
        self.assertTrue(backend.entered.wait(5))
        second = worker.request(t0, t1, "how many people?")

        self.assertFalse(second.deduped)
        self.assertIsNot(second.job, first.job)
        gate.set()
        self.finished(worker, first.job)
        self.finished(worker, second.job)

    def test_dedupe_can_be_switched_off_from_config(self) -> None:
        self.write_segments(T0)
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend, dedupe_identical_ranges=False)
        self.addCleanup(gate.set)

        t0, t1 = T0 + timedelta(seconds=1), T0 + timedelta(seconds=6)
        first = worker.request(t0, t1, "q")
        second = worker.request(t0, t1, "q")
        self.assertFalse(second.deduped)
        self.assertIsNot(second.job, first.job)
        gate.set()

    def test_a_finished_job_does_not_dedupe_a_later_ask(self) -> None:
        """Dedupe is about work in flight, not a cache of stale answers."""
        self.write_segments(T0)
        worker = self.make_worker()
        t0, t1 = T0 + timedelta(seconds=1), T0 + timedelta(seconds=6)
        first = worker.analyze(t0, t1, "q")
        second = worker.request(t0, t1, "q")
        self.assertFalse(second.deduped)
        self.assertIsNot(second.job, first)


# --------------------------------------------------------------------------------------
# Backstop 3 — the 90 s timeout
# --------------------------------------------------------------------------------------


class TimeoutTests(WorkerFixture):
    def test_timeout_is_a_state_with_a_sentence(self) -> None:
        self.write_segments(T0)
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend)
        self.addCleanup(gate.set)

        job = worker.submit(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        self.assertTrue(backend.entered.wait(5))

        self.clock.advance(TIMEOUT_SECONDS + 1)
        polled = worker.poll(job)

        self.assertIs(polled.state, JobState.TIMEOUT)
        self.assertIn("timed out", (polled.error or "").lower())
        self.assertIn(f"{TIMEOUT_SECONDS:g}s", polled.error or "")
        self.assertEqual(polled.answer, "")
        gate.set()

    def test_a_late_result_does_not_overwrite_a_timeout(self) -> None:
        """The user was already told it timed out; changing the story is worse."""
        self.write_segments(T0)
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend)

        job = worker.submit(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        self.assertTrue(backend.entered.wait(5))
        self.clock.advance(TIMEOUT_SECONDS + 1)
        worker.poll(job)
        self.assertIs(job.state, JobState.TIMEOUT)

        gate.set()
        worker.shutdown(timeout=5.0)
        self.assertIs(job.state, JobState.TIMEOUT)
        self.assertEqual(job.answer, "")

    def test_a_job_that_never_gets_a_slot_times_out(self) -> None:
        self.write_segments(T0)
        gate = threading.Event()
        backend = ScriptedBackend()
        backend.gate = gate
        worker = self.make_worker(backend=backend)
        self.addCleanup(gate.set)

        first = worker.submit(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q1")
        self.assertTrue(backend.entered.wait(5))
        second = worker.submit(T0 + timedelta(seconds=20), T0 + timedelta(seconds=25), "q2")

        self.clock.advance(TIMEOUT_SECONDS + 1)
        self.assertIs(worker.poll(second).state, JobState.TIMEOUT)
        self.assertIs(worker.poll(first).state, JobState.TIMEOUT)
        gate.set()

    def test_elapsed_is_measured_from_the_request(self) -> None:
        """SPEC §11.2 prints elapsed against the 90 s budget; it starts when asked."""
        self.write_segments(T0)
        worker = self.make_worker()
        job = worker.submit(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        self.finished(worker, job)
        self.assertEqual(job.requested_at, T0)
        self.assertEqual(job.elapsed, 0.0, "the injected clock did not move")


# --------------------------------------------------------------------------------------
# Range length — SPEC §5's latency warning
# --------------------------------------------------------------------------------------


class RangeBudgetTests(WorkerFixture):
    def test_an_over_long_range_is_refused_not_truncated(self) -> None:
        self.write_segments(T0, T0 + timedelta(seconds=SEGMENT_SECONDS))
        worker = self.make_worker()
        submission = worker.request(T0, T0 + timedelta(seconds=MAX_RANGE_SECONDS + 30), "q")

        self.assertIs(submission.job.state, JobState.FAILED)
        self.assertFalse(submission.deduped)
        self.assertIn("max_range_seconds", submission.job.error or "")
        self.assertIn("Nothing was truncated", submission.job.error or "")
        self.assertEqual(len(self.backend.calls), 0)
        self.assertEqual(len(self.extractor.plans), 0)

    def test_a_range_at_the_cap_is_accepted(self) -> None:
        starts = [T0 + timedelta(seconds=SEGMENT_SECONDS * i) for i in range(3)]
        self.write_segments(*starts)
        worker = self.make_worker()
        job = worker.analyze(T0, T0 + timedelta(seconds=MAX_RANGE_SECONDS), "q")
        self.assertIs(job.state, JobState.DONE)


# --------------------------------------------------------------------------------------
# Invariant 3 — stitching, and holes that are stated rather than swallowed
# --------------------------------------------------------------------------------------


class StitchingTests(WorkerFixture):
    def test_a_boundary_spanning_range_is_stitched(self) -> None:
        second_start = T0 + timedelta(seconds=SEGMENT_SECONDS)
        names = self.write_segments(T0, second_start)
        worker = self.make_worker()

        t0 = second_start - timedelta(seconds=5)
        t1 = second_start + timedelta(seconds=5)
        job = worker.analyze(t0, t1, "what happened across the boundary?")

        self.assertIs(job.state, JobState.DONE)
        report = worker.report(job)
        assert report is not None
        self.assertEqual(report.segments, names, "both files must be read")
        self.assertEqual(report.gaps, [])
        self.assertAlmostEqual(report.covered_seconds, 10.0)

        # One decode invocation per span, and the frames arrive in wall-clock order.
        plan = self.extractor.plans[0]
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual([Path(s.source).name for s in plan.steps], names)
        frames = [p.name for p in self.backend.calls[0].frames]
        self.assertEqual(frames, sorted(frames))
        self.assertTrue(frames[0].startswith("s000_"))
        self.assertTrue(frames[-1].startswith("s001_"))

        # And the clip is a concat of both parts, not a single cut of one.
        clip_plan = self.cutter.plans[0]
        self.assertEqual(len(clip_plan.slices), 2)
        self.assertIsNotNone(clip_plan.concat_list_path)

    def test_a_hole_is_reported_in_the_answer_and_in_the_confidence(self) -> None:
        """Invariant 3: a short answer must never masquerade as a complete one."""
        # Two segments with a whole segment's worth of time missing between them.
        second_start = T0 + timedelta(seconds=SEGMENT_SECONDS * 2)
        self.write_segments(T0, second_start)
        worker = self.make_worker()

        t0 = T0 + timedelta(seconds=SEGMENT_SECONDS - 5)
        t1 = T0 + timedelta(seconds=SEGMENT_SECONDS + 5)
        job = worker.analyze(t0, t1, "q")

        self.assertIs(job.state, JobState.DONE)
        self.assertIn("[worker note]", job.answer)
        self.assertIn("never recorded", job.answer)
        report = worker.report(job)
        assert report is not None
        self.assertEqual(len(report.gaps), 1)
        self.assertAlmostEqual(report.gap_seconds, 5.0)
        self.assertAlmostEqual(report.covered_seconds, 5.0)
        # Half the range existed, so at most half the confidence.
        self.assertAlmostEqual(job.confidence or 0.0, 0.5)

    def test_a_range_with_no_footage_at_all_fails_with_a_sentence(self) -> None:
        self.write_segments(T0)
        worker = self.make_worker()
        far = T0 + timedelta(days=1)
        job = worker.analyze(far, far + timedelta(seconds=10), "q")

        self.assertIs(job.state, JobState.FAILED)
        self.assertIn("no footage", (job.error or "").lower())
        self.assertIsNone(job.evidence_clip)
        self.assertEqual(len(self.backend.calls), 0)


# --------------------------------------------------------------------------------------
# The decode plan — invariants 7 and 8, asserted as argv
# --------------------------------------------------------------------------------------


class DecodePlanTests(WorkerFixture):
    def _plan(self, **overrides: object):  # type: ignore[no-untyped-def]
        self.write_segments(T0)
        settings = self.settings(**overrides)
        spans = tc.resolve_range(
            T0 + timedelta(seconds=5),
            T0 + timedelta(seconds=15),
            archive_dir=self.archive,
        )
        return build_decode_plan(spans, self.root / "out", settings=settings), settings

    def test_no_scale_filter_ever(self) -> None:
        """Invariant 7. The deep path exists to read detail; a resize here is the bug."""
        plan, _ = self._plan()
        for step in plan.steps:
            joined = " ".join(step.argv)
            self.assertNotIn("scale", joined)
            self.assertNotIn("-s ", joined)

    def test_samples_at_the_configured_deep_rate(self) -> None:
        plan, _ = self._plan()
        vf = plan.steps[0].argv[plan.steps[0].argv.index("-vf") + 1]
        self.assertTrue(vf.startswith(f"fps={SAMPLE_FPS:g},"))

    def test_seeks_by_pts_and_carries_wall_clock(self) -> None:
        """Invariant 2: the PTS offset locates the pixels, the wall clock names the moment."""
        plan, _ = self._plan()
        step = plan.steps[0]
        argv = list(step.argv)
        self.assertEqual(argv[argv.index("-ss") + 1], "5.000")
        self.assertEqual(argv[argv.index("-t") + 1], "10.000")
        # -ss precedes -i so the seek is a fast one.
        self.assertLess(argv.index("-ss"), argv.index("-i"))
        self.assertEqual(step.wall_start, T0 + timedelta(seconds=5))

    def test_overlay_is_burned_with_absolute_utc(self) -> None:
        """Invariant 8. The VLM reads this for temporal localization."""
        plan, settings = self._plan()
        vf = plan.steps[0].argv[plan.steps[0].argv.index("-vf") + 1]
        self.assertIn("drawtext=", vf)
        # fps first, drawtext second: burn onto the frames we keep, not the ones we drop.
        self.assertLess(vf.index("fps="), vf.index("drawtext="))
        epoch = int((T0 + timedelta(seconds=5)).timestamp())
        self.assertIn(f"gmtime\\:{epoch}\\:", vf)
        self.assertIn(settings.overlay_fontfile, vf)

    def test_drawtext_escaping_is_exact(self) -> None:
        """One backslash for the argument separators, three for the format's own colons.

        Verified against ffmpeg 6.1.1: with one, ffmpeg reports "%{pts} requires at most 3
        arguments"; with two, "Stray %". Neither fails the job — they render garbage into
        the frame and localization degrades silently, which is invariant 8's whole warning.
        """
        text = drawtext_expansion(T0, OVERLAY_FORMAT)
        self.assertEqual(
            text,
            "%{pts\\:gmtime\\:" + str(int(T0.timestamp())) + "\\:"
            + OVERLAY_FORMAT.replace(":", "\\\\\\:")
            + "}",
        )
        self.assertNotIn("\\\\\\\\", text)

    def test_overlay_can_be_switched_off(self) -> None:
        chain = video_filter(T0, self.settings(overlay_enabled=False))
        self.assertEqual(chain, f"fps={SAMPLE_FPS:g}")

    def test_gap_spans_are_not_decoded(self) -> None:
        self.write_segments(T0)
        spans = tc.resolve_range(
            T0 + timedelta(seconds=SEGMENT_SECONDS - 5),
            T0 + timedelta(seconds=SEGMENT_SECONDS + 5),
            archive_dir=self.archive,
        )
        self.assertTrue(any(s.is_gap for s in spans))
        plan = build_decode_plan(spans, self.root / "out", settings=self.settings())
        self.assertEqual(len(plan.steps), 1)

    def test_an_entirely_missing_range_has_no_plan(self) -> None:
        self.write_segments(T0)
        far = T0 + timedelta(days=1)
        spans = tc.resolve_range(far, far + timedelta(seconds=10), archive_dir=self.archive)
        with self.assertRaises(ValueError):
            build_decode_plan(spans, self.root / "out", settings=self.settings())


# --------------------------------------------------------------------------------------
# Priority — SPEC §7. The caller says which lane it is in.
# --------------------------------------------------------------------------------------


class PriorityTests(WorkerFixture):
    def test_interactive_by_default_and_verification_when_asked(self) -> None:
        self.write_segments(T0)
        worker = self.make_worker()
        t0 = T0 + timedelta(seconds=1)

        worker.analyze(t0, t0 + timedelta(seconds=5), "user is waiting")
        worker.analyze(
            t0 + timedelta(seconds=10),
            t0 + timedelta(seconds=15),
            "m5 is verifying",
            priority=Priority.VERIFICATION,
        )
        lanes = [p for p, _ in self.queue.submitted]
        self.assertEqual(lanes, [Priority.INTERACTIVE, Priority.VERIFICATION])
        self.assertTrue(all(label.startswith("deep:") for _, label in self.queue.submitted))

    def test_the_vlm_is_only_reached_through_the_queue(self) -> None:
        """Invariant 1: one VLM process, and everything queues in front of it."""
        self.write_segments(T0)
        worker = self.make_worker()
        job = worker.analyze(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        report = worker.report(job)
        assert report is not None
        self.assertIsNotNone(report.queue_job_id)
        self.assertEqual(len(self.queue.submitted), 1)


# --------------------------------------------------------------------------------------
# Confidence — a heuristic, and it must behave like one
# --------------------------------------------------------------------------------------


class ConfidenceTests(unittest.TestCase):
    def _confidence(self, **overrides: object) -> float:
        kwargs: dict[str, object] = {
            "requested_seconds": 10.0,
            "covered_seconds": 10.0,
            "frames_decoded": 40,
            "expected_frames": 40,
            "hedged": False,
            "hedged_factor": HEDGED_FACTOR,
        }
        kwargs.update(overrides)
        return derive_confidence(**kwargs)  # type: ignore[arg-type]

    def test_full_coverage_unhedged_is_one(self) -> None:
        self.assertEqual(self._confidence(), 1.0)

    def test_a_hole_costs_its_share(self) -> None:
        self.assertAlmostEqual(self._confidence(covered_seconds=5.0, expected_frames=20), 0.5)

    def test_a_short_decode_costs_its_share(self) -> None:
        self.assertAlmostEqual(self._confidence(frames_decoded=10), 0.25)

    def test_no_frames_is_no_confidence(self) -> None:
        """An answer produced without looking at anything cannot carry a number."""
        self.assertEqual(self._confidence(frames_decoded=0), 0.0)
        self.assertEqual(self._confidence(expected_frames=0, frames_decoded=0), 0.0)

    def test_hedging_applies_the_configured_factor(self) -> None:
        self.assertAlmostEqual(self._confidence(hedged=True), HEDGED_FACTOR)

    def test_it_never_leaves_the_unit_interval(self) -> None:
        self.assertEqual(self._confidence(covered_seconds=99.0, frames_decoded=999), 1.0)

    def test_an_empty_range_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            self._confidence(requested_seconds=0.0)

    def test_hedge_detection_reads_the_model_at_its_word(self) -> None:
        self.assertTrue(detect_hedge("The rear door is not visible from this angle."))
        self.assertTrue(detect_hedge("I cannot determine whether anyone approached."))
        self.assertFalse(detect_hedge("Yes. The rear doors are open from 21:11:19 onward."))


# --------------------------------------------------------------------------------------
# The stub backend — how the escalation path is demonstrated while D1 is open
# --------------------------------------------------------------------------------------


class StubBackendTests(WorkerFixture):
    def _request(self, frames: int = 3) -> AnalysisRequest:
        return AnalysisRequest(
            chunk_id="job001",
            question="Was the rear door open?",
            t_start=T0,
            t_end=T0 + timedelta(seconds=10),
            frames=tuple(Path(f"/frames/{i}.jpg") for i in range(frames)),
            segments=("cam01_20260814_211100.mp4",),
            covered_seconds=10.0,
            gap_seconds=0.0,
        )

    def test_every_stub_answer_is_marked_as_one(self) -> None:
        result = StubAnalysisBackend().analyze(self._request())
        self.assertTrue(result.is_stub)
        self.assertIn(STUB_MARKER, result.answer)
        self.assertIn(STUB_MARKER, result.reasoning)
        self.assertIn("No model read these pixels", result.answer)

    def test_a_stub_answer_can_never_look_confident(self) -> None:
        """It is asserted, not sniffed: a synthetic answer is not grounded in footage."""
        self.assertTrue(StubAnalysisBackend().analyze(self._request()).hedged)

    def test_it_is_deterministic(self) -> None:
        a = StubAnalysisBackend().analyze(self._request())
        b = StubAnalysisBackend().analyze(self._request())
        self.assertEqual(a.answer, b.answer)

    def test_it_reports_what_the_pipeline_actually_did(self) -> None:
        result = StubAnalysisBackend().analyze(self._request(frames=40))
        self.assertIn("40 frames", result.answer)
        self.assertIn("cam01_20260814_211100.mp4", result.answer)

    def test_end_to_end_through_the_worker(self) -> None:
        """The escalation path M3 and M5 exercise today, start to finish."""
        self.write_segments(T0)
        worker = self.make_worker(backend=StubAnalysisBackend())
        job = worker.analyze(T0 + timedelta(seconds=5), T0 + timedelta(seconds=15), "q")

        self.assertIs(job.state, JobState.DONE)
        self.assertIn(STUB_MARKER, job.answer)
        self.assertIsNotNone(job.evidence_clip)
        # Full coverage, full decode, but asserted-hedged: the coverage heuristic lands on
        # the hedged factor and never on something that reads as certainty.
        self.assertAlmostEqual(job.confidence or 0.0, HEDGED_FACTOR)
        report = worker.report(job)
        assert report is not None
        self.assertTrue(report.is_stub)
        self.assertIn("hedged", report.confidence_detail)


# --------------------------------------------------------------------------------------
# The report — the audit trail behind the number
# --------------------------------------------------------------------------------------


class ReportTests(WorkerFixture):
    def test_the_report_explains_the_confidence(self) -> None:
        self.write_segments(T0)
        worker = self.make_worker(extractor=RecordingExtractor(yield_ratio=0.5))
        job = worker.analyze(T0 + timedelta(seconds=5), T0 + timedelta(seconds=15), "q")

        report = worker.report(job)
        assert report is not None
        self.assertEqual(report.frames_decoded, report.expected_frames // 2)
        self.assertAlmostEqual(job.confidence or 0.0, 0.5)
        self.assertIn("coverage", report.confidence_detail)
        self.assertIn("decode", report.confidence_detail)
        payload = report.to_dict()
        self.assertEqual(payload["job_id"], job.job_id)
        self.assertEqual(payload["native_resolution"], True)
        self.assertEqual(payload["sample_fps"], SAMPLE_FPS)

    def test_stats_count_the_backstops(self) -> None:
        self.write_segments(T0)
        worker = self.make_worker()
        worker.analyze(T0 + timedelta(seconds=1), T0 + timedelta(seconds=6), "q")
        stats = worker.stats()
        self.assertEqual(stats["submitted"], 1)
        self.assertEqual(stats["done"], 1)
        self.assertEqual(stats["max_inflight"], MAX_INFLIGHT)
        self.assertEqual(stats["timeout_seconds"], TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
