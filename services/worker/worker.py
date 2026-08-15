"""M4 — the deep worker. One entry point, shared by M3 and M5 (SPEC §5).

    deep_analyze(t_start, t_end, question) -> DeepJob   # {answer, evidence_clip, confidence}

The job it does, in order:

1. Resolve the wall-clock range to segment files, **stitching across boundaries** and
   never taking a filename (CLAUDE.md invariant 3). ``shared/timecode.py`` owns that
   derivation; holes come back as gap spans and are reported, never swallowed.
2. Cut the evidence clip. Deliberately *before* the analysis: a copy-cut is milliseconds,
   and a job that times out at 90 s should still hand the user the footage it was looking
   at rather than nothing at all.
3. Re-decode at 4 fps, native resolution (``vlm.profiles.deep``). No resize, ever —
   invariant 7. The wall-clock overlay is burned onto the sampled frames so the model can
   localize in time (invariant 8); see ``decode.py``.
4. One deep VLM request, through ``shared/queue.py`` at the caller's priority —
   ``INTERACTIVE`` when a human is waiting on M3, ``VERIFICATION`` when M5 is checking an
   alert it has already fired (SPEC §7). The profile is the client's business, not ours.
5. A confidence number, derived and explained in ``analysis.derive_confidence``. It is a
   coverage heuristic. It is not the model's opinion and it is not a probability.

Never blocking a user turn — CLAUDE.md invariant 4
--------------------------------------------------
:meth:`DeepWorker.submit` returns a ``DeepJob`` in state ``QUEUED`` **immediately**, before
any archive is touched. M3 answers provisionally, hands the UI the ``job_id``, and streams
the refinement when the state reaches ``DONE`` (SPEC §4.3, §11.2). :func:`deep_analyze` is
the blocking convenience form over the same machinery, for callers already on a background
thread — M5's stage 3, and scripts.

The ``DeepJob`` is **mutated in place**, so a caller that kept the object from ``submit``
sees the refinement without a lookup. Fields are written under the worker's lock and
``state`` is written *last*, after ``answer``, ``reasoning``, ``confidence`` and
``evidence_clip`` — so a reader that observes ``state is DONE`` is guaranteed to see a
fully populated job, with no lock of its own.

The three backstops — SPEC §4.3
-------------------------------
* **One in flight.** ``agent.deep.max_inflight`` slots, held by a semaphore. A job that
  cannot get one stays ``QUEUED`` and waits; it is never dropped, and its 90 s budget runs
  from ``requested_at``, so queueing can never hide behind the timer the UI prints.
* **Dedupe.** ``agent.deep.dedupe_identical_ranges``. An impatient user clicking twice gets
  the *same* ``DeepJob`` back, and :class:`Submission` says so, because §11.2 wants
  "already running — job 7f3a" on screen: a silent no-op reads as a bug in rehearsal. The
  key is the range **and** the question — same range, different question is genuinely
  different work, and handing back the first job's answer would be worse than doing the
  work twice.
* **Timeout.** ``agent.deep.timeout_seconds`` (90), surfaced as ``JobState.TIMEOUT`` with a
  sentence in ``error``. It is enforced at every observation point (:meth:`poll`,
  :meth:`wait`, :func:`deep_analyze`) as well as between stages inside the job, so a caller
  that only polls still gets the truth. A terminal state is final: a result that arrives
  after the deadline is logged and dropped, because the user has already been told it timed
  out and changing the story afterwards is worse than losing the answer.

None of these raise. A caller asking a question gets a ``DeepJob`` describing what
happened, always — an exception escaping into a chat turn is a demo that ends.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shared import config, timecode
from shared.queue import Priority, QueueTimeout, VLMQueue
from shared.schema import DeepJob, JobState, to_iso, utcnow
from shared.timecode import SegmentSpan
from services.mcp.clips import (
    ClipCutter,
    FfmpegClipCutter,
    NullClipCutter,
    SegmentResolver,
    SegmentSlice,
    build_clip_plan,
    clip_path_for,
    timecode_segment_resolver,
)

from .analysis import (
    AnalysisBackend,
    AnalysisRequest,
    build_analysis_backend,
    confidence_explanation,
    derive_confidence,
    segments_of,
)
from .decode import (
    DecodePlan,
    FfmpegFrameExtractor,
    FrameExtractor,
    build_decode_plan,
    frames_for_seconds,
)
from .settings import WorkerSettings

__all__ = [
    "DeepReport",
    "Submission",
    "DeepWorker",
    "archive_resolver",
    "deep_analyze",
    "submit",
    "default_worker",
    "set_default_worker",
    "ffmpeg_worker_from_config",
]

logger = logging.getLogger("services.worker")

#: States from which a job never moves again. Reached exactly once, by one writer.
TERMINAL_STATES = frozenset({JobState.DONE, JobState.TIMEOUT, JobState.FAILED})


class _Deadline(Exception):
    """Internal: the 90 s budget ran out mid-job. Becomes ``JobState.TIMEOUT``."""


class _Refusal(Exception):
    """Internal: the job cannot be done at all. Becomes ``JobState.FAILED``."""


# --------------------------------------------------------------------------------------
# What the job did — inspectable, and the audit trail for the confidence number
# --------------------------------------------------------------------------------------


@dataclass
class DeepReport:
    """Everything about a job that does not belong on the shared ``DeepJob`` contract.

    ``DeepJob`` is the wire shape M3, M5 and the UI agree on (``shared/schema.py``), and it
    is not M4's to extend. This is where the worker keeps its own account of what it read —
    which segments, how much of the range existed, how many frames came out, and the
    arithmetic behind ``confidence``. The UI can render it beside the answer; a human
    debugging "why is this 0.35?" reads ``confidence_detail`` and gets the whole story.
    """

    job_id: str
    priority: str
    backend: str
    is_stub: bool
    deadline: datetime
    requested_seconds: float
    sample_fps: float
    native_resolution: bool = True
    covered_seconds: float = 0.0
    gap_seconds: float = 0.0
    gaps: list[tuple[str, str]] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    expected_frames: int = 0
    frames_decoded: int = 0
    hedged: bool | None = None
    confidence: float | None = None
    confidence_detail: str = ""
    evidence_clip: str | None = None
    queue_job_id: str | None = None
    dedupe_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "priority": self.priority,
            "backend": self.backend,
            "is_stub": self.is_stub,
            "deadline": to_iso(self.deadline),
            "requested_seconds": round(self.requested_seconds, 3),
            "sample_fps": self.sample_fps,
            "native_resolution": self.native_resolution,
            "covered_seconds": round(self.covered_seconds, 3),
            "gap_seconds": round(self.gap_seconds, 3),
            "gaps": [list(g) for g in self.gaps],
            "segments": list(self.segments),
            "expected_frames": self.expected_frames,
            "frames_decoded": self.frames_decoded,
            "hedged": self.hedged,
            "confidence": self.confidence,
            "confidence_detail": self.confidence_detail,
            "evidence_clip": self.evidence_clip,
            "queue_job_id": self.queue_job_id,
            "dedupe_hits": self.dedupe_hits,
        }


@dataclass(frozen=True)
class Submission:
    """The result of asking for a deep analysis.

    ``deduped`` is not decoration: SPEC §11.2 lists "already running — job 7f3a" as a
    required element of the Ask pane, precisely because a dedupe that looks like nothing
    happening gets reported as a bug during rehearsal.
    """

    job: DeepJob
    deduped: bool
    detail: str = ""


# --------------------------------------------------------------------------------------
# Clip resolution
# --------------------------------------------------------------------------------------


def archive_resolver(archive_dir: Path, camera_id: str | None = None) -> SegmentResolver:
    """``timecode_segment_resolver`` bound to a non-default archive directory.

    The production worker uses ``timecode_segment_resolver`` itself, which reads
    ``paths.archive``. This exists for the two cases that cannot: a test with a tempdir
    archive, and a hand-run of the worker against footage that lives somewhere else.
    Same semantics — gaps are dropped from the clip with a warning naming the hole, so the
    clip holds the footage that genuinely exists and the shortfall is stated, never
    implied.
    """

    def resolve(t_start: datetime, t_end: datetime) -> list[SegmentSlice]:
        slices: list[SegmentSlice] = []
        for span in timecode.resolve_range(
            t_start, t_end, archive_dir=archive_dir, camera_id=camera_id
        ):
            if span.is_gap or span.path is None:
                logger.warning(
                    "archive gap inside a clip range; that footage was never recorded",
                    extra={
                        "fields": {
                            "gap_start": to_iso(span.t_start),
                            "gap_end": to_iso(span.t_end),
                            "gap_seconds": span.duration,
                        }
                    },
                )
                continue
            slices.append(
                SegmentSlice(
                    path=str(span.path),
                    seek_seconds=span.pts_in,
                    duration_seconds=span.duration,
                )
            )
        return slices

    return resolve


# --------------------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------------------


class DeepWorker:
    """The deep path. Construct one per process; M3 and M5 share it.

    Every collaborator is injectable, and for one reason each: the queue so that M4 shares
    the *process's* single in-flight VLM budget rather than opening a second one
    (invariant 1); the backend so ``vlm.backend`` can be stub today and vLLM tomorrow; the
    extractor and cutter so the test suite never shells out; the clock so timeout tests are
    arithmetic rather than a race.
    """

    def __init__(
        self,
        *,
        settings: WorkerSettings | None = None,
        queue: VLMQueue | None = None,
        backend: AnalysisBackend | None = None,
        extractor: FrameExtractor | None = None,
        clip_cutter: ClipCutter | None = None,
        segment_resolver: SegmentResolver | None = None,
        archive_dir: str | Path | None = None,
        clock: Callable[[], datetime] = utcnow,
        id_factory: Callable[[], str] | None = None,
        frames_root: str | Path | None = None,
        keep_frames: bool = False,
    ) -> None:
        self._settings = settings or WorkerSettings.from_config(archive_dir=archive_dir)
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex[:8])
        self._keep_frames = keep_frames
        self._frames_root = Path(frames_root) if frames_root is not None else None

        # A queue we were given belongs to the process; one we build belongs to us and is
        # ours to stop. Either way there is exactly one, because there is exactly one VLM.
        self._owns_queue = queue is None
        self._queue = queue or VLMQueue()
        if self._owns_queue:
            self._queue.start()

        self._backend = backend or build_analysis_backend(self._settings)
        self._extractor: FrameExtractor = extractor or FfmpegFrameExtractor(
            ffmpeg_bin=self._settings.ffmpeg_bin,
            timeout_seconds=self._settings.decode_timeout_seconds,
        )
        self._cutter: ClipCutter = clip_cutter or NullClipCutter()
        if segment_resolver is not None:
            self._resolver: SegmentResolver = segment_resolver
        elif archive_dir is None:
            self._resolver = timecode_segment_resolver
        else:
            self._resolver = archive_resolver(Path(archive_dir), self._settings.camera_id)

        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self._settings.max_inflight)
        self._jobs: dict[str, DeepJob] = {}
        self._reports: dict[str, DeepReport] = {}
        self._events: dict[str, threading.Event] = {}
        #: Live jobs by dedupe key. Entries are removed the moment a job goes terminal, so
        #: a repeat click after an answer lands starts fresh work rather than replaying a
        #: stale one.
        self._live: dict[tuple[str, str, str], DeepJob] = {}
        self._threads: list[threading.Thread] = []
        self._counters = {"submitted": 0, "deduped": 0, "done": 0, "timeout": 0, "failed": 0}

    # -- properties -------------------------------------------------------------------

    @property
    def settings(self) -> WorkerSettings:
        return self._settings

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def is_stub(self) -> bool:
        return self._backend.is_stub

    # ----------------------------------------------------------------------------------
    # Submission — SPEC §5's entry point, in its two forms
    # ----------------------------------------------------------------------------------

    def request(
        self,
        t_start: datetime,
        t_end: datetime,
        question: str,
        *,
        priority: Priority | str = Priority.INTERACTIVE,
    ) -> Submission:
        """Queue a deep analysis. Returns immediately — invariant 4.

        The returned job is ``QUEUED`` and nothing has touched the archive yet. Use
        :meth:`submit` if you do not need the dedupe flag.
        """
        t0, t1 = self._validate(t_start, t_end, question)
        prio = Priority(priority)
        requested_seconds = (t1 - t0).total_seconds()
        key = self._dedupe_key(t0, t1, question)

        with self._lock:
            if self._settings.dedupe_identical_ranges:
                existing = self._live.get(key)
                if existing is not None and existing.state not in TERMINAL_STATES:
                    self._counters["deduped"] += 1
                    report = self._reports.get(existing.job_id)
                    if report is not None:
                        report.dedupe_hits += 1
                    detail = f"already running — job {existing.job_id}"
                    self._emit(
                        "deep_deduped",
                        job_id=existing.job_id,
                        state=existing.state.value,
                        t_start=to_iso(t0),
                        t_end=to_iso(t1),
                    )
                    return Submission(existing, True, detail)

            job = DeepJob(
                job_id=self._id_factory(),
                t_start=t0,
                t_end=t1,
                question=question,
                state=JobState.QUEUED,
                requested_at=self._clock(),
            )
            report = DeepReport(
                job_id=job.job_id,
                priority=prio.value,
                backend=self._backend.name,
                is_stub=self._backend.is_stub,
                deadline=self._deadline(job),
                requested_seconds=requested_seconds,
                sample_fps=self._settings.sample_fps,
                native_resolution=self._settings.native_resolution,
            )
            self._register(job, report)
            self._counters["submitted"] += 1

            # The one refusal that happens on the caller's thread, because it is pure
            # arithmetic and because saying it late would be saying it quietly.
            if requested_seconds > self._settings.max_range_seconds:
                detail = self._too_long_message(requested_seconds)
                self._finish_locked(job, JobState.FAILED, error=detail)
                self._emit(
                    "deep_refused",
                    job_id=job.job_id,
                    requested_seconds=round(requested_seconds, 3),
                    max_range_seconds=self._settings.max_range_seconds,
                )
                return Submission(job, False, detail)

            self._live[key] = job
            # Finished threads are dropped rather than accumulated; ``shutdown`` joins
            # whatever is genuinely still running.
            self._threads = [t for t in self._threads if t.is_alive()]
            thread = threading.Thread(
                target=self._run,
                args=(job, report, prio, key),
                name=f"deep-{job.job_id}",
                daemon=True,
            )
            self._threads.append(thread)

        self._emit(
            "deep_submitted",
            job_id=job.job_id,
            priority=prio.value,
            t_start=to_iso(t0),
            t_end=to_iso(t1),
            requested_seconds=round(requested_seconds, 3),
            timeout_seconds=self._settings.timeout_seconds,
            backend=self._backend.name,
        )
        thread.start()
        return Submission(job, False, "")

    def submit(
        self,
        t_start: datetime,
        t_end: datetime,
        question: str,
        *,
        priority: Priority | str = Priority.INTERACTIVE,
    ) -> DeepJob:
        """The async form. Returns a ``QUEUED`` ``DeepJob`` without blocking — invariant 4."""
        return self.request(t_start, t_end, question, priority=priority).job

    def analyze(
        self,
        t_start: datetime,
        t_end: datetime,
        question: str,
        *,
        priority: Priority | str = Priority.INTERACTIVE,
        timeout: float | None = None,
    ) -> DeepJob:
        """The blocking form — SPEC §5's ``deep_analyze``. **Never call this on a user turn.**

        Correct callers are already on a background thread: M5's stage-3 verification, and
        scripts. M3 uses :meth:`submit` and streams the refinement (SPEC §4.3).
        """
        return self.wait(self.submit(t_start, t_end, question, priority=priority), timeout)

    # ----------------------------------------------------------------------------------
    # Observation — where the timeout is surfaced to a caller that never waits
    # ----------------------------------------------------------------------------------

    def poll(self, job: DeepJob | str) -> DeepJob:
        """Current state of a job, with the 90 s deadline applied.

        A UI polling on ``ui.poll_interval_ms`` calls this. It is what turns "the job is
        still RUNNING and the box is wedged" into ``TIMEOUT`` with a sentence, rather than
        a spinner that never stops.
        """
        resolved = self._job_for(job)
        with self._lock:
            self._apply_deadline_locked(resolved)
        return resolved

    def wait(self, job: DeepJob | str, timeout: float | None = None) -> DeepJob:
        """Block until the job is terminal or its budget is spent. Never raises."""
        resolved = self._job_for(job)
        event = self._events.get(resolved.job_id)
        budget = timeout if timeout is not None else self._settings.timeout_seconds
        if event is not None:
            event.wait(budget)
        return self.poll(resolved)

    def job(self, job_id: str) -> DeepJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def report(self, job: DeepJob | str) -> DeepReport | None:
        job_id = job if isinstance(job, str) else job.job_id
        with self._lock:
            return self._reports.get(job_id)

    def jobs(self) -> list[DeepJob]:
        with self._lock:
            return list(self._jobs.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._counters,
                "live": len(self._live),
                "backend": self._backend.name,
                "is_stub": self._backend.is_stub,
                "max_inflight": self._settings.max_inflight,
                "timeout_seconds": self._settings.timeout_seconds,
                "max_range_seconds": self._settings.max_range_seconds,
            }

    def shutdown(self, *, timeout: float | None = 5.0) -> None:
        """Stop the queue we own and join whatever is still running."""
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout)
        if self._owns_queue:
            self._queue.stop(drain=False)

    # ----------------------------------------------------------------------------------
    # The job itself
    # ----------------------------------------------------------------------------------

    def _run(
        self,
        job: DeepJob,
        report: DeepReport,
        priority: Priority,
        key: tuple[str, str, str],
    ) -> None:
        acquired = False
        frames_dir: Path | None = None
        try:
            # ``max_inflight`` is enforced here rather than by rejecting: work is never
            # dropped, it waits — and it waits against the same 90 s the user is watching,
            # so a queued job cannot hide behind the timer.
            remaining = self._remaining(job)
            acquired = remaining > 0 and self._slots.acquire(timeout=remaining)
            if not acquired:
                raise _Deadline("waiting for the single in-flight deep slot")
            frames_dir = self._execute(job, report, priority)
        except _Deadline as exc:
            self._finish(job, JobState.TIMEOUT, error=self._timeout_message(job, str(exc)))
        except _Refusal as exc:
            self._finish(job, JobState.FAILED, error=str(exc))
        except BaseException as exc:  # noqa: BLE001 - a failed job is a state, not a traceback
            logger.exception("deep job failed", extra={"fields": {"job_id": job.job_id}})
            self._finish(job, JobState.FAILED, error=f"{type(exc).__name__}: {exc}")
        finally:
            if acquired:
                self._slots.release()
            with self._lock:
                if self._live.get(key) is job:
                    del self._live[key]
            if frames_dir is not None and not self._keep_frames:
                shutil.rmtree(frames_dir, ignore_errors=True)

    def _execute(self, job: DeepJob, report: DeepReport, priority: Priority) -> Path | None:
        with self._lock:
            if job.state in TERMINAL_STATES:
                return None
            job.state = JobState.RUNNING
        self._check_deadline(job, "starting")

        # 1. Resolve — invariant 3. A range is never a filename, and an event that starts
        #    at 21:11:58 and runs 12 s is two files.
        spans = timecode.resolve_range(
            job.t_start,
            job.t_end,
            archive_dir=self._settings.archive_dir,
            camera_id=self._settings.camera_id,
        )
        self._record_coverage(job, report, spans)

        # 2. Evidence clip, before the expensive part. See the module docstring.
        self._check_deadline(job, "resolving the range")
        clip = self._cut_clip(job)
        with self._lock:
            job.evidence_clip = clip
        report.evidence_clip = clip

        # 3. Decode — 4 fps, native resolution, overlay burned. Invariants 7 and 8.
        self._check_deadline(job, "cutting the evidence clip")
        frames_dir = Path(tempfile.mkdtemp(prefix=f"deep-{job.job_id}-", dir=self._frames_root))
        plan = build_decode_plan(spans, frames_dir, settings=self._settings)
        report.expected_frames = plan.expected_frames
        frames = tuple(self._extractor.extract(plan))
        report.frames_decoded = len(frames)
        self._log_decode(job, report, plan)

        # 4. One deep request, at the caller's priority, through the one queue.
        self._check_deadline(job, "decoding frames")
        request = AnalysisRequest(
            chunk_id=job.job_id,
            question=job.question,
            t_start=job.t_start,
            t_end=job.t_end,
            frames=frames,
            segments=tuple(report.segments),
            covered_seconds=report.covered_seconds,
            gap_seconds=report.gap_seconds,
        )
        result = self._analyze(request, job, report, priority)

        # 5. Confidence — a coverage heuristic, explained in analysis.derive_confidence.
        report.hedged = result.hedged
        report.confidence = derive_confidence(
            requested_seconds=report.requested_seconds,
            covered_seconds=report.covered_seconds,
            frames_decoded=report.frames_decoded,
            expected_frames=report.expected_frames,
            hedged=result.hedged,
            hedged_factor=self._settings.hedged_confidence_factor,
        )
        report.confidence_detail = confidence_explanation(
            requested_seconds=report.requested_seconds,
            covered_seconds=report.covered_seconds,
            frames_decoded=report.frames_decoded,
            expected_frames=report.expected_frames,
            hedged=result.hedged,
            hedged_factor=self._settings.hedged_confidence_factor,
        )

        self._finish(
            job,
            JobState.DONE,
            answer=result.answer + self._gap_note(report),
            reasoning=result.reasoning,
            confidence=report.confidence,
        )
        self._emit(
            "deep_done",
            job_id=job.job_id,
            elapsed_s=round(job.elapsed, 3),
            frames=report.frames_decoded,
            confidence=report.confidence,
            confidence_detail=report.confidence_detail,
            evidence_clip=report.evidence_clip,
            is_stub=report.is_stub,
        )
        return frames_dir

    def _analyze(
        self,
        request: AnalysisRequest,
        job: DeepJob,
        report: DeepReport,
        priority: Priority,
    ) -> Any:
        """Run the deep request through ``shared/queue.py``. Never around it — invariant 1.

        The wait is bounded by what is left of the job's 90 s rather than by the
        transport's own 120 s (``vlm.profiles.deep.request_timeout_seconds``), which is why
        settings.yaml sets those two numbers in that order: the job-level timeout fires
        first, with a sentence, instead of the transport dying with a stack trace.
        """
        queued = self._queue.submit(
            priority,
            lambda: self._backend.analyze(request),
            label=f"deep:{job.job_id}",
        )
        report.queue_job_id = queued.job_id
        remaining = self._remaining(job)
        if remaining <= 0:
            self._queue.cancel(queued)
            raise _Deadline("waiting for the VLM")
        try:
            return queued.result(timeout=remaining)
        except QueueTimeout as exc:
            self._queue.cancel(queued)
            raise _Deadline("waiting for the VLM") from exc

    # -- steps -------------------------------------------------------------------------

    def _record_coverage(
        self, job: DeepJob, report: DeepReport, spans: Sequence[SegmentSpan]
    ) -> None:
        """Fold the archive's holes into the report. Never into silence — invariant 3."""
        holes = [s for s in spans if s.is_gap]
        report.covered_seconds = timecode.covered_seconds(list(spans))
        report.gap_seconds = max(0.0, report.requested_seconds - report.covered_seconds)
        report.gaps = [(to_iso(h.t_start), to_iso(h.t_end)) for h in holes]
        report.segments = list(segments_of(spans))
        if holes:
            logger.warning(
                "archive is missing part of a deep-analysis range",
                extra={
                    "fields": {
                        "job_id": job.job_id,
                        "gap_seconds": round(report.gap_seconds, 3),
                        "gaps": report.gaps,
                    }
                },
            )
        if report.covered_seconds <= 0:
            raise _Refusal(
                f"the archive holds no footage for {to_iso(job.t_start)}..{to_iso(job.t_end)}; "
                f"there is nothing to re-watch. This is a fact about the recording, not a "
                f"failure of the analysis."
            )

    def _cut_clip(self, job: DeepJob) -> str | None:
        """Cut the evidence clip for the range, or return None if we cannot.

        A stream copy can only *begin* on a keyframe, which is why the recorder writes a
        1 s GOP (``recorder.device.keyframe_interval_seconds``): that interval is the floor
        on how accurately this clip can land on the requested range. Returns None rather
        than a path when nothing was cut — a job naming a clip that does not exist is worse
        than one admitting it has none, because the UI will offer it to a human.
        """
        slices = list(self._resolver(job.t_start, job.t_end))
        if not slices:
            logger.warning(
                "no archive segments cover the range; no evidence clip",
                extra={"fields": {"job_id": job.job_id}},
            )
            return None
        plan = build_clip_plan(
            slices,
            clip_path_for(
                job.t_start,
                job.t_end,
                clips_dir=self._settings.clips_dir,
                camera_id=self._settings.camera_id,
                container=self._settings.clip_container,
            ),
            ffmpeg_bin=self._settings.ffmpeg_bin,
            copy_codec=self._settings.copy_codec,
        )
        return self._cutter.cut(plan)

    def _gap_note(self, report: DeepReport) -> str:
        """A sentence appended to the answer when part of the range was never recorded.

        The confidence number already drops, but a number is easy to miss and the answer
        text is not. The note is attributed to the worker so it cannot be mistaken for
        something the model said.
        """
        if not report.gaps:
            return ""
        holes = "; ".join(f"{a}..{b}" for a, b in report.gaps)
        return (
            f"\n\n[worker note] {report.gap_seconds:.2f}s of the requested "
            f"{report.requested_seconds:.2f}s range was never recorded ({holes}). "
            f"This answer is based only on the {report.covered_seconds:.2f}s that exists."
        )

    # -- state -------------------------------------------------------------------------

    def _validate(
        self, t_start: datetime, t_end: datetime, question: str
    ) -> tuple[datetime, datetime]:
        if t_start.tzinfo is None or t_end.tzinfo is None:
            raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
        t0 = t_start.astimezone(timezone.utc)
        t1 = t_end.astimezone(timezone.utc)
        if t1 <= t0:
            raise ValueError(f"empty or inverted range: {to_iso(t0)} .. {to_iso(t1)}")
        if not question or not question.strip():
            raise ValueError("a deep analysis needs a question; an empty one has no answer")
        return t0, t1

    def _dedupe_key(self, t0: datetime, t1: datetime, question: str) -> tuple[str, str, str]:
        # Whitespace and case are noise from a text box, not a different question.
        return (to_iso(t0), to_iso(t1), " ".join(question.split()).lower())

    def _register(self, job: DeepJob, report: DeepReport) -> None:
        self._jobs[job.job_id] = job
        self._reports[job.job_id] = report
        self._events[job.job_id] = threading.Event()
        while len(self._jobs) > self._settings.job_history:
            oldest = next(iter(self._jobs))
            self._jobs.pop(oldest, None)
            self._reports.pop(oldest, None)
            self._events.pop(oldest, None)

    def _job_for(self, job: DeepJob | str) -> DeepJob:
        if isinstance(job, DeepJob):
            return job
        with self._lock:
            found = self._jobs.get(job)
        if found is None:
            raise KeyError(f"unknown deep job: {job!r}")
        return found

    def _deadline(self, job: DeepJob) -> datetime:
        return job.requested_at + timedelta(seconds=self._settings.timeout_seconds)

    def _remaining(self, job: DeepJob) -> float:
        return (self._deadline(job) - self._clock()).total_seconds()

    def _check_deadline(self, job: DeepJob, stage: str) -> None:
        if self._remaining(job) <= 0:
            raise _Deadline(stage)

    def _timeout_message(self, job: DeepJob, stage: str) -> str:
        return (
            f"deep analysis timed out after {job.elapsed:.1f}s while {stage} "
            f"(agent.deep.timeout_seconds is {self._settings.timeout_seconds:g}s). "
            f"No answer was produced; any evidence clip already cut is still attached."
        )

    def _too_long_message(self, requested_seconds: float) -> str:
        frames = frames_for_seconds(requested_seconds, self._settings.sample_fps)
        return (
            f"requested range is {requested_seconds:.1f}s, which is "
            f"{frames} frames at {self._settings.sample_fps:g} fps — past the "
            f"{self._settings.max_range_seconds:g}s the deep path is budgeted for "
            f"(agent.deep.max_range_seconds, {self._settings.max_frames} frames). SPEC §5 "
            f"targets 20-60s per job and this range would blow that. Nothing was truncated: "
            f"split the range, or raise the setting behind a benchmark."
        )

    def _apply_deadline_locked(self, job: DeepJob) -> None:
        if job.state in TERMINAL_STATES:
            return
        if self._remaining(job) <= 0:
            self._finish_locked(
                job, JobState.TIMEOUT, error=self._timeout_message(job, "waiting for the worker")
            )

    def _finish(
        self,
        job: DeepJob,
        state: JobState,
        *,
        answer: str = "",
        reasoning: str = "",
        confidence: float | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            return self._finish_locked(
                job,
                state,
                answer=answer,
                reasoning=reasoning,
                confidence=confidence,
                error=error,
            )

    def _finish_locked(
        self,
        job: DeepJob,
        state: JobState,
        *,
        answer: str = "",
        reasoning: str = "",
        confidence: float | None = None,
        error: str | None = None,
    ) -> bool:
        """Move a job to a terminal state, once. Returns False if it was already terminal.

        A late result is dropped rather than applied: the caller has already been shown
        ``TIMEOUT`` with a stated 90 s budget (SPEC §4.3), and an answer appearing after
        that contradicts what is on screen.

        ``state`` is assigned **last**, after every field a reader would want with it. That
        ordering is the contract in the module docstring: observing ``DONE`` implies the
        answer, reasoning, confidence and clip are all already visible.
        """
        if job.state in TERMINAL_STATES:
            if state is JobState.DONE:
                self._emit(
                    "deep_late_result",
                    job_id=job.job_id,
                    already=job.state.value,
                    elapsed_s=round(job.elapsed, 3),
                )
            return False
        if answer:
            job.answer = answer
        if reasoning:
            job.reasoning = reasoning
        if confidence is not None:
            job.confidence = confidence
        job.error = error
        job.completed_at = self._clock()
        job.state = state
        self._counters[state.value] = self._counters.get(state.value, 0) + 1
        event = self._events.get(job.job_id)
        if event is not None:
            event.set()
        if state is not JobState.DONE:
            self._emit("deep_" + state.value, job_id=job.job_id, error=error)
        return True

    # -- logging — CLAUDE.md: we cannot tune what we cannot see -------------------------

    def _log_decode(self, job: DeepJob, report: DeepReport, plan: DecodePlan) -> None:
        self._emit(
            "deep_decoded",
            job_id=job.job_id,
            steps=len(plan.steps),
            segments=report.segments,
            fps=plan.fps,
            native_resolution=self._settings.native_resolution,
            frames=report.frames_decoded,
            expected_frames=report.expected_frames,
        )

    def _emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, **fields}
        line = (
            json.dumps(record, sort_keys=True, default=str)
            if str(config.get("logging.format", "json")) == "json"
            else " ".join(f"{k}={v}" for k, v in record.items())
        )
        logger.info(line)


# --------------------------------------------------------------------------------------
# The process-wide worker — SPEC §5's entry point as M3 and M5 import it
# --------------------------------------------------------------------------------------

_default: DeepWorker | None = None
_default_lock = threading.Lock()


def default_worker() -> DeepWorker:
    """The lazily-built process worker.

    Built on first use rather than at import, because constructing it reads config and (on
    the ``vllm`` backend) a model name that is UNRESOLVED while SPEC §10 D1 is open. A
    process that already owns a ``VLMQueue`` should build its own :class:`DeepWorker` with
    that queue and install it via :func:`set_default_worker`, so that M1, M3, M4 and M5
    share one in-flight budget — there is one VLM (invariant 1).
    """
    global _default
    with _default_lock:
        if _default is None:
            _default = DeepWorker()
        return _default


def set_default_worker(worker: DeepWorker | None) -> None:
    """Install (or clear) the process worker. Call before the first ``deep_analyze``."""
    global _default
    with _default_lock:
        _default = worker


def deep_analyze(
    t_start: datetime,
    t_end: datetime,
    question: str,
    *,
    priority: Priority | str = Priority.INTERACTIVE,
    timeout: float | None = None,
    worker: DeepWorker | None = None,
) -> DeepJob:
    """SPEC §5. Re-watch a range and answer a question about it. **Blocks.**

    Returns a terminal ``DeepJob`` carrying ``answer``, ``evidence_clip``, ``confidence``,
    ``reasoning`` and ``elapsed``. It does not raise on timeout or on missing footage —
    those are states on the job, with a sentence in ``error``.

    Callers on a user's turn must use :func:`submit` instead (CLAUDE.md invariant 4).
    ``priority`` is ``INTERACTIVE`` when a human is waiting and ``VERIFICATION`` when M5 is
    checking an alert it has already fired (SPEC §7).
    """
    return (worker or default_worker()).analyze(
        t_start, t_end, question, priority=priority, timeout=timeout
    )


def submit(
    t_start: datetime,
    t_end: datetime,
    question: str,
    *,
    priority: Priority | str = Priority.INTERACTIVE,
    worker: DeepWorker | None = None,
) -> DeepJob:
    """SPEC §4.3. Queue a deep analysis and return a ``QUEUED`` ``DeepJob`` immediately.

    The job mutates in place as it progresses; poll it with ``DeepWorker.poll`` (which is
    also what applies the 90 s timeout) and stream the refinement when it reaches ``DONE``.
    """
    return (worker or default_worker()).submit(t_start, t_end, question, priority=priority)


def ffmpeg_worker_from_config(
    *, archive_dir: str | Path | None = None, **kwargs: Any
) -> DeepWorker:
    """The production worker: real ffmpeg for both the decode and the evidence clip.

    Never constructed by the test suite — nothing under ``tests/`` shells out.
    """
    settings = WorkerSettings.from_config(archive_dir=archive_dir)
    if archive_dir is not None:
        kwargs.setdefault("archive_dir", archive_dir)
    kwargs.setdefault(
        "clip_cutter",
        FfmpegClipCutter(
            ffmpeg_bin=settings.ffmpeg_bin,
            timeout_seconds=settings.clip_timeout_seconds,
        ),
    )
    kwargs.setdefault(
        "extractor",
        FfmpegFrameExtractor(
            ffmpeg_bin=settings.ffmpeg_bin,
            timeout_seconds=settings.decode_timeout_seconds,
        ),
    )
    kwargs.setdefault("settings", settings)
    return DeepWorker(**kwargs)
