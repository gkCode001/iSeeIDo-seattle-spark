"""M1 — the ingest pipeline. SPEC §2.

One window at a time, in order:

1. **Plan** a 5 s window on a 4 s stride (``windows.py``, SPEC §2.2). It is a time range
   pointing into the archive; no video is ever copied or cut.
2. **Resolve** it onto segment files through ``shared/timecode.py`` — never by filename
   (CLAUDE.md invariant 3), because a window at 21:11:58 lives in two of them.
3. **Gate** it (``gate.py``, SPEC §2.3). Most windows stop here. This is the step that
   makes real time arithmetically possible.
4. **Sample, resize, burn the clock** (``frames.py``, SPEC §2.4 and invariant 8).
5. **Caption** through ``shared/queue.py`` at ``Priority.INGEST`` (SPEC §7) and
   ``shared/vlm_client.py`` on the ``live`` profile (invariant 6).
6. **Write** a ``ChunkRecord`` to the index (SPEC §3.1, invariant 2 — wall clock *and*
   ``segment`` + ``pts_offset``).

Every window produces a record, including a skipped one
-------------------------------------------------------
SPEC §2.3: a gated window writes a null record with ``gated=True`` and an empty
caption. Not an optimisation to skip — the skip rate is the health metric that says
whether the gate is tuned, and **a gap in the record stream is indistinguishable from
crashed ingest**. The one case that produces no record at all is a window whose start
instant the archive does not cover: ``segment`` is required by the schema and a record
pointing at a file that does not contain the moment is worse than no record. That case
is counted and logged as its own event so it can never be mistaken for a skip.

Undecodable footage is not a skip
---------------------------------
A segment whose recorder was hard-killed has no moov atom (CLAUDE.md: SIGTERM, never
SIGKILL) and ffmpeg cannot read it. Those windows get a gated record too — it is the
only null shape ``ChunkRecord`` permits — but they are counted separately as decode
failures and reported separately, because a 100% skip rate made of unreadable files is a
broken archive wearing a healthy gate's numbers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import TracebackType
from typing import Any, Protocol

from shared.queue import Priority, VLMQueue
from shared.schema import ChunkRecord, Tier, to_iso, utcnow
from shared.timecode import (
    MissingFootageError,
    SegmentSpan,
    covered_seconds,
    resolve_range,
    segment_and_offset,
)
from shared.vlm_client import VLMChunk, encode_frame

from .captioner import Captioner, build_captioner
from .ffmpeg import FFmpegDecodeError, resolve_ffmpeg
from .frames import FrameExtractor
from .gate import Gate, GateDecision, build_gate
from .settings import IngestSettings
from .telemetry import log_event, timed
from .windows import Window, archive_bounds, plan_windows

__all__ = ["IngestStats", "ChunkSink", "IngestPipeline"]

LOGGER = logging.getLogger("services.ingest")


class ChunkSink(Protocol):
    """Where records go. ``services.index.IndexStore`` satisfies this.

    Narrowed to the one method M1 uses so that a run with ``--no-index`` — or a test —
    needs a three-line object rather than a whole store. It takes a **list**, which is
    invariant 9 in the signature.
    """

    def insert(self, chunks: list[ChunkRecord]) -> int:
        ...


@dataclass
class IngestStats:
    """What a run did. Printed at the end and logged as one structured event.

    The counters are deliberately not collapsible into each other: ``skipped`` is the
    gate working, ``decode_failures`` is the archive being unreadable, and
    ``no_footage`` is the archive not existing for that second. All three look like
    "the VLM did not run", and only the first one is good news.
    """

    windows: int = 0
    captioned: int = 0
    skipped: int = 0
    decode_failures: int = 0
    no_footage: int = 0
    records_written: int = 0
    gate_ms: float = 0.0
    extract_ms: float = 0.0
    caption_ms: float = 0.0
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None

    @property
    def decided(self) -> int:
        """Windows the gate actually ruled on — the honest skip-rate denominator."""
        return self.captioned + self.skipped

    @property
    def skip_rate(self) -> float:
        """SPEC §2.3's number. Zero windows is 0.0, not a division error."""
        return (self.skipped / self.decided) if self.decided else 0.0

    @property
    def elapsed(self) -> float:
        return ((self.finished_at or utcnow()) - self.started_at).total_seconds()

    def health(self, settings: IngestSettings) -> str:
        """``ok`` / ``low`` / ``empty`` against ``ingest.gate.*``.

        Matches ``IndexStore.gate_health`` deliberately: M1 and M2 must not disagree
        about whether the gate is healthy, and two thresholds read from the same keys is
        the cheapest way to guarantee that.
        """
        if self.decided == 0:
            return "empty"
        return "ok" if self.skip_rate >= settings.warn_skip_rate else "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": self.windows,
            "captioned": self.captioned,
            "skipped": self.skipped,
            "decode_failures": self.decode_failures,
            "no_footage": self.no_footage,
            "records_written": self.records_written,
            "skip_rate": round(self.skip_rate, 4),
            "gate_ms": round(self.gate_ms, 2),
            "extract_ms": round(self.extract_ms, 2),
            "caption_ms": round(self.caption_ms, 2),
            "elapsed_s": round(self.elapsed, 3),
            "started_at": to_iso(self.started_at),
            "finished_at": to_iso(self.finished_at) if self.finished_at else None,
        }


class IngestPipeline:
    """The M1 loop. Construct it, call :meth:`run`, read the stats.

    Every collaborator is injectable — gate, frame extractor, captioner, queue, sink —
    which is what lets the whole loop be tested against byte strings with no ffmpeg, no
    model and no archive, and is also what lets ``bench.py`` reuse the frame path without
    inheriting the index.

    Use it as a context manager. It owns the VLM queue's worker thread when it built one,
    and a run that exits without stopping that thread leaves a daemon holding the single
    in-flight slot.
    """

    def __init__(
        self,
        settings: IngestSettings,
        *,
        gate: Gate | None = None,
        extractor: FrameExtractor | None = None,
        captioner: Captioner | None = None,
        queue: VLMQueue | None = None,
        sink: ChunkSink | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._s = settings
        self._gate = gate if gate is not None else build_gate(settings)
        self._extractor = extractor if extractor is not None else FrameExtractor(settings)
        self._captioner = captioner if captioner is not None else build_captioner(settings)
        self._sink = sink
        self._log = logger or LOGGER

        # An injected queue belongs to the caller; one we build here is ours to stop.
        self._queue = queue if queue is not None else VLMQueue()
        self._owns_queue = queue is None
        self._started = False

        self.stats = IngestStats()

    # -- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        """Bring up the queue worker. Idempotent."""
        if self._started:
            return
        self._queue.start()
        self._started = True

    def close(self) -> None:
        """Drain and stop the queue if we own it. Safe to call twice."""
        if self._owns_queue and self._started:
            self._queue.stop(drain=True)
        self._started = False

    def __enter__(self) -> IngestPipeline:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- planning ------------------------------------------------------------------

    def plan(
        self,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        *,
        start_index: int = 0,
    ) -> Iterator[Window]:
        """Windows over ``[t_from, t_to]``, defaulting to whatever the archive covers.

        The bounds come from filenames alone (``shared/timecode.py``) — no ffprobe, no
        decode. The last segment's end is nominal rather than measured, so the walk may
        plan a window or two past the true end of a half-written file; those decode-fail
        and are counted, which is the honest outcome for an archive that claimed to cover
        a second it does not.
        """
        if t_from is None or t_to is None:
            bounds = archive_bounds(str(self._s.archive_dir), self._s.camera_id)
            if bounds is None:
                log_event("ingest.archive_empty", archive=str(self._s.archive_dir))
                return iter(())
            t_from = t_from or bounds[0]
            t_to = t_to or bounds[1]
        return plan_windows(
            t_from,
            t_to,
            self._s.window_seconds,
            self._s.stride_seconds,
            start_index=start_index,
        )

    # -- the loop ------------------------------------------------------------------

    def run(
        self,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        *,
        limit: int | None = None,
    ) -> IngestStats:
        """Walk the archive once. Returns the run's stats and logs the summary."""
        self.start()
        self._preflight()
        windows = self.plan(t_from, t_to)
        if limit is not None:
            windows = _take(windows, limit)
        for window in windows:
            self.process(window)
        self.stats.finished_at = utcnow()
        self.log_summary()
        return self.stats

    def _resume_point(self) -> datetime | None:
        """The end of the newest chunk already indexed, or None when nothing is.

        Best-effort: a sink that cannot answer (a plain list in a test, a backend that
        has no browse) simply means "start from the archive", which is the old behaviour
        and is never wrong, only slower.
        """
        browse = getattr(self._sink, "browse", None)
        if not callable(browse):
            return None
        try:
            page = browse(limit=1, newest_first=True)
        except Exception as exc:  # noqa: BLE001 - resuming is an optimisation
            log_event("ingest.resume_failed", level=logging.WARNING, error=repr(exc))
            return None
        records = getattr(page, "records", None) or getattr(page, "chunks", None) or []
        if not records:
            return None
        newest = max(r.t_end for r in records)
        log_event("ingest.resumed", after=to_iso(newest))
        return newest

    def follow(
        self,
        t_from: datetime | None = None,
        *,
        poll_interval: float | None = None,
        max_passes: int | None = None,
    ) -> IngestStats:
        """Keep walking as the recorder extends the archive.

        Picks up where the last pass stopped rather than re-planning from the beginning,
        so a window is analysed exactly once. Windows that are not yet complete are not
        emitted at all (``plan_windows``), which is the "wait for the window to close"
        latency floor of SPEC §2.2 rather than a bug.

        ``t_from`` starts the walk somewhere other than the oldest segment on disk. Its
        use is skipping a backlog to caption the live tail: on a long archive the default
        re-walks hours of footage — already-indexed footage, on a restart — before it
        reaches the present. Footage before ``t_from`` is never analysed by this run.
        """
        self.start()
        self._preflight()
        interval = poll_interval if poll_interval is not None else self._s.poll_interval_seconds
        cursor: datetime | None = None
        index = 0
        passes = 0
        try:
            while max_passes is None or passes < max_passes:
                passes += 1
                # exclude_open: never walk into the file the recorder still has open.
                # Its windows cannot decode until it closes, and the cursor below only
                # moves forward — so they would be burned as decode failures and never
                # retried. See archive_bounds().
                bounds = archive_bounds(
                    str(self._s.archive_dir), self._s.camera_id, exclude_open=True
                )
                if bounds is not None:
                    # max(), not t_from alone: a t_from past the end of the archive would
                    # otherwise plan nothing until the recorder caught up to it, and a
                    # t_from before the archive starts would plan windows with no footage.
                    first = bounds[0] if t_from is None else max(t_from, bounds[0])
                    if cursor is None and t_from is None:
                        # RESUME where the index already got to. Without this every
                        # restart re-walks the archive from its first segment: the walk
                        # is fast enough to look healthy in the log while the newest
                        # caption stays hours old, so questions get answered from stale
                        # footage and nothing says why. Re-captioning also burns the one
                        # VLM slot on windows that are already indexed.
                        #
                        # Derived from the INDEX, not from a cursor file: clearing the
                        # index has to mean "index it again", and a file would happily
                        # skip everything it had already seen.
                        resumed = self._resume_point()
                        if resumed is not None:
                            first = max(first, resumed)
                    start = cursor if cursor is not None else first
                    for window in plan_windows(
                        start,
                        bounds[1],
                        self._s.window_seconds,
                        self._s.stride_seconds,
                        start_index=index,
                    ):
                        self.process(window)
                        index = window.index + 1
                        # Advance by a stride, not to the window end: consecutive windows
                        # overlap by 1 s (SPEC §2.2) and jumping to the end would drop it.
                        cursor = window.t_start + timedelta(seconds=self._s.stride_seconds)
                if max_passes is None or passes < max_passes:
                    time.sleep(interval)
        except KeyboardInterrupt:
            self._log.info("ingest interrupted; stopping after %d windows", self.stats.windows)
        self.stats.finished_at = utcnow()
        self.log_summary()
        return self.stats

    def process(self, window: Window) -> ChunkRecord | None:
        """One window, end to end. Returns the record written, or None for no footage."""
        self.stats.windows += 1
        spans = resolve_range(
            window.t_start, window.t_end, str(self._s.archive_dir), self._s.camera_id
        )

        try:
            segment, pts_offset = segment_and_offset(
                window.t_start, str(self._s.archive_dir), self._s.camera_id
            )
        except MissingFootageError as exc:
            # No record: ``segment`` is required by the schema (invariant 2) and there is
            # no honest value for it. Logged loudly so it is never read as a gate skip.
            self.stats.no_footage += 1
            log_event(
                "ingest.no_footage",
                level=logging.WARNING,
                window=str(window),
                t_start=to_iso(window.t_start),
                error=str(exc),
            )
            return None

        with timed("ingest.gate", window=str(window)) as gate_span:
            decision = self._gate.evaluate(window, spans)
            # Inside the block: `timed` writes its record on __exit__, so fields set after
            # the `with` are merged into an object nobody reads again. This update used to
            # sit below it, which is why no ingest.gate line has ever carried the verdict.
            gate_span.fields.update(
                passed=decision.passed,
                reason=decision.reason.value,
                score=decision.score,
                # The denominator the score was averaged over. A score without it cannot
                # be compared against one from a differently framed scene, which is
                # exactly how a letterboxed clip read as "still" for twenty minutes.
                active_px=decision.active_px,
            )
        self.stats.gate_ms += gate_span.elapsed_ms

        if decision.skipped:
            self.stats.skipped += 1
            return self._write(self._null_record(window, segment, pts_offset), decision)

        frames = self._sample(window, spans)
        if frames is None:
            # Undecodable. A gated record is the only null shape the schema allows, but
            # the counter and the log line keep it distinguishable from a real skip.
            self.stats.decode_failures += 1
            return self._write(self._null_record(window, segment, pts_offset), decision)

        caption = self._caption(window, frames)
        if caption is None:
            self.stats.decode_failures += 1
            return self._write(self._null_record(window, segment, pts_offset), decision)

        self.stats.captioned += 1
        record = ChunkRecord(
            chunk_id=window.chunk_id(self._s.camera_id),
            camera_id=self._s.camera_id,
            t_start=window.t_start,
            t_end=window.t_end,
            segment=segment,
            pts_offset=round(pts_offset, 3),
            tier=Tier.LIVE,
            gated=False,
            caption=caption,
            # Left empty on purpose: the index embeds captions itself, with the embedder
            # its own queries use (SPEC §3.4). A corpus embedded by one model and queried
            # by another does not error, it just quietly stops finding things.
            embedding=[],
        )
        return self._write(record, decision, frames=len(frames), covered=covered_seconds(spans))

    # -- steps ---------------------------------------------------------------------

    def _sample(self, window: Window, spans: Sequence[SegmentSpan]) -> list[bytes] | None:
        """Frames for a surviving window, or None when the footage cannot be read."""
        try:
            with timed("ingest.extract", window=str(window)) as span:
                frames = self._extractor.extract(spans)
                span.fields["frames"] = len(frames)
            self.stats.extract_ms += span.elapsed_ms
        except FFmpegDecodeError as exc:
            self.stats.extract_ms += 0.0
            log_event(
                "ingest.decode_failed",
                level=logging.WARNING,
                window=str(window),
                stage="extract",
                error=str(exc),
            )
            return None
        if not frames:
            log_event(
                "ingest.decode_failed",
                level=logging.WARNING,
                window=str(window),
                stage="extract",
                error="ffmpeg produced no frames for a window the archive claims to cover",
            )
            return None
        return frames

    def _caption(self, window: Window, frames: Sequence[bytes]) -> str | None:
        """Queue the caption at ingest priority and wait for it.

        A **list of one** — CLAUDE.md invariant 9. Not a batch of many: SPEC §0 is explicit
        that batching does not help a single-camera workload, and holding a closed window
        back to pair it with the next one would add a stride to SPEC §8's latency budget
        for no throughput at all. The list is the interface; the batch is a later config.
        """
        chunk_id = window.chunk_id(self._s.camera_id)
        vlm_chunk = VLMChunk(
            chunk_id=chunk_id,
            frames=[encode_frame(frame) for frame in frames],
        )
        started = time.perf_counter()
        job = self._queue.submit(
            Priority.INGEST,
            lambda: self._captioner.caption([vlm_chunk]),
            label=f"caption {chunk_id}",
        )
        try:
            results = job.result(timeout=self._s.caption_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - any caption failure is one failure to us
            self.stats.caption_ms += (time.perf_counter() - started) * 1000.0
            log_event(
                "ingest.caption_failed",
                level=logging.WARNING,
                window=str(window),
                chunk_id=chunk_id,
                error=repr(exc),
            )
            return None
        self.stats.caption_ms += (time.perf_counter() - started) * 1000.0

        text = results[0].text.strip() if results else ""
        if not text:
            log_event(
                "ingest.caption_failed",
                level=logging.WARNING,
                window=str(window),
                chunk_id=chunk_id,
                error="model returned an empty caption",
            )
            return None
        return text

    def _null_record(self, window: Window, segment: str, pts_offset: float) -> ChunkRecord:
        """The SPEC §2.3 null record: located in time, with nothing to say."""
        return ChunkRecord(
            chunk_id=window.chunk_id(self._s.camera_id),
            camera_id=self._s.camera_id,
            t_start=window.t_start,
            t_end=window.t_end,
            segment=segment,
            pts_offset=round(pts_offset, 3),
            tier=Tier.LIVE,
            gated=True,
            caption="",
            embedding=[],
        )

    def _write(
        self,
        record: ChunkRecord,
        decision: GateDecision,
        *,
        frames: int = 0,
        covered: float | None = None,
    ) -> ChunkRecord:
        """Send one record to the index and log it. Takes a list — invariant 9."""
        if self._sink is not None:
            self.stats.records_written += self._sink.insert([record])
        log_event(
            "ingest.chunk",
            chunk_id=record.chunk_id,
            t_start=to_iso(record.t_start),
            t_end=to_iso(record.t_end),
            segment=record.segment,
            pts_offset=record.pts_offset,
            gated=record.gated,
            gate_reason=decision.reason.value,
            gate_score=None if decision.score is None else round(decision.score, 5),
            frames=frames,
            covered_s=None if covered is None else round(covered, 3),
            caption=record.caption,
        )
        return record

    # -- health --------------------------------------------------------------------

    def _preflight(self) -> None:
        """Fail before the first window rather than once per window.

        A missing ffmpeg would otherwise surface as every window decode-failing, which
        reads as a corrupt archive rather than as a missing binary.
        """
        resolve_ffmpeg(self._s.ffmpeg_bin)
        log_event(
            "ingest.started",
            archive=str(self._s.archive_dir),
            camera_id=self._s.camera_id,
            window_s=self._s.window_seconds,
            stride_s=self._s.stride_seconds,
            sample_fps=self._s.sample_fps,
            short_side_px=self._s.live_short_side_px,
            gate_backend=self._s.gate_backend.value if self._s.gate_enabled else "disabled",
            motion_threshold=self._s.motion_threshold,
            vlm_backend=self._s.vlm_backend,
            captioner_model=self._captioner.model,
        )

    def log_summary(self) -> None:
        """One structured line, plus a warning when the gate is mistuned.

        SPEC §2.3: *log the skip rate. If it's below 60% the gate is mistuned and
        real-time is gone.* That sentence is this method.
        """
        stats = self.stats
        health = stats.health(self._s)
        log_event(
            "ingest.summary",
            level=logging.INFO if health != "low" else logging.WARNING,
            **stats.to_dict(),
            target_skip_rate=self._s.target_skip_rate,
            warn_skip_rate=self._s.warn_skip_rate,
            gate_health=health,
            captioner_model=self._captioner.model,
        )
        if health == "low":
            self._log.warning(
                "detector gate skipped %.1f%% of %d windows, under ingest.gate."
                "warn_skip_rate (%.0f%%). SPEC §2.3: the gate is mistuned and real-time "
                "is gone. Raise ingest.gate.motion_threshold, or accept that this scene "
                "is genuinely busy.",
                stats.skip_rate * 100,
                stats.decided,
                self._s.warn_skip_rate * 100,
            )
        if stats.decode_failures:
            self._log.warning(
                "%d of %d windows could not be decoded. A skip rate computed over an "
                "unreadable archive is not a gate measurement — check for segments with "
                "no moov atom, which is what a SIGKILLed recorder leaves behind.",
                stats.decode_failures,
                stats.windows,
            )


def _take(items: Iterable[Window], limit: int) -> Iterator[Window]:
    """``itertools.islice`` with a name that says why: ``--limit`` for a smoke run."""
    for n, item in enumerate(items):
        if n >= limit:
            return
        yield item
