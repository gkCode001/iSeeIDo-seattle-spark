"""Re-decode a resolved time range at 4 fps, native resolution (SPEC §5 step 2).

The live path samples 1 fps at ~512 px short side because it is protecting KV-cache
headroom (SPEC §2.5). This path does the opposite on purpose: **no scale filter is ever
emitted here**. CLAUDE.md invariant 7 — the archive stays native and the deep worker is
the reason it does. If you find yourself adding a resize to make a job faster, the
correct dial is the range length or ``max_tokens``, not the pixels.

Two shapes, mirroring ``services/mcp/clips.py``:

* :func:`build_decode_plan` is pure. Given the spans ``shared/timecode.py`` resolved, it
  returns the exact argv it would run. The tests assert on it with no subprocess
  anywhere, which is the only way the drawtext escaping below is checkable at all.
* :class:`FfmpegFrameExtractor` executes a plan. Nothing under ``tests/`` constructs one.

Stitching
---------
One ffmpeg invocation per span, never one per file-name-guess: an event at 21:11:58
running 12 s is two files (invariant 3), and the spans arrive already tiled by
``resolve_range``. Gap spans are dropped here and reported by the caller — this module
must not be the place that decides a hole is acceptable. Frames are named
``s{span}_{n}.jpg`` so a plain ``sorted()`` puts them back in wall-clock order across the
boundary, which is what the VLM needs to reason about "then".

The overlay — CLAUDE.md invariant 8
-----------------------------------
The VLM reads a burned-in wall clock for temporal localization, and ``vlm.prompts.deep``
asks it to cite those timestamps back to us. The archive itself carries none (the
recorder stream-copies or encodes without drawtext), so the deep path burns its own onto
the sampled frames, exactly as SPEC §2.4 step 3 does on the live path. There is no resize
here, so "after any resize" is satisfied trivially, and at native resolution the
configured ``ingest.overlay.fontsize`` is comfortably above ``min_height_px``.

The clock shown is **absolute UTC**, derived from ``span.t_start`` plus the frame's
position in the trimmed output — not the segment's PTS, which restarts at zero every file
and names no moment in history (invariant 2). ffmpeg computes it per frame via
``%{pts:gmtime:<epoch>:<strftime>}``; the epoch base is the wall-clock instant the span
begins at.

That expansion has to survive two rounds of unescaping — the filtergraph option parser and
then drawtext's own ``%{}`` splitter — so the argument separators need one backslash and
the colons *inside* the strftime format need three. This was established empirically
against ffmpeg 6.1.1 on real 1080p footage from this box, not derived from the docs, and
:func:`drawtext_expansion` is unit-tested character for character because a wrong count
here does not fail: it renders ``Stray %`` into the frame and localization degrades
silently, which is precisely what invariant 8 warns about.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from shared.timecode import SegmentSpan

from .settings import WorkerSettings

__all__ = [
    "DecodeStep",
    "DecodePlan",
    "FrameExtractor",
    "NullFrameExtractor",
    "FfmpegFrameExtractor",
    "build_decode_plan",
    "drawtext_expansion",
    "frames_for_seconds",
    "sorted_frames",
    "video_filter",
]

logger = logging.getLogger("services.worker.decode")

#: Extension of the extracted frames. JPEG because ``shared.vlm_client.encode_frame``
#: defaults to ``image/jpeg`` and the frames go straight into a ``data:`` URI.
FRAME_SUFFIX = "jpg"

#: Separator between the arguments of drawtext's ``%{pts:...}`` expansion. One backslash:
#: the filtergraph parser eats it and drawtext sees a bare colon.
_ARG_SEP = "\\:"


def _escape_strftime(fmt: str) -> str:
    """Escape a strftime format so drawtext's ``%{}`` splitter keeps it in one piece.

    Three backslashes before each colon, not one: the filtergraph parser unescapes once
    (``\\\\\\:`` -> ``\\:``) and drawtext's expansion parser unescapes again (``\\:`` ->
    ``:``). With one backslash the colon separates arguments and ffmpeg reports
    ``%{pts} requires at most 3 arguments``; with two it reports ``Stray %``. Verified
    against ffmpeg 6.1.1.
    """
    return fmt.replace("\\", "\\\\").replace(":", "\\\\\\:")


def _escape_option(value: str) -> str:
    """Escape a plain filter option value (a font path) for the filtergraph parser."""
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def drawtext_expansion(wall_start: datetime, fmt: str) -> str:
    """The ``%{pts:gmtime:...}`` text that renders absolute UTC on each frame.

    ``wall_start`` is the wall-clock instant of the span's first decoded frame. Output PTS
    restart at zero after the trim, so ``epoch + pts`` is the true time of every frame
    that follows — including the ones on the far side of a segment boundary, which is the
    whole point.
    """
    if wall_start.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    epoch = int(wall_start.astimezone(timezone.utc).timestamp())
    return f"%{{pts{_ARG_SEP}gmtime{_ARG_SEP}{epoch}{_ARG_SEP}{_escape_strftime(fmt)}}}"


def video_filter(wall_start: datetime, settings: WorkerSettings) -> str:
    """The ``-vf`` chain for one span: sample, then burn the clock. Never scale.

    Order matters and is the invariant-8 order: ``fps`` first so the overlay is drawn once
    per kept frame rather than on frames we are about to throw away, and no ``scale``
    anywhere, so the text is burned at full size onto a native-resolution frame.
    """
    chain = [f"fps={settings.sample_fps:g}"]
    if not settings.overlay_enabled:
        return ",".join(chain)

    pad = settings.overlay_padding_px
    y = f"h-text_h-{pad}" if settings.overlay_position == "bottom" else str(pad)
    text = drawtext_expansion(wall_start, settings.overlay_format)
    chain.append(
        "drawtext="
        f"fontfile={_escape_option(settings.overlay_fontfile)}"
        f":text='{text}'"
        f":fontsize={settings.overlay_fontsize}"
        ":fontcolor=white"
        # A box, not bare text: a white timestamp over a white wall is illegible, and an
        # illegible overlay is invariant 8's silent failure.
        ":box=1:boxcolor=black@0.6:boxborderw=4"
        f":x=(w-text_w)/2:y={y}"
    )
    return ",".join(chain)


@dataclass(frozen=True)
class DecodeStep:
    """One ffmpeg invocation: the frames of one span of one segment file.

    ``wall_start`` is carried alongside ``seek_seconds`` for the same reason a
    ``ChunkRecord`` carries both — the PTS offset locates the pixels, the wall clock names
    the moment, and neither substitutes for the other (invariant 2).
    """

    span_index: int
    source: Path
    seek_seconds: float
    duration_seconds: float
    wall_start: datetime
    wall_end: datetime
    expected_frames: int
    argv: tuple[str, ...]


@dataclass(frozen=True)
class DecodePlan:
    """Every invocation needed to turn a resolved range into frames on disk. Runs nothing."""

    out_dir: Path
    fps: float
    steps: tuple[DecodeStep, ...]

    @property
    def expected_frames(self) -> int:
        return sum(step.expected_frames for step in self.steps)

    @property
    def covered_seconds(self) -> float:
        return sum(step.duration_seconds for step in self.steps)


def build_decode_plan(
    spans: Sequence[SegmentSpan],
    out_dir: Path,
    *,
    settings: WorkerSettings,
) -> DecodePlan:
    """Construct the ffmpeg invocations for a resolved range. Pure — runs nothing.

    ``-ss`` before ``-i`` so the seek is a fast one. Unlike the copy-cut in
    ``services/mcp/clips.py`` this path re-encodes to stills, so ffmpeg decodes from the
    preceding keyframe and discards up to the seek point: the frames land on the requested
    instant rather than on the nearest keyframe. The 1 s GOP the recorder writes still
    matters — it bounds how much pre-roll is decoded and thrown away.
    """
    out_dir = Path(out_dir)
    steps: list[DecodeStep] = []
    for index, span in enumerate(spans):
        if span.is_gap or span.path is None:
            # Reported by the caller, folded into confidence, never silently absorbed.
            continue
        duration = span.duration
        if duration <= 0:
            continue
        pattern = f"s{index:03d}_%05d.{FRAME_SUFFIX}"
        argv = (
            settings.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{span.pts_in:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(span.path),
            "-an",
            "-sn",
            "-vf",
            video_filter(span.t_start, settings),
            # The fps filter already fixed the rate; passthrough stops ffmpeg from
            # duplicating or dropping frames a second time on the way out.
            "-fps_mode",
            "passthrough",
            "-q:v",
            str(settings.frame_quality),
            "-f",
            "image2",
            str(out_dir / pattern),
        )
        steps.append(
            DecodeStep(
                span_index=index,
                source=span.path,
                seek_seconds=span.pts_in,
                duration_seconds=duration,
                wall_start=span.t_start,
                wall_end=span.t_end,
                expected_frames=max(1, int(round(duration * settings.sample_fps))),
                argv=argv,
            )
        )
    if not steps:
        raise ValueError("no decodable spans; the requested range is entirely archive gap")
    return DecodePlan(out_dir=out_dir, fps=settings.sample_fps, steps=tuple(steps))


def sorted_frames(out_dir: Path) -> list[Path]:
    """Frames in wall-clock order. Relies on the ``s{span}_{n}`` naming, not on mtime."""
    return sorted(Path(out_dir).glob(f"s*_*.{FRAME_SUFFIX}"))


class FrameExtractor(Protocol):
    """Executes a :class:`DecodePlan`. Returns the frame paths, in wall-clock order."""

    def extract(self, plan: DecodePlan) -> list[Path]: ...


class NullFrameExtractor:
    """Default extractor: plans, records, decodes nothing.

    The honest behaviour when ffmpeg is missing. A worker holding this returns zero frames
    and the confidence heuristic collapses accordingly, rather than an answer being
    produced about footage nobody read.
    """

    def __init__(self) -> None:
        self.plans: list[DecodePlan] = []

    def extract(self, plan: DecodePlan) -> list[Path]:
        self.plans.append(plan)
        logger.warning(
            "frames not decoded: no extractor configured",
            extra={"fields": {"steps": len(plan.steps), "expected": plan.expected_frames}},
        )
        return []


class FfmpegFrameExtractor:
    """Runs the plan through ffmpeg. Never constructed by the test suite.

    ``timeout_seconds`` is not optional: this decode holds the single in-flight deep slot
    (``agent.deep.max_inflight``), so an ffmpeg that wedges on a truncated segment blocks
    every later job behind it. A partial decode is kept rather than discarded — the frames
    that did come out are real footage, and the shortfall shows up in the frame count that
    feeds confidence.
    """

    def __init__(self, *, ffmpeg_bin: str, timeout_seconds: float) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        return shutil.which(self.ffmpeg_bin) is not None

    def extract(self, plan: DecodePlan) -> list[Path]:
        if not self.available():
            logger.warning(
                "ffmpeg not on PATH; no frames decoded",
                extra={"fields": {"ffmpeg_bin": self.ffmpeg_bin}},
            )
            return []
        plan.out_dir.mkdir(parents=True, exist_ok=True)
        for step in plan.steps:
            try:
                subprocess.run(  # noqa: S603 - argv is built here, never shell-interpolated
                    list(step.argv),
                    check=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                # One bad segment must not lose the other side of a boundary-spanning
                # range: log it, keep going, let the frame count tell the truth.
                logger.warning(
                    "decode step failed; continuing with the frames that exist",
                    extra={
                        "fields": {
                            "source": str(step.source),
                            "span_index": step.span_index,
                            "error": str(exc),
                        }
                    },
                )
        frames = sorted_frames(plan.out_dir)
        logger.info(
            "decoded frames",
            extra={
                "fields": {
                    "frames": len(frames),
                    "expected": plan.expected_frames,
                    "fps": plan.fps,
                    "steps": len(plan.steps),
                }
            },
        )
        return frames


def frames_for_seconds(seconds: float, fps: float) -> int:
    """Frames a range of this length yields at this rate. Reported in refusal messages."""
    return int(math.ceil(seconds * fps))
