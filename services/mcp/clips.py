"""Evidence clips — SPEC §6.4 ("with the clip attached"), rendered by §11.4.

``ffmpeg`` is **not installed on this box** (CLAUDE.md machine state), so this module is
split in two on purpose:

* ``build_clip_plan`` is pure. Given segment slices and an output path it returns the
  exact argv it would run. It is a value, not an effect, and the tests assert on it
  without a subprocess anywhere.
* ``FfmpegClipCutter`` executes a plan. Nothing in the test suite constructs one.

The default cutter is ``NullClipCutter``: it records the plan, returns no path, and the
log honestly says there is no clip. A row claiming a clip that was never cut is worse
than a row admitting it has none, because the Timeline pane will offer it to a human.

Invariant 3: a clip is requested by *time range*, never by filename. Resolving a range
to segment files and PTS offsets belongs to ``shared/timecode.py``, which owns that
derivation exclusively. This module takes the resolver as an injected callable so it
neither duplicates that logic nor imports a module that may not exist yet.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from shared.schema import chunk_id_for

__all__ = [
    "SegmentSlice",
    "ClipPlan",
    "ClipCutter",
    "NullClipCutter",
    "FfmpegClipCutter",
    "SegmentResolver",
    "timecode_segment_resolver",
    "clip_path_for",
    "build_clip_plan",
]

logger = logging.getLogger("mcp.clips")


@dataclass(frozen=True)
class SegmentSlice:
    """One archive file and the span of it a clip needs.

    ``seek_seconds`` is a PTS offset *within that file* — PTS restarts at zero every
    segment, so this number is meaningless without ``path`` beside it (invariant 2). The
    shape matches what ``shared/timecode.py`` is expected to return.
    """

    path: str
    seek_seconds: float
    duration_seconds: float


#: Resolves a wall-clock range to the archive slices covering it. Defaults to
#: ``timecode_segment_resolver`` below; a caller may inject a stub instead.
SegmentResolver = Callable[[datetime, datetime], Sequence[SegmentSlice]]


def timecode_segment_resolver(t_start: datetime, t_end: datetime) -> list[SegmentSlice]:
    """Bind the clip cutter to ``shared/timecode.py``, which owns range resolution.

    Gap spans (a recorder restart, ``path is None``) are dropped, with a warning naming
    the hole. Dropping is deliberate: the clip then contains the footage that genuinely
    exists rather than failing outright, which matters when an alert lands seconds after
    a restart. What it must never do is *silently* present a short clip as complete — so
    the hole is logged with its exact bounds, and the caller can consult
    ``timecode.gaps()`` on the same range when it needs to say so on screen.
    """
    from shared import timecode

    slices: list[SegmentSlice] = []
    for span in timecode.resolve_range(t_start, t_end):
        if span.is_gap or span.path is None:
            logger.warning(
                "archive gap inside a clip range; that footage was never recorded",
                extra={
                    "fields": {
                        "gap_start": span.t_start.isoformat(),
                        "gap_end": span.t_end.isoformat(),
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


@dataclass(frozen=True)
class ClipPlan:
    """Everything needed to cut a clip, and nothing that has been done yet.

    ``commands`` run in order. A range inside one segment is a single copy-cut; a range
    spanning a segment boundary is one copy-cut per part followed by a concat demux,
    which needs ``concat_list_text`` written to ``concat_list_path`` first.
    """

    out_path: Path
    slices: tuple[SegmentSlice, ...]
    commands: tuple[tuple[str, ...], ...]
    part_paths: tuple[Path, ...] = ()
    concat_list_path: Path | None = None
    concat_list_text: str | None = None


def clip_path_for(
    t_start: datetime,
    t_end: datetime,
    *,
    clips_dir: Path,
    camera_id: str,
    container: str,
) -> Path:
    """Name a clip after the footage range it holds, using the §3.1 chunk id form.

    Deriving the name from the range rather than a counter means the same range always
    names the same file — a re-cut overwrites rather than accumulating near-duplicates,
    and a human reading ``data/clips`` can see what they are looking at.
    """
    return Path(clips_dir) / f"{chunk_id_for(camera_id, t_start, t_end)}.{container}"


def _fmt(seconds: float) -> str:
    """Millisecond-precision seconds; ffmpeg accepts this and it stays diff-readable."""
    return f"{seconds:.3f}"


def build_clip_plan(
    slices: Sequence[SegmentSlice],
    out_path: Path,
    *,
    ffmpeg_bin: str,
    copy_codec: bool,
) -> ClipPlan:
    """Construct the ffmpeg invocations for a clip. Pure — runs nothing.

    ``-ss`` goes *before* ``-i`` so the seek is a fast one, and ``-c copy`` keeps the
    archive's native resolution intact (invariant 7): an evidence clip that has been
    re-encoded is no longer evidence of what the camera saw.
    """
    if not slices:
        raise ValueError("no segment slices; a clip needs at least one")
    out_path = Path(out_path)
    codec_args: tuple[str, ...] = ("-c", "copy") if copy_codec else ()
    base = (ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error", "-y")

    def cut(source: SegmentSlice, target: Path) -> tuple[str, ...]:
        return (
            *base,
            "-ss",
            _fmt(source.seek_seconds),
            "-t",
            _fmt(source.duration_seconds),
            "-i",
            source.path,
            *codec_args,
            "-avoid_negative_ts",
            "make_zero",
            str(target),
        )

    if len(slices) == 1:
        return ClipPlan(
            out_path=out_path,
            slices=tuple(slices),
            commands=(cut(slices[0], out_path),),
        )

    # Boundary-spanning range: cut each part, then concat-demux them losslessly.
    parts = tuple(
        out_path.with_name(f"{out_path.stem}.part{i}{out_path.suffix}")
        for i in range(len(slices))
    )
    commands = [cut(source, part) for source, part in zip(slices, parts, strict=True)]
    list_path = out_path.with_name(f"{out_path.stem}.concat.txt")
    list_text = "".join(f"file '{part}'\n" for part in parts)
    commands.append(
        (
            *base,
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            *codec_args,
            str(out_path),
        )
    )
    return ClipPlan(
        out_path=out_path,
        slices=tuple(slices),
        commands=tuple(commands),
        part_paths=parts,
        concat_list_path=list_path,
        concat_list_text=list_text,
    )


class ClipCutter:
    """Executes a :class:`ClipPlan`. Returns the clip path, or None if nothing was cut."""

    def cut(self, plan: ClipPlan) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError


class NullClipCutter(ClipCutter):
    """Default cutter: plans, records, cuts nothing.

    This is the correct behaviour on a box without ffmpeg. The plan is kept on
    ``plans`` so a caller (or a test) can inspect exactly what would have run.
    """

    def __init__(self) -> None:
        self.plans: list[ClipPlan] = []

    def cut(self, plan: ClipPlan) -> str | None:
        self.plans.append(plan)
        logger.info(
            "clip not cut: no cutter configured",
            extra={"fields": {"out_path": str(plan.out_path), "slices": len(plan.slices)}},
        )
        return None


class FfmpegClipCutter(ClipCutter):
    """Runs the plan through ffmpeg. Never constructed by the tests.

    ``timeout_seconds`` is not optional: this runs inside the action log's write lock so
    that the brake check, the cut and the append are one atomic step, and an ffmpeg that
    hangs would take both M3 and M5 down with it. A copy-cut of a 60 s segment is
    milliseconds; anything near the timeout means the archive is wrong, not slow.
    """

    def __init__(self, *, ffmpeg_bin: str, timeout_seconds: float) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        return shutil.which(self.ffmpeg_bin) is not None

    def cut(self, plan: ClipPlan) -> str | None:
        if not self.available():
            logger.warning(
                "ffmpeg not on PATH; clip skipped",
                extra={"fields": {"ffmpeg_bin": self.ffmpeg_bin}},
            )
            return None
        plan.out_path.parent.mkdir(parents=True, exist_ok=True)
        if plan.concat_list_path is not None and plan.concat_list_text is not None:
            plan.concat_list_path.write_text(plan.concat_list_text, encoding="utf-8")
        for command in plan.commands:
            try:
                subprocess.run(  # noqa: S603 - argv is built here, never shell-interpolated
                    list(command),
                    check=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                logger.warning(
                    "clip cut failed",
                    extra={"fields": {"command": list(command), "error": str(exc)}},
                )
                return None
        for part in plan.part_paths:
            part.unlink(missing_ok=True)
        if plan.concat_list_path is not None:
            plan.concat_list_path.unlink(missing_ok=True)
        return str(plan.out_path)
