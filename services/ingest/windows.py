"""Analysis windows — SPEC §2.2.

    | Window | 5 s | Latency floor is "wait for window to close". |
    | Stride | 4 s | 1 s overlap so a boundary event is not halved. |

**Windows are time ranges pointing into the segment files. No video is copied or cut**
(SPEC §2.2, CLAUDE.md invariant 3). Nothing in this module opens a file: a window is
four numbers, and turning it into pixels is ``shared/timecode.py``'s job at the moment
someone actually wants them.

Everything here is pure and clock-free. The tests that matter — that consecutive windows
overlap by exactly one second, that a partial trailing window is never emitted — are
arithmetic, and arithmetic should not need an archive.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from shared.schema import chunk_id_for, to_iso
from shared.timecode import SegmentInfo, list_segments

__all__ = ["Window", "plan_windows", "archive_bounds"]


@dataclass(frozen=True)
class Window:
    """One analysis window: a wall-clock range and its position in the walk.

    ``index`` is the ordinal within this run and exists for one reason —
    ``ingest.gate.warmup_windows`` (SPEC §2.3). The first windows have no reference frame
    to diff against, so they are never gated away, and "first" needs a definition that
    does not depend on wall clock.
    """

    index: int
    t_start: datetime
    t_end: datetime

    @property
    def duration(self) -> float:
        return (self.t_end - self.t_start).total_seconds()

    def chunk_id(self, camera_id: str) -> str:
        """The SPEC §3.1 id. Derived here so the gate path and the caption path cannot
        disagree about what a gated window was called."""
        return chunk_id_for(camera_id, self.t_start, self.t_end)

    def __str__(self) -> str:
        return f"#{self.index} {to_iso(self.t_start)}..{to_iso(self.t_end)}"


def plan_windows(
    t_from: datetime,
    t_to: datetime,
    window_seconds: float,
    stride_seconds: float,
    *,
    start_index: int = 0,
) -> Iterator[Window]:
    """Walk ``[t_from, t_to]`` in windows of ``window_seconds`` every ``stride_seconds``.

    **Only complete windows are emitted.** A trailing 2 s of footage is not a window: the
    VLM needs ~5 frames to tell "reversing toward" from "parked near" (SPEC §2.2), and a
    short chunk would be captioned badly and indexed as though it were not. In live
    ``--follow`` mode the remainder is picked up on the next pass, once the archive has
    grown past it — which is exactly what "wait for the window to close" means.

    Yields lazily. An overnight archive is tens of thousands of windows and there is no
    reason to hold them all.
    """
    if t_from.tzinfo is None or t_to.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError(
            f"window_seconds and stride_seconds must be positive; got "
            f"{window_seconds} and {stride_seconds}"
        )

    start = t_from.astimezone(timezone.utc)
    end = t_to.astimezone(timezone.utc)
    width = timedelta(seconds=window_seconds)
    stride = timedelta(seconds=stride_seconds)

    index = start_index
    cursor = start
    while cursor + width <= end:
        yield Window(index=index, t_start=cursor, t_end=cursor + width)
        index += 1
        cursor += stride


def archive_bounds(
    archive_dir: str | None = None,
    camera_id: str | None = None,
    *,
    exclude_open: bool = False,
) -> tuple[datetime, datetime] | None:
    """The wall-clock span the archive covers, or None when it is empty.

    Derived from filenames alone via ``shared/timecode.py`` — no ffprobe, no decode.

    ``exclude_open`` stops the walk at the end of the last **closed** segment. While the
    recorder is running, the newest file has no moov atom yet and *nothing* in it can be
    decoded — not "a window or two" at the tail, the whole file. A follower that walks
    into it burns every one of those windows as a decode failure and, because the cursor
    only moves forward, never comes back for them once the file closes. Started against
    a live recorder on an empty archive, that is every window there is.

    A segment is treated as closed when a later one exists: the recorder opens the next
    file in the same breath as it closes the previous one, which is the same signal
    ``shared/timecode.py`` uses to derive a segment's effective end.
    """
    segments: list[SegmentInfo] = list_segments(archive_dir, camera_id)
    if not segments:
        return None
    if exclude_open:
        if len(segments) < 2:
            # Only the file currently being written exists; nothing is analysable yet.
            return None
        return segments[0].start, segments[-2].end
    return segments[0].start, segments[-1].end
