"""Wall clock <-> (segment file, PTS offset). The join between a text hit and its pixels.

SPEC §2.1 (recorder naming), §3.1 (chunk record), §5 (deep worker stitching), §11.5 (the
one local-time conversion).

Recorders restart PTS at zero in every segment file, so a PTS value on its own names no
moment in history. The recorder's *filename* carries the segment start time, and that is
the only durable anchor we have — hence CLAUDE.md invariant 2 (a chunk stores wall clock
**and** ``segment`` + ``pts_offset``) and invariant 3 (fetching video takes a time range,
never a filename, because an event at 21:11:58 running 12 s lives in two files).

Everything here is that arithmetic, in one place, so no service re-derives it slightly
differently.

Two things this module is deliberately honest about, because getting them wrong fails
silently — the worst failure mode we have:

**Gaps.** The recorder restarts, the disk fills, the process is killed. The archive then
has holes. ``resolve_range`` never returns "the bits that exist" as though they were the
whole range: the returned span list always tiles the *requested* range end to end, and a
hole appears as a span with ``path is None`` (``span.is_gap``). A caller that ignores the
distinction hands ``None`` to ffmpeg and gets a loud TypeError instead of a quietly short
answer. ``require_complete`` turns the same information into an exception for callers who
would rather not think about it.

**Clock drift.** A "60 s" segment is rarely exactly 60 s. We do not probe real durations:
that needs ``ffprobe``, ffmpeg is not installed on this box yet (CLAUDE.md machine state),
and shelling out per lookup is not free on the deep path. Instead we use a fact that is
already on disk and costs nothing: **the next segment's filename**. The recorder closes
one file and opens the next in the same breath, so segment N effectively runs until
segment N+1 starts, whatever its nominal length. That absorbs drift in both directions up
to ``recorder.max_drift_seconds``; a larger jump is read as a recorder restart, i.e. a
gap. The final segment in the archive has no successor and is credited with the nominal
duration only — an under-claim, which surfaces as a reported gap rather than as footage we
promise and cannot deliver. Stated assumption, not a measurement.

Filename second-resolution matters here: ``%H%M%S`` cannot express 60.4 s, so sub-second
drift accumulates invisibly *inside* a file and ``pts_out`` may land a fraction past true
EOF. ffmpeg clamps to EOF, so the failure is a slightly short clip, never a wrong one.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from shared import config
from shared.schema import to_iso

__all__ = [
    "TimecodeError",
    "MissingFootageError",
    "SegmentInfo",
    "SegmentSpan",
    "segment_start_from_name",
    "segment_name_for",
    "pts_offset_for",
    "list_segments",
    "segment_and_offset",
    "resolve_range",
    "require_complete",
    "gaps",
    "covered_seconds",
    "display_timezone",
    "to_local",
    "format_local",
]

log = logging.getLogger(__name__)

# Fallback for a setting that is not in config/settings.yaml yet — see module docstring.
# CLAUDE.md forbids magic numbers in service code, and this is one wearing a coat: the
# real home for it is ``recorder.max_drift_seconds``. Two seconds is chosen to be larger
# than any plausible segmenter overshoot at 60 s nominal and far smaller than the
# shortest gap worth reporting. Add the key and this constant stops being consulted.
_FALLBACK_MAX_DRIFT_SECONDS = 2.0

_CAMERA_PLACEHOLDER = "{camera_id}"

# strftime directives the recorder pattern may use, and what they look like on disk.
# Anything outside this set is rejected loudly rather than guessed at.
_DIRECTIVE_RE: dict[str, str] = {
    "%Y": r"\d{4}",
    "%y": r"\d{2}",
    "%m": r"\d{2}",
    "%d": r"\d{2}",
    "%H": r"\d{2}",
    "%M": r"\d{2}",
    "%S": r"\d{2}",
    "%j": r"\d{3}",
    "%f": r"\d{1,6}",
}


class TimecodeError(Exception):
    """A time range could not be mapped onto the archive."""


class MissingFootageError(TimecodeError):
    """The requested range is not fully covered by segment files.

    Carries the holes so the caller can say *which* seconds are missing rather than
    "something went wrong". Never swallow this into a shorter range — that is the exact
    silent failure CLAUDE.md invariant 3 exists to prevent.
    """

    def __init__(self, message: str, holes: list[SegmentSpan] | None = None) -> None:
        super().__init__(message)
        self.holes: list[SegmentSpan] = list(holes or [])


# --------------------------------------------------------------------------------------
# Filename <-> segment start
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _compile_pattern(pattern: str) -> tuple[re.Pattern[str], tuple[str, ...]]:
    """Turn ``recorder.filename_pattern`` into a matcher plus its directive order.

    We build the regex rather than reversing ``strptime`` because the pattern mixes a
    ``{camera_id}`` placeholder with strftime directives, and ``strptime`` has no idea
    what to do with the former. The directive tuple lets us hand the captured pieces back
    to ``strptime`` for the actual date arithmetic — leap years and month lengths are not
    a wheel worth reinventing.
    """
    parts: list[str] = ["^"]
    directives: list[str] = []
    i = 0
    seen_camera = False
    while i < len(pattern):
        if pattern.startswith(_CAMERA_PLACEHOLDER, i):
            if seen_camera:
                raise TimecodeError(
                    f"filename_pattern names {_CAMERA_PLACEHOLDER} twice: {pattern!r}"
                )
            seen_camera = True
            parts.append(r"(?P<camera_id>.+?)")
            i += len(_CAMERA_PLACEHOLDER)
        elif pattern[i] == "%":
            directive = pattern[i : i + 2]
            if directive == "%%":
                parts.append("%")
            elif directive in _DIRECTIVE_RE:
                # Named per position so the captures can be read back without having to
                # reason about how many other groups the pattern happened to introduce.
                parts.append(f"(?P<d{len(directives)}>{_DIRECTIVE_RE[directive]})")
                directives.append(directive)
            else:
                raise TimecodeError(
                    f"unsupported strftime directive {directive!r} in filename_pattern "
                    f"{pattern!r}; add it to _DIRECTIVE_RE if the recorder needs it"
                )
            i += 2
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    parts.append("$")
    if not directives:
        raise TimecodeError(f"filename_pattern carries no time directives: {pattern!r}")
    return re.compile("".join(parts)), tuple(directives)


def _pattern() -> str:
    return str(config.get("recorder.filename_pattern"))


def _segment_seconds() -> float:
    return float(config.get("recorder.segment_seconds"))


def _max_drift_seconds() -> float:
    return float(config.get("recorder.max_drift_seconds", _FALLBACK_MAX_DRIFT_SECONDS))


def segment_start_from_name(name: str) -> datetime:
    """Parse a recorder filename back to its start time, as timezone-aware UTC.

    The filename is the source of truth for segment start (SPEC §3.1). Accepts a bare
    name or a full path; the directory part is ignored.
    """
    stem = Path(name).name
    regex, directives = _compile_pattern(_pattern())
    match = regex.match(stem)
    if match is None:
        raise TimecodeError(f"filename does not match recorder.filename_pattern: {stem!r}")
    if "%Y" not in directives and "%y" not in directives:
        raise TimecodeError(f"filename_pattern has no year; cannot place {stem!r} in time")
    # A separator no filename component can contain, so the recomposed value cannot be
    # mis-split by strptime when two fixed-width fields sit flush against each other.
    values = [match.group(f"d{n}") for n in range(len(directives))]
    parsed = datetime.strptime("\x1f".join(values), "\x1f".join(directives))
    return parsed.replace(tzinfo=timezone.utc)


def segment_name_for(camera_id: str, dt: datetime) -> str:
    """Inverse of :func:`segment_start_from_name`.

    ``dt`` must be the segment's **start**, not an arbitrary instant inside it — the
    recorder chooses cut points, we do not, so there is no honest way to round here.
    Sub-second precision is dropped because the pattern cannot express it.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    return dt.astimezone(timezone.utc).strftime(_pattern().replace(_CAMERA_PLACEHOLDER, camera_id))


def pts_offset_for(t: datetime, segment_name: str) -> float:
    """Seconds from the start of ``segment_name`` to ``t`` — the ``pts_offset`` of §3.1.

    Rejects an instant before the segment starts, and one implausibly far past its end
    (nominal duration plus the drift allowance), because both mean the caller matched the
    wrong file and both would otherwise produce a seek that lands somewhere plausible and
    wrong.
    """
    if t.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    start = segment_start_from_name(segment_name)
    offset = (t.astimezone(timezone.utc) - start).total_seconds()
    if offset < 0:
        raise TimecodeError(f"{to_iso(t)} precedes the start of {Path(segment_name).name}")
    limit = _segment_seconds() + _max_drift_seconds()
    if offset > limit:
        raise TimecodeError(
            f"{to_iso(t)} is {offset:.2f}s into {Path(segment_name).name}, past its "
            f"{limit:.2f}s outer bound — wrong segment for this instant"
        )
    return offset


# --------------------------------------------------------------------------------------
# The archive
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentInfo:
    """One file on disk, plus where we believe it ends.

    ``end`` is derived from the *next* file's start when there is one (drift-absorbing),
    otherwise from the nominal duration (drift-blind). ``end_is_nominal`` says which, so a
    caller can tell a measured boundary from an assumed one.
    """

    path: Path
    camera_id: str
    start: datetime
    end: datetime
    end_is_nominal: bool

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def duration(self) -> float:
        return (self.end - self.start).total_seconds()


def _archive_dir(archive_dir: str | Path | None) -> Path:
    return Path(archive_dir) if archive_dir is not None else config.repo_path("paths.archive")


def list_segments(
    archive_dir: str | Path | None = None,
    camera_id: str | None = None,
) -> list[SegmentInfo]:
    """Every segment in the archive, in time order, with its effective end filled in.

    Files that do not match the recorder pattern are ignored (ffmpeg's in-progress
    temporaries, stray clips) but logged at DEBUG — a whole archive that "has no segments"
    is usually a pattern mismatch, and the log line is the fastest way to see that.

    With ``camera_id`` unset every camera present is accepted, but a mixed archive raises:
    interleaving two cameras' segments would produce spans that tile the timeline twice
    and the resulting stitch would be nonsense. One camera today (SPEC §0), so this is a
    tripwire, not a feature.
    """
    directory = _archive_dir(archive_dir)
    if not directory.is_dir():
        raise TimecodeError(f"archive directory does not exist: {directory}")

    regex, _ = _compile_pattern(_pattern())
    found: list[tuple[datetime, str, Path]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = regex.match(path.name)
        if match is None:
            log.debug("ignoring non-segment file in archive: %s", path.name)
            continue
        cam = match.groupdict().get("camera_id") or ""
        if camera_id is not None and cam != camera_id:
            continue
        found.append((segment_start_from_name(path.name), cam, path))

    if not found:
        return []
    cameras = {cam for _, cam, _ in found}
    if len(cameras) > 1:
        raise TimecodeError(
            f"archive {directory} mixes cameras {sorted(cameras)}; pass camera_id to "
            f"disambiguate before stitching"
        )

    found.sort(key=lambda row: row[0])
    nominal = timedelta(seconds=_segment_seconds())
    drift = timedelta(seconds=_max_drift_seconds())

    segments: list[SegmentInfo] = []
    for idx, (start, cam, path) in enumerate(found):
        nominal_end = start + nominal
        if idx + 1 < len(found):
            next_start = found[idx + 1][0]
            # Contiguous recording: whatever the true duration, this file runs until the
            # next one begins. Beyond nominal + drift the recorder restarted, and the
            # interval belongs to nobody — credit only the nominal length and let the
            # remainder surface as a gap.
            if next_start <= nominal_end + drift:
                segments.append(SegmentInfo(path, cam, start, next_start, False))
                continue
        segments.append(SegmentInfo(path, cam, start, nominal_end, True))
    return segments


# --------------------------------------------------------------------------------------
# SegmentSpan — the unit the deep worker actually consumes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentSpan:
    """A contiguous slice of one segment file, or a hole where footage should have been.

    ``pts_in``/``pts_out`` are seconds from that file's PTS zero and are what a decoder
    wants; ``t_start``/``t_end`` are the same interval in wall clock and are what a human,
    a caption and the index want. Both, always — see CLAUDE.md invariant 2.

    A gap span has ``path is None``. It still carries ``t_start``/``t_end`` so the hole can
    be reported precisely, and its PTS fields are zero and meaningless.
    """

    path: Path | None
    segment_start: datetime | None
    pts_in: float
    pts_out: float
    t_start: datetime
    t_end: datetime

    @property
    def is_gap(self) -> bool:
        return self.path is None

    @property
    def segment(self) -> str:
        """Matches ``ChunkRecord.segment``. Empty for a gap."""
        return "" if self.path is None else self.path.name

    @property
    def duration(self) -> float:
        return (self.t_end - self.t_start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "path": None if self.path is None else str(self.path),
            "segment_start": None if self.segment_start is None else to_iso(self.segment_start),
            "pts_in": self.pts_in,
            "pts_out": self.pts_out,
            "t_start": to_iso(self.t_start),
            "t_end": to_iso(self.t_end),
            "is_gap": self.is_gap,
        }


def _gap(t_start: datetime, t_end: datetime) -> SegmentSpan:
    return SegmentSpan(None, None, 0.0, 0.0, t_start, t_end)


def resolve_range(
    t_start: datetime,
    t_end: datetime,
    archive_dir: str | Path | None = None,
    camera_id: str | None = None,
) -> list[SegmentSpan]:
    """Map a wall-clock range onto the archive. **The** function of this module.

    Callers pass a time range and never a filename (CLAUDE.md invariant 3): an event at
    21:11:58 running 12 s is two files, and every caller that special-cased that would get
    it subtly wrong in a different way.

    The returned spans are in time order and **tile ``[t_start, t_end)`` exactly** —
    ``spans[0].t_start == t_start``, ``spans[-1].t_end == t_end``, no overlaps, no
    implicit holes. Missing footage is present as a span with ``is_gap`` True rather than
    omitted, so a short answer can never masquerade as a complete one. Use
    :func:`require_complete` to reject it outright.

    An empty archive, or a range entirely outside it, yields a single gap span covering
    the whole request. That is the correct answer, not an error: "we did not record that"
    is a fact about the world.
    """
    if t_start.tzinfo is None or t_end.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    t0 = t_start.astimezone(timezone.utc)
    t1 = t_end.astimezone(timezone.utc)
    if t1 <= t0:
        raise ValueError(f"empty or inverted range: {to_iso(t0)} .. {to_iso(t1)}")

    spans: list[SegmentSpan] = []
    cursor = t0
    for seg in list_segments(archive_dir, camera_id):
        if seg.end <= cursor:
            continue
        if seg.start >= t1:
            break
        if seg.start > cursor:
            spans.append(_gap(cursor, min(seg.start, t1)))
            cursor = min(seg.start, t1)
            if cursor >= t1:
                break
        span_start = max(cursor, seg.start)
        span_end = min(seg.end, t1)
        spans.append(
            SegmentSpan(
                path=seg.path,
                segment_start=seg.start,
                pts_in=(span_start - seg.start).total_seconds(),
                pts_out=(span_end - seg.start).total_seconds(),
                t_start=span_start,
                t_end=span_end,
            )
        )
        cursor = span_end
        if cursor >= t1:
            break

    if cursor < t1:
        spans.append(_gap(cursor, t1))
    return spans


def gaps(spans: list[SegmentSpan]) -> list[SegmentSpan]:
    """The holes in a resolution, for reporting. Empty list means fully covered."""
    return [s for s in spans if s.is_gap]


def covered_seconds(spans: list[SegmentSpan]) -> float:
    """Seconds of real footage in a resolution — gaps excluded."""
    return sum(s.duration for s in spans if not s.is_gap)


def require_complete(spans: list[SegmentSpan]) -> list[SegmentSpan]:
    """Return ``spans`` if there is no hole in them, else raise :class:`MissingFootageError`.

    For callers whose answer would be wrong rather than merely partial if footage were
    missing — the deep worker answering "did anyone approach the door" cannot say "no"
    from a range it only half read.
    """
    holes = gaps(spans)
    if holes:
        detail = ", ".join(f"{to_iso(h.t_start)}..{to_iso(h.t_end)}" for h in holes)
        raise MissingFootageError(
            f"archive is missing {sum(h.duration for h in holes):.2f}s of the requested "
            f"range: {detail}",
            holes,
        )
    return spans


def segment_and_offset(
    t: datetime,
    archive_dir: str | Path | None = None,
    camera_id: str | None = None,
) -> tuple[str, float]:
    """``(segment, pts_offset)`` for one instant — the pair every ``ChunkRecord`` carries.

    This is the ingest-side entry point for CLAUDE.md invariant 2. Raises
    :class:`MissingFootageError` when ``t`` falls in a hole, because a record whose
    ``segment`` points at a file that does not contain the moment is worse than no record.
    """
    if t.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    moment = t.astimezone(timezone.utc)
    for seg in list_segments(archive_dir, camera_id):
        if seg.start <= moment < seg.end:
            return seg.name, (moment - seg.start).total_seconds()
    raise MissingFootageError(f"no segment covers {to_iso(moment)}")


# --------------------------------------------------------------------------------------
# Local time — SPEC §11.5. The only place local time is produced, anywhere.
# --------------------------------------------------------------------------------------


def display_timezone(tz: str | ZoneInfo | None = None) -> ZoneInfo:
    """Resolve ``ui.display_timezone`` (or an override) to a zoneinfo object.

    A missing tzdata is raised, never defaulted to UTC: a clock that is silently five and
    a half hours out looks exactly like a clock that is right, and the UI is where a human
    decides whether the alert at "21:11" matters.
    """
    if isinstance(tz, ZoneInfo):
        return tz
    name = tz or str(config.get("ui.display_timezone"))
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - depends on host tzdata
        raise TimecodeError(
            f"timezone {name!r} is not available; install tzdata rather than falling back "
            f"to UTC, which would silently shift every displayed time"
        ) from exc


def to_local(dt: datetime, tz: str | ZoneInfo | None = None) -> datetime:
    """UTC -> display timezone. The single conversion helper of SPEC §11.5, called at render.

    DST is a non-event in this direction: an absolute instant maps to exactly one local
    time, ambiguous or not, so nothing here can throw on a fall-back hour or shift on a
    spring-forward one. The dangerous direction is local-naive -> UTC (M5's ``Task.active``
    window), which deliberately does not live here.

    **Render the result; do not compute with it.** Two aware datetimes in the same zone
    compare by wall clock and ignore ``fold`` (PEP 495), so the two distinct instants
    inside a fall-back hour test *equal* as local times. Sorting a timeline or deduping an
    action log on these values silently collapses an hour once a year. Sort in UTC, then
    convert the rows you are about to draw.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    return dt.astimezone(display_timezone(tz))


def format_local(
    dt: datetime,
    fmt: str | None = None,
    tz: str | ZoneInfo | None = None,
) -> str:
    """Render an instant for a human, using ``ui.time_format`` unless told otherwise."""
    return to_local(dt, tz).strftime(fmt or str(config.get("ui.time_format")))
