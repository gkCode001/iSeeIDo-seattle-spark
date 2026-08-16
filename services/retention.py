"""Deleting old footage and the captions that describe it.

The one operation here that nothing else in the system performs: **destroying evidence**.
Everything else appends — the archive grows, the index grows, the action log is
append-only by construction. So this module is deliberately shaped around the fact that
it cannot be undone:

* :func:`plan_retention` reads. It resolves a cutoff, decides exactly which segment files
  and which index rows would go, and returns that with byte counts. It deletes nothing.
* :func:`apply_retention` writes, against a plan a human has seen.

Two calls rather than one because the interesting failure is not "the delete errored", it
is "the delete succeeded and took more than anyone expected". A count in front of the
button is the only thing standing between a mistyped age and the footage the demo is
about to be asked about.

**Why deleting footage is not like deleting a row.** The archive is the only thing the
deep worker can re-read (CLAUDE.md invariant 7). A caption survives its footage perfectly
well as text and will still be retrieved, still be cited, still be answered from — and
the escalation it triggers will find nothing to watch. That asymmetry is why both sides
are swept against the same cutoff here rather than left to two policies that drift.

Three things this deliberately does not touch:

* **The segment ffmpeg has open.** Unlinking it takes the mp4 finalisation with it, and
  the resulting file is undecodable for every window overlapping it. The newest segment
  is always kept, as is anything modified recently — see ``retention.*`` in settings.
* **Evidence clips** under ``paths.clips``. The action log referencing them is
  append-only and SPEC §6.4's "why did you alert at 21:11?" renders from them.
* **The action log and the chat log.** History of what the system *did* is not footage.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from shared import config, timecode
from shared.schema import to_iso, utcnow

__all__ = [
    "DoomedSegment",
    "RetentionPlan",
    "RetentionResult",
    "RetentionSettings",
    "apply_retention",
    "plan_retention",
]

_LOGGER = logging.getLogger("services.retention")


def log_event(event: str, **fields: Any) -> None:
    """One JSON line, same shape as the per-service telemetry modules emit.

    A near-copy of those for the reason ``services/agent/telemetry.py`` states: ``shared/``
    is owned elsewhere. Unlike the others this one is not optional — a sweep that deleted
    47 files and left no record of which is a question nobody can answer afterwards.
    """
    payload = {"event": event, **fields}
    _LOGGER.info(json.dumps(payload, default=str, sort_keys=True))


class DeletableIndex(Protocol):
    """The slice of :class:`services.index.IndexStore` this module needs."""

    def select_before(self, cutoff: datetime) -> list[str]: ...

    def delete(self, chunk_ids: list[str]) -> int: ...


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionSettings:
    """``retention.*`` plus the two recorder numbers the live-segment guard needs."""

    max_age_seconds: float
    min_age_seconds: float
    live_guard_seconds: float
    archive_dir: Path
    camera_id: str

    @classmethod
    def from_config(cls) -> RetentionSettings:
        segment_seconds = float(config.get("recorder.segment_seconds"))
        guard_segments = float(config.get("retention.live_segment_guard_segments"))
        return cls(
            max_age_seconds=float(config.get("retention.max_age_seconds")),
            min_age_seconds=float(config.get("retention.min_age_seconds")),
            live_guard_seconds=segment_seconds * guard_segments,
            archive_dir=config.repo_path("paths.archive"),
            camera_id=str(config.get("camera.id")),
        )


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DoomedSegment:
    """One archive file the plan would unlink, with what it costs to keep."""

    path: Path
    t_start: datetime
    t_end: datetime
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.path.name,
            "t_start": to_iso(self.t_start),
            "t_end": to_iso(self.t_end),
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RetentionPlan:
    """What a sweep would destroy. Produced by reads only; safe to compute and discard."""

    cutoff: datetime
    older_than_seconds: float
    segments: list[DoomedSegment] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    #: Files old enough to sweep that the live-segment guard held back. Surfaced rather
    #: than silently excluded: "delete everything older than 3 h" leaving a 60 s file
    #: behind is correct and looks like a bug unless the reason is on screen.
    #:
    #: A list, not a name. In steady state this is at most the one file ffmpeg has open,
    #: but the guard also catches anything freshly *written* with an old name — a file
    #: source replayed into the archive, a restored backup — and reporting one of several
    #: would misname which footage survived.
    kept_live_segments: list[str] = field(default_factory=list)
    #: Archive directory absent, so there is nothing on disk to sweep. Not an error —
    #: index rows may still be swept — but the UI should say so rather than report zero.
    archive_missing: bool = False

    @property
    def bytes_to_free(self) -> int:
        return sum(segment.size_bytes for segment in self.segments)

    @property
    def is_empty(self) -> bool:
        return not self.segments and not self.chunk_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff": to_iso(self.cutoff),
            "older_than_seconds": self.older_than_seconds,
            "segments": [segment.to_dict() for segment in self.segments],
            "segment_count": len(self.segments),
            "chunk_count": len(self.chunk_ids),
            "bytes_to_free": self.bytes_to_free,
            "kept_live_segments": list(self.kept_live_segments),
            "archive_missing": self.archive_missing,
            "empty": self.is_empty,
        }


@dataclass(frozen=True)
class RetentionResult:
    """What a sweep actually destroyed. ``errors`` is per-file and never fatal."""

    plan: RetentionPlan
    segments_deleted: int
    bytes_freed: int
    chunks_deleted: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff": to_iso(self.plan.cutoff),
            "older_than_seconds": self.plan.older_than_seconds,
            "segments_deleted": self.segments_deleted,
            "bytes_freed": self.bytes_freed,
            "chunks_deleted": self.chunks_deleted,
            "kept_live_segments": list(self.plan.kept_live_segments),
            "errors": list(self.errors),
        }


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------


def resolve_age(
    older_than_seconds: float | None, settings: RetentionSettings | None = None
) -> float:
    """Validate a requested age against ``retention.min_age_seconds``.

    ``None`` means "the configured default". Anything below the floor raises rather than
    clamping: a caller asking to delete the last 30 seconds has made a mistake, and
    silently doing something adjacent to what they asked is how a floor becomes a story
    about the tool deleting the wrong thing.
    """
    resolved = settings or RetentionSettings.from_config()
    age = resolved.max_age_seconds if older_than_seconds is None else float(older_than_seconds)
    if not math.isfinite(age):
        raise ValueError("older_than_seconds must be a finite number of seconds")
    if age < resolved.min_age_seconds:
        raise ValueError(
            f"refusing to delete footage newer than {resolved.min_age_seconds:g} s "
            f"(asked for {age:g} s); the live window, ingest's backlog and any in-flight "
            f"deep job all live in the recent past. See retention.min_age_seconds."
        )
    return age


def plan_retention(
    index: DeletableIndex,
    *,
    older_than_seconds: float | None = None,
    now: datetime | None = None,
    settings: RetentionSettings | None = None,
) -> RetentionPlan:
    """Decide what a sweep would delete. **Reads only.**

    A segment qualifies when it *ends* at or before the cutoff, so a file straddling the
    boundary is kept whole — half its footage is newer than the cutoff and the archive
    has no way to delete half a file. Index rows follow the same rule (``t_end <=
    cutoff``), which is deliberately narrower than the overlap rule retrieval uses: a
    boundary-straddling window is widened into a search and excluded from a deletion.
    """
    resolved = settings or RetentionSettings.from_config()
    age = resolve_age(older_than_seconds, resolved)
    moment = now or utcnow()
    cutoff = moment - timedelta(seconds=age)

    chunk_ids = index.select_before(cutoff)

    if not resolved.archive_dir.is_dir():
        # The recorder has never run here, or the archive lives elsewhere. Index rows can
        # still be swept; reporting this beats reporting "0 files" as if the disk were
        # already clean.
        return RetentionPlan(
            cutoff=cutoff,
            older_than_seconds=age,
            chunk_ids=chunk_ids,
            archive_missing=True,
        )

    segments = timecode.list_segments(resolved.archive_dir, resolved.camera_id)
    guard_after = moment - timedelta(seconds=resolved.live_guard_seconds)

    doomed: list[DoomedSegment] = []
    kept_live: list[str] = []
    for position, segment in enumerate(segments):
        if segment.end > cutoff:
            continue
        # The newest file on disk is the one ffmpeg is writing into. `list_segments` can
        # only give it a nominal end, so with the recorder stopped it eventually ages past
        # any cutoff — and unlinking it mid-write is what leaves an unplayable moov-less
        # segment behind. One kept minute is cheaper than that every time.
        is_newest = position == len(segments) - 1
        try:
            stat = segment.path.stat()
        except OSError:
            # It vanished between listing and stat. Nothing to delete.
            continue
        # A name is a claim about when recording *started*; mtime is evidence about now.
        # They disagree whenever a file source is replayed into the archive.
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        if is_newest or mtime > guard_after:
            kept_live.append(segment.path.name)
            continue
        doomed.append(
            DoomedSegment(
                path=segment.path,
                t_start=segment.start,
                t_end=segment.end,
                size_bytes=stat.st_size,
            )
        )

    plan = RetentionPlan(
        cutoff=cutoff,
        older_than_seconds=age,
        segments=doomed,
        chunk_ids=chunk_ids,
        kept_live_segments=kept_live,
    )
    log_event(
        "retention.planned",
        cutoff=to_iso(cutoff),
        older_than_seconds=age,
        segments=len(doomed),
        chunks=len(chunk_ids),
        bytes=plan.bytes_to_free,
        kept_live_segments=kept_live,
    )
    return plan


# --------------------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------------------


def apply_retention(index: DeletableIndex, plan: RetentionPlan) -> RetentionResult:
    """Execute ``plan``. Irreversible.

    **Index rows go first, then the files.** Both orders can be interrupted; only one of
    them is safe when it is. Captions outliving their footage means the agent cites a
    range the player cannot load and the deep worker cannot re-watch — a confident wrong
    answer. Footage outliving its captions is invisible: unindexed bytes nothing queries.

    A file that will not unlink is recorded in ``errors`` and the sweep continues. Half a
    sweep leaves the system in a state it already handles (the archive has holes and
    ``shared/timecode.py`` reports them); aborting on the first permission error leaves
    the *index* half-swept instead, which nothing handles.
    """
    chunks_deleted = index.delete(plan.chunk_ids)

    errors: list[str] = []
    deleted = 0
    freed = 0
    for segment in plan.segments:
        try:
            segment.path.unlink()
        except FileNotFoundError:
            # Already gone — the sweep is idempotent, which matters because the button
            # can be clicked twice and a plan can outlive the disk it described.
            continue
        except OSError as exc:
            errors.append(f"{segment.path.name}: {exc.strerror or exc}")
            continue
        deleted += 1
        freed += segment.size_bytes

    result = RetentionResult(
        plan=plan,
        segments_deleted=deleted,
        bytes_freed=freed,
        chunks_deleted=chunks_deleted,
        errors=errors,
    )
    log_event(
        "retention.applied",
        cutoff=to_iso(plan.cutoff),
        older_than_seconds=plan.older_than_seconds,
        segments_deleted=deleted,
        bytes_freed=freed,
        chunks_deleted=chunks_deleted,
        errors=len(errors),
    )
    return result
