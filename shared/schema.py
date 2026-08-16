"""Single source of truth for records that cross a module boundary.

SPEC §3.1 (ChunkRecord), §6.1 (Task), §6.4 + §11.4 (ActionLogEntry).

Every timestamp in this module is timezone-aware UTC. Serialization is ISO 8601 with a
``Z`` suffix. Local time is a UI-layer concern (SPEC §11.5) and must never appear in a
payload produced here.

Stdlib dataclasses on purpose. This module is imported by every service, including ones
running in container images we do not build. It must never be the reason an import
fails.

What this module deliberately does NOT do:

* Map wall-clock time to segment files. That is ``shared/timecode.py`` (SPEC §3.1) and
  it owns the ``segment`` / ``pts_offset`` derivation exclusively.
* Hold tunable numbers. Windows, strides, thresholds and model names live in
  ``config/settings.yaml``. The only constants below are format strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

__all__ = [
    "Tier",
    "ActionKind",
    "ActionStatus",
    "JobState",
    "ChunkRecord",
    "Task",
    "ActionLogEntry",
    "DeepJob",
    "ChatTurn",
    "utcnow",
    "to_iso",
    "from_iso",
    "chunk_id_for",
]


# --------------------------------------------------------------------------------------
# Time helpers — the only correct way to move a timestamp in or out of a payload.
# --------------------------------------------------------------------------------------


def utcnow() -> datetime:
    """Timezone-aware current UTC time. Never use ``datetime.utcnow()``; it is naive."""
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """Serialize to ISO 8601 with a ``Z`` suffix.

    Seconds precision when the value is whole, matching the SPEC §3.1 example exactly;
    full microseconds otherwise. The rule is that ``from_iso(to_iso(dt)) == dt`` always
    holds — window boundaries land on sub-second offsets once stride drift accumulates,
    and a lossy roundtrip here would misalign a chunk against its own pixels. Prettier
    millisecond output is not worth an inexact join.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be timezone-aware UTC")
    dt = dt.astimezone(timezone.utc)
    spec = "seconds" if dt.microsecond == 0 else "microseconds"
    return dt.isoformat(timespec=spec).replace("+00:00", "Z")


def from_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, requiring an explicit UTC designator."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp missing timezone designator: {value!r}")
    return parsed.astimezone(timezone.utc)


def chunk_id_for(camera_id: str, t_start: datetime, t_end: datetime) -> str:
    """Build the SPEC §3.1 chunk id, e.g. ``cam01_20260814T211107_211112``."""
    start = t_start.astimezone(timezone.utc)
    end = t_end.astimezone(timezone.utc)
    return f"{camera_id}_{start:%Y%m%dT%H%M%S}_{end:%H%M%S}"


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------


class Tier(str, Enum):
    """SPEC §3.3. ``LIVE`` is the alert path; ``ROLLUP`` is the search path (D4)."""

    LIVE = "live"
    ROLLUP = "rollup"


class ActionKind(str, Enum):
    """SPEC §4.1 / §6.1. Severity ordering matters — see SPEC §6.3.

    ``SAVE_CLIP`` is low stakes and fires on stage-2 confidence with no verification.
    The other two reach a human: they fire provisionally as ``UNVERIFIED`` and are
    amended when the worker finishes.
    """

    SAVE_CLIP = "save_clip"
    RAISE_ALERT = "raise_alert"
    FILE_TICKET = "file_ticket"
    #: Posts to a Discord channel through AlertBridge (services/mcp/alertbridge.py). The
    #: only action with an effect outside this box, which is why it is the one that most
    #: needs the brakes in front of it: a message cannot be un-posted, and thirty of them
    #: for one event is the SPEC §6.4 failure mode with an audience.
    NOTIFY_DISCORD = "notify_discord"

    @property
    def reaches_a_human(self) -> bool:
        return self is not ActionKind.SAVE_CLIP


class ActionStatus(str, Enum):
    """SPEC §6.3. An entry is written ``UNVERIFIED`` and amended by a *later* entry."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    RETRACTED = "retracted"


class JobState(str, Enum):
    """Lifecycle of a ``deep_analyze`` call (SPEC §5)."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    TIMEOUT = "timeout"
    FAILED = "failed"


# --------------------------------------------------------------------------------------
# ChunkRecord — SPEC §3.1
# --------------------------------------------------------------------------------------


@dataclass
class ChunkRecord:
    """One analysis window. The only join between a text hit and the pixels it came from.

    ``t_start``/``t_end`` are absolute wall clock; ``segment`` + ``pts_offset`` locate
    the same moment inside a recorder file whose PTS restarted at zero. Both are
    required — see CLAUDE.md invariant 2. A record carrying only one of them is a bug,
    not a shortcut.

    When ``gated`` is True the detector found nothing (SPEC §2.3), inference was skipped
    entirely, and ``caption``/``embedding`` are empty. These null records are still
    written: the skip rate is a health metric, and a gap in the record stream is
    indistinguishable from a crashed ingest.
    """

    chunk_id: str
    camera_id: str
    t_start: datetime
    t_end: datetime
    segment: str
    pts_offset: float
    tier: Tier = Tier.LIVE
    gated: bool = False
    caption: str = ""
    embedding: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (self.t_end - self.t_start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "camera_id": self.camera_id,
            "t_start": to_iso(self.t_start),
            "t_end": to_iso(self.t_end),
            "segment": self.segment,
            "pts_offset": self.pts_offset,
            "tier": self.tier.value,
            "gated": self.gated,
            "caption": self.caption,
            "embedding": list(self.embedding),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkRecord:
        return cls(
            chunk_id=d["chunk_id"],
            camera_id=d["camera_id"],
            t_start=from_iso(d["t_start"]),
            t_end=from_iso(d["t_end"]),
            segment=d["segment"],
            pts_offset=float(d["pts_offset"]),
            tier=Tier(d.get("tier", Tier.LIVE.value)),
            gated=bool(d.get("gated", False)),
            caption=d.get("caption", ""),
            embedding=list(d.get("embedding") or []),
        )


# --------------------------------------------------------------------------------------
# Task — SPEC §6.1
# --------------------------------------------------------------------------------------


@dataclass
class Task:
    """A standing task evaluated against every chunk M1 emits.

    ``active`` is a **local** wall-clock window ("18:00-06:00") and may wrap midnight.
    It is the one deliberate exception to UTC-everywhere: a human writing "overnight"
    means their night, not UTC's. Resolve it against the configured display timezone at
    match time, in M5 — never store a UTC-converted copy, because it breaks on the next
    DST transition.

    ``embedding`` is the ``describe`` text embedded once at registration and reused for
    the free stage-1 cosine (SPEC §6.2).
    """

    task_id: str
    describe: str
    window: int
    action: ActionKind
    cooldown: int = 300
    active: str = "00:00-24:00"
    enabled: bool = True
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "describe": self.describe,
            "window": self.window,
            "action": self.action.value,
            "cooldown": self.cooldown,
            "active": self.active,
            "enabled": self.enabled,
            "embedding": list(self.embedding),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            task_id=d["task_id"],
            describe=d["describe"],
            window=int(d["window"]),
            action=ActionKind(d["action"]),
            cooldown=int(d.get("cooldown", 300)),
            active=d.get("active", "00:00-24:00"),
            enabled=bool(d.get("enabled", True)),
            embedding=list(d.get("embedding") or []),
        )


# --------------------------------------------------------------------------------------
# ActionLogEntry — SPEC §6.4, rendered per SPEC §11.4
# --------------------------------------------------------------------------------------


@dataclass
class ActionLogEntry:
    """One row of the append-only action log.

    **Rows are never mutated.** Verification and retraction (SPEC §6.3) append a *new*
    row carrying ``parent_id`` pointing at the original. This is what lets §11.4 render
    a retraction as the original struck through with the amendment beneath it, and what
    makes ``read_action_log`` (SPEC §4.1) able to answer "why did you alert at 21:11?"
    honestly rather than showing a tidied-up history.

    ``t_start``/``t_end`` are the *footage* range this action is about, which is not the
    same as ``ts``, the moment the row was appended. The dedupe brake (SPEC §6.4)
    compares footage ranges; the cooldown brake compares ``ts``.

    ``task_id`` is None for actions M3 fired on a user's behalf rather than M5.
    """

    entry_id: str
    ts: datetime
    action: ActionKind
    t_start: datetime
    t_end: datetime
    status: ActionStatus = ActionStatus.UNVERIFIED
    task_id: str | None = None
    reason: str = ""
    clip_path: str | None = None
    parent_id: str | None = None
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "ts": to_iso(self.ts),
            "action": self.action.value,
            "t_start": to_iso(self.t_start),
            "t_end": to_iso(self.t_end),
            "status": self.status.value,
            "task_id": self.task_id,
            "reason": self.reason,
            "clip_path": self.clip_path,
            "parent_id": self.parent_id,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionLogEntry:
        return cls(
            entry_id=d["entry_id"],
            ts=from_iso(d["ts"]),
            action=ActionKind(d["action"]),
            t_start=from_iso(d["t_start"]),
            t_end=from_iso(d["t_end"]),
            status=ActionStatus(d.get("status", ActionStatus.UNVERIFIED.value)),
            task_id=d.get("task_id"),
            reason=d.get("reason", ""),
            clip_path=d.get("clip_path"),
            parent_id=d.get("parent_id"),
            job_id=d.get("job_id"),
        )


# --------------------------------------------------------------------------------------
# DeepJob — SPEC §5, surfaced by SPEC §4.3 and §11.2
# --------------------------------------------------------------------------------------


@dataclass
class DeepJob:
    """A ``deep_analyze`` request and its eventual result.

    Shared by M3, M4, M5 and the UI, which is why it lives here. The ``job_id`` is
    returned to the caller immediately (CLAUDE.md invariant 4) and everything below
    ``state`` fills in later, over the WebSocket.

    The dedupe key is ``(t_start, t_end, question)`` — SPEC §4.3 requires that an
    impatient user clicking twice does not queue the work twice. The question is part of
    the key deliberately: the same range asked a *different* question is different work,
    and handing back the first job's answer would be a wrong answer rather than merely a
    duplicated one.
    """

    job_id: str
    t_start: datetime
    t_end: datetime
    question: str
    state: JobState = JobState.QUEUED
    requested_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None
    answer: str = ""
    reasoning: str = ""
    confidence: float | None = None
    evidence_clip: str | None = None
    error: str | None = None

    @property
    def elapsed(self) -> float:
        end = self.completed_at or utcnow()
        return (end - self.requested_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "t_start": to_iso(self.t_start),
            "t_end": to_iso(self.t_end),
            "question": self.question,
            "state": self.state.value,
            "requested_at": to_iso(self.requested_at),
            "completed_at": to_iso(self.completed_at) if self.completed_at else None,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "evidence_clip": self.evidence_clip,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeepJob:
        return cls(
            job_id=d["job_id"],
            t_start=from_iso(d["t_start"]),
            t_end=from_iso(d["t_end"]),
            question=d["question"],
            state=JobState(d.get("state", JobState.QUEUED.value)),
            requested_at=from_iso(d["requested_at"]),
            completed_at=from_iso(d["completed_at"]) if d.get("completed_at") else None,
            answer=d.get("answer", ""),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence"),
            evidence_clip=d.get("evidence_clip"),
            error=d.get("error"),
        )


# --------------------------------------------------------------------------------------
# ChatTurn — SPEC §11.4
# --------------------------------------------------------------------------------------


@dataclass
class ChatTurn:
    """One question/answer exchange, persisted to ``data/chats.jsonl``.

    Stores ``job_id``, not just the answer text: the refinement lands *after* the turn
    ends, and a page reload must not lose it (SPEC §11.4).

    ``grounded`` is the §4.2 groundedness verdict — True when the reranked context could
    answer the question alone. It drives the badge that SPEC §11.2 calls the most
    important pixel in the build, so it is persisted rather than recomputed.
    """

    turn_id: str
    ts: datetime
    question: str
    provisional_answer: str = ""
    grounded: bool | None = None
    cited_chunk_ids: list[str] = field(default_factory=list)
    job_id: str | None = None
    latency_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "ts": to_iso(self.ts),
            "question": self.question,
            "provisional_answer": self.provisional_answer,
            "grounded": self.grounded,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "job_id": self.job_id,
            "latency_s": self.latency_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatTurn:
        return cls(
            turn_id=d["turn_id"],
            ts=from_iso(d["ts"]),
            question=d["question"],
            provisional_answer=d.get("provisional_answer", ""),
            grounded=d.get("grounded"),
            cited_chunk_ids=list(d.get("cited_chunk_ids") or []),
            job_id=d.get("job_id"),
            latency_s=d.get("latency_s"),
        )
