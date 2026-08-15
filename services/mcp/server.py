"""The action server — the only thing in the system allowed to change the outside world.

SPEC §6.3 / §6.4, CLAUDE.md invariant 5. Three actions (``save_clip``, ``raise_alert``,
``file_ticket``), three brakes in front of all of them, one append-only log behind them.
There is no direct path to an action that skips this class, and no argument on any method
that disables a brake — the cooldown *duration* is a per-task dial (SPEC §6.1) but the
brake itself always runs, and dedupe is not tunable per call at all.

The demo failure mode is not missing an event. It is firing thirty alerts for one, and
actions cannot be un-fired.

Severity split (SPEC §6.3), which is behaviour rather than decoration:

* ``save_clip`` is low stakes. It fires on stage-2 confidence and is complete on arrival;
  ``ActionResult.awaits_verification`` is False and no worker time is owed.
* ``raise_alert`` and ``file_ticket`` reach a human. They fire *provisionally*, marked
  ``UNVERIFIED``, and M5 is told to queue a stage-3 verification. When the worker returns
  the caller calls :meth:`verify` or :meth:`retract`, which **append** an amendment.

Nothing here mutates a row. Ever. See ``log.py``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from shared import config
from shared.schema import (
    ActionKind,
    ActionLogEntry,
    ActionStatus,
    Task,
    utcnow,
)

from services.mcp.brakes import Brake, BrakeDecision, check_brakes, originating_entries
from services.mcp.clips import (
    ClipCutter,
    FfmpegClipCutter,
    NullClipCutter,
    SegmentResolver,
    build_clip_plan,
    clip_path_for,
)
from services.mcp.log import ActionLog, ResolvedAction

__all__ = ["ActionResult", "ActionServer", "read_action_log", "ffmpeg_cutter_from_config"]

logger = logging.getLogger("mcp.actions")

#: Not in settings.yaml yet — see the report accompanying this module. Both are
#: constructor arguments so a caller can override without touching code.
_FFMPEG_BIN_SETTING = "recorder.ffmpeg_bin"
_CLIP_TIMEOUT_SETTING = "recorder.clip_timeout_seconds"


@dataclass(frozen=True)
class ActionResult:
    """What happened when an action was requested — fired, or which brake stopped it.

    A suppressed request is a normal, expected outcome, not an error. M5 asks on every
    matching chunk and most of those asks are supposed to be refused; raising here would
    turn the working case into exception handling.
    """

    fired: bool
    entry: ActionLogEntry | None = None
    brake: Brake | None = None
    blocked_by: ActionLogEntry | None = None
    detail: str = ""
    #: Every brake that would have stopped this, not only the one named in ``brake``.
    #: A brake that is always masked by another is a brake nobody would notice breaking.
    engaged_brakes: tuple[Brake, ...] = ()

    @property
    def entry_id(self) -> str | None:
        return self.entry.entry_id if self.entry else None

    @property
    def awaits_verification(self) -> bool:
        """True when M5 owes this action a stage-3 verification (SPEC §6.3).

        Only ever True for actions that reach a human. ``save_clip`` is done when it is
        written.
        """
        if self.entry is None:
            return False
        return (
            self.entry.action.reaches_a_human
            and self.entry.status is ActionStatus.UNVERIFIED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fired": self.fired,
            "entry": self.entry.to_dict() if self.entry else None,
            "brake": self.brake.value if self.brake else None,
            "engaged_brakes": [b.value for b in self.engaged_brakes],
            "blocked_by": self.blocked_by.entry_id if self.blocked_by else None,
            "detail": self.detail,
            "awaits_verification": self.awaits_verification,
        }


class ActionServer:
    """Cooldown + time-range dedupe + append-only log, in front of three actions.

    Construct one per process. M3 and M5 each hold their own instance pointed at the same
    file; correctness across them comes from the log's ``flock``, not from sharing this
    object.

    Everything tunable comes from ``config/settings.yaml``. The constructor arguments
    exist so tests can inject a temp path and a fake clock — not so production code can
    pick its own numbers.
    """

    def __init__(
        self,
        *,
        log_path: str | Path | None = None,
        clips_dir: str | Path | None = None,
        camera_id: str | None = None,
        default_cooldown_seconds: float | None = None,
        dedupe_overlap_seconds: float | None = None,
        clock: Callable[[], datetime] = utcnow,
        id_factory: Callable[[], str] | None = None,
        segment_resolver: SegmentResolver | None = None,
        clip_cutter: ClipCutter | None = None,
        clip_container: str | None = None,
        copy_codec: bool | None = None,
    ) -> None:
        self.log = ActionLog(
            log_path if log_path is not None else config.repo_path("paths.action_log")
        )
        self.clips_dir = Path(
            clips_dir if clips_dir is not None else config.repo_path("paths.clips")
        )
        self.camera_id = camera_id if camera_id is not None else str(config.get("camera.id"))
        self.default_cooldown_seconds = float(
            default_cooldown_seconds
            if default_cooldown_seconds is not None
            else config.get("monitor.default_cooldown_seconds")
        )
        self.dedupe_overlap_seconds = float(
            dedupe_overlap_seconds
            if dedupe_overlap_seconds is not None
            else config.get("monitor.dedupe_overlap_seconds")
        )
        self.clip_container = str(
            clip_container if clip_container is not None else config.get("recorder.container")
        )
        self.copy_codec = bool(
            copy_codec if copy_codec is not None else config.get("recorder.copy_codec")
        )
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex[:12])
        self._segment_resolver = segment_resolver
        self._clip_cutter = clip_cutter or NullClipCutter()
        #: Instance-local telemetry for the Watch pane. Not a source of truth — the log
        #: is. Deliberately not persisted, because a second store would drift from it.
        self.stats: dict[str, int] = {
            "fired": 0,
            "suppressed": 0,
            # Per-brake counters sum to more than ``suppressed`` when both engage on the
            # same request. That is the point: a counter stuck at zero is how you find
            # out a brake stopped working.
            "suppressed_cooldown": 0,
            "suppressed_dedupe": 0,
            "amended": 0,
        }

    # ----------------------------------------------------------------------------------
    # The three actions
    # ----------------------------------------------------------------------------------

    def save_clip(
        self,
        t_start: datetime,
        t_end: datetime,
        *,
        task: Task | None = None,
        task_id: str | None = None,
        reason: str = "",
        job_id: str | None = None,
        cooldown_seconds: float | None = None,
    ) -> ActionResult:
        """Low stakes: fires on stage-2 confidence, no verification owed (SPEC §6.3)."""
        return self.fire(
            ActionKind.SAVE_CLIP,
            t_start,
            t_end,
            task=task,
            task_id=task_id,
            reason=reason,
            job_id=job_id,
            cooldown_seconds=cooldown_seconds,
        )

    def raise_alert(
        self,
        t_start: datetime,
        t_end: datetime,
        *,
        task: Task | None = None,
        task_id: str | None = None,
        reason: str = "",
        job_id: str | None = None,
        cooldown_seconds: float | None = None,
    ) -> ActionResult:
        """Reaches a human: fires provisionally as ``UNVERIFIED``, amended on stage 3."""
        return self.fire(
            ActionKind.RAISE_ALERT,
            t_start,
            t_end,
            task=task,
            task_id=task_id,
            reason=reason,
            job_id=job_id,
            cooldown_seconds=cooldown_seconds,
        )

    def file_ticket(
        self,
        t_start: datetime,
        t_end: datetime,
        *,
        task: Task | None = None,
        task_id: str | None = None,
        reason: str = "",
        job_id: str | None = None,
        cooldown_seconds: float | None = None,
    ) -> ActionResult:
        """Reaches a human: fires provisionally as ``UNVERIFIED``, amended on stage 3."""
        return self.fire(
            ActionKind.FILE_TICKET,
            t_start,
            t_end,
            task=task,
            task_id=task_id,
            reason=reason,
            job_id=job_id,
            cooldown_seconds=cooldown_seconds,
        )

    def fire(
        self,
        action: ActionKind,
        t_start: datetime,
        t_end: datetime,
        *,
        task: Task | None = None,
        task_id: str | None = None,
        reason: str = "",
        job_id: str | None = None,
        cooldown_seconds: float | None = None,
    ) -> ActionResult:
        """Request an action. The brakes decide whether it happens.

        ``t_start``/``t_end`` are the **footage** range, not the current time. Passing
        ``utcnow()`` here would defeat the dedupe brake entirely, since every request
        would carry a range nothing had ever collided with.

        ``cooldown_seconds`` overrides the per-task dial from SPEC §6.1 for callers that
        do not hold a ``Task``. It cannot disable the brake — negative values raise, and
        the dedupe brake is not overridable at all.

        The brake check, the clip cut and the append happen inside one lock. Splitting
        them is the race that produces two alerts for one event across M3 and M5.
        """
        if t_end < t_start:
            raise ValueError(f"footage range ends before it starts: {t_start} .. {t_end}")
        resolved_task_id = task.task_id if task is not None else task_id
        cooldown = self._resolve_cooldown(task, cooldown_seconds)
        now = self._clock()

        with self.log.transaction():
            prior = originating_entries(
                self.log.entries(), action=action, task_id=resolved_task_id
            )
            decision: BrakeDecision = check_brakes(
                prior,
                now=now,
                t_start=t_start,
                t_end=t_end,
                cooldown_seconds=cooldown,
                dedupe_pad_seconds=self.dedupe_overlap_seconds,
            )
            if not decision.allowed:
                return self._suppressed(action, resolved_task_id, t_start, t_end, decision)

            clip_path = self._cut_clip(t_start, t_end)
            entry = ActionLogEntry(
                entry_id=self._id_factory(),
                ts=now,
                action=action,
                t_start=t_start,
                t_end=t_end,
                # Everything is written UNVERIFIED. For save_clip that is a statement
                # about provenance, not a promise of a pending check — read
                # ``awaits_verification``, which consults ActionKind.reaches_a_human.
                status=ActionStatus.UNVERIFIED,
                task_id=resolved_task_id,
                reason=reason,
                clip_path=clip_path,
                parent_id=None,
                job_id=job_id,
            )
            self.log.append(entry)

        self.stats["fired"] += 1
        result = ActionResult(fired=True, entry=entry, detail="fired")
        logger.info(
            "action fired",
            extra={
                "fields": {
                    "entry_id": entry.entry_id,
                    "action": action.value,
                    "task_id": resolved_task_id,
                    "t_start": entry.t_start.isoformat(),
                    "t_end": entry.t_end.isoformat(),
                    "cooldown_seconds": cooldown,
                    "clip_path": clip_path,
                    "awaits_verification": result.awaits_verification,
                }
            },
        )
        return result

    # ----------------------------------------------------------------------------------
    # Amendments — append, never mutate
    # ----------------------------------------------------------------------------------

    def verify(
        self,
        entry_id: str,
        *,
        reason: str = "",
        clip_path: str | None = None,
        job_id: str | None = None,
    ) -> ActionLogEntry:
        """Stage 3 agreed. Appends a ``VERIFIED`` row pointing at ``entry_id``."""
        return self.amend(
            entry_id,
            ActionStatus.VERIFIED,
            reason=reason,
            clip_path=clip_path,
            job_id=job_id,
        )

    def retract(
        self,
        entry_id: str,
        *,
        reason: str = "",
        clip_path: str | None = None,
        job_id: str | None = None,
    ) -> ActionLogEntry:
        """Stage 3 disagreed. Appends a ``RETRACTED`` row pointing at ``entry_id``.

        The original stays exactly as written, which is what §11.4 renders struck
        through. Retracting does **not** release the brakes: the task stays in cooldown
        and the footage range stays deduped, because "we were wrong, so let us try again
        immediately" is how one event becomes thirty.
        """
        return self.amend(
            entry_id,
            ActionStatus.RETRACTED,
            reason=reason,
            clip_path=clip_path,
            job_id=job_id,
        )

    def amend(
        self,
        entry_id: str,
        status: ActionStatus,
        *,
        reason: str = "",
        clip_path: str | None = None,
        job_id: str | None = None,
    ) -> ActionLogEntry:
        """Append an amendment against an existing row.

        Amendments carry the original's action, task and footage range so a row read in
        isolation still says what it is about. They are excluded from the brake key —
        commentary on an action is not a second action.
        """
        if status is ActionStatus.UNVERIFIED:
            raise ValueError(
                "an amendment must resolve something: use VERIFIED or RETRACTED. "
                "UNVERIFIED is the state a row is born in, not one it returns to."
            )
        with self.log.transaction():
            target = self.log.entry(entry_id)
            if target is None:
                raise KeyError(f"no such action log entry: {entry_id!r}")
            amendment = ActionLogEntry(
                entry_id=self._id_factory(),
                ts=self._clock(),
                action=target.action,
                t_start=target.t_start,
                t_end=target.t_end,
                status=status,
                task_id=target.task_id,
                reason=reason,
                clip_path=clip_path,
                parent_id=target.entry_id,
                job_id=job_id or target.job_id,
            )
            self.log.append(amendment)

        self.stats["amended"] += 1
        logger.info(
            "action amended",
            extra={
                "fields": {
                    "entry_id": amendment.entry_id,
                    "parent_id": target.entry_id,
                    "status": status.value,
                    "clip_path": clip_path,
                    "reason": reason,
                }
            },
        )
        return amendment

    # ----------------------------------------------------------------------------------
    # Reads — SPEC §4.1 and §11.4 share these rows, so there is exactly one of each
    # ----------------------------------------------------------------------------------

    def read_action_log(self, t_from: datetime, t_to: datetime) -> list[ActionLogEntry]:
        """SPEC §4.1. Raw rows in append order — amendments included, nothing folded.

        This is what the agent introspects and what the Timeline pane renders, so they
        cannot disagree.
        """
        return self.log.read_range(t_from, t_to)

    def resolved_log(self, t_from: datetime, t_to: datetime) -> list[ResolvedAction]:
        """The same window, folded: one row per action with its amendments attached."""
        return self.log.resolved_range(t_from, t_to)

    def resolve(self, entry_id: str) -> ResolvedAction | None:
        """Current status of one action. Use this instead of re-walking ``parent_id``."""
        return self.log.resolve(entry_id)

    def pending_verification(
        self, t_from: datetime, t_to: datetime
    ) -> list[ResolvedAction]:
        """Human-reaching actions in this window that stage 3 still owes a verdict."""
        return [r for r in self.resolved_log(t_from, t_to) if r.awaits_verification]

    # ----------------------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------------------

    def _resolve_cooldown(self, task: Task | None, override: float | None) -> float:
        if override is not None:
            cooldown = float(override)
        elif task is not None:
            cooldown = float(task.cooldown)
        else:
            cooldown = self.default_cooldown_seconds
        if cooldown < 0:
            raise ValueError(f"cooldown must be non-negative, got {cooldown!r}")
        return cooldown

    def _suppressed(
        self,
        action: ActionKind,
        task_id: str | None,
        t_start: datetime,
        t_end: datetime,
        decision: BrakeDecision,
    ) -> ActionResult:
        self.stats["suppressed"] += 1
        for brake in decision.engaged:
            self.stats[f"suppressed_{brake.value}"] += 1
        logger.info(
            "action suppressed",
            extra={
                "fields": {
                    "action": action.value,
                    "task_id": task_id,
                    "brake": decision.brake.value if decision.brake else None,
                    "engaged_brakes": [b.value for b in decision.engaged],
                    "blocked_by": decision.blocked_by.entry_id if decision.blocked_by else None,
                    "t_start": t_start.isoformat(),
                    "t_end": t_end.isoformat(),
                    "detail": decision.detail,
                }
            },
        )
        return ActionResult(
            fired=False,
            entry=None,
            brake=decision.brake,
            blocked_by=decision.blocked_by,
            detail=decision.detail,
            engaged_brakes=decision.engaged,
        )

    def _cut_clip(self, t_start: datetime, t_end: datetime) -> str | None:
        """Cut the evidence clip for a range, or return None if we cannot.

        Runs inside the log lock so that the row and the file it names are written as one
        step. The cutter carries its own timeout for that reason. Returns None rather
        than a path when no resolver is wired or ffmpeg is absent — the log should never
        name a file that does not exist.
        """
        if self._segment_resolver is None:
            return None
        slices = list(self._segment_resolver(t_start, t_end))
        if not slices:
            logger.warning(
                "no archive segments cover the requested range; no clip",
                extra={"fields": {"t_start": t_start.isoformat(), "t_end": t_end.isoformat()}},
            )
            return None
        plan = build_clip_plan(
            slices,
            clip_path_for(
                t_start,
                t_end,
                clips_dir=self.clips_dir,
                camera_id=self.camera_id,
                container=self.clip_container,
            ),
            ffmpeg_bin=str(config.get(_FFMPEG_BIN_SETTING, "ffmpeg")),
            copy_codec=self.copy_codec,
        )
        return self._clip_cutter.cut(plan)


def ffmpeg_cutter_from_config() -> FfmpegClipCutter:
    """The production cutter, wired from settings.

    Never constructed by the test suite — nothing under ``tests/`` shells out. Neither
    setting exists in ``config/settings.yaml`` yet; the defaults here are placeholders
    flagged in this module's handover notes, not a decision.
    """
    return FfmpegClipCutter(
        ffmpeg_bin=str(config.get(_FFMPEG_BIN_SETTING, "ffmpeg")),
        timeout_seconds=float(config.get(_CLIP_TIMEOUT_SETTING, 20.0)),
    )


def read_action_log(
    t_from: datetime,
    t_to: datetime,
    *,
    log_path: str | Path | None = None,
) -> list[ActionLogEntry]:
    """Module-level form of the SPEC §4.1 tool, for M3 to bind directly.

    Read-only, so it is safe to call without owning an ``ActionServer``.
    """
    return ActionServer(log_path=log_path).read_action_log(t_from, t_to)
