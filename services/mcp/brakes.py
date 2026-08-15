"""The two suppression brakes, as pure functions over prior log entries.

SPEC §6.4, CLAUDE.md invariant 5. Kept free of I/O so they can be reasoned about and
tested without a log file, a clock or a config load. ``ActionServer`` is the only caller;
it holds the log lock while it asks these questions so that the answer cannot go stale
between the check and the append.

Two brakes, two different fields, and the difference is the whole point:

* **Cooldown** compares ``ts`` — the wall-clock moment a row was appended. It answers
  "have we already bothered a human about this task recently?"
* **Dedupe** compares ``t_start``/``t_end`` — the *footage* range the action is about.
  It answers "have we already acted on this moment?" Consecutive analysis windows share
  a second of footage (5 s window, 4 s stride) and will hand us the same event several
  times within a few seconds of wall clock; comparing ``ts`` there would let every one
  of them through the instant the cooldown lapsed.

Neither brake is released by an outcome. A retracted alert still holds both, because the
alternative is a retraction that immediately re-fires — which is the thirty-alerts
failure mode with extra steps.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from shared.schema import ActionKind, ActionLogEntry

__all__ = [
    "Brake",
    "BrakeDecision",
    "originating_entries",
    "ranges_collide",
    "cooldown_blocker",
    "dedupe_blocker",
    "check_brakes",
]


class Brake(str, Enum):
    """Which brake stopped an action. Surfaced to the caller and to the Watch pane."""

    COOLDOWN = "cooldown"
    DEDUPE = "dedupe"


@dataclass(frozen=True)
class BrakeDecision:
    """Outcome of the brake check. ``blocked_by`` is the row that did the blocking.

    ``engaged`` lists **every** brake that would have stopped this action, not just the
    one reported as ``brake``. Both are always evaluated: if the first to be asked
    short-circuited the second, a brake could be silently broken for the whole build and
    the only symptom would be its counter sitting at zero — which is exactly what a
    working cooldown looks like too.
    """

    allowed: bool
    brake: Brake | None = None
    blocked_by: ActionLogEntry | None = None
    detail: str = ""
    engaged: tuple[Brake, ...] = ()


def originating_entries(
    entries: Iterable[ActionLogEntry],
    *,
    action: ActionKind,
    task_id: str | None,
) -> list[ActionLogEntry]:
    """Prior rows that share this action's brake key, in append order.

    The key is ``(task_id, action)``. Two different tasks watching the same doorway are
    two different concerns and each is allowed its own alert; the same task firing twice
    is the thing we are here to stop.

    Amendments (``parent_id`` set) are excluded — they are commentary on an action that
    already happened, not an action of their own, and counting them would let a single
    fire hold the cooldown twice over.
    """
    return [
        e
        for e in entries
        if e.parent_id is None and e.action is action and e.task_id == task_id
    ]


def ranges_collide(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
    pad_seconds: float,
) -> bool:
    """True when two footage ranges are close enough to be the same moment.

    ``pad_seconds`` is ``monitor.dedupe_overlap_seconds``. Ranges collide when they
    actually overlap *or* when the gap between them is smaller than the pad: at a 5 s
    window and a 4 s stride, windows two apart no longer overlap yet still describe one
    event, and a bare intersection test would let the third chunk of a staged event fire
    a second alert.
    """
    if pad_seconds < 0:
        raise ValueError(f"pad_seconds must be non-negative, got {pad_seconds!r}")
    gap = max(
        (b_start - a_end).total_seconds(),
        (a_start - b_end).total_seconds(),
    )
    return gap < pad_seconds


def cooldown_blocker(
    prior: Sequence[ActionLogEntry],
    *,
    now: datetime,
    cooldown_seconds: float,
) -> ActionLogEntry | None:
    """Most recent prior row still inside its cooldown, or None.

    Compares ``ts`` (when we acted), never the footage range.
    """
    if cooldown_seconds < 0:
        raise ValueError(f"cooldown_seconds must be non-negative, got {cooldown_seconds!r}")
    if cooldown_seconds == 0:
        return None
    blocker: ActionLogEntry | None = None
    for entry in prior:
        if (now - entry.ts).total_seconds() < cooldown_seconds:
            if blocker is None or entry.ts >= blocker.ts:
                blocker = entry
    return blocker


def dedupe_blocker(
    prior: Sequence[ActionLogEntry],
    *,
    t_start: datetime,
    t_end: datetime,
    pad_seconds: float,
) -> ActionLogEntry | None:
    """Most recent prior row covering the same footage, or None.

    Compares ``t_start``/``t_end`` (what we acted *about*), never ``ts``. A range fired
    on last week is still fired on: this brake has no expiry, because footage time is
    absolute and a moment cannot stop having been alerted about.
    """
    blocker: ActionLogEntry | None = None
    for entry in prior:
        if ranges_collide(entry.t_start, entry.t_end, t_start, t_end, pad_seconds):
            if blocker is None or entry.ts >= blocker.ts:
                blocker = entry
    return blocker


def check_brakes(
    prior: Sequence[ActionLogEntry],
    *,
    now: datetime,
    t_start: datetime,
    t_end: datetime,
    cooldown_seconds: float,
    dedupe_pad_seconds: float,
) -> BrakeDecision:
    """Run both brakes over the already-keyed ``prior`` rows.

    Both always run; neither short-circuits the other. When both engage, dedupe is the
    one reported, because "we already acted on this exact footage" is a more specific
    answer than "we acted recently" — and it is the answer the Watch pane should show
    while a staged event is still in frame.

    There is no argument that turns either off.
    """
    cool = cooldown_blocker(prior, now=now, cooldown_seconds=cooldown_seconds)
    dupe = dedupe_blocker(prior, t_start=t_start, t_end=t_end, pad_seconds=dedupe_pad_seconds)

    engaged: tuple[Brake, ...] = tuple(
        brake
        for brake, hit in ((Brake.DEDUPE, dupe), (Brake.COOLDOWN, cool))
        if hit is not None
    )
    if not engaged:
        return BrakeDecision(allowed=True, detail="no brake engaged")

    if dupe is not None:
        detail = (
            f"dedupe: footage range overlaps entry {dupe.entry_id} "
            f"({dupe.t_start.isoformat()} .. {dupe.t_end.isoformat()}) "
            f"within {dedupe_pad_seconds:g}s"
        )
        return BrakeDecision(
            allowed=False,
            brake=Brake.DEDUPE,
            blocked_by=dupe,
            detail=detail,
            engaged=engaged,
        )

    assert cool is not None  # noqa: S101 - the `not engaged` branch above ruled this out
    remaining = cooldown_seconds - (now - cool.ts).total_seconds()
    return BrakeDecision(
        allowed=False,
        brake=Brake.COOLDOWN,
        blocked_by=cool,
        detail=(
            f"cooldown: {remaining:.1f}s remaining of {cooldown_seconds:g}s "
            f"since entry {cool.entry_id}"
        ),
        engaged=engaged,
    )
