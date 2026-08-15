"""The Watch pane's contract — SPEC §11.3, shape per ``ui/mock/monitor_state.json``.

The three stages of SPEC §6.2 are invisible by default, and SPEC §6.4 warns that the demo
failure mode is firing thirty alerts for one. Showing the funnel per task, with the
cooldown running, fixes both: the brake gets *proved* on stage rather than asserted.

This is live state, not a persisted record, so it is not in ``shared/schema.py``. It is
still a dataclass rather than a hand-built dict because the UI is already written against
these field names and a typo in a dict literal is a blank panel with no error anywhere.

**Absolute timestamps only. Never remaining-seconds.** Every countdown the UI draws is
derived by the UI::

    cooldown remaining = cooldown_seconds - (now - last_fired_ts)
    sustain elapsed    = now - stage2.since

A server that sent "247 s remaining" would be sending a number that was already stale
when it left, and a slow poll or a paused tab would then make the brake look *shorter*
than it is — the one direction of error that matters, because it makes a working brake
look broken while it is holding. ``cooldown_seconds`` and ``sustain_window_s`` are
durations, not countdowns: they are the dial's setting and do not tick.

M3 owns the HTTP route. This module is a plain function over plain data so that binding
``GET /api/monitor/state`` to it is ``json.dumps(monitor.state().to_dict())``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from shared.schema import to_iso, utcnow

__all__ = [
    "Stage1State",
    "Stage2State",
    "Stage3State",
    "TimeRange",
    "TaskFunnelState",
    "MonitorState",
]


def _iso_or_none(dt: datetime | None) -> str | None:
    return None if dt is None else to_iso(dt)


@dataclass(frozen=True)
class TimeRange:
    """A footage range, in UTC. The Watch pane scrubs the player to it."""

    t_start: datetime
    t_end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {"t_start": to_iso(self.t_start), "t_end": to_iso(self.t_end)}


@dataclass(frozen=True)
class Stage1State:
    """Embedding match — free, every chunk. ``threshold`` is echoed so the pane can draw
    the gate without re-reading settings, and so a threshold changed at runtime shows."""

    score: float = 0.0
    threshold: float | None = None
    matched: bool = False
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "matched": self.matched,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class Stage2State:
    """LLM confirm plus the sustain window.

    ``since`` is the **footage** start of the current run of consecutive matches, not the
    wall-clock moment we noticed. On the live path those are within a window of each
    other; on replayed footage they are not, and the sustain bar should measure the event
    rather than the operator's patience.
    """

    verdict: str | None = None  # "match" | "no_match" | None
    since: datetime | None = None
    sustain_window_s: int = 0
    last_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "since": _iso_or_none(self.since),
            "sustain_window_s": self.sustain_window_s,
            "last_chunk_id": self.last_chunk_id,
        }


@dataclass(frozen=True)
class Stage3State:
    """Worker verify. ``state`` is the job's lifecycle; ``verdict`` is what it decided.

    They are separate because a job can be ``done`` and inconclusive — the worker ran and
    told us nothing decisive — and collapsing that into "no verdict yet" would leave the
    pane spinning forever on a job that has finished.
    """

    state: str = "idle"  # idle | queued | running | done | failed
    job_id: str | None = None
    verdict: str | None = None  # "verified" | "retracted" | None

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "job_id": self.job_id, "verdict": self.verdict}


@dataclass(frozen=True)
class TaskFunnelState:
    """One card in the Watch pane. One task, all three stages, and the brake."""

    task_id: str
    state: str  # armed | matching | cooling | out_of_window | disabled
    in_active_window: bool
    stage1: Stage1State
    stage2: Stage2State
    stage3: Stage3State
    last_fired_ts: datetime | None
    cooldown_seconds: float
    match_range: TimeRange | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "in_active_window": self.in_active_window,
            "stage1": self.stage1.to_dict(),
            "stage2": self.stage2.to_dict(),
            "stage3": self.stage3.to_dict(),
            "last_fired_ts": _iso_or_none(self.last_fired_ts),
            "cooldown_seconds": self.cooldown_seconds,
            "match_range": None if self.match_range is None else self.match_range.to_dict(),
        }


@dataclass(frozen=True)
class MonitorState:
    """What ``GET /api/monitor/state`` returns.

    ``generated_at`` is what the mock-mode rebaser in ``ui/static/data.js`` anchors to; in
    live mode it is simply the stamp the pane prints so a frozen panel is visibly frozen
    rather than merely wrong.
    """

    generated_at: datetime = field(default_factory=utcnow)
    tasks: tuple[TaskFunnelState, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": to_iso(self.generated_at),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def task(self, task_id: str) -> TaskFunnelState | None:
        for row in self.tasks:
            if row.task_id == task_id:
                return row
        return None
