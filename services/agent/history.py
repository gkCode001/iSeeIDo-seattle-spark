"""Chat history — ``data/chats.jsonl`` (``paths.chat_log``), SPEC §11.4.

Append-only JSONL, same shape of thing as the action log, for one specific reason:
**refinements arrive after the turn ends.** Persisting only the message text loses the
34-second answer on the next page reload, so the turn persists its *job*, and the job's
own rows are appended here as they land.

Two row kinds share the file, tagged by ``kind``:

* ``{"kind": "turn", ...}`` — ``ChatTurn.to_dict()``
* ``{"kind": "job",  ...}`` — ``DeepJob.to_dict()``

One file rather than two because they are read together (``GET /api/chat/history``
returns ``{turns, jobs}``) and written in the same causal order. Nothing is ever
mutated: a job that goes ``queued → running → done`` appends three rows and the reader
keeps the last one per ``job_id``. That is what makes a crash mid-refinement leave a
readable file instead of a truncated one.

This is *not* the action log. CLAUDE.md is explicit that the action log is the only
history store for actions, and nothing about actions is written here.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path

from shared.schema import ChatTurn, DeepJob

from .telemetry import log_event

__all__ = ["ChatLog", "ChatHistory"]

_KIND = "kind"
_TURN = "turn"
_JOB = "job"


class ChatHistory:
    """What one read of the log produced: turns in append order, jobs by id."""

    def __init__(self, turns: list[ChatTurn], jobs: dict[str, DeepJob]) -> None:
        self.turns = turns
        self.jobs = jobs

    def to_dict(self) -> dict[str, object]:
        """The ``GET /api/chat/history`` payload the UI expects (ui/static/data.js)."""
        return {
            "turns": [t.to_dict() for t in self.turns],
            "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
        }


class ChatLog:
    """Append-only reader/writer for ``paths.chat_log``.

    A ``threading.Lock`` serialises writers inside this process, which is all that is
    needed: M3 is one process and nothing else writes chat turns. (The action log takes
    an ``flock`` because M3 *and* M5 both write it.) Each row is one ``write`` of one
    line, so a reader mid-write sees whole rows.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    # -- writes ---------------------------------------------------------------------

    def append_turn(self, turn: ChatTurn) -> ChatTurn:
        self._append({_KIND: _TURN, **turn.to_dict()})
        return turn

    def append_job(self, job: DeepJob) -> DeepJob:
        """Append the job's current state. Called on every transition, never in place."""
        self._append({_KIND: _JOB, **job.to_dict()})
        return job

    def _append(self, row: dict[str, object]) -> None:
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    # -- reads ----------------------------------------------------------------------

    def read(self, max_turns: int | None = None) -> ChatHistory:
        """Fold the log into ``{turns, jobs}``.

        Turns keep append order, deduped by ``turn_id`` with the last row winning — a
        turn is written once today, but a future amendment should not double the card.
        Jobs keep the newest row per ``job_id``, which is how ``queued → running → done``
        collapses to "done" on reload.

        A corrupt line is skipped and logged rather than raising: chat history is the
        soft log (SPEC §11.4), and losing the pane because one row was half-written
        during a kill would be a worse trade than losing that row.
        """
        turns: dict[str, ChatTurn] = {}
        jobs: dict[str, DeepJob] = {}
        if not self.path.is_file():
            return ChatHistory([], {})

        with self.path.open("r", encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    kind = row.get(_KIND)
                    if kind == _TURN:
                        turn = ChatTurn.from_dict(row)
                        turns[turn.turn_id] = turn
                    elif kind == _JOB:
                        job = DeepJob.from_dict(row)
                        jobs[job.job_id] = job
                    else:
                        raise ValueError(f"unknown row kind {kind!r}")
                except Exception as exc:  # noqa: BLE001 — one bad row, not one bad file
                    log_event(
                        "agent.chatlog.skipped_row",
                        path=str(self.path),
                        line=number,
                        error=f"{type(exc).__name__}: {exc}",
                    )

        ordered = sorted(turns.values(), key=lambda t: (t.ts, t.turn_id))
        if max_turns is not None and len(ordered) > max_turns:
            # The tail, not the head: the UI scrolls to the end and the newest turns are
            # the ones a refinement may still be inbound for.
            ordered = ordered[-max_turns:]
        return ChatHistory(ordered, jobs)

    def turn_ids_by_job(self) -> dict[str, list[str]]:
        """Which turns are waiting on which job — the WebSocket fan-out map, restored."""
        mapping: dict[str, list[str]] = {}
        for turn in self.read().turns:
            if turn.job_id:
                mapping.setdefault(turn.job_id, []).append(turn.turn_id)
        return mapping

    def unfinished_jobs(self) -> list[DeepJob]:
        """Jobs whose last written state was not terminal.

        These are the honest casualties of a restart: the process that was going to
        finish them is gone. The server marks them failed on boot rather than leaving a
        spinner counting up forever.
        """
        from shared.schema import JobState  # noqa: PLC0415 — local to keep the import list honest

        return [
            job
            for job in self.read().jobs.values()
            if job.state in (JobState.QUEUED, JobState.RUNNING)
        ]

    def extend(self, turns: Iterable[ChatTurn], jobs: Iterable[DeepJob]) -> None:
        """Bulk append, for seeding a fixture history in a test or a rehearsal."""
        for turn in turns:
            self.append_turn(turn)
        for job in jobs:
            self.append_job(job)
