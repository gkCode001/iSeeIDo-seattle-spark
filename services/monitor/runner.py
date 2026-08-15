"""The thing that makes M5 run — SPEC §6.

The funnel, the brakes and the task registry were all built and tested, and none of it
ever executed: there was no ``__main__`` and no production caller of ``build_monitor``.
Standing tasks rendered in the Watch pane and could never fire. This module is the loop
that was missing.

**How chunks arrive.** SPEC §6 says M5 "subscribes to every chunk M1 emits". M1 and M5
are separate processes, so the subscription is a tail of the index M1 writes — the same
shape as ingest tailing the archive, and for the same reason: the writer owns its file
and the reader keeps a cursor. A push channel between the two would be a second thing to
keep alive during a demo, for a stream that averages one record every four seconds.

**What it deliberately does not do.** It never fires an action itself. Everything goes
through ``services/mcp``'s ActionServer so cooldown, time-range dedupe and the
append-only log apply (CLAUDE.md invariant 5), and it never writes the index.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from shared.schema import ChunkRecord

from services.monitor.funnel import Monitor

logger = logging.getLogger("monitor.runner")

__all__ = ["ChunkSource", "IndexTail", "MonitorRunner"]


class ChunkSource:
    """Where new chunks come from. Injected so tests need no files and no clock."""

    def poll(self) -> list[ChunkRecord]:  # pragma: no cover - interface
        raise NotImplementedError


class IndexTail(ChunkSource):
    """Tail the JSONL corpus M2 persists, returning records not yet seen.

    Cursor is the byte offset, not a timestamp: records are appended in the order ingest
    produced them, and a timestamp cursor would silently skip a chunk written slightly
    out of order after a restart.

    A file that SHRINKS is treated as a rewrite rather than as corruption — the
    in-memory backend rewrites the whole corpus on every insert, so this is normal — and
    the cursor resets. That re-delivers records, which is safe: the funnel's own dedupe
    and the MCP brakes both key on the footage range, so a replayed chunk cannot produce
    a second alert.
    """

    def __init__(self, path: str, *, seek_to_end: bool = True) -> None:
        from pathlib import Path

        self._path = Path(path)
        self._offset = 0
        self._seen: set[str] = set()
        if seek_to_end and self._path.is_file():
            # Start from *now*: on a first run the archive may hold hours of history, and
            # replaying it would evaluate standing tasks against footage nobody is
            # watching — firing alerts for events that ended yesterday.
            self._offset = self._path.stat().st_size

    def poll(self) -> list[ChunkRecord]:
        if not self._path.is_file():
            return []
        size = self._path.stat().st_size
        if size < self._offset:
            logger.info(
                "index shrank; re-reading from the start",
                extra={"fields": {"was": self._offset, "now": size}},
            )
            self._offset = 0
        if size == self._offset:
            return []
        records: list[ChunkRecord] = []
        with self._path.open("r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            for line in fh:
                if not line.endswith("\n"):
                    # A partial line: the writer is mid-append. Leave the cursor before
                    # it and pick it up whole on the next pass.
                    break
                self._offset += len(line.encode("utf-8"))
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    record = ChunkRecord.from_dict(json.loads(line))
                except Exception as exc:  # noqa: BLE001 - one bad row, not one dead loop
                    logger.warning("skipping unreadable index row: %s", exc)
                    continue
                if record.chunk_id in self._seen:
                    continue
                self._seen.add(record.chunk_id)
                records.append(record)
        records.sort(key=lambda r: (r.t_start, r.chunk_id))
        return records


class MonitorRunner:
    """Drive the funnel: observe new chunks, then collect stage-3 verdicts.

    Runs on one thread. The funnel is not thread-safe and does not need to be — a stream
    of one chunk every four seconds is not a concurrency problem, and a single thread
    means the Watch pane's funnel state is always a consistent snapshot rather than a
    half-applied one.
    """

    def __init__(
        self,
        monitor: Monitor,
        source: ChunkSource,
        *,
        poll_interval: float = 2.0,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        on_action: Callable[[dict[str, Any]], None] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._monitor = monitor
        self._source = source
        self._interval = poll_interval
        self._on_state = on_state
        self._on_action = on_action
        self._log = logger_ or logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.chunks_seen = 0
        self.actions_fired = 0

    # -- one pass, so tests never need a thread or a sleep ------------------------------

    def tick(self) -> int:
        """One poll. Returns how many chunks were observed."""
        try:
            chunks = self._source.poll()
        except Exception as exc:  # noqa: BLE001 - a bad read must not kill the monitor
            self._log.warning("chunk source failed: %s", exc)
            return 0

        fired = 0
        if chunks:
            self.chunks_seen += len(chunks)
            # A LIST, even though the tail usually yields one (invariant 9).
            for outcome in self._monitor.observe(chunks):
                # `fired` is a property on FunnelOutcome; the row itself is action.entry.
                if not outcome.fired:
                    continue
                fired += 1
                entry = outcome.action.entry if outcome.action else None
                if entry is not None and self._on_action is not None:
                    with _swallow(self._log, "action callback"):
                        self._on_action(entry.to_dict())
                self._log.info(
                    "action fired",
                    extra={"fields": {
                        "task_id": outcome.task_id,
                        "action": entry.action.value if entry else None,
                        "entry_id": entry.entry_id if entry else None,
                    }},
                )

        # Stage 3 lands asynchronously; collecting it is what turns UNVERIFIED into
        # VERIFIED or a retraction (SPEC §6.3).
        try:
            self._monitor.pump_verifications()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("verification pump failed: %s", exc)

        self.actions_fired += fired
        if self._on_state is not None and (chunks or fired):
            with _swallow(self._log, "state callback"):
                self._on_state(self._monitor.state().to_dict())
        return len(chunks)

    # -- background loop ----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()
        self._log.info("monitor runner started")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None


class _swallow:
    """Context manager: log and continue. A UI callback must never stop the funnel."""

    def __init__(self, log: logging.Logger, what: str) -> None:
        self._log, self._what = log, what

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: Any) -> bool:
        if exc is not None:
            self._log.warning("%s failed: %s", self._what, exc)
        return True
