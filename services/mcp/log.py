"""The append-only action log — SPEC §6.4, read by M3 per §4.1, rendered per §11.4.

This file is the only history store in the system (CLAUDE.md conventions). Nothing here
ever rewrites, reorders or deletes a line. ``verify`` and ``retract`` are not updates:
they append a *new* ``ActionLogEntry`` whose ``parent_id`` points at the original, which
is what lets §11.4 draw a retraction as the original struck through with the amendment
beneath it. A log you can tidy up is a log that cannot answer "why did you alert at
21:11?" honestly.

Two writers, two processes. M3 fires actions on a user's behalf and M5 fires them from
standing tasks, so every append takes an exclusive ``flock`` on a sidecar lock file and
writes one whole line in one ``write``. The lock is also held across the *read* that
feeds the brake check — checking cooldown and then appending under separate locks is a
race that shows up as exactly the duplicate alert the brakes exist to prevent.

Reads are incremental: the file only ever grows, so bytes before the cursor can never
change and are parsed once. That keeps a per-chunk brake check cheap without introducing
a second copy of the log that could drift from the file.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from shared.schema import ActionLogEntry, ActionStatus

__all__ = ["ActionLog", "ResolvedAction", "resolve_all", "LogCorruptionError"]


class LogCorruptionError(RuntimeError):
    """A line in the log is not a parseable ``ActionLogEntry``.

    Raised rather than skipped. A log with a hole in it is worse than no log, because it
    reads as authoritative right up until the row you needed is the missing one.
    """


# --------------------------------------------------------------------------------------
# Resolved view — fold parent_id chains once, here, so nobody reimplements it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedAction:
    """An original action plus every amendment appended against it, in append order.

    The raw log is the truth and both the UI (§11.4) and the agent (§4.1) render it as
    written. This view answers the *other* question — "what is the current status of
    entry X" — without three call sites each walking ``parent_id`` slightly differently.
    """

    original: ActionLogEntry
    amendments: tuple[ActionLogEntry, ...] = ()

    @property
    def entry_id(self) -> str:
        return self.original.entry_id

    @property
    def latest(self) -> ActionLogEntry:
        """The row that currently speaks for this action."""
        return self.amendments[-1] if self.amendments else self.original

    @property
    def status(self) -> ActionStatus:
        return self.latest.status

    @property
    def retracted(self) -> bool:
        return self.status is ActionStatus.RETRACTED

    @property
    def verified(self) -> bool:
        return self.status is ActionStatus.VERIFIED

    @property
    def awaits_verification(self) -> bool:
        """True only for actions that reach a human and have not been amended yet.

        ``save_clip`` is low stakes and fires on stage-2 confidence with no verification
        (SPEC §6.3), so it never awaits anything — it is not "pending", it is done.
        """
        return self.original.action.reaches_a_human and self.status is ActionStatus.UNVERIFIED

    @property
    def clip_path(self) -> str | None:
        """Most recent clip mentioned anywhere in the chain.

        A ``save_clip`` carries its clip on the original row; an alert usually gains one
        when the deep worker's verification lands (§11.4, "verified · clip attached").
        """
        for entry in reversed((self.original, *self.amendments)):
            if entry.clip_path:
                return entry.clip_path
        return None

    @property
    def job_id(self) -> str | None:
        for entry in reversed((self.original, *self.amendments)):
            if entry.job_id:
                return entry.job_id
        return None

    @property
    def reason(self) -> str:
        """The reason a reader should show: the amendment's if there is one."""
        for entry in reversed((self.original, *self.amendments)):
            if entry.reason:
                return entry.reason
        return ""

    @property
    def resolved_at(self) -> datetime:
        return self.latest.ts

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "original": self.original.to_dict(),
            "amendments": [a.to_dict() for a in self.amendments],
            "status": self.status.value,
            "retracted": self.retracted,
            "awaits_verification": self.awaits_verification,
            "clip_path": self.clip_path,
            "job_id": self.job_id,
            "reason": self.reason,
        }


def _root_id(entry: ActionLogEntry, by_id: dict[str, ActionLogEntry]) -> str:
    """Walk ``parent_id`` to the originating row.

    A parent outside the slice being resolved, or a cycle written by a buggy caller,
    stops the walk rather than looping — a wrong grouping is recoverable, a hang in the
    Timeline pane is not.
    """
    seen: set[str] = {entry.entry_id}
    current = entry
    while current.parent_id is not None:
        parent = by_id.get(current.parent_id)
        if parent is None or parent.entry_id in seen:
            break
        seen.add(parent.entry_id)
        current = parent
    return current.entry_id


def resolve_all(entries: Iterable[ActionLogEntry]) -> list[ResolvedAction]:
    """Fold a flat, append-ordered run of rows into one ``ResolvedAction`` per action.

    Output order is the order the originals were appended, which is the order §11.4
    renders them in.
    """
    rows = list(entries)
    by_id = {e.entry_id: e for e in rows}
    originals: dict[str, ActionLogEntry] = {}
    amendments: dict[str, list[ActionLogEntry]] = {}
    order: list[str] = []

    for entry in rows:
        root = _root_id(entry, by_id)
        if root not in amendments:
            amendments[root] = []
            order.append(root)
        if entry.entry_id == root:
            originals[root] = entry
        else:
            amendments[root].append(entry)

    resolved: list[ResolvedAction] = []
    for root in order:
        original = originals.get(root)
        if original is None:
            # Orphaned amendment: its parent is outside this slice. Promote the earliest
            # amendment so the row is still visible rather than silently dropped.
            chain = amendments[root]
            if not chain:
                continue
            original, rest = chain[0], chain[1:]
            resolved.append(ResolvedAction(original=original, amendments=tuple(rest)))
            continue
        resolved.append(ResolvedAction(original=original, amendments=tuple(amendments[root])))
    return resolved


# --------------------------------------------------------------------------------------
# The log itself
# --------------------------------------------------------------------------------------


class ActionLog:
    """Append-only JSONL at ``paths.action_log``.

    Not a general store. The only mutating operation is ``append``; there is deliberately
    no update, no delete and no compaction, because every one of those would be a way to
    make the log disagree with what actually happened.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._entries = []
        self._cursor = 0
        self._depth = 0

    @property
    def lock_path(self) -> Path:
        """Sidecar lock file. Separate so the lock exists before the log does."""
        return self.path.with_suffix(self.path.suffix + ".lock")

    # -- transaction ---------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[ActionLog]:
        """Hold the cross-process write lock and refresh the in-memory view.

        The brake check must happen inside this, together with the append it authorises.
        Anything less and two processes can both observe "no recent alert" and both fire.
        Re-entrant: nested use by ``append`` inside a caller's transaction is a no-op.
        """
        with self._lock:
            if self._depth > 0:
                self._depth += 1
                try:
                    yield self
                finally:
                    self._depth -= 1
                return
            with open(self.lock_path, "a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                self._depth += 1
                try:
                    self._refresh()
                    yield self
                finally:
                    self._depth -= 1
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # -- reads ---------------------------------------------------------------------

    def _refresh(self) -> None:
        """Parse whatever has been appended since the cursor, by us or by anyone else."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < self._cursor:
            # The file shrank. It is append-only by contract, so this means someone
            # replaced or truncated it out of band; re-read from scratch rather than
            # serving a view that no longer matches the bytes on disk.
            self._entries = []
            self._cursor = 0
        if size == self._cursor:
            return
        with open(self.path, "rb") as handle:
            handle.seek(self._cursor)
            blob = handle.read()
        consumed = 0
        for raw in blob.splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                # Partial trailing line: a writer is mid-append. Leave it for next time.
                break
            consumed += len(raw)
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            try:
                self._entries.append(ActionLogEntry.from_dict(json.loads(text)))
            except Exception as exc:  # noqa: BLE001 - re-raised with the offending line
                raise LogCorruptionError(
                    f"{self.path}: unparseable line at byte {self._cursor + consumed}: {text!r}"
                ) from exc
        self._cursor += consumed

    def entries(self) -> list[ActionLogEntry]:
        """Every row, in append order."""
        with self.transaction():
            return list(self._entries)

    def entry(self, entry_id: str) -> ActionLogEntry | None:
        with self.transaction():
            for candidate in self._entries:
                if candidate.entry_id == entry_id:
                    return candidate
        return None

    def read_range(self, t_from: datetime, t_to: datetime) -> list[ActionLogEntry]:
        """Rows relevant to a wall-clock window — SPEC §4.1 ``read_action_log``.

        "Why did you alert at 21:11?" can mean either *you acted at 21:11* or *something
        happened at 21:11*, and the asker does not know which field they mean. A row
        matches if its ``ts`` falls in the window **or** its footage range overlaps it.

        Amendments travel with their originals in both directions: a retraction appended
        minutes after the alert must still appear under it, and a retraction found inside
        the window must bring its original along or it renders as a strike-through with
        nothing above it.
        """
        with self.transaction():
            rows = list(self._entries)

        by_id = {e.entry_id: e for e in rows}
        matched: set[str] = set()
        for entry in rows:
            if t_from <= entry.ts <= t_to or (entry.t_start < t_to and t_from < entry.t_end):
                matched.add(entry.entry_id)

        # Pull in parents of matched amendments, then children of matched originals.
        roots = {_root_id(by_id[eid], by_id) for eid in matched}
        selected = set(matched) | roots
        for entry in rows:
            if entry.parent_id is not None and _root_id(entry, by_id) in roots:
                selected.add(entry.entry_id)
        return [e for e in rows if e.entry_id in selected]

    def resolved_range(self, t_from: datetime, t_to: datetime) -> list[ResolvedAction]:
        """``read_range`` folded into one row per action."""
        return resolve_all(self.read_range(t_from, t_to))

    def resolve(self, entry_id: str) -> ResolvedAction | None:
        """Current status of one action, amendments included."""
        with self.transaction():
            rows = list(self._entries)
        by_id = {e.entry_id: e for e in rows}
        if entry_id not in by_id:
            return None
        root = _root_id(by_id[entry_id], by_id)
        chain = [e for e in rows if e.entry_id == root or _root_id(e, by_id) == root]
        for resolved in resolve_all(chain):
            if resolved.entry_id == root:
                return resolved
        return None

    # -- the one write -------------------------------------------------------------

    def append(self, entry: ActionLogEntry) -> ActionLogEntry:
        """Append one row. The only mutating operation this class has.

        One ``write`` of one newline-terminated line to an ``O_APPEND`` handle, under the
        exclusive lock, then ``fsync``. Interleaving would produce a line that parses as
        neither action — and the log's whole job is to still be readable after the crash
        that made you want to read it.
        """
        with self.transaction():
            payload = json.dumps(entry.to_dict(), separators=(",", ":"))
            if "\n" in payload:
                raise ValueError("serialized entry contains a newline; would corrupt the log")
            with open(self.path, "ab") as handle:
                handle.write((payload + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            self._entries.append(entry)
            self._cursor = self.path.stat().st_size
        return entry
