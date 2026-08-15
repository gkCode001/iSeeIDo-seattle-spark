"""The standing-task registry — SPEC §6.1, §11.3, and SPEC §10 D5.

Two ways a task arrives, one code path:

* ``config/tasks.yaml`` is the **cold-start seed** so a fresh boot has tasks without
  anyone clicking (SPEC §11.3). This module reads that file and never writes it.
* ``register_task`` is the runtime path. The §11.3 form POSTs the six §6.1 fields to
  M3's endpoint, which delegates here; binding M3's conversational registration ("alert
  me if that happens again") to the same call later is a tool schema, not new plumbing.

**The description is embedded once, at registration.** That is what makes stage 1 free
(SPEC §6.2): every chunk that arrives afterwards costs one dot product per task, not one
model call. A task whose ``embedding`` is empty at match time is a task that silently
never matches, so registration embeds eagerly and refuses to store a task it could not
embed.

Validation happens here rather than at match time. A task with an unparseable ``active``
window should fail the POST that created it, in front of the person who typed it — not at
03:00, once, into a log nobody is reading.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from shared.schema import ActionKind, Task

from services.index.embedding import Embedder

from services.monitor.active import ActiveWindow, parse_active_window

__all__ = ["TaskRegistry", "TaskRegistrationError", "load_task_seed"]

logger = logging.getLogger("monitor.registry")


class TaskRegistrationError(ValueError):
    """A task was rejected at registration. Never raised at match time."""


def load_task_seed(path: str | Path) -> list[Task]:
    """Read ``config/tasks.yaml`` into unembedded :class:`Task` objects.

    Read-only, by design: tasks registered at runtime live in the running monitor, and
    persisting them back into the seed is a deliberate separate decision (see the header
    comment in ``config/tasks.yaml``). A missing seed file is not an error — a box that
    has only ever been given tasks through the form is a legitimate state — but an empty
    registry is logged, because "the monitor fires nothing" and "the monitor watches
    nothing" look identical from the Watch pane.
    """
    seed = Path(path)
    if not seed.is_file():
        logger.warning(
            "task seed file absent; starting with no standing tasks",
            extra={"fields": {"tasks_file": str(seed)}},
        )
        return []
    with seed.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TaskRegistrationError(f"task seed is not a mapping: {seed}")
    rows = data.get("tasks") or []
    if not isinstance(rows, list):
        raise TaskRegistrationError(f"'tasks:' in {seed} is not a list")
    return [_task_from_mapping(row, source=str(seed)) for row in rows]


def _task_from_mapping(row: Any, *, source: str) -> Task:
    if not isinstance(row, Mapping):
        raise TaskRegistrationError(f"task entry in {source} is not a mapping: {row!r}")
    missing = [k for k in ("task_id", "describe", "window", "action") if k not in row]
    if missing:
        raise TaskRegistrationError(
            f"task in {source} is missing required SPEC §6.1 field(s): {', '.join(missing)}"
        )
    try:
        # shared/schema.py owns the shape. Parsing it anywhere else is how two modules end
        # up disagreeing about what a Task is.
        return Task.from_dict(dict(row))
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskRegistrationError(f"task in {source} is malformed: {exc}") from exc


class TaskRegistry:
    """Tasks plus their once-computed description embeddings, in registration order.

    Not thread-safe by itself. M5 evaluates chunks on one thread (one camera, one stream
    — SPEC §0) and registration arrives from M3's HTTP handler; if those ever become
    genuinely concurrent, wrap ``register``/``tasks`` rather than making every reader
    take a lock it does not need.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._tasks: dict[str, Task] = {}
        self._windows: dict[str, ActiveWindow] = {}

    # ----------------------------------------------------------------------------------
    # Registration
    # ----------------------------------------------------------------------------------

    def load_seed(self, path: str | Path) -> list[Task]:
        """Register every task in the cold-start seed. Returns what was registered."""
        registered = [self.register(task) for task in load_task_seed(path)]
        logger.info(
            "task seed loaded",
            extra={"fields": {"tasks_file": str(path), "count": len(registered)}},
        )
        return registered

    def register(self, task: Task | Mapping[str, Any]) -> Task:
        """Validate, embed ``describe`` once, and store. SPEC §6.2 stage 1 depends on it.

        Returns the stored task — the caller's object is not mutated, so an M3 handler
        can hand its request payload straight in without acquiring a second owner of the
        same object.
        """
        candidate = (
            task if isinstance(task, Task) else _task_from_mapping(task, source="register_task")
        )
        self._validate(candidate)
        if candidate.task_id in self._tasks:
            raise TaskRegistrationError(f"task_id already registered: {candidate.task_id!r}")

        # embed_query, not embed_passages: the task description plays the role of the
        # question and the caption is the document (SPEC §3.4). nemoretriever is
        # asymmetric and encodes the two differently; the hashing stand-in is symmetric,
        # so getting this wrong today would cost nothing and cost recall the day NGC
        # credentials arrive.
        embedding = list(self._embedder.embed_query(candidate.describe))
        if not embedding:
            raise TaskRegistrationError(
                f"embedder returned no vector for {candidate.task_id!r}; a task with no "
                f"embedding silently never matches"
            )
        stored = replace(candidate, embedding=embedding)
        self._tasks[stored.task_id] = stored
        self._windows[stored.task_id] = parse_active_window(stored.active)
        logger.info(
            "task registered",
            extra={
                "fields": {
                    "task_id": stored.task_id,
                    "action": stored.action.value,
                    "window_seconds": stored.window,
                    "cooldown_seconds": stored.cooldown,
                    "active": stored.active,
                    "enabled": stored.enabled,
                    "embed_model": self._embedder.model,
                    "embed_dims": len(embedding),
                }
            },
        )
        return stored

    def _validate(self, task: Task) -> None:
        if not task.task_id or not task.task_id.strip():
            raise TaskRegistrationError("task_id must be a non-empty slug")
        if not task.describe or not task.describe.strip():
            raise TaskRegistrationError(
                f"{task.task_id!r} has no describe text; stage 1 has nothing to embed"
            )
        if task.window < 0:
            raise TaskRegistrationError(f"{task.task_id!r} has a negative sustain window")
        if task.cooldown < 0:
            raise TaskRegistrationError(f"{task.task_id!r} has a negative cooldown")
        if not isinstance(task.action, ActionKind):
            raise TaskRegistrationError(f"{task.task_id!r} has an unknown action")
        # Raises ActiveWindowError, which is a ValueError, in front of whoever typed it.
        parse_active_window(task.active)

    # ----------------------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------------------

    def tasks(self) -> list[Task]:
        """Every registered task, enabled or not, in registration order.

        Disabled tasks are included: SPEC §11.3 renders them with a DISABLED badge, and a
        task that vanishes from the pane when it is switched off looks like a task that
        was deleted.
        """
        return list(self._tasks.values())

    def enabled_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.enabled]

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def remove(self, task_id: str) -> Task:
        """Delete a task. Returns the task that was removed.

        The task stops being evaluated immediately: no stage-1 cosine, no promotion, no
        action. What it does NOT do is touch the append-only action log — anything this
        task already fired stays on the Timeline, with its reason and its clip.
        SPEC §6.4's "why did you alert at 21:11?" has to keep working for a task that no
        longer exists, and deleting history to tidy up a task list is exactly the kind of
        rewriting an append-only log exists to prevent.

        Raises ``KeyError`` for an unknown id rather than returning quietly: "delete that
        one" silently doing nothing is worse than an error, because the operator walks
        away believing the task is gone.
        """
        task = self._tasks.pop(task_id, None)
        if task is None:
            raise KeyError(f"no such task: {task_id!r}")
        self._windows.pop(task_id, None)
        logger.info("task removed", extra={"fields": {"task_id": task_id}})
        return task

    def update(self, task_id: str, changes: Mapping[str, Any]) -> Task:
        """Apply a partial change to a registered task. Returns the stored result.

        ``describe`` is re-embedded when it changes — stage 1 matches against that vector
        (SPEC §6.2), so a task whose description was edited but whose embedding was not
        would keep matching the old wording, which is the sort of divergence nobody
        thinks to look for.

        ``task_id`` cannot be changed here: it is the cooldown and dedupe key, and moving
        it would orphan the brakes' history for the event currently in flight. Remove and
        re-register if you really mean to rename.
        """
        current = self._tasks.get(task_id)
        if current is None:
            raise KeyError(f"no such task: {task_id!r}")
        if "task_id" in changes and changes["task_id"] != task_id:
            raise TaskRegistrationError(
                "task_id is the cooldown and dedupe key and cannot be edited; remove the "
                "task and register it under the new id if that is what you mean."
            )
        merged = {**current.to_dict(), **dict(changes), "task_id": task_id}
        # Drop the stored vector when the wording changed, so register() recomputes it.
        if merged.get("describe") != current.describe:
            merged["embedding"] = []
        else:
            merged["embedding"] = list(current.embedding)
        self._tasks.pop(task_id, None)
        self._windows.pop(task_id, None)
        try:
            return self.register(merged)
        except Exception:
            # Put the original back: a rejected edit must not delete the task.
            self._tasks[task_id] = current
            self._windows[task_id] = parse_active_window(current.active)
            raise

    def window_for(self, task_id: str) -> ActiveWindow:
        """The parsed ``active`` window. Parsed at registration, never at match time."""
        try:
            return self._windows[task_id]
        except KeyError:  # pragma: no cover - a caller holding an unregistered id
            raise KeyError(f"no such task: {task_id!r}") from None

    def set_enabled(self, task_id: str, enabled: bool) -> Task:
        """Flip a task on or off without losing its embedding or its funnel history."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"no such task: {task_id!r}")
        updated = replace(task, enabled=bool(enabled))
        self._tasks[task_id] = updated
        logger.info(
            "task enabled state changed",
            extra={"fields": {"task_id": task_id, "enabled": updated.enabled}},
        )
        return updated

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._tasks
