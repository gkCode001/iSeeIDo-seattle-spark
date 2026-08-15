"""``register_task`` and the Watch pane's reads — SPEC §10 D5, §11.3.

**M5 owns the task registry.** D5's resolution is to build ``register_task`` as an
endpoint now so the §11.3 form has something to POST to, and to bind M3 (and the
monitor) to it later — at which point conversational registration ("alert me if that
happens again") costs a tool schema, not new plumbing.

So this module is a seam, not an implementation. :class:`TaskRegistry` is the interface
the server talks to; :class:`SeedTaskRegistry` is the placeholder behind it, which:

* reads ``config/tasks.yaml`` (``monitor.tasks_file``) as the cold-start seed, so a
  fresh boot has tasks without anyone clicking;
* keeps runtime registrations in memory and **does not** write them back — persisting
  them is M5's call, and a half-owned file that two processes both edit is how a demo
  loses its seed;
* leaves ``Task.embedding`` empty. M5 embeds ``describe`` once at registration
  (SPEC §6.2); doing it here would need the index's embedder and would be the wrong
  module deciding when.

The funnel state it reports is honest about being a placeholder: stages are idle and
``source`` says ``seed``. The one part that is real is ``last_fired_ts``, read from the
**action log**, because that is the authoritative history (CLAUDE.md) and it is what
makes the §11.3 cooldown count down truthfully even before M5 exists.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from services.mcp import ActionServer
from shared import config
from shared.schema import ActionKind, Task, to_iso, utcnow

from .telemetry import log_event

__all__ = ["TaskRegistry", "SeedTaskRegistry", "DuplicateTaskError"]


class DuplicateTaskError(ValueError):
    """A ``task_id`` that is already registered. The id is the cooldown key (SPEC §6.1)."""


@runtime_checkable
class TaskRegistry(Protocol):
    """What the server needs from M5. Injected, so binding M5 later is wiring."""

    def tasks(self) -> list[Task]: ...

    def register(self, task: Task) -> Task: ...

    def monitor_state(self) -> dict[str, Any]:
        """Funnel state for the §11.3 Watch pane, shaped like ui/mock/monitor_state.json."""
        ...


class SeedTaskRegistry:
    """Cold-start seed plus in-memory registrations. Replaced by M5, not extended."""

    def __init__(
        self,
        tasks_file: str | Path | None = None,
        actions: ActionServer | None = None,
    ) -> None:
        raw = tasks_file if tasks_file is not None else config.get("monitor.tasks_file")
        self._path = (config.REPO_ROOT / str(raw)).resolve()
        self._actions = actions
        self._lock = threading.Lock()
        self._registered: dict[str, Task] = {}
        # Ids removed this process. A seeded task can be deleted from the pane, but the
        # seed file is the cold-start truth and would bring it back on restart — so the
        # removal is a tombstone, not a rewrite of config/tasks.yaml.
        self._removed: set[str] = set()
        self._seed: list[Task] | None = None

    # -- reads ------------------------------------------------------------------------

    def _seeded(self) -> list[Task]:
        """Parse ``config/tasks.yaml`` once. Read-only — the file is owned elsewhere."""
        if self._seed is not None:
            return self._seed
        seed: list[Task] = []
        if self._path.is_file():
            with self._path.open("r", encoding="utf-8") as fh:
                document = yaml.safe_load(fh) or {}
            for row in document.get("tasks") or []:
                try:
                    seed.append(Task.from_dict(row))
                except Exception as exc:  # noqa: BLE001 — one bad task, not one dead pane
                    log_event(
                        "agent.tasks.seed_skipped",
                        path=str(self._path),
                        task=row.get("task_id") if isinstance(row, dict) else None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        else:
            log_event("agent.tasks.no_seed", path=str(self._path))
        self._seed = seed
        return seed

    def tasks(self) -> list[Task]:
        with self._lock:
            registered = dict(self._registered)
        # Registrations shadow the seed by id, so re-registering a seeded task edits it
        # rather than showing the Watch pane two panels with the same name.
        merged = {task.task_id: task for task in self._seeded()}
        merged.update(registered)
        for gone in self._removed:
            merged.pop(gone, None)
        return list(merged.values())

    # -- writes -----------------------------------------------------------------------

    def register(self, task: Task) -> Task:
        """SPEC §11.3's form target. Rejects a duplicate id rather than shadowing one.

        Not persisted: see the module docstring. ``config/tasks.yaml`` stays the
        cold-start seed and M5 decides what survives a restart.
        """
        with self._lock:
            existing = ({t.task_id for t in self._seeded()} | set(self._registered)) - self._removed
            if task.task_id in existing:
                raise DuplicateTaskError(f"task_id already registered: {task.task_id}")
            self._registered[task.task_id] = task
            self._removed.discard(task.task_id)
        log_event(
            "agent.tasks.registered",
            task_id=task.task_id,
            action=task.action.value,
            window=task.window,
            cooldown=task.cooldown,
            active=task.active,
            persisted=False,
        )
        return task

    def remove(self, task_id: str) -> Task:
        """Delete a task so it stops being evaluated. Returns what was removed.

        A **seeded** task can be removed too, but only for this process: the seed lives
        in ``config/tasks.yaml`` and would come back on restart. Rather than pretend
        otherwise, the removal is recorded as a tombstone that shadows the seed, and the
        log line says ``persisted: false`` — same honesty as ``register``.

        The append-only action log is deliberately untouched. Anything this task already
        fired stays on the Timeline with its reason and its clip: SPEC §6.4's "why did
        you alert at 21:11?" must keep working for a task that no longer exists.
        """
        with self._lock:
            seeded = {t.task_id for t in self._seeded()}
            task = self._registered.get(task_id)
            if task is None:
                task = next((t for t in self._seeded() if t.task_id == task_id), None)
            if task is None or task_id in self._removed:
                raise KeyError(f"no such task: {task_id!r}")
            self._registered.pop(task_id, None)
            if task_id in seeded:
                # Shadow the seed for the life of this process.
                self._removed.add(task_id)
        log_event(
            "agent.tasks.removed",
            task_id=task_id,
            was_seeded=task_id in seeded,
            persisted=False,
        )
        return task

    def update(self, task_id: str, changes: dict[str, Any]) -> Task:
        """Apply a partial change. ``task_id`` itself is not editable — it is the
        cooldown and dedupe key, and moving it would orphan the brakes' history."""
        if "task_id" in changes and changes["task_id"] != task_id:
            raise ValueError(
                "task_id is the cooldown and dedupe key and cannot be edited; remove the "
                "task and register it under the new id if that is what you mean."
            )
        # Resolved BEFORE taking the lock: `tasks()` acquires it too and threading.Lock
        # is not reentrant, so calling it from inside the critical section deadlocks the
        # request thread — the connection just hangs, with no error anywhere.
        current = next((t for t in self.tasks() if t.task_id == task_id), None)
        if current is None:
            raise KeyError(f"no such task: {task_id!r}")
        with self._lock:
            merged = {**current.to_dict(), **changes, "task_id": task_id}
            updated = task_from_payload(merged)
            self._registered[task_id] = updated
            self._removed.discard(task_id)
        log_event("agent.tasks.updated", task_id=task_id, fields=sorted(changes), persisted=False)
        return updated

    # -- the Watch pane ----------------------------------------------------------------

    def monitor_state(self) -> dict[str, Any]:
        """Placeholder funnel state until M5 publishes its own.

        Every stage is idle because nothing here evaluates chunks. ``last_fired_ts`` is
        real — it comes from the action log — so the §11.3 cooldown is a countdown
        against something that actually happened, which is the part of that pane that
        proves a brake rather than asserting one.
        """
        last_fired = self._last_fired_by_task()
        rows: list[dict[str, Any]] = []
        for task in self.tasks():
            rows.append(
                {
                    "task_id": task.task_id,
                    "state": "armed" if task.enabled else "disabled",
                    "in_active_window": True,
                    "stage1": {
                        "score": 0.0,
                        "threshold": float(config.get("monitor.stage1_cosine_threshold")),
                        "matched": False,
                        "chunk_id": None,
                    },
                    "stage2": {
                        "verdict": None,
                        "since": None,
                        "sustain_window_s": task.window,
                        "last_chunk_id": None,
                    },
                    "stage3": {"state": "idle", "job_id": None, "verdict": None},
                    "last_fired_ts": last_fired.get(task.task_id),
                    "cooldown_seconds": task.cooldown,
                    "match_range": None,
                }
            )
        return {
            "generated_at": to_iso(utcnow()),
            "tasks": rows,
            # Legibility, not decoration: a Watch pane full of idle stages should say
            # why, rather than looking like a monitor that is running and seeing nothing.
            "source": "seed",
            "detail": "M5 is not bound yet; stages are idle and last_fired_ts is from the action log",
        }

    def _last_fired_by_task(self) -> dict[str, str]:
        if self._actions is None:
            return {}
        t_to = utcnow()
        t_from = t_to.replace(year=t_to.year - 1)
        latest: dict[str, str] = {}
        for entry in self._actions.read_action_log(t_from, t_to):
            # Amendments carry the original's task_id; they are commentary, not a firing.
            if entry.task_id and entry.parent_id is None:
                latest[entry.task_id] = to_iso(entry.ts)
        return latest


def task_from_payload(payload: dict[str, Any]) -> Task:
    """Build a :class:`~shared.schema.Task` from the §11.3 form's six fields.

    Defaults come from ``config/settings.yaml`` rather than from ``shared/schema.py``'s
    dataclass defaults, so the form and the monitor agree on what "unset" means.
    """
    if not payload.get("task_id"):
        raise ValueError("task_id is required; it is the cooldown and dedupe key")
    if not payload.get("describe"):
        raise ValueError("describe is required; it is what stage 1 embeds")
    action = payload.get("action")
    try:
        kind = ActionKind(action)
    except ValueError as exc:
        raise ValueError(
            f"unknown action {action!r}; expected one of "
            f"{[k.value for k in ActionKind]}"
        ) from exc
    return Task(
        task_id=str(payload["task_id"]),
        describe=str(payload["describe"]),
        window=int(payload.get("window") or config.get("monitor.stage2_sustain_default")),
        action=kind,
        cooldown=int(payload.get("cooldown") or config.get("monitor.default_cooldown_seconds")),
        active=str(payload.get("active") or "00:00-24:00"),
        enabled=bool(payload.get("enabled", True)),
        # M5 embeds `describe` once at registration (SPEC §6.2). Not our call.
        embedding=[],
    )
