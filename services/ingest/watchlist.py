"""Task-aware captioning — the watchlist M1 appends to the caption prompt.

**Why this exists.** M5's stage 1 is ``cosine(task.embedding, caption_vector)`` and its
stage 2 reads ``chunk.caption``. Neither ever sees a pixel. That makes the caption the
*only* substrate standing-task detection has: if the VLM did not happen to mention the
thing a task cares about, the task cannot fire, however plainly the footage shows it.

That is not hypothetical. ``config/tasks.yaml`` carries a comment recording that
``red-tag-person`` had to be reworded from caption vocabulary ("a person wearing a red
wristband") rather than from the event ("having red band on hand", which scored 0.07-0.26
and never once cleared the gate). Writing task descriptions by reverse-engineering the
captioner's word choice is a trap that only gets worse as the prompt changes underneath.

So instead of hoping a general caption mentions the right noun, we *ask*. The active
tasks are appended to the caption prompt as an explicit checklist and the model answers
them by name. Detection stops depending on luck.

**Why the cost is acceptable.** This is prefill, not decode. SPEC §2.5 puts prefill at
~5% of per-chunk cost, so a few extra prompt lines are close to free — where raising
``max_tokens`` would spend decode, which is ~95%. The output budget is untouched.

**Why priming the model is safe here.** Telling a VLM to look for a red wristband biases
it toward reporting one. That is a real effect and the reason ``max_items`` exists. It is
also precisely what the funnel is built to absorb: SPEC §6.2 calls stage 1 "deliberately
loose - over-trigger here and filter later", stage 2 is an LLM re-read, and stage 3
re-watches the footage at 4 fps before anything reaches a human. A false positive costs
one stage-2 call; a false *negative* is an event the system can never recover, because the
window has closed and the caption is all that survives.

**Why a file.** M1 and M5 are separate processes (``services/ingest/__main__.py`` and
``services/monitor/__main__.py``, or M5 wired into the agent server), so M1 cannot read
M5's in-memory registry. Tasks reach it the same way chunks reach the agent's index: a
file, polled by mtime. ``config/tasks.yaml`` is the cold-start seed; the agent server
writes runtime CRUD through to ``ingest.watchlist.path`` so a task created in the UI
reaches the captioner within ``refresh_seconds`` instead of at the next M1 restart.

A missing watchlist file is not an error. It means "no active tasks", the suffix is empty,
and captioning behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

__all__ = ["WatchItem", "Watchlist", "write_watchlist"]


@dataclass(frozen=True)
class WatchItem:
    """One line of the checklist: a task's id and its natural-language description."""

    task_id: str
    describe: str


def _coerce(raw: Any) -> list[WatchItem]:
    """Parse either the watchlist JSON or the ``tasks.yaml`` seed shape.

    Both are ``{"tasks": [...]}`` over mappings carrying ``task_id`` and ``describe``,
    which is why one parser serves both and the seed needs no conversion step.
    """
    if isinstance(raw, dict):
        rows = raw.get("tasks", [])
    elif isinstance(raw, list):
        rows = raw
    else:
        return []

    items: list[WatchItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # `enabled: false` is how the UI disables a task without deleting it. A disabled
        # task must not steer the captioner, or it keeps costing prompt budget and keeps
        # biasing captions for something nobody is watching for.
        if not row.get("enabled", True):
            continue
        task_id = str(row.get("task_id") or "").strip()
        describe = str(row.get("describe") or "").strip()
        if not task_id or not describe:
            continue
        items.append(WatchItem(task_id=task_id, describe=describe))
    return items


def _read(path: Path) -> list[WatchItem]:
    """Load one watchlist source. Never raises — a bad file must not stop ingest.

    Captioning without a checklist degrades to the old behaviour; captioning not at all
    loses footage permanently. The failure is logged and swallowed on purpose.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        LOG.warning("watchlist unreadable", extra={"fields": {"path": str(path), "error": str(exc)}})
        return []

    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml  # local: only the seed path needs it

            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
    except Exception as exc:  # noqa: BLE001 - see docstring
        LOG.warning("watchlist unparseable", extra={"fields": {"path": str(path), "error": str(exc)}})
        return []
    return _coerce(raw)


def write_watchlist(path: Path, tasks: list[Any]) -> None:
    """Publish the live task set for M1 to pick up. Called by whoever owns task CRUD.

    Written atomically via a sibling temp file and ``replace``: M1 polls this path on a
    timer and a torn read would silently drop the checklist for one window.
    """
    rows = []
    for task in tasks:
        get = task.get if isinstance(task, dict) else lambda k, d=None: getattr(task, k, d)  # noqa: E731
        task_id = str(get("task_id", "") or "").strip()
        describe = str(get("describe", "") or "").strip()
        if not task_id or not describe:
            continue
        rows.append(
            {"task_id": task_id, "describe": describe, "enabled": bool(get("enabled", True))}
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"tasks": rows}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    LOG.info("watchlist published", extra={"fields": {"path": str(path), "count": len(rows)}})


class Watchlist:
    """The caption-prompt suffix for the currently active tasks, refreshed by mtime.

    Reads ``path`` (runtime CRUD) and falls back to ``seed_path`` (``config/tasks.yaml``)
    when the former is absent, so a fresh box that has never opened the UI still gets a
    task-aware caption. Both are stat-checked rather than re-parsed, matching how
    ``InMemoryBackend`` picks up index changes across processes.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        seed_path: Path | None = None,
        preamble: str = "",
        max_items: int = 8,
        enabled: bool = True,
    ) -> None:
        self._path = Path(path) if path else None
        self._seed = Path(seed_path) if seed_path else None
        self._preamble = preamble.strip()
        self._max_items = max(0, int(max_items))
        self._enabled = bool(enabled)
        self._stamp: tuple[Any, ...] | None = None
        self._suffix = ""
        self._items: list[WatchItem] = []
        self._loaded = False

    @property
    def items(self) -> list[WatchItem]:
        """The checklist as last refreshed. For logging and tests, not for the prompt."""
        return list(self._items)

    def _stat(self, path: Path | None) -> tuple[Any, ...]:
        if path is None:
            return (None,)
        try:
            st = path.stat()
        except OSError:
            return (None,)
        return (st.st_size, st.st_mtime_ns)

    def refresh(self) -> bool:
        """Re-read if either source changed on disk. Returns True when the suffix moved."""
        if not self._enabled:
            return False
        stamp = self._stat(self._path) + self._stat(self._seed)
        if self._loaded and stamp == self._stamp:
            return False

        items = _read(self._path) if self._path else []
        if not items and self._seed:
            items = _read(self._seed)

        dropped = 0
        if self._max_items and len(items) > self._max_items:
            # The prompt grows one line per task and prefill is not free at every scale.
            # Truncating silently would look like a detection bug months later, so say it.
            dropped = len(items) - self._max_items
            items = items[: self._max_items]
            LOG.warning(
                "watchlist truncated; tasks beyond the cap steer no captions",
                extra={"fields": {"kept": len(items), "dropped": dropped}},
            )

        before = self._suffix
        self._items = items
        self._suffix = self._render(items)
        self._stamp = stamp
        self._loaded = True
        changed = before != self._suffix
        if changed:
            LOG.info(
                "watchlist refreshed",
                extra={"fields": {"count": len(items), "dropped": dropped}},
            )
        return changed

    def _render(self, items: list[WatchItem]) -> str:
        if not items or not self._preamble:
            return ""
        lines = "\n".join(f"- {item.describe}" for item in items)
        return f"\n\n{self._preamble}\n{lines}"

    def suffix(self) -> str:
        """The text to append to ``vlm.prompts.caption``. Empty when there is nothing to watch."""
        if not self._enabled:
            return ""
        self.refresh()
        return self._suffix

    def apply(self, prompt: str) -> str:
        """``prompt`` plus the checklist. The general description always comes first.

        Order is deliberate: the checklist must not displace the four-point description,
        because that description is what the Ask surface searches. Watch is additive.
        """
        return prompt + self.suffix()
