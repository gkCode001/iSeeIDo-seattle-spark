"""Structured logging for M1.

CLAUDE.md: every VLM call logs model, profile, token counts and wall time — we cannot
tune what we cannot see. The gate is the same story for a different reason: SPEC §2.3
says to *log the skip rate*, because a mistuned gate does not fail, it just quietly
stops being real-time.

Deliberately not a shared module, for the same reason ``services/index/telemetry.py``
is not: ``shared/`` is owned elsewhere. If a common logging helper ever appears there,
delete both files and import that one.
"""

from __future__ import annotations

import json
import logging
import time
from types import TracebackType
from typing import Any

__all__ = ["log_event", "timed"]

_LOGGER = logging.getLogger("services.ingest")


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one JSON line. ``logging.format: json`` in settings.yaml is the intent."""
    if not _LOGGER.isEnabledFor(level):
        return
    payload = {"event": event, **fields}
    _LOGGER.log(level, json.dumps(payload, default=str, sort_keys=True))


class timed:
    """Context manager that logs ``event`` with ``wall_time_ms`` on exit.

    Extra fields discovered inside the block go on ``.fields`` and are merged into the
    record. Failures are logged too: a window that dies on a corrupt segment should show
    up as an event, not as a hole in the counters.
    """

    def __init__(self, event: str, **fields: Any) -> None:
        self.event = event
        self.fields: dict[str, Any] = dict(fields)
        self._t0 = 0.0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def __enter__(self) -> timed:
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed_ms = round(self.elapsed_ms, 2)
        level = logging.INFO
        if exc is not None:
            self.fields["error"] = f"{exc_type.__name__ if exc_type else '?'}: {exc}"
            level = logging.WARNING
        log_event(self.event, level=level, wall_time_ms=elapsed_ms, **self.fields)
