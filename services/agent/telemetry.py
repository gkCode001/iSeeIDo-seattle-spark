"""Structured logging for M3.

CLAUDE.md: every model call logs model, profile, token counts and wall time — we cannot
tune what we cannot see. The ask model is a model call, so it logs the same shape the
VLM does, and so do the two decisions the demo turns on: the groundedness verdict and
the escalation.

Deliberately not a shared module, and a near-copy of ``services/index/telemetry.py`` for
the same stated reason: ``shared/`` is owned elsewhere. If a common logging helper
appears there, delete both files and import that one.
"""

from __future__ import annotations

import json
import logging
import time
from types import TracebackType
from typing import Any

__all__ = ["log_event", "timed"]

_LOGGER = logging.getLogger("services.agent")


def log_event(event: str, **fields: Any) -> None:
    """Emit one JSON line. ``logging.format: json`` in settings.yaml is the intent."""
    if not _LOGGER.isEnabledFor(logging.INFO):
        return
    payload = {"event": event, **fields}
    _LOGGER.info(json.dumps(payload, default=str, sort_keys=True))


class timed:
    """Context manager that logs ``event`` with ``wall_time_ms`` on exit.

    Extra fields discovered inside the block go on ``.fields``; they are merged into the
    record. Failures are logged too — an ask that 500s mid-demo should not be silent.
    """

    def __init__(self, event: str, **fields: Any) -> None:
        self.event = event
        self.fields: dict[str, Any] = dict(fields)
        self._t0 = 0.0

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    def __enter__(self) -> timed:
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        elapsed_ms = round((time.perf_counter() - self._t0) * 1000.0, 2)
        if exc is not None:
            self.fields["error"] = f"{exc_type.__name__ if exc_type else '?'}: {exc}"
        log_event(self.event, wall_time_ms=elapsed_ms, **self.fields)
