"""Structured logging for the importer.

Same shape and same reasoning as ``services/ingest/telemetry.py``: one JSON line per
event, its own logger name so an import run is greppable apart from the live path. An
import is the one operation that writes into the archive without the recorder, so what it
placed and where needs to be in the log rather than only on disk.
"""

from __future__ import annotations

import json
import logging
from typing import Any

__all__ = ["log_event"]

_LOGGER = logging.getLogger("services.importer")


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one JSON line. ``logging.format: json`` in settings.yaml is the intent."""
    if not _LOGGER.isEnabledFor(level):
        return
    payload = {"event": event, **fields}
    _LOGGER.log(level, json.dumps(payload, default=str, sort_keys=True))
