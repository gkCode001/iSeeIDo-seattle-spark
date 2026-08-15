"""Structured logging for the recorder.

CLAUDE.md conventions: structured logging, UTC ISO 8601 with a ``Z`` suffix. One line
of JSON per event on stdout, so that ``make ingest`` running beside a 30-hour recorder
produces something greppable rather than something to squint at.

Timestamps come from ``shared.schema`` — there is exactly one correct way to put a
timestamp in a payload and it is not ``datetime.utcnow()``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from shared import config
from shared.schema import to_iso, utcnow

# Keys the logging machinery sets on every record. Anything else a caller attached via
# ``extra=`` is ours and belongs in the event payload.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, service, event, then the caller's fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": to_iso(utcnow()),
            "level": record.levelname,
            "service": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Fallback for ``logging.format: text`` — same fields, easier on human eyes."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        tail = " ".join(f"{k}={v!r}" for k, v in extras.items())
        head = f"{to_iso(utcnow())} {record.levelname:<7} {record.name} {record.getMessage()}"
        return f"{head} {tail}".rstrip()


def configure(service: str) -> logging.Logger:
    """Attach a single stdout handler and return the service logger.

    Idempotent: calling this twice does not double every line, which matters because the
    supervisor and the entry point both want a logger.
    """
    logger = logging.getLogger(service)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = str(config.get("logging.format", "json")).lower()
        handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(str(config.get("logging.level", "INFO")).upper())
    return logger


def event(logger: logging.Logger, level: int, name: str, **fields: Any) -> None:
    """Emit one structured event. ``name`` is a dotted event name, not a sentence."""
    logger.log(level, name, extra=fields)
