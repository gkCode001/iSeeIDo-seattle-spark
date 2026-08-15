"""Recorder — the ffmpeg segmenter that writes the archive (SPEC §2.1).

Runs independently of every AI component. If ingest crashes, if the VLM OOMs, if Milvus
never comes up, recording continues: the archive is what M4 re-watches, and it is the
only copy of the pixels the system ever gets.

Layout follows one rule — **command construction is separate from command execution**,
because ffmpeg is not installed on this box yet (CLAUDE.md machine state) and the module
must still import and still be testable:

* ``settings`` — configuration, source classification. Pure.
* ``command``  — the argv list. Pure; this is what the tests assert on.
* ``supervisor`` — spawning, watching, restarting. The only module that needs a process.
* ``log``      — structured JSON events.

Run it: ``python3 -m services.recorder [--source ...] [--dry-run]``
"""

from __future__ import annotations

from .command import (
    FFmpegMissingError,
    build_ffmpeg_command,
    describe_command,
    resolve_ffmpeg,
)
from .settings import (
    PENDING_SETTINGS,
    RecorderError,
    RecorderSettings,
    SourceKind,
    SourceUnsetError,
    classify_source,
    normalize_source,
    redact_source,
)
from .supervisor import RecorderSupervisor, archive_segments, backoff_delay, spawn_ffmpeg

__all__ = [
    "PENDING_SETTINGS",
    "FFmpegMissingError",
    "RecorderError",
    "RecorderSettings",
    "RecorderSupervisor",
    "SourceKind",
    "SourceUnsetError",
    "archive_segments",
    "backoff_delay",
    "build_ffmpeg_command",
    "classify_source",
    "describe_command",
    "normalize_source",
    "redact_source",
    "resolve_ffmpeg",
    "spawn_ffmpeg",
]
