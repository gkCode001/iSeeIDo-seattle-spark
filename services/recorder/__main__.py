"""Entry point: ``python3 -m services.recorder``.

Start this before ingest and leave it running. It is the one process in the build that
should still be alive at hour 40.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .command import FFmpegMissingError, build_ffmpeg_command, describe_command, resolve_ffmpeg
from .log import configure, event
from .settings import RecorderError, RecorderSettings, redact_source
from .supervisor import RecorderSupervisor


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m services.recorder",
        description="Continuous full-resolution segment recorder (SPEC §2.1).",
    )
    p.add_argument(
        "--source",
        default=None,
        help=(
            "RTSP url or path to a recording, overriding recorder.source. Both are "
            "supported — SPEC §10 D2 has not chosen between them."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg command and exit. Does not require ffmpeg to be installed.",
    )
    p.add_argument(
        "--max-starts",
        type=int,
        default=None,
        help="Stop after this many ffmpeg spawns. For smoke tests; omit for the real run.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    log = configure("recorder")

    try:
        settings = RecorderSettings.from_config(source=args.source)
    except RecorderError as exc:
        # Unset or unusable source. This is a human-readable message on purpose — the
        # source is UNRESOLVED (SPEC §10 D2) and someone has to decide, not debug.
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        command = build_ffmpeg_command(settings)
        print(describe_command(command))
        try:
            resolve_ffmpeg(command[0])
        except FFmpegMissingError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2
        return 0

    supervisor = RecorderSupervisor(settings)
    supervisor.install_signal_handlers()
    try:
        return supervisor.run_forever(max_starts=args.max_starts)
    except FFmpegMissingError as exc:
        event(log, logging.CRITICAL, "recorder.ffmpeg_missing", detail=str(exc))
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except RecorderError as exc:
        event(
            log,
            logging.CRITICAL,
            "recorder.failed",
            detail=str(exc),
            source=redact_source(settings.source),
        )
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by hand, not by unittest
    raise SystemExit(main())
