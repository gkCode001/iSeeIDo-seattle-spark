"""``python3 -m services.importer`` — put a recorded video into the system.

    python3 -m services.importer clip.mp4          # one file
    python3 -m services.importer --inbox           # everything in data/inbox, once
    python3 -m services.importer --watch           # keep watching the drop folder
    python3 -m services.importer clip.mp4 --no-ingest    # place it, caption it later

Import is two steps and both run here: slice the file into correctly named archive
segments (``services/importer/importer.py``), then walk that range with M1 exactly as if
the recorder had produced it. The second step is the one that costs — it is a VLM caption
per unskipped window, through the same single queue the live path uses, so importing an
hour of footage while the camera is running makes both slower. That is a scheduling fact,
not a bug: CLAUDE.md invariant 1 means there is one model and it is shared.

``--watch`` polls rather than uses inotify: the drop folder is a human dropping a file
every few minutes, the poll costs one ``listdir``, and a dependency-free stdlib loop is
worth more here than an event API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from shared import config

from .importer import ImportError_, ImportResult, import_video, inbox_files, ingest_settings_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m services.importer",
        description="Import recorded video: slice it into the archive on a real "
        "wall-clock timeline, then ingest it (SPEC §2 unchanged).",
    )
    parser.add_argument("files", nargs="*", help="video files to import")
    parser.add_argument(
        "--inbox",
        action="store_true",
        help="import everything in importer.inbox (data/inbox) and exit",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep watching the inbox and import files as they land; Ctrl-C to stop",
    )
    parser.add_argument(
        "--camera-id",
        help="identity for this recording (default: the next free clipNN). Its own id "
        "keeps the import from interleaving with the live camera on the timeline.",
    )
    parser.add_argument(
        "--start",
        help="ISO 8601 UTC instant the recording should start at "
        "(default: now - duration, so it ends at the moment of import)",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="place the segments but do not caption them. The footage is fetchable and "
        "the deep worker can read it; nothing is in the index until M1 walks it.",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="re-encode even when the source is h264. Slower, but it imposes our own "
        "keyframe interval, which bounds evidence-clip precision (CLAUDE.md).",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="remove the input file once it is safely in the archive",
    )
    parser.add_argument("--archive", help="override paths.archive for this run")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else str(config.get("logging.level", "INFO")).upper(),
        format="%(message)s",
        stream=sys.stderr,
    )

    if not args.files and not args.inbox and not args.watch:
        build_parser().print_help()
        return 2

    start = _parse_instant(args.start)
    results: list[ImportResult] = []

    try:
        for path in args.files:
            results.append(_one(Path(path), args, start))
        if args.inbox or args.watch:
            results.extend(_drain_inbox(args, start))
        if args.watch:
            _watch(args, start)
    except ImportError_ as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True))
    elif results:
        print(_summary(results))
    elif not args.watch:
        print("nothing to import", file=sys.stderr)
        return 1
    return 0


def _one(path: Path, args: argparse.Namespace, start: datetime | None) -> ImportResult:
    """Import one file and, unless told not to, caption the range it now occupies."""
    result = import_video(
        path,
        camera_id=args.camera_id,
        start=start,
        archive_dir=args.archive,
        force_reencode=args.reencode,
        keep_source=not args.delete_source,
    )
    print(
        f"placed {path.name} as {result.camera_id}: {len(result.segments)} segment(s), "
        f"{result.info.duration:.1f}s, {result.t_start:%H:%M:%S}–{result.t_end:%H:%M:%S} UTC",
        file=sys.stderr,
    )
    if not args.no_ingest:
        _ingest(result, args.archive)
    return result


def _ingest(result: ImportResult, archive_dir: str | None) -> None:
    """Walk the imported range with M1 — the same gate, prompt and index as the camera."""
    from services.index import build_index
    from services.ingest.pipeline import IngestPipeline

    settings = ingest_settings_for(result, archive_dir)
    sink = build_index()
    sink.ensure_ready()
    try:
        with IngestPipeline(settings, sink=sink) as pipeline:
            stats = pipeline.run(result.t_start, result.t_end)
    finally:
        sink.close()
    d = stats.to_dict()
    print(
        f"  ingested: {d['windows']} window(s), {d['captioned']} captioned, "
        f"{d['skipped']} skipped by the gate, {d['records_written']} record(s)",
        file=sys.stderr,
    )


def _drain_inbox(args: argparse.Namespace, start: datetime | None) -> list[ImportResult]:
    out: list[ImportResult] = []
    for path in inbox_files():
        try:
            out.append(_one(path, args, start))
        except ImportError_ as exc:
            # One unreadable file must not stop the rest: the folder is a human's drop
            # target and half of what lands in it will be the wrong thing.
            print(f"skipped {path.name}: {exc}", file=sys.stderr)
    return out


def _watch(args: argparse.Namespace, start: datetime | None) -> None:
    interval = float(config.get("importer.poll_interval_seconds", 5.0))
    inbox = config.repo_path("importer.inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    print(f"watching {inbox} every {interval:.0f}s — drop a video in", file=sys.stderr)
    seen: dict[Path, int] = {}
    while True:
        for path in inbox_files():
            size = path.stat().st_size
            # A file still being copied grows between polls. Import it on the pass after
            # its size stops changing, so a 400 MB scp does not become 40 MB of segments.
            if seen.get(path) != size:
                seen[path] = size
                continue
            try:
                _one(path, args, start)
            except ImportError_ as exc:
                print(f"skipped {path.name}: {exc}", file=sys.stderr)
            finally:
                seen.pop(path, None)
        time.sleep(interval)


def _summary(results: list[ImportResult]) -> str:
    lines = ["imported:"]
    for r in results:
        lines.append(
            f"  {r.source.name}  ->  {r.camera_id}  "
            f"{r.t_start:%Y-%m-%d %H:%M:%S}–{r.t_end:%H:%M:%S} UTC  "
            f"({r.info.duration:.1f}s, {r.info.width}x{r.info.height}, {r.info.codec}"
            f"{', re-encoded' if r.reencoded else ''})"
        )
    lines.append("")
    lines.append("Ask about it at http://127.0.0.1:8080/ — it is in the index under its own")
    lines.append("camera id, on the wall clock it was placed at.")
    return "\n".join(lines)


def _parse_instant(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(f"--start is not ISO 8601: {raw!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
