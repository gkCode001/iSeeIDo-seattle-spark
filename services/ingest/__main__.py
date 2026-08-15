"""``make ingest`` — run M1 against the configured archive.

    python3 -m services.ingest                    # walk everything the archive covers
    python3 -m services.ingest --follow           # keep walking as the recorder writes
    python3 -m services.ingest --limit 20 --dry-run
    SPARK_SETTINGS=/path/to/settings.yaml python3 -m services.ingest

Ingest reads the **archive**, not the camera. The recorder (SPEC §2.1) owns the camera
and runs independently of any AI, which is why this can be stopped, restarted and
re-pointed without risking a single frame of footage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime

from shared import config
from shared.schema import from_iso

from .pipeline import IngestPipeline
from .settings import IngestSettings

__all__ = ["main"]


def _parse_instant(value: str | None) -> datetime | None:
    """Parse a ``--from``/``--to`` argument, demanding an explicit UTC designator.

    ``shared/schema.py``'s parser rejects a naive timestamp rather than assuming UTC. A
    range silently interpreted in the wrong zone would walk real footage and index it
    against times that are five and a half hours out — and every record would look fine.
    """
    if value is None:
        return None
    return from_iso(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m services.ingest",
        description="M1 — walk the archive in analysis windows, gate them, caption the "
        "survivors, write chunk records (SPEC §2).",
    )
    parser.add_argument("--archive", help="override paths.archive for this run")
    parser.add_argument(
        "--from",
        dest="t_from",
        help="start of the walk, ISO 8601 UTC (default: the first segment on disk)",
    )
    parser.add_argument(
        "--to",
        dest="t_to",
        help="end of the walk, ISO 8601 UTC (default: the last segment on disk)",
    )
    parser.add_argument("--limit", type=int, help="stop after N windows — a smoke run")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="keep walking as the recorder extends the archive; Ctrl-C to stop",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="do not write to the index; gate and caption only. For measuring the gate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the windows and print them; touch no footage and no model",
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging, including every ignored file in the archive",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG
        if args.verbose
        else str(config.get("logging.level", "INFO")).upper(),
        format="%(message)s",
        stream=sys.stderr,
    )

    settings = IngestSettings.from_config(args.archive)
    t_from = _parse_instant(args.t_from)
    t_to = _parse_instant(args.t_to)

    if args.dry_run:
        # No sink, no queue start, no ffmpeg: this exists to answer "which windows would
        # you analyse" before spending an afternoon of decode on the wrong range.
        planner = IngestPipeline(settings, sink=None)
        for window in planner.plan(t_from, t_to):
            if args.limit is not None and window.index >= args.limit:
                break
            print(window)
        planner.close()
        return 0

    sink = None
    if not args.no_index:
        # Imported here rather than at module scope so that --dry-run and --no-index work
        # on a box where M2's backend cannot be constructed.
        from services.index import build_index

        sink = build_index()
        sink.ensure_ready()

    try:
        with IngestPipeline(settings, sink=sink) as pipeline:
            if args.follow:
                stats = pipeline.follow(t_from)
            else:
                stats = pipeline.run(t_from, t_to, limit=args.limit)
    finally:
        if sink is not None:
            sink.close()

    if args.json:
        print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    else:
        print(_summary(stats, settings))
    # Non-zero when the gate is mistuned: SPEC §2.3 calls a sub-60% skip rate the end of
    # real time, and a build step that says so only in prose gets read as passing.
    return 0 if stats.health(settings) != "low" else 1


def _summary(stats: object, settings: IngestSettings) -> str:
    """The human-readable block a run ends with."""
    d = stats.to_dict()  # type: ignore[attr-defined]
    health = stats.health(settings)  # type: ignore[attr-defined]
    return "\n".join(
        [
            "M1 ingest — SPEC §2",
            f"  windows            {d['windows']}",
            f"  captioned          {d['captioned']}",
            f"  skipped by gate    {d['skipped']}",
            f"  decode failures    {d['decode_failures']}",
            f"  no footage         {d['no_footage']}",
            f"  records written    {d['records_written']}",
            "",
            f"  skip rate          {d['skip_rate'] * 100:.1f}%  "
            f"(target >= {settings.target_skip_rate * 100:.0f}%, "
            f"warn < {settings.warn_skip_rate * 100:.0f}%)  [{health}]",
            f"  gate               {d['gate_ms']:.0f} ms total",
            f"  frame sampling     {d['extract_ms']:.0f} ms total",
            f"  captioning         {d['caption_ms']:.0f} ms total",
            f"  wall time          {d['elapsed_s']:.1f} s",
        ]
    )


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
