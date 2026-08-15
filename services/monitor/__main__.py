"""``python3 -m services.monitor`` — M5 as its own process (SPEC §6).

Two ways to run the monitor, and the difference matters:

* **In the M3 process** (the default; see ``services/agent/server.build_app``). M3 and
  M5 then share one task registry, so a task registered through the Watch pane is the
  same object the funnel evaluates, and funnel state reaches the pane over the WebSocket
  that already exists. This is what the demo runs.

* **Standalone, here.** Useful for the compose topology and for watching the funnel
  without a UI. The cost is real and is not hidden: this process has its OWN task
  registry seeded from ``config/tasks.yaml``, so tasks registered through M3's form do
  not reach it. Run it this way for seeded tasks, or when M3 is not running at all.

Both paths fire actions only through ``services/mcp``, so cooldown, time-range dedupe
and the append-only log apply either way (CLAUDE.md invariant 5).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

from shared import config

from services.monitor import build_monitor
from services.monitor.runner import IndexTail, MonitorRunner
from services.monitor.verify import WorkerVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m services.monitor",
        description="M5 — evaluate standing tasks against every chunk M1 emits (SPEC §6).",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="index JSONL to tail (default: index.store.memory_path)",
    )
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="replay the whole index instead of starting at its end. Off by default: "
        "on a first run the index may hold hours of history, and replaying it evaluates "
        "standing tasks against footage nobody is watching.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip stage 3. Alerts then stay UNVERIFIED and are never retracted.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    parser.add_argument("--max-ticks", type=int, default=None, help="stop after N polls")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("monitor")

    verifier: Any | None = None if args.no_verify else WorkerVerifier()
    monitor = build_monitor(verifier=verifier)
    index_path = args.index or str(config.repo_path("index.store.memory_path"))
    tail = IndexTail(index_path, seek_to_end=not args.from_start)
    runner = MonitorRunner(monitor, tail, poll_interval=args.interval)

    log.info(
        "monitor starting: %d task(s), tailing %s%s",
        len(monitor.tasks()),
        index_path,
        "" if verifier is not None else " (stage 3 disabled)",
    )
    for task in monitor.tasks():
        log.info(
            "  %-22s %-12s window %3ds  cooldown %4ds  active %s",
            task.task_id,
            task.action.value,
            task.window,
            task.cooldown,
            task.active,
        )

    if args.max_ticks is not None:
        for _ in range(args.max_ticks):
            runner.tick()
        log.info("observed %d chunk(s), fired %d action(s)", runner.chunks_seen, runner.actions_fired)
        return 0

    stopping = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        runner.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    runner.start()
    try:
        while not stopping:
            signal.pause()
    except KeyboardInterrupt:  # pragma: no cover - Ctrl-C outside the handler
        pass
    finally:
        runner.stop()
    log.info("observed %d chunk(s), fired %d action(s)", runner.chunks_seen, runner.actions_fired)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
