"""Supervising the ffmpeg segmenter (SPEC §2.1).

The recorder is the one component that must survive everything else. If ingest crashes,
if the VLM OOMs, if Milvus never comes up — recording continues, because the archive is
what M4 re-reads and it is the only copy of the pixels we ever get.

So this module assumes ffmpeg will die and plans for it:

* **Exit** — logged with the return code, restarted with exponential backoff.
* **Stall** — alive, but no segment has been closed in ``stall_timeout_seconds``. A
  camera that stops sending without closing the socket looks exactly like a healthy
  process from the outside. The watchdog kills it so the restart path can run.
* **Flap** — backoff grows to a ceiling, and resets after a run long enough to count as
  healthy, so an hour-old failure streak does not punish a fresh restart.

Everything that touches the outside world — spawning, sleeping, reading the clock,
listing the archive — is injectable, so the tests exercise the real supervision logic
with no ffmpeg, no subprocesses and no elapsed wall time.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from .command import build_ffmpeg_command, describe_command, resolve_ffmpeg
from .log import configure, event
from .settings import RecorderSettings, redact_source, setting

__all__ = ["ProcessHandle", "RecorderSupervisor", "backoff_delay", "spawn_ffmpeg"]


class ProcessHandle(Protocol):
    """The slice of ``subprocess.Popen`` the supervisor uses.

    Narrow on purpose: a fake that implements these four methods is a complete stand-in,
    which is what makes the restart logic testable on a box with no ffmpeg.
    """

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


Spawn = Callable[[Sequence[str]], ProcessHandle]


def spawn_ffmpeg(argv: Sequence[str]) -> ProcessHandle:
    """Start ffmpeg detached from our stdin, with stderr piped for the log pump."""
    return subprocess.Popen(  # noqa: S603 — argv is built by us, never shell-interpolated
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )


def backoff_delay(
    consecutive_failures: int, initial: float, maximum: float, multiplier: float
) -> float:
    """Delay before restart attempt number ``consecutive_failures`` (1-based).

    Exponential, capped. No jitter: there is one camera and one recorder, so there is no
    thundering herd to spread out, and a predictable sequence is one a human can read off
    the log and recognise.
    """
    if consecutive_failures <= 1:
        return float(initial)
    try:
        delay = float(initial) * (float(multiplier) ** (consecutive_failures - 1))
    except OverflowError:
        # A source that has been unavailable for days. The exponent stopped meaning
        # anything long before the float did; the cap is the answer either way.
        return float(maximum)
    return min(delay, float(maximum))


class RecorderSupervisor:
    """Keeps exactly one ffmpeg segmenter alive for the lifetime of the process."""

    def __init__(
        self,
        settings: RecorderSettings,
        *,
        spawn: Spawn | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
        resolve_binary: bool = True,
    ) -> None:
        self.settings = settings
        self._spawn: Spawn = spawn or spawn_ffmpeg
        self._clock = clock
        self._log = logger or configure("recorder")
        # A test harness passes its own spawn and has no ffmpeg to resolve; the real run
        # resolves up front so a missing binary is one clear message, not a restart loop
        # around FileNotFoundError.
        self._resolve_binary = resolve_binary
        self._stop = threading.Event()
        self._proc: ProcessHandle | None = None
        self._starts = 0
        self._term_sent = False
        # Backoff sleeps wake early on stop, so Ctrl-C during a 60 s backoff does not
        # look like a hung process.
        self._sleep: Callable[[float], None] = sleep or (lambda d: self._stop.wait(d))

    # -- lifecycle ---------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the supervisor to shut down, interrupting the current run.

        Safe to call from a signal handler: it sets a flag and sends one signal. It does
        not wait — waiting for a process from inside a handler that interrupted a wait on
        that same process is how shutdown paths hang.
        """
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._signal_stop(proc, reason="stop_requested")

    def install_signal_handlers(self) -> None:
        """Ctrl-C and ``docker stop`` should finalise the open segment, not orphan it."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda _s, _f: self.request_stop())
            except ValueError:
                # Not the main thread. The caller owns shutdown in that case.
                return

    def run_forever(self, max_starts: int | None = None) -> int:
        """Run until stopped. Returns a process exit status.

        ``max_starts`` bounds the number of spawns — used by the tests and by
        ``--max-starts`` for a smoke run. ``None`` means "until someone stops us", which
        is the only value the demo uses.
        """
        argv = build_ffmpeg_command(self.settings)
        if self._resolve_binary:
            argv[0] = resolve_ffmpeg(argv[0])

        self.settings.archive_dir.mkdir(parents=True, exist_ok=True)

        max_failures = int(setting("recorder.restart.max_consecutive_failures"))
        initial = float(setting("recorder.restart.initial_backoff_seconds"))
        maximum = float(setting("recorder.restart.max_backoff_seconds"))
        multiplier = float(setting("recorder.restart.backoff_multiplier"))
        healthy = float(setting("recorder.restart.healthy_seconds"))

        event(
            self._log,
            logging.INFO,
            "recorder.supervisor_start",
            camera_id=self.settings.camera_id,
            source=redact_source(self.settings.source),
            source_kind=self.settings.kind.value,
            segment_seconds=self.settings.segment_seconds,
            copy_codec=self.settings.copy_codec,
            archive_dir=str(self.settings.archive_dir),
            command=describe_command(argv, redact=True),
        )
        if not self.settings.copy_codec:
            event(
                self._log,
                logging.WARNING,
                "recorder.reencoding_archive",
                detail=(
                    "recorder.copy_codec is false, so the archive is re-encoded. CLAUDE.md "
                    "invariant 7 says the archive stays at native resolution and is what "
                    "the deep worker re-reads."
                ),
            )

        failures = 0
        status = 0
        while not self._stop.is_set():
            if max_starts is not None and self._starts >= max_starts:
                break
            started_at = self._clock()
            returncode = self._run_once(argv)
            ran_for = self._clock() - started_at

            if self._stop.is_set():
                event(self._log, logging.INFO, "recorder.stopped", returncode=returncode)
                break

            failures = 1 if ran_for >= healthy else failures + 1
            event(
                self._log,
                logging.ERROR if returncode not in (0, None) else logging.WARNING,
                "recorder.exited",
                returncode=returncode,
                ran_for_s=round(ran_for, 3),
                consecutive_failures=failures,
                clean_exit=returncode == 0,
            )

            if max_failures and failures >= max_failures:
                event(
                    self._log,
                    logging.CRITICAL,
                    "recorder.giving_up",
                    consecutive_failures=failures,
                    detail=(
                        "ffmpeg failed to stay up. Nothing is being recorded. Check the "
                        "source with: python3 -m services.recorder --dry-run"
                    ),
                )
                status = 1
                break

            if max_starts is not None and self._starts >= max_starts:
                break

            delay = backoff_delay(failures, initial, maximum, multiplier)
            event(
                self._log,
                logging.INFO,
                "recorder.restart_scheduled",
                delay_s=delay,
                consecutive_failures=failures,
            )
            self._sleep(delay)

        return status

    # -- one run -----------------------------------------------------------------------

    def _run_once(self, argv: Sequence[str]) -> int | None:
        """Spawn ffmpeg, watch it, return its exit code."""
        # Counted before the spawn, not after: a spawn that raises is still an attempt,
        # and a loop that only counts successes never terminates when nothing succeeds.
        self._starts += 1
        self._term_sent = False
        try:
            proc = self._spawn(argv)
        except OSError as exc:
            event(self._log, logging.ERROR, "recorder.spawn_failed", error=str(exc))
            return None
        self._proc = proc
        event(
            self._log,
            logging.INFO,
            "recorder.started",
            pid=proc.pid,
            start_number=self._starts,
        )
        self._pump_stderr(proc)
        try:
            return self._watch(proc)
        finally:
            self._proc = None

    def _watch(self, proc: ProcessHandle) -> int | None:
        """Wait for exit, killing the process if the archive stops growing.

        Freshness is measured on the archive directory rather than on ffmpeg's stderr:
        stderr is quiet by design at our log level, and "a segment was closed recently" is
        the only signal that actually means what we care about.
        """
        poll_interval = float(setting("recorder.restart.poll_interval_seconds"))
        stall_timeout = float(setting("recorder.restart.stall_timeout_seconds"))
        stop_timeout = float(setting("recorder.stop_timeout_seconds"))
        # ffmpeg writes nothing until the first segment closes, so the watchdog cannot
        # start counting down before one segment length has passed.
        first_write_grace = float(self.settings.segment_seconds) + stall_timeout

        last_progress = self._clock()
        last_seen = self._archive_marker()
        seen_progress = False
        kill_deadline: float | None = None

        while True:
            try:
                return proc.wait(timeout=poll_interval)
            except subprocess.TimeoutExpired:
                pass
            now = self._clock()

            # Already asked it to go. Give it stop_timeout to finalise the open mp4, then
            # stop being polite — a killed segment has no moov atom and will not play, so
            # this order matters.
            if kill_deadline is not None:
                if now >= kill_deadline:
                    event(self._log, logging.WARNING, "recorder.killing", pid=proc.pid)
                    try:
                        proc.kill()
                        return proc.wait(timeout=stop_timeout)
                    except (OSError, subprocess.TimeoutExpired):
                        return proc.poll()
                continue

            if self._stop.is_set():
                # ``request_stop`` may have arrived before this process existed — during
                # a backoff sleep, or in the gap between spawning and recording the
                # handle. The signal is sent here so there is one path that always runs.
                self._signal_stop(proc, reason="stop_requested")
                kill_deadline = now + stop_timeout
                continue

            marker = self._archive_marker()
            if marker != last_seen:
                last_seen, last_progress, seen_progress = marker, now, True
                continue

            idle = now - last_progress
            limit = stall_timeout if seen_progress else first_write_grace
            if idle >= limit:
                event(
                    self._log,
                    logging.ERROR,
                    "recorder.stalled",
                    idle_s=round(idle, 3),
                    limit_s=limit,
                    detail=(
                        "ffmpeg is alive but the archive has not grown. Ending it so the "
                        "restart path can run; nothing has been recorded since the segment "
                        "named below."
                    ),
                    last_segment=last_seen[0] if last_seen else None,
                )
                self._signal_stop(proc, reason="stalled")
                kill_deadline = now + stop_timeout

    def _archive_marker(self) -> tuple[str, float, int] | None:
        """(name, mtime, size) of the newest segment, or None if there is none yet.

        Size is in the tuple because the last file keeps growing between segment
        boundaries — that is progress too, and waiting a whole segment to notice it would
        make the watchdog needlessly slow to trust.

        ``os.scandir`` rather than ``iterdir``: this runs every poll interval for the life
        of the process, against a directory holding 1440 files per day.
        """
        newest: tuple[str, float, int] | None = None
        try:
            with os.scandir(self.settings.archive_dir) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file():
                            continue
                        stat = entry.stat()
                    except OSError:
                        continue
                    if newest is None or stat.st_mtime > newest[1]:
                        newest = (entry.name, stat.st_mtime, stat.st_size)
        except OSError:
            return None
        return newest

    def _signal_stop(self, proc: ProcessHandle, *, reason: str) -> None:
        """Send SIGTERM once and return immediately. ``_watch`` escalates to SIGKILL."""
        if self._term_sent:
            return
        self._term_sent = True
        event(self._log, logging.INFO, "recorder.terminating", reason=reason, pid=proc.pid)
        try:
            proc.terminate()
        except OSError as exc:
            event(self._log, logging.WARNING, "recorder.terminate_failed", error=str(exc))

    def _pump_stderr(self, proc: ProcessHandle) -> None:
        """Drain ffmpeg's stderr into the structured log.

        Not optional: with stderr on a pipe and nobody reading, a chatty ffmpeg fills the
        buffer and blocks forever — a recorder that is alive, silent and archiving
        nothing, which is the exact failure this module exists to prevent.
        """
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return

        def pump() -> None:
            try:
                for line in stream:
                    text = line.rstrip()
                    if text:
                        event(self._log, logging.WARNING, "recorder.ffmpeg", line=text)
            except (OSError, ValueError):
                pass

        threading.Thread(target=pump, name="ffmpeg-stderr", daemon=True).start()


def archive_segments(archive_dir: Path, camera_id: str) -> list[Path]:
    """Segment files on disk for a camera, oldest first.

    Convenience for a human checking that the recorder is doing its job. Mapping a time
    range onto these files is ``shared/timecode.py``'s exclusive job (invariant 3) — do
    not grow a second implementation here.
    """
    try:
        found = [p for p in archive_dir.iterdir() if p.is_file() and p.name.startswith(camera_id)]
    except OSError:
        return []
    return sorted(found)
