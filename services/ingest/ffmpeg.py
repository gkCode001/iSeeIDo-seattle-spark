"""The one place M1 spawns ffmpeg.

Two calls happen per window and both live behind this module: the gate's 32x32 grayscale
thumbnails (``gate.py``) and the captioner's overlaid JPEGs (``frames.py``). Both write
to stdout and are read as bytes — nothing is ever written to a temporary file, because
the archive is the only copy of the pixels and a scratch directory filling up at hour 30
is not a failure mode worth inventing.

Why ffmpeg does the heavy lifting: the alternative is decoding H.264 in Python, which
means numpy and OpenCV, which means wheels for ARM64 + sm_121 (CLAUDE.md platform
gotchas). ``-vf scale=32:32,format=gray`` hands back 1 KB per frame that pure Python can
diff at speed. The dependency we did not add is the feature.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence

from .settings import IngestError

__all__ = [
    "FFmpegError",
    "FFmpegMissingError",
    "FFmpegDecodeError",
    "resolve_ffmpeg",
    "run_ffmpeg",
    "BASE_ARGS",
]

#: Flags every call shares. ``-nostdin`` matters more than it looks: ingest runs under a
#: supervisor with no terminal, and an ffmpeg that decides to prompt hangs the loop.
BASE_ARGS: tuple[str, ...] = ("-hide_banner", "-nostdin", "-loglevel", "error")


class FFmpegError(IngestError):
    """An ffmpeg invocation failed."""


class FFmpegMissingError(FFmpegError):
    """ffmpeg is not installed, or the configured binary is not executable."""


class FFmpegDecodeError(FFmpegError):
    """ffmpeg ran and could not read the footage.

    Distinct from :class:`FFmpegMissingError` on purpose: a corrupt segment is a normal
    fact about an archive whose recorder was killed (CLAUDE.md: SIGTERM, never SIGKILL —
    the last open segment of a hard-killed recorder has no moov atom), and M1 must keep
    walking past it rather than stopping the run. A missing binary is the opposite: every
    subsequent window will fail the same way.
    """

    def __init__(self, message: str, *, returncode: int = 0, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def resolve_ffmpeg(binary: str) -> str:
    """Resolve ``binary`` to an executable path, or explain how to get one.

    Never let this surface as a raw ``FileNotFoundError`` from ``Popen``: that names the
    binary and nothing else, and M1 without ffmpeg cannot gate, cannot sample and cannot
    burn an overlay — it is not a degraded pipeline, it is no pipeline.
    """
    if os.sep in binary:
        if os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary
        found = None
    else:
        found = shutil.which(binary)
    if found:
        return found
    raise FFmpegMissingError(
        f"ffmpeg not found (looked for {binary!r}). M1 needs it for all three of the gate, "
        f"frame sampling and the burned wall-clock overlay.\n"
        f"  Install it:             sudo apt install ffmpeg\n"
        f"  Point at another build: export SPARK_FFMPEG=/opt/ffmpeg-cuda/bin/ffmpeg\n"
        f"  Then verify the box:    python3 scripts/doctor.py"
    )


def run_ffmpeg(argv: Sequence[str], *, timeout: float, expect_output: bool = True) -> bytes:
    """Run ffmpeg and return its stdout.

    ``timeout`` is not optional and has no ``None`` spelling. Ingest is a loop over a
    growing archive; one wedged decode with no bound stops every window after it, and the
    symptom — captions simply stop arriving — is indistinguishable from a crashed service
    from anywhere downstream.

    A non-zero exit, a timeout, or an empty stdout when output was expected all raise
    :class:`FFmpegDecodeError` carrying ffmpeg's own stderr. The caller decides whether
    that window is fatal; it never is, in practice.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - argv is built here, never user text
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegMissingError(f"ffmpeg binary {argv[0]!r} is not executable: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegDecodeError(
            f"ffmpeg exceeded ingest.ffmpeg_timeout_seconds ({timeout}s) and was killed; "
            f"the decode is wedged, not slow",
            stderr=_text(exc.stderr),
        ) from exc

    stderr = _text(completed.stderr)
    if completed.returncode != 0:
        raise FFmpegDecodeError(
            f"ffmpeg exited {completed.returncode}: {stderr.strip()[:400] or '(no stderr)'}",
            returncode=completed.returncode,
            stderr=stderr,
        )
    if expect_output and not completed.stdout:
        # Exit 0 with nothing on stdout happens when a seek lands past EOF — a window
        # whose wall clock the archive claims to cover but whose pixels the file does not.
        raise FFmpegDecodeError(
            f"ffmpeg produced no output: {stderr.strip()[:400] or '(no stderr)'}",
            stderr=stderr,
        )
    return completed.stdout


def _text(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")
