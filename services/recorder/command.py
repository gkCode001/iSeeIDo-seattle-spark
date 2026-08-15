"""Building the ffmpeg segmenter command line (SPEC §2.1).

Pure. Nothing here spawns a process or requires ffmpeg to exist — ffmpeg is *not*
installed on this box yet (CLAUDE.md machine state) and the module must still import,
and its output must still be testable. Execution lives in ``supervisor.py``; what gets
tested is the argv list, not a live process.
"""

from __future__ import annotations

import os
import shutil

from .settings import RecorderError, RecorderSettings, SourceKind, redact_source

__all__ = ["FFmpegMissingError", "build_ffmpeg_command", "resolve_ffmpeg", "describe_command"]


class FFmpegMissingError(RecorderError):
    """ffmpeg is not installed, or the configured binary is not executable."""


def build_ffmpeg_command(settings: RecorderSettings, ffmpeg_bin: str | None = None) -> list[str]:
    """Return the argv for the continuous segmenter.

    The three flags that are not negotiable:

    ``-f segment -strftime 1``
        SPEC §2.1. The filename carries the segment's wall-clock start time, which is
        what ``shared/timecode.py`` maps a fetch range onto.

    ``-reset_timestamps 1``
        Each segment's PTS restarts at zero. Invariant 2 states this as a fact about the
        archive; this flag is what makes it true. Without it ``pts_offset`` would be
        measured against a monotonic stream clock and every deep-worker seek would be
        wrong by however long the recorder had been up.

    ``-c copy``
        Invariant 7. Stream-copy at native resolution: no downscale, no re-encode, no
        generation loss. Downscaling belongs to the live analysis path only. It is also
        why a 30-hour recording costs ~5% of one CPU core.

    ``-segment_atclocktime 1`` is the cosmetic-looking one that earns its place: cuts land
    on wall-clock multiples of ``segment_seconds``, so files are named 21:10:00, 21:11:00
    exactly as SPEC §2.1 shows, instead of drifting off whenever the recorder restarted.
    """
    settings.validate()
    binary = ffmpeg_bin or settings.ffmpeg_bin
    argv: list[str] = [binary, "-hide_banner", "-nostdin", "-loglevel", settings.loglevel]

    if settings.kind is SourceKind.DEVICE:
        # v4l2 must be named explicitly: without -f, ffmpeg probes /dev/video0 as a file
        # and fails with a demuxer error. Capture mode is stated rather than negotiated,
        # because a webcam asked for nothing in particular will happily hand back 640x480
        # and nobody notices until the footage is on screen.
        argv += ["-f", "v4l2", "-framerate", str(settings.device_framerate)]
        argv += ["-video_size", settings.device_video_size]
        if settings.device_input_format:
            argv += ["-input_format", settings.device_input_format]
    elif settings.kind is SourceKind.RTSP:
        # UDP is the default and it drops packets under load; the archive is the one copy
        # of the pixels we ever get.
        argv += ["-rtsp_transport", settings.rtsp_transport]
        # No socket-timeout flag here on purpose: rtsp's -stimeout/-timeout spelling moved
        # between ffmpeg 5 and 6 and we have no ffmpeg on this box to verify against, and
        # an unknown option makes ffmpeg exit instead of record. A camera that stalls
        # without closing the socket is caught by the supervisor's stall watchdog, which
        # covers every stall mode rather than just the socket one.
    elif settings.realtime_file_playback:
        # A file is read as fast as the disk allows, which would write a day of segments
        # in a minute and make every wall-clock filename a lie. -re paces playback at the
        # native frame rate so the archive means what invariant 2 says it means.
        argv += ["-re"]

    argv += ["-i", settings.source]

    if settings.kind is SourceKind.DEVICE:
        # A webcam emits rawvideo or mjpeg; neither stream-copies into mp4 as anything a
        # player will open. So this path encodes — at the capture resolution, with no
        # scale filter anywhere, which is what invariant 7 actually protects. The rule is
        # "never downscale the archive", not "never touch a codec".
        argv += ["-c:v", settings.encoder]
        # Keyframe interval. This is the flag that makes invariant 3's "fetch by time
        # range" honest. A stream-copied clip can only begin on a keyframe, so clip
        # accuracy is bounded by the GOP: measured on this box, an 8.3 s GOP turned a
        # 2.0 s request into 8.07 s of footage that started 6 s early, while a 1 s GOP
        # returned 2.1 s. The deep worker seeks the same way, so this bounds its
        # accuracy too.
        gop = max(1, round(settings.device_framerate * settings.device_keyframe_interval_seconds))
        argv += ["-g", str(gop)]
        argv += [*settings.encoder_args]
    elif settings.copy_codec:
        argv += ["-c", "copy"]

    argv += [
        "-f",
        "segment",
        "-strftime",
        "1",
        "-segment_time",
        str(settings.segment_seconds),
        "-segment_format",
        settings.container,
        "-segment_atclocktime",
        "1",
        "-reset_timestamps",
        "1",
        settings.output_template,
    ]

    if settings.preview_enabled and settings.preview_path is not None:
        # A second OUTPUT, appended after the archive's. The archive argv above is
        # untouched by design: whatever happens to the preview, the segment writer is
        # byte-for-byte the command it would have been without this block.
        #
        # `-update 1` rewrites one file in place rather than accumulating frames, so the
        # UI can poll a single path and the disk cost is one small JPEG, forever.
        # The scale here is a downscale of the PREVIEW ONLY — invariant 7 is about the
        # archive, which is written by the output above at capture resolution.
        argv += [
            "-map", "0:v",
            "-vf", f"fps={settings.preview_fps},scale={settings.preview_width}:-2",
            "-q:v", str(settings.preview_quality),
            "-update", "1",
            "-y",
            str(settings.preview_path),
        ]
    return argv


def resolve_ffmpeg(binary: str) -> str:
    """Resolve ``binary`` to an executable path, or explain how to get one.

    Never let this surface as a raw ``FileNotFoundError`` from ``Popen``: that error names
    the binary and nothing else, and the person reading it at hour 3 has no idea that the
    apt candidate on this box is documented as having no NVDEC.
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
        f"ffmpeg not found (looked for {binary!r}). The recorder cannot write the archive "
        f"without it, and the archive is the deep worker's entire reason for existing.\n"
        f"  Install it:            sudo apt install ffmpeg\n"
        f"  Point at another build: export SPARK_FFMPEG=/opt/ffmpeg-cuda/bin/ffmpeg\n"
        f"  Then verify the box:    python3 scripts/doctor.py\n"
        f"Note: the apt candidate (6.1.1) advertises no NVDEC. That is fine here — the "
        f"recorder stream-copies and never decodes — but SPEC §2.4 GPU decode will want a "
        f"CUDA build. See the CLAUDE.md machine-state table."
    )


def describe_command(argv: list[str], redact: bool = False) -> str:
    """A copy-pasteable shell rendering of ``argv``.

    ``redact`` strips ``user:password@`` out of any argument that looks like a url. Pass
    it for anything that lands in a log — camera urls carry credentials far more often
    than not, and these logs are what gets pasted into a chat window when the recorder
    misbehaves. ``--dry-run`` leaves them in: the point there is a line you can paste
    into a shell.
    """
    args = [redact_source(a) for a in argv] if redact else argv
    return " ".join(_quote(a) for a in args)


def _quote(arg: str) -> str:
    if arg and not any(c in arg for c in " \t\n\"'\\$&|<>();*?[]{}#~!"):
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"
