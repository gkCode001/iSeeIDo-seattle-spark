"""Recorder configuration, resolved from ``config/settings.yaml`` (SPEC §2.1).

Nothing in this module touches a process or the filesystem. It exists so that command
construction stays pure and testable on a box where ``ffmpeg`` is not installed — which
is this box, today (CLAUDE.md machine state).

Pending settings
----------------
``settings.yaml`` does not yet carry the supervision dials or the RTSP transport. They
are listed once in ``PENDING_SETTINGS`` below with the values we would put in the YAML,
and read through :func:`setting` so that adding the keys to ``settings.yaml`` takes over
with no code change. This is deliberately the *only* place in the recorder where a
number has a fallback — CLAUDE.md's "no magic numbers" rule is about numbers scattered
through service code, and one labelled table that names its own migration is the least
bad way to hold them until the config file can be edited.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from shared import config

__all__ = [
    "PENDING_SETTINGS",
    "RecorderError",
    "SourceUnsetError",
    "SourceKind",
    "RecorderSettings",
    "classify_source",
    "normalize_source",
    "redact_source",
    "setting",
]


# Proposed ``settings.yaml`` additions, under the existing ``recorder:`` block. Every
# value here is a number or a knob the recorder needs and the YAML does not have yet.
PENDING_SETTINGS: dict[str, Any] = {
    # RTSP over UDP silently drops packets under load and the archive is what the deep
    # worker re-reads; TCP trades latency we do not care about for frames we do.
    "recorder.rtsp_transport": "tcp",
    # Name/path of the binary. Overridable so a CUDA build in /opt can be pointed at
    # without touching code (SPEC §2.4 wants NVDEC eventually).
    "recorder.ffmpeg_bin": "ffmpeg",
    # A file source is paced with -re so one second of footage costs one second of wall
    # clock. Set false only to blast a recording onto disk as fast as it reads, which
    # breaks the wall-clock ↔ footage correspondence of invariant 2.
    "recorder.realtime_file_playback": True,
    # Supervision. A recorder that dies quietly at hour 30 loses the demo footage.
    "recorder.restart.initial_backoff_seconds": 1.0,
    "recorder.restart.max_backoff_seconds": 60.0,
    "recorder.restart.backoff_multiplier": 2.0,
    # A run that lasted this long counts as healthy: the next death starts backoff over
    # rather than inheriting an hour-old failure streak.
    "recorder.restart.healthy_seconds": 30.0,
    # 0 = never give up. A camera unplugged for an hour must still be recording when it
    # comes back.
    "recorder.restart.max_consecutive_failures": 0,
    # Death without an exit code: ffmpeg alive, no new segment closed. Three segments of
    # silence is unambiguous, and the watchdog kills the process so the restart path can
    # do its job.
    "recorder.restart.stall_timeout_seconds": 180.0,
    "recorder.restart.poll_interval_seconds": 1.0,
    # SIGTERM, then this long, then SIGKILL. Long enough for ffmpeg to finalise the mp4
    # it is holding open — a killed segment is an unplayable segment.
    "recorder.stop_timeout_seconds": 10.0,
    # v4l2 webcam capture — D2's resolved primary source.
    "recorder.device.video_size": "1280x720",
    "recorder.device.framerate": 30,
    "recorder.device.input_format": None,
    "recorder.device.encoder": "h264_nvenc",
    "recorder.device.encoder_args": [],
    "recorder.device.keyframe_interval_seconds": 1.0,
}

# ffmpeg's own verbosity, derived from ``logging.level`` rather than configured twice.
# ffmpeg's "info" is a progress line several times a second, which is noise, not INFO.
_FFMPEG_LOGLEVEL = {
    "DEBUG": "verbose",
    "INFO": "warning",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "fatal",
}

_RTSP_SCHEMES = frozenset({"rtsp", "rtsps"})

#: v4l2 capture devices. Linux-only, which is what this box is.
_DEVICE_PREFIX = "/dev/video"


class RecorderError(RuntimeError):
    """Base class for recorder failures that a human is expected to read and act on."""


class SourceUnsetError(RecorderError):
    """``recorder.source`` is null — SPEC §10 D2 is unresolved."""


class SourceKind(str, Enum):
    """What we are recording from.

    D2 resolved to ``DEVICE`` — a USB webcam — as the primary demo source: it is live,
    needs no network, and an event can be staged in front of it on cue. The other two
    stay supported because they cost nothing to keep and a webcam that fails on the day
    should not also be a code change.
    """

    RTSP = "rtsp"
    FILE = "file"
    DEVICE = "device"


def setting(dotted: str) -> Any:
    """Read a setting, falling back to :data:`PENDING_SETTINGS` for keys not in the YAML.

    Raises ``KeyError`` for a key that is in neither, which is a programming error rather
    than a configuration one.
    """
    return config.get(dotted, PENDING_SETTINGS[dotted])


def classify_source(source: str) -> SourceKind:
    """Decide whether ``source`` is a stream URL or a path on disk.

    SPEC §10 D2 — live camera vs pre-ingested recording — is open, so this returns a kind
    rather than asserting one. Anything with a scheme we do not speak is rejected loudly:
    a typo'd ``rtp://`` would otherwise reach ffmpeg and fail as a demuxer error.
    """
    if not source or not source.strip():
        raise SourceUnsetError("recorder source is empty")
    text = source.strip()
    # Checked before the path branch: /dev/video0 is a path by spelling but a live
    # capture device by behaviour. Treating it as a file would pace it with -re and
    # stream-copy raw frames into mp4, which produces an unplayable archive.
    if text.startswith(_DEVICE_PREFIX):
        return SourceKind.DEVICE
    parsed = urlparse(text)
    # A Windows-style drive letter or a bare relative path parses as a one-char scheme.
    if len(parsed.scheme) <= 1:
        return SourceKind.FILE
    if parsed.scheme in _RTSP_SCHEMES:
        return SourceKind.RTSP
    if parsed.scheme == "file":
        return SourceKind.FILE
    raise RecorderError(
        f"unsupported recorder.source scheme {parsed.scheme!r} in {redact_source(text)!r}. "
        f"Supported: an RTSP url (rtsp:// or rtsps://) or a path to a file on disk "
        f"(SPEC §10 D2 is still open, so both are allowed)."
    )


def normalize_source(source: str) -> str:
    """Canonicalise a source for handing to ffmpeg.

    A file source becomes an absolute path resolved against the repo root, because the
    recorder is a long-lived process whose working directory is nobody's business, and
    ``file://`` is unwrapped — ffmpeg spells that protocol ``file:`` and would treat the
    ``//`` as a hostname. URLs are passed through untouched.
    """
    text = source.strip()
    kind = classify_source(text)
    if kind is SourceKind.RTSP:
        return text
    if kind is SourceKind.DEVICE:
        # Device nodes are already absolute and must not be resolved: /dev/video0 is
        # often a symlink and ffmpeg wants the name the user configured.
        return text
    if text.startswith("file://"):
        text = urlparse(text).path
    return str((config.REPO_ROOT / text).resolve())


def redact_source(source: str) -> str:
    """Strip credentials out of an RTSP url before it reaches a log line.

    Camera urls carry ``user:password@`` far more often than not, and these logs are the
    thing we paste into a chat window when the recorder misbehaves.
    """
    try:
        parsed = urlparse(source)
    except ValueError:
        return source
    if not parsed.netloc or "@" not in parsed.netloc:
        return source
    host = parsed.netloc.rsplit("@", 1)[1]
    return urlunparse(parsed._replace(netloc=f"***@{host}"))


@dataclass(frozen=True)
class RecorderSettings:
    """Everything the recorder needs, resolved once and then immutable.

    ``copy_codec`` is CLAUDE.md invariant 7 in a boolean: the archive is stream-copied at
    native resolution and never re-encoded to save disk. It is the deep worker's entire
    reason for existing.
    """

    camera_id: str
    source: str
    archive_dir: Path
    segment_seconds: int
    filename_pattern: str
    container: str
    copy_codec: bool
    rtsp_transport: str
    realtime_file_playback: bool
    ffmpeg_bin: str
    loglevel: str
    # v4l2 capture. `device_video_size` selects the camera's *capture mode*; it is not a
    # downscale and does not touch invariant 7 — set it to the highest mode the camera
    # offers. `device_input_format` picks between the camera's own encodings (mjpeg
    # gives higher frame rates at 1080p than raw yuyv422 over USB 2.0).
    device_video_size: str = "1280x720"
    device_framerate: int = 30
    device_input_format: str | None = None
    # A webcam cannot be stream-copied into mp4 — it emits rawvideo or mjpeg — so the
    # device path encodes. That is not a violation of invariant 7: the rule is never to
    # *downscale* the archive, and this encodes at capture resolution. h264_nvenc uses
    # the dedicated encoder block rather than SM time, so it does not compete with the
    # VLM for compute.
    encoder: str = "h264_nvenc"
    encoder_args: tuple[str, ...] = ()
    # Keyframe interval, in seconds. This is load-bearing for evidence clips, not a
    # quality knob: a stream-copied cut can only start on a keyframe, so a clip is
    # accurate to at best one GOP. Measured on this box — an 8.3 s GOP turned a 2.0 s
    # request into 8.07 s of footage starting 6 s early; at 1 s it returns 2.1 s.
    # Invariant 3 promises a fetch by time range; this is what makes that promise true
    # to the second rather than to the nearest keyframe.
    device_keyframe_interval_seconds: float = 1.0
    # Live preview — a second output of the same ffmpeg, for the UI. Never a second
    # camera reader: v4l2 access to /dev/video0 is exclusive.
    preview_enabled: bool = False
    preview_path: Path | None = None
    preview_fps: float = 2.0
    preview_width: int = 640
    preview_quality: int = 6

    @property
    def kind(self) -> SourceKind:
        return classify_source(self.source)

    @property
    def output_template(self) -> str:
        """Absolute strftime template ffmpeg writes segments to.

        ``{camera_id}`` is substituted by us; the ``%Y%m%d_%H%M%S`` escapes are left for
        ffmpeg's ``-strftime`` to expand at segment-open time. That is what makes the
        filename carry the segment's wall-clock start (SPEC §2.1) — the anchor
        ``shared/timecode.py`` maps a time range onto, and half of invariant 2.
        """
        try:
            name = self.filename_pattern.format(camera_id=self.camera_id)
        except (KeyError, IndexError) as exc:
            raise RecorderError(
                f"recorder.filename_pattern {self.filename_pattern!r} references an "
                f"unknown placeholder {exc}; only {{camera_id}} is substituted (the "
                f"%-escapes are ffmpeg's, via -strftime)."
            ) from exc
        return str(self.archive_dir / name)

    @classmethod
    def from_config(cls, source: str | None = None) -> RecorderSettings:
        """Build from ``settings.yaml``. ``source`` overrides ``recorder.source``.

        Fails with a readable message when the source is unset, because it *is* unset:
        SPEC §10 D2 (live camera vs pre-ingested recording) has no owner and no answer,
        and picking one here silently would be the wrong kind of helpful.
        """
        resolved = source if source is not None else config.get("recorder.source", None)
        resolved = resolved.strip() if isinstance(resolved, str) else resolved
        if not resolved:
            raise SourceUnsetError(
                "recorder.source is not set. It is UNRESOLVED — SPEC §10 D2, live camera "
                "vs pre-ingested recording, and the recorder will not pick for you.\n"
                "  Set it in config/settings.yaml:\n"
                "    recorder:\n"
                "      source: rtsp://user:pass@camera.local:554/stream1   # live camera\n"
                "      source: tests/fixtures/demo.mp4                     # a recording\n"
                "  Or override for one run:  python3 -m services.recorder --source ...\n"
                "  Both forms are supported; the decision is which one the demo runs on."
            )

        settings = cls(
            camera_id=str(config.get("camera.id")),
            source=normalize_source(str(resolved)),
            archive_dir=config.repo_path("paths.archive"),
            segment_seconds=int(config.get("recorder.segment_seconds")),
            filename_pattern=str(config.get("recorder.filename_pattern")),
            container=str(config.get("recorder.container")),
            copy_codec=bool(config.get("recorder.copy_codec")),
            rtsp_transport=str(setting("recorder.rtsp_transport")),
            realtime_file_playback=bool(setting("recorder.realtime_file_playback")),
            ffmpeg_bin=str(os.environ.get("SPARK_FFMPEG") or setting("recorder.ffmpeg_bin")),
            loglevel=_FFMPEG_LOGLEVEL.get(
                str(config.get("logging.level", "INFO")).upper(), "warning"
            ),
            device_video_size=str(setting("recorder.device.video_size")),
            device_framerate=int(setting("recorder.device.framerate")),
            device_input_format=(
                str(fmt) if (fmt := setting("recorder.device.input_format")) else None
            ),
            encoder=str(setting("recorder.device.encoder")),
            encoder_args=tuple(str(a) for a in setting("recorder.device.encoder_args")),
            device_keyframe_interval_seconds=float(
                setting("recorder.device.keyframe_interval_seconds")
            ),
            preview_enabled=bool(config.get("recorder.preview.enabled", False)),
            preview_path=(
                config.repo_path("recorder.preview.path")
                if config.get("recorder.preview.enabled", False)
                else None
            ),
            preview_fps=float(config.get("recorder.preview.fps", 2.0)),
            preview_width=int(config.get("recorder.preview.width", 640)),
            preview_quality=int(config.get("recorder.preview.quality", 6)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Catch the configuration mistakes that produce a corrupt archive rather than a
        crash — the ones nobody notices until the deep worker cannot open a file."""
        if self.segment_seconds <= 0:
            raise RecorderError(
                f"recorder.segment_seconds must be positive, got {self.segment_seconds}"
            )
        suffix = Path(self.filename_pattern).suffix.lstrip(".").lower()
        if suffix and suffix != self.container.lower():
            raise RecorderError(
                f"recorder.filename_pattern ends in {suffix!r} but recorder.container is "
                f"{self.container!r}. ffmpeg would mux {self.container} into files named "
                f"{suffix} and every downstream tool would trust the extension."
            )
        classify_source(self.source)
