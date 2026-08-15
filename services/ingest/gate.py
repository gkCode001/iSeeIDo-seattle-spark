"""The detector gate — SPEC §2.3.

**This is the main reason real-time is achievable.** On a fixed camera most windows are
an empty scene. Something cheap decides whether the VLM runs; the VLM costs ~2 s and the
gate costs ~0.1 s (SPEC §8), so skipping 80% of windows is worth more than any amount of
inference tuning. CLAUDE.md says it plainly: prefer deleting work over parallelizing it.

How this one works, and why it needs no new dependency
------------------------------------------------------
DeepStream is absent from this box and its sm_121 support is unverified, so the gate that
ships is a frame-diff (``ingest.gate.backend: motion``). ffmpeg does the expensive half —

    -vf "fps=N,scale=32:32,format=gray" -f rawvideo

— which hands back **1024 bytes per frame**. Diffing 1 KB buffers is something pure
Python does perfectly well, so there is no numpy, no OpenCV, and no ARM64 wheel hunt.
The score is the mean absolute delta between consecutive thumbnails, normalised to 0..1,
compared against ``ingest.gate.motion_threshold``.

Three decisions that are not obvious
------------------------------------
**The window score is the maximum over frame pairs, not the mean.** A gate answers "did
anything happen", and one second of movement inside five seconds of stillness averages
away to nothing. The max is what "anything" means.

**The last thumbnail of the previous window is carried forward** as the first reference
of the next. Windows overlap by 1 s already (SPEC §2.2), but a movement that begins in
the final frame of a window would otherwise have nothing to be compared against.

**The gate fails open.** Warmup windows, an undecodable segment, an unreadable thumbnail
stream — all pass. A false skip is invisible: there is no record of what the VLM would
have said, and nothing downstream can tell "nothing happened" from "we did not look".
A false caption merely costs 2 s.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from shared.timecode import SegmentSpan

from .ffmpeg import BASE_ARGS, FFmpegDecodeError, run_ffmpeg
from .settings import GateBackend, IngestError, IngestSettings
from .telemetry import log_event
from .windows import Window

__all__ = [
    "GateReason",
    "GateDecision",
    "Gate",
    "MotionGate",
    "PassthroughGate",
    "build_gate",
    "mean_abs_delta",
    "motion_score",
    "split_thumbnails",
    "thumbnail_command",
]

#: Maximum value of one grayscale sample. Normalising by it is what makes
#: ``ingest.gate.motion_threshold`` a unit-free 0..1 number rather than a number that
#: silently means something different if the pixel format ever changes.
_GRAY_MAX = 255


class GateReason(str, Enum):
    """Why the gate decided what it decided. Logged verbatim — these strings are the
    only way to tell a healthy 85% skip rate from an 85% made of decode failures."""

    MOTION = "motion"  # passed: something moved
    STILL = "still"  # skipped: nothing moved
    WARMUP = "warmup"  # passed: no reference frame yet (ingest.gate.warmup_windows)
    DISABLED = "disabled"  # passed: ingest.gate.enabled is false
    UNDECODABLE = "undecodable"  # passed: we could not look, so we do not claim to know
    NO_FOOTAGE = "no_footage"  # passed: the range resolved to nothing but gaps


@dataclass(frozen=True)
class GateDecision:
    """One gate verdict, with the number behind it.

    ``score`` is the normalised 0..1 motion figure, or None when no comparison was
    possible. It is kept even for a skip: tuning ``motion_threshold`` means looking at
    the distribution of scores that were just under it.
    """

    passed: bool
    reason: GateReason
    score: float | None = None
    frames: int = 0
    error: str = ""

    @property
    def skipped(self) -> bool:
        return not self.passed


class Gate(Protocol):
    """What the pipeline needs from a gate. ``deepstream`` slots in here unchanged."""

    def evaluate(self, window: Window, spans: Sequence[SegmentSpan]) -> GateDecision:
        """Decide whether ``window`` reaches the VLM."""
        ...

    def reset(self) -> None:
        """Forget the reference frame. Called when the walk jumps across a gap."""
        ...


# --------------------------------------------------------------------------------------
# The arithmetic — pure, and the part worth testing
# --------------------------------------------------------------------------------------


def mean_abs_delta(a: bytes, b: bytes) -> float:
    """Mean absolute difference between two equal-length grayscale buffers, 0..1.

    1024 bytes per frame at a handful of frames per window: a Python loop is not the
    bottleneck here, ffmpeg's decode is. ``zip`` over two ``bytes`` objects yields ints,
    so this allocates nothing per pixel.
    """
    if len(a) != len(b):
        raise ValueError(f"thumbnail size mismatch: {len(a)} vs {len(b)} bytes")
    if not a:
        raise ValueError("empty thumbnail; nothing to compare")
    total = sum(abs(x - y) for x, y in zip(a, b))
    return total / (len(a) * _GRAY_MAX)


def motion_score(frames: Sequence[bytes]) -> float | None:
    """Peak inter-frame delta across a window, or None with fewer than two frames.

    Maximum rather than mean: see the module docstring. One frame is not a failure — it
    is a window too short or a decode that gave up early — and the caller turns None into
    a fail-open pass rather than a zero, which would read as "definitely still".
    """
    if len(frames) < 2:
        return None
    return max(mean_abs_delta(frames[i - 1], frames[i]) for i in range(1, len(frames)))


def split_thumbnails(raw: bytes, frame_bytes: int) -> list[bytes]:
    """Slice a rawvideo stream into fixed-size frames, discarding any partial tail.

    A truncated final frame means ffmpeg was killed mid-write. Diffing it would compare
    real pixels against whatever the buffer happened to hold, and the resulting spurious
    delta would pass a window that should have been skipped — a silent tax on the skip
    rate rather than an error.
    """
    if frame_bytes <= 0:
        raise ValueError(f"frame_bytes must be positive, got {frame_bytes}")
    count = len(raw) // frame_bytes
    return [raw[i * frame_bytes : (i + 1) * frame_bytes] for i in range(count)]


def thumbnail_command(
    ffmpeg_bin: str,
    span: SegmentSpan,
    *,
    sample_fps: float,
    size: int,
) -> list[str]:
    """argv for the thumbnail extraction over one span. Pure — spawns nothing.

    ``-ss`` precedes ``-i`` deliberately: input seeking lets ffmpeg jump by index instead
    of decoding the file from the top, which is the difference between a gate that costs
    0.1 s and one that costs as much as the caption it was meant to avoid.

    ``scale=32:32`` ignores aspect ratio on purpose. The output is never looked at by a
    human or a model; it is a fixed-size fingerprint, and a fixed size is what lets the
    diff be a flat byte comparison.
    """
    if span.path is None:
        raise ValueError("cannot extract thumbnails from a gap span")
    return [
        ffmpeg_bin,
        *BASE_ARGS,
        "-ss",
        f"{span.pts_in:.3f}",
        "-i",
        str(span.path),
        "-t",
        f"{span.duration:.3f}",
        "-an",
        "-vf",
        f"fps={sample_fps},scale={size}:{size},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]


# --------------------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------------------


class MotionGate:
    """``ingest.gate.backend: motion`` — ffmpeg thumbnails plus a pure-Python diff.

    ``extract`` is injected so the decision logic can be tested as arithmetic against a
    list of byte strings, with no subprocess and no footage. The default implementation
    is :meth:`_extract_via_ffmpeg`.
    """

    def __init__(
        self,
        settings: IngestSettings,
        extract: Callable[[Sequence[SegmentSpan]], list[bytes]] | None = None,
    ) -> None:
        self._s = settings
        self._extract = extract if extract is not None else self._extract_via_ffmpeg
        self._reference: bytes | None = None

    def reset(self) -> None:
        self._reference = None

    def evaluate(self, window: Window, spans: Sequence[SegmentSpan]) -> GateDecision:
        # Warmup first, and before any decode: SPEC §2.3's warmup exists because the first
        # windows have no reference frame, and a window we were never going to skip is a
        # window whose thumbnails we need not pay for.
        if window.index < self._s.warmup_windows:
            self._prime(spans)
            return GateDecision(True, GateReason.WARMUP)

        usable = [s for s in spans if not s.is_gap]
        if not usable:
            self.reset()
            return GateDecision(True, GateReason.NO_FOOTAGE)

        try:
            frames = self._extract(usable)
        except FFmpegDecodeError as exc:
            # Fail open. We did not look, so we do not get to say nothing happened.
            self.reset()
            return GateDecision(True, GateReason.UNDECODABLE, error=str(exc))

        # The previous window's last frame is the reference for this one's first, so a
        # movement starting on a boundary is still a comparison rather than a lone frame.
        sequence = frames if self._reference is None else [self._reference, *frames]
        score = motion_score(sequence)
        if frames:
            self._reference = frames[-1]

        if score is None:
            return GateDecision(True, GateReason.UNDECODABLE, frames=len(frames))
        if score >= self._s.motion_threshold:
            return GateDecision(True, GateReason.MOTION, score=score, frames=len(frames))
        return GateDecision(False, GateReason.STILL, score=score, frames=len(frames))

    # -- internals ----------------------------------------------------------------

    def _prime(self, spans: Sequence[SegmentSpan]) -> None:
        """Leave a warmup window with a usable reference frame for the next one."""
        usable = [s for s in spans if not s.is_gap]
        if not usable:
            return
        try:
            frames = self._extract(usable)
        except FFmpegDecodeError:
            self.reset()
            return
        if frames:
            self._reference = frames[-1]

    def _extract_via_ffmpeg(self, spans: Sequence[SegmentSpan]) -> list[bytes]:
        """One ffmpeg call per span, concatenated.

        A window straddling two segment files is two calls (invariant 3 — an event on a
        boundary lives in two files). Concatenating across the seam is correct for a diff:
        the recorder cut mid-scene, so consecutive frames either side are genuinely
        consecutive in time.
        """
        frames: list[bytes] = []
        for span in spans:
            raw = run_ffmpeg(
                thumbnail_command(
                    self._s.ffmpeg_bin,
                    span,
                    sample_fps=self._s.gate_sample_fps,
                    size=self._s.thumbnail_size,
                ),
                timeout=self._s.ffmpeg_timeout_seconds,
            )
            got = split_thumbnails(raw, self._s.thumbnail_bytes)
            if len(raw) % self._s.thumbnail_bytes:
                log_event(
                    "gate.partial_thumbnail",
                    segment=Path(str(span.path)).name,
                    bytes=len(raw),
                    frame_bytes=self._s.thumbnail_bytes,
                )
            frames.extend(got)
        return frames


class PassthroughGate:
    """``ingest.gate.enabled: false`` — everything reaches the VLM.

    Not a debugging toy: it is the measurement baseline. The skip rate only means
    something against a run that captioned every window, and SPEC §9 wants the gate's
    contribution logged rather than assumed.
    """

    def reset(self) -> None:
        return None

    def evaluate(self, window: Window, spans: Sequence[SegmentSpan]) -> GateDecision:
        return GateDecision(True, GateReason.DISABLED)


def build_gate(
    settings: IngestSettings,
    extract: Callable[[Sequence[SegmentSpan]], list[bytes]] | None = None,
) -> Gate:
    """Select the gate named by ``ingest.gate.backend``.

    ``deepstream`` and ``tensorrt`` are named in settings.yaml as the upgrade path and are
    not implemented. They raise rather than falling back to motion: a gate silently
    running a different backend than the one configured is precisely the kind of thing
    that makes a skip-rate number meaningless.
    """
    if not settings.gate_enabled:
        return PassthroughGate()
    if settings.gate_backend is GateBackend.MOTION:
        return MotionGate(settings, extract)
    raise IngestError(
        f"ingest.gate.backend is {settings.gate_backend.value!r}, which is not implemented. "
        f"DeepStream is absent from this box and its sm_121 support is unverified "
        f"(CLAUDE.md machine state); {GateBackend.MOTION.value!r} is the backend that "
        f"ships. Set it back, or set ingest.gate.enabled: false to caption every window."
    )
