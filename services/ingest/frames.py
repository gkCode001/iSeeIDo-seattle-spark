"""Sampling, resizing and burning the wall-clock overlay — SPEC §2.4, steps 2 and 3.

    2. Sample to 5 frames, resize.
    3. **Burn wall-clock into the bottom of each frame, after the resize.**

CLAUDE.md invariant 8, restated because it is the one that fails silently: the VLM reads
the burned timestamp for temporal localization, so if the overlay is illegible the model
does not error — it simply stops citing times, and nobody notices until the deep worker
cannot locate anything. Hence two rules enforced mechanically here rather than by
convention:

**Scale comes first in the filter chain.** ``scale=...,drawtext=...`` — the text is
rendered at output resolution, at its configured pixel size, and is never resampled.
Reversing the order shrinks 20 px of text to 5 px of grey mush.

**The burned height is measured, not assumed.** :func:`resolve_overlay_fontsize` renders
the real string with the real font through the real ffmpeg and reads the ink height back
out; if it falls under ``ingest.overlay.min_height_px`` the fontsize is raised until it
does not, and the raise is logged. A configured fontsize is a preference, a minimum
height is a floor, and honouring a floor is what a floor is for.

Per-frame time, not per-window time
-----------------------------------
The overlay is ``%{pts:gmtime:<epoch>:<fmt>}``, so each of the five frames carries *its
own* wall clock rather than the window's start repeated five times. A window is 5 s wide;
a single timestamp would be wrong for four of the frames the model is asked to localize
against. The epoch base is the span's ``t_start`` because input seeking (``-ss`` before
``-i``) resets output PTS to zero — verified on this box against real archive footage.

**The escaping is M4's, imported rather than re-derived.** That expansion survives two
rounds of unescaping — the filtergraph option parser, then drawtext's own ``%{}``
splitter — so argument separators take one backslash and colons *inside* the strftime
format take three. Getting the count wrong does not crash: it renders ``Stray %`` into
the frame and localization degrades silently, which is exactly invariant 8's failure
mode. ``services/worker/decode.py`` established it empirically against ffmpeg 6.1.1 and
unit-tests it character for character, so this module imports
:func:`~services.worker.decode.drawtext_expansion` instead of keeping a second copy that
could drift. The two paths must burn identical text or the deep worker and the live
index would cite times in different formats. *This helper arguably belongs in
``shared/`` — neither service owns the other's file; see the M1 report.*

Resolution note (SPEC §2.5): the downscale is a **KV-cache pressure** dial, not a speed
one. Halving it saves ~0.3 s. It applies to the live analysis path only — the archive
stays native (invariant 7), and this module never writes to the archive.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from shared.timecode import SegmentSpan

# The one cross-service import in M1, and a deliberate one: see the module docstring.
# Duplicating this escaping is how the two burned clocks silently drift apart.
from services.worker.decode import drawtext_expansion

from .ffmpeg import BASE_ARGS, FFmpegDecodeError, run_ffmpeg
from .settings import IngestError, IngestSettings, OverlayPosition
from .telemetry import log_event

__all__ = [
    "OverlayTooSmallError",
    "escape_drawtext_value",
    "gmtime_expression",
    "scale_filter",
    "drawtext_filter",
    "filter_chain",
    "frame_command",
    "split_jpeg_stream",
    "measure_text_height_px",
    "resolve_overlay_fontsize",
    "FrameExtractor",
]

#: JPEG start-of-image marker. ``-f image2pipe`` concatenates whole JPEGs with no frame
#: header of its own, so this is the only boundary there is. It cannot occur inside the
#: entropy-coded data: a literal 0xFF there is byte-stuffed as 0xFF00.
_JPEG_SOI = b"\xff\xd8\xff"

#: A fixed instant used only to render a probe string of the right shape. Digits are
#: monospaced within a proportional font, so any concrete time measures the same height.
_PROBE_INSTANT = datetime(2026, 8, 14, 21, 11, 7, tzinfo=timezone.utc)

#: Threshold above which a probe pixel counts as ink. The probe renders a fully opaque
#: white box on pure black, so anything but pure black is the box; the margin is for
#: chroma-subsampling ringing, not for taste.
_INK_LEVEL = 40

#: Fallback ratio of rendered text height to fontsize, used only when the ffmpeg probe
#: itself fails. Measured on this box across fontsizes 10–32 with DejaVuSans: 0.70–0.75.
#: The low end is chosen so the fallback over-estimates the required fontsize rather than
#: under-estimating it — an overlay slightly larger than needed costs nothing.
_FALLBACK_HEIGHT_RATIO = 0.70

#: Padding drawn around the burned text. Purely cosmetic on the live path, but it is what
#: keeps the clock off the very edge of the frame where JPEG ringing is worst.
_BOX_BORDER_PX = 4


class OverlayTooSmallError(IngestError):
    """The burned wall clock cannot be made to reach ``ingest.overlay.min_height_px``."""


# --------------------------------------------------------------------------------------
# Filter-graph construction — pure, and where the invariants are actually enforced
# --------------------------------------------------------------------------------------


def escape_drawtext_value(text: str) -> str:
    """Escape a literal for use inside a single-quoted drawtext option value.

    Only backslash and the single quote need handling: the filtergraph parser treats
    everything between the quotes as literal, which is why ``:`` and ``,`` survive. This
    is *not* enough for the ``%{...}`` expansion layer — see :func:`gmtime_expression`,
    which has its own, worse, escaping problem.
    """
    return text.replace("\\", "\\\\").replace("'", r"\'")


def gmtime_expression(wall_start: datetime, fmt: str) -> str:
    """``%{pts:gmtime:<epoch>:<fmt>}`` — per-frame UTC wall clock, correctly escaped.

    A thin alias for ``services.worker.decode.drawtext_expansion``, kept so that M1 reads
    in M1's vocabulary while there is exactly one implementation of the escaping in the
    repo. Do not reimplement it here: the backslash counts are empirical, a wrong one
    fails silently, and two copies would drift.

    UTC, per SPEC §10 D8: the overlay is baked into the frames the model reads and cannot
    be changed afterwards, so it carries the same timezone as everything else that crosses
    a module boundary. The UI converts at render, once, in one direction.
    """
    return drawtext_expansion(wall_start, fmt)


def escape_option_value(value: str) -> str:
    """Escape a plain filter option value — a font path — for the filtergraph parser.

    Mirrors ``services/worker/decode.py``'s private helper. Trivial next to the expansion
    escaping above, but a fontfile path containing a colon would otherwise be read as the
    start of the next drawtext option and the filter would fail to load.
    """
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def scale_filter(short_side_px: int) -> str:
    """Downscale to ``short_side_px`` on the **short** side, never upscaling.

    Written as expressions rather than ``-2:512`` because the camera offers portrait modes
    (1080x1920 and 1728x3072 — CLAUDE.md machine state) and a fixed height would blow a
    portrait frame up to 512 px *wide*, quadrupling the token count of the thing SPEC §2.5
    asks us to shrink. ``min()`` keeps a source that is already small alone: upscaling
    invents no detail and costs real KV cache.

    ``-2`` on the free axis keeps it even, which mjpeg's yuvj420p requires.
    """
    if short_side_px <= 0:
        raise ValueError(f"short_side_px must be positive, got {short_side_px}")
    w = f"if(gt(iw,ih),-2,min({short_side_px},iw))"
    h = f"if(gt(iw,ih),min({short_side_px},ih),-2)"
    return f"scale=w='{w}':h='{h}'"


def drawtext_filter(
    text: str,
    *,
    fontfile: str,
    fontsize: int,
    fontcolor: str,
    box_opacity: float,
    padding_px: int,
    position: OverlayPosition = OverlayPosition.BOTTOM,
    boxcolor: str = "black",
    boxborderw: int = _BOX_BORDER_PX,
    escape: bool = True,
) -> str:
    """One drawtext filter, positioned by ``ingest.overlay.position``.

    ``y=h-text_h-<padding>`` rather than a literal offset — the same expression
    ``services/worker/decode.py`` uses. ``text_h`` is the rendered text height, so the
    overlay sits the same distance from the edge whatever fontsize
    :func:`resolve_overlay_fontsize` settled on. A hardcoded ``y=h-30`` silently pushes
    the text off the bottom of the frame the moment the fontsize is raised.

    The box behind the text is not decoration. A white clock on a white wall — this
    camera's actual scene — is unreadable, and an unreadable overlay is invariant 8's
    silent failure. ``boxcolor``/``boxborderw`` are parameters only so that
    :func:`measure_text_height_px` can render an opaque, borderless box whose inked rows
    are exactly ffmpeg's ``text_h``.
    """
    value = escape_drawtext_value(text) if escape else text
    if position is OverlayPosition.TOP:
        y = f"{padding_px}"
    else:
        y = f"h-text_h-{padding_px}"
    return (
        f"drawtext=fontfile={escape_option_value(fontfile)}"
        f":text='{value}'"
        f":x={padding_px}"
        f":y={y}"
        f":fontsize={fontsize}"
        f":fontcolor={fontcolor}"
        f":box=1"
        f":boxcolor={boxcolor}@{box_opacity}"
        f":boxborderw={boxborderw}"
    )


def filter_chain(settings: IngestSettings, wall_start: datetime, fontsize: int) -> str:
    """The full ``-vf`` chain for one span: sample, **then** scale, **then** burn.

    The order is CLAUDE.md invariant 8 and it is the whole reason this function exists
    rather than three string concatenations at the call site.
    """
    parts = [f"fps={settings.sample_fps}", scale_filter(settings.live_short_side_px)]
    if settings.overlay_enabled:
        parts.append(
            drawtext_filter(
                gmtime_expression(wall_start, settings.overlay_format),
                fontfile=settings.overlay_fontfile,
                fontsize=fontsize,
                fontcolor=settings.overlay_fontcolor,
                box_opacity=settings.overlay_box_opacity,
                padding_px=settings.overlay_padding_px,
                position=settings.overlay_position,
                # Already escaped for all three parsers; escaping again would double the
                # backslashes and produce a literal "%{pts..." burned into the frame.
                escape=False,
            )
        )
    return ",".join(parts)


def frame_command(settings: IngestSettings, span: SegmentSpan, fontsize: int) -> list[str]:
    """argv that turns one span into a stream of overlaid JPEGs on stdout.

    ``-ss`` before ``-i`` for the same reason the gate does it: index seeking rather than
    decoding from the top of the file. It also resets output PTS to zero, which is what
    makes ``span.t_start`` the correct epoch base for the burned clock.
    """
    if span.path is None:
        raise ValueError("cannot sample frames from a gap span")
    return [
        settings.ffmpeg_bin,
        *BASE_ARGS,
        "-ss",
        f"{span.pts_in:.3f}",
        "-i",
        str(span.path),
        "-t",
        f"{span.duration:.3f}",
        "-an",
        "-vf",
        filter_chain(settings, span.t_start, fontsize),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        str(settings.frame_jpeg_quality),
        "-",
    ]


def split_jpeg_stream(raw: bytes) -> list[bytes]:
    """Split a concatenated MJPEG stream into individual JPEG images.

    ``image2pipe`` writes whole JPEGs back to back with no framing of its own, so the
    start-of-image marker is the only boundary available. Preferred over writing numbered
    files to a temp directory: ingest runs for hours and a scratch directory nobody
    reaps is a failure at hour 30, not at hour 3.
    """
    if not raw:
        return []
    starts = []
    pos = raw.find(_JPEG_SOI)
    while pos != -1:
        starts.append(pos)
        pos = raw.find(_JPEG_SOI, pos + len(_JPEG_SOI))
    if not starts:
        return []
    bounds = [*starts, len(raw)]
    return [raw[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


# --------------------------------------------------------------------------------------
# The overlay legibility check — invariant 8, measured rather than asserted
# --------------------------------------------------------------------------------------


def measure_text_height_px(
    settings: IngestSettings,
    fontsize: int,
    *,
    text: str | None = None,
) -> int:
    """Render the overlay string on a black canvas and count the rows it inks.

    One ~30 ms lavfi call. The alternative is a font-metrics constant, and a constant is
    exactly the thing that goes stale when the fontfile changes and takes invariant 8 with
    it — silently, because an illegible overlay produces no error anywhere.

    The probe draws a fully opaque white box with ``boxborderw=0``, so the inked rows are
    ffmpeg's own ``text_h`` rather than an estimate of it.
    """
    probe = text if text is not None else _PROBE_INSTANT.strftime(settings.overlay_format)
    # Generous canvas: DejaVuSans averages well under one em per character, so
    # fontsize * len(text) cannot clip, and a clipped probe would under-report.
    width = max(2, fontsize * max(1, len(probe)))
    height = max(2, fontsize * 4)
    chain = drawtext_filter(
        probe,
        fontfile=settings.overlay_fontfile,
        fontsize=fontsize,
        fontcolor="white",
        box_opacity=1.0,
        padding_px=fontsize,
        position=OverlayPosition.TOP,
        boxcolor="white",
        boxborderw=0,
    )

    raw = run_ffmpeg(
        [
            settings.ffmpeg_bin,
            *BASE_ARGS,
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:d=1",
            "-vf",
            chain,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        timeout=settings.ffmpeg_timeout_seconds,
    )
    inked = [
        row
        for row in range(height)
        if raw[row * width : (row + 1) * width] and max(raw[row * width : (row + 1) * width]) > _INK_LEVEL
    ]
    return (max(inked) - min(inked) + 1) if inked else 0


def resolve_overlay_fontsize(
    settings: IngestSettings,
    measure: Callable[[int], int] | None = None,
) -> int:
    """Smallest fontsize >= the configured one whose burned text clears the floor.

    ``ingest.overlay.fontsize`` is a preference; ``ingest.overlay.min_height_px`` is a
    floor, and this honours the floor by raising the preference rather than by failing the
    run. The raise is logged as its own event, because a number the operator set being
    quietly overridden is the sort of thing that should appear in the log at hour 3 and
    not be discovered at hour 30.

    Measured on this box: DejaVuSans renders roughly 0.75 of its em as ink for a digit
    string, so the shipped ``fontsize: 20`` yields **15 px** against a ``min_height_px``
    of 16 — one pixel short, and this raises it to 22. That is a real finding about the
    shipped configuration, not a hypothetical.

    ``measure`` is injected so the search is testable without ffmpeg.
    """
    if not settings.overlay_enabled:
        return settings.overlay_fontsize

    probe = measure if measure is not None else (lambda size: measure_text_height_px(settings, size))
    floor = settings.overlay_min_height_px
    configured = settings.overlay_fontsize

    for size in range(configured, settings.overlay_max_fontsize + 1):
        try:
            height = probe(size)
        except FFmpegDecodeError as exc:
            # Probing failed — usually a fontfile ffmpeg cannot open. Fall back to the
            # measured ratio and over-estimate, because refusing to run over a font metric
            # would be a worse failure than an overlay two pixels taller than needed.
            fallback = max(configured, math.ceil(floor / _FALLBACK_HEIGHT_RATIO))
            log_event(
                "overlay.probe_failed",
                configured_fontsize=configured,
                fallback_fontsize=fallback,
                min_height_px=floor,
                error=str(exc),
            )
            return fallback
        if height >= floor:
            if size != configured:
                log_event(
                    "overlay.fontsize_raised",
                    configured_fontsize=configured,
                    resolved_fontsize=size,
                    measured_height_px=height,
                    min_height_px=floor,
                    reason=(
                        "burned wall clock would be under ingest.overlay.min_height_px; "
                        "the VLM reads it for temporal localization (invariant 8)"
                    ),
                )
            return size

    raise OverlayTooSmallError(
        f"no fontsize up to {settings.overlay_max_fontsize} renders the wall-clock overlay "
        f"at ingest.overlay.min_height_px ({floor} px) with font "
        f"{settings.overlay_fontfile!r}. The VLM reads that text for temporal localization "
        f"and an illegible one fails silently (CLAUDE.md invariant 8), so this stops rather "
        f"than indexing footage nothing can localize against."
    )


# --------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------


class FrameExtractor:
    """Turns a window's spans into the frames the VLM sees.

    The resolved fontsize is computed once, lazily, and reused: it is a property of the
    font and the settings, not of the window, and paying 30 ms per window for a constant
    would be a strange way to spend the gate's savings.
    """

    def __init__(self, settings: IngestSettings, fontsize: int | None = None) -> None:
        self._s = settings
        self._fontsize = fontsize

    @property
    def fontsize(self) -> int:
        """The fontsize actually burned, after the min-height floor is applied."""
        if self._fontsize is None:
            self._fontsize = resolve_overlay_fontsize(self._s)
        return self._fontsize

    def extract(self, spans: Sequence[SegmentSpan]) -> list[bytes]:
        """JPEG bytes for every sampled frame in ``spans``, in time order.

        A window straddling two segment files is two ffmpeg calls and one concatenated
        list — invariant 3's "an event at 21:11:58 running 12 s is two files", handled
        once here instead of at every call site.
        """
        frames: list[bytes] = []
        for span in spans:
            if span.is_gap:
                continue
            raw = run_ffmpeg(
                frame_command(self._s, span, self.fontsize),
                timeout=self._s.ffmpeg_timeout_seconds,
            )
            frames.extend(split_jpeg_stream(raw))
        return frames
