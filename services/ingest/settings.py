"""Ingest configuration, resolved from ``config/settings.yaml`` (SPEC §2.2–§2.4).

Nothing here touches ffmpeg, the VLM or the filesystem. It exists so that everything
downstream can be a pure function of one immutable object, and so that the numbers M1
turns live in exactly one place.

Pending settings
----------------
``settings.yaml`` carries every dial SPEC §2.2–§2.4 names, but a handful of mechanical
knobs the implementation needs are not there yet — the gate's own sample rate, the
subprocess timeout, the drawtext cosmetics. They are listed once in
:data:`PENDING_SETTINGS` with the values we would put in the YAML, and read through
:func:`setting`, so that adding the keys takes over with no code change. This mirrors
``services/recorder/settings.py`` and is deliberately the *only* place in M1 where a
number has a fallback: CLAUDE.md's "no magic numbers" rule is about literals scattered
through service code, and one labelled table that names its own migration is the least
bad way to hold them until the config file can be edited.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from shared import config

__all__ = [
    "PENDING_SETTINGS",
    "IngestError",
    "GateBackend",
    "OverlayPosition",
    "IngestSettings",
    "setting",
]


# Proposed ``settings.yaml`` additions. Every entry is a knob M1 needs and the YAML does
# not have yet; the dotted key is where it should live.
# Every M1 tunable now lives in config/settings.yaml, where the rationale lives with it.
# This table is intentionally EMPTY: `setting()` raises KeyError for anything missing
# from both, so a typo'd key fails loudly instead of silently taking a default. A default
# in code is a magic number wearing a disguise (CLAUDE.md), and a fallback that shadows
# nothing just confuses the next person hunting for the dial.
PENDING_SETTINGS: dict[str, Any] = {}


class IngestError(RuntimeError):
    """Base class for M1 failures a human is expected to read and act on."""


class GateBackend(str, Enum):
    """SPEC §2.3. ``motion`` is what ships; the other two are the upgrade path.

    DeepStream is absent from this box and its sm_121 support is unverified (CLAUDE.md
    machine state), so the gate that exists is a frame-diff one. The architecture is the
    same either way — something cheap decides whether the VLM runs — which is the point
    of naming the backend in config rather than assuming one.
    """

    MOTION = "motion"
    DEEPSTREAM = "deepstream"
    TENSORRT = "tensorrt"


class OverlayPosition(str, Enum):
    """Where the burned wall clock sits. ``bottom`` matches SPEC §2.4."""

    BOTTOM = "bottom"
    TOP = "top"


def setting(dotted: str) -> Any:
    """Read a setting, falling back to :data:`PENDING_SETTINGS` for keys not in the YAML.

    Raises for a key in neither, which is a programming error rather than a
    configuration one. Note the fallback is looked up ONLY when the table actually
    carries the key — indexing it eagerly as ``config.get``'s default argument would
    raise ``KeyError`` for every setting once the table empties, which is exactly what
    happens when the pending keys finally land in the YAML.
    """
    if dotted in PENDING_SETTINGS:
        return config.get(dotted, PENDING_SETTINGS[dotted])
    return config.get(dotted)


@dataclass(frozen=True)
class IngestSettings:
    """Everything M1 needs, resolved once and then immutable.

    The three groups map onto SPEC §2.2 (windows), §2.3 (gate) and §2.4 (captioning), and
    the field order follows that so the object reads like the spec section it implements.
    """

    # -- identity and location ----------------------------------------------------
    camera_id: str
    archive_dir: Path
    ffmpeg_bin: str
    ffmpeg_timeout_seconds: float

    # -- SPEC §2.2 analysis windows -----------------------------------------------
    window_seconds: float
    stride_seconds: float
    sample_fps: float
    live_short_side_px: int
    frame_jpeg_quality: int

    # -- SPEC §2.3 the detector gate ----------------------------------------------
    gate_enabled: bool
    gate_backend: GateBackend
    gate_sample_fps: float
    motion_threshold: float
    thumbnail_size: int
    warmup_windows: int
    target_skip_rate: float
    warn_skip_rate: float
    active_area_enabled: bool
    active_area_noise_level: int
    active_area_min_px: int

    # -- SPEC §2.4 captioning ------------------------------------------------------
    overlay_enabled: bool
    overlay_format: str
    overlay_position: OverlayPosition
    overlay_min_height_px: int
    overlay_fontfile: str
    overlay_fontsize: int
    overlay_fontcolor: str
    overlay_box_opacity: float
    overlay_padding_px: int
    overlay_max_fontsize: int
    vlm_backend: str
    caption_prompt: str
    watchlist_enabled: bool
    watchlist_path: Path
    watchlist_seed_path: Path
    watchlist_preamble: str
    watchlist_max_items: int
    caption_timeout_seconds: float
    poll_interval_seconds: float
    bench_target_seconds: float

    # -- derived --------------------------------------------------------------------

    @property
    def frames_per_window(self) -> int:
        """How many frames the VLM sees per window. SPEC §2.2: 5 s at 1 fps → 5 frames."""
        return max(1, round(self.window_seconds * self.sample_fps))

    @property
    def thumbnail_bytes(self) -> int:
        """Bytes per gate thumbnail. 32x32 gray = 1024, which is the whole argument for
        diffing them in pure Python instead of adding numpy."""
        return self.thumbnail_size * self.thumbnail_size

    @classmethod
    def from_config(cls, archive_dir: str | Path | None = None) -> IngestSettings:
        """Build from ``settings.yaml``. ``archive_dir`` overrides ``paths.archive``."""
        live_timeout = config.get("vlm.profiles.live.request_timeout_seconds", None)
        pause = config.get("vlm.queue.max_ingest_pause_seconds", None)
        caption_timeout = setting("ingest.caption_timeout_seconds")
        if caption_timeout is None:
            # The longest a well-behaved ingest caption can take: its own HTTP timeout
            # plus the queue's bounded pause (SPEC §7 — ingest may be paused, never
            # starved). Anything past that is a wedge, not a slow model.
            caption_timeout = float(live_timeout or 0.0) + float(pause or 0.0)

        settings = cls(
            camera_id=str(config.get("camera.id")),
            archive_dir=(
                Path(archive_dir).expanduser().resolve()
                if archive_dir is not None
                else config.repo_path("paths.archive")
            ),
            ffmpeg_bin=str(
                os.environ.get("SPARK_FFMPEG")
                or setting("ingest.ffmpeg_bin")
                or config.get("recorder.ffmpeg_bin")
            ),
            ffmpeg_timeout_seconds=float(setting("ingest.ffmpeg_timeout_seconds")),
            window_seconds=float(config.get("ingest.window_seconds")),
            stride_seconds=float(config.get("ingest.stride_seconds")),
            sample_fps=float(config.get("ingest.sample_fps")),
            live_short_side_px=int(config.get("ingest.live_short_side_px")),
            frame_jpeg_quality=int(setting("ingest.frame_jpeg_quality")),
            gate_enabled=bool(config.get("ingest.gate.enabled")),
            gate_backend=GateBackend(str(config.get("ingest.gate.backend"))),
            gate_sample_fps=float(setting("ingest.gate.sample_fps")),
            motion_threshold=float(config.get("ingest.gate.motion_threshold")),
            thumbnail_size=int(config.get("ingest.gate.thumbnail_size")),
            warmup_windows=int(config.get("ingest.gate.warmup_windows")),
            target_skip_rate=float(config.get("ingest.gate.target_skip_rate")),
            warn_skip_rate=float(config.get("ingest.gate.warn_skip_rate")),
            active_area_enabled=bool(config.get("ingest.gate.active_area.enabled")),
            active_area_noise_level=int(config.get("ingest.gate.active_area.noise_level")),
            active_area_min_px=int(config.get("ingest.gate.active_area.min_px")),
            overlay_enabled=bool(config.get("ingest.overlay.enabled")),
            overlay_format=str(config.get("ingest.overlay.format")),
            overlay_position=OverlayPosition(str(config.get("ingest.overlay.position"))),
            overlay_min_height_px=int(config.get("ingest.overlay.min_height_px")),
            overlay_fontfile=str(config.get("ingest.overlay.fontfile")),
            overlay_fontsize=int(config.get("ingest.overlay.fontsize")),
            overlay_fontcolor=str(setting("ingest.overlay.fontcolor")),
            overlay_box_opacity=float(setting("ingest.overlay.box_opacity")),
            overlay_padding_px=int(config.get("ingest.overlay.padding_px")),
            overlay_max_fontsize=int(setting("ingest.overlay.max_fontsize")),
            vlm_backend=str(config.get("vlm.backend")),
            caption_prompt=str(config.get("vlm.prompts.caption")).strip(),
            watchlist_enabled=bool(setting("ingest.watchlist.enabled")),
            watchlist_path=config.repo_path("ingest.watchlist.path"),
            watchlist_seed_path=config.repo_path("monitor.tasks_file"),
            watchlist_preamble=str(setting("vlm.prompts.watchlist_preamble") or "").strip(),
            watchlist_max_items=int(setting("ingest.watchlist.max_items")),
            caption_timeout_seconds=float(caption_timeout),
            poll_interval_seconds=float(setting("ingest.poll_interval_seconds")),
            bench_target_seconds=float(setting("ingest.bench.target_caption_seconds")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Catch the configurations that produce a *wrong* index rather than a crash.

        These are the ones nobody notices: a stride longer than the window silently drops
        footage between chunks, and a gate that never fires silently costs real-time.
        """
        if self.window_seconds <= 0:
            raise IngestError(f"ingest.window_seconds must be positive, got {self.window_seconds}")
        if self.stride_seconds <= 0:
            raise IngestError(f"ingest.stride_seconds must be positive, got {self.stride_seconds}")
        if self.stride_seconds > self.window_seconds:
            raise IngestError(
                f"ingest.stride_seconds ({self.stride_seconds}) exceeds "
                f"ingest.window_seconds ({self.window_seconds}); footage between windows "
                f"would never be analysed and the gap would be invisible in the index. "
                f"SPEC §2.2 wants a 1 s overlap, not a hole."
            )
        if self.sample_fps <= 0:
            raise IngestError(f"ingest.sample_fps must be positive, got {self.sample_fps}")
        if self.gate_sample_fps <= 0:
            raise IngestError(f"ingest.gate.sample_fps must be positive, got {self.gate_sample_fps}")
        if not 0.0 <= self.motion_threshold <= 1.0:
            raise IngestError(
                f"ingest.gate.motion_threshold is a normalised 0..1 mean absolute delta; "
                f"got {self.motion_threshold}"
            )
        if self.thumbnail_size <= 1:
            raise IngestError(
                f"ingest.gate.thumbnail_size must be >1 px, got {self.thumbnail_size}"
            )
        if self.warmup_windows < 0:
            raise IngestError(f"ingest.gate.warmup_windows must be >=0, got {self.warmup_windows}")
        if not 0 <= self.active_area_noise_level <= 255:
            raise IngestError(
                f"ingest.gate.active_area.noise_level is a 0..255 grayscale level; "
                f"got {self.active_area_noise_level}"
            )
        if not 0 <= self.active_area_min_px <= self.thumbnail_bytes:
            # Above the pixel count every window would score 0.0 and the VLM would never
            # run again — the exact silent failure this whole block exists to catch.
            raise IngestError(
                f"ingest.gate.active_area.min_px must be between 0 and the thumbnail's "
                f"{self.thumbnail_bytes} pixels, got {self.active_area_min_px}"
            )
        if self.live_short_side_px <= 0:
            raise IngestError(
                f"ingest.live_short_side_px must be positive, got {self.live_short_side_px}"
            )
        if self.overlay_enabled and self.overlay_min_height_px <= 0:
            raise IngestError(
                f"ingest.overlay.min_height_px must be positive when the overlay is on, "
                f"got {self.overlay_min_height_px}"
            )
        if self.overlay_enabled and not Path(self.overlay_fontfile).is_file():
            raise IngestError(
                f"ingest.overlay.fontfile {self.overlay_fontfile!r} does not exist. "
                f"drawtext fails the whole ffmpeg call without one, so every window would "
                f"decode-fail. Install fonts-dejavu-core or point the setting elsewhere."
            )
        if not self.caption_prompt:
            raise IngestError(
                "vlm.prompts.caption is empty; an empty prompt produces an empty caption "
                "and an index full of nothing."
            )
        if self.watchlist_enabled and not self.watchlist_preamble:
            # Silently degrading here is the bad outcome: captions stay general, standing
            # tasks keep missing details, and nothing in the logs says why.
            raise IngestError(
                "ingest.watchlist.enabled is true but vlm.prompts.watchlist_preamble is "
                "empty, so no task checklist would reach the captioner and standing tasks "
                "would silently go back to matching on whatever the caption happened to "
                "mention. Set the preamble or set ingest.watchlist.enabled: false."
            )
        if self.watchlist_enabled and self.watchlist_max_items <= 0:
            raise IngestError(
                f"ingest.watchlist.max_items must be positive when the watchlist is on, "
                f"got {self.watchlist_max_items}"
            )
