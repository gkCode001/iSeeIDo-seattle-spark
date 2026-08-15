"""Deep-worker configuration, resolved from ``config/settings.yaml`` (SPEC §5, §4.3).

Nothing in this module touches a process, a socket or the filesystem. It exists so that
the decode plan, the confidence heuristic and the job backstops are all built from one
resolved value object rather than from a dozen scattered ``config.get`` calls — and so
that the two invariants this path can violate silently are checked **once, loudly, at
construction**:

* ``vlm.profiles.deep.native_resolution`` must be true. CLAUDE.md invariant 7: the deep
  path exists to read fine detail. A downscale here does not fail, it just quietly
  answers worse, which is the failure mode we can least afford.
* ``vlm.profiles.deep.sample_fps`` must be positive. Zero frames is an answer about
  nothing.

Pending settings
----------------
Five keys the deep worker needs are not in ``settings.yaml`` yet, and M4 may not edit
that file. They are listed once in :data:`PENDING_SETTINGS` with the values we would
write into the YAML, and read through :func:`setting`, so that adding them takes over
with no code change. This follows the precedent in ``services/recorder/settings.py``:
CLAUDE.md's "no magic numbers" rule is about literals scattered through service code, and
one labelled table that names its own migration is the least bad way to hold them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared import config
from shared.vlm_client import Profile, ProfileSpec

__all__ = [
    "PENDING_SETTINGS",
    "WorkerSettings",
    "setting",
]


#: Proposed ``settings.yaml`` additions, each shown under the block it belongs in.
PENDING_SETTINGS: dict[str, Any] = {
    # agent.deep — the job backstops of SPEC §4.3 live here already (timeout_seconds,
    # max_inflight, dedupe_identical_ranges); these are the ones M4 adds.
    #
    # SPEC §5's own worked example is 30 s -> ~120 frames at 4 fps, with a realistic
    # target of 20-60 s per job. 60 s (240 frames) is twice that example and is the point
    # past which the prefill, not the 600-token trace, starts to dominate. A longer range
    # is refused with a sentence rather than truncated, because a silently halved range
    # produces a confident answer about footage nobody asked about.
    "agent.deep.max_range_seconds": 60.0,
    # A copy-free decode of <=60 s of 1080p is a couple of seconds. Anything near this
    # means the archive is wrong, not slow — and the decode holds the single in-flight
    # deep slot while it runs.
    "agent.deep.decode_timeout_seconds": 30.0,
    # ffmpeg -q:v for the extracted JPEGs. 2 is the top of the useful range: the frames
    # are native resolution (invariant 7) and this is the only lossy step between the
    # archive and the VLM, so it is not where bytes get saved.
    "agent.deep.frame_quality": 2,
    # Confidence multiplier applied when the answer text hedges — see
    # ``services.worker.analysis.derive_confidence``. A heuristic weight, deliberately
    # harsh: the deep prompt tells the model to say so explicitly when the footage does
    # not show enough, so when it does say so we believe it.
    "agent.deep.hedged_confidence_factor": 0.35,
    # How many finished jobs stay addressable by ``job_id``. The UI reloads and asks for
    # a job that finished before the reload (SPEC §11.4), so this cannot be zero; it is a
    # memory guard, not a tuning dial.
    "agent.deep.job_history": 256,
    # ingest.overlay — the block exists; this is the one field it lacks. Distance from
    # the frame edge to the overlay box, in pixels.
    "ingest.overlay.padding_px": 8,
}


def setting(dotted: str) -> Any:
    """``config.get`` with the pending-settings table as the fallback.

    A key present in ``settings.yaml`` always wins, so landing these in the YAML silently
    retires the fallback rather than leaving two sources of truth.
    """
    if dotted not in PENDING_SETTINGS:
        return config.get(dotted)
    return config.get(dotted, PENDING_SETTINGS[dotted])


@dataclass(frozen=True)
class WorkerSettings:
    """Everything M4 needs, resolved once.

    Frozen because a job must not be able to retune the worker mid-flight: the elapsed
    timer the UI prints against ``timeout_seconds`` (SPEC §11.2) would then be measuring
    against a number that changed underneath it.
    """

    # -- identity and the archive ------------------------------------------------------
    camera_id: str
    archive_dir: Path
    clips_dir: Path
    clip_container: str
    copy_codec: bool
    ffmpeg_bin: str
    clip_timeout_seconds: float

    # -- the deep profile — SPEC §5 ----------------------------------------------------
    sample_fps: float
    native_resolution: bool
    max_tokens: int
    deep_prompt: str
    backend: str

    # -- job backstops — SPEC §4.3 -----------------------------------------------------
    timeout_seconds: float
    max_inflight: int
    dedupe_identical_ranges: bool
    max_range_seconds: float
    job_history: int

    # -- decode ------------------------------------------------------------------------
    decode_timeout_seconds: float
    frame_quality: int

    # -- confidence --------------------------------------------------------------------
    hedged_confidence_factor: float

    # -- overlay — CLAUDE.md invariant 8 -----------------------------------------------
    overlay_enabled: bool
    overlay_format: str
    overlay_fontfile: str
    overlay_fontsize: int
    overlay_position: str
    overlay_padding_px: int

    @classmethod
    def from_config(cls, *, archive_dir: str | Path | None = None) -> WorkerSettings:
        deep: ProfileSpec = ProfileSpec.from_config(Profile.DEEP)

        # SPEC §5 states these two as facts about the deep path, and settings.yaml carries
        # them as ``sample_fps: 4`` / ``native_resolution: true``. Absent means someone
        # deleted them, which must not degrade into a guess.
        if deep.sample_fps is None or deep.sample_fps <= 0:
            raise config.ConfigError(
                "vlm.profiles.deep.sample_fps must be a positive number; SPEC §5 puts the "
                "deep path at 4 fps and the worker has no honest default for 'no frames'"
            )
        if not deep.native_resolution:
            raise config.ConfigError(
                "vlm.profiles.deep.native_resolution is false. CLAUDE.md invariant 7: "
                "downscaling only ever applies to the live path — the deep path exists to "
                "read detail the caption missed, and a resize here fails silently by "
                "simply answering worse."
            )

        position = str(config.get("ingest.overlay.position")).strip().lower()
        if position not in ("bottom", "top"):
            raise config.ConfigError(
                f"ingest.overlay.position must be 'bottom' or 'top', got {position!r}"
            )

        return cls(
            camera_id=str(config.get("camera.id")),
            archive_dir=(
                Path(archive_dir) if archive_dir is not None else config.repo_path("paths.archive")
            ),
            clips_dir=config.repo_path("paths.clips"),
            clip_container=str(config.get("recorder.container")),
            copy_codec=bool(config.get("recorder.copy_codec")),
            ffmpeg_bin=str(config.get("recorder.ffmpeg_bin")),
            clip_timeout_seconds=float(config.get("recorder.clip_timeout_seconds")),
            sample_fps=float(deep.sample_fps),
            native_resolution=True,
            max_tokens=int(deep.max_tokens),
            deep_prompt=str(config.get("vlm.prompts.deep")),
            backend=str(config.get("vlm.backend")).strip().lower(),
            timeout_seconds=float(config.get("agent.deep.timeout_seconds")),
            max_inflight=int(config.get("agent.deep.max_inflight")),
            dedupe_identical_ranges=bool(config.get("agent.deep.dedupe_identical_ranges")),
            max_range_seconds=float(setting("agent.deep.max_range_seconds")),
            job_history=int(setting("agent.deep.job_history")),
            decode_timeout_seconds=float(setting("agent.deep.decode_timeout_seconds")),
            frame_quality=int(setting("agent.deep.frame_quality")),
            hedged_confidence_factor=float(setting("agent.deep.hedged_confidence_factor")),
            overlay_enabled=bool(config.get("ingest.overlay.enabled")),
            overlay_format=str(config.get("ingest.overlay.format")),
            overlay_fontfile=str(config.get("ingest.overlay.fontfile")),
            overlay_fontsize=int(config.get("ingest.overlay.fontsize")),
            overlay_position=position,
            overlay_padding_px=int(setting("ingest.overlay.padding_px")),
        )

    @property
    def max_frames(self) -> int:
        """Frames a maximum-length range decodes to. Reported, never used to truncate."""
        return int(math.ceil(self.max_range_seconds * self.sample_fps))

    @property
    def is_stub_backend(self) -> bool:
        return self.backend == "stub"
