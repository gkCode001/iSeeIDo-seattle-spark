"""M1 — ingest (SPEC §2).

Walks the archive in 5 s windows on a 4 s stride, throws most of them away with a cheap
detector gate, captions what survives through the one VLM client, and writes a
``ChunkRecord`` for every window either way.

Typical use::

    from services.index import build_index
    from services.ingest import IngestPipeline, IngestSettings

    settings = IngestSettings.from_config()
    with build_index() as index, IngestPipeline(settings, sink=index) as pipeline:
        stats = pipeline.run()
        print(stats.skip_rate)          # SPEC §2.3's health metric

Three things this package does not do, on purpose:

* **Touch the camera.** The recorder (SPEC §2.1) owns it and runs independently, so
  ingest can be restarted without risking footage.
* **Copy or cut video.** A window is a time range; ``shared/timecode.py`` maps it onto
  segment files (CLAUDE.md invariant 3).
* **Define a record.** ``shared/schema.py`` owns ``ChunkRecord``. Nothing here redefines
  it, and every record carries wall clock *and* ``segment`` + ``pts_offset``
  (invariant 2).

``vlm.backend: stub`` selects :class:`StubCaptioner`, which produces deterministic
synthetic captions so this whole path runs on real footage before an NGC key exists. It
is not a test mock — see ``captioner.py``.
"""

from .captioner import (
    STUB_MODEL,
    STUB_PREFIX,
    Captioner,
    StubCaptioner,
    VLMCaptioner,
    build_captioner,
)
from .ffmpeg import FFmpegDecodeError, FFmpegError, FFmpegMissingError, resolve_ffmpeg
from .frames import (
    FrameExtractor,
    OverlayTooSmallError,
    drawtext_filter,
    filter_chain,
    gmtime_expression,
    measure_text_height_px,
    resolve_overlay_fontsize,
    scale_filter,
    split_jpeg_stream,
)
from .gate import (
    Gate,
    GateDecision,
    GateReason,
    MotionGate,
    PassthroughGate,
    build_gate,
    mean_abs_delta,
    motion_score,
    split_thumbnails,
    thumbnail_command,
)
from .pipeline import ChunkSink, IngestPipeline, IngestStats
from .settings import (
    PENDING_SETTINGS,
    GateBackend,
    IngestError,
    IngestSettings,
    OverlayPosition,
)
from .windows import Window, archive_bounds, plan_windows

__all__ = [
    # the two things most callers need
    "IngestPipeline",
    "IngestSettings",
    # results
    "IngestStats",
    "ChunkSink",
    # SPEC §2.2 — windows
    "Window",
    "plan_windows",
    "archive_bounds",
    # SPEC §2.3 — the gate
    "Gate",
    "GateDecision",
    "GateReason",
    "MotionGate",
    "PassthroughGate",
    "build_gate",
    "mean_abs_delta",
    "motion_score",
    "split_thumbnails",
    "thumbnail_command",
    # SPEC §2.4 — frames and overlay
    "FrameExtractor",
    "drawtext_filter",
    "filter_chain",
    "gmtime_expression",
    "measure_text_height_px",
    "resolve_overlay_fontsize",
    "scale_filter",
    "split_jpeg_stream",
    "OverlayTooSmallError",
    # SPEC §2.4 — captioning
    "Captioner",
    "StubCaptioner",
    "VLMCaptioner",
    "build_captioner",
    "STUB_MODEL",
    "STUB_PREFIX",
    # configuration and errors
    "GateBackend",
    "OverlayPosition",
    "PENDING_SETTINGS",
    "IngestError",
    "FFmpegError",
    "FFmpegMissingError",
    "FFmpegDecodeError",
    "resolve_ffmpeg",
]
