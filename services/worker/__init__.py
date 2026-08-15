"""M4 — the deep worker (SPEC §5). Headless, one entry point, shared by M3 and M5.

    from services.worker import deep_analyze, submit

    job = submit(t_start, t_end, "was the rear door open?")     # QUEUED, returns at once
    ...                                                          # M3's turn ends here
    job = worker.poll(job)                                       # DONE, with the answer

    job = deep_analyze(t0, t1, q, priority=Priority.VERIFICATION)  # M5 stage 3, blocking

Which form to use is not a preference. **A user turn must never block on this**
(CLAUDE.md invariant 4): M3 answers provisionally from the index, hands the UI a
``job_id``, and streams the refinement. :func:`deep_analyze` is for callers that are
already on a background thread.

``vlm.backend`` is ``stub`` on this box — there is no NGC key and nothing serving. The
stub decodes the real frames, cuts a real clip and travels through the real queue; what it
does not do is look at the pixels, and every answer it returns is stamped
:data:`~services.worker.analysis.STUB_MARKER` so it cannot be mistaken for a real one on
stage.
"""

from services.worker.analysis import (
    STUB_MARKER,
    AnalysisBackend,
    AnalysisRequest,
    AnalysisResult,
    StubAnalysisBackend,
    VLMAnalysisBackend,
    build_analysis_backend,
    confidence_explanation,
    derive_confidence,
    detect_hedge,
)
from services.worker.decode import (
    DecodePlan,
    DecodeStep,
    FfmpegFrameExtractor,
    FrameExtractor,
    NullFrameExtractor,
    build_decode_plan,
    drawtext_expansion,
    video_filter,
)
from services.worker.settings import PENDING_SETTINGS, WorkerSettings
from services.worker.worker import (
    TERMINAL_STATES,
    DeepReport,
    DeepWorker,
    Submission,
    archive_resolver,
    deep_analyze,
    default_worker,
    ffmpeg_worker_from_config,
    set_default_worker,
    submit,
)

__all__ = [
    "deep_analyze",
    "submit",
    "DeepWorker",
    "DeepReport",
    "Submission",
    "TERMINAL_STATES",
    "default_worker",
    "set_default_worker",
    "ffmpeg_worker_from_config",
    "archive_resolver",
    "WorkerSettings",
    "PENDING_SETTINGS",
    "AnalysisBackend",
    "AnalysisRequest",
    "AnalysisResult",
    "StubAnalysisBackend",
    "VLMAnalysisBackend",
    "build_analysis_backend",
    "derive_confidence",
    "confidence_explanation",
    "detect_hedge",
    "STUB_MARKER",
    "DecodePlan",
    "DecodeStep",
    "FrameExtractor",
    "NullFrameExtractor",
    "FfmpegFrameExtractor",
    "build_decode_plan",
    "drawtext_expansion",
    "video_filter",
]
