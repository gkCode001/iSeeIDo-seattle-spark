"""Recorded video in, indexed footage out — the drop-folder import path.

The live path (recorder → archive → M1) is not the only way footage should be able to
enter this system. A demo needs a clip that is *chosen* rather than staged in front of a
lens, and the hackathon's own rules want an open dataset run through the pipeline rather
than only footage we shot ourselves.

**The insight that makes this small: wall-clock time lives in the filename.** Every
downstream piece — the deep worker's seek, evidence-clip cutting, the timeline, invariant
2's join between a caption and its pixels — resolves time by parsing
``recorder.filename_pattern`` off a segment name (``shared/timecode.py``). So a recorded
file does not need a new ingest path, a new schema field or a second archive. It needs to
be *sliced into correctly named segments*. Everything after that is the system that
already exists, unchanged, and the burned overlay comes out right for free because
``services/ingest/frames.py`` renders ``%{pts:gmtime:<epoch>}`` from the same filename.

Three decisions worth stating, because each has a quieter alternative that is wrong:

**Placement — the clip is laid down ending NOW.** ``t_end`` is the moment of import and
``t_start`` is that minus the file's real duration, so the recording occupies the minutes
that just passed. The alternative, honouring the file's own creation date, puts a 2019
dataset clip outside every default window: the ask surface looks back 30 minutes
(``agent.search.default_lookback_seconds``), so a truthful timestamp means every demo
question needs an explicit range and the standing-task funnel never looks at it at all.
Import time is not a lie about when the events happened — it is a statement about when
the system saw them, which is what the index is for. ``--start`` overrides it.

**Identity — an import is its own camera.** ``clip01``, not ``cam01``. Two sources
sharing an id would interleave in the timeline the moment their ranges overlapped, and
``list_segments`` would have to guess which file covers 00:41:17. A separate id keeps
both queryable and keeps the stitcher honest. It is also why ``list_segments`` raises on
a mixed archive without a ``camera_id`` — that tripwire is now load-bearing rather than
theoretical, and the readers pass an id.

**Cut points — the segment list is read back, never assumed.** With ``-c copy`` ffmpeg
can only cut on a keyframe, so a "60 s" segment is 60 s only if the source's GOP happens
to divide it. Naming files ``start + n*60`` would then drift a filename away from the
footage it holds, which is invariant 2 breaking silently and in the one direction nobody
checks. ffmpeg's ``-segment_list`` reports each part's true offset; we name from that.
"""

from __future__ import annotations

from .importer import (
    ImportError_,
    ImportResult,
    VideoInfo,
    default_camera_id,
    import_video,
    inbox_files,
    probe_video,
    slice_into_archive,
)

__all__ = [
    "ImportError_",
    "ImportResult",
    "VideoInfo",
    "default_camera_id",
    "import_video",
    "inbox_files",
    "probe_video",
    "slice_into_archive",
]
