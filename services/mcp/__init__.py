"""M-MCP — the action server. The brakes (SPEC §6.4, CLAUDE.md invariant 5).

Every action the system takes in the outside world goes through :class:`ActionServer`.
There is no other path, and none of the three brakes — cooldown, footage-range dedupe,
append-only log — can be switched off by a caller. Actions cannot be un-fired.

Typical use from M5::

    server = ActionServer()
    result = server.raise_alert(chunk.t_start, chunk.t_end, task=task, reason=caption)
    if result.fired and result.awaits_verification:
        job_id = queue_deep_verify(...)          # SPEC §6.3 stage 3, non-blocking
        ...
        server.verify(result.entry_id, reason=verdict, clip_path=clip)   # or retract()

and from M3 (SPEC §4.1)::

    rows = server.read_action_log(t_from, t_to)      # raw, for "why did you alert?"
    folded = server.resolved_log(t_from, t_to)       # one row per action, for status
"""

from services.mcp.brakes import (
    Brake,
    BrakeDecision,
    check_brakes,
    cooldown_blocker,
    dedupe_blocker,
    originating_entries,
    ranges_collide,
)
from services.mcp.clips import (
    ClipCutter,
    ClipPlan,
    FfmpegClipCutter,
    NullClipCutter,
    SegmentResolver,
    timecode_segment_resolver,
    SegmentSlice,
    build_clip_plan,
    clip_path_for,
)
from services.mcp.log import ActionLog, LogCorruptionError, ResolvedAction, resolve_all
from services.mcp.server import (
    ActionResult,
    ActionServer,
    ffmpeg_cutter_from_config,
    read_action_log,
)

__all__ = [
    "ActionServer",
    "ActionResult",
    "read_action_log",
    "ffmpeg_cutter_from_config",
    "ActionLog",
    "ResolvedAction",
    "resolve_all",
    "LogCorruptionError",
    "Brake",
    "BrakeDecision",
    "check_brakes",
    "cooldown_blocker",
    "dedupe_blocker",
    "originating_entries",
    "ranges_collide",
    "SegmentSlice",
    "SegmentResolver",
    "timecode_segment_resolver",
    "ClipPlan",
    "ClipCutter",
    "NullClipCutter",
    "FfmpegClipCutter",
    "build_clip_plan",
    "clip_path_for",
]
