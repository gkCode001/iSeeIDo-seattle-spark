"""The SPEC §4.1 tools, and the schemas the model is shown.

| Tool | Purpose |
|---|---|
| ``search_index(query, t_from?, t_to?)`` | M2 retrieval |
| ``request_deep_analysis(t_start, t_end, question)`` | dispatch to M4, returns a ``job_id`` immediately |
| ``read_action_log(t_from, t_to)`` | answer "why did you alert at 21:11?" |
| MCP actions | ``save_clip``, ``raise_alert``, ``file_ticket`` |

Two rules this module exists to enforce.

**Every action goes through ``services/mcp``** — cooldown, footage-range dedupe,
append-only log (CLAUDE.md invariant 5). There is no path from here to an effect that
skips :class:`~services.mcp.ActionServer`, and a suppressed action is reported as a
normal outcome rather than raised: the brakes refusing is the system working.

**The tool descriptions are load-bearing.** SPEC §4.2's second escalation mechanism is
the model reaching for ``request_deep_analysis`` on fine visual detail. That behaviour
lives in the ``description`` strings below, which is why they read like instructions
rather than labels — they are the mechanism, not documentation of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from services.index import IndexStore, SearchHit
from services.mcp import ActionResult, ActionServer
from shared.schema import ActionKind, ActionLogEntry, DeepJob, from_iso, to_iso, utcnow

from .deep import JobRegistry
from .llm import ContextChunk
from .settings import AgentSettings
from .telemetry import log_event, timed

__all__ = [
    "TOOL_SCHEMAS",
    "ToolInvocation",
    "Toolbox",
    "ACTION_TOOLS",
    "DEEP_TOOL",
    "SEARCH_TOOL",
    "ACTION_LOG_TOOL",
]

SEARCH_TOOL = "search_index"
DEEP_TOOL = "request_deep_analysis"
ACTION_LOG_TOOL = "read_action_log"
ACTION_TOOLS: dict[str, ActionKind] = {
    "save_clip": ActionKind.SAVE_CLIP,
    "raise_alert": ActionKind.RAISE_ALERT,
    "file_ticket": ActionKind.FILE_TICKET,
}

_ISO_HINT = "ISO 8601 UTC with a Z suffix, e.g. 2026-08-14T21:11:07Z"


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


#: OpenAI-shaped function schemas, passed straight through as ``tools``.
TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _schema(
        SEARCH_TOOL,
        "Search the caption index for windows of footage matching a description. "
        "Returns the top reranked chunks, each with its wall-clock range. Captions were "
        "written live at 1 fps from downscaled frames: they record what was present and "
        "what it was doing, never fine visual detail.",
        {
            "query": {"type": "string", "description": "What to look for, in plain language."},
            "t_from": {"type": "string", "description": f"Optional start of the search window. {_ISO_HINT}"},
            "t_to": {"type": "string", "description": f"Optional end of the search window. {_ISO_HINT}"},
        },
        ["query"],
    ),
    _schema(
        DEEP_TOOL,
        "Re-watch a range of footage at native resolution and 4 fps to answer a question "
        "the captions cannot. USE THIS whenever the question turns on fine visual detail "
        "— whether a door was open, what a sign or plate reads, a colour, what someone "
        "was wearing or holding, an exact count — or whenever the retrieved captions "
        "simply do not mention what was asked. It returns a job_id immediately and the "
        "refined answer arrives later; it never blocks your reply, so asking for it costs "
        "the user nothing but gains a real look at the pixels.",
        {
            "t_start": {"type": "string", "description": f"Start of the footage range to re-watch. {_ISO_HINT}"},
            "t_end": {"type": "string", "description": f"End of the footage range to re-watch. {_ISO_HINT}"},
            "question": {
                "type": "string",
                "description": "The question to answer from the footage, self-contained.",
            },
            "why": {
                "type": "string",
                "description": "One line on why the captions are insufficient. Printed in the UI.",
            },
        },
        ["t_start", "t_end", "question"],
    ),
    _schema(
        ACTION_LOG_TOOL,
        "Read the append-only action log for a time window — every alert, ticket and "
        "saved clip, with the reason it fired and any later verification or retraction. "
        "This is how 'why did you alert at 21:11?' is answered from the record rather "
        "than from memory.",
        {
            "t_from": {"type": "string", "description": f"Start of the window. {_ISO_HINT}"},
            "t_to": {"type": "string", "description": f"End of the window. {_ISO_HINT}"},
        },
        ["t_from", "t_to"],
    ),
    _schema(
        "save_clip",
        "Save an evidence clip of a footage range. Low stakes: no human is notified. "
        "Subject to the cooldown and time-range dedupe brakes, which may refuse it.",
        {
            "t_start": {"type": "string", "description": f"Start of the footage range. {_ISO_HINT}"},
            "t_end": {"type": "string", "description": f"End of the footage range. {_ISO_HINT}"},
            "reason": {"type": "string", "description": "Why this range is worth keeping."},
        },
        ["t_start", "t_end"],
    ),
    _schema(
        "raise_alert",
        "Raise an alert about a footage range. THIS REACHES A HUMAN and cannot be "
        "un-fired; it is written unverified and amended later. Only when the user asks "
        "for it or the situation plainly warrants it.",
        {
            "t_start": {"type": "string", "description": f"Start of the footage range. {_ISO_HINT}"},
            "t_end": {"type": "string", "description": f"End of the footage range. {_ISO_HINT}"},
            "reason": {"type": "string", "description": "What the human needs to know."},
        },
        ["t_start", "t_end"],
    ),
    _schema(
        "file_ticket",
        "File a ticket about a footage range. THIS REACHES A HUMAN and cannot be "
        "un-fired; it is written unverified and amended later.",
        {
            "t_start": {"type": "string", "description": f"Start of the footage range. {_ISO_HINT}"},
            "t_end": {"type": "string", "description": f"End of the footage range. {_ISO_HINT}"},
            "reason": {"type": "string", "description": "What the ticket is for."},
        },
        ["t_start", "t_end"],
    ),
)


@dataclass
class ToolInvocation:
    """One tool call and what came of it — rendered in the UI, logged, never hidden."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    detail: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "detail": self.detail,
            "result": self.result,
        }


class Toolbox:
    """The four SPEC §4.1 tools, bound to the real M2, M4 and MCP.

    Every dependency is injected. The agent above holds no reference to an index or an
    action server, and a test can hand this class fakes without a server socket, a
    Milvus, or an M4 that has not been written yet.
    """

    def __init__(
        self,
        index: IndexStore,
        actions: ActionServer,
        jobs: JobRegistry,
        settings: AgentSettings,
    ) -> None:
        self._index = index
        self._actions = actions
        self._jobs = jobs
        self._s = settings

    # -- search_index ---------------------------------------------------------------

    def search_index(
        self,
        query: str,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        *,
        lookback_seconds: float | None = None,
    ) -> list[SearchHit]:
        """SPEC §3.4 retrieval: embed → ANN k=20 → rerank → top 5, ranges attached.

        With no range given, the search covers ``agent.search.default_lookback_seconds``
        ending now — an unbounded scan is not wrong, but it lets a week-old caption
        outrank this afternoon's on a lexical tie, and the ask surface is asked about
        today.
        """
        if t_from is None and t_to is None:
            window = (
                lookback_seconds
                if lookback_seconds is not None
                else self._s.search_lookback_seconds
            )
            t_to = utcnow()
            t_from = t_to - timedelta(seconds=window)
        with timed("agent.search", query=query) as span:
            hits = self._index.search(query, t_from, t_to)
            span.fields["hits"] = len(hits)
            span.fields["t_from"] = to_iso(t_from) if t_from else None
            span.fields["t_to"] = to_iso(t_to) if t_to else None
        return hits

    @staticmethod
    def as_context(hits: Sequence[SearchHit]) -> tuple[ContextChunk, ...]:
        """Reranked hits as the model sees them — caption plus wall-clock range."""
        return tuple(
            ContextChunk(
                chunk_id=hit.chunk_id,
                t_start=hit.record.t_start,
                t_end=hit.record.t_end,
                caption=hit.caption,
                rank=hit.rank,
            )
            for hit in hits
        )

    # -- request_deep_analysis ------------------------------------------------------

    def request_deep_analysis(
        self,
        t_start: datetime,
        t_end: datetime,
        question: str,
        *,
        turn_id: str,
    ) -> tuple[DeepJob, str | None]:
        """Dispatch to M4 and return ``(job, dedupe_of)`` immediately.

        Never blocks (CLAUDE.md invariant 4). The registry owns the in-flight cap, the
        dedupe and the timeout; this method owns nothing but the call.
        """
        return self._jobs.request(t_start, t_end, question, turn_id=turn_id)

    def deep_range(
        self,
        hits: Sequence[SearchHit],
        arguments: Mapping[str, Any] | None = None,
    ) -> tuple[datetime, datetime]:
        """Decide which footage the worker re-watches.

        The model's own ``t_start``/``t_end`` win when it gave usable ones — that is the
        point of letting it choose. Otherwise the range spans the cited chunks, padded by
        ``agent.deep.range_pad_seconds`` because a 5 s window that clipped the event at
        its boundary is precisely the case being escalated, and clamped to
        ``agent.deep.max_range_seconds`` because 4 fps at native resolution is the SPEC §5
        latency warning made real.

        With no hits at all there is nothing to cite, so the most recent
        ``agent.deep.fallback_window_seconds`` is used and that fact is logged.
        """
        proposed = _range_from_arguments(arguments)
        if proposed is not None:
            t_start, t_end = proposed
        elif hits:
            t_start = min(hit.record.t_start for hit in hits)
            t_end = max(hit.record.t_end for hit in hits)
            pad = timedelta(seconds=self._s.deep_range_pad_seconds)
            t_start, t_end = t_start - pad, t_end + pad
        else:
            t_end = utcnow()
            t_start = t_end - timedelta(seconds=self._s.deep_fallback_window_seconds)
            log_event("agent.deep.range_fallback", seconds=self._s.deep_fallback_window_seconds)

        span = (t_end - t_start).total_seconds()
        if span > self._s.deep_max_range_seconds:
            # Keep the *end*: the escalating question is nearly always about how
            # something finished, and the tail is where the answer lives.
            t_start = t_end - timedelta(seconds=self._s.deep_max_range_seconds)
            log_event(
                "agent.deep.range_clamped",
                requested_seconds=round(span, 2),
                max_seconds=self._s.deep_max_range_seconds,
            )
        return t_start, t_end

    # -- read_action_log ------------------------------------------------------------

    def read_action_log(self, t_from: datetime, t_to: datetime) -> list[ActionLogEntry]:
        """SPEC §4.1. The same rows the Timeline pane renders — one source, no drift."""
        return self._actions.read_action_log(t_from, t_to)

    # -- MCP actions ----------------------------------------------------------------

    def fire_action(
        self,
        kind: ActionKind,
        t_start: datetime,
        t_end: datetime,
        *,
        reason: str = "",
        job_id: str | None = None,
    ) -> ActionResult:
        """Fire through the action server. **The only path** — CLAUDE.md invariant 5.

        ``task_id`` stays None: this action is M3 acting on a user's behalf, not M5
        acting on a standing task, and the log distinguishes them (``shared/schema.py``).
        """
        return self._actions.fire(
            kind, t_start, t_end, reason=reason, job_id=job_id
        )

    # -- dispatch -------------------------------------------------------------------

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        turn_id: str,
        hits: Sequence[SearchHit],
    ) -> ToolInvocation:
        """Execute one model-requested tool call and return a legible record of it.

        ``request_deep_analysis`` is handled by the caller (it changes the shape of the
        turn), so it is refused here rather than silently double-dispatched.
        """
        arguments = dict(arguments)
        if name == DEEP_TOOL:
            raise ValueError(f"{DEEP_TOOL} is handled by the agent, not by dispatch()")

        if name == SEARCH_TOOL:
            found = self.search_index(
                str(arguments.get("query", "")),
                _parse_iso(arguments.get("t_from")),
                _parse_iso(arguments.get("t_to")),
            )
            return ToolInvocation(
                name=name,
                arguments=arguments,
                detail=f"{len(found)} chunks",
                result={"hits": [hit.to_dict() for hit in found]},
            )

        if name == ACTION_LOG_TOOL:
            t_from = _parse_iso(arguments.get("t_from"))
            t_to = _parse_iso(arguments.get("t_to"))
            if t_from is None or t_to is None:
                t_to = t_to or utcnow()
                t_from = t_from or t_to - timedelta(seconds=self._s.search_lookback_seconds)
            rows = self.read_action_log(t_from, t_to)
            return ToolInvocation(
                name=name,
                arguments=arguments,
                detail=f"{len(rows)} entries",
                result={"entries": [row.to_dict() for row in rows]},
            )

        if name in ACTION_TOOLS:
            span = _range_from_arguments(arguments) or _span_of(hits)
            if span is None:
                return ToolInvocation(
                    name=name,
                    arguments=arguments,
                    ok=False,
                    detail="no footage range: an action names the footage it is about",
                )
            result = self.fire_action(
                ACTION_TOOLS[name], span[0], span[1], reason=str(arguments.get("reason", ""))
            )
            return ToolInvocation(
                name=name,
                arguments=arguments,
                # A brake refusing is a normal outcome, not a failure of the call.
                ok=True,
                detail=result.detail,
                result=result.to_dict(),
            )

        return ToolInvocation(
            name=name, arguments=arguments, ok=False, detail=f"unknown tool {name!r}"
        )


# --------------------------------------------------------------------------------------
# Argument parsing — a model writes strings, and some of them are wrong
# --------------------------------------------------------------------------------------


def _parse_iso(value: Any) -> datetime | None:
    """Parse a model-supplied timestamp, or None. Never raises into a turn."""
    if not value or not isinstance(value, str):
        return None
    try:
        return from_iso(value)
    except (ValueError, TypeError):
        log_event("agent.tool.bad_timestamp", value=value)
        return None


def _range_from_arguments(
    arguments: Mapping[str, Any] | None,
) -> tuple[datetime, datetime] | None:
    """``(t_start, t_end)`` from tool arguments, if both parse and are ordered."""
    if not arguments:
        return None
    t_start = _parse_iso(arguments.get("t_start"))
    t_end = _parse_iso(arguments.get("t_end"))
    if t_start is None or t_end is None or t_end <= t_start:
        return None
    return t_start, t_end


def _span_of(hits: Sequence[SearchHit]) -> tuple[datetime, datetime] | None:
    if not hits:
        return None
    return (
        min(hit.record.t_start for hit in hits),
        max(hit.record.t_end for hit in hits),
    )
