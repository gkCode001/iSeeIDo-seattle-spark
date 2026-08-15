"""M3 — the ask agent. SPEC §4.

One turn, in order:

1. **Retrieve.** ``search_index`` — embed → ANN k=20 → rerank → top 5, ranges attached.
2. **Gate.** Give the reranked chunks to the model and ask, *before answering*: can this
   question be answered from this context alone, yes/no? (SPEC §4.2, mechanism 1.)
3. **Answer.** The provisional answer, with the tools offered so the model can reach for
   ``request_deep_analysis`` itself. (SPEC §4.2, mechanism 2.)
4. **Escalate, without blocking.** If either mechanism fired, dispatch to M4 and return
   the ``job_id``. The turn ends here (CLAUDE.md invariant 4); the refinement arrives
   over the WebSocket and is *appended*, never substituted.

Both mechanisms are on, and both are recorded on :class:`Escalation` so the decision can
be printed. SPEC §4.2 says to print it; SPEC §11.2 calls the resulting badge the most
important pixel in the build. ``ChatTurn.grounded`` persists the verdict so a reload
re-renders the same badge rather than a recomputed one.

This module is pure logic: no HTTP, no sockets, no threads. The server is a thin shell
around :meth:`AskAgent.ask`, which is what makes a later FastAPI swap a new file rather
than a rewrite.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from services.index import SearchHit
from shared.schema import ChatTurn, DeepJob, to_iso, utcnow

from .history import ChatLog
from .llm import (
    LLMBackend,
    LLMError,
    LLMRequest,
    LLMResponse,
    Purpose,
    ToolCall,
)
from .settings import AgentSettings
from .telemetry import log_event, timed
from .tools import DEEP_TOOL, TOOL_SCHEMAS, ToolInvocation, Toolbox

__all__ = ["AskAgent", "AskResult", "Escalation", "GateVerdict"]

#: Trigger names, as they appear in the payload and in the log. SPEC §4.2's two
#: mechanisms, named so the UI can say *which* one fired rather than only *that* one did.
TRIGGER_GATE = "groundedness_gate"
TRIGGER_TOOL = "tool_choice"


@dataclass(frozen=True)
class GateVerdict:
    """The §4.2 groundedness gate's answer. One extra model call, one yes/no.

    ``grounded`` is None only when the gate did not run — disabled in config, or the
    model call failed. None is *not* "probably fine": the UI renders an unknown verdict
    differently from a negative one, because a gate that silently stopped running looks
    exactly like a system that is always confident.
    """

    grounded: bool | None
    reason: str = ""
    ran: bool = True
    error: str | None = None

    @property
    def badge(self) -> str:
        """The §11.2 badge text, decided here so two surfaces cannot word it differently."""
        if self.grounded is True:
            return "answered from index"
        if self.grounded is False:
            return "not answerable from index"
        return "groundedness unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "reason": self.reason,
            "ran": self.ran,
            "error": self.error,
            "badge": self.badge,
        }


@dataclass
class Escalation:
    """Why this turn did or did not reach for the deep worker — the printable record."""

    escalated: bool
    triggers: tuple[str, ...]
    gate: GateVerdict
    tool_requested: bool
    why: str = ""
    t_start: datetime | None = None
    t_end: datetime | None = None
    timeout_seconds: float = 0.0
    #: The turn WOULD have escalated, but the recent window is the thing that came up
    #: short, so the wider search is being offered instead. `escalated` is False and no
    #: job exists — the record still says the gate fired, which is what the badge reads.
    deferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalated": self.escalated,
            "deferred": self.deferred,
            "triggers": list(self.triggers),
            "gate": self.gate.to_dict(),
            "tool_requested": self.tool_requested,
            "why": self.why,
            "t_start": to_iso(self.t_start) if self.t_start else None,
            "t_end": to_iso(self.t_end) if self.t_end else None,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class AskResult:
    """One completed turn. ``job`` is queued, not finished — that is the whole design."""

    turn: ChatTurn
    escalation: Escalation
    hits: list[SearchHit] = field(default_factory=list)
    job: DeepJob | None = None
    dedupe_of: str | None = None
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)
    #: Set when the recent window could not answer and the user is being OFFERED a wider
    #: search rather than given one. No job is queued while this is pending: the deep
    #: path costs tens of seconds of the single VLM slot, and spending that on a guess
    #: about what the user meant is how "I don't know" becomes a 90 s wait.
    widen_offer: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """``POST /api/ask``'s body: ``ChatTurn.to_dict()`` plus the optional extras.

        The UI reads ``dedupe_of`` and ``job`` (ui/static/data.js); ``escalation``,
        ``retrieval`` and ``tool_calls`` are additive and ignored by a client that does
        not know them. The ChatTurn keys stay exactly as ``shared/schema.py`` writes
        them — this payload extends the contract, it does not reshape it.
        """
        payload = self.turn.to_dict()
        payload["dedupe_of"] = self.dedupe_of
        payload["job"] = self.job.to_dict() if self.job else None
        payload["escalation"] = self.escalation.to_dict()
        payload["retrieval"] = dict(self.retrieval)
        payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        payload["cited"] = [hit.to_dict() for hit in self.hits]
        payload["widen_offer"] = dict(self.widen_offer) if self.widen_offer else None
        return payload


class AskAgent:
    """The ask surface. Construct with :func:`services.agent.build_agent` in production."""

    def __init__(
        self,
        backend: LLMBackend,
        tools: Toolbox,
        chat_log: ChatLog,
        settings: AgentSettings,
    ) -> None:
        self._llm = backend
        self._tools = tools
        self._log = chat_log
        self._s = settings

    # -- the turn --------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        turn_id: str | None = None,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        widen: bool = False,
    ) -> AskResult:
        """Answer a question provisionally. **Never blocks on deep analysis.**

        Returns as soon as the retrieval, the gate and the answer are done — typically a
        couple of seconds. If the turn escalated, ``result.job`` is a QUEUED job whose
        refinement will arrive later over the WebSocket.
        """
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"

        # The surface is asked about NOW, so an ask covers the recent window unless the
        # caller gave an explicit range or has already agreed to look further back.
        explicit_range = t_from is not None or t_to is not None
        lookback = (
            self._s.search_extended_lookback_seconds
            if widen
            else self._s.search_lookback_seconds
        )

        with timed("agent.ask", turn_id=turn_id, question=question, widened=widen) as span:
            hits = self._tools.search_index(question, t_from, t_to, lookback_seconds=lookback)
            context = self._tools.as_context(hits)

            gate = self._gate(question, context)
            answer, calls = self._answer(question, context)

            deep_call = next((c for c in calls if c.name == DEEP_TOOL), None)
            other_calls = [c for c in calls if c.name != DEEP_TOOL]
            invocations = [
                self._tools.dispatch(call.name, call.arguments, turn_id=turn_id, hits=hits)
                for call in other_calls
            ]

            triggers: list[str] = []
            if gate.grounded is False:
                triggers.append(TRIGGER_GATE)
            if deep_call is not None:
                triggers.append(TRIGGER_TOOL)

            escalation = Escalation(
                escalated=bool(triggers),
                triggers=tuple(triggers),
                gate=gate,
                tool_requested=deep_call is not None,
                why=_why(gate, deep_call),
                timeout_seconds=self._s.deep_timeout_seconds,
            )

            # Offer the wider search instead of taking it. Only when the recent window
            # is what failed: an explicit range is the caller's decision, and a question
            # already widened has nowhere further to go — that one escalates for real.
            widen_offer: dict[str, Any] | None = None
            if (
                escalation.escalated
                and self._s.confirm_before_widening
                and not widen
                and not explicit_range
            ):
                widen_offer = {
                    "reason": escalation.why,
                    "searched_seconds": int(self._s.search_lookback_seconds),
                    "offer_seconds": int(self._s.search_extended_lookback_seconds),
                    "question": question,
                }
                escalation.escalated = False
                escalation.deferred = True

            job: DeepJob | None = None
            dedupe_of: str | None = None
            if escalation.escalated:
                t_start, t_end = self._tools.deep_range(
                    hits, deep_call.arguments if deep_call else None
                )
                escalation.t_start, escalation.t_end = t_start, t_end
                job, dedupe_of = self._tools.request_deep_analysis(
                    t_start, t_end, question, turn_id=turn_id
                )

            turn = ChatTurn(
                turn_id=turn_id,
                ts=utcnow(),
                question=question,
                provisional_answer=answer,
                # The persisted §4.2 verdict — the badge is re-rendered from this on
                # reload, never recomputed (SPEC §11.2 / §11.4).
                grounded=gate.grounded,
                cited_chunk_ids=[hit.chunk_id for hit in hits],
                job_id=job.job_id if job else dedupe_of,
                latency_s=round(span.elapsed_s, 3),
            )
            self._log.append_turn(turn)
            if job is not None and dedupe_of is None:
                # Persist the job with the turn (SPEC §11.4): the refinement lands after
                # the turn ends and a reload must not lose it.
                self._log.append_job(job)

            span.fields.update(
                grounded=gate.grounded,
                escalated=escalation.escalated,
                triggers=triggers,
                job_id=turn.job_id,
                dedupe_of=dedupe_of,
                hits=len(hits),
            )

        log_event(
            "agent.turn",
            turn_id=turn_id,
            grounded=gate.grounded,
            badge=gate.badge,
            escalated=escalation.escalated,
            triggers=triggers,
            job_id=turn.job_id,
            dedupe_of=dedupe_of,
            latency_s=turn.latency_s,
            cited=turn.cited_chunk_ids,
        )
        return AskResult(
            turn=turn,
            escalation=escalation,
            hits=list(hits),
            job=job,
            dedupe_of=dedupe_of,
            tool_calls=invocations,
            retrieval=self._retrieval_meta(hits),
            widen_offer=widen_offer,
        )

    # -- the two model calls ---------------------------------------------------------

    def _gate(self, question: str, context: Sequence[Any]) -> GateVerdict:
        """SPEC §4.2 mechanism 1. One extra call, before answering, yes or no.

        Retrieval distance is **not** consulted, here or anywhere: ANN always returns a
        plausible top-k even when the answer was never indexed, so a threshold on it
        would be a confidence number with nothing behind it.
        """
        if not self._s.groundedness_gate:
            return GateVerdict(grounded=None, reason="gate disabled in config", ran=False)

        request = LLMRequest(
            purpose=Purpose.GROUNDEDNESS,
            system=self._s.groundedness_prompt,
            user=_gate_prompt(question, context),
            # A yes/no plus one sentence. Same reasoning as CLAUDE.md invariant 6:
            # output tokens are the latency dial, and this call sits on the user's turn.
            max_tokens=min(self._s.max_tokens, 96),
            temperature=self._s.temperature,
            context=tuple(context),
            question=question,
        )
        try:
            response = self._llm.complete(request)
        except LLMError as exc:
            # A gate that cannot run reports unknown. It must never report "grounded" by
            # default — that would turn an outage into silent overconfidence.
            log_event("agent.gate.failed", error=f"{type(exc).__name__}: {exc}")
            return GateVerdict(
                grounded=None,
                reason="the groundedness gate could not run",
                ran=False,
                error=str(exc),
            )
        grounded, reason = _parse_verdict(response.text)
        log_event(
            "agent.gate",
            grounded=grounded,
            reason=reason,
            context_chunks=len(context),
            wall_time_ms=round(response.wall_time_ms, 2),
        )
        return GateVerdict(grounded=grounded, reason=reason)

    def _answer(
        self, question: str, context: Sequence[Any]
    ) -> tuple[str, tuple[ToolCall, ...]]:
        """The provisional answer, with the tools offered (SPEC §4.2 mechanism 2)."""
        request = LLMRequest(
            purpose=Purpose.ANSWER,
            system=self._s.answer_prompt,
            user=_answer_prompt(question, context),
            max_tokens=self._s.max_tokens,
            temperature=self._s.temperature,
            tools=TOOL_SCHEMAS,
            context=tuple(context),
            question=question,
        )
        try:
            response: LLMResponse = self._llm.complete(request)
        except LLMError as exc:
            log_event("agent.answer.failed", error=f"{type(exc).__name__}: {exc}")
            return (
                f"The ask model is unreachable ({exc}). The retrieved windows are listed "
                f"below; the index itself is fine.",
                (),
            )
        return response.text, response.tool_calls

    # -- helpers ---------------------------------------------------------------------

    def _retrieval_meta(self, hits: Sequence[SearchHit]) -> dict[str, Any]:
        """The §11.2 retrieval line, as data: "⌕ searched index · 20 → 5 chunks · range"."""
        return {
            "returned": len(hits),
            "t_start": to_iso(min(h.record.t_start for h in hits)) if hits else None,
            "t_end": to_iso(max(h.record.t_end for h in hits)) if hits else None,
        }


# --------------------------------------------------------------------------------------
# Prompt rendering and verdict parsing
# --------------------------------------------------------------------------------------


def _render_context(context: Sequence[Any]) -> str:
    if not context:
        return "(no captioned windows were retrieved for this range)"
    return "\n".join(chunk.render() for chunk in context)


def _gate_prompt(question: str, context: Sequence[Any]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT ({len(context)} captions):\n{_render_context(context)}\n\n"
        "Can the question be answered from this context alone? YES or NO, then one "
        "sentence."
    )


def _answer_prompt(question: str, context: Sequence[Any]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT ({len(context)} captions, wall clock is UTC):\n"
        f"{_render_context(context)}\n\n"
        "Answer from this context. If it does not record what was asked, say so and "
        "call request_deep_analysis on the relevant range."
    )


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    """Read YES/NO out of the gate's completion.

    Tolerant of a model that writes "Yes." or leads with a courtesy line, and honest
    when it writes neither: an unparseable verdict is *unknown*, not grounded. A gate
    that fails open is not a gate.
    """
    stripped = text.strip()
    reason = ""
    verdict: bool | None = None
    for line in stripped.splitlines():
        token = line.strip().lstrip("*#- ").lower()
        if not token:
            continue
        if verdict is None and token.startswith(("yes", "no")):
            verdict = token.startswith("yes")
            remainder = line.strip()
            # "NO — the captions do not..." keeps its explanation on the same line.
            for separator in ("—", "-", ":", "."):
                head, sep, tail = remainder.partition(separator)
                if sep and len(head.strip()) <= 4 and tail.strip():
                    reason = tail.strip()
                    break
            continue
        if verdict is not None and not reason:
            reason = line.strip()
            break
    if verdict is None:
        return None, f"unparseable groundedness verdict: {stripped[:120]!r}"
    return verdict, reason or stripped


def _why(gate: GateVerdict, deep_call: ToolCall | None) -> str:
    """One line, for the escalation card. Both mechanisms get to speak."""
    parts: list[str] = []
    if gate.grounded is False and gate.reason:
        parts.append(f"gate: {gate.reason}")
    if deep_call is not None:
        detail = str(deep_call.arguments.get("why") or "").strip()
        parts.append(f"tool choice: {detail}" if detail else "the model called request_deep_analysis")
    if not parts and gate.grounded is True:
        parts.append("answered from the index; no re-watch needed")
    return " · ".join(parts)
