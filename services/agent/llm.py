"""The ask model, behind a two-implementation seam — SPEC §4.

* :class:`OpenAICompatBackend` — the real path. Nemotron served by NIM on an
  OpenAI-compatible route (``agent.endpoint``), with the tool schemas passed as
  ``tools`` so SPEC §4.2's *tool choice* mechanism is the model's, not ours. The model
  name comes from ``config.require("agent.model")`` — SPEC §10 D3 is open and this
  module does not pick one.
* :class:`StubBackend` — deterministic, stdlib only, no network. ``agent.backend: stub``
  is the shipped default and it is **not** a test mock: there is no NGC key on this box
  (CLAUDE.md machine state), and the ask surface, the groundedness gate and the whole
  escalation arc have to be exercisable end to end against the real index *today*.

Two things about the seam are deliberate.

**The request carries its retrieval context structurally**, not only as prompt text.
The HTTP backend ignores :attr:`LLMRequest.context` and sends the rendered prompt; the
stub reads it and never parses its own prompt back out with a regex. A stub that
re-parses English is a stub that fails differently from the thing it stands in for.

**Purpose is a field.** Two calls per turn — the §4.2 gate and the answer — are
different questions with different budgets, and naming which one is running is what
makes the structured log readable when the demo misbehaves. The HTTP backend passes it
through to the log and nowhere else.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from shared.schema import to_iso

from .settings import AgentSettings
from .telemetry import log_event

__all__ = [
    "Purpose",
    "ContextChunk",
    "ToolCall",
    "LLMRequest",
    "LLMResponse",
    "LLMBackend",
    "Transport",
    "RequestsTransport",
    "StubBackend",
    "OpenAICompatBackend",
    "build_backend",
    "LLMError",
    "LLMTransportError",
    "LLMResponseError",
]

LOGGER = logging.getLogger("services.agent.llm")

_CHAT_COMPLETIONS = "/chat/completions"


class LLMError(RuntimeError):
    """Base class for every failure raised by this module."""


class LLMTransportError(LLMError):
    """The endpoint could not be reached, or returned a non-2xx status."""


class LLMResponseError(LLMError):
    """The endpoint answered, but not in the shape an OpenAI-compatible server owes us."""


class Purpose(str, Enum):
    """Which of the turn's two calls this is. See SPEC §4.2."""

    #: "Can this question be answered from this context alone, yes/no?"
    GROUNDEDNESS = "groundedness"
    #: The provisional answer, with the tools offered.
    ANSWER = "answer"


# --------------------------------------------------------------------------------------
# Values crossing the seam
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextChunk:
    """One reranked chunk, as the model sees it.

    Carries the wall-clock range beside the caption because that range is what the model
    must cite and what ``request_deep_analysis`` re-watches — CLAUDE.md invariant 2 all
    the way to the prompt.
    """

    chunk_id: str
    t_start: datetime
    t_end: datetime
    caption: str
    rank: int = 0

    def render(self) -> str:
        """The prompt line for this chunk. UTC, ``Z``-suffixed, same as every payload."""
        return f"[{self.chunk_id}] {to_iso(self.t_start)} .. {to_iso(self.t_end)}: {self.caption}"


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked for. ``arguments`` is already decoded."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True)
class LLMRequest:
    """One completion request. ``context`` is the reranked top-N, structurally."""

    purpose: Purpose
    system: str
    user: str
    max_tokens: int
    temperature: float = 0.0
    tools: Sequence[Mapping[str, Any]] = ()
    context: Sequence[ContextChunk] = ()
    question: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """One completion, plus the numbers we tune on."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    backend: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMBackend(Protocol):
    """Text in, text (and optional tool calls) out. The only thing the agent knows."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


# --------------------------------------------------------------------------------------
# Transport — the seam that keeps tests off the real endpoint
# --------------------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Minimal HTTP seam, mirroring ``shared/vlm_client.py`` so the two read alike."""

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout: float | None
    ) -> dict[str, Any]: ...


class RequestsTransport:
    """Default transport: a pooled ``requests.Session``.

    ``requests`` is imported lazily so importing this module never depends on it being
    installed — and so the stub path has no dependency at all.
    """

    def __init__(self, session: Any | None = None) -> None:
        if session is None:
            import requests  # noqa: PLC0415 — deferred; see class docstring

            session = requests.Session()
        self._session = session

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout: float | None
    ) -> dict[str, Any]:
        try:
            response = self._session.post(url, json=dict(payload), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — any transport failure is one failure to us
            raise LLMTransportError(f"POST {url} failed: {exc}") from exc
        status = getattr(response, "status_code", 200)
        if not 200 <= status < 300:
            body = getattr(response, "text", "")[:500]
            raise LLMTransportError(f"POST {url} returned HTTP {status}: {body}")
        try:
            body_json = response.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMResponseError(f"POST {url} returned non-JSON body: {exc}") from exc
        if not isinstance(body_json, dict):
            raise LLMResponseError(
                f"POST {url} returned a {type(body_json).__name__}, not an object"
            )
        return body_json


# --------------------------------------------------------------------------------------
# The real path
# --------------------------------------------------------------------------------------


class OpenAICompatBackend:
    """Nemotron over an OpenAI-compatible NIM route (SPEC §4).

    Never exercised on this box — there is no NGC key and ``agent.model`` is null
    (SPEC §10 D3) — so it is written to the documented API and that is stated rather
    than hidden. The tool schemas ride in ``tools``: SPEC §4.2's second escalation
    mechanism is the *model* reaching for ``request_deep_analysis``, and reimplementing
    that choice on our side would be a different mechanism wearing its name.
    """

    def __init__(
        self,
        settings: AgentSettings,
        transport: Transport | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self._s = settings
        self._transport: Transport = transport if transport is not None else RequestsTransport()
        self._url = settings.endpoint.rstrip("/") + _CHAT_COMPLETIONS
        # SPEC §10 D3. ``require`` (via AgentSettings.model) turns "model is null" into a
        # sentence naming the open decision instead of a 404.
        self._model = str(model) if model is not None else settings.model

    @property
    def name(self) -> str:
        return "nim"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            payload["tools"] = [dict(t) for t in request.tools]
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            body = self._transport.post(
                self._url, payload, timeout=self._s.request_timeout
            )
        except LLMError as exc:
            _log_call(
                self.name,
                self._model,
                request,
                wall_time_ms=(time.perf_counter() - started) * 1000.0,
                prompt_tokens=0,
                completion_tokens=0,
                tool_calls=(),
                ok=False,
                error=str(exc),
            )
            raise

        wall_time_ms = (time.perf_counter() - started) * 1000.0
        text, calls = self._parse_message(body)
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        _log_call(
            self.name,
            self._model,
            request,
            wall_time_ms=wall_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=calls,
            ok=True,
            error=None,
        )
        return LLMResponse(
            text=text,
            tool_calls=calls,
            model=self._model,
            backend=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_ms=wall_time_ms,
            raw=body,
        )

    @staticmethod
    def _parse_message(body: Mapping[str, Any]) -> tuple[str, tuple[ToolCall, ...]]:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError(f"response has no choices: {_clip(body)}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise LLMResponseError(f"choice has no message: {_clip(body)}")

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            name = str(function.get("name") or "")
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    # A malformed tool call is a call we cannot honour, but it is also
                    # evidence the model wanted to escalate. Keep the intent, drop the
                    # arguments, and let the agent fall back to the cited range.
                    LOGGER.warning("tool call %s had unparseable arguments; dropped", name)
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(ToolCall(name=name, arguments=arguments, call_id=str(raw.get("id") or "")))

        content = str(message.get("content") or "").strip()
        if not calls:
            inline, content = _extract_inline_tool_call(content)
            calls.extend(inline)
        return content, tuple(calls)


# --------------------------------------------------------------------------------------
# Inline tool calls — normalising a wire format that is not OpenAI's
# --------------------------------------------------------------------------------------

#: `name{key:<|"|>value<|"|>,...}` — how the served model emits a tool call in `content`
#: instead of in `tool_calls`. Observed live from gemma-4-E2B via llama-server with jinja
#: templating on; llama.cpp does not normalise this shape into the OpenAI field, so the
#: raw call reaches us as prose.
#:
#: The delimiter is matched loosely (`<|>`, `<|"|>`, and similar) on purpose: it is a
#: model/template detail with no specification behind it, and pinning the exact spelling
#: means the next template revision silently reinstates the bug. What identifies a call
#: here is the *shape* — a known-looking name, braces, and delimited key/value pairs.
_DELIM = r"<\|(?:[^|>]{0,4}\|)?>"
_INLINE_CALL_RE = re.compile(r"(?P<name>[a-z_][a-z0-9_]*)\s*\{(?P<body>.*?)\}\s*$", re.S | re.I)
_INLINE_ARG_RE = re.compile(
    rf"(?P<key>[a-z_][a-z0-9_]*)\s*:\s*{_DELIM}(?P<value>.*?){_DELIM}", re.S | re.I
)


def _extract_inline_tool_call(content: str) -> tuple[list[ToolCall], str]:
    """Pull a tool call out of message *content*, returning it and the leftover prose.

    Without this the serialized call is rendered to the user as the provisional answer —
    literally ``request_deep_analysis{question:<|>...<|>}`` in the Ask pane, which is the
    surface SPEC §11.2 calls the most important pixel in the build. The escalation still
    happened; it just looked broken.

    Deliberately conservative: only a call that is the WHOLE remaining message is taken,
    and only with the ``<|>``-delimited argument shape. A model that merely mentions a
    tool name mid-sentence keeps its prose.

    The tool's own ``why`` argument, when present, becomes the leftover text — it is the
    model's stated reason for escalating, which is exactly what the pane wants to show
    while the deep job runs.
    """
    text = content.strip()
    if not text or "<|" not in text:
        return [], content
    match = _INLINE_CALL_RE.search(text)
    if not match:
        return [], content
    arguments = {m.group("key"): m.group("value").strip() for m in _INLINE_ARG_RE.finditer(match.group("body"))}
    if not arguments:
        return [], content
    call = ToolCall(name=match.group("name"), arguments=arguments)
    LOGGER.info(
        "normalised an inline tool call the backend did not put in tool_calls: %s", call.name
    )
    # Anything the model wrote before the call is real prose and is kept; otherwise fall
    # back to its own `why`, and never to the serialized call itself.
    leading = text[: match.start()].strip()
    return [call], leading or str(arguments.get("why") or "")


# --------------------------------------------------------------------------------------
# The stub — how the demo runs today
# --------------------------------------------------------------------------------------

# Stub corpus data, NOT tunables. The real backend has no word lists: it is a language
# model answering a yes/no question. These exist only so the stand-in reaches the same
# verdicts on this footage that a model would, deterministically and offline.
#: No apostrophes: "van's" must tokenise to "van" so it matches a caption that says
#: "van". Same shape as the index's tokenizer for the same reason.
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Function words carry no information about whether a caption covers a question.
#: Mirrors the spirit of the index's tokenizer, kept local so the two can diverge
#: without one silently changing the other's behaviour.
_STOPWORDS = frozenset(
    """
    a an the this that these those there here and or but if then than so as at by for
    from in into of on onto to with without about over under between during it its is
    are was were be been being do did does done have has had can could would should
    will shall may might must i you he she they we me him her them us my your his their
    our what when where which who whom why how did any some all no not yes up down out
    off again once more most other same such only own too very s t just now
    """.split()
)

#: The §4.2 fine-visual-detail lexicon. A question containing one of these asks about
#: something a 1 fps caption at 512 px does not record, whatever it says about the scene
#: — which is exactly when the model should reach for ``request_deep_analysis``.
_DETAIL_TERMS = frozenset(
    """
    open closed ajar shut colour color read reads reading text sign signage logo brand
    plate plates licence license number numbers digit digits label letters writing wrote
    face faces expression wearing wore clothing shirt jacket hat mask holding carrying
    gesture gesturing exactly precisely detail details clearly zoom close
    """.split()
)
# "visible" is deliberately NOT in the set above. It reads as a fine-detail word but is
# used generically far more often — "what is visible in the scene?" is precisely the
# question the index SHOULD answer, and treating it as a detail term escalated it, which
# breaks the SPEC §10 D6 pairing the demo depends on. The terms that earn a place here
# name a specific attribute a 1 fps caption cannot have recorded.


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _STOPWORDS and len(w) > 1]


class StubBackend:
    """Deterministic stand-in for the ask model. No network, no model, no NGC key.

    It is crude on purpose and honest about which crude rule it is applying:

    * **Groundedness** is decided by *coverage* — what fraction of the question's
      content words appear in the reranked captions. That is a real signal about
      whether the context mentions what was asked, and it is deliberately **not** the
      same rule the tool choice uses, so the two §4.2 mechanisms can disagree on stage
      the way two real mechanisms would.
    * **Tool choice** is decided by the fine-visual-detail lexicon above. "Was the rear
      door open" asks about something a caption never records, even when the caption
      talks at length about the van.

    Neither rule is a confidence score and neither is tunable retrieval distance —
    SPEC §4.2 is explicit that ANN distance is not a confidence signal, and this stub
    never looks at one.
    """

    _MODEL_TAG = "stub-ask"

    def __init__(self, settings: AgentSettings) -> None:
        self._s = settings

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return self._MODEL_TAG

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if request.purpose is Purpose.GROUNDEDNESS:
            text, calls = self._groundedness(request), ()
        else:
            text, calls = self._answer(request)
        wall_time_ms = (time.perf_counter() - started) * 1000.0
        _log_call(
            self.name,
            self._MODEL_TAG,
            request,
            wall_time_ms=wall_time_ms,
            # A stub has no tokeniser. Reporting words keeps the log field populated and
            # comparable in shape; it is not a token count and is not claimed to be.
            prompt_tokens=len(_words(request.user)),
            completion_tokens=len(_words(text)),
            tool_calls=calls,
            ok=True,
            error=None,
        )
        return LLMResponse(
            text=text,
            tool_calls=calls,
            model=self._MODEL_TAG,
            backend=self.name,
            prompt_tokens=len(_words(request.user)),
            completion_tokens=len(_words(text)),
            wall_time_ms=wall_time_ms,
        )

    # -- the two calls --------------------------------------------------------------

    def coverage(self, question: str, context: Sequence[ContextChunk]) -> float:
        """Fraction of the question's content words present in the context captions."""
        terms = set(_content_words(question))
        if not terms:
            return 0.0
        present: set[str] = set()
        for chunk in context:
            present.update(_words(chunk.caption))
        return len(terms & present) / len(terms)

    def _groundedness(self, request: LLMRequest) -> str:
        question = request.question or request.user
        if not request.context:
            return (
                "NO\nThe retrieval returned no captioned windows for this range, so "
                "there is no context to answer from."
            )
        score = self.coverage(question, request.context)
        threshold = self._s.stub_coverage_threshold
        context_words = {w for c in request.context for w in _words(c.caption)}
        asked = set(_content_words(question))
        missing = sorted(asked - context_words)

        # A question turning on a specific visual attribute is NOT answerable from
        # captions that never mention that attribute, however much of the rest overlaps.
        # Without this, "was the rear door open" scores 60% on captions saying "van",
        # "door" and "the rear of the van" — none of which record open or shut — and the
        # gate fails open on exactly the question SPEC §4.2 exists to escalate. Lexical
        # overlap measures vocabulary, not whether the answer is present.
        missing_detail = sorted((asked & _DETAIL_TERMS) - context_words)
        if missing_detail:
            return (
                f"NO\nThe captions never mention {', '.join(missing_detail)}. "
                f"They describe the scene at {score:.0%} term overlap but do not record "
                f"the specific detail asked about; answering means re-watching."
            )

        if score >= threshold:
            return (
                f"YES\nThe captions state what was asked "
                f"({score:.0%} of the question's terms appear in the retrieved context)."
            )
        detail = ", ".join(missing[:4]) or "the specific detail asked about"
        return (
            f"NO\nThe captions do not record {detail} "
            f"({score:.0%} of the question's terms appear in the retrieved context); "
            f"answering would mean looking at the footage again."
        )

    def _answer(self, request: LLMRequest) -> tuple[str, tuple[ToolCall, ...]]:
        question = request.question or request.user
        detail = sorted(set(_content_words(question)) & _DETAIL_TERMS)

        if not request.context:
            text = (
                "Nothing in the index covers that. No captioned window was retrieved for "
                "the requested range, so there is no caption to answer from."
            )
        else:
            best = request.context[0]
            # Times are cited in UTC and labelled. SPEC §10 D8 is open — the overlay is
            # burned UTC while the UI chrome renders local — so the label is the honest
            # move until that decision lands. Do not silently convert here.
            span = f"{to_iso(best.t_start)} .. {to_iso(best.t_end)}"
            text = f"From the indexed captions, {span} (UTC): {best.caption}"
            if len(request.context) > 1:
                text += f" The next-closest window is {to_iso(request.context[1].t_start)} (UTC)."
            if detail:
                text += (
                    " The captions do not record "
                    + ", ".join(detail)
                    + "; that is a level of visual detail the live path never wrote down."
                )

        calls: tuple[ToolCall, ...] = ()
        if detail and _has_tool(request.tools, "request_deep_analysis"):
            t_start, t_end = _context_span(request.context)
            calls = (
                ToolCall(
                    name="request_deep_analysis",
                    arguments={
                        "t_start": to_iso(t_start) if t_start else None,
                        "t_end": to_iso(t_end) if t_end else None,
                        "question": question,
                        "why": (
                            "asks for fine visual detail ("
                            + ", ".join(detail)
                            + ") that a 1 fps caption does not record"
                        ),
                    },
                ),
            )
        return text, calls


def _context_span(context: Sequence[ContextChunk]) -> tuple[datetime | None, datetime | None]:
    if not context:
        return None, None
    return min(c.t_start for c in context), max(c.t_end for c in context)


def _has_tool(tools: Sequence[Mapping[str, Any]], name: str) -> bool:
    return any((t.get("function") or {}).get("name") == name for t in tools)


# --------------------------------------------------------------------------------------
# Logging and construction
# --------------------------------------------------------------------------------------


def _log_call(
    backend: str,
    model: str,
    request: LLMRequest,
    *,
    wall_time_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: Sequence[ToolCall],
    ok: bool,
    error: str | None,
) -> None:
    """One line per model call — CLAUDE.md: we cannot tune what we cannot see."""
    log_event(
        "agent.llm",
        backend=backend,
        model=model,
        purpose=request.purpose.value,
        max_tokens=request.max_tokens,
        context_chunks=len(request.context),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_time_ms=round(wall_time_ms, 2),
        tool_calls=[c.name for c in tool_calls],
        ok=ok,
        error=error,
    )


def _clip(body: Any, limit: int = 300) -> str:
    return repr(body)[:limit]


def build_backend(
    settings: AgentSettings, transport: Transport | None = None
) -> LLMBackend:
    """Pick an implementation from ``agent.backend``.

    ``stub`` is the shipped default and needs nothing. ``nim`` needs ``agent.model``
    (SPEC §10 D3) and a reachable ``agent.endpoint``; it fails at construction with a
    readable message rather than at the first user turn.
    """
    backend = settings.backend.lower()
    if backend == "stub":
        return StubBackend(settings)
    if backend == "nim":
        return OpenAICompatBackend(settings, transport)
    raise ValueError(f"unknown agent.backend: {settings.backend!r}")
