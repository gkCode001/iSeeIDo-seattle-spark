"""The one VLM client. Two request profiles against a single process.

CLAUDE.md invariant 1: there is **one VLM process, ever**. Memory is 128 GB unified with
no separate VRAM, so a second model instance does not run slowly — it kills the box.
Every service reaches the VLM through this module (and ``shared/queue.py`` in front of
it). Nothing else opens a connection to the inference endpoint.

CLAUDE.md invariant 6 / SPEC §2.4 and §5 — the two profiles, and why they are not
parameters:

===========  ==================  ============  ==================================
profile      ``enable_reasoning``  ``max_tokens``  used by
===========  ==================  ============  ==================================
``live``     false               80            M1 ingest captioning
``deep``     true                ~600          M4 deep worker, M5 stage-3 verify
===========  ==================  ============  ==================================

Decode is ~95% of chunk latency and is bandwidth-bound (~17 tok/s single stream on an
8B bf16 model). **Output token count is the primary dial.** A caller may ask for *fewer*
tokens than its profile allows; asking for more raises ``ProfileViolation`` rather than
quietly turning a 2 s caption into a 60 s one. There is no ``enable_reasoning``
parameter at all — reasoning is a property of the profile, and a CoT caption on the live
path is the single fastest way to lose real-time.

Two shapes worth understanding before editing:

* **The caption API takes a list of chunks** even though we always pass one
  (CLAUDE.md invariant 9). One camera means one request in flight, so the list is
  serviced sequentially — the batch dimension exists so that 40 cameras is a config
  change rather than a refactor.
* **The transport is injected.** ``VLMClient(transport=...)`` is how tests mock the
  endpoint; CLAUDE.md forbids tests touching the real one, because it contends with
  ingest. The default transport is only constructed when no transport is given, so
  importing this module never requires ``requests``.

The model name is UNRESOLVED — SPEC §10 D1, decided by the block-0 caption benchmark.
It is fetched with ``config.require`` so an unset model fails with a readable message
instead of a mystery 404 from a server that was never told what to serve. Passing
``model=`` explicitly is supported for exactly one reason: ``make bench`` compares
variants to *settle* D1.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from shared import config

__all__ = [
    "Profile",
    "ProfileSpec",
    "VLMChunk",
    "VLMResult",
    "Transport",
    "RequestsTransport",
    "VLMClient",
    "VLMError",
    "VLMTransportError",
    "VLMResponseError",
    "ProfileViolation",
    "encode_frame",
]

LOGGER = logging.getLogger("shared.vlm_client")

# Not yet present in config/settings.yaml. Looked up with a default of None so the
# absence is not fatal, but a hung request against a wedged vLLM would block the single
# in-flight slot forever — this key should be added. See the module report.
_TIMEOUT_SETTING = "vlm.request_timeout_seconds"

_CHAT_COMPLETIONS = "/chat/completions"

# vLLM surfaces a separated reasoning trace as ``reasoning_content`` when a reasoning
# parser is configured, and inline ``<think>...</think>`` when it is not. Which one we
# get depends on the variant chosen in D1, so handle both rather than betting.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class VLMError(RuntimeError):
    """Base class for every failure raised by this module."""


class VLMTransportError(VLMError):
    """The endpoint could not be reached, or returned a non-2xx status."""


class VLMResponseError(VLMError):
    """The endpoint answered, but not in the shape an OpenAI-compatible server owes us."""


class ProfileViolation(ValueError):
    """A caller tried to exceed its profile's budget. See CLAUDE.md invariant 6."""


# --------------------------------------------------------------------------------------
# Profiles — SPEC §2.4 (live) and §5 (deep)
# --------------------------------------------------------------------------------------


class Profile(str, Enum):
    """The two request profiles. There is no third, and no runtime-computed profile."""

    LIVE = "live"
    DEEP = "deep"


@dataclass(frozen=True)
class ProfileSpec:
    """Resolved settings for one profile, read from ``vlm.profiles.<name>``.

    ``sample_fps`` and ``native_resolution`` are decode-side hints rather than request
    fields — M4 reads them from here so that the deep path's "4 fps, native resolution"
    lives in one place instead of being re-derived per caller (SPEC §5).
    """

    name: str
    enable_reasoning: bool
    max_tokens: int
    temperature: float
    sample_fps: float | None = None
    native_resolution: bool | None = None
    request_timeout_seconds: float | None = None

    @classmethod
    def from_config(cls, profile: Profile) -> ProfileSpec:
        base = f"vlm.profiles.{profile.value}"
        sample_fps = config.get(f"{base}.sample_fps", None)
        native = config.get(f"{base}.native_resolution", None)
        # Per-profile timeout, falling back to the shared one. The two profiles differ by
        # an order of magnitude in expected duration, so a single value either strangles
        # the deep path or lets a wedged live call hold the only slot for minutes.
        timeout = config.get(f"{base}.request_timeout_seconds", None)
        if timeout is None:
            timeout = config.get(_TIMEOUT_SETTING, None)
        return cls(
            name=profile.value,
            enable_reasoning=bool(config.get(f"{base}.enable_reasoning")),
            max_tokens=int(config.get(f"{base}.max_tokens")),
            temperature=float(config.get(f"{base}.temperature")),
            sample_fps=None if sample_fps is None else float(sample_fps),
            native_resolution=None if native is None else bool(native),
            request_timeout_seconds=None if timeout is None else float(timeout),
        )

    def budget(self, requested: int | None) -> int:
        """Clamp downward only. Raising ``max_tokens`` past the profile is a violation."""
        if requested is None:
            return self.max_tokens
        if requested <= 0:
            raise ProfileViolation(f"max_tokens must be positive, got {requested}")
        if requested > self.max_tokens:
            raise ProfileViolation(
                f"profile {self.name!r} caps max_tokens at {self.max_tokens}; "
                f"{requested} was requested. Decode is ~95% of latency and this is the "
                f"dial — raise it in config/settings.yaml with a benchmark, not here."
            )
        return requested


# --------------------------------------------------------------------------------------
# Call inputs and outputs
# --------------------------------------------------------------------------------------


@dataclass
class VLMChunk:
    """The frames of one analysis window, ready to send.

    ``chunk_id`` is the SPEC §3.1 id and is echoed back on the result so a batched call
    cannot lose track of which caption belongs to which window.

    ``frames`` are image URLs — ``data:`` URIs from :func:`encode_frame` in practice.
    This module does no decoding, resizing or overlay burning: by the time frames arrive
    here the wall-clock overlay must already be burned in, *after* any resize
    (CLAUDE.md invariant 8), because the VLM reads it for temporal localization.
    """

    chunk_id: str
    frames: Sequence[str]
    extra_text: str = ""


@dataclass
class VLMResult:
    """One completion, plus the numbers we tune on.

    ``reasoning`` is populated on the deep profile only (SPEC §5 — the
    ``reasoning_description`` the worker shows as proof in SPEC §11.2). On the live path
    it is always empty, by construction.
    """

    chunk_id: str
    text: str
    model: str
    profile: str
    prompt_tokens: int
    completion_tokens: int
    wall_time_ms: float
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def encode_frame(data: bytes, mime: str = "image/jpeg") -> str:
    """Wrap raw image bytes as a ``data:`` URI for the ``image_url`` content part."""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# --------------------------------------------------------------------------------------
# Transport — the seam that keeps tests off the real endpoint
# --------------------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Minimal HTTP seam. Implement this to mock the VLM in a test."""

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout: float | None
    ) -> dict[str, Any]:
        """POST ``payload`` as JSON, return the decoded JSON body.

        Raise :class:`VLMTransportError` for anything that is not a 2xx JSON response.
        """
        ...


class RequestsTransport:
    """Default transport: a pooled ``requests.Session``.

    ``requests`` is imported lazily so that importing this module — which every service
    does — never depends on it being installed in that container image.
    """

    def __init__(self, session: Any | None = None) -> None:
        if session is None:
            import requests  # local: see class docstring

            session = requests.Session()
        self._session = session

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout: float | None
    ) -> dict[str, Any]:
        try:
            response = self._session.post(url, json=dict(payload), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - any transport failure is one failure to us
            raise VLMTransportError(f"POST {url} failed: {exc}") from exc
        status = getattr(response, "status_code", 200)
        if not 200 <= status < 300:
            body = getattr(response, "text", "")[:500]
            raise VLMTransportError(f"POST {url} returned HTTP {status}: {body}")
        try:
            body_json = response.json()
        except Exception as exc:  # noqa: BLE001
            raise VLMResponseError(f"POST {url} returned non-JSON body: {exc}") from exc
        if not isinstance(body_json, dict):
            raise VLMResponseError(
                f"POST {url} returned a {type(body_json).__name__}, not an object"
            )
        return body_json


# --------------------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------------------

# Every field this module can log. ``logging.vlm_calls`` in settings.yaml selects from
# this set; naming a field that is not here fails at construction rather than producing
# a log line with a hole in it.
_LOGGABLE_FIELDS = frozenset(
    {
        "model",
        "profile",
        "prompt_tokens",
        "completion_tokens",
        "wall_time_ms",
        "chunk_id",
        "frames",
        "max_tokens",
        "enable_reasoning",
        "ok",
        "error",
    }
)


class VLMClient:
    """The single entry point to the VLM process.

    Not thread-safe by design of the system rather than of the class: ``shared/queue.py``
    admits one request at a time (``vlm.queue.max_inflight``), because one camera means
    there is never more than one request to make. Reusing one client across threads is
    fine — the transport is, and this class holds no per-call state.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        logger: logging.Logger | None = None,
    ) -> None:
        self._transport: Transport = transport if transport is not None else RequestsTransport()
        self._endpoint = (endpoint or str(config.get("vlm.endpoint"))).rstrip("/")
        # SPEC §10 D1. ``require`` turns "model is null" into a sentence instead of a 404.
        self._model = str(model) if model is not None else str(config.require("vlm.model"))
        # An explicit argument overrides every profile; otherwise the timeout is
        # per-profile and resolved on ProfileSpec. See _timeout_for().
        self._timeout_s = timeout_s
        self._clock = clock
        self._log = logger or LOGGER

        self.live = ProfileSpec.from_config(Profile.LIVE)
        self.deep = ProfileSpec.from_config(Profile.DEEP)
        self._specs = {Profile.LIVE: self.live, Profile.DEEP: self.deep}

        self._log_fields = self._resolve_log_fields()
        self._log_json = str(config.get("logging.format", "json")) == "json"

    # -- properties ---------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _timeout_for(self, spec: ProfileSpec) -> float | None:
        """Explicit constructor argument wins; otherwise the profile's own timeout."""
        return self._timeout_s if self._timeout_s is not None else spec.request_timeout_seconds

    def spec(self, profile: Profile) -> ProfileSpec:
        return self._specs[profile]

    # -- public API ---------------------------------------------------------------

    def caption(
        self,
        chunks: Sequence[VLMChunk],
        *,
        prompt: str,
        max_tokens: int | None = None,
    ) -> list[VLMResult]:
        """Live profile: caption a list of windows (M1, SPEC §2.4).

        Takes a **list** even though ingest always passes one — CLAUDE.md invariant 9.
        Results come back in input order, one per chunk.
        """
        return self._run(Profile.LIVE, chunks, prompt=prompt, max_tokens=max_tokens)

    def analyze(
        self,
        chunks: Sequence[VLMChunk],
        *,
        prompt: str,
        max_tokens: int | None = None,
    ) -> list[VLMResult]:
        """Deep profile: re-watch a window with reasoning on (M4, SPEC §5).

        Same VLM process as :meth:`caption`, different request. Callers reach this only
        via ``deep_analyze`` and never on a user's blocking turn (CLAUDE.md invariant 4).
        """
        return self._run(Profile.DEEP, chunks, prompt=prompt, max_tokens=max_tokens)

    # -- internals ----------------------------------------------------------------

    def _run(
        self,
        profile: Profile,
        chunks: Sequence[VLMChunk],
        *,
        prompt: str,
        max_tokens: int | None,
    ) -> list[VLMResult]:
        spec = self._specs[profile]
        budget = spec.budget(max_tokens)
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required; an empty prompt produces an empty caption")
        # Serviced sequentially: one camera, one request in flight, batching does not
        # help us (SPEC §0). The list is the interface, not a promise of parallelism.
        return [self._one(profile, spec, chunk, prompt, budget) for chunk in chunks]

    def _one(
        self,
        profile: Profile,
        spec: ProfileSpec,
        chunk: VLMChunk,
        prompt: str,
        budget: int,
    ) -> VLMResult:
        if not chunk.frames:
            raise ValueError(f"chunk {chunk.chunk_id!r} has no frames; nothing to caption")
        payload = self._build_payload(spec, chunk, prompt, budget)
        url = f"{self._endpoint}{_CHAT_COMPLETIONS}"

        started = self._clock()
        try:
            body = self._transport.post(url, payload, timeout=self._timeout_for(spec))
        except VLMError as exc:
            # Log failures too. A run of transport errors is a tuning signal, and an
            # unlogged one looks like the VLM simply got slow.
            self._emit(
                spec,
                chunk,
                budget,
                wall_time_ms=(self._clock() - started) * 1000.0,
                prompt_tokens=0,
                completion_tokens=0,
                ok=False,
                error=str(exc),
            )
            raise

        wall_time_ms = (self._clock() - started) * 1000.0
        text, reasoning = self._parse_message(body, spec)
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        self._emit(
            spec,
            chunk,
            budget,
            wall_time_ms=wall_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ok=True,
            error=None,
        )
        return VLMResult(
            chunk_id=chunk.chunk_id,
            text=text,
            reasoning=reasoning,
            model=self._model,
            profile=spec.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_ms=wall_time_ms,
            raw=body,
        )

    def _build_payload(
        self, spec: ProfileSpec, chunk: VLMChunk, prompt: str, budget: int
    ) -> dict[str, Any]:
        # Images first, instruction last: the frames are the subject and the prompt is
        # what to do with them. The wall-clock overlay is already burned into each frame
        # (invariant 8) — that is what the model cites times from.
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}} for url in chunk.frames
        ]
        text = prompt if not chunk.extra_text else f"{prompt}\n\n{chunk.extra_text}"
        content.append({"type": "text", "text": text})
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": budget,
            "temperature": spec.temperature,
            # The reasoning switch rides in chat_template_kwargs, which is how vLLM
            # passes flags into a model's chat template. The exact kwarg name belongs to
            # the variant D1 settles; if it changes, it changes here and nowhere else.
            "chat_template_kwargs": {"enable_reasoning": spec.enable_reasoning},
        }

    def _parse_message(self, body: Mapping[str, Any], spec: ProfileSpec) -> tuple[str, str]:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLMResponseError(f"response has no choices: {_clip(body)}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise VLMResponseError(f"choice has no message: {_clip(body)}")

        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not reasoning and _THINK_CLOSE in text:
            reasoning, _, text = text.partition(_THINK_CLOSE)
            reasoning = reasoning.replace(_THINK_OPEN, "")
        if not spec.enable_reasoning:
            # The live profile has reasoning off; anything that looks like a trace here
            # means the request did not take effect. Drop it rather than indexing it.
            reasoning = ""
        return text.strip(), reasoning.strip()

    # -- structured logging — CLAUDE.md: we cannot tune what we cannot see ----------

    def _resolve_log_fields(self) -> tuple[str, ...]:
        fields = config.get("logging.vlm_calls")
        if not isinstance(fields, list) or not fields:
            raise config.ConfigError("logging.vlm_calls must be a non-empty list of field names")
        unknown = [f for f in fields if f not in _LOGGABLE_FIELDS]
        if unknown:
            raise config.ConfigError(
                f"logging.vlm_calls names field(s) this client cannot produce: {unknown}. "
                f"Known fields: {sorted(_LOGGABLE_FIELDS)}"
            )
        return tuple(str(f) for f in fields)

    def _emit(
        self,
        spec: ProfileSpec,
        chunk: VLMChunk,
        budget: int,
        *,
        wall_time_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        ok: bool,
        error: str | None,
    ) -> None:
        available: dict[str, Any] = {
            "model": self._model,
            "profile": spec.name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_time_ms": round(wall_time_ms, 2),
            "chunk_id": chunk.chunk_id,
            "frames": len(chunk.frames),
            "max_tokens": budget,
            "enable_reasoning": spec.enable_reasoning,
            "ok": ok,
            "error": error,
        }
        record = {name: available[name] for name in self._log_fields}
        if not ok and "error" not in record:
            # A failure that logs like a success is worse than no log at all.
            record["error"] = error
        line = (
            json.dumps(record, sort_keys=True)
            if self._log_json
            else " ".join(f"{k}={v}" for k, v in record.items())
        )
        (self._log.info if ok else self._log.warning)(line)


def _clip(body: Any, limit: int = 300) -> str:
    return repr(body)[:limit]
