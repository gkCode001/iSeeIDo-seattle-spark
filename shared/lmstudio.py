"""Find the model LM Studio currently has loaded, so the ask surface can use it.

`scripts/serve_models.sh` normally launches llama.cpp *itself*, borrowing the CUDA-13
ARM64 binary that LM Studio ships. This module is the other half of that arrangement: if
LM Studio is already running with a model loaded, the console's model selector can point
M3 at **that** server instead — which is the cheapest way to answer "would model X be
better here?". Load the candidate in the GUI, pick it in the topbar, ask the same
question twice. No re-download into our model dir, no second copy on disk, no editing
``agent.model`` to a filename only our launcher understands.

**This module only reports; it never loads or unloads anything.** Choosing LM Studio in
the UI changes which endpoint M3 talks to. It does not start a server, and it cannot
stop the one `make serve` started — so if both are up, both are resident, and that is a
CLAUDE.md invariant 1 problem that no toggle can fix. :func:`probe` reports when both
answer so the UI can say so out loud.

**What we give up by not launching the server ourselves:**

``--reasoning off``
    Load-bearing, not a preference: gemma-4 is a thinking model, and left on it spends
    the whole budget inside ``reasoning_content`` and returns an **empty** answer with
    ``finish_reason=length``. We cannot pass a launch flag to a server we did not launch,
    so the switch moves into the request body — ``lmstudio.reasoning_off_payload``, which
    :class:`services.agent.llm.OpenAICompatBackend` merges into every request.

``-c 32768``
    Sized from a measurement, not a guess: a 1080p frame costs ~261 prompt tokens through
    the vision encoder, so the deep path's 4 fps fills 32k in ~30 s of footage. LM Studio
    loads whatever context its GUI slider says, and a short one returns HTTP 400 — which
    reaches the console as a deep job that never completes. :func:`resolve` reads
    ``loaded_context_length`` back and refuses below ``lmstudio.min_context_tokens``.

``--mmproj``
    A text-only model answers questions fine but cannot caption a frame. The ask surface
    does not need vision, so ``lmstudio.require_vision`` defaults to false; set it true
    if you ever point the captioner here.

**Two APIs, in preference order.** LM Studio's native ``GET /api/v0/models`` reports
``state``, ``type`` and ``loaded_context_length`` — everything the checks above need. The
OpenAI-compatible ``GET /v1/models`` reports an id and nothing else, so it is only a
fallback, and the checks it cannot perform are reported as such rather than silently
passing.

The fetcher is injected for the same reason :class:`shared.vlm_client.Transport` is:
tests must never touch a real endpoint. Stdlib ``urllib`` is the default because this is
one GET at startup and does not justify a dependency.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shared import config

__all__ = [
    "BACKEND",
    "Fetch",
    "LMStudioError",
    "LMStudioSettings",
    "LMStudioUnreachable",
    "LoadedModel",
    "NoModelLoaded",
    "Probe",
    "UnusableModel",
    "merge_payload",
    "probe",
    "resolve",
    "urllib_fetch",
]

LOGGER = logging.getLogger("shared.lmstudio")

#: The value ``vlm.backend`` / ``agent.backend`` take to select this path.
BACKEND = "lmstudio"

#: LM Studio's own REST API. Reports state/type/context; the OpenAI route does not.
_NATIVE_PATH = "/api/v0/models"
_OPENAI_PATH = "/v1/models"

#: ``type`` values the native API uses for models that can read an image.
_VISION_KINDS = frozenset({"vlm"})

#: ``type`` values that cannot serve any of our three roles.
_NON_CHAT_KINDS = frozenset({"embeddings"})

_UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class LMStudioError(RuntimeError):
    """Base class for every failure raised by this module."""


class LMStudioUnreachable(LMStudioError):
    """Nothing answered on the configured endpoint."""


class NoModelLoaded(LMStudioError):
    """LM Studio is running, but no model is loaded — or too many are to choose."""


class UnusableModel(LMStudioError):
    """A model is loaded, but it cannot do this system's job."""


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LMStudioSettings:
    """Resolved ``lmstudio:`` block.

    One block, read by both the VLM client and the ask backend, because there is one
    process. Separate ``vlm.lmstudio`` / ``agent.lmstudio`` blocks would let M1 and M3
    drift onto different servers, which is the exact failure the shared ``agent.backend``
    key already exists to prevent for M3/M5.
    """

    endpoint: str
    model: str | None
    require_vision: bool
    min_context_tokens: int | None
    resolve_timeout_seconds: float
    reasoning_off_payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> LMStudioSettings:
        model = config.get("lmstudio.model", None)
        min_ctx = config.get("lmstudio.min_context_tokens", None)
        payload = config.get("lmstudio.reasoning_off_payload", None)
        return cls(
            endpoint=str(config.get("lmstudio.endpoint")).rstrip("/"),
            # null means "whatever is loaded" — the whole point of this backend.
            model=str(model) if model else None,
            require_vision=bool(config.get("lmstudio.require_vision", True)),
            min_context_tokens=int(min_ctx) if min_ctx else None,
            resolve_timeout_seconds=float(config.get("lmstudio.resolve_timeout_seconds", 5.0)),
            reasoning_off_payload=dict(payload) if isinstance(payload, Mapping) else {},
        )

    @property
    def base_url(self) -> str:
        """Server root, with the OpenAI ``/v1`` suffix removed if present."""
        return self.endpoint[: -len("/v1")] if self.endpoint.endswith("/v1") else self.endpoint


# --------------------------------------------------------------------------------------
# The loaded model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """What LM Studio says it is currently serving.

    ``kind`` and ``context_length`` are ``None``/``unknown`` when only the
    OpenAI-compatible route was available — that route reports neither, and guessing
    would defeat the checks in :func:`resolve`.
    """

    id: str
    kind: str = _UNKNOWN
    state: str = _UNKNOWN
    context_length: int | None = None
    source: str = _OPENAI_PATH

    @property
    def is_vision(self) -> bool:
        return self.kind in _VISION_KINDS

    @property
    def kind_known(self) -> bool:
        return self.kind != _UNKNOWN

    def describe(self) -> str:
        bits = [self.id]
        if self.kind_known:
            bits.append(self.kind)
        if self.context_length:
            bits.append(f"ctx {self.context_length}")
        return ", ".join(bits)


# --------------------------------------------------------------------------------------
# Transport — injected so tests never reach a real server
# --------------------------------------------------------------------------------------

#: ``(url, timeout) -> decoded JSON``. Raises anything on failure; callers catch broadly.
Fetch = Callable[[str, float], Any]


def urllib_fetch(url: str, timeout: float) -> Any:
    """Default fetcher. Stdlib only — one GET does not justify a dependency."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local http
        return json.loads(response.read().decode("utf-8"))


# --------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------


def resolve(
    settings: LMStudioSettings | None = None,
    *,
    fetch: Fetch | None = None,
    logger: logging.Logger | None = None,
) -> LoadedModel:
    """Find the model LM Studio is serving, and refuse it if it cannot do the job.

    Fails at startup with a sentence rather than at request time with a 404 or a
    never-completing deep job — the same bargain ``config.require`` makes for
    ``vlm.model``.
    """
    settings = settings or LMStudioSettings.from_config()
    fetch = fetch or urllib_fetch
    log = logger or LOGGER

    models, source = _list_models(settings, fetch)
    if not models:
        raise NoModelLoaded(
            f"LM Studio answered on {settings.base_url} but listed no models. "
            "Load one in the GUI (or start it with `lms load <model>`)."
        )

    chosen = _choose(models, settings)
    _check_usable(chosen, settings)

    log.info(
        "lmstudio: serving %s (via %s)",
        chosen.describe(),
        source,
        extra={"model": chosen.id, "kind": chosen.kind, "context_length": chosen.context_length},
    )
    return chosen


def _list_models(settings: LMStudioSettings, fetch: Fetch) -> tuple[list[LoadedModel], str]:
    """Native API first — it is the only one that reports state, type and context."""
    native_url = f"{settings.base_url}{_NATIVE_PATH}"
    try:
        return _parse(fetch(native_url, settings.resolve_timeout_seconds), _NATIVE_PATH), (
            _NATIVE_PATH
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "try the other route"
        native_error = exc

    openai_url = f"{settings.endpoint}/models"
    try:
        models = _parse(fetch(openai_url, settings.resolve_timeout_seconds), _OPENAI_PATH)
    except Exception as exc:  # noqa: BLE001
        raise LMStudioUnreachable(
            f"no LM Studio server answered on {settings.base_url} "
            f"({_NATIVE_PATH}: {native_error}; {_OPENAI_PATH}: {exc}). "
            "Start LM Studio, load a model, and switch its local server on "
            "(Developer tab)."
        ) from exc

    LOGGER.warning(
        "lmstudio: %s unavailable (%s); falling back to %s, which reports no model type "
        "or context length — the vision and context checks cannot run",
        _NATIVE_PATH,
        native_error,
        _OPENAI_PATH,
    )
    return models, _OPENAI_PATH


def _parse(body: Any, source: str) -> list[LoadedModel]:
    data = body.get("data") if isinstance(body, Mapping) else body
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise LMStudioError(f"GET {source} returned no model list: {str(body)[:200]}")

    models: list[LoadedModel] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        model_id = entry.get("id")
        if not model_id:
            continue
        # loaded_context_length is what the model was actually loaded with; the
        # max_context_length ceiling is irrelevant if the GUI slider was left low.
        ctx = entry.get("loaded_context_length") or entry.get("max_context_length")
        models.append(
            LoadedModel(
                id=str(model_id),
                kind=str(entry.get("type") or _UNKNOWN).lower(),
                state=str(entry.get("state") or _UNKNOWN).lower(),
                context_length=int(ctx) if ctx else None,
                source=source,
            )
        )
    return models


def _choose(models: Sequence[LoadedModel], settings: LMStudioSettings) -> LoadedModel:
    """Pin if configured, else take the loaded one — and say so when that is ambiguous."""
    if settings.model:
        for model in models:
            if model.id == settings.model:
                return model
        listed = ", ".join(m.id for m in models) or "nothing"
        raise NoModelLoaded(
            f"lmstudio.model is {settings.model!r}, which LM Studio does not list. "
            f"Listed: {listed}. Set lmstudio.model to null to use whatever is loaded."
        )

    # `state` is only populated by the native route. On the OpenAI fallback every model
    # is state-unknown, and LM Studio lists loaded models there anyway.
    loaded = [m for m in models if m.state == "loaded"]
    if not loaded:
        loaded = [m for m in models if m.state == _UNKNOWN]
    if not loaded:
        listed = ", ".join(f"{m.id} ({m.state})" for m in models)
        raise NoModelLoaded(
            f"LM Studio has no model loaded — listed: {listed}. Load one in the GUI, "
            "or run `lms load <model>`."
        )
    if len(loaded) == 1:
        return loaded[0]

    # More than one loaded is already an invariant-1 problem, but it may be an embedding
    # model alongside the chat one, which is legitimate. Narrow to what we can use.
    usable = [m for m in loaded if m.kind not in _NON_CHAT_KINDS]
    if settings.require_vision:
        vision = [m for m in usable if m.is_vision]
        if len(vision) == 1:
            return vision[0]
        usable = vision or usable
    if len(usable) == 1:
        return usable[0]
    listed = ", ".join(m.describe() for m in loaded)
    raise NoModelLoaded(
        f"LM Studio has {len(loaded)} models loaded and no way to pick between them: "
        f"{listed}. Set lmstudio.model to the one to use — and note that two chat models "
        "resident at once is what CLAUDE.md invariant 1 exists to prevent."
    )


def _check_usable(model: LoadedModel, settings: LMStudioSettings) -> None:
    if model.kind in _NON_CHAT_KINDS:
        raise UnusableModel(
            f"LM Studio is serving {model.id!r}, a {model.kind} model — it cannot answer "
            "a chat request. Load a vision-language model."
        )

    if settings.require_vision:
        if model.kind_known and not model.is_vision:
            raise UnusableModel(
                f"LM Studio is serving {model.id!r}, reported as {model.kind!r}, not a "
                "vision model — it cannot caption a frame. Load a model with a vision "
                "projector, or set lmstudio.require_vision: false to use it for the ask "
                "surface only."
            )
        if not model.kind_known:
            LOGGER.warning(
                "lmstudio: cannot confirm %s is vision-capable (%s does not report a "
                "model type). A text-only model fails at the first caption.",
                model.id,
                model.source,
            )

    min_ctx = settings.min_context_tokens
    if min_ctx and model.context_length and model.context_length < min_ctx:
        raise UnusableModel(
            f"LM Studio loaded {model.id!r} with a {model.context_length}-token context, "
            f"below the {min_ctx} this system needs. A 1080p frame costs ~261 prompt "
            "tokens, so the deep path's 25 s at 4 fps is ~26k tokens; a short context "
            "returns HTTP 400, which reaches the UI as a deep job that never completes. "
            "Raise the context length in LM Studio's loader, or lower "
            "lmstudio.min_context_tokens if you are only using the ask surface."
        )
    if min_ctx and not model.context_length:
        LOGGER.warning(
            "lmstudio: cannot confirm %s has a %d-token context (%s does not report one)",
            model.id,
            min_ctx,
            model.source,
        )


@dataclass(frozen=True)
class Probe:
    """Whether this source can be selected right now, and what to say if not.

    The selector needs an answer for *both* options before the user picks one — an
    option that turns out to be dead only after you choose it is worse than one greyed
    out with a reason. So this never raises: a failed probe is a result.
    """

    available: bool
    model: str | None = None
    detail: str = ""

    @classmethod
    def from_error(cls, exc: Exception) -> Probe:
        return cls(available=False, detail=str(exc))


def probe(
    settings: LMStudioSettings | None = None,
    *,
    fetch: Fetch | None = None,
) -> Probe:
    """:func:`resolve`, with the exception turned into a message the UI can render."""
    try:
        settings = settings or LMStudioSettings.from_config()
    except Exception as exc:  # noqa: BLE001 - a missing config block is just "unavailable"
        return Probe.from_error(exc)
    try:
        model = resolve(settings, fetch=fetch)
    except LMStudioError as exc:
        return Probe.from_error(exc)
    except Exception as exc:  # noqa: BLE001
        return Probe.from_error(exc)
    return Probe(available=True, model=model.id, detail=model.describe())


def merge_payload(payload: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Merge request-body overrides one level deep.

    One level is deliberate: the only nested key anyone sets here is
    ``chat_template_kwargs``, and merging it lets a config add ``enable_thinking``
    without having to restate the switch :mod:`shared.vlm_client` already writes.
    """
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(payload.get(key), Mapping):
            merged = dict(payload[key])
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return payload
