"""The deep request itself, and the confidence number attached to its answer (SPEC §5).

Two backends behind one protocol, chosen by ``vlm.backend``:

* :class:`VLMAnalysisBackend` — the real thing. Frames go to ``VLMClient.analyze()``,
  which enforces the deep profile (``enable_reasoning=true``, ``max_tokens≈600``,
  CLAUDE.md invariant 6). Needs ``vlm.model``, which is UNRESOLVED pending SPEC §10 D1.
* :class:`StubAnalysisBackend` — deterministic synthetic answers, no model, no network.

**The stub is not a test mock.** There is no NGC key and nothing serving on this box, so
the stub is how M3's escalation and M5's stage 3 get exercised end to end on real footage
today. Everything it returns is stamped with :data:`STUB_MARKER` and reports
``is_stub``, in the answer text itself as well as structurally, because the one thing
that must never happen is a synthetic answer being read off a screen on stage as though
the system had watched the video. It still decodes the real frames and still travels
through the real queue at the real priority — what it does not do is look at pixels, and
it says so in its first sentence.

Confidence
----------
See :func:`derive_confidence`. Short version: it is a **coverage heuristic**, not a
probability, and nothing in it comes from the model. SPEC §4.2 makes the same point about
retrieval distance — a plausible-looking score that measures the wrong thing is worse
than no score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from shared.schema import to_iso
from shared.vlm_client import VLMChunk, VLMClient, encode_frame

from .settings import WorkerSettings

__all__ = [
    "STUB_MARKER",
    "HEDGE_MARKERS",
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisBackend",
    "StubAnalysisBackend",
    "VLMAnalysisBackend",
    "build_analysis_backend",
    "detect_hedge",
    "derive_confidence",
]

logger = logging.getLogger("services.worker.analysis")

#: Prefixed to every stub answer. Long and ugly on purpose — it has to be unmissable in a
#: chat bubble on a projector.
STUB_MARKER = "[STUB — no VLM is serving on this box; SPEC §10 D1 is open]"

#: Phrases that mean "the footage did not show me enough". ``vlm.prompts.deep`` explicitly
#: instructs the model to say so rather than guess, so this is the model's own uncertainty
#: signal, read out of the only channel it has: the text.
HEDGE_MARKERS: tuple[str, ...] = (
    "cannot tell",
    "can't tell",
    "cannot determine",
    "cannot confirm",
    "not visible",
    "not clear",
    "unclear",
    "unable to",
    "does not show",
    "doesn't show",
    "no way to tell",
    "not enough",
    "insufficient",
    "obscured",
    "too blurry",
    "i cannot",
)


def detect_hedge(text: str) -> bool:
    """True when the answer text says, in so many words, that it could not see enough."""
    lowered = text.lower()
    return any(marker in lowered for marker in HEDGE_MARKERS)


@dataclass(frozen=True)
class AnalysisRequest:
    """One deep look at one range. Assembled by the worker, consumed by a backend."""

    chunk_id: str
    question: str
    t_start: datetime
    t_end: datetime
    frames: tuple[Path, ...]
    segments: tuple[str, ...]
    covered_seconds: float
    gap_seconds: float

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class AnalysisResult:
    """What a backend returns. Deliberately not a ``VLMResult``.

    ``hedged`` is a property of this layer, not of the model: it is our reading of the
    answer text, and the confidence heuristic consumes it. Keeping it here rather than
    inferring it in the worker means the stub can assert its own uncertainty honestly
    instead of relying on its synthetic prose happening to trip a keyword.
    """

    answer: str
    reasoning: str
    hedged: bool
    is_stub: bool
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_ms: float = 0.0
    raw: dict[str, object] = field(default_factory=dict)


class AnalysisBackend(Protocol):
    """Frames + a question -> an answer. The seam ``vlm.backend`` selects across."""

    @property
    def name(self) -> str: ...

    @property
    def is_stub(self) -> bool: ...

    def analyze(self, request: AnalysisRequest) -> AnalysisResult: ...


# --------------------------------------------------------------------------------------
# Stub — how the escalation path is demonstrated while D1 is open
# --------------------------------------------------------------------------------------


class StubAnalysisBackend:
    """Deterministic synthetic answers. Same range and question -> same bytes, always.

    Determinism is the feature: it is what lets a test assert that the dedupe brake
    returned the *same* job rather than quietly re-running the work, and what lets a
    rehearsal produce the same screen twice.

    The answer reports what the pipeline actually did — how many frames came off which
    segment files, over which wall-clock range — because that is the part of the deep path
    that is genuinely being proven today. It reports nothing about the pixels, and says so.
    """

    _MODEL_TAG = "stub-deep"

    @property
    def name(self) -> str:
        return self._MODEL_TAG

    @property
    def is_stub(self) -> bool:
        return True

    def _digest(self, request: AnalysisRequest) -> str:
        material = "|".join(
            [
                request.question.strip().lower(),
                to_iso(request.t_start),
                to_iso(request.t_end),
                str(request.frame_count),
                ",".join(request.segments),
            ]
        )
        # blake2b rather than hash(): salted string hashing differs per process, and two
        # runs of the same demo must print the same job digest.
        return hashlib.blake2b(material.encode("utf-8"), digest_size=3).hexdigest()

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        started = time.perf_counter()
        digest = self._digest(request)
        segments = ", ".join(request.segments) or "no segment files"
        answer = (
            f"{STUB_MARKER} "
            f"Re-watched {to_iso(request.t_start)} to {to_iso(request.t_end)} — "
            f"{request.frame_count} frames at native resolution from {segments}. "
            f"No model read these pixels, so this answer cannot address "
            f"{request.question.strip().rstrip('?')!r}. "
            f"Stub digest {digest}."
        )
        reasoning = (
            f"{STUB_MARKER} Synthetic trace. The deep path resolved the range to "
            f"{len(request.segments)} segment file(s), decoded {request.frame_count} "
            f"frames covering {request.covered_seconds:.2f}s of footage "
            f"({request.gap_seconds:.2f}s missing from the archive), and would have sent "
            f"them to the VLM with reasoning enabled. Everything downstream of the frame "
            f"extraction is real; the words in the answer are not."
        )
        wall_time_ms = (time.perf_counter() - started) * 1000.0
        # The real client logs model/profile/tokens/wall time on every call (CLAUDE.md:
        # we cannot tune what we cannot see). The stub path must be just as visible, or
        # a demo run leaves no trace of what it did.
        logger.info(
            json.dumps(
                {
                    "model": self._MODEL_TAG,
                    "profile": "deep",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "wall_time_ms": round(wall_time_ms, 2),
                    "chunk_id": request.chunk_id,
                    "frames": request.frame_count,
                    "stub": True,
                },
                sort_keys=True,
            )
        )
        return AnalysisResult(
            answer=answer,
            reasoning=reasoning,
            # Asserted, not sniffed. A stub answer is by construction not grounded in the
            # footage, so it can never be allowed to carry a confident-looking number.
            hedged=True,
            is_stub=True,
            model=self._MODEL_TAG,
            wall_time_ms=wall_time_ms,
        )


# --------------------------------------------------------------------------------------
# The real backend
# --------------------------------------------------------------------------------------


class VLMAnalysisBackend:
    """``VLMClient.analyze()`` on the deep profile — SPEC §5 step 3.

    The client owns the profile: ``enable_reasoning=true`` and ``max_tokens≈600`` are read
    from ``vlm.profiles.deep`` and cannot be raised from here (invariant 6). This class
    only assembles frames and reads the result apart.

    ``analyze`` takes a **list** of chunks even though we always pass one (invariant 9) —
    that list is the client's interface, not a promise of parallelism.
    """

    def __init__(self, client: VLMClient, *, prompt: str, max_tokens: int | None = None) -> None:
        self._client = client
        self._prompt = prompt
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self._client.model

    @property
    def is_stub(self) -> bool:
        return False

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.frames:
            raise ValueError(
                f"no frames decoded for {request.chunk_id!r}; refusing to ask the VLM "
                f"about footage that was never read"
            )
        frames = tuple(encode_frame(Path(p).read_bytes()) for p in request.frames)
        chunk = VLMChunk(
            chunk_id=request.chunk_id,
            frames=frames,
            # The question rides as extra_text so the shared deep prompt in
            # ``vlm.prompts.deep`` stays the single place the instruction is worded.
            extra_text=f"Question: {request.question.strip()}",
        )
        results = self._client.analyze([chunk], prompt=self._prompt, max_tokens=self._max_tokens)
        result = results[0]
        return AnalysisResult(
            answer=result.text,
            reasoning=result.reasoning,
            hedged=detect_hedge(result.text),
            is_stub=False,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            wall_time_ms=result.wall_time_ms,
            raw=dict(result.raw),
        )


def build_analysis_backend(
    settings: WorkerSettings,
    *,
    client: VLMClient | None = None,
) -> AnalysisBackend:
    """Pick a backend from ``vlm.backend``.

    The real client is constructed **lazily and only when asked for**: it resolves
    ``vlm.model`` through ``config.require``, which raises while D1 is open. Building it
    eagerly would make the stub path unusable on the box the stub exists for.
    """
    if settings.is_stub_backend:
        return StubAnalysisBackend()
    if settings.backend == "vllm":
        return VLMAnalysisBackend(
            client if client is not None else VLMClient(),
            prompt=settings.deep_prompt,
            max_tokens=settings.max_tokens,
        )
    raise ValueError(f"unknown vlm.backend: {settings.backend!r} (expected 'stub' or 'vllm')")


# --------------------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------------------


def derive_confidence(
    *,
    requested_seconds: float,
    covered_seconds: float,
    frames_decoded: int,
    expected_frames: int,
    hedged: bool,
    hedged_factor: float,
) -> float:
    """A **heuristic**, in [0, 1]. Not a probability, and not the model's opinion.

    It answers one question — *how much of what was asked about did this answer actually
    get to look at?* — and nothing else. It is deliberately not called "certainty",
    because it says nothing about whether the answer is true.

    Three factors, multiplied:

    1. ``coverage = covered_seconds / requested_seconds``. Footage that exists in the
       archive, over footage that was requested. A recorder restart in the middle of the
       range means a third of the question was never seen, so the number falls by a third.
       This is the term that stops a gap being silently swallowed (invariant 3).
    2. ``decode_yield = frames_decoded / expected_frames``, clamped to 1. Catches the
       cases coverage cannot see: a truncated segment whose moov atom is missing, an
       ffmpeg step that failed, an extractor that decoded nothing at all. Zero frames is
       zero confidence, which is the correct reading of "answered without looking".
    3. ``hedged_factor`` (``agent.deep.hedged_confidence_factor``, 0.35) when the answer
       hedges — either because the text trips :data:`HEDGE_MARKERS`, or because the
       backend asserted it, as the stub always does. ``vlm.prompts.deep`` tells the model
       to say when the footage does not show enough; taking it at its word is the only
       content signal available without a second inference call.

    Deliberately **not** used, and each for a reason that has already burned someone:

    * Token counts or logprobs. A 600-token trace is not more likely to be right than a
      200-token one; it is just longer.
    * Retrieval distance. SPEC §4.2 — ANN always returns a plausible top-k, including for
      answers that were never indexed.
    * A constant floor to make the number "look reasonable". A confident-looking number we
      made up is exactly what this function exists to avoid.

    Consequence worth stating: on a complete range with a full decode and an unhedged
    answer this returns exactly ``1.0``. That is honest about what it measures — we read
    all of the footage — and it is *not* a claim that the answer is certainly correct. Any
    UI rendering it should label it coverage, not certainty.
    """
    if requested_seconds <= 0:
        raise ValueError(f"requested_seconds must be positive, got {requested_seconds}")
    coverage = max(0.0, min(1.0, covered_seconds / requested_seconds))
    if expected_frames <= 0:
        decode_yield = 0.0
    else:
        decode_yield = max(0.0, min(1.0, frames_decoded / expected_frames))
    value = coverage * decode_yield
    if hedged:
        value *= hedged_factor
    return round(max(0.0, min(1.0, value)), 4)


def confidence_explanation(
    *,
    requested_seconds: float,
    covered_seconds: float,
    frames_decoded: int,
    expected_frames: int,
    hedged: bool,
    hedged_factor: float,
) -> str:
    """One line naming every factor that produced the number. For the log and the report."""
    coverage = 0.0 if requested_seconds <= 0 else covered_seconds / requested_seconds
    yield_ = 0.0 if expected_frames <= 0 else frames_decoded / expected_frames
    parts = [
        f"coverage {covered_seconds:.2f}/{requested_seconds:.2f}s = {min(coverage, 1.0):.3f}",
        f"decode {frames_decoded}/{expected_frames} frames = {min(yield_, 1.0):.3f}",
    ]
    parts.append(f"hedged x{hedged_factor:g}" if hedged else "unhedged x1")
    return "; ".join(parts)


def segments_of(spans: Sequence[object]) -> tuple[str, ...]:
    """Segment file names touched by a resolution, gaps excluded, in time order."""
    names: list[str] = []
    for span in spans:
        name = getattr(span, "segment", "")
        if name and name not in names:
            names.append(name)
    return tuple(names)
