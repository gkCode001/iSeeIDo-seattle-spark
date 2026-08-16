"""Captioning — SPEC §2.4, steps 4 and 5.

Two implementations behind one interface, chosen by ``vlm.backend``:

``vllm``
    :class:`VLMCaptioner` — ``shared/vlm_client.py``, the one client, on the ``live``
    profile. It does not pass ``max_tokens``: the profile caps it at 80 and the client
    raises on any attempt to exceed that (CLAUDE.md invariant 6). Decode is ~95% of
    latency, so the token budget is the whole latency story and it is not M1's to set.

``stub``
    :class:`StubCaptioner` — deterministic synthetic captions. **This is not a test
    mock.** SPEC §10 D1 (which Cosmos variant) is open, there is no NGC key on this box
    (CLAUDE.md machine state), and the plumbing still has to be proven end to end on real
    footage today: real segments, real gate decisions, real wall clock, real records in
    the index. The stub is what makes the other 95% of M1 runnable while the 5% that
    needs a model is blocked on a credential.

Both take a **list** of chunks (invariant 9) and both return ``VLMResult``, so the
pipeline, the logs and ``bench.py`` cannot tell them apart structurally — only by the
``model`` field, which is the point. Every stub caption is prefixed so that a synthetic
sentence can never be mistaken for a real one on a screen at hour 38.

Neither implementation is called directly. The pipeline submits them through
``shared/queue.py`` at ``Priority.INGEST`` (SPEC §7), which is what keeps one camera's
captioning from ever competing with a user's turn.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Protocol

from shared.vlm_client import Profile, ProfileSpec, VLMChunk, VLMClient, VLMResult

from .settings import IngestError, IngestSettings
from .telemetry import log_event
from .watchlist import Watchlist

__all__ = [
    "Captioner",
    "StubCaptioner",
    "VLMCaptioner",
    "build_captioner",
    "STUB_MODEL",
    "STUB_PREFIX",
]

#: What a stub caption reports as its model. ``ChunkRecord`` has no model field, so this
#: is the only place a run's provenance survives — and ``bench.py`` reads it to decide
#: whether the number it just printed means anything.
STUB_MODEL = "stub"

#: Prefix on every synthetic caption. Deliberately visible in the index and in the UI:
#: an un-marked synthetic caption looks exactly like a real one, and the demo's whole
#: claim is that the captions came from watching the footage.
STUB_PREFIX = "[stub]"

#: Synthetic caption bodies. Written in the register the live prompt asks for — objects,
#: people, vehicles and what they are doing, two short sentences — so that retrieval,
#: reranking and M5's task matching are exercised against text of the right shape rather
#: than against lorem ipsum.
_STUB_SCENES: tuple[str, ...] = (
    "A person in a dark top sits at a desk facing the camera and moves slightly.",
    "An indoor room with a monitor and an office chair. Nobody is moving.",
    "A person leans forward toward the desk and then settles back into the chair.",
    "A door at the left of frame is closed. A person is seated near the centre.",
    "A person turns their head to the right and looks off camera.",
    "An office chair is empty in the foreground. Light comes through blinds behind it.",
    "A person reaches across the desk toward something out of frame.",
    "The scene is static: furniture, a screen and a wall. No people are visible.",
)


class Captioner(Protocol):
    """One caption per chunk, in input order. Takes a list — CLAUDE.md invariant 9."""

    @property
    def model(self) -> str:
        """What produced these captions. Logged, and shouted about by ``bench.py``."""
        ...

    def caption(self, chunks: Sequence[VLMChunk]) -> list[VLMResult]:
        ...


class VLMCaptioner:
    """The real path: ``shared/vlm_client.py`` on the ``live`` profile.

    Thin on purpose. Everything that could be a knob here — reasoning off, 80 tokens,
    temperature, timeout — belongs to the profile, and the client raises rather than
    letting a caller widen it. The prompt comes from ``vlm.prompts.caption`` so that M1
    and the rollup job (SPEC §3.3) cannot drift into two incomparable caption styles in
    one index.
    """

    def __init__(
        self,
        settings: IngestSettings,
        client: VLMClient | None = None,
        watchlist: Watchlist | None = None,
    ) -> None:
        self._s = settings
        self._client = client if client is not None else VLMClient()
        self._watchlist = watchlist if watchlist is not None else Watchlist(
            settings.watchlist_path,
            seed_path=settings.watchlist_seed_path,
            preamble=settings.watchlist_preamble,
            max_items=settings.watchlist_max_items,
            enabled=settings.watchlist_enabled,
        )

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def watchlist(self) -> Watchlist:
        """Exposed so the pipeline can log what the captioner is currently watching for."""
        return self._watchlist

    def caption(self, chunks: Sequence[VLMChunk]) -> list[VLMResult]:
        # No max_tokens argument. The live profile owns the output budget (invariant 6)
        # and asking for more is a ProfileViolation, not a tuning decision M1 gets to make.
        #
        # The watchlist suffix is re-resolved per call rather than cached at construction:
        # tasks are created and deleted through the UI while ingest is running, and a
        # captioner that read the task list once at boot would keep steering captions
        # toward a task the user deleted an hour ago. It is an mtime check, not a re-parse.
        prompt = self._watchlist.apply(self._s.caption_prompt)
        return self._client.caption(list(chunks), prompt=prompt)


class StubCaptioner:
    """Deterministic synthetic captions, so the pipeline runs today.

    Deterministic in the strict sense: the caption for a given ``chunk_id`` is the same
    across processes and machines, because it is derived from a hash of the id rather
    than from a random draw or a counter. Re-running ingest over the same archive
    produces a byte-identical index, which is what makes "did my change break retrieval?"
    an answerable question while D1 is open.

    The caption reports the frame count and the number of bytes it was handed, so a
    silently-empty frame list shows up in the text rather than as a caption about a scene
    nobody sampled.
    """

    def __init__(self, settings: IngestSettings, logger: logging.Logger | None = None) -> None:
        self._s = settings
        self._spec: ProfileSpec = ProfileSpec.from_config(Profile.LIVE)
        self._log = logger or logging.getLogger("services.ingest")
        self._warned = False

    @property
    def model(self) -> str:
        return STUB_MODEL

    def caption(self, chunks: Sequence[VLMChunk]) -> list[VLMResult]:
        self._warn_once()
        return [self._one(chunk) for chunk in chunks]

    def _one(self, chunk: VLMChunk) -> VLMResult:
        if not chunk.frames:
            raise ValueError(f"chunk {chunk.chunk_id!r} has no frames; nothing to caption")
        digest = hashlib.sha256(chunk.chunk_id.encode("utf-8")).digest()
        scene = _STUB_SCENES[digest[0] % len(_STUB_SCENES)]
        text = f"{STUB_PREFIX} {scene} Sampled {len(chunk.frames)} frames."
        result = VLMResult(
            chunk_id=chunk.chunk_id,
            text=text,
            model=STUB_MODEL,
            profile=self._spec.name,
            # Zero, not a plausible-looking guess. A fabricated token count would flow
            # straight into bench.py's tokens-per-second and turn a meaningless number
            # into a meaningless number that looks credible.
            prompt_tokens=0,
            completion_tokens=0,
            wall_time_ms=0.0,
        )
        # Same shape the real client logs, so a stub run and a vLLM run are comparable
        # line for line in the log.
        log_event(
            "vlm.caption",
            model=STUB_MODEL,
            profile=self._spec.name,
            chunk_id=chunk.chunk_id,
            frames=len(chunk.frames),
            max_tokens=self._spec.max_tokens,
            enable_reasoning=self._spec.enable_reasoning,
            prompt_tokens=0,
            completion_tokens=0,
            wall_time_ms=0.0,
            synthetic=True,
        )
        return result

    def _warn_once(self) -> None:
        if self._warned:
            return
        self._warned = True
        self._log.warning(
            "vlm.backend=stub — captions in this run are SYNTHETIC and describe nothing "
            "that is actually in the footage. They exist to prove the M1 plumbing while "
            "SPEC §10 D1 is open and this box has no NGC key. Every caption is prefixed "
            "%r. Set vlm.backend: vllm once a model is serving.",
            STUB_PREFIX,
        )


def build_captioner(settings: IngestSettings, client: VLMClient | None = None) -> Captioner:
    """Select the captioner named by ``vlm.backend``.

    An unknown backend raises rather than defaulting to the stub. Falling back silently
    would mean a misspelled backend produces a full, plausible-looking index of captions
    of nothing — the single worst outcome available to this module.
    """
    backend = settings.vlm_backend.strip().lower()
    if backend == STUB_MODEL:
        return StubCaptioner(settings)
    if backend == "vllm":
        return VLMCaptioner(settings, client)
    raise IngestError(
        f"vlm.backend is {settings.vlm_backend!r}; M1 knows {STUB_MODEL!r} and 'vllm'. "
        f"There is no silent fallback: a typo here would produce an index full of "
        f"captions of nothing."
    )
