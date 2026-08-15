"""Stage 2 — LLM confirm. SPEC §6.2, ~1 s, runs on stage-1 candidates only.

Stage 1 is deliberately loose and over-triggers. Stage 2 is the filter: it reads the
caption and the task and says match / no match. It is also where the **sustain window**
lives, but that is bookkeeping over a series of verdicts and belongs to the funnel — this
module answers one question about one caption and holds no state.

Two implementations behind one protocol, chosen by ``agent.backend`` — the same key M3
reads, so the two surfaces cannot end up on different models:

* :class:`NIMConfirmer` — Nemotron over an OpenAI-compatible route. Needs NGC
  credentials, which this box does not have (CLAUDE.md machine state), and needs SPEC §10
  D3 resolved before it knows which model to ask for.
* :class:`StubConfirmer` — deterministic, stdlib only, no network, no model.

**The stub is not a mock.** It is how the whole funnel — stage 1 through the brakes to
the action log and the Watch pane — gets proven end to end before an LLM is serving, the
same role ``vlm.backend: stub`` plays for M1 and ``index.embed.backend: hashing`` plays
for M2. It is a crude *lexical coverage* test: what fraction of the task's content words
appear in the caption. That is genuinely a different signal from stage 1's hashed cosine
(coverage vs. direction), so the two stages are not the same test run twice — but it is
not semantic, "vehicle" will not confirm "van", and its number is not a quality signal.
Do not tune thresholds against it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from shared import config
from shared.schema import Task

from services.index.embedding import tokenize

from services.monitor.settings import MonitorSettings

__all__ = [
    "ConfirmVerdict",
    "Stage2Confirmer",
    "StubConfirmer",
    "NIMConfirmer",
    "build_confirmer",
]

logger = logging.getLogger("monitor.confirm")

#: Words carrying no scene content. A closed list, not a tunable — it exists so that "a
#: vehicle stopped in front of the fire door" is scored on *vehicle/stopped/front/fire/
#: door* rather than on how many times it says "the". Keep it short: every word removed
#: here is a word the stub can no longer distinguish two tasks by.
_STOPWORDS = frozenset(
    """
    a an the and or of in on at to from for with without into onto is are was were be
    being been by near next it its that this these those there here as
    """.split()
)

#: What stage 2 is instructed to answer. Compared case-insensitively against the first
#: word the model returns, so a model that says "Yes, a van is parked..." still parses.
_YES = "yes"
_NO = "no"


@dataclass(frozen=True)
class ConfirmVerdict:
    """One stage-2 answer about one caption.

    ``score`` is whatever the backend can honestly offer — lexical coverage for the stub,
    None for the LLM, which returns a word and not a number. The Watch pane renders
    stage 2 as a verdict, not a bar, precisely because a confidence here would be made up.
    """

    match: bool
    detail: str = ""
    score: float | None = None
    model: str = ""

    @property
    def verdict(self) -> str:
        """The string the Watch pane renders (``ui/mock/monitor_state.json``)."""
        return "match" if self.match else "no_match"


class Stage2Confirmer(Protocol):
    """Caption + task → match / no match."""

    @property
    def model(self) -> str: ...

    def confirm(self, caption: str, task: Task) -> ConfirmVerdict:
        """Answer for one caption. Must not raise on a plausible caption."""
        ...


# --------------------------------------------------------------------------------------
# Deterministic stand-in
# --------------------------------------------------------------------------------------


def content_words(text: str) -> set[str]:
    """Lowercase word tokens with the stopwords removed.

    Shares ``tokenize`` with M2's embedder and lexical reranker so the three agree about
    what a word is — a stage-1 hit on a token stage 2 cannot see would be very hard to
    explain from the Watch pane.
    """
    return {w for w in tokenize(text) if w not in _STOPWORDS}


class StubConfirmer:
    """Lexical coverage of the task description by the caption. No model, no network.

    Deterministic across processes and across runs: same caption plus same task always
    gives the same verdict, which is what makes a rehearsed demo rehearsable and a test
    of the sustain window a test rather than a coin flip.
    """

    _MODEL_TAG = "stub-lexical-confirm"

    def __init__(self, min_overlap: float) -> None:
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError(f"min_overlap must be a fraction, got {min_overlap!r}")
        self._min_overlap = min_overlap

    @property
    def model(self) -> str:
        return f"{self._MODEL_TAG}-{self._min_overlap:g}"

    def coverage(self, caption: str, describe: str) -> float:
        wanted = content_words(describe)
        if not wanted:
            return 0.0
        seen = content_words(caption)
        return len(wanted & seen) / len(wanted)

    def confirm(self, caption: str, task: Task) -> ConfirmVerdict:
        score = self.coverage(caption, task.describe)
        match = score >= self._min_overlap
        return ConfirmVerdict(
            match=match,
            detail=(
                f"lexical coverage {score:.2f} "
                f"{'>=' if match else '<'} {self._min_overlap:g}"
            ),
            score=score,
            model=self.model,
        )


# --------------------------------------------------------------------------------------
# Real LLM client
# --------------------------------------------------------------------------------------


class NIMConfirmer:
    """One yes/no chat completion per candidate chunk, against ``agent.endpoint``.

    Two things this deliberately does:

    * **Asks for one word.** ``monitor.confirm_max_tokens`` is single digits. Decode is
      ~95% of latency (CLAUDE.md invariant 6) and stage 2's whole budget is ~1 s; a
      confirmer that explains itself is a confirmer that misses the next chunk.
    * **Fails closed.** A transport error returns *no match* rather than raising. Stage 2
      is the gate in front of an action that cannot be un-fired, and "the LLM was
      unreachable" is not evidence that a vehicle is blocking the fire door. The failure
      is logged, and it surfaces in the Watch pane as a task that stops sustaining.

    ``requests`` is imported inside the call so this module stays importable on a box
    without it, matching how M2 treats pymilvus.
    """

    def __init__(self, settings: MonitorSettings) -> None:
        self._s = settings
        self._url = settings.confirm_endpoint.rstrip("/") + settings.confirm_path
        # Re-read through `require` so a null model names SPEC §10 D3 instead of 404ing.
        self._model = str(config.require("agent.model"))

    @property
    def model(self) -> str:
        return self._model

    def _prompt(self, caption: str, task: Task) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._s.confirm_prompt},
            {
                "role": "user",
                "content": (
                    f"CONDITION: {task.describe}\n"
                    f"SCENE: {caption}\n"
                    f"Does the scene satisfy the condition?"
                ),
            },
        ]

    def confirm(self, caption: str, task: Task) -> ConfirmVerdict:
        import requests  # noqa: PLC0415 — deferred so import works without the dep

        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._prompt(caption, task),
            "max_tokens": self._s.confirm_max_tokens,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(self._url, json=body, timeout=self._s.confirm_timeout)
            resp.raise_for_status()
            text = str(resp.json()["choices"][0]["message"]["content"]).strip()
        except Exception as exc:  # noqa: BLE001 - fail closed, see the class docstring
            logger.warning(
                "stage 2 confirm failed; treating as no match",
                extra={
                    "fields": {
                        "model": self._model,
                        "task_id": task.task_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "wall_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                },
            )
            return ConfirmVerdict(
                match=False, detail=f"confirm unavailable: {type(exc).__name__}", model=self._model
            )

        first = text.lower().lstrip().split(maxsplit=1)
        head = first[0].strip(".,:;!\"'") if first else ""
        # Anything that is not an affirmative is a no. An unparseable answer must not be
        # read as consent to fire.
        match = head == _YES
        logger.info(
            "stage 2 confirm",
            extra={
                "fields": {
                    "model": self._model,
                    "profile": "confirm",
                    "task_id": task.task_id,
                    "answer": text[:40],
                    "match": match,
                    "max_tokens": self._s.confirm_max_tokens,
                    "wall_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
            },
        )
        if head not in (_YES, _NO):
            logger.warning(
                "stage 2 answer was not yes/no; read as no match",
                extra={"fields": {"task_id": task.task_id, "answer": text[:80]}},
            )
        return ConfirmVerdict(match=match, detail=text[:200], model=self._model)


def build_confirmer(settings: MonitorSettings) -> Stage2Confirmer:
    """Pick an implementation from ``agent.backend``."""
    backend = settings.confirm_backend.lower()
    if backend == "stub":
        return StubConfirmer(settings.stub_min_overlap)
    if backend == "nim":
        return NIMConfirmer(settings)
    raise ValueError(f"unknown agent.backend for stage 2: {settings.confirm_backend!r}")
