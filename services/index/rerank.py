"""Cross-encoder rerank — SPEC §3.4, ``index.rerank``.

ANN gets us 20 plausible neighbours; the reranker decides which 5 actually answer the
question. SPEC §4.2 is blunt about why this matters: retrieval distance is not a
confidence signal, and at a 5 s window neighbouring embeddings look alike, so the
bi-encoder has very little to discriminate on (SPEC §3.3).

Two implementations, same protocol:

* :class:`NIMReranker` — ``llama-3.2-nv-rerankqa-1b-v2``. Needs NGC credentials.
* :class:`LexicalReranker` — IDF-weighted term overlap computed *over the candidate
  set*, stdlib only.

The stand-in deliberately scores against the candidates rather than a fixed corpus,
because that is the one behaviour a reranker has that a bi-encoder does not: a word
every candidate contains carries no information, and dropping it is what makes the
ordering change. It is not a cross-encoder and will not catch negation or entity
binding — the two things a real one earns its 1B parameters on.
"""

from __future__ import annotations

import math
from typing import Protocol

from .embedding import tokenize
from .settings import IndexSettings
from .telemetry import timed

__all__ = ["Reranker", "LexicalReranker", "NIMReranker", "build_reranker"]


class Reranker(Protocol):
    """(query, passages) → ``(passage_index, score)`` pairs, best first."""

    @property
    def model(self) -> str: ...

    def rank(self, query: str, passages: list[str]) -> list[tuple[int, float]]: ...


class LexicalReranker:
    """IDF-weighted query-term coverage, IDF computed over the candidate set only.

    Score is the fraction of the query's *information* the passage covers: 1.0 means the
    passage contains every query term that discriminates within this candidate set, 0.0
    means none. Bounded in [0, 1] so it reads like a relevance score rather than an
    unscaled logit — but it is still a stand-in, so do not calibrate the §4.2
    groundedness gate against it.
    """

    MODEL_TAG = "lexical-stub"

    @property
    def model(self) -> str:
        return self.MODEL_TAG

    def rank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []

        with timed("index.rerank", model=self.model, count=len(passages)):
            passage_terms = [set(tokenize(p)) for p in passages]
            n_docs = len(passages)

            # df over the candidates, not a global corpus: a term in every candidate
            # separates nothing here, whatever it means in general English.
            doc_freq: dict[str, int] = {}
            for terms in passage_terms:
                for term in terms:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

            query_terms = set(tokenize(query))
            weights = {
                term: math.log(1.0 + n_docs / (1.0 + doc_freq.get(term, 0)))
                for term in query_terms
            }
            total = sum(weights.values())

            scored: list[tuple[int, float]] = []
            for i, terms in enumerate(passage_terms):
                if total <= 0.0:
                    score = 0.0
                else:
                    score = sum(w for t, w in weights.items() if t in terms) / total
                scored.append((i, score))

        # Ties broken by original ANN order: when the reranker cannot separate two
        # candidates, the bi-encoder's opinion is the next best thing, and a stable
        # order keeps test expectations meaningful.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


class NIMReranker:
    """NVIDIA NeMo Retriever reranking NIM.

    Not an OpenAI-shaped route — it is ``POST {endpoint}/ranking`` with a ``query``
    object and a ``passages`` list, answering with ``rankings`` already sorted by
    ``logit`` descending. The path has moved between NIM releases, so it comes from
    ``index.rerank.path``.

    ``logit`` is unbounded and un-calibrated. It orders candidates; it is not a
    probability, and SPEC §4.2 warns against reading it as confidence.
    """

    def __init__(self, settings: IndexSettings) -> None:
        self._s = settings
        self._url = settings.rerank_endpoint.rstrip("/") + settings.rerank_path

    @property
    def model(self) -> str:
        return self._s.rerank_model

    def rank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []
        import requests  # noqa: PLC0415 — deferred so import works without the dep

        body = {
            "model": self._s.rerank_model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            # Captions are short, but a rollup passage (SPEC §3.3) is 12 of them merged
            # and can outrun the context window. Truncating beats a 422 mid-demo.
            "truncate": "END",
        }
        with timed(
            "index.rerank",
            model=self._s.rerank_model,
            count=len(passages),
            chars=sum(len(p) for p in passages),
        ):
            resp = requests.post(self._url, json=body, timeout=self._s.rerank_timeout)
            resp.raise_for_status()
            rankings = resp.json()["rankings"]

        return [(int(r["index"]), float(r["logit"])) for r in rankings]


def build_reranker(settings: IndexSettings) -> Reranker:
    """Pick an implementation from ``index.rerank.backend``."""
    backend = settings.rerank_backend.lower()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "nim":
        return NIMReranker(settings)
    raise ValueError(f"unknown index.rerank.backend: {settings.rerank_backend!r}")
