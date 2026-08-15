"""Caption/question embedding — SPEC §3.4, ``index.embed``.

Two implementations behind one protocol:

* :class:`NIMEmbedder` — the real ``llama-3.2-nemoretriever-300m-embed-v1`` over an
  OpenAI-compatible route. Needs NGC credentials, which this box does not have yet.
* :class:`HashingEmbedder` — signed feature hashing over stdlib only. Deterministic
  across processes, no model, no network.

The stand-in is not a stub. It is a real (crude) lexical retriever: a question sharing
words with a caption lands close to it in cosine space, which is enough for M3 to be
built and tested against the whole pipeline before NGC access arrives. It is *not*
semantic — "vehicle" will not find "van" — so do not tune retrieval thresholds against
it, and do not read its scores as a quality signal.

**The same embedder must serve ingest and retrieval** (SPEC §3.4: "same model as
ingest"). Both sides construct it from ``index.embed``, so switching backends switches
both together or neither. A corpus embedded by one and queried by the other is a silent
recall collapse, not an error, which is why the store re-embeds captions itself rather
than trusting whatever arrived on the record.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .settings import IndexSettings
from .telemetry import timed

__all__ = ["Embedder", "HashingEmbedder", "NIMEmbedder", "build_embedder", "tokenize"]

# Words, plain and simple. Captions are two tight sentences of English (max_tokens=80),
# so a smarter tokenizer buys nothing the reranker will not also see.
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared with the lexical reranker so the two agree."""
    return _WORD_RE.findall(text.lower())


class Embedder(Protocol):
    """Text → unit-norm vector of ``dims`` floats."""

    @property
    def dims(self) -> int: ...

    @property
    def model(self) -> str: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed indexed documents (captions). Takes a list — CLAUDE.md invariant 9."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question. Asymmetric models encode this differently to a passage."""
        ...


# --------------------------------------------------------------------------------------
# Deterministic stand-in
# --------------------------------------------------------------------------------------


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Signed feature hashing (the "hashing trick") over word unigrams and bigrams.

    ``hash()`` on ``str`` is salted per process, so this uses blake2b: an embedding
    written to disk today must still match a query embedded tomorrow in a different
    process. That is the whole point of persisting a corpus.

    Signed hashing — each feature contributes ``+w`` or ``-w`` depending on a second
    hash bit — keeps collisions from all pushing scores the same way, so unrelated text
    stays near zero cosine instead of drifting positive.

    Term weight is ``1 + log(tf)``, the standard sub-linear damp: a caption that says
    "van" three times is about a van, but not three times as much.
    """

    _MODEL_TAG = "hashing-stub"

    def __init__(self, dims: int) -> None:
        if dims <= 0:
            raise ValueError(f"dims must be positive, got {dims}")
        self._dims = dims

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def model(self) -> str:
        # Tagged with the width so a corpus embedded at 768 cannot be silently queried
        # at some other width by a future config edit.
        return f"{self._MODEL_TAG}-{self._dims}"

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        # Bigrams give a little word-order sensitivity: "van reverses" vs "reverses van".
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def _embed_one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for feature in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1

        vec = [0.0] * self._dims
        for feature, tf in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            bucket = raw % self._dims
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            vec[bucket] += sign * (1.0 + math.log(tf))
        return _l2_normalize(vec)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        with timed("index.embed", model=self.model, kind="passage", count=len(texts)):
            return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        with timed("index.embed", model=self.model, kind="query", count=1):
            return self._embed_one(text)


# --------------------------------------------------------------------------------------
# Real NIM client
# --------------------------------------------------------------------------------------


class NIMEmbedder:
    """OpenAI-compatible ``/embeddings`` against the NeMo Retriever embedding NIM.

    Two details that are easy to get wrong and expensive to notice:

    * ``input_type`` — nemoretriever is asymmetric. Sending a question as a passage
      returns a perfectly plausible vector with quietly worse recall.
    * ``dimensions`` — the model is Matryoshka-trained and ``index.embed.dims`` is a
      *truncation*, not a description. Ask for it explicitly; do not assume the server's
      default matches the collection we built.

    ``requests`` is imported inside the call so this module stays importable on a box
    without it, matching how the Milvus backend treats pymilvus.
    """

    def __init__(self, settings: IndexSettings) -> None:
        self._s = settings
        self._url = settings.embed_endpoint.rstrip("/") + settings.embed_path

    @property
    def dims(self) -> int:
        return self._s.embed_dims

    @property
    def model(self) -> str:
        return self._s.embed_model

    def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        import requests  # noqa: PLC0415 — deferred so import works without the dep

        body = {
            "model": self._s.embed_model,
            "input": texts,
            "input_type": input_type,
            "dimensions": self._s.embed_dims,
            "encoding_format": "float",
        }
        with timed(
            "index.embed",
            model=self._s.embed_model,
            kind=input_type,
            count=len(texts),
            chars=sum(len(t) for t in texts),
        ):
            resp = requests.post(self._url, json=body, timeout=self._s.embed_timeout)
            resp.raise_for_status()
            data = resp.json()["data"]

        # The API is allowed to return these out of order; it carries an index for
        # exactly that reason. Re-sort rather than trusting arrival order.
        ordered = sorted(data, key=lambda row: row["index"])
        vectors = [[float(v) for v in row["embedding"]] for row in ordered]
        for vec in vectors:
            if len(vec) != self._s.embed_dims:
                raise ValueError(
                    f"embed endpoint returned {len(vec)} dims, "
                    f"index.embed.dims is {self._s.embed_dims}"
                )
        return vectors

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._post(texts, self._s.embed_passage_input_type)

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], self._s.embed_query_input_type)[0]


def build_embedder(settings: IndexSettings) -> Embedder:
    """Pick an implementation from ``index.embed.backend``."""
    backend = settings.embed_backend.lower()
    if backend == "hashing":
        return HashingEmbedder(settings.embed_dims)
    if backend == "nim":
        return NIMEmbedder(settings)
    raise ValueError(f"unknown index.embed.backend: {settings.embed_backend!r}")
