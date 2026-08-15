"""M2 — the index (SPEC §3).

One store holding the vector *and* the full ``ChunkRecord`` payload, and one retrieval
path over it: embed → ANN k=20 → cross-encoder rerank → top 5, where the top 5 carry
their wall-clock time ranges.

Typical use::

    from services.index import build_index

    with build_index() as index:
        index.insert([chunk])                      # a list, always — invariant 9
        hits = index.search("white van at the loading door")
        t_start, t_end = hits[0].time_range        # what M3 hands to M4

Every implementation choice is config (see ``services/index/settings.py``). The
defaults run with no NGC credentials and no Milvus, so the full pipeline works on this
box today; flipping ``index.store.backend`` to ``milvus`` and the model backends to
``nim`` swaps in the real thing without touching a caller.

``shared/schema.py`` owns ``ChunkRecord``. Nothing here redefines it.
"""

from .backend import (
    BrowsePage,
    IndexCounts,
    InMemoryBackend,
    MilvusBackend,
    ScoredChunk,
    VectorBackend,
    build_backend,
)
from .embedding import Embedder, HashingEmbedder, NIMEmbedder, build_embedder
from .rerank import LexicalReranker, NIMReranker, Reranker, build_reranker
from .settings import PENDING_SETTINGS, IndexSettings
from .store import IndexStore, SearchHit, build_index

__all__ = [
    # the two things most callers need
    "build_index",
    "IndexStore",
    "SearchHit",
    # configuration
    "IndexSettings",
    "PENDING_SETTINGS",
    # seams, for tests and for swapping implementations by hand
    "VectorBackend",
    "InMemoryBackend",
    "MilvusBackend",
    "build_backend",
    "Embedder",
    "HashingEmbedder",
    "NIMEmbedder",
    "build_embedder",
    "Reranker",
    "LexicalReranker",
    "NIMReranker",
    "build_reranker",
    # values that cross the boundary
    "BrowsePage",
    "IndexCounts",
    "ScoredChunk",
]
