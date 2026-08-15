"""Every dial M2 reads, resolved from ``config/settings.yaml`` in exactly one place.

CLAUDE.md: no magic numbers in service code. The keys that already exist in
settings.yaml are read with :func:`shared.config.get` and no default — a missing one
should fail loudly.

**Keys this module needs that settings.yaml does not have yet** are listed in
``_PENDING`` below and read *with* a default. A default in code is a magic number
wearing a disguise (shared/config.py says so), so each one is temporary and named here
rather than scattered through the backends. Every default is chosen so the module runs
today on a box with no NGC credentials and no Milvus — see CLAUDE.md's machine-state
table. Add them to settings.yaml and the defaults become dead code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared import config
from shared.schema import Tier

__all__ = ["IndexSettings", "PENDING_SETTINGS"]


# --------------------------------------------------------------------------------------
# Settings that belong in config/settings.yaml under ``index:`` and are not there yet.
# Reported to whoever owns settings.yaml; until then these defaults apply.
# --------------------------------------------------------------------------------------
_PENDING: dict[str, object] = {
    # Which implementation to use. "memory" needs nothing; "milvus" needs pymilvus and
    # a running Milvus. Default is the one that works with no credentials.
    "index.store.backend": "memory",
    # Optional JSONL file the in-memory backend loads at startup and rewrites on insert,
    # so M3 can develop against a stable corpus without re-ingesting. null = ephemeral.
    "index.store.memory_path": None,
    # "nim" = the real OpenAI-compatible NIM; "hashing" = the deterministic stdlib
    # stand-in. Same choice for rerank: "nim" or "lexical".
    "index.embed.backend": "hashing",
    "index.rerank.backend": "lexical",
    "index.embed.timeout_seconds": 30,
    "index.rerank.timeout_seconds": 30,
    # NIM's reranker is not an OpenAI-shaped route; it is POST {endpoint}/ranking.
    # Kept configurable because the path has moved between NIM releases.
    "index.rerank.path": "/ranking",
    "index.embed.path": "/embeddings",
    # nemoretriever asymmetric models want to know whether the text is a query or a
    # document. Getting this backwards quietly costs recall.
    "index.embed.query_input_type": "query",
    "index.embed.passage_input_type": "passage",
    # Milvus index build. AUTOINDEX is Milvus' own "don't tune this" choice, which is
    # correct at our scale: SPEC §3.2 puts the whole corpus at ~13 MB/day.
    "index.milvus.index_type": "AUTOINDEX",
    "index.milvus.metric_type": "COSINE",
    # Partition names. Gated records live in their own partition so they never enter
    # the searched graph — see the docstring in backend.py.
    "index.milvus.live_partition": "captioned",
    "index.milvus.gated_partition": "gated",
}

PENDING_SETTINGS: tuple[str, ...] = tuple(_PENDING)


def _pending(dotted: str) -> object:
    """Read a not-yet-in-YAML setting, falling back to the documented default above."""
    return config.get(dotted, _PENDING[dotted])


@dataclass(frozen=True)
class IndexSettings:
    """Resolved configuration for one :class:`~services.index.store.IndexStore`."""

    # --- store -------------------------------------------------------------------
    store_backend: str
    memory_path: Path | None
    milvus_host: str
    milvus_port: int
    milvus_collection: str
    milvus_index_type: str
    milvus_metric_type: str
    live_partition: str
    gated_partition: str

    # --- embed -------------------------------------------------------------------
    embed_backend: str
    embed_model: str
    embed_dims: int
    embed_endpoint: str
    embed_path: str
    embed_timeout: float
    embed_query_input_type: str
    embed_passage_input_type: str

    # --- rerank ------------------------------------------------------------------
    rerank_backend: str
    rerank_model: str
    rerank_endpoint: str
    rerank_path: str
    rerank_timeout: float

    # --- retrieval ---------------------------------------------------------------
    ann_k: int
    rerank_top_n: int

    # --- browse (listing, not retrieval) ------------------------------------------
    browse_scan_batch: int

    # --- tiers (SPEC §3.3) --------------------------------------------------------
    rollup_enabled: bool
    rollup_window_seconds: int

    @property
    def search_tier(self) -> Tier:
        """The tier retrieval reads from.

        SPEC §3.3: ``live`` is the alert path, ``rollup`` is the *search* path. While
        rollup is unbuilt (D4) there is nothing else to search, so we read ``live``.
        Flipping ``index.rollup.enabled`` moves retrieval onto the merged windows — that
        is the whole seam, and it is deliberately the only thing tier costs us today.
        """
        return Tier.ROLLUP if self.rollup_enabled else Tier.LIVE

    @classmethod
    def from_config(cls) -> IndexSettings:
        """Build from ``config/settings.yaml``. Every existing key is required."""
        raw_memory_path = _pending("index.store.memory_path")
        return cls(
            store_backend=str(_pending("index.store.backend")),
            memory_path=Path(str(raw_memory_path)) if raw_memory_path else None,
            milvus_host=str(config.get("index.milvus.host")),
            milvus_port=int(config.get("index.milvus.port")),
            milvus_collection=str(config.get("index.milvus.collection")),
            milvus_index_type=str(_pending("index.milvus.index_type")),
            milvus_metric_type=str(_pending("index.milvus.metric_type")),
            live_partition=str(_pending("index.milvus.live_partition")),
            gated_partition=str(_pending("index.milvus.gated_partition")),
            embed_backend=str(_pending("index.embed.backend")),
            embed_model=str(config.require("index.embed.model")),
            embed_dims=int(config.get("index.embed.dims")),
            embed_endpoint=str(config.get("index.embed.endpoint")),
            embed_path=str(_pending("index.embed.path")),
            embed_timeout=float(_pending("index.embed.timeout_seconds")),  # type: ignore[arg-type]
            embed_query_input_type=str(_pending("index.embed.query_input_type")),
            embed_passage_input_type=str(_pending("index.embed.passage_input_type")),
            rerank_backend=str(_pending("index.rerank.backend")),
            rerank_model=str(config.require("index.rerank.model")),
            rerank_endpoint=str(config.get("index.rerank.endpoint")),
            rerank_path=str(_pending("index.rerank.path")),
            rerank_timeout=float(_pending("index.rerank.timeout_seconds")),  # type: ignore[arg-type]
            ann_k=int(config.get("index.search.ann_k")),
            rerank_top_n=int(config.get("index.search.rerank_top_n")),
            browse_scan_batch=int(config.get("index.browse.scan_batch")),
            rollup_enabled=bool(config.get("index.rollup.enabled")),
            rollup_window_seconds=int(config.get("index.rollup.window_seconds")),
        )
