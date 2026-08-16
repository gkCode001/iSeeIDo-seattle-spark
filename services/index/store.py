"""M2 — the index. SPEC §3.

    question → embed (same model as ingest) → Milvus ANN, k=20
             → rerank (cross-encoder) → top 5

The top 5 carry their time ranges. That is the whole point: SPEC §3.4's last line, and
the reason CLAUDE.md invariant 2 exists. A hit whose ``t_start``/``t_end``/``segment``/
``pts_offset`` did not survive retrieval intact is not a hit — M4 cannot re-watch
footage it cannot locate.

SPEC §3.2's arithmetic is the argument for keeping everything: 768 floats ≈ 3 KB, so
~4,300 captioned chunks/day ≈ 13 MB of index against 43 GB of video. The index is
effectively free. Keep every window, pay to look closely only when asked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from types import TracebackType
from typing import Any

from shared import config
from shared.captions import split_caption
from shared.schema import ChunkRecord, Tier, to_iso

from .backend import BrowsePage, IndexCounts, ScoredChunk, VectorBackend, build_backend
from .embedding import Embedder, build_embedder
from .rerank import Reranker, build_reranker
from .settings import IndexSettings
from .telemetry import log_event, timed

__all__ = ["SearchHit", "IndexStore", "build_index"]


@dataclass(frozen=True)
class SearchHit:
    """One reranked result, handed to M3 and from there to M4.

    ``record`` is the untouched ``ChunkRecord`` (minus its vector, which nothing
    downstream reads) — so ``segment`` and ``pts_offset`` ride along with the wall-clock
    range and ``shared/timecode.py`` has everything it needs.

    Two scores, deliberately separate. ``ann_score`` is cosine similarity from the
    vector store; ``rerank_score`` is the cross-encoder's. SPEC §4.2: neither is a
    confidence signal — ANN always returns a plausible top-k even when the answer was
    never indexed. They are exposed for debugging and for the UI, not for a threshold.
    """

    record: ChunkRecord
    ann_score: float
    rerank_score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.record.chunk_id

    @property
    def caption(self) -> str:
        return self.record.caption

    @property
    def time_range(self) -> tuple[datetime, datetime]:
        """The footage range, ready for ``request_deep_analysis`` (SPEC §4.1)."""
        return (self.record.t_start, self.record.t_end)

    def to_dict(self) -> dict[str, Any]:
        """Serialization for M3's ``search_index`` tool result.

        The vector is not here and the time range is. That is the correct shape for
        something about to be pasted into an LLM's context.
        """
        return {
            "chunk_id": self.record.chunk_id,
            "camera_id": self.record.camera_id,
            "t_start": to_iso(self.record.t_start),
            "t_end": to_iso(self.record.t_end),
            "segment": self.record.segment,
            "pts_offset": self.record.pts_offset,
            "tier": self.record.tier.value,
            "caption": self.record.caption,
            "ann_score": self.ann_score,
            "rerank_score": self.rerank_score,
            "rank": self.rank,
        }


class IndexStore:
    """The public face of M2. Construct with :func:`build_index` unless you are testing.

    Ingest calls :meth:`insert`. M3's ``search_index`` tool calls :meth:`search`. Nothing
    else in the system should reach past this object into a backend.
    """

    def __init__(
        self,
        backend: VectorBackend,
        embedder: Embedder,
        reranker: Reranker,
        settings: IndexSettings,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._reranker = reranker
        self._s = settings
        if embedder.dims != settings.embed_dims:
            raise ValueError(
                f"embedder produces {embedder.dims} dims but index.embed.dims is "
                f"{settings.embed_dims}; the collection would be built at the wrong width"
            )

    # -- lifecycle ---------------------------------------------------------------

    def ensure_ready(self) -> None:
        self._backend.ensure_ready()

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> IndexStore:
        self.ensure_ready()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- writes ------------------------------------------------------------------

    def insert(self, chunks: list[ChunkRecord]) -> int:
        """Index a batch of chunk records. **Takes a list** — CLAUDE.md invariant 9.

        Ingest passes one today. The batch dimension is here so that captioning more
        than one window at a time later is a config change, not a refactor through
        every caller.

        Records arriving without an embedding are embedded here, from their caption,
        with the embedder configured in ``index.embed`` — the same one :meth:`search`
        uses. SPEC §3.4 requires that; a corpus embedded by one model and queried by
        another does not error, it just quietly stops finding things.

        Gated records (SPEC §2.3) are written as-is: no caption, no vector, routed out
        of the searchable partition by the backend. They are not an error and must not
        be dropped — the skip rate is a health metric, and a gap in the record stream is
        indistinguishable from crashed ingest.
        """
        if not chunks:
            return 0
        self.ensure_ready()

        self._validate(chunks)
        prepared = self._with_embeddings(chunks)

        with timed("index.insert", count=len(prepared)) as span:
            written = self._backend.upsert(prepared)
            span.fields["gated"] = sum(1 for c in prepared if c.gated)
        return written

    def _validate(self, chunks: list[ChunkRecord]) -> None:
        """Enforce the ``shared/schema.py`` contract at the boundary.

        These are contract violations from M1, not data conditions. Failing loudly here
        beats a mystery two modules downstream — a gated record carrying a caption means
        the gate and the captioner disagree about whether inference ran.
        """
        for chunk in chunks:
            if not chunk.chunk_id:
                raise ValueError("chunk_id is required; it is the primary key")
            if chunk.t_end < chunk.t_start:
                raise ValueError(f"{chunk.chunk_id}: t_end precedes t_start")
            if not chunk.segment:
                raise ValueError(
                    f"{chunk.chunk_id}: segment is required — wall clock alone cannot "
                    "locate the pixels (CLAUDE.md invariant 2)"
                )
            if chunk.gated:
                if chunk.caption or chunk.embedding:
                    raise ValueError(
                        f"{chunk.chunk_id}: gated record carries a caption or embedding; "
                        "SPEC §2.3 skips inference entirely for these"
                    )
            else:
                if not chunk.caption:
                    raise ValueError(
                        f"{chunk.chunk_id}: ungated record has no caption — nothing to "
                        "embed and nothing for the reranker to read"
                    )
                if chunk.embedding and len(chunk.embedding) != self._s.embed_dims:
                    raise ValueError(
                        f"{chunk.chunk_id}: embedding is {len(chunk.embedding)} dims, "
                        f"index.embed.dims is {self._s.embed_dims}"
                    )

    def _with_embeddings(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Fill in missing vectors in one call rather than one call per record.

        Returns copies. The caller's ``ChunkRecord`` objects are left alone — a store
        that mutates what you hand it is a store that aliases its own contents, and the
        resulting "the index changed by itself" bug is exactly the slow kind.
        """
        missing = [i for i, c in enumerate(chunks) if not c.gated and not c.embedding]
        if not missing:
            return list(chunks)

        # The DESCRIPTION, not the whole caption. A task-aware caption ends in a WATCHING
        # block quoting every standing task's wording (services/ingest/watchlist.py), so
        # embedding it whole would give every chunk in the corpus an identical tail —
        # flattening exactly the differences search exists to rank on, and pulling
        # unrelated footage into any query that happens to share a word with a task.
        # split_caption() is a no-op on captions without a block.
        vectors = self._embedder.embed_passages(
            [split_caption(chunks[i].caption).description for i in missing]
        )
        if len(vectors) != len(missing):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(missing)} captions"
            )

        prepared = list(chunks)
        for position, vector in zip(missing, vectors):
            prepared[position] = replace(chunks[position], embedding=vector)
        return prepared

    # -- deletes -----------------------------------------------------------------

    def select_before(self, cutoff: datetime) -> list[str]:
        """Ids of every record — captioned and gated — that ended at or before ``cutoff``.

        The read half of the retention sweep (``services/retention.py``). Reading and
        deleting are two calls rather than one so the count can be shown to a human
        first: this is the only operation in the system that destroys the join between a
        caption and its pixels, and it does not come back.
        """
        self.ensure_ready()
        return self._backend.select_before(cutoff)

    def delete(self, chunk_ids: list[str]) -> int:
        """Remove records by id. Returns how many existed to remove.

        Gated rows go too. They are the skip-rate denominator (SPEC §2.3), so keeping
        them past their captioned neighbours would inflate the measured skip rate of
        every window that survived — a health metric quietly reporting on a corpus that
        is no longer there.
        """
        if not chunk_ids:
            return 0
        self.ensure_ready()
        with timed("index.delete", count=len(chunk_ids)) as span:
            removed = self._backend.delete(chunk_ids)
            span.fields["removed"] = removed
        return removed

    # -- reads -------------------------------------------------------------------

    def search(
        self,
        query: str,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        *,
        tier: Tier | None = None,
        ann_k: int | None = None,
        top_n: int | None = None,
    ) -> list[SearchHit]:
        """SPEC §3.4 retrieval. Signature matches M3's ``search_index`` tool (SPEC §4.1).

        ``t_from``/``t_to`` filter on **overlap**, not containment: a window straddling
        the requested boundary holds pixels the question is about, and dropping it is
        how "what happened around 21:11" silently loses the event at 21:10:58.

        ``tier`` defaults to ``index.rollup.enabled`` — ``live`` today, ``rollup`` the
        moment D4 ships. That is the entire seam; retrieval reads one tier and nothing
        else in this module knows the difference.

        Returns at most ``index.search.rerank_top_n`` hits, best first, each carrying its
        wall-clock range. Gated windows never appear: they have no caption to match.
        """
        self.ensure_ready()
        k = ann_k if ann_k is not None else self._s.ann_k
        n = top_n if top_n is not None else self._s.rerank_top_n
        resolved_tier = tier if tier is not None else self._s.search_tier

        with timed(
            "index.search",
            tier=resolved_tier.value,
            ann_k=k,
            top_n=n,
            t_from=to_iso(t_from) if t_from else None,
            t_to=to_iso(t_to) if t_to else None,
        ) as span:
            embedding = self._embedder.embed_query(query)
            candidates = self._backend.search(
                embedding, k, tier=resolved_tier, t_from=t_from, t_to=t_to
            )
            hits = self._rerank(query, candidates, n)
            span.fields["candidates"] = len(candidates)
            span.fields["returned"] = len(hits)
        return hits

    def _rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[SearchHit]:
        """Cross-encoder pass over the ANN candidates, truncated to ``top_n``."""
        if not candidates:
            return []
        # Description only, for the same reason as the embedding above: a reranker scoring
        # a query against three verbatim task descriptions ranks the watchlist, not the
        # footage.
        ranked = self._reranker.rank(
            query, [split_caption(c.record.caption).description for c in candidates]
        )
        for position, _ in ranked:
            if not 0 <= position < len(candidates):
                # A reranker that indexes outside the passages it was given is a bug we
                # cannot paper over — the wrong caption would be attributed to the wrong
                # time range, which is the one mistake this module must never make.
                raise ValueError(
                    f"reranker returned passage index {position} for "
                    f"{len(candidates)} candidates"
                )
        ranked = self._blend_recency(ranked, candidates)

        hits: list[SearchHit] = []
        for rank, (position, score) in enumerate(ranked[:top_n]):
            candidate = candidates[position]
            hits.append(
                SearchHit(
                    record=candidate.record,
                    ann_score=candidate.score,
                    rerank_score=score,
                    rank=rank,
                )
            )
        return hits

    def _blend_recency(
        self, ranked: list[tuple[int, float]], candidates: list[ScoredChunk]
    ) -> list[tuple[int, float]]:
        """Reorder ``ranked`` so newer candidates win ties and near-ties.

        Returns the same ``(position, score)`` pairs, resorted. The score is left as the
        reranker produced it: ``SearchHit.rerank_score`` promises the reranker's opinion,
        and quietly returning a blended number there would make the §4.2 warning about
        not reading it as confidence harder to honour, not easier.

        Both terms are min-max normalised over the candidate set before blending, because
        the two rerankers are not on one scale — ``LexicalReranker`` is bounded in [0, 1]
        and a NIM ``logit`` is unbounded and often negative. Normalising is what lets one
        ``recency_weight`` mean the same thing on both.
        """
        weight = self._s.recency_weight
        half_life = self._s.recency_half_life_seconds
        if weight <= 0.0 or half_life <= 0.0 or len(ranked) < 2:
            return ranked

        # Newest candidate, not wall clock — see the config comment. An explicit past
        # range would otherwise score every candidate as equally old.
        reference = max(candidates[position].record.t_end for position, _ in ranked)

        scores = [score for _, score in ranked]
        low, high = min(scores), max(scores)
        span = high - low

        blended: list[tuple[float, int, int, float]] = []
        for order, (position, score) in enumerate(ranked):
            relevance = (score - low) / span if span > 0 else 0.5
            age = max(0.0, (reference - candidates[position].record.t_end).total_seconds())
            recency = 0.5 ** (age / half_life)
            combined = (1.0 - weight) * relevance + weight * recency
            blended.append((combined, order, position, score))

        # Ties fall back to the reranker's original order, same rule as the reranker's own
        # fallback to ANN order.
        blended.sort(key=lambda item: (-item[0], item[1]))
        return [(position, score) for _, _, position, score in blended]

    def fetch(self, chunk_ids: list[str], *, with_embedding: bool = False) -> list[ChunkRecord]:
        """Look up records by id, gated ones included. Order follows ``chunk_ids``."""
        self.ensure_ready()
        return self._backend.fetch(chunk_ids, with_embedding=with_embedding)

    def browse(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        tier: Tier | None = None,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        gated: bool | None = None,
        contains: str | None = None,
        newest_first: bool = True,
    ) -> BrowsePage:
        """Page the index in wall-clock order — listing, not retrieval.

        :meth:`search` answers "what is relevant to this question"; this answers "what
        did the system write down, in order". They are different enough to be separate
        methods: search is capped at ``rerank_top_n``, reorders by relevance, has no
        total to paginate against, and never returns gated records — every one of which
        is wrong for someone reading the index.

        ``tier`` defaults to **every** tier rather than :attr:`IndexSettings.search_tier`.
        A reader paging the index wants to see that a rollup row exists (SPEC §3.3), not
        have it filtered out by a dial they cannot see.

        Records come back without their vectors, exactly as search hits do — 3 KB of
        floats per row is not what a page of captions is for. ``fetch`` gets them back.
        """
        self.ensure_ready()
        # Clamped rather than rejected: an offset past the end is what "next page" does
        # at the boundary, and it should return an empty page, not a 500.
        safe_offset = max(0, int(offset))
        safe_limit = max(0, int(limit))
        return self._backend.browse(
            safe_offset,
            safe_limit,
            tier=tier,
            t_from=t_from,
            t_to=t_to,
            gated=gated,
            contains=contains,
            newest_first=newest_first,
        )

    # -- health ------------------------------------------------------------------

    def stats(self) -> IndexCounts:
        """Row counts and the gate skip rate (SPEC §2.3).

        The index is the only place the skip rate is observable after the fact, which is
        the second reason null records are stored rather than dropped.
        """
        self.ensure_ready()
        counts = self._backend.counts()
        log_event(
            "index.stats",
            total=counts.total,
            captioned=counts.captioned,
            gated=counts.gated,
            skip_rate=round(counts.skip_rate, 4),
            gate_health=self.gate_health(counts),
        )
        return counts

    @staticmethod
    def gate_health(counts: IndexCounts) -> str:
        """``ok`` / ``low`` / ``empty`` against the ``ingest.gate.*`` thresholds.

        SPEC §2.3: below the warn rate the gate is mistuned and real-time is gone. The
        thresholds live in settings.yaml because they are M1's dials — M2 only reports.
        """
        if counts.total == 0:
            return "empty"
        warn = float(config.get("ingest.gate.warn_skip_rate"))
        return "ok" if counts.skip_rate >= warn else "low"


def build_index(settings: IndexSettings | None = None) -> IndexStore:
    """Construct an :class:`IndexStore` from ``config/settings.yaml``.

    Every implementation choice is config: ``index.store.backend``,
    ``index.embed.backend``, ``index.rerank.backend``. The defaults are the ones that
    run on a box with no NGC credentials and no Milvus, so this returns a working store
    today — see ``services/index/settings.py`` for the keys still to be added.
    """
    resolved = settings or IndexSettings.from_config()
    store = IndexStore(
        backend=build_backend(resolved),
        embedder=build_embedder(resolved),
        reranker=build_reranker(resolved),
        settings=resolved,
    )
    log_event(
        "index.built",
        store=resolved.store_backend,
        embed=resolved.embed_backend,
        rerank=resolved.rerank_backend,
        dims=resolved.embed_dims,
        tier=resolved.search_tier.value,
    )
    return store
