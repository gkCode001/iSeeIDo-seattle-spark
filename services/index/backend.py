"""Vector store behind a two-implementation seam — SPEC §3.2.

``pymilvus`` is not installed on this box and Milvus is not running (CLAUDE.md's
machine-state table). Both implementations satisfy :class:`VectorBackend`:

* :class:`InMemoryBackend` — dicts and brute-force cosine, stdlib only. At SPEC §3.2's
  numbers (~4,300 captioned chunks/day, 768 floats each) a full scan of a *week* is
  about 23 M multiply-adds; in pure Python that is well under a second, and one camera
  means one query in flight. It is not a toy — it is the whole retrieval path, running
  today, so M3 can be built before NGC access lands.
* :class:`MilvusBackend` — written against pymilvus' documented ORM API and imported
  lazily, so this module imports fine on a box without it. Untested against a live
  server; that is stated rather than hidden.

Everything is a ``ChunkRecord`` (``shared/schema.py``) in and out. This module never
redefines the record.


Gated records — where the null rows live
----------------------------------------
SPEC §2.3 skips inference on ~80% of windows and still writes a record, because the
skip rate is a health metric and a gap in the record stream is indistinguishable from
crashed ingest. Those records have no caption and no embedding, and they must not
pollute vector search.

They are stored in the **same collection**, in a **separate partition**. Not a separate
database (SPEC §3.2 says one store), not a scalar filter alone, and not dropped.

The reason it is a partition rather than a ``gated == false`` filter: at the target skip
rate the null rows *outnumber* the real ones four to one. Putting 17,000 unsearchable
entries a day into an ANN graph to filter them out at query time degrades recall and
wastes the graph — filtered ANN search prunes candidates it has already paid to visit.
A partition keeps them out of the searched segments entirely while leaving them
queryable for the health metric, which is exactly the split we want.

Gated rows still need *a* vector because a Milvus collection must have a vector field.
They get :data:`NULL_VECTOR`, a well-defined unit vector (a zero vector makes COSINE
normalization produce NaN). It is never scored: nothing searches that partition.

The in-memory backend mirrors this with two dicts, so the behaviour is the same on
either side of the swap.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from shared.schema import ChunkRecord, Tier

from .settings import IndexSettings
from .telemetry import log_event, timed

__all__ = [
    "IndexCounts",
    "ScoredChunk",
    "VectorBackend",
    "InMemoryBackend",
    "MilvusBackend",
    "build_backend",
    "NULL_VECTOR",
    "overlaps",
]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Milvus VARCHAR fields need a declared width. These are schema shape, not tunables —
# a chunk_id is ``cam01_20260814T211107_211112`` (31 chars) and a segment filename is
# similar, so both have an order of magnitude of headroom.
_ID_MAX_LEN = 128
_NAME_MAX_LEN = 256
_TIER_MAX_LEN = 16


def NULL_VECTOR(dims: int) -> list[float]:  # noqa: N802 — reads as a constant at call sites
    """Placeholder vector for a gated record. Unit norm, never searched."""
    vec = [0.0] * dims
    vec[0] = 1.0
    return vec


def _to_us(dt: datetime) -> int:
    """UTC datetime → epoch microseconds.

    Microseconds, not milliseconds: ``ChunkRecord`` timestamps land on sub-second
    offsets once stride drift accumulates (see ``shared/schema.to_iso``), and an
    integer filter that rounds is an integer filter that drops boundary chunks.
    """
    return int((dt.astimezone(timezone.utc) - _EPOCH).total_seconds() * 1_000_000)


def overlaps(
    record: ChunkRecord, t_from: datetime | None, t_to: datetime | None
) -> bool:
    """Does this chunk's footage range intersect ``[t_from, t_to]``?

    **Overlap, not containment.** A 5 s window straddling the requested boundary holds
    the pixels the question is about; requiring containment silently drops exactly the
    chunk at the edge of "what happened around 21:11". Same reasoning as CLAUDE.md
    invariant 3 — ranges, never points.
    """
    if t_from is not None and record.t_end < t_from:
        return False
    if t_to is not None and record.t_start > t_to:
        return False
    return True


@dataclass(frozen=True)
class IndexCounts:
    """Health metrics. SPEC §2.3 wants the skip rate visible; this is where it lands."""

    total: int
    captioned: int
    gated: int

    @property
    def skip_rate(self) -> float:
        """Fraction of windows the detector gate skipped. Target ≥0.80, warn <0.60."""
        return self.gated / self.total if self.total else 0.0


@dataclass(frozen=True)
class BrowsePage:
    """One page of the index in time order, plus the size of the full match set.

    This is the listing counterpart to :class:`ScoredChunk`: no query, no vector, no
    relevance — the ordering is wall-clock and the filters are exact. It exists because
    "show me what the system wrote down" is a different question from "what is relevant
    to this", and answering it through ``search("")`` gives an ANN ranking of a
    zero-vector query, capped at ``rerank_top_n``, with no total to paginate against.

    ``total`` counts every record matching the filters, not the ones on this page —
    without it the UI cannot say "page 3 of 27" and cannot know when to stop.
    """

    records: list[ChunkRecord]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True)
class ScoredChunk:
    """One ANN neighbour: the record plus its vector-space similarity.

    ``score`` is COSINE similarity in [-1, 1], higher is better, identical on both
    backends. SPEC §4.2: this is a distance, not a confidence. It orders candidates for
    the reranker and means nothing on its own.
    """

    record: ChunkRecord
    score: float


class VectorBackend(Protocol):
    """The Milvus seam. Everything above this line is backend-agnostic."""

    def ensure_ready(self) -> None:
        """Create/load the collection. Idempotent; safe to call on every startup."""
        ...

    def upsert(self, records: list[ChunkRecord]) -> int:
        """Write records, routing gated ones out of the searchable partition."""
        ...

    def search(
        self,
        embedding: list[float],
        limit: int,
        *,
        tier: Tier,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
    ) -> list[ScoredChunk]:
        """ANN over captioned records of ``tier`` only. Returns ≤ ``limit``, best first."""
        ...

    def fetch(self, chunk_ids: list[str], *, with_embedding: bool = False) -> list[ChunkRecord]:
        """Look up records by id, gated ones included. Order follows ``chunk_ids``."""
        ...

    def browse(
        self,
        offset: int,
        limit: int,
        *,
        tier: Tier | None = None,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        gated: bool | None = None,
        contains: str | None = None,
        newest_first: bool = True,
    ) -> BrowsePage:
        """A page of records in **wall-clock order**, with the full match count.

        Unlike :meth:`search` this reads gated records too (``gated=None``), because the
        skip rate is only legible when the null rows are visible next to the captioned
        ones. ``gated=False`` restricts to captioned, ``gated=True`` to the skipped.

        ``contains`` is a case-insensitive substring test on the caption — a text filter,
        deliberately not a semantic one. Someone paging the index is checking what was
        written; an ANN ranking would reorder the very thing they are trying to read.
        """
        ...

    def counts(self) -> IndexCounts:
        ...

    def close(self) -> None:
        ...


# --------------------------------------------------------------------------------------
# In-memory
# --------------------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float], a_norm: float) -> float:
    """Cosine similarity given ``a``'s precomputed norm. ``b`` is the query."""
    b_norm = math.sqrt(sum(v * v for v in b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (a_norm * b_norm)


class InMemoryBackend:
    """Brute-force cosine over dicts. Needs nothing but stdlib.

    Two stores, mirroring the Milvus partition split: ``_searchable`` holds captioned
    records and is the only thing scanned; ``_gated`` holds the null rows so the skip
    rate and the continuity of the record stream stay observable.

    Optional JSONL persistence (``index.store.memory_path``) exists so M3 can ingest a
    fixture corpus once and keep querying it across restarts. The file is rewritten
    whole and swapped atomically — at SPEC §3.2's volumes it is megabytes, and a
    half-written index that survives a crash is worse than one that does not.
    """

    def __init__(self, dims: int, path: Path | None = None) -> None:
        self._dims = dims
        self._path = path
        self._searchable: dict[str, ChunkRecord] = {}
        self._gated: dict[str, ChunkRecord] = {}
        self._norms: dict[str, float] = {}
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------------

    def ensure_ready(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._path and self._path.is_file():
            with self._path.open("r", encoding="utf-8") as fh:
                records = [ChunkRecord.from_dict(json.loads(line)) for line in fh if line.strip()]
            self._absorb(records)
            log_event(
                "index.memory.loaded", path=str(self._path), records=len(records)
            )

    def close(self) -> None:
        self._persist()

    # -- writes ------------------------------------------------------------------

    def _absorb(self, records: list[ChunkRecord]) -> None:
        for record in records:
            if record.gated:
                self._gated[record.chunk_id] = record
                self._searchable.pop(record.chunk_id, None)
                self._norms.pop(record.chunk_id, None)
            else:
                self._searchable[record.chunk_id] = record
                self._gated.pop(record.chunk_id, None)
                self._norms[record.chunk_id] = math.sqrt(
                    sum(v * v for v in record.embedding)
                )

    def upsert(self, records: list[ChunkRecord]) -> int:
        self.ensure_ready()
        self._absorb(records)
        self._persist()
        return len(records)

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for record in self._all():
                    fh.write(json.dumps(record.to_dict()) + "\n")
            os.replace(tmp_name, self._path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _all(self) -> list[ChunkRecord]:
        merged = list(self._searchable.values()) + list(self._gated.values())
        return sorted(merged, key=lambda r: (r.t_start, r.chunk_id))

    # -- reads -------------------------------------------------------------------

    def search(
        self,
        embedding: list[float],
        limit: int,
        *,
        tier: Tier,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
    ) -> list[ScoredChunk]:
        self.ensure_ready()
        with timed(
            "index.ann", backend="memory", tier=tier.value, limit=limit
        ) as span:
            hits: list[ScoredChunk] = []
            for chunk_id, record in self._searchable.items():
                if record.tier is not tier or not record.embedding:
                    continue
                if not overlaps(record, t_from, t_to):
                    continue
                score = _cosine(record.embedding, embedding, self._norms[chunk_id])
                hits.append(ScoredChunk(record=self._strip(record), score=score))
            # Descending score; chunk_id breaks ties so repeated queries are stable.
            hits.sort(key=lambda h: (-h.score, h.record.chunk_id))
            span.fields["scanned"] = len(self._searchable)
            span.fields["hits"] = min(len(hits), limit)
        return hits[:limit]

    @staticmethod
    def _strip(record: ChunkRecord) -> ChunkRecord:
        """Drop the vector from a search hit.

        Nothing downstream reads it — M3 wants the caption and the time range, M4 wants
        the time range — and it is 3 KB per hit on the way to an LLM context window.
        ``fetch(..., with_embedding=True)`` gets it back. Milvus does the same, so the
        two backends behave identically.
        """
        return replace(record, embedding=[])

    def fetch(self, chunk_ids: list[str], *, with_embedding: bool = False) -> list[ChunkRecord]:
        self.ensure_ready()
        found: list[ChunkRecord] = []
        for chunk_id in chunk_ids:
            record = self._searchable.get(chunk_id) or self._gated.get(chunk_id)
            if record is not None:
                found.append(record if with_embedding else self._strip(record))
        return found

    def browse(
        self,
        offset: int,
        limit: int,
        *,
        tier: Tier | None = None,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        gated: bool | None = None,
        contains: str | None = None,
        newest_first: bool = True,
    ) -> BrowsePage:
        self.ensure_ready()
        needle = (contains or "").strip().lower()
        with timed("index.browse", backend="memory", offset=offset, limit=limit) as span:
            # ``_all()`` is already sorted by (t_start, chunk_id) — the same order the
            # JSONL is persisted in, so what a reader sees on the page and what they see
            # in the file are the same sequence.
            matched = [
                record
                for record in self._all()
                if (tier is None or record.tier is tier)
                and (gated is None or record.gated is gated)
                and overlaps(record, t_from, t_to)
                and (not needle or needle in record.caption.lower())
            ]
            if newest_first:
                matched.reverse()
            window = [self._strip(r) for r in matched[offset : offset + limit]]
            span.fields["matched"] = len(matched)
            span.fields["returned"] = len(window)
        return BrowsePage(records=window, total=len(matched), offset=offset, limit=limit)

    def counts(self) -> IndexCounts:
        self.ensure_ready()
        captioned, gated = len(self._searchable), len(self._gated)
        return IndexCounts(total=captioned + gated, captioned=captioned, gated=gated)


# --------------------------------------------------------------------------------------
# Milvus
# --------------------------------------------------------------------------------------


class MilvusBackend:
    """pymilvus ORM backend. **Written from the docs, never run** — pymilvus is absent.

    Schema shape and why: the full ``ChunkRecord`` goes in a JSON ``payload`` field, so
    the record round-trips through ``to_dict``/``from_dict`` and cannot be silently
    truncated by a schema that drifted from ``shared/schema.py``. The scalar columns
    alongside it exist only because Milvus filters on scalars, not on JSON: ``camera_id``
    for the day we have a second camera, ``tier`` for SPEC §3.3, ``gated`` as a
    belt-and-braces guard behind the partition split, and ``t_start_us``/``t_end_us``
    for the time-range filter that ``search_index`` (SPEC §4.1) takes.

    The collection is created with strong consistency: ingest writes a chunk and the
    monitor may query for it a second later, and a demo is not the place to discover
    that Milvus' default bounded staleness is measured in seconds.
    """

    _VECTOR_FIELD = "embedding"

    def __init__(self, settings: IndexSettings) -> None:
        self._s = settings
        self._alias = f"m2-{settings.milvus_collection}"
        self._collection: Any = None

    # -- lifecycle ---------------------------------------------------------------

    def _connect(self) -> None:
        from pymilvus import connections  # noqa: PLC0415 — deferred; pymilvus is optional

        if not connections.has_connection(self._alias):
            connections.connect(
                alias=self._alias, host=self._s.milvus_host, port=str(self._s.milvus_port)
            )

    def ensure_ready(self) -> None:
        if self._collection is not None:
            return
        from pymilvus import (  # noqa: PLC0415 — deferred; pymilvus is optional
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        self._connect()
        name = self._s.milvus_collection

        if not utility.has_collection(name, using=self._alias):
            schema = CollectionSchema(
                fields=[
                    FieldSchema("chunk_id", DataType.VARCHAR, max_length=_ID_MAX_LEN, is_primary=True),
                    FieldSchema("camera_id", DataType.VARCHAR, max_length=_ID_MAX_LEN),
                    FieldSchema("segment", DataType.VARCHAR, max_length=_NAME_MAX_LEN),
                    FieldSchema("tier", DataType.VARCHAR, max_length=_TIER_MAX_LEN),
                    FieldSchema("gated", DataType.BOOL),
                    FieldSchema("t_start_us", DataType.INT64),
                    FieldSchema("t_end_us", DataType.INT64),
                    FieldSchema("payload", DataType.JSON),
                    FieldSchema(
                        self._VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self._s.embed_dims
                    ),
                ],
                description="SPEC §3.1 chunk records — vector plus full payload.",
            )
            Collection(
                name=name, schema=schema, using=self._alias, consistency_level="Strong"
            )

        collection = Collection(name=name, using=self._alias)

        for partition in (self._s.live_partition, self._s.gated_partition):
            if not collection.has_partition(partition):
                collection.create_partition(partition)

        if not collection.has_index():
            collection.create_index(
                field_name=self._VECTOR_FIELD,
                index_params={
                    "index_type": self._s.milvus_index_type,
                    "metric_type": self._s.milvus_metric_type,
                },
            )
        collection.load()
        self._collection = collection
        log_event("index.milvus.ready", collection=name, dims=self._s.embed_dims)

    def close(self) -> None:
        # Nothing was ever opened, so there is nothing to import. Closing a backend that
        # failed to start must not raise a *second*, more confusing error on the way out.
        if self._collection is None:
            return
        from pymilvus import connections  # noqa: PLC0415 — deferred; pymilvus is optional

        self._collection.release()
        self._collection = None
        if connections.has_connection(self._alias):
            connections.disconnect(self._alias)

    # -- writes ------------------------------------------------------------------

    def _row(self, record: ChunkRecord) -> dict[str, Any]:
        payload = record.to_dict()
        # The vector lives in its own column; duplicating 3 KB of floats into the JSON
        # payload would double the index for a field nothing reads from there.
        payload.pop("embedding", None)
        vector = (
            NULL_VECTOR(self._s.embed_dims) if record.gated else list(record.embedding)
        )
        return {
            "chunk_id": record.chunk_id,
            "camera_id": record.camera_id,
            "segment": record.segment,
            "tier": record.tier.value,
            "gated": record.gated,
            "t_start_us": _to_us(record.t_start),
            "t_end_us": _to_us(record.t_end),
            "payload": payload,
            self._VECTOR_FIELD: vector,
        }

    def upsert(self, records: list[ChunkRecord]) -> int:
        """Write a batch, split by partition.

        A chunk's ``gated`` flag is decided once, at ingest, and ``chunk_id`` is derived
        from its time range — so a record never moves between partitions. Re-inserting
        the same id with a flipped flag would leave a duplicate in the other partition;
        that is unsupported rather than defended against, because doing it means ingest
        re-ran a window it already gated, which is a different bug.
        """
        self.ensure_ready()
        by_partition: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            partition = (
                self._s.gated_partition if record.gated else self._s.live_partition
            )
            by_partition.setdefault(partition, []).append(self._row(record))

        written = 0
        with timed("index.upsert", backend="milvus", count=len(records)):
            for partition, rows in by_partition.items():
                self._collection.upsert(rows, partition_name=partition)
                written += len(rows)
        return written

    # -- reads -------------------------------------------------------------------

    def _time_expr(self, tier: Tier, t_from: datetime | None, t_to: datetime | None) -> str:
        # Overlap, not containment — see ``overlaps``. A chunk qualifies when it ends at
        # or after t_from AND starts at or before t_to.
        clauses = [f'tier == "{tier.value}"', "gated == false"]
        if t_from is not None:
            clauses.append(f"t_end_us >= {_to_us(t_from)}")
        if t_to is not None:
            clauses.append(f"t_start_us <= {_to_us(t_to)}")
        return " and ".join(clauses)

    @staticmethod
    def _record_from_entity(entity: Any, vector: list[float] | None = None) -> ChunkRecord:
        payload = entity.get("payload")
        record = ChunkRecord.from_dict(payload)
        return replace(record, embedding=list(vector)) if vector else record

    def search(
        self,
        embedding: list[float],
        limit: int,
        *,
        tier: Tier,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
    ) -> list[ScoredChunk]:
        self.ensure_ready()
        expr = self._time_expr(tier, t_from, t_to)
        with timed(
            "index.ann", backend="milvus", tier=tier.value, limit=limit, expr=expr
        ) as span:
            results = self._collection.search(
                data=[embedding],
                anns_field=self._VECTOR_FIELD,
                param={"metric_type": self._s.milvus_metric_type},
                limit=limit,
                expr=expr,
                # The gated partition is never searched. This is the load-bearing half
                # of the null-record decision; the ``gated == false`` clause above is
                # only a guard against a mis-routed write.
                partition_names=[self._s.live_partition],
                output_fields=["payload"],
            )
            hits = [
                ScoredChunk(record=self._record_from_entity(hit.entity), score=float(hit.score))
                for hit in results[0]
            ]
            span.fields["hits"] = len(hits)
        return hits

    def fetch(self, chunk_ids: list[str], *, with_embedding: bool = False) -> list[ChunkRecord]:
        self.ensure_ready()
        if not chunk_ids:
            return []
        quoted = ", ".join(f'"{cid}"' for cid in chunk_ids)
        output = ["payload"] + ([self._VECTOR_FIELD] if with_embedding else [])
        rows = self._collection.query(expr=f"chunk_id in [{quoted}]", output_fields=output)
        by_id = {
            row["payload"]["chunk_id"]: self._record_from_entity(
                row, row.get(self._VECTOR_FIELD) if with_embedding else None
            )
            for row in rows
        }
        # Preserve the caller's order: rerank hands us ids already ranked.
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def browse(
        self,
        offset: int,
        limit: int,
        *,
        tier: Tier | None = None,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        gated: bool | None = None,
        contains: str | None = None,
        newest_first: bool = True,
    ) -> BrowsePage:
        """Scan the matching scalars, order them here, then fetch one page of payloads.

        Milvus ``query`` has no ORDER BY, so its ``offset``/``limit`` paginate an order
        Milvus does not promise: page 2 could repeat a row from page 1. Since the whole
        point of this call is wall-clock order, the ordering is done here — the iterator
        below reads only ``chunk_id`` and ``t_start_us`` (two scalars per row, ~40 bytes),
        and only the page that is actually rendered pays for its payload.

        ``contains`` filters on a JSON path, which needs a Milvus new enough to support
        infix ``like`` on JSON fields. If that turns out to be false on the box this
        finally runs against, the fix is a scalar ``caption`` column beside the JSON,
        not a client-side filter — a filter applied after paging returns short pages.
        """
        self.ensure_ready()
        expr = self._browse_expr(tier, t_from, t_to, gated, contains)
        partitions = self._browse_partitions(gated)
        with timed(
            "index.browse", backend="milvus", offset=offset, limit=limit, expr=expr
        ) as span:
            keys: list[tuple[int, str]] = []
            iterator = self._collection.query_iterator(
                batch_size=self._s.browse_scan_batch,
                expr=expr,
                partition_names=partitions,
                output_fields=["chunk_id", "t_start_us"],
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    keys.extend((int(row["t_start_us"]), str(row["chunk_id"])) for row in batch)
            finally:
                iterator.close()

            keys.sort(reverse=newest_first)
            page_ids = [chunk_id for _, chunk_id in keys[offset : offset + limit]]
            records = self.fetch(page_ids)
            span.fields["matched"] = len(keys)
            span.fields["returned"] = len(records)
        return BrowsePage(records=records, total=len(keys), offset=offset, limit=limit)

    def _browse_expr(
        self,
        tier: Tier | None,
        t_from: datetime | None,
        t_to: datetime | None,
        gated: bool | None,
        contains: str | None,
    ) -> str:
        """Like :meth:`_time_expr` but every clause is optional — listing, not searching.

        In particular ``gated`` is a filter here rather than a fixed ``== false``: the
        skipped windows are half of what a reader is checking when they page the index.
        """
        clauses: list[str] = []
        if tier is not None:
            clauses.append(f'tier == "{tier.value}"')
        if gated is not None:
            clauses.append(f"gated == {str(bool(gated)).lower()}")
        if t_from is not None:
            clauses.append(f"t_end_us >= {_to_us(t_from)}")
        if t_to is not None:
            clauses.append(f"t_start_us <= {_to_us(t_to)}")
        needle = (contains or "").strip()
        if needle:
            # Milvus has no lower(); the caption is stored as written, so a case-folded
            # match is not available here. The in-memory backend is case-insensitive and
            # this one is not — a divergence worth naming rather than hiding.
            escaped = needle.replace("\\", "\\\\").replace('"', '\\"').replace("%", "\\%")
            clauses.append(f'payload["caption"] like "%{escaped}%"')
        # An always-true clause: pymilvus rejects an empty expression, and "no filters"
        # is the default page the browser opens on.
        return " and ".join(clauses) if clauses else 'chunk_id != ""'

    def _browse_partitions(self, gated: bool | None) -> list[str]:
        """Read both partitions unless the caller asked for one side of the split."""
        if gated is True:
            return [self._s.gated_partition]
        if gated is False:
            return [self._s.live_partition]
        return [self._s.live_partition, self._s.gated_partition]

    def counts(self) -> IndexCounts:
        self.ensure_ready()

        def _count(expr: str) -> int:
            rows = self._collection.query(expr=expr, output_fields=["count(*)"])
            return int(rows[0]["count(*)"]) if rows else 0

        gated = _count("gated == true")
        captioned = _count("gated == false")
        return IndexCounts(total=gated + captioned, captioned=captioned, gated=gated)


def build_backend(settings: IndexSettings) -> VectorBackend:
    """Pick an implementation from ``index.store.backend``."""
    backend = settings.store_backend.lower()
    if backend == "memory":
        return InMemoryBackend(settings.embed_dims, settings.memory_path)
    if backend == "milvus":
        return MilvusBackend(settings)
    raise ValueError(f"unknown index.store.backend: {settings.store_backend!r}")
