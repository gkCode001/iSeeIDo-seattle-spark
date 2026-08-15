"""M2 tests — SPEC §3, run entirely on the in-memory backend.

    python3 -m unittest discover -s tests -t . -v

Stdlib ``unittest``, no third-party packages, no network. CLAUDE.md forbids calling a
real model endpoint from tests, and on this box there is nothing to call anyway:
pymilvus is not installed and the embed/rerank NIMs need NGC credentials that do not
exist yet. The whole point of the two-implementation seam is that this file exercises
the real retrieval pipeline regardless.

The load-bearing test is :meth:`TestWallClockRetrieval.test_query_returns_correct_wall_clock`
— SPEC §9 says block 6–12 h is done when "a text query returns a chunk with correct
wall-clock times", and that is the assertion.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.index import (
    HashingEmbedder,
    IndexSettings,
    IndexStore,
    InMemoryBackend,
    LexicalReranker,
    MilvusBackend,
    build_backend,
    build_embedder,
    build_index,
    build_reranker,
)
from shared.schema import ChunkRecord, Tier, chunk_id_for

# --------------------------------------------------------------------------------------
# Fixture corpus
#
# A staged afternoon on cam01. Five-second windows on a four-second stride (settings.yaml
# ingest.window_seconds / stride_seconds), one 60 s segment file per SPEC §2.1, and the
# event we go looking for sits at a timestamp we can assert on exactly.
# --------------------------------------------------------------------------------------

CAMERA = "cam01"
SEGMENT_START = datetime(2026, 8, 14, 21, 11, 0, tzinfo=timezone.utc)
SEGMENT = "cam01_20260814_211100.mp4"

# The event. 21:11:07 → 21:11:12, seven seconds into the segment file.
EVENT_START = datetime(2026, 8, 14, 21, 11, 7, tzinfo=timezone.utc)
EVENT_END = datetime(2026, 8, 14, 21, 11, 12, tzinfo=timezone.utc)
EVENT_PTS = 7.0
EVENT_CAPTION = "A white panel van reverses toward the loading door and stops."

# Distractors: same scene, same minute, plausible neighbours. Short windows dilute
# retrieval (SPEC §3.3), which is exactly the condition the reranker exists for.
DISTRACTORS = [
    (0.0, "An empty loading bay under sodium lighting. Nothing moves."),
    (4.0, "A person in a hi-vis jacket walks left to right past the shutter."),
    (16.0, "Two people stand near the shutter talking. The bay is otherwise still."),
    (20.0, "A forklift crosses the bay carrying a stack of pallets."),
    (24.0, "The shutter door lowers halfway and stops."),
    (28.0, "An empty bay. The overhead light flickers once."),
]


def _chunk(
    offset_s: float,
    caption: str,
    *,
    gated: bool = False,
    tier: Tier = Tier.LIVE,
    duration: float = 5.0,
) -> ChunkRecord:
    """Build one record at ``offset_s`` into the segment, with a consistent pts_offset."""
    t_start = SEGMENT_START + timedelta(seconds=offset_s)
    t_end = t_start + timedelta(seconds=duration)
    return ChunkRecord(
        chunk_id=chunk_id_for(CAMERA, t_start, t_end),
        camera_id=CAMERA,
        t_start=t_start,
        t_end=t_end,
        segment=SEGMENT,
        # pts_offset = t_start − segment_start (SPEC §3.1). PTS restarts at zero every
        # file, so this is the only thing that locates the frame inside it.
        pts_offset=offset_s,
        tier=tier,
        gated=gated,
        caption="" if gated else caption,
    )


def event_chunk() -> ChunkRecord:
    return _chunk(EVENT_PTS, EVENT_CAPTION)


def corpus() -> list[ChunkRecord]:
    return [_chunk(off, cap) for off, cap in DISTRACTORS] + [event_chunk()]


GATED_FIRST_OFFSET = 36.0  # clear of the last captioned window, which ends at 33.0


def gated_chunks(count: int) -> list[ChunkRecord]:
    """Null records — SPEC §2.3. No caption, no embedding, still written.

    Placed after every captioned window so a time-filtered search over their range
    genuinely contains nothing else — otherwise the "no gated hits" assertion would pass
    for the wrong reason.
    """
    return [_chunk(GATED_FIRST_OFFSET + 4.0 * i, "", gated=True) for i in range(count)]


class _ReversingReranker:
    """A reranker whose only opinion is "the opposite of whatever ANN said".

    Used to prove the cross-encoder's verdict is what reaches the caller, without
    depending on which way the lexical stand-in happens to break a particular tie.
    """

    @property
    def model(self) -> str:
        return "reversing-test-double"

    def rank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        n = len(passages)
        # Best first, scores descending — the protocol's contract.
        return [(i, float(n - rank)) for rank, i in enumerate(reversed(range(n)))]


def make_store(settings: IndexSettings | None = None, path: Path | None = None) -> IndexStore:
    """An IndexStore on the stdlib-only backends, built from the real settings.yaml."""
    resolved = settings or IndexSettings.from_config()
    return IndexStore(
        backend=InMemoryBackend(resolved.embed_dims, path),
        embedder=HashingEmbedder(resolved.embed_dims),
        reranker=LexicalReranker(),
        settings=resolved,
    )


# --------------------------------------------------------------------------------------
# SPEC §9's done-criterion for this block
# --------------------------------------------------------------------------------------


class TestWallClockRetrieval(unittest.TestCase):
    """A text query returns a chunk with correct wall-clock times."""

    def setUp(self) -> None:
        self.store = make_store()
        self.store.insert(corpus())

    def test_query_returns_correct_wall_clock(self) -> None:
        hits = self.store.search("white van reversing at the loading door")

        self.assertTrue(hits, "retrieval returned nothing")
        top = hits[0]

        # The caption we were looking for...
        self.assertEqual(top.caption, EVENT_CAPTION)

        # ...and, the part that matters, the absolute time it happened. Not a segment
        # offset, not a relative position: UTC wall clock, exact.
        self.assertEqual(top.record.t_start, EVENT_START)
        self.assertEqual(top.record.t_end, EVENT_END)
        self.assertEqual(top.time_range, (EVENT_START, EVENT_END))

    def test_hit_carries_the_full_locator_tuple(self) -> None:
        """CLAUDE.md invariant 2: wall clock *and* segment + pts_offset, or M4 is blind."""
        top = self.store.search("white van reversing at the loading door")[0]

        self.assertEqual(top.record.segment, SEGMENT)
        self.assertEqual(top.record.pts_offset, EVENT_PTS)
        self.assertEqual(top.record.camera_id, CAMERA)
        self.assertEqual(top.record.tier, Tier.LIVE)

        # pts_offset must agree with wall clock against the segment filename's start.
        self.assertEqual(
            top.record.t_start - timedelta(seconds=top.record.pts_offset), SEGMENT_START
        )

    def test_serialized_hit_keeps_the_time_range(self) -> None:
        """What M3 pastes into a context window still lets M4 find the pixels."""
        payload = self.store.search("white van reversing at the loading door")[0].to_dict()

        self.assertEqual(payload["t_start"], "2026-08-14T21:11:07Z")
        self.assertEqual(payload["t_end"], "2026-08-14T21:11:12Z")
        self.assertEqual(payload["segment"], SEGMENT)
        self.assertEqual(payload["pts_offset"], EVENT_PTS)
        # The 3 KB vector has no business in an LLM prompt.
        self.assertNotIn("embedding", payload)

    def test_record_round_trips_unchanged(self) -> None:
        """Everything except the vector survives insert → search byte-for-byte."""
        original = event_chunk()
        top = self.store.search("white van reversing at the loading door")[0]

        self.assertEqual(
            {k: v for k, v in original.to_dict().items() if k != "embedding"},
            {k: v for k, v in top.record.to_dict().items() if k != "embedding"},
        )

    def test_search_hits_omit_the_vector(self) -> None:
        top = self.store.search("white van reversing at the loading door")[0]
        self.assertEqual(top.record.embedding, [])

        # ...but it is still stored, and fetch can hand it back.
        (full,) = self.store.fetch([top.chunk_id], with_embedding=True)
        self.assertEqual(len(full.embedding), IndexSettings.from_config().embed_dims)


# --------------------------------------------------------------------------------------
# insert()
# --------------------------------------------------------------------------------------


class TestInsert(unittest.TestCase):
    def setUp(self) -> None:
        self.store = make_store()

    def test_insert_takes_a_list(self) -> None:
        """CLAUDE.md invariant 9. Ingest passes one; the signature takes many."""
        self.assertEqual(self.store.insert([event_chunk()]), 1)
        self.assertEqual(self.store.insert(_shifted(corpus(), minutes=1)), len(corpus()))
        self.assertEqual(self.store.insert([]), 0)

    def test_insert_embeds_captions_that_arrive_without_a_vector(self) -> None:
        """SPEC §3.4: same model as ingest. The store owns that guarantee."""
        chunk = event_chunk()
        self.assertEqual(chunk.embedding, [])

        self.store.insert([chunk])

        (stored,) = self.store.fetch([chunk.chunk_id], with_embedding=True)
        self.assertEqual(len(stored.embedding), IndexSettings.from_config().embed_dims)
        # The caller's record is untouched — the store does not mutate its inputs.
        self.assertEqual(chunk.embedding, [])

    def test_insert_preserves_a_vector_that_was_supplied(self) -> None:
        settings = IndexSettings.from_config()
        vector = HashingEmbedder(settings.embed_dims).embed_query(EVENT_CAPTION)
        chunk = replace(event_chunk(), embedding=vector)

        self.store.insert([chunk])

        (stored,) = self.store.fetch([chunk.chunk_id], with_embedding=True)
        self.assertEqual(stored.embedding, vector)

    def test_reinsert_updates_in_place(self) -> None:
        """chunk_id is the primary key; re-ingesting a window must not duplicate it."""
        self.store.insert([event_chunk()])
        self.store.insert([replace(event_chunk(), caption="A blue van pulls away.")])

        self.assertEqual(self.store.stats().total, 1)
        (stored,) = self.store.fetch([event_chunk().chunk_id])
        self.assertEqual(stored.caption, "A blue van pulls away.")

    def test_rejects_contract_violations(self) -> None:
        cases = {
            "gated record with a caption": replace(
                event_chunk(), gated=True, caption="something the gate never saw"
            ),
            "ungated record with no caption": replace(event_chunk(), caption=""),
            "missing segment": replace(event_chunk(), segment=""),
            "reversed time range": replace(event_chunk(), t_end=EVENT_START - timedelta(seconds=1)),
            "wrong embedding width": replace(event_chunk(), embedding=[0.1, 0.2, 0.3]),
        }
        for name, chunk in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.store.insert([chunk])


def _shifted(chunks: list[ChunkRecord], *, minutes: int) -> list[ChunkRecord]:
    """Same corpus, a different minute — for tests that need two distinct batches."""
    delta = timedelta(minutes=minutes)
    out = []
    for chunk in chunks:
        t_start, t_end = chunk.t_start + delta, chunk.t_end + delta
        out.append(
            replace(
                chunk,
                chunk_id=chunk_id_for(chunk.camera_id, t_start, t_end),
                t_start=t_start,
                t_end=t_end,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Gated / null records — SPEC §2.3
# --------------------------------------------------------------------------------------


class TestGatedRecords(unittest.TestCase):
    """Stored, never searched, always counted.

    The three properties the design has to hold simultaneously: they must not pollute
    vector search, the skip rate must stay measurable, and the record stream must have
    no gaps (a gap is indistinguishable from crashed ingest).
    """

    def setUp(self) -> None:
        self.store = make_store()
        self.store.insert(corpus())
        self.store.insert(gated_chunks(20))

    def test_gated_records_are_stored_not_dropped(self) -> None:
        gated = gated_chunks(20)
        found = self.store.fetch([c.chunk_id for c in gated])

        self.assertEqual(len(found), len(gated))
        for record in found:
            self.assertTrue(record.gated)
            self.assertEqual(record.caption, "")
            self.assertEqual(record.embedding, [])

    def test_gated_records_never_appear_in_search(self) -> None:
        gated_ids = {c.chunk_id for c in gated_chunks(20)}

        for query in ("empty bay", "white van", "nothing at all", "forklift pallets"):
            with self.subTest(query):
                hits = self.store.search(query)
                self.assertFalse(gated_ids & {h.chunk_id for h in hits})

    def test_gated_records_are_not_returned_by_a_time_filtered_search(self) -> None:
        """Even when the requested range contains nothing but gated windows."""
        gated = gated_chunks(20)
        hits = self.store.search("anything at all", t_from=gated[0].t_start, t_to=gated[-1].t_end)
        self.assertEqual(hits, [])

    def test_skip_rate_is_observable(self) -> None:
        stats = self.store.stats()

        self.assertEqual(stats.captioned, len(corpus()))
        self.assertEqual(stats.gated, 20)
        self.assertEqual(stats.total, len(corpus()) + 20)
        self.assertAlmostEqual(stats.skip_rate, 20 / (len(corpus()) + 20))

    def test_gate_health_flags_a_mistuned_gate(self) -> None:
        """settings.yaml warns below 0.60 — SPEC §2.3 says real-time is gone there."""
        healthy = make_store()
        healthy.insert(corpus()[:1])
        healthy.insert(gated_chunks(20))
        self.assertEqual(healthy.gate_health(healthy.stats()), "ok")

        # 7 captioned to 2 gated is a 22% skip rate: the VLM is doing 4x its budget.
        mistuned = make_store()
        mistuned.insert(corpus())
        mistuned.insert(gated_chunks(2))
        self.assertEqual(mistuned.gate_health(mistuned.stats()), "low")

        self.assertEqual(make_store().gate_health(make_store().stats()), "empty")


# --------------------------------------------------------------------------------------
# Time filtering — M3's search_index(query, t_from?, t_to?) — SPEC §4.1
# --------------------------------------------------------------------------------------


class TestTimeFiltering(unittest.TestCase):
    def setUp(self) -> None:
        self.store = make_store()
        self.store.insert(corpus())

    def test_range_excludes_chunks_outside_it(self) -> None:
        hits = self.store.search(
            "van at the loading door",
            t_from=SEGMENT_START,
            t_to=SEGMENT_START + timedelta(seconds=10),
        )
        self.assertTrue(hits)
        for hit in hits:
            self.assertLessEqual(hit.record.t_start, SEGMENT_START + timedelta(seconds=10))
            self.assertGreaterEqual(hit.record.t_end, SEGMENT_START)

    def test_a_window_straddling_the_boundary_is_kept(self) -> None:
        """Overlap, not containment.

        The event runs 21:11:07–21:11:12. Asking about "around 21:11:11" must return it
        even though the window is not contained in the request — otherwise the chunk
        holding the pixels is the one chunk we drop.
        """
        mid_event = EVENT_START + timedelta(seconds=4)
        hits = self.store.search(
            "white van reversing", t_from=mid_event, t_to=mid_event + timedelta(seconds=1)
        )

        self.assertIn(event_chunk().chunk_id, {h.chunk_id for h in hits})
        self.assertEqual(hits[0].time_range, (EVENT_START, EVENT_END))

    def test_open_ended_ranges(self) -> None:
        after = self.store.search("bay", t_from=EVENT_START)
        self.assertTrue(all(h.record.t_end >= EVENT_START for h in after))

        before = self.store.search("bay", t_to=EVENT_END)
        self.assertTrue(all(h.record.t_start <= EVENT_END for h in before))

    def test_range_with_nothing_in_it_returns_empty(self) -> None:
        far_future = SEGMENT_START + timedelta(days=1)
        hits = self.store.search("white van", t_from=far_future, t_to=far_future + timedelta(hours=1))
        self.assertEqual(hits, [])


# --------------------------------------------------------------------------------------
# The retrieval pipeline itself — SPEC §3.4
# --------------------------------------------------------------------------------------


class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = IndexSettings.from_config()
        self.store = make_store(self.settings)
        # More candidates than ann_k so truncation is actually exercised.
        for minute in range(6):
            self.store.insert(_shifted(corpus(), minutes=minute))

    def test_returns_at_most_rerank_top_n(self) -> None:
        hits = self.store.search("white van at the loading door")
        self.assertLessEqual(len(hits), self.settings.rerank_top_n)
        self.assertEqual(self.settings.rerank_top_n, 5)  # SPEC §3.4

    def test_ann_k_bounds_what_the_reranker_sees(self) -> None:
        """k=20 in, 5 out. Nothing outside the ANN candidate set can be returned."""
        self.assertEqual(self.settings.ann_k, 20)

        narrow = self.store.search("white van", ann_k=1)
        self.assertEqual(len(narrow), 1)

        wide = self.store.search("white van", ann_k=self.settings.ann_k, top_n=20)
        self.assertLessEqual(len(wide), self.settings.ann_k)

    def test_results_are_ordered_by_rerank_score(self) -> None:
        hits = self.store.search("forklift carrying pallets")
        self.assertEqual([h.rank for h in hits], list(range(len(hits))))
        scores = [h.rerank_score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_both_scores_are_exposed_separately(self) -> None:
        """SPEC §4.2: neither is confidence, but the UI has to be able to show both."""
        top = self.store.search("white van reversing")[0]
        self.assertIsInstance(top.ann_score, float)
        self.assertIsInstance(top.rerank_score, float)
        self.assertGreater(top.ann_score, 0.0)

    def test_the_reranker_decides_the_final_order(self) -> None:
        """Rerank is wired in, not decoration.

        Asserted with a reranker that simply reverses the ANN order, because that is
        deterministic: whatever the bi-encoder thought, what comes back is what the
        cross-encoder said. Whether the *lexical* stand-in reorders a given query is a
        property of the stand-in and is covered in :class:`TestReranker`.
        """
        settings = self.settings
        query = "white van at the loading door"

        ann_only = InMemoryBackend(settings.embed_dims)
        store = IndexStore(
            ann_only, HashingEmbedder(settings.embed_dims), _ReversingReranker(), settings
        )
        store.insert(corpus())

        embedding = HashingEmbedder(settings.embed_dims).embed_query(query)
        ann_order = [
            c.record.chunk_id
            for c in ann_only.search(embedding, settings.ann_k, tier=Tier.LIVE)
        ]
        returned = [h.chunk_id for h in store.search(query, top_n=settings.ann_k)]

        self.assertEqual(returned, list(reversed(ann_order)))

    def test_rerank_scores_travel_with_the_hit(self) -> None:
        """The score attached to a hit is the one the reranker gave *that* passage."""
        hits = self.store.search("a forklift crossing the bay with pallets")
        for hit in hits:
            self.assertGreaterEqual(hit.rerank_score, 0.0)
        # The passage that actually mentions a forklift must outrank the ones that do not.
        self.assertIn("forklift", hits[0].caption.lower())

    def test_empty_index_returns_no_hits(self) -> None:
        self.assertEqual(make_store().search("anything"), [])

    def test_search_is_deterministic(self) -> None:
        first = [h.chunk_id for h in self.store.search("white van at the loading door")]
        second = [h.chunk_id for h in self.store.search("white van at the loading door")]
        self.assertEqual(first, second)


# --------------------------------------------------------------------------------------
# Tiers — SPEC §3.3 / §10 D4
# --------------------------------------------------------------------------------------


class TestTiers(unittest.TestCase):
    """``live`` ships; ``rollup`` is a seam, not an implementation."""

    def setUp(self) -> None:
        self.settings = IndexSettings.from_config()
        self.store = make_store(self.settings)
        self.store.insert(corpus())
        self.rollup = _chunk(
            0.0,
            "Over the last minute: an empty bay, a hi-vis walker, a white van reversing "
            "to the door, and a forklift crossing with pallets.",
            tier=Tier.ROLLUP,
            duration=60.0,
        )
        self.store.insert([self.rollup])

    def test_default_tier_follows_the_rollup_flag(self) -> None:
        self.assertFalse(self.settings.rollup_enabled)  # SPEC §10 D4, unresolved
        self.assertIs(self.settings.search_tier, Tier.LIVE)
        self.assertIs(replace(self.settings, rollup_enabled=True).search_tier, Tier.ROLLUP)

    def test_live_search_does_not_see_rollup_chunks(self) -> None:
        hits = self.store.search("white van reversing", tier=Tier.LIVE)
        self.assertNotIn(self.rollup.chunk_id, {h.chunk_id for h in hits})
        self.assertEqual(hits[0].chunk_id, event_chunk().chunk_id)

    def test_rollup_search_sees_only_rollup_chunks(self) -> None:
        hits = self.store.search("white van reversing", tier=Tier.ROLLUP)
        self.assertEqual([h.chunk_id for h in hits], [self.rollup.chunk_id])
        # And it still carries a usable time range — a 60 s one.
        self.assertEqual(hits[0].record.duration, 60.0)


# --------------------------------------------------------------------------------------
# The embedder and reranker stand-ins
# --------------------------------------------------------------------------------------


class TestEmbedder(unittest.TestCase):
    def setUp(self) -> None:
        self.dims = IndexSettings.from_config().embed_dims

    def test_dims_come_from_config(self) -> None:
        self.assertEqual(self.dims, 768)  # SPEC §3.4, Matryoshka-truncated
        self.assertEqual(len(HashingEmbedder(self.dims).embed_query("a van")), self.dims)

    def test_deterministic_across_instances(self) -> None:
        """Not ``hash()``: a vector written to disk today must match a query tomorrow."""
        a = HashingEmbedder(self.dims).embed_query(EVENT_CAPTION)
        b = HashingEmbedder(self.dims).embed_query(EVENT_CAPTION)
        self.assertEqual(a, b)

    def test_vectors_are_unit_norm(self) -> None:
        vector = HashingEmbedder(self.dims).embed_query(EVENT_CAPTION)
        self.assertAlmostEqual(sum(v * v for v in vector) ** 0.5, 1.0, places=9)

    def test_shared_wording_scores_above_unrelated_wording(self) -> None:
        embedder = HashingEmbedder(self.dims)
        query = embedder.embed_query("white van at the loading door")
        related = embedder.embed_passages([EVENT_CAPTION])[0]
        unrelated = embedder.embed_passages(["The overhead light flickers once."])[0]

        self.assertGreater(_dot(query, related), _dot(query, unrelated))

    def test_empty_text_does_not_explode(self) -> None:
        vector = HashingEmbedder(self.dims).embed_query("")
        self.assertEqual(len(vector), self.dims)
        self.assertEqual(sum(vector), 0.0)

    def test_passage_api_takes_a_list(self) -> None:
        self.assertEqual(HashingEmbedder(self.dims).embed_passages([]), [])
        self.assertEqual(len(HashingEmbedder(self.dims).embed_passages(["a", "b", "c"])), 3)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestReranker(unittest.TestCase):
    def test_empty_passages(self) -> None:
        self.assertEqual(LexicalReranker().rank("anything", []), [])

    def test_returns_every_passage_exactly_once(self) -> None:
        passages = [c for _, c in DISTRACTORS] + [EVENT_CAPTION]
        ranked = LexicalReranker().rank("white van loading door", passages)

        self.assertEqual(sorted(i for i, _ in ranked), list(range(len(passages))))

    def test_ranks_the_matching_passage_first(self) -> None:
        passages = [c for _, c in DISTRACTORS] + [EVENT_CAPTION]
        top, _ = LexicalReranker().rank("white panel van reverses", passages)[0]
        self.assertEqual(passages[top], EVENT_CAPTION)

    def test_scores_are_bounded_and_descending(self) -> None:
        passages = [c for _, c in DISTRACTORS] + [EVENT_CAPTION]
        scores = [s for _, s in LexicalReranker().rank("white van", passages)]

        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_a_query_matching_nothing_scores_zero(self) -> None:
        ranked = LexicalReranker().rank("helicopter refuelling", [c for _, c in DISTRACTORS])
        self.assertTrue(all(s == 0.0 for _, s in ranked))


# --------------------------------------------------------------------------------------
# The backend seam
# --------------------------------------------------------------------------------------


class TestBackendSeam(unittest.TestCase):
    """Both implementations must be constructible on a box with neither dependency."""

    def test_defaults_run_without_credentials(self) -> None:
        """CLAUDE.md machine state: no NGC key, no Milvus, no pymilvus. Still works."""
        settings = IndexSettings.from_config()
        self.assertEqual(settings.store_backend, "memory")
        self.assertEqual(settings.embed_backend, "hashing")
        self.assertEqual(settings.rerank_backend, "lexical")

        with build_index() as store:
            store.insert([event_chunk()])
            self.assertEqual(
                store.search("white van")[0].time_range, (EVENT_START, EVENT_END)
            )

    def test_milvus_backend_constructs_without_pymilvus(self) -> None:
        """Lazy import: selecting the backend must not require the package to exist."""
        settings = replace(IndexSettings.from_config(), store_backend="milvus")
        backend = build_backend(settings)

        self.assertIsInstance(backend, MilvusBackend)
        with self.assertRaises(ImportError):
            backend.ensure_ready()
        # And closing one that never opened must not raise a second, confusing error.
        backend.close()

    def test_nim_clients_construct_without_a_server(self) -> None:
        """Constructing a client must not touch the network. Nor must an empty batch."""
        settings = replace(
            IndexSettings.from_config(), embed_backend="nim", rerank_backend="nim"
        )
        embedder = build_embedder(settings)
        reranker = build_reranker(settings)

        self.assertEqual(embedder.model, settings.embed_model)
        self.assertEqual(embedder.dims, settings.embed_dims)
        self.assertEqual(reranker.model, settings.rerank_model)
        self.assertEqual(embedder.embed_passages([]), [])
        self.assertEqual(reranker.rank("q", []), [])

    def test_unknown_backend_names_fail_loudly(self) -> None:
        settings = IndexSettings.from_config()
        for field, value in (
            ("store_backend", "postgres"),
            ("embed_backend", "word2vec"),
            ("rerank_backend", "vibes"),
        ):
            with self.subTest(field), self.assertRaises(ValueError):
                builder = {
                    "store_backend": build_backend,
                    "embed_backend": build_embedder,
                    "rerank_backend": build_reranker,
                }[field]
                builder(replace(settings, **{field: value}))

    def test_dims_mismatch_is_caught_at_construction(self) -> None:
        settings = IndexSettings.from_config()
        with self.assertRaises(ValueError):
            IndexStore(
                InMemoryBackend(settings.embed_dims),
                HashingEmbedder(64),  # not index.embed.dims
                LexicalReranker(),
                settings,
            )


class TestPersistence(unittest.TestCase):
    """The in-memory backend can survive a restart, so M3 need not re-ingest to iterate."""

    def test_corpus_survives_a_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"

            with make_store(path=path) as store:
                store.insert(corpus())
                store.insert(gated_chunks(3))

            with make_store(path=path) as reopened:
                stats = reopened.stats()
                self.assertEqual(stats.captioned, len(corpus()))
                self.assertEqual(stats.gated, 3)

                top = reopened.search("white van reversing at the loading door")[0]
                self.assertEqual(top.time_range, (EVENT_START, EVENT_END))
                self.assertEqual(top.record.segment, SEGMENT)
                self.assertEqual(top.record.pts_offset, EVENT_PTS)


class TestBrowse(unittest.TestCase):
    """Listing the index — what the /api/index browser pages through.

    Different guarantees from :meth:`search`, and the difference is the point: wall-clock
    order rather than relevance, gated rows included rather than filtered out, and a
    total so a caller can paginate at all.
    """

    def setUp(self) -> None:
        self.store = make_store()
        self.store.insert(corpus())
        self.store.insert(gated_chunks(3))
        self.total = len(corpus()) + 3

    def test_lists_captioned_and_gated_together(self) -> None:
        """A browser that hides the skipped windows shows a corpus 4× too small."""
        page = self.store.browse(limit=100)

        self.assertEqual(page.total, self.total)
        self.assertEqual(len(page.records), self.total)
        self.assertEqual(sum(1 for r in page.records if r.gated), 3)

    def test_newest_first_by_default(self) -> None:
        page = self.store.browse(limit=100)
        starts = [r.t_start for r in page.records]
        self.assertEqual(starts, sorted(starts, reverse=True))

    def test_oldest_first_is_the_exact_reverse(self) -> None:
        newest = self.store.browse(limit=100)
        oldest = self.store.browse(limit=100, newest_first=False)
        self.assertEqual(
            [r.chunk_id for r in newest.records],
            list(reversed([r.chunk_id for r in oldest.records])),
        )

    def test_pages_tile_the_corpus_without_gaps_or_repeats(self) -> None:
        """The property that makes pagination trustworthy: page 2 starts where 1 ended."""
        seen: list[str] = []
        for offset in range(0, self.total, 3):
            page = self.store.browse(offset=offset, limit=3)
            self.assertEqual(page.total, self.total)
            seen.extend(r.chunk_id for r in page.records)

        self.assertEqual(len(seen), self.total)
        self.assertEqual(len(set(seen)), self.total)
        self.assertEqual(seen, [r.chunk_id for r in self.store.browse(limit=100).records])

    def test_offset_past_the_end_is_an_empty_page_not_an_error(self) -> None:
        page = self.store.browse(offset=10_000, limit=10)
        self.assertEqual(page.records, [])
        self.assertEqual(page.total, self.total)

    def test_negative_offset_is_clamped(self) -> None:
        self.assertEqual(self.store.browse(offset=-5, limit=2).offset, 0)

    def test_total_counts_the_match_set_not_the_page(self) -> None:
        page = self.store.browse(limit=2)
        self.assertEqual(len(page.records), 2)
        self.assertEqual(page.total, self.total)

    def test_gated_filter_selects_one_side_of_the_split(self) -> None:
        captioned = self.store.browse(limit=100, gated=False)
        skipped = self.store.browse(limit=100, gated=True)

        self.assertEqual(captioned.total, len(corpus()))
        self.assertEqual(skipped.total, 3)
        self.assertTrue(all(not r.gated for r in captioned.records))
        self.assertTrue(all(r.gated for r in skipped.records))
        self.assertTrue(all(r.caption == "" for r in skipped.records))

    def test_contains_is_a_case_insensitive_substring(self) -> None:
        page = self.store.browse(limit=100, contains="WHITE PANEL VAN")
        self.assertEqual([r.caption for r in page.records], [EVENT_CAPTION])
        self.assertEqual(page.total, 1)

    def test_contains_matching_nothing_is_an_empty_page(self) -> None:
        page = self.store.browse(limit=100, contains="helicopter")
        self.assertEqual(page.records, [])
        self.assertEqual(page.total, 0)

    def test_time_range_filters_on_overlap(self) -> None:
        """Invariant 3's rule again: a window straddling the boundary is in range."""
        # A one-second range strictly inside the event window. Containment would drop it.
        inside_start = EVENT_START + timedelta(seconds=2)
        page = self.store.browse(
            limit=100, t_from=inside_start, t_to=inside_start + timedelta(seconds=1)
        )
        self.assertIn(EVENT_CAPTION, [r.caption for r in page.records])

    def test_records_carry_the_full_locator_tuple(self) -> None:
        """Invariant 2 — a row without segment + pts_offset cannot be re-watched."""
        page = self.store.browse(limit=100, contains="white panel van")
        record = page.records[0]
        self.assertEqual(record.t_start, EVENT_START)
        self.assertEqual(record.t_end, EVENT_END)
        self.assertEqual(record.segment, SEGMENT)
        self.assertEqual(record.pts_offset, EVENT_PTS)

    def test_vectors_are_not_shipped(self) -> None:
        """3 KB of floats per row, for a page nobody reads them on."""
        page = self.store.browse(limit=100, gated=False)
        self.assertTrue(all(r.embedding == [] for r in page.records))

    def test_tier_filter(self) -> None:
        self.store.insert([_chunk(200.0, "A merged minute of the bay.", tier=Tier.ROLLUP)])
        live = self.store.browse(limit=100, tier=Tier.LIVE)
        rollup = self.store.browse(limit=100, tier=Tier.ROLLUP)

        self.assertEqual(rollup.total, 1)
        self.assertEqual(live.total, self.total)
        # Unfiltered means every tier: a reader should see that the rollup row exists,
        # rather than have it hidden by index.rollup.enabled, a dial they cannot see.
        self.assertEqual(self.store.browse(limit=100).total, self.total + 1)


class TestSettings(unittest.TestCase):
    def test_reads_the_real_settings_file(self) -> None:
        settings = IndexSettings.from_config()

        self.assertEqual(settings.milvus_collection, "chunks")
        self.assertEqual(settings.milvus_port, 19530)
        self.assertEqual(settings.embed_model, "llama-3.2-nemoretriever-300m-embed-v1")
        self.assertEqual(settings.rerank_model, "llama-3.2-nv-rerankqa-1b-v2")
        self.assertEqual(settings.embed_dims, 768)
        self.assertEqual(settings.ann_k, 20)
        self.assertEqual(settings.rerank_top_n, 5)


if __name__ == "__main__":
    unittest.main()
