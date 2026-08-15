"""M3 tests — SPEC §4, run against the real in-memory index.

    python3 -m unittest discover -s tests -t . -v

Stdlib ``unittest``, no third-party packages, no network beyond a loopback socket this
file opens itself. CLAUDE.md forbids calling a real model endpoint from tests, and on
this box there is nothing to call: ``agent.backend`` is ``stub`` and ``agent.model`` is
null pending SPEC §10 D3. The retrieval underneath is *not* stubbed — every ask below
goes through M2's real embed → ANN → rerank path.

The load-bearing tests are :class:`TestEscalation` (both §4.2 mechanisms, and the
verdict persisted on the turn) and :class:`TestNeverBlocks` (a user turn that returns
while the deep job is still running — CLAUDE.md invariant 4). Everything else supports
those two claims.

M4 is built concurrently, so the deep worker is injected here as a fake in every test.
Nothing in this file imports ``services.worker``.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.agent import (
    AgentApp,
    AskAgent,
    AskServer,
    ChatLog,
    JobRegistry,
    SeedTaskRegistry,
    StubBackend,
    Toolbox,
    UnavailableAnalyzer,
    WebSocketHub,
    accept_key,
    build_app,
)
from services.agent.agent import TRIGGER_GATE, TRIGGER_TOOL, _parse_verdict
from services.agent.llm import LLMError, LLMRequest, LLMResponse, Purpose, ToolCall
from services.agent.settings import AgentSettings
from services.agent.tasks import task_from_payload
from services.agent.ws import client_key, mask_frame
from services.index import build_index
from services.mcp import ActionServer, NullClipCutter
from shared import config
from shared.schema import (
    ActionKind,
    ChatTurn,
    ChunkRecord,
    DeepJob,
    JobState,
    chunk_id_for,
    from_iso,
    to_iso,
    utcnow,
)

# Numbers come from settings.yaml, not from this file. CLAUDE.md: no magic numbers.
WINDOW_SECONDS = float(config.get("ingest.window_seconds"))
STRIDE_SECONDS = float(config.get("ingest.stride_seconds"))
ANN_K = int(config.get("index.search.ann_k"))
TOP_N = int(config.get("index.search.rerank_top_n"))
DEEP_TIMEOUT = float(config.get("agent.deep.timeout_seconds"))

CAMERA = "cam01"
SEGMENT = "cam01_20260814_211100.mp4"
SEGMENT_START = datetime(2026, 8, 14, 21, 11, 0, tzinfo=timezone.utc)

#: The SPEC §10 D6 pair, on one staged afternoon: one question the captions answer and
#: one they genuinely cannot.
GROUNDED_QUESTION = "When did the van arrive at the loading door?"
ESCALATING_QUESTION = "Was the van's rear door open when it backed up?"

#: A caption is two tight sentences (vlm.profiles.live.max_tokens = 80). None of them
#: records whether a door was open — that is the point of the escalating question.
CAPTIONS: list[tuple[float, str]] = [
    (0.0, "An empty loading bay under sodium lighting. Nothing moves."),
    (7.0, "A white panel van reverses toward the loading door and stops."),
    (11.0, "A white panel van is stopped across the loading bay in front of the fire door."),
    (19.0, "The van remains stationary while a person in a hi-vis jacket walks past the shutter."),
    (47.0, "Two figures approach the rear of the van from the right."),
]


def fixture_chunks() -> list[ChunkRecord]:
    """The staged corpus, on ingest's real window/stride."""
    chunks: list[ChunkRecord] = []
    for offset, caption in CAPTIONS:
        t_start = SEGMENT_START + timedelta(seconds=offset)
        t_end = t_start + timedelta(seconds=WINDOW_SECONDS)
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id_for(CAMERA, t_start, t_end),
                camera_id=CAMERA,
                t_start=t_start,
                t_end=t_end,
                segment=SEGMENT,
                pts_offset=offset,
                caption=caption,
            )
        )
    return chunks


# --------------------------------------------------------------------------------------
# Fakes. The deep worker (M4) is built concurrently; its contract is SPEC §5 plus
# shared/schema.py, and that is all these stand in for.
# --------------------------------------------------------------------------------------


class FakeAnalyzer:
    """An M4 that answers on command. ``submit`` returns QUEUED without doing anything."""

    def __init__(self, answer: str = "Yes. The rear doors are open from 21:11:19 onward.") -> None:
        self.answer = answer
        self.submitted: list[tuple[datetime, datetime, str]] = []
        self.release = threading.Event()
        self.entered = threading.Event()
        self.concurrent = 0
        self.max_concurrent = 0
        self.delay_s = 0.0
        self._lock = threading.Lock()
        self._seq = 0

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        self.submitted.append((t_start, t_end, question))
        with self._lock:
            self._seq += 1
            job_id = f"job{self._seq:02d}"
        return DeepJob(
            job_id=job_id,
            t_start=t_start,
            t_end=t_end,
            question=question,
            state=JobState.QUEUED,
        )

    def result(self, job: DeepJob, timeout_s: float) -> DeepJob:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.entered.set()
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            elif not self.release.is_set():
                self.release.wait(timeout=5.0)
            return dataclasses.replace(
                job,
                state=JobState.DONE,
                completed_at=utcnow(),
                answer=self.answer,
                reasoning="Re-decoded at native resolution, 4 fps.",
                confidence=0.88,
                evidence_clip="data/clips/fake.mp4",
            )
        finally:
            with self._lock:
                self.concurrent -= 1


class ScriptedBackend:
    """An LLM whose two answers are dictated by the test, not by a heuristic."""

    def __init__(self, verdict: str, answer: str = "provisional", calls: tuple[ToolCall, ...] = ()) -> None:
        self.verdict = verdict
        self.answer = answer
        self.calls = calls
        self.seen: list[LLMRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        if request.purpose is Purpose.GROUNDEDNESS:
            return LLMResponse(text=self.verdict, backend=self.name)
        return LLMResponse(text=self.answer, tool_calls=self.calls, backend=self.name)


class ExplodingBackend:
    """The ask model, down. The turn must survive it."""

    @property
    def name(self) -> str:
        return "exploding"

    @property
    def model(self) -> str:
        return "exploding"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("connection refused")


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


class AgentCase(unittest.TestCase):
    """Builds a real index, a real action server on a temp log, and a fake M4."""

    backend_factory = None  # type: ignore[var-annotated]
    settings_overrides: dict[str, object] = {}

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="m3-"))
        self.addCleanup(self._teardown)

        self.settings = dataclasses.replace(
            AgentSettings.from_config(),
            chat_log=self.tmp / "chats.jsonl",
            **self.settings_overrides,
        )
        self.index = build_index()
        self.index.ensure_ready()
        self.index.insert(fixture_chunks())

        self.actions = ActionServer(
            log_path=self.tmp / "actions.jsonl",
            clips_dir=self.tmp / "clips",
            clip_cutter=NullClipCutter(),
        )
        self.analyzer = FakeAnalyzer()
        self.jobs = JobRegistry(self.analyzer, self.settings)
        self.chat_log = ChatLog(self.settings.chat_log)
        self.tools = Toolbox(self.index, self.actions, self.jobs, self.settings)
        self.llm = (
            self.backend_factory() if self.backend_factory else StubBackend(self.settings)
        )
        self.agent = AskAgent(self.llm, self.tools, self.chat_log, self.settings)
        self.updates: list = []
        self.jobs.subscribe(self.updates.append)

    def _teardown(self) -> None:
        self.jobs.stop(timeout_s=2.0)
        self.index.close()

    def ask(self, question: str, **kwargs):
        """Ask over the fixture's day, so the default lookback cannot hide it."""
        kwargs.setdefault("t_from", SEGMENT_START - timedelta(hours=1))
        kwargs.setdefault("t_to", SEGMENT_START + timedelta(hours=1))
        return self.agent.ask(question, **kwargs)

    def wait_for_state(self, job_id: str, state: JobState, timeout: float = 5.0) -> DeepJob:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.jobs.job(job_id)
            if job is not None and job.state is state:
                return job
            time.sleep(0.01)
        self.fail(f"job {job_id} never reached {state}")


# --------------------------------------------------------------------------------------
# SPEC §4.2 — the escalation decision, and making it legible
# --------------------------------------------------------------------------------------


class TestEscalation(AgentCase):
    """The heart of the demo: the system knowing its own summary is not good enough."""

    def test_grounded_question_answers_from_the_index(self) -> None:
        result = self.ask(GROUNDED_QUESTION)

        self.assertIs(result.turn.grounded, True)
        self.assertFalse(result.escalation.escalated)
        self.assertEqual(result.escalation.triggers, ())
        self.assertIsNone(result.job)
        self.assertIsNone(result.turn.job_id)
        self.assertEqual(result.escalation.gate.badge, "answered from index")
        # It answered from something: the cited chunks are real ids from the corpus.
        self.assertTrue(result.turn.cited_chunk_ids)
        self.assertLessEqual(len(result.turn.cited_chunk_ids), TOP_N)

    def test_fine_visual_detail_escalates_on_both_mechanisms(self) -> None:
        result = self.ask(ESCALATING_QUESTION)

        self.assertIs(result.turn.grounded, False)
        self.assertTrue(result.escalation.escalated)
        # SPEC §4.2 runs both mechanisms. On this question they agree, and the record
        # says which fired rather than only that something did.
        self.assertIn(TRIGGER_GATE, result.escalation.triggers)
        self.assertIn(TRIGGER_TOOL, result.escalation.triggers)
        self.assertIsNotNone(result.job)
        self.assertEqual(result.turn.job_id, result.job.job_id)
        self.assertEqual(result.escalation.gate.badge, "not answerable from index")
        self.assertTrue(result.escalation.why, "the escalation must be printable")

    def test_the_verdict_is_persisted_on_the_turn(self) -> None:
        """§11.2's badge is re-rendered from the record, never recomputed."""
        self.ask(GROUNDED_QUESTION)
        self.ask(ESCALATING_QUESTION)

        turns = ChatLog(self.settings.chat_log).read().turns
        self.assertEqual([t.grounded for t in turns], [True, False])
        self.assertEqual([t.question for t in turns], [GROUNDED_QUESTION, ESCALATING_QUESTION])

    def test_retrieval_distance_is_never_the_signal(self) -> None:
        """SPEC §4.2: ANN always returns a plausible top-k. Ask about something absent."""
        result = self.ask("Did anyone read the licence plate on the blue lorry?")

        # Retrieval still returns its best guesses — that is exactly the trap.
        self.assertTrue(result.hits)
        # And the gate still says no.
        self.assertIs(result.turn.grounded, False)
        self.assertTrue(result.escalation.escalated)

    def test_escalated_range_comes_from_the_cited_chunks(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        cited = [h.record for h in result.hits]
        earliest = min(r.t_start for r in cited)
        latest = max(r.t_end for r in cited)

        span = (result.escalation.t_end - result.escalation.t_start).total_seconds()
        maximum = self.settings.deep_max_range_seconds
        self.assertLessEqual(span, maximum)
        # The tail always covers the cited chunks; the HEAD is what deep_range gives up
        # when the padded span exceeds agent.deep.max_range_seconds, because an
        # escalating question is nearly always about how something finished.
        self.assertGreaterEqual(result.escalation.t_end, latest)
        if span < maximum:
            self.assertLessEqual(result.escalation.t_start, earliest)
        else:
            self.assertEqual(
                result.escalation.t_start,
                result.escalation.t_end - timedelta(seconds=maximum),
            )
        # And that is the range the worker was actually handed.
        self.assertEqual(
            self.analyzer.submitted[0][:2],
            (result.escalation.t_start, result.escalation.t_end),
        )

    def test_timeout_is_stated_to_the_user(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        self.assertEqual(result.escalation.timeout_seconds, DEEP_TIMEOUT)


class TestGate(AgentCase):
    """The §4.2 gate on its own — one extra call, one yes/no, no distance anywhere."""

    def test_gate_runs_before_the_answer(self) -> None:
        backend = ScriptedBackend(verdict="YES\nthe captions state it")
        agent = AskAgent(backend, self.tools, self.chat_log, self.settings)
        agent.ask(GROUNDED_QUESTION)

        purposes = [r.purpose for r in backend.seen]
        self.assertEqual(purposes, [Purpose.GROUNDEDNESS, Purpose.ANSWER])
        # The gate is shown the reranked chunks, not the raw query.
        self.assertTrue(backend.seen[0].context)

    def test_gate_receives_the_reranked_context_only(self) -> None:
        backend = ScriptedBackend(verdict="NO\nnot recorded")
        agent = AskAgent(backend, self.tools, self.chat_log, self.settings)
        agent.ask(ESCALATING_QUESTION)
        self.assertLessEqual(len(backend.seen[0].context), TOP_N)

    def test_a_no_verdict_escalates_even_without_a_tool_call(self) -> None:
        backend = ScriptedBackend(verdict="NO\nthe captions do not record it", calls=())
        agent = AskAgent(backend, self.tools, self.chat_log, self.settings)
        result = agent.ask(GROUNDED_QUESTION)

        self.assertEqual(result.escalation.triggers, (TRIGGER_GATE,))
        self.assertIsNotNone(result.job)

    def test_a_tool_call_escalates_even_on_a_yes_verdict(self) -> None:
        """Mechanism 2 stands alone: the model may want the pixels regardless."""
        call = ToolCall(name="request_deep_analysis", arguments={"why": "needs the pixels"})
        backend = ScriptedBackend(verdict="YES\ncovered", calls=(call,))
        agent = AskAgent(backend, self.tools, self.chat_log, self.settings)
        result = agent.ask(GROUNDED_QUESTION)

        self.assertEqual(result.escalation.triggers, (TRIGGER_TOOL,))
        self.assertIs(result.turn.grounded, True)
        self.assertIsNotNone(result.job)
        self.assertIn("needs the pixels", result.escalation.why)

    def test_a_failed_gate_reports_unknown_not_grounded(self) -> None:
        agent = AskAgent(ExplodingBackend(), self.tools, self.chat_log, self.settings)
        result = agent.ask(GROUNDED_QUESTION)

        self.assertIsNone(result.turn.grounded)
        self.assertFalse(result.escalation.gate.ran)
        self.assertEqual(result.escalation.gate.badge, "groundedness unknown")
        # The turn survives a dead model; the index answer is still worth something.
        self.assertTrue(result.turn.provisional_answer)

    def test_gate_can_be_switched_off_and_says_so(self) -> None:
        settings = dataclasses.replace(self.settings, groundedness_gate=False)
        agent = AskAgent(
            StubBackend(settings),
            Toolbox(self.index, self.actions, self.jobs, settings),
            self.chat_log,
            settings,
        )
        result = agent.ask(ESCALATING_QUESTION)

        self.assertIsNone(result.turn.grounded)
        self.assertFalse(result.escalation.gate.ran)
        # Tool choice is untouched by the gate's switch — still one live mechanism.
        self.assertEqual(result.escalation.triggers, (TRIGGER_TOOL,))

    def test_verdict_parsing(self) -> None:
        self.assertEqual(_parse_verdict("YES\nthe captions state it")[0], True)
        self.assertEqual(_parse_verdict("no — not recorded")[0], False)
        self.assertEqual(_parse_verdict("No. The captions do not say.")[0], False)
        self.assertEqual(_parse_verdict("Yes")[0], True)
        # An unparseable verdict is unknown. A gate that fails open is not a gate.
        self.assertIsNone(_parse_verdict("perhaps, it depends")[0])
        self.assertIsNone(_parse_verdict("")[0])

    def test_reason_survives_onto_the_payload(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        payload = result.to_payload()
        self.assertTrue(payload["escalation"]["gate"]["reason"])
        self.assertEqual(payload["escalation"]["gate"]["grounded"], False)


# --------------------------------------------------------------------------------------
# SPEC §4.3 / CLAUDE.md invariant 4 — never block
# --------------------------------------------------------------------------------------


class TestNeverBlocks(AgentCase):
    def test_the_turn_returns_while_the_job_is_still_running(self) -> None:
        started = time.monotonic()
        result = self.ask(ESCALATING_QUESTION)
        elapsed = time.monotonic() - started

        # The fake worker is still sitting on release.wait() right now.
        self.assertLess(elapsed, 2.0, "the turn waited on deep analysis")
        self.assertIn(result.job.state, (JobState.QUEUED, JobState.RUNNING))
        self.assertTrue(self.analyzer.entered.wait(timeout=2.0))
        self.assertIsNone(result.job.completed_at)

        self.analyzer.release.set()
        done = self.wait_for_state(result.job.job_id, JobState.DONE)
        self.assertIn("rear doors are open", done.answer)

    def test_the_refinement_is_published_to_subscribers(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        self.analyzer.release.set()
        self.wait_for_state(result.job.job_id, JobState.DONE)

        states = [u.job.state for u in self.updates]
        self.assertIn(JobState.RUNNING, states)
        self.assertIs(states[-1], JobState.DONE)
        # Addressed to the turn that asked, so the UI can stack it under that card.
        self.assertEqual(self.updates[-1].turn_ids, (result.turn.turn_id,))
        # And it is an addition: the provisional answer is untouched.
        self.assertNotEqual(self.updates[-1].job.answer, result.turn.provisional_answer)

    def test_identical_ranges_are_deduped(self) -> None:
        """SPEC §4.3: an impatient user clicking twice must not queue the work twice."""
        first = self.ask(ESCALATING_QUESTION)
        second = self.ask(ESCALATING_QUESTION)

        self.assertEqual(second.dedupe_of, first.job.job_id)
        self.assertEqual(second.turn.job_id, first.job.job_id)
        self.assertEqual(len(self.analyzer.submitted), 1)
        # Both turns are waiting on the one job, and both get told.
        self.analyzer.release.set()
        self.wait_for_state(first.job.job_id, JobState.DONE)
        self.assertEqual(
            set(self.jobs.turns_for(first.job.job_id)),
            {first.turn.turn_id, second.turn.turn_id},
        )

    def test_one_deep_job_in_flight(self) -> None:
        """``agent.deep.max_inflight`` — one camera, one VLM, one job at a time."""
        self.analyzer.delay_s = 0.05
        ranges = [
            (SEGMENT_START + timedelta(seconds=i * 10), SEGMENT_START + timedelta(seconds=i * 10 + 5))
            for i in range(4)
        ]
        jobs = [
            self.jobs.request(t0, t1, "q", turn_id=f"turn-{i}")[0]
            for i, (t0, t1) in enumerate(ranges)
        ]
        for job in jobs:
            self.wait_for_state(job.job_id, JobState.DONE)
        self.assertEqual(self.analyzer.max_concurrent, self.settings.deep_max_inflight)

    def test_a_job_past_the_deadline_is_a_timeout_with_a_sentence(self) -> None:
        settings = dataclasses.replace(self.settings, deep_timeout_seconds=0.05)
        registry = JobRegistry(self.analyzer, settings)
        self.addCleanup(registry.stop, 2.0)
        self.analyzer.delay_s = 0.3

        job, _ = registry.request(SEGMENT_START, SEGMENT_START + timedelta(seconds=5), "q", turn_id="t")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current = registry.job(job.job_id)
            if current and current.state is JobState.TIMEOUT:
                break
            time.sleep(0.01)
        current = registry.job(job.job_id)
        self.assertIs(current.state, JobState.TIMEOUT)
        self.assertIn("timeout", (current.error or "").lower())

    def test_a_worker_that_cannot_start_fails_the_job_not_the_turn(self) -> None:
        registry = JobRegistry(UnavailableAnalyzer("M4 is not running"), self.settings)
        self.addCleanup(registry.stop, 2.0)
        tools = Toolbox(self.index, self.actions, registry, self.settings)
        agent = AskAgent(StubBackend(self.settings), tools, self.chat_log, self.settings)

        result = agent.ask(ESCALATING_QUESTION)
        self.assertTrue(result.escalation.escalated)
        self.assertTrue(result.turn.provisional_answer)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current = registry.job(result.job.job_id)
            if current and current.state is JobState.FAILED:
                break
            time.sleep(0.01)
        self.assertIs(registry.job(result.job.job_id).state, JobState.FAILED)


# --------------------------------------------------------------------------------------
# SPEC §4.1 — the tools
# --------------------------------------------------------------------------------------


class TestTools(AgentCase):
    def test_search_index_returns_wall_clock_ranges(self) -> None:
        hits = self.tools.search_index(
            "white van at the loading door",
            SEGMENT_START - timedelta(minutes=1),
            SEGMENT_START + timedelta(minutes=1),
        )
        self.assertTrue(hits)
        t_start, t_end = hits[0].time_range
        self.assertEqual(t_start.tzinfo, timezone.utc)
        self.assertEqual(hits[0].record.segment, SEGMENT)
        self.assertEqual(
            hits[0].record.pts_offset,
            (t_start - SEGMENT_START).total_seconds(),
            "pts_offset must still agree with wall clock after retrieval",
        )

    def test_actions_go_through_the_action_server(self) -> None:
        """CLAUDE.md invariant 5: no direct path to an effect."""
        t_start = SEGMENT_START + timedelta(seconds=7)
        t_end = t_start + timedelta(seconds=5)
        invocation = self.tools.dispatch(
            "save_clip",
            {"t_start": to_iso(t_start), "t_end": to_iso(t_end), "reason": "user asked"},
            turn_id="turn-1",
            hits=[],
        )
        self.assertTrue(invocation.ok)
        rows = self.actions.read_action_log(t_start - timedelta(hours=1), utcnow())
        self.assertEqual(len(rows), 1)
        self.assertIs(rows[0].action, ActionKind.SAVE_CLIP)
        # M3 acting for a user, not M5 acting on a standing task.
        self.assertIsNone(rows[0].task_id)

    def test_the_brakes_still_apply_to_the_agent(self) -> None:
        t_start = SEGMENT_START + timedelta(seconds=7)
        t_end = t_start + timedelta(seconds=5)
        args = {"t_start": to_iso(t_start), "t_end": to_iso(t_end)}
        self.tools.dispatch("raise_alert", args, turn_id="t1", hits=[])
        second = self.tools.dispatch("raise_alert", args, turn_id="t2", hits=[])

        # A suppressed action is a normal outcome, reported rather than raised.
        self.assertTrue(second.ok)
        self.assertFalse(second.result["fired"])
        self.assertIsNotNone(second.result["brake"])
        rows = self.actions.read_action_log(t_start - timedelta(hours=1), utcnow())
        self.assertEqual(len(rows), 1, "the second alert must not reach the log")

    def test_read_action_log_answers_why_did_you_alert(self) -> None:
        t_start = SEGMENT_START + timedelta(seconds=7)
        t_end = t_start + timedelta(seconds=5)
        self.actions.raise_alert(t_start, t_end, reason="van across the fire door")

        rows = self.tools.read_action_log(t_start - timedelta(minutes=5), utcnow())
        self.assertEqual([r.reason for r in rows], ["van across the fire door"])

    def test_deep_range_is_padded_and_clamped(self) -> None:
        hits = self.tools.search_index(
            ESCALATING_QUESTION,
            SEGMENT_START - timedelta(minutes=1),
            SEGMENT_START + timedelta(minutes=1),
        )
        t_start, t_end = self.tools.deep_range(hits)
        pad = timedelta(seconds=self.settings.deep_range_pad_seconds)
        maximum = self.settings.deep_max_range_seconds
        span = (t_end - t_start).total_seconds()

        # The end is always the cited tail plus the pad — deep_range clamps the *start*,
        # because the escalating question is nearly always about how something finished.
        self.assertEqual(t_end, max(h.record.t_end for h in hits) + pad)
        self.assertLessEqual(span, maximum)
        if span < maximum:
            # Unclamped: the padded start survives intact.
            self.assertEqual(t_start, min(h.record.t_start for h in hits) - pad)
        else:
            # Clamped: exactly `maximum` of footage, ending at the padded tail.
            self.assertEqual(t_start, t_end - timedelta(seconds=maximum))

    def test_deep_range_clamp_keeps_the_tail(self) -> None:
        """A range longer than agent.deep.max_range_seconds must give up its head, not
        its tail, and must never exceed what M4 will accept — the worker refuses an
        over-long range outright rather than truncating it."""
        maximum = self.settings.deep_max_range_seconds
        t0 = SEGMENT_START
        t1 = t0 + timedelta(seconds=maximum * 3)
        start, end = self.tools.deep_range([], {"t_start": to_iso(t0), "t_end": to_iso(t1)})
        self.assertEqual(end, t1, "the tail is where the answer lives")
        self.assertEqual((end - start).total_seconds(), maximum)

    def test_deep_range_prefers_the_models_own_arguments(self) -> None:
        t0 = SEGMENT_START + timedelta(seconds=10)
        t1 = SEGMENT_START + timedelta(seconds=20)
        start, end = self.tools.deep_range([], {"t_start": to_iso(t0), "t_end": to_iso(t1)})
        self.assertEqual((start, end), (t0, t1))

    def test_deep_range_survives_a_model_writing_nonsense(self) -> None:
        start, end = self.tools.deep_range([], {"t_start": "half past three", "t_end": None})
        self.assertLess(start, end)
        # The fallback window is itself subject to the clamp — settings keeps the two
        # aligned so this is normally an identity, but asserting the min() means the
        # test states the real rule instead of a coincidence between two config values.
        expected = min(
            self.settings.deep_fallback_window_seconds, self.settings.deep_max_range_seconds
        )
        self.assertAlmostEqual((end - start).total_seconds(), expected, places=1)

    def test_tool_schemas_describe_the_escalation_path(self) -> None:
        """SPEC §4.2 mechanism 2 lives in the description string, so assert it exists."""
        from services.agent import TOOL_SCHEMAS

        names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        self.assertEqual(
            names,
            {
                "search_index",
                "request_deep_analysis",
                "read_action_log",
                "save_clip",
                "raise_alert",
                "file_ticket",
            },
        )
        deep = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "request_deep_analysis")
        self.assertIn("job_id", deep["function"]["description"])
        self.assertIn("never blocks", deep["function"]["description"])


# --------------------------------------------------------------------------------------
# SPEC §11.4 — chat history that outlives the turn
# --------------------------------------------------------------------------------------


class TestChatLog(AgentCase):
    def test_a_refinement_survives_a_reload(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        self.chat_log.append_job(
            dataclasses.replace(
                result.job, state=JobState.DONE, completed_at=utcnow(), answer="refined"
            )
        )

        reloaded = ChatLog(self.settings.chat_log).read()
        turn = reloaded.turns[-1]
        self.assertEqual(turn.job_id, result.job.job_id)
        # The turn persists the JOB, not the text: the refined answer is rebuilt from it.
        self.assertEqual(reloaded.jobs[turn.job_id].answer, "refined")
        self.assertIs(reloaded.jobs[turn.job_id].state, JobState.DONE)
        self.assertNotEqual(turn.provisional_answer, "refined")

    def test_the_log_is_append_only(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        self.chat_log.append_job(dataclasses.replace(result.job, state=JobState.RUNNING))
        self.chat_log.append_job(dataclasses.replace(result.job, state=JobState.DONE))

        rows = [
            json.loads(line)
            for line in self.settings.chat_log.read_text(encoding="utf-8").splitlines()
        ]
        kinds = [row["kind"] for row in rows]
        self.assertEqual(kinds.count("turn"), 1)
        self.assertEqual(kinds.count("job"), 3)  # queued, running, done — nothing rewritten
        # The reader keeps the last state per job.
        self.assertIs(ChatLog(self.settings.chat_log).read().jobs[result.job.job_id].state, JobState.DONE)

    def test_a_corrupt_row_costs_one_row_not_the_pane(self) -> None:
        self.ask(GROUNDED_QUESTION)
        with self.settings.chat_log.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        self.ask(ESCALATING_QUESTION)

        self.assertEqual(len(ChatLog(self.settings.chat_log).read().turns), 2)

    def test_unfinished_jobs_are_reported(self) -> None:
        result = self.ask(ESCALATING_QUESTION)
        unfinished = ChatLog(self.settings.chat_log).unfinished_jobs()
        self.assertEqual([j.job_id for j in unfinished], [result.job.job_id])

    def test_every_payload_timestamp_is_z_suffixed_utc(self) -> None:
        payload = self.ask(ESCALATING_QUESTION).to_payload()
        self.assertTrue(payload["ts"].endswith("Z"))
        self.assertTrue(payload["job"]["t_start"].endswith("Z"))
        self.assertTrue(payload["escalation"]["t_start"].endswith("Z"))
        self.assertEqual(from_iso(payload["ts"]).tzinfo, timezone.utc)


# --------------------------------------------------------------------------------------
# SPEC §10 D5 / §11.3 — register_task
# --------------------------------------------------------------------------------------


class TestTaskRegistry(AgentCase):
    def test_the_seed_file_is_the_cold_start(self) -> None:
        registry = SeedTaskRegistry(actions=self.actions)
        ids = {task.task_id for task in registry.tasks()}
        self.assertIn("fire-door-blocked", ids)

    def test_registering_a_task(self) -> None:
        registry = SeedTaskRegistry(actions=self.actions)
        task = task_from_payload(
            {
                "task_id": "rear-door-open",
                "describe": "a van with its rear doors open at the loading bay",
                "window": 60,
                "action": "save_clip",
                "cooldown": 120,
                "active": "00:00-24:00",
            }
        )
        registered = registry.register(task)
        self.assertEqual(registered.task_id, "rear-door-open")
        # M5 embeds `describe` once at registration (SPEC §6.2) — not us.
        self.assertEqual(registered.embedding, [])
        self.assertIn("rear-door-open", {t.task_id for t in registry.tasks()})

    def test_duplicate_ids_are_refused(self) -> None:
        from services.agent import DuplicateTaskError

        registry = SeedTaskRegistry(actions=self.actions)
        with self.assertRaises(DuplicateTaskError):
            registry.register(
                task_from_payload(
                    {
                        "task_id": "fire-door-blocked",
                        "describe": "again",
                        "action": "raise_alert",
                    }
                )
            )

    def test_monitor_state_reads_last_fired_from_the_action_log(self) -> None:
        from shared.schema import Task

        registry = SeedTaskRegistry(actions=self.actions)
        task = next(t for t in registry.tasks() if t.task_id == "fire-door-blocked")
        self.assertIsInstance(task, Task)
        self.actions.raise_alert(
            SEGMENT_START, SEGMENT_START + timedelta(seconds=5), task=task, reason="staged"
        )
        state = registry.monitor_state()
        row = next(r for r in state["tasks"] if r["task_id"] == "fire-door-blocked")
        self.assertTrue(row["last_fired_ts"].endswith("Z"))
        self.assertEqual(row["cooldown_seconds"], task.cooldown)

    def test_bad_payloads_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            task_from_payload({"describe": "no id", "action": "save_clip"})
        with self.assertRaises(ValueError):
            task_from_payload({"task_id": "x", "describe": "y", "action": "detonate"})


# --------------------------------------------------------------------------------------
# The server — the contract ui/static/data.js already expects
# --------------------------------------------------------------------------------------


class TestServer(AgentCase):
    def setUp(self) -> None:
        super().setUp()
        self.app = AgentApp(
            agent=self.agent,
            index=self.index,
            actions=self.actions,
            jobs=self.jobs,
            chat_log=self.chat_log,
            tasks=SeedTaskRegistry(actions=self.actions),
            hub=WebSocketHub(),
            settings=self.settings,
            clip_cutter=NullClipCutter(),
        )
        self.server = AskServer(self.app, host="127.0.0.1", port=0)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.port}"

    def get(self, path: str) -> tuple[int, dict]:
        with urllib.request.urlopen(self.base + path, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_config(self) -> None:
        status, body = self.get("/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(body["ui"]["display_timezone"], config.get("ui.display_timezone"))

    def test_chunks(self) -> None:
        t_from = to_iso(SEGMENT_START - timedelta(minutes=1))
        t_to = to_iso(SEGMENT_START + timedelta(minutes=1))
        status, body = self.get(f"/api/chunks?t_from={t_from}&t_to={t_to}")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["chunks"]), len(CAPTIONS))
        # In time order, and carrying the §3.1 join.
        starts = [c["t_start"] for c in body["chunks"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(body["chunks"][0]["segment"], SEGMENT)

    def test_ask_grounded(self) -> None:
        status, body = self.post("/api/ask", {"question": GROUNDED_QUESTION})
        self.assertEqual(status, 200)
        # ChatTurn.to_dict() keys, unchanged.
        for key in ("turn_id", "ts", "question", "provisional_answer", "grounded",
                    "cited_chunk_ids", "job_id", "latency_s"):
            self.assertIn(key, body)
        self.assertIs(body["grounded"], True)
        self.assertIsNone(body["job"])
        self.assertIsNone(body["dedupe_of"])

    def test_ask_escalates_without_blocking(self) -> None:
        started = time.monotonic()
        status, body = self.post("/api/ask", {"question": ESCALATING_QUESTION})
        elapsed = time.monotonic() - started

        self.assertEqual(status, 200)
        self.assertLess(elapsed, 3.0)
        self.assertIs(body["grounded"], False)
        self.assertIsNotNone(body["job"])
        self.assertEqual(body["job"]["state"], JobState.QUEUED.value)
        self.assertEqual(body["job_id"], body["job"]["job_id"])
        self.assertTrue(body["escalation"]["escalated"])
        self.analyzer.release.set()

    def test_ask_dedupe_is_reported(self) -> None:
        _, first = self.post("/api/ask", {"question": ESCALATING_QUESTION})
        _, second = self.post("/api/ask", {"question": ESCALATING_QUESTION})
        self.assertEqual(second["dedupe_of"], first["job"]["job_id"])
        self.analyzer.release.set()

    def test_ask_requires_a_question(self) -> None:
        status, body = self.post("/api/ask", {"question": "   "})
        self.assertEqual(status, 400)
        self.assertIn("detail", body)

    def test_history_returns_turns_and_jobs(self) -> None:
        self.post("/api/ask", {"question": ESCALATING_QUESTION})
        self.analyzer.release.set()
        status, body = self.get("/api/chat/history")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["turns"]), 1)
        job_id = body["turns"][0]["job_id"]
        self.assertIn(job_id, body["jobs"])

    def test_tasks_and_register_task(self) -> None:
        status, body = self.get("/api/tasks")
        self.assertEqual(status, 200)
        self.assertTrue(body["tasks"])

        payload = {
            "task_id": "rear-door-open",
            "describe": "a van with its rear doors open",
            "window": 60,
            "action": "save_clip",
            "cooldown": 120,
            "active": "00:00-24:00",
        }
        status, created = self.post("/api/register_task", payload)
        self.assertEqual(status, 200)
        self.assertEqual(created["task_id"], "rear-door-open")

        status, conflict = self.post("/api/register_task", payload)
        self.assertEqual(status, 409)
        self.assertIn("already registered", conflict["detail"])

    def test_monitor_state_shape(self) -> None:
        status, body = self.get("/api/monitor/state")
        self.assertEqual(status, 200)
        self.assertTrue(body["generated_at"].endswith("Z"))
        row = body["tasks"][0]
        for key in ("task_id", "state", "stage1", "stage2", "stage3", "cooldown_seconds"):
            self.assertIn(key, row)

    def test_actions_endpoint_reads_the_same_rows_as_the_agent(self) -> None:
        self.actions.raise_alert(
            SEGMENT_START, SEGMENT_START + timedelta(seconds=5), reason="staged"
        )
        status, body = self.get("/api/actions")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["entries"]), 1)
        self.assertEqual(body["entries"][0]["reason"], "staged")

    def test_video_requires_a_range_never_a_filename(self) -> None:
        """Invariant 3: footage is fetched by time range, never by filename. Asking
        without one is a 400, not a helpful default."""
        # `get` raises on any non-2xx, so the 400 arrives as an exception — same shape
        # as test_video_reports_a_hole_rather_than_short_footage below.
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/video")
        self.assertEqual(caught.exception.code, 400)
        detail = json.loads(caught.exception.read().decode("utf-8"))["detail"]
        self.assertIn("range", detail)

    def test_video_reports_a_hole_rather_than_short_footage(self) -> None:
        t_from = to_iso(SEGMENT_START)
        t_to = to_iso(SEGMENT_START + timedelta(seconds=5))
        try:
            self.get(f"/api/video?t_from={t_from}&t_to={t_to}")
            self.fail("expected a 404 for footage that was never recorded")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            self.assertIn("no footage", json.loads(exc.read().decode("utf-8"))["detail"])

    def test_unknown_endpoint(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/nonsense")
        self.assertEqual(caught.exception.code, 404)

    def test_the_ui_is_served_from_the_same_origin(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=10) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"<html", response.read().lower())

    def test_static_paths_cannot_escape_the_ui_directory(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/../config/settings.yaml")
        self.assertEqual(caught.exception.code, 404)


# --------------------------------------------------------------------------------------
# The WebSocket — SPEC §4.3's third arrow
# --------------------------------------------------------------------------------------


class TestWebSocket(TestServer):
    """Speaks RFC 6455 by hand, because that is what the browser will do."""

    def connect(self) -> socket.socket:
        sock = socket.create_connection(("127.0.0.1", self.server.port), timeout=10)
        key = client_key()
        sock.sendall(
            (
                "GET /ws HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{self.server.port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode("ascii")
        )
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = sock.recv(1)
            if not chunk:
                self.fail("server closed during the handshake")
            header += chunk
        self.assertIn(b"101", header.split(b"\r\n")[0])
        self.assertIn(accept_key(key).encode("ascii"), header)
        self.addCleanup(sock.close)
        return sock

    def test_accept_key_matches_the_rfc_example(self) -> None:
        # RFC 6455 §1.3's worked example. If this drifts, no browser will connect.
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_a_refinement_arrives_over_the_socket(self) -> None:
        sock = self.connect()
        _, body = self.post("/api/ask", {"question": ESCALATING_QUESTION})
        self.analyzer.release.set()

        message = self._await_type(sock, "refinement", state=JobState.DONE.value)
        self.assertEqual(message["turn_id"], body["turn_id"])
        self.assertEqual(message["job"]["job_id"], body["job"]["job_id"])
        self.assertIn("rear doors are open", message["job"]["answer"])
        # Appended, never substituted: the provisional text is not what came back.
        self.assertNotEqual(message["job"]["answer"], body["provisional_answer"])

    def test_the_socket_answers_a_ping(self) -> None:
        sock = self.connect()
        sock.sendall(mask_frame(b"", opcode=0x9))
        opcode, _ = self._read_frame(sock)
        self.assertEqual(opcode, 0xA)

    def test_monitor_state_and_action_messages_have_the_shape_the_ui_reads(self) -> None:
        sock = self.connect()
        self.app.publish_monitor_state({"generated_at": to_iso(utcnow()), "tasks": []})
        message = self._await_type(sock, "monitor_state")
        self.assertIn("tasks", message["state"])

        entry = self.actions.raise_alert(
            SEGMENT_START, SEGMENT_START + timedelta(seconds=5), reason="staged"
        ).entry
        self.app.publish_action(entry.to_dict())
        message = self._await_type(sock, "action")
        self.assertEqual(message["entry"]["reason"], "staged")

    # -- frame reading ---------------------------------------------------------------

    def _read_frame(self, sock: socket.socket) -> tuple[int, bytes]:
        header = self._recv(sock, 2)
        opcode = header[0] & 0x0F
        length = header[1] & 0x7F
        if length == 126:
            length = int.from_bytes(self._recv(sock, 2), "big")
        elif length == 127:
            length = int.from_bytes(self._recv(sock, 8), "big")
        return opcode, self._recv(sock, length) if length else b""

    def _recv(self, sock: socket.socket, count: int) -> bytes:
        buffer = b""
        while len(buffer) < count:
            chunk = sock.recv(count - len(buffer))
            if not chunk:
                self.fail("socket closed mid-frame")
            buffer += chunk
        return buffer

    def _await_type(self, sock: socket.socket, kind: str, state: str | None = None) -> dict:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            opcode, payload = self._read_frame(sock)
            if opcode != 0x1:
                continue
            message = json.loads(payload.decode("utf-8"))
            if message.get("type") != kind:
                continue
            if state is not None and message.get("job", {}).get("state") != state:
                continue
            return message
        self.fail(f"no {kind} message arrived")


# --------------------------------------------------------------------------------------
# Wiring — build_app() must produce a working agent on this box, today
# --------------------------------------------------------------------------------------


class TestWiring(unittest.TestCase):
    def test_build_app_uses_the_configured_backends(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="m3-wiring-"))
        # backend pinned to the stub, NOT inherited from settings.yaml: SPEC §10 D3 is
        # resolved to `nim`, so an inherited backend would fire real HTTP at the serving
        # model. CLAUDE.md forbids tests touching the real endpoint (it contends with
        # ingest), and a unit test whose meaning changes when config moves is not testing
        # what it claims to. The wiring under test is the assembly, not the model choice.
        settings = dataclasses.replace(
            AgentSettings.from_config(), chat_log=tmp / "chats.jsonl", backend="stub"
        )
        index = build_index()
        index.ensure_ready()
        index.insert(fixture_chunks())
        self.addCleanup(index.close)

        app = build_app(
            settings=settings,
            index=index,
            actions=ActionServer(log_path=tmp / "actions.jsonl", clip_cutter=NullClipCutter()),
            analyzer=FakeAnalyzer(),
            clip_cutter=NullClipCutter(),
        )
        self.addCleanup(app.stop)
        app.start()

        status, payload = app.post_ask({"question": GROUNDED_QUESTION})
        self.assertEqual(int(status), 200)
        self.assertIs(payload["grounded"], True)
        self.assertEqual(payload["escalation"]["gate"]["badge"], "answered from index")

    def test_settings_do_not_invent_a_model(self) -> None:
        """SPEC §10 D3 is resolved, so this pins the GUARD, not the configured value.

        If agent.model is ever cleared, reading it must name the decision rather than
        quietly guessing a model the endpoint was never told to serve.
        """
        root = config.load()
        previous = root["agent"]["model"]
        root["agent"]["model"] = None
        try:
            with self.assertRaises(config.ConfigError) as caught:
                _ = AgentSettings.from_config().model
            self.assertIn("agent.model", str(caught.exception))
        finally:
            root["agent"]["model"] = previous

    def test_the_resolved_model_is_read_from_config(self) -> None:
        """D3 landed: M3 and the VLM deliberately share one served model, so that the
        two surfaces can never end up talking to different ones."""
        settings = AgentSettings.from_config()
        self.assertEqual(settings.model, str(config.get("agent.model")))
        self.assertEqual(str(config.get("agent.model")), str(config.get("vlm.model")))

    def test_pending_settings_are_all_absent_from_the_yaml(self) -> None:
        """A pending key that has since reached settings.yaml must be dropped from the
        table — otherwise the fallback shadows nothing and confuses the next person to
        go looking for the dial. Same guard as tests/test_ingest.py, deliberately: one
        convention for the whole repo, not one per module."""
        from services.agent import PENDING_SETTINGS

        for key in PENDING_SETTINGS:
            self.assertTrue(key.startswith("agent."), key)
            with self.subTest(setting=key):
                self.assertIsNone(
                    config.get(key, None),
                    f"{key} is now in settings.yaml; drop it from _PENDING",
                )


if __name__ == "__main__":
    unittest.main()


class TestInlineToolCalls(unittest.TestCase):
    """The served model emits tool calls inside `content`, not in `tool_calls`.

    Observed live from gemma-4-E2B via llama-server: llama.cpp does not normalise that
    shape, so without this the serialized call is rendered to the user as the provisional
    answer — the pane SPEC §11.2 calls the most important pixel in the build.
    """

    RAW = (
        "request_deep_analysis{question:<|>was the door open?<|>,"
        "t_start:<|>2026-08-15T17:32:06Z<|>,t_end:<|>2026-08-15T17:33:31Z<|>,"
        "why:<|>The captions do not record whether the door was open.<|>}"
    )

    def test_an_inline_call_is_lifted_out_of_the_answer_text(self) -> None:
        from services.agent.llm import _extract_inline_tool_call

        calls, text = _extract_inline_tool_call(self.RAW)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "request_deep_analysis")
        self.assertEqual(calls[0].arguments["t_start"], "2026-08-15T17:32:06Z")
        # The user must never see the serialized call.
        self.assertNotIn("<|>", text)
        self.assertNotIn("request_deep_analysis{", text)

    def test_the_models_reason_becomes_the_provisional_text(self) -> None:
        """`why` is the model's stated reason for escalating, which is exactly what the
        Ask pane should show while the deep job runs."""
        from services.agent.llm import _extract_inline_tool_call

        _, text = _extract_inline_tool_call(self.RAW)
        self.assertIn("do not record whether the door was open", text)

    def test_ordinary_prose_is_untouched(self) -> None:
        from services.agent.llm import _extract_inline_tool_call

        answer = "The person is wearing an orange shirt and sitting in a gaming chair."
        calls, text = _extract_inline_tool_call(answer)
        self.assertEqual(calls, [])
        self.assertEqual(text, answer)

    def test_merely_naming_a_tool_is_not_a_call(self) -> None:
        """Conservative on purpose: a model discussing its tools keeps its prose."""
        from services.agent.llm import _extract_inline_tool_call

        calls, _ = _extract_inline_tool_call("I could use request_deep_analysis for this.")
        self.assertEqual(calls, [])

    def test_structured_tool_calls_still_win(self) -> None:
        """When the backend does it properly, the inline fallback must not double-count."""
        from services.agent.llm import OpenAICompatBackend

        body = {
            "choices": [{"message": {
                "content": self.RAW,
                "tool_calls": [{"id": "c1", "function": {
                    "name": "request_deep_analysis", "arguments": '{"t_start": "x"}'}}],
            }}]
        }
        text, calls = OpenAICompatBackend._parse_message(body)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call_id, "c1")
        self.assertEqual(text, self.RAW)


class TestTaskCrud(unittest.TestCase):
    """Standing tasks are CRUD, not create-only — SPEC §11.3's pane has to be usable
    after the third time you get a task's wording wrong."""

    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="crud-"))
        seed = tmp / "tasks.yaml"
        seed.write_text(
            "tasks:\n"
            "  - task_id: seeded-one\n"
            "    describe: a vehicle stopped in front of the fire door\n"
            "    window: 120\n"
            "    action: raise_alert\n"
            "    cooldown: 300\n"
            "    active: \"00:00-24:00\"\n",
            encoding="utf-8",
        )
        self.registry = SeedTaskRegistry(tasks_file=seed)
        self.app = build_app(
            settings=dataclasses.replace(
                AgentSettings.from_config(), chat_log=tmp / "chats.jsonl", backend="stub"
            ),
            index=build_index(),
            actions=ActionServer(log_path=tmp / "actions.jsonl", clip_cutter=NullClipCutter()),
            analyzer=FakeAnalyzer(),
            clip_cutter=NullClipCutter(),
            tasks=self.registry,
        )
        self.addCleanup(self.app.stop)

    def _ids(self) -> list[str]:
        return [t["task_id"] for t in self.app.get_tasks()[1]["tasks"]]

    def test_delete_removes_a_registered_task(self) -> None:
        self.app.post_register_task(
            {"task_id": "temp", "describe": "someone at the door", "window": 30,
             "action": "save_clip", "cooldown": 60, "active": "00:00-24:00"}
        )
        self.assertIn("temp", self._ids())
        status, payload = self.app.delete_task("temp")
        self.assertEqual(int(status), 200)
        self.assertEqual(payload["deleted"]["task_id"], "temp")
        self.assertNotIn("temp", self._ids())

    def test_a_seeded_task_can_be_deleted_for_this_process(self) -> None:
        """config/tasks.yaml is the cold-start truth and is not rewritten, so the
        removal is a tombstone — it must still disappear from the pane."""
        self.assertIn("seeded-one", self._ids())
        self.app.delete_task("seeded-one")
        self.assertNotIn("seeded-one", self._ids())

    def test_deleting_twice_is_a_404_not_a_silent_success(self) -> None:
        """'Delete that one' quietly doing nothing is worse than an error: the operator
        walks away believing the task is gone."""
        self.app.delete_task("seeded-one")
        status, _ = self.app.delete_task("seeded-one")
        self.assertEqual(int(status), 404)

    def test_delete_leaves_the_action_log_alone(self) -> None:
        """SPEC §6.4: 'why did you alert at 21:11?' must keep working for a task that no
        longer exists. Tidying a task list must never rewrite history."""
        t0 = SEGMENT_START
        self.app.actions.fire(
            ActionKind.RAISE_ALERT, t0, t0 + timedelta(seconds=5), task_id="seeded-one",
            reason="staged",
        )
        before = self.app.get_actions(None, None)[1]["entries"]
        self.assertTrue(before)
        self.app.delete_task("seeded-one")
        after = self.app.get_actions(None, None)[1]["entries"]
        self.assertEqual(len(after), len(before))
        self.assertEqual(after[0]["task_id"], "seeded-one")

    def test_patch_edits_in_place(self) -> None:
        status, payload = self.app.patch_task("seeded-one", {"cooldown": 900})
        self.assertEqual(int(status), 200)
        self.assertEqual(payload["cooldown"], 900)
        self.assertEqual(payload["describe"], "a vehicle stopped in front of the fire door")

    def test_patch_refuses_to_rename(self) -> None:
        """task_id keys the cooldown and dedupe brakes; moving it orphans the history of
        whatever event is in flight."""
        status, payload = self.app.patch_task("seeded-one", {"task_id": "other"})
        self.assertEqual(int(status), 400)
        self.assertIn("cooldown", payload["detail"])

    def test_patch_rejects_unknown_fields_rather_than_ignoring_them(self) -> None:
        status, payload = self.app.patch_task("seeded-one", {"colour": "red"})
        self.assertEqual(int(status), 400)
        self.assertIn("colour", payload["detail"])

    def test_patch_and_delete_report_a_missing_task(self) -> None:
        self.assertEqual(int(self.app.patch_task("ghost", {"cooldown": 5})[0]), 404)
        self.assertEqual(int(self.app.delete_task("ghost")[0]), 404)

    def test_an_empty_patch_is_refused(self) -> None:
        self.assertEqual(int(self.app.patch_task("seeded-one", {})[0]), 400)

    def test_a_deleted_id_can_be_registered_again(self) -> None:
        """Delete then re-create must not trip the duplicate-id guard."""
        self.app.delete_task("seeded-one")
        status, _ = self.app.post_register_task(
            {"task_id": "seeded-one", "describe": "something else entirely", "window": 10,
             "action": "save_clip", "cooldown": 30, "active": "00:00-24:00"}
        )
        self.assertEqual(int(status), 200)
        self.assertIn("seeded-one", self._ids())
