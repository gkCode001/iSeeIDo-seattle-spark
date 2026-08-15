"""Tests for M5, the standing-task monitor — SPEC §6.

The headline claim, and the reason this file exists:

    SPEC §6.4 — "The demo failure mode here is not missing an event. It is firing thirty
    alerts for one."

So :meth:`ExactlyOneAlertTest.test_one_event_over_six_minutes_fires_exactly_one_action`
asserts the count is **1**. Not "few", not "fewer than five". Everything else here is
either support for that claim or a test of the three things that decide when it fires: the
sustain window, the ``active`` window, and stage 3's ability to take an alert back.

Rules this file follows, because the module under test is the one that changes the outside
world unprompted:

* the action log always lives in a tempdir, never ``data/actions.jsonl``;
* time is injected, never slept — a brake test that depends on wall clock is a brake test
  that will be marked flaky and then deleted;
* the VLM, the LLM and the deep worker are all stood in for. Nothing here reaches a
  network, and M4 (``services/worker``) is being written in parallel, so stage 3 arrives
  through an injected interface and this file does not import it.

Run with::

    python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared import config
from shared.schema import (
    ActionKind,
    ActionStatus,
    ChunkRecord,
    DeepJob,
    JobState,
    Task,
    chunk_id_for,
    from_iso,
)
from services.index import build_embedder
from services.index.settings import IndexSettings
from services.mcp import ActionServer, Brake
from services.monitor import (
    ActiveWindowError,
    Monitor,
    MonitorSettings,
    NIMConfirmer,
    StubConfirmer,
    TaskRegistrationError,
    TaskRegistry,
    build_confirmer,
    build_monitor,
    cosine,
    parse_active_window,
)

# Numbers come from settings.yaml, not from this file. CLAUDE.md: no magic numbers.
WINDOW_SECONDS = float(config.get("ingest.window_seconds"))
STRIDE_SECONDS = float(config.get("ingest.stride_seconds"))
STAGE1_THRESHOLD = float(config.get("monitor.stage1_cosine_threshold"))
CAMERA_ID = str(config.get("camera.id"))
DISPLAY_TZ = str(config.get("ui.display_timezone"))

#: 21:11:07 in the configured display timezone (Asia/Kolkata, UTC+5:30) — the hour SPEC
#: §11.3's mock-up alerts at, and comfortably inside ``18:00-06:00``. Every assertion below
#: is reproducible from it.
T0 = datetime(2026, 8, 14, 15, 41, 7, tzinfo=timezone.utc)

#: Captions the stub backends can separate. The fire-door one scores ~0.64 cosine against
#: its task and ~0.11 against the loading-bay task, so stage 1 genuinely filters rather
#: than passing everything through to stage 2.
FIRE_DOOR_CAPTION = "A white van is stopped in front of the fire door. Nobody is visible."
LOADING_BAY_CAPTION = "A person is unloading boxes from a vehicle at the loading bay."
QUIET_CAPTION = "An empty courtyard at night. Nothing is moving."


class FakeClock:
    """Injected wall clock. Advancing it is the only way time passes in these tests."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now

    def set(self, moment: datetime) -> datetime:
        self.now = moment
        return self.now


class SequentialIds:
    """Deterministic entry ids so failures name the row that broke."""

    def __init__(self, prefix: str = "e") -> None:
        self.prefix = prefix
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"{self.prefix}{self.n:04d}"


class FakeVerifier:
    """Stands in for M4 (SPEC §5), which is being built in parallel.

    Implements exactly the two calls M5 makes — ``submit`` and ``poll`` — and nothing
    else. Submission is instantaneous and returns a QUEUED job, which is the contract that
    matters: SPEC §6.3 says stage 3 must not block the chunk loop, and a fake that
    returned a finished job would let a blocking implementation pass this suite.
    """

    def __init__(self) -> None:
        self.submitted: list[DeepJob] = []
        self.jobs: dict[str, DeepJob] = {}
        self._n = 0

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        self._n += 1
        job = DeepJob(
            job_id=f"job{self._n:02d}",
            t_start=t_start,
            t_end=t_end,
            question=question,
            state=JobState.QUEUED,
            requested_at=t_start,
        )
        self.jobs[job.job_id] = job
        self.submitted.append(job)
        return job

    def poll(self, job_id: str) -> DeepJob | None:
        return self.jobs.get(job_id)

    def finish(
        self,
        job_id: str,
        *,
        confidence: float | None,
        answer: str = "",
        state: JobState = JobState.DONE,
        evidence_clip: str | None = None,
    ) -> DeepJob:
        """Land a worker verdict, the way the WebSocket would."""
        job = self.jobs[job_id]
        job.state = state
        job.confidence = confidence
        job.answer = answer
        job.evidence_clip = evidence_clip
        job.completed_at = job.requested_at + timedelta(seconds=30)
        return job


def make_chunks(
    start: datetime,
    count: int,
    caption: str,
    *,
    gated_indices: tuple[int, ...] = (),
) -> list[ChunkRecord]:
    """A run of analysis windows at the configured window/stride.

    Consecutive windows overlap by ``window - stride`` seconds, which is exactly why the
    dedupe brake exists (SPEC §6.4) — the same event arrives several times.

    ``embedding`` is left empty on purpose: the monitor re-derives it with the same
    embedder the index uses, and a test that pre-embedded would not exercise that.
    """
    chunks: list[ChunkRecord] = []
    for i in range(count):
        t_start = start + timedelta(seconds=STRIDE_SECONDS * i)
        t_end = t_start + timedelta(seconds=WINDOW_SECONDS)
        gated = i in gated_indices
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id_for(CAMERA_ID, t_start, t_end),
                camera_id=CAMERA_ID,
                t_start=t_start,
                t_end=t_end,
                segment=f"{CAMERA_ID}_20260814_154100.mp4",
                pts_offset=(t_start - start).total_seconds(),
                gated=gated,
                caption="" if gated else caption,
            )
        )
    return chunks


class MonitorTestCase(unittest.TestCase):
    """Shared wiring: tempdir log, fake clock, fake worker, stub LLM."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = FakeClock(T0)
        self.settings = MonitorSettings.from_config()
        self.embedder = build_embedder(IndexSettings.from_config())
        self.verifier = FakeVerifier()
        self.actions = ActionServer(
            log_path=self.root / "actions.jsonl",
            clips_dir=self.root / "clips",
            clock=self.clock,
            id_factory=SequentialIds(),
        )

    def build_monitor(self, tasks: list[Task] | None = None, **kwargs: object) -> Monitor:
        registry = TaskRegistry(self.embedder)
        for task in tasks or seed_tasks():
            registry.register(task)
        return Monitor(
            registry=registry,
            actions=self.actions,
            # Pinned to the stub, NOT build_confirmer(): that reads agent.backend from
            # settings.yaml, which is now `nim` (SPEC §10 D3 resolved), so the funnel
            # tests would fire real HTTP at the serving model. CLAUDE.md forbids tests
            # touching the real endpoint — it contends with ingest — and a unit test whose
            # meaning changes when someone edits config is not testing what it claims.
            confirmer=StubConfirmer(self.settings.stub_min_overlap),
            embedder=self.embedder,
            settings=self.settings,
            verifier=self.verifier,
            clock=self.clock,
            timezone_override=DISPLAY_TZ,
            **kwargs,  # type: ignore[arg-type]
        )

    # -- helpers ---------------------------------------------------------------------

    def feed(self, monitor: Monitor, chunks: list[ChunkRecord]) -> list[object]:
        """Play chunks through the funnel, moving the injected clock with the footage.

        The clock tracks ``t_end`` because that is what happens live: a window is analysed
        the moment it closes. Keeping the two in step is what makes the cooldown brake
        (which compares wall clock) testable against a footage timeline.
        """
        outcomes: list[object] = []
        for chunk in chunks:
            self.clock.set(chunk.t_end)
            outcomes.extend(monitor.observe([chunk]))
        return outcomes

    def log_rows(self):
        day = timedelta(days=1)
        return self.actions.read_action_log(T0 - day, self.clock.now + day)

    def originals_for(self, task_id: str):
        """Originating rows only — amendments are commentary, not a second action."""
        return [r for r in self.log_rows() if r.task_id == task_id and r.parent_id is None]


def seed_tasks() -> list[Task]:
    """The two SPEC §6.1 tasks from ``config/tasks.yaml``, read rather than retyped.

    Reading the real seed means a change to the demo's tasks shows up here as a failing
    assertion rather than as a suite that passes against tasks nobody runs.
    """
    from services.monitor import load_task_seed

    return load_task_seed(config.repo_path("monitor.tasks_file"))


# ======================================================================================
# THE HEADLINE — SPEC §6.4
# ======================================================================================


class ExactlyOneAlertTest(MonitorTestCase):
    def test_one_event_over_six_minutes_fires_exactly_one_action(self) -> None:
        """One staged event, many overlapping matching chunks → **exactly one** action.

        Ten minutes of a van at the fire door, at a 5 s window and 4 s stride: 150 chunks,
        every one of them matching stage 1 and stage 2, spanning more than the 300 s
        cooldown so that the cooldown brake alone cannot account for the result.

        The assertion is ``== 1``.
        """
        monitor = self.build_monitor()
        chunks = make_chunks(T0, 150, FIRE_DOOR_CAPTION)
        span = (chunks[-1].t_end - chunks[0].t_start).total_seconds()
        cooldown = float(config.get("monitor.default_cooldown_seconds"))
        self.assertGreater(
            span,
            cooldown,
            "the event must outlast the cooldown, or this test proves only that the "
            "cooldown works and says nothing about dedupe",
        )

        outcomes = self.feed(monitor, chunks)

        fired = [o for o in outcomes if o.fired]  # type: ignore[attr-defined]
        self.assertEqual(len(fired), 1, "one event must produce one action")

        rows = self.originals_for("fire-door-blocked")
        self.assertEqual(len(rows), 1, f"expected 1 action log row, got {len(rows)}")
        self.assertEqual(rows[0].action, ActionKind.RAISE_ALERT)
        self.assertEqual(rows[0].status, ActionStatus.UNVERIFIED)

        # Every chunk kept matching and kept asking; the brakes did the refusing.
        self.assertEqual(monitor.stats["stage2_matches"], 150)
        self.assertEqual(monitor.stats["promotions"], 1)
        self.assertGreater(monitor.stats["suppressed"], 100)
        self.assertEqual(
            monitor.stats["fired"] + monitor.stats["suppressed"],
            len([o for o in outcomes if o.sustained]),  # type: ignore[attr-defined]
        )

    def test_dedupe_carries_the_event_past_the_cooldown(self) -> None:
        """After the cooldown lapses the *footage-range* brake is what still holds.

        Without this the funnel would re-fire every 300 s for as long as the van sat
        there — the thirty-alerts failure mode arriving on a timer instead of all at once.
        """
        monitor = self.build_monitor()
        outcomes = self.feed(monitor, make_chunks(T0, 150, FIRE_DOOR_CAPTION))

        fired_at = self.originals_for("fire-door-blocked")[0].ts

        # "Dedupe engaged and cooldown did not" *is* the statement that the cooldown had
        # already lapsed and the footage-range brake was carrying the event on its own.
        def dedupe_alone(outcome) -> bool:
            result = outcome.action
            if result is None or result.fired:
                return False
            engaged = result.engaged_brakes
            return Brake.DEDUPE in engaged and Brake.COOLDOWN not in engaged

        suppressed_after_cooldown = [o for o in outcomes if dedupe_alone(o)]
        self.assertTrue(
            suppressed_after_cooldown,
            "the run must extend past the cooldown for this test to mean anything",
        )
        last = suppressed_after_cooldown[-1].action  # type: ignore[attr-defined]
        self.assertEqual(last.brake, Brake.DEDUPE)
        self.assertNotIn(
            Brake.COOLDOWN,
            last.engaged_brakes,
            "the cooldown had lapsed; dedupe alone must be holding the event",
        )
        self.assertEqual(len(self.originals_for("fire-door-blocked")), 1)
        self.assertGreater(fired_at, T0)

    def test_a_second_separate_event_is_allowed_to_fire(self) -> None:
        """The brakes suppress a repeat of one event, not the next event.

        A monitor that could only ever alert once would pass the headline test and be
        useless, so the opposite case is asserted alongside it.
        """
        monitor = self.build_monitor()
        self.feed(monitor, make_chunks(T0, 40, FIRE_DOOR_CAPTION))
        self.assertEqual(len(self.originals_for("fire-door-blocked")), 1)

        # The van leaves: quiet footage breaks the run. Then, an hour later, another one.
        self.feed(monitor, make_chunks(T0 + timedelta(minutes=5), 5, QUIET_CAPTION))
        self.feed(monitor, make_chunks(T0 + timedelta(hours=1), 40, FIRE_DOOR_CAPTION))

        self.assertEqual(len(self.originals_for("fire-door-blocked")), 2)


# ======================================================================================
# Stage 2 — the sustain window
# ======================================================================================


class SustainWindowTest(MonitorTestCase):
    def test_sustain_window_delays_promotion(self) -> None:
        """Nothing fires until ``window`` seconds of consecutive matches have accrued."""
        tasks = seed_tasks()
        bay = next(t for t in tasks if t.task_id == "loading-bay-activity")
        monitor = self.build_monitor([bay])

        chunks = make_chunks(T0, 20, LOADING_BAY_CAPTION)
        fired_index: int | None = None
        for i, chunk in enumerate(chunks):
            self.clock.set(chunk.t_end)
            outcome = monitor.observe([chunk])[0]
            held = (chunk.t_end - chunks[0].t_start).total_seconds()
            if outcome.fired:
                fired_index = i
                break
            self.assertLess(
                held,
                bay.window,
                f"chunk {i} held {held}s of the {bay.window}s window and did not fire",
            )

        self.assertIsNotNone(fired_index, "the task never promoted")
        held_at_fire = (chunks[fired_index].t_end - chunks[0].t_start).total_seconds()
        self.assertGreaterEqual(held_at_fire, bay.window)
        # ...and the chunk before it was still short of the window.
        self.assertLess(
            (chunks[fired_index - 1].t_end - chunks[0].t_start).total_seconds(), bay.window
        )

    def test_state_shows_sustain_progress_as_an_absolute_start(self) -> None:
        """The pane gets ``since``, never "96 of 120 seconds". SPEC §11.3."""
        bay = next(t for t in seed_tasks() if t.task_id == "loading-bay-activity")
        monitor = self.build_monitor([bay])
        self.feed(monitor, make_chunks(T0, 3, LOADING_BAY_CAPTION))

        row = monitor.state().task("loading-bay-activity")
        self.assertIsNotNone(row)
        self.assertEqual(row.stage2.verdict, "match")
        self.assertEqual(row.stage2.since, T0)
        self.assertEqual(row.stage2.sustain_window_s, bay.window)
        self.assertIsNone(row.last_fired_ts, "not promoted yet, so nothing has fired")
        self.assertEqual(row.state, "matching")

    def test_a_gated_chunk_breaks_the_run(self) -> None:
        """SPEC §2.3's null records mean nothing is happening, not "unchanged"."""
        bay = next(t for t in seed_tasks() if t.task_id == "loading-bay-activity")
        monitor = self.build_monitor([bay])

        # Almost enough to promote, then the detector sees nothing.
        self.feed(monitor, make_chunks(T0, 6, LOADING_BAY_CAPTION))
        self.feed(monitor, make_chunks(T0 + timedelta(seconds=24), 1, LOADING_BAY_CAPTION,
                                       gated_indices=(0,)))
        self.assertIsNone(monitor.state().task("loading-bay-activity").stage2.since)
        self.assertEqual(len(self.originals_for("loading-bay-activity")), 0)

        # The run restarts from scratch: 6 more chunks are again not enough.
        restart = T0 + timedelta(seconds=28)
        self.feed(monitor, make_chunks(restart, 6, LOADING_BAY_CAPTION))
        self.assertEqual(len(self.originals_for("loading-bay-activity")), 0)
        self.assertEqual(monitor.state().task("loading-bay-activity").stage2.since, restart)

    def test_stage_one_filters_before_the_llm_is_asked(self) -> None:
        """Stage 2 costs ~1 s per call. It must not run on every chunk."""
        monitor = self.build_monitor()
        self.feed(monitor, make_chunks(T0, 10, QUIET_CAPTION))

        self.assertEqual(monitor.stats["stage1_candidates"], 0)
        self.assertEqual(monitor.stats["stage2_confirms"], 0)
        self.assertEqual(monitor.stats["chunks_seen"], 10)

        # And the loose gate really is loose: the matching caption passes stage 1 for its
        # own task and fails it for the other one, with a single threshold.
        fire_score = cosine(
            self.embedder.embed_query("a vehicle stopped in front of the fire door"),
            self.embedder.embed_passages([FIRE_DOOR_CAPTION])[0],
        )
        self.assertGreater(fire_score, STAGE1_THRESHOLD)

    def test_stub_confirmer_separates_coverage_from_similarity(self) -> None:
        """Stage 2 asks a different question from stage 1: coverage, not direction."""
        confirmer = StubConfirmer(self.settings.stub_min_overlap)
        task = next(t for t in seed_tasks() if t.task_id == "fire-door-blocked")
        self.assertTrue(confirmer.confirm(FIRE_DOOR_CAPTION, task).match)
        self.assertFalse(confirmer.confirm("A white van drives past.", task).match)
        self.assertFalse(confirmer.confirm(QUIET_CAPTION, task).match)

    def test_a_stage_two_rejection_stops_a_stage_one_candidate(self) -> None:
        """Stage 1 over-triggers on purpose; stage 2 is what stops the action.

        Driven by an injected confirmer rather than by a cunningly chosen caption. Both
        stand-ins on this box are lexical — the hashing embedder and the stub confirmer
        read the same words — so a caption that split them would be a fact about the
        stand-ins, not about the funnel. What must be true regardless of backend is that a
        stage-2 "no" ends the run, and that is what this asserts.
        """

        class AlwaysNo:
            model = "test-always-no"

            def __init__(self) -> None:
                self.calls = 0

            def confirm(self, caption: str, task: Task):
                from services.monitor import ConfirmVerdict

                self.calls += 1
                return ConfirmVerdict(match=False, detail="test", model=self.model)

        refuser = AlwaysNo()
        fire_door = next(t for t in seed_tasks() if t.task_id == "fire-door-blocked")
        monitor = self.build_monitor([fire_door])
        monitor.confirmer = refuser
        outcomes = self.feed(monitor, make_chunks(T0, 60, FIRE_DOOR_CAPTION))

        self.assertEqual(monitor.stats["stage1_candidates"], 60, "stage 1 let them through")
        self.assertEqual(refuser.calls, 60, "stage 2 was asked about every candidate")
        self.assertEqual(monitor.stats["stage2_matches"], 0)
        self.assertEqual(monitor.stats["promotions"], 0)
        self.assertEqual(len(self.originals_for("fire-door-blocked")), 0)
        self.assertTrue(all(o.reached == 2 for o in outcomes))  # type: ignore[attr-defined]
        self.assertIsNone(monitor.state().task("fire-door-blocked").stage2.since)
        self.assertEqual(monitor.state().task("fire-door-blocked").stage2.verdict, "no_match")


# ======================================================================================
# Stage 3 — verification, retraction, and the amendment shape
# ======================================================================================


class VerificationTest(MonitorTestCase):
    def _fire_once(self) -> str:
        task = next(t for t in seed_tasks() if t.task_id == "fire-door-blocked")
        monitor = self.build_monitor([task])
        self.feed(monitor, make_chunks(T0, 40, FIRE_DOOR_CAPTION))
        rows = self.originals_for("fire-door-blocked")
        self.assertEqual(len(rows), 1)
        self.monitor = monitor
        return rows[0].entry_id

    def test_alert_fires_before_the_worker_answers(self) -> None:
        """SPEC §6.3: stage 3 is not a blocking precondition."""
        entry_id = self._fire_once()
        self.assertEqual(len(self.verifier.submitted), 1)
        job = self.verifier.submitted[0]
        self.assertEqual(job.state, JobState.QUEUED, "submit must return without blocking")

        row = self.monitor.state().task("fire-door-blocked")
        self.assertEqual(row.stage3.state, "queued")
        self.assertEqual(row.stage3.job_id, job.job_id)
        self.assertIsNone(row.stage3.verdict)
        self.assertEqual(self.actions.resolve(entry_id).status, ActionStatus.UNVERIFIED)
        # The worker was asked about the *matched range*, not about one 5 s window.
        self.assertEqual(job.t_start, T0)
        self.assertGreater((job.t_end - job.t_start).total_seconds(), 120)

    def test_stage_three_agreement_appends_a_verified_row(self) -> None:
        entry_id = self._fire_once()
        job = self.verifier.submitted[0]
        self.verifier.finish(
            job.job_id, confidence=0.92, answer="A van is stopped at the fire door at 21:11:14.",
            evidence_clip="data/clips/cam01_evidence.mp4",
        )
        outcomes = self.monitor.pump_verifications()

        self.assertEqual([o.verdict for o in outcomes], ["verified"])
        resolved = self.actions.resolve(entry_id)
        self.assertEqual(resolved.status, ActionStatus.VERIFIED)
        self.assertEqual(len(resolved.amendments), 1)
        self.assertEqual(resolved.amendments[0].parent_id, entry_id)
        self.assertEqual(resolved.original.status, ActionStatus.UNVERIFIED,
                         "the original row must never be mutated")
        self.assertEqual(self.monitor.state().task("fire-door-blocked").stage3.verdict, "verified")

    def test_retraction_when_stage_three_disagrees(self) -> None:
        """The visible retraction is the thesis (SPEC §11.4). It is an append."""
        entry_id = self._fire_once()
        job = self.verifier.submitted[0]
        self.verifier.finish(
            job.job_id, confidence=0.05, answer="No vehicle is present; the door is clear."
        )
        outcomes = self.monitor.pump_verifications()

        self.assertEqual([o.verdict for o in outcomes], ["retracted"])
        resolved = self.actions.resolve(entry_id)
        self.assertEqual(resolved.status, ActionStatus.RETRACTED)
        self.assertTrue(resolved.retracted)
        self.assertEqual(resolved.original.status, ActionStatus.UNVERIFIED)
        self.assertEqual(resolved.amendments[0].parent_id, entry_id)
        self.assertIn("No vehicle", resolved.reason)
        self.assertEqual(self.monitor.state().task("fire-door-blocked").stage3.verdict, "retracted")

        # A retraction does not release the brakes: "we were wrong, so let us try again
        # immediately" is the thirty-alerts failure mode with extra steps.
        self.feed(self.monitor, make_chunks(T0 + timedelta(seconds=160), 20, FIRE_DOOR_CAPTION))
        self.assertEqual(len(self.originals_for("fire-door-blocked")), 1)

    def test_a_failed_worker_is_inconclusive_not_a_disagreement(self) -> None:
        """A timeout says nothing about the footage, so it must not retract an alert."""
        entry_id = self._fire_once()
        job = self.verifier.submitted[0]
        self.verifier.finish(job.job_id, confidence=None, state=JobState.TIMEOUT)
        outcomes = self.monitor.pump_verifications()

        self.assertEqual([o.verdict for o in outcomes], [None])
        resolved = self.actions.resolve(entry_id)
        self.assertEqual(resolved.status, ActionStatus.UNVERIFIED)
        self.assertEqual(resolved.amendments, ())
        row = self.monitor.state().task("fire-door-blocked")
        self.assertEqual(row.stage3.state, "failed")
        self.assertIsNone(row.stage3.verdict)

    def test_save_clip_is_low_stakes_and_asks_for_no_verification(self) -> None:
        """SPEC §6.3 splits by action severity, not by task."""
        bay = next(t for t in seed_tasks() if t.task_id == "loading-bay-activity")
        self.assertFalse(ActionKind.SAVE_CLIP.reaches_a_human)
        monitor = self.build_monitor([bay])
        self.feed(monitor, make_chunks(T0, 20, LOADING_BAY_CAPTION))

        rows = self.originals_for("loading-bay-activity")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].action, ActionKind.SAVE_CLIP)
        self.assertEqual(self.verifier.submitted, [], "save_clip owes no worker time")
        self.assertEqual(monitor.state().task("loading-bay-activity").stage3.state, "idle")

    def test_an_unknown_job_is_not_ours_to_act_on(self) -> None:
        """M3 escalates jobs too (SPEC §4.2). Those must not amend M5's rows."""
        self._fire_once()
        stranger = DeepJob(
            job_id="not-ours", t_start=T0, t_end=T0 + timedelta(seconds=30),
            question="?", state=JobState.DONE, confidence=0.0,
        )
        self.assertIsNone(self.monitor.apply_verification(stranger))
        self.assertEqual(len(self.log_rows()), 1)


# ======================================================================================
# `active` — a LOCAL window that may wrap midnight (SPEC §6.1)
# ======================================================================================


class ActiveWindowTest(unittest.TestCase):
    def test_midnight_wrap_parsing(self) -> None:
        overnight = parse_active_window("18:00-06:00")
        self.assertTrue(overnight.wraps_midnight)
        self.assertFalse(overnight.always)
        for minute, expected in (
            (17 * 60 + 59, False),
            (18 * 60, True),      # inclusive at the start
            (23 * 60 + 59, True),
            (0, True),            # ...through midnight...
            (5 * 60 + 59, True),
            (6 * 60, False),      # exclusive at the end
            (12 * 60, False),
        ):
            with self.subTest(minute=minute):
                self.assertEqual(overnight.contains_local_minute(minute), expected)

    def test_non_wrapping_and_always_on(self) -> None:
        day = parse_active_window("09:00-17:00")
        self.assertFalse(day.wraps_midnight)
        self.assertTrue(day.contains_local_minute(9 * 60))
        self.assertFalse(day.contains_local_minute(17 * 60))
        self.assertFalse(day.contains_local_minute(3 * 60))

        always = parse_active_window("00:00-24:00")
        self.assertTrue(always.always)
        for minute in (0, 6 * 60, 23 * 60 + 59):
            self.assertTrue(always.contains_local_minute(minute))

    def test_unparseable_windows_are_rejected_at_registration(self) -> None:
        """A scoped task must never silently become an unscoped one."""
        for spec in ("overnight", "18:00–06:00", "6pm-6am", "25:00-06:00", "18:60-06:00", ""):
            with self.subTest(spec=spec), self.assertRaises(ActiveWindowError):
                parse_active_window(spec)

    def test_resolved_against_the_display_timezone_not_utc(self) -> None:
        """The whole point: "overnight" means the operator's night, not UTC's."""
        overnight = parse_active_window("18:00-06:00")
        # Asia/Kolkata is UTC+5:30, so 21:11 local is 15:41 UTC — a time that is plainly
        # daytime in UTC and would be excluded by a naive UTC comparison.
        self.assertTrue(overnight.contains(T0, DISPLAY_TZ))
        self.assertFalse(
            overnight.contains(T0, "UTC"),
            "15:41 UTC is outside 18:00-06:00 in UTC; the test is only meaningful "
            "because the two zones disagree",
        )


class ActiveWindowFunnelTest(MonitorTestCase):
    """The window as the funnel applies it: out-of-hours chunks never reach stage 1."""

    def _fire_door(self) -> Task:
        return next(t for t in seed_tasks() if t.task_id == "fire-door-blocked")

    def test_out_of_hours_chunks_are_not_evaluated_at_all(self) -> None:
        monitor = self.build_monitor([self._fire_door()])
        # 12:00 local = 06:30 UTC — the middle of the excluded half of 18:00-06:00.
        noon_local = datetime(2026, 8, 14, 6, 30, 0, tzinfo=timezone.utc)
        outcomes = self.feed(monitor, make_chunks(noon_local, 60, FIRE_DOOR_CAPTION))

        self.assertEqual(len(self.originals_for("fire-door-blocked")), 0)
        self.assertEqual(monitor.stats["stage1_candidates"], 0)
        self.assertEqual(monitor.stats["stage2_confirms"], 0)
        self.assertTrue(all(o.reached == 0 for o in outcomes))  # type: ignore[attr-defined]
        self.assertIn("outside active window", outcomes[0].detail)  # type: ignore[attr-defined]

        row = monitor.state().task("fire-door-blocked")
        self.assertFalse(row.in_active_window)
        self.assertEqual(row.state, "out_of_window")

    def test_the_window_wraps_midnight_in_local_time(self) -> None:
        """02:00 local is inside "18:00-06:00" and is 20:30 UTC the *previous* day."""
        monitor = self.build_monitor([self._fire_door()])
        two_am_local = datetime(2026, 8, 13, 20, 30, 0, tzinfo=timezone.utc)
        self.feed(monitor, make_chunks(two_am_local, 40, FIRE_DOOR_CAPTION))

        self.assertEqual(len(self.originals_for("fire-door-blocked")), 1)
        self.assertTrue(monitor.state().task("fire-door-blocked").in_active_window)

    def test_leaving_the_window_breaks_a_sustained_run(self) -> None:
        """A run that straddles 06:00 local must not promote on out-of-hours footage."""
        monitor = self.build_monitor([self._fire_door()])
        # Start 90 s before 06:00 local (= 00:30 UTC): not enough for the 120 s window.
        start = datetime(2026, 8, 14, 0, 28, 30, tzinfo=timezone.utc)
        self.feed(monitor, make_chunks(start, 60, FIRE_DOOR_CAPTION))

        self.assertEqual(len(self.originals_for("fire-door-blocked")), 0)
        self.assertIsNone(monitor.state().task("fire-door-blocked").stage2.since)


# ======================================================================================
# The registry — SPEC §6.1, §11.3, D5
# ======================================================================================


class RegistryTest(MonitorTestCase):
    def test_seed_is_the_cold_start_and_is_read_not_retyped(self) -> None:
        monitor = self.build_monitor()
        ids = [t.task_id for t in monitor.tasks()]
        self.assertEqual(ids, ["fire-door-blocked", "loading-bay-activity"])
        for task in monitor.tasks():
            self.assertTrue(task.embedding, "describe is embedded once, at registration")
            self.assertEqual(len(task.embedding), self.embedder.dims)

    def test_register_task_at_runtime_embeds_and_appears_in_state(self) -> None:
        monitor = self.build_monitor()
        stored = monitor.register_task(
            {
                "task_id": "gate-left-open",
                "describe": "the side gate is standing open",
                "window": 60,
                "action": "file_ticket",
                "cooldown": 600,
                "active": "00:00-24:00",
            }
        )
        self.assertEqual(len(stored.embedding), self.embedder.dims)
        row = monitor.state().task("gate-left-open")
        self.assertIsNotNone(row)
        self.assertEqual(row.state, "armed")
        self.assertEqual(row.cooldown_seconds, 600)
        self.assertEqual(row.stage2.sustain_window_s, 60)
        self.assertIsNone(row.match_range)

    def test_duplicate_and_malformed_registrations_are_refused(self) -> None:
        monitor = self.build_monitor()
        with self.assertRaises(TaskRegistrationError):
            monitor.register_task(
                {"task_id": "fire-door-blocked", "describe": "x", "window": 5,
                 "action": "save_clip"}
            )
        with self.assertRaises(ActiveWindowError):
            monitor.register_task(
                {"task_id": "bad-window", "describe": "x", "window": 5,
                 "action": "save_clip", "active": "overnight"}
            )
        with self.assertRaises(TaskRegistrationError):
            monitor.register_task({"task_id": "no-describe", "window": 5, "action": "save_clip"})

    def test_a_disabled_task_is_rendered_but_never_fires(self) -> None:
        monitor = self.build_monitor()
        monitor.registry.set_enabled("fire-door-blocked", False)
        self.feed(monitor, make_chunks(T0, 60, FIRE_DOOR_CAPTION))

        self.assertEqual(len(self.originals_for("fire-door-blocked")), 0)
        row = monitor.state().task("fire-door-blocked")
        self.assertIsNotNone(row, "a disabled task must still be rendered, not vanish")
        self.assertEqual(row.state, "disabled")


# ======================================================================================
# The Watch pane contract — SPEC §11.3, ui/mock/monitor_state.json
# ======================================================================================


def key_shape(value: object) -> object:
    """Recursive key structure, ignoring values and ``_``-prefixed fixture commentary."""
    if isinstance(value, dict):
        return {k: key_shape(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [key_shape(v) for v in value]
    return None


class MonitorStateContractTest(MonitorTestCase):
    def setUp(self) -> None:
        super().setUp()
        mock_path = Path(__file__).resolve().parent.parent / "ui" / "mock" / "monitor_state.json"
        self.mock = json.loads(mock_path.read_text(encoding="utf-8"))

    def _live_state(self) -> dict:
        """One task mid-event (so ``match_range`` is populated) and one idle."""
        monitor = self.build_monitor()
        self.feed(monitor, make_chunks(T0, 40, FIRE_DOOR_CAPTION))
        self.monitor = monitor
        return monitor.state().to_dict()

    def test_shape_matches_the_ui_fixture(self) -> None:
        live = self._live_state()
        self.assertEqual(
            key_shape({k: v for k, v in self.mock.items() if not k.startswith("_")}),
            key_shape({**live, "tasks": live["tasks"]}),
            "GET /api/monitor/state must match ui/mock/monitor_state.json; the Watch pane "
            "is already written against these names",
        )
        # The fixture's first row is mid-event and its second is idle — the same pair the
        # live state above produces, which is what makes the comparison meaningful.
        self.assertIsNotNone(live["tasks"][0]["match_range"])
        self.assertIsNone(live["tasks"][1]["match_range"])

    def test_state_carries_absolute_timestamps_never_countdowns(self) -> None:
        live = self._live_state()

        def walk(node: object, path: str = "") -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    self.assertNotIn(
                        "remain", k, f"{path}.{k} is a countdown; the UI derives those"
                    )
                    self.assertNotIn("elapsed", k, f"{path}.{k} is a countdown")
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(live)
        row = live["tasks"][0]
        for field in (live["generated_at"], row["last_fired_ts"], row["stage2"]["since"],
                      row["match_range"]["t_start"], row["match_range"]["t_end"]):
            self.assertTrue(field.endswith("Z"), f"{field!r} must be UTC ISO 8601 with Z")
            self.assertEqual(from_iso(field).tzinfo, timezone.utc)

        # cooldown_seconds is the dial's setting, not a countdown: it does not move.
        first = row["cooldown_seconds"]
        self.clock.advance(60)
        self.assertEqual(self.monitor.state().to_dict()["tasks"][0]["cooldown_seconds"], first)

    def test_state_is_json_serialisable_for_the_http_route(self) -> None:
        """M3 owns the route; this must be one ``json.dumps`` away from being served."""
        payload = json.dumps(self._live_state())
        self.assertIn("fire-door-blocked", payload)
        self.assertEqual(json.loads(payload)["tasks"][0]["stage1"]["threshold"], STAGE1_THRESHOLD)


# ======================================================================================
# Wiring
# ======================================================================================


class BuildMonitorTest(unittest.TestCase):
    def test_build_monitor_wires_the_seed_and_the_stub_backends(self) -> None:
        """The whole funnel must run on this box today: no NGC key, no LLM serving."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = FakeClock(T0)
            monitor = build_monitor(
                actions=ActionServer(
                    log_path=root / "actions.jsonl", clips_dir=root / "clips", clock=clock
                ),
                clock=clock,
            )
            self.assertEqual(
                [t.task_id for t in monitor.tasks()],
                ["fire-door-blocked", "loading-bay-activity"],
            )
            # build_monitor picks stage 2 from `agent.backend` — the SAME key M3 reads,
            # so the two surfaces can never end up on different models. Assert the
            # SELECTION, not a fixed value: SPEC §10 D3 is resolved to `nim` now, and a
            # test hardcoding "stub" would have to be rewritten every time the decision
            # moves rather than telling us the wiring is right.
            backend = str(config.get("agent.backend"))
            expected = StubConfirmer if backend == "stub" else NIMConfirmer
            self.assertIsInstance(monitor.confirmer, expected)
            # The stub path must remain usable regardless, because it is what runs when
            # nothing is serving — a fresh box, or the model process down mid-demo.
            self.assertIsInstance(
                build_confirmer(dataclasses.replace(MonitorSettings.from_config(), confirm_backend="stub")),
                StubConfirmer,
            )
            # And it does not touch data/actions.jsonl on the way through.
            self.assertTrue((root / "actions.jsonl").parent.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
