"""M5 — the standing-task monitor. SPEC §6.

Long-running and push-triggered: it subscribes to every chunk M1 emits and is **the only
module that changes the outside world unprompted**, which is why it is the one that needs
brakes and why it does not own them.

The three-stage funnel (SPEC §6.2), narrowing hard at each step:

===== ============== ======= ==========================================================
Stage Runs on        Cost    Does
===== ============== ======= ==========================================================
1     every chunk    free    Cosine between the task's once-embedded ``describe`` and
                             the caption. Deliberately loose — over-trigger here and
                             filter later.
2     candidates     ~1 s    LLM reads caption + task, says match/no. Also holds the
                             **sustain window**: ``window`` seconds of consecutive
                             matches before promoting.
3     promoted       20–60 s ``deep_analyze`` re-watches the footage at 4 fps. Captions
                             are lossy; do not file a ticket on a 1 fps guess.
===== ============== ======= ==========================================================

**Acting is non-blocking (SPEC §6.3).** Stage 3 is not a precondition. The action fires on
stage-2 confidence and the verified verdict is attached when the worker finishes — the
same provisional-then-refined shape M3 uses on the Ask surface. The split is by *action
severity*, not by task, and ``ActionKind.reaches_a_human`` already encodes it:
``save_clip`` fires and is done; ``raise_alert``/``file_ticket`` fire as ``UNVERIFIED``
and are amended, or retracted, on stage 3.

**The brakes are not here.** Cooldown, footage-range dedupe and the append-only log live
in ``services/mcp`` and are already tested (CLAUDE.md invariant 5). This module fires
through :class:`~services.mcp.server.ActionServer` on *every* sustained chunk and lets the
brakes refuse — one brake, in one place. A second cooldown here would be a second thing to
get wrong, and the two would disagree on the day one of them was edited.

Two behaviours worth stating outright, because both look like bugs and neither is:

* **A sustained run keeps asking and keeps being refused.** Once promoted, every
  subsequent matching chunk re-requests the action. The dedupe brake sees a footage range
  that still overlaps the one already fired on and suppresses it, forever, for as long as
  the run continues. That is how one four-minute event produces exactly one alert, and it
  is what the Watch pane renders as COOLING while stage 1 and stage 2 visibly keep
  matching (SPEC §11.3).
* **Firing does not reset the run.** Resetting would let the same unbroken event promote
  again a ``window`` later, on a footage range that no longer overlaps — which is the
  thirty-alerts failure mode arriving on a timer. The run ends when stage 2 says no, when
  the chunk is gated, or when the clock leaves the task's ``active`` hours.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from shared.schema import ChunkRecord, DeepJob, JobState, Task, utcnow

from services.index.embedding import Embedder, build_embedder
from services.index.settings import IndexSettings
from services.mcp import ActionResult, ActionServer

from services.monitor.confirm import Stage2Confirmer, build_confirmer
from services.monitor.registry import TaskRegistry
from services.monitor.settings import MonitorSettings
from services.monitor.state import (
    MonitorState,
    Stage1State,
    Stage2State,
    Stage3State,
    TaskFunnelState,
    TimeRange,
)
from services.monitor.verify import (
    DeepVerifier,
    NullVerifier,
    VerdictFn,
    confidence_verdict,
)

__all__ = ["Monitor", "FunnelOutcome", "VerificationOutcome", "build_monitor", "cosine"]

logger = logging.getLogger("monitor.funnel")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, tolerant of a zero vector.

    Not imported from M2's backend: that one is private and takes a precomputed norm
    because it runs over a whole corpus. Here it runs once per task per chunk over two
    vectors, so the honest form is cheaper than the cached one.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --------------------------------------------------------------------------------------
# What one evaluation produced — returned so callers can log, test and stream it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FunnelOutcome:
    """One task's verdict on one chunk. ``reached`` is how far down the funnel it got.

    Returned rather than only logged so that a test can assert *which stage* stopped a
    chunk. "No alert fired" is true of a stage-1 miss and of a suppressed stage-2
    promotion, and those are opposite health signals.
    """

    chunk_id: str
    task_id: str
    reached: int  # 0 = not evaluated (disabled / out of window / gated), 1, 2, 3
    detail: str
    stage1_score: float = 0.0
    stage1_matched: bool = False
    stage2_match: bool | None = None
    sustained: bool = False
    action: ActionResult | None = None
    job_id: str | None = None

    @property
    def fired(self) -> bool:
        return self.action is not None and self.action.fired


@dataclass(frozen=True)
class VerificationOutcome:
    """What stage 3 did to an already-fired action."""

    task_id: str
    entry_id: str
    job_id: str
    verdict: str | None  # "verified" | "retracted" | None (inconclusive)
    detail: str = ""


# --------------------------------------------------------------------------------------
# Per-task runtime state. Not persisted: the action log is the only history store
# (CLAUDE.md conventions), and this is the live funnel, which is not history.
# --------------------------------------------------------------------------------------


@dataclass
class _Runtime:
    stage1_score: float = 0.0
    stage1_matched: bool = False
    stage1_chunk_id: str | None = None
    stage2_verdict: str | None = None
    stage2_last_chunk_id: str | None = None
    #: Footage start of the current unbroken run of stage-2 matches.
    run_since: datetime | None = None
    #: Footage end of the most recent chunk in that run.
    run_until: datetime | None = None
    #: Has this run already reached its sustain window at least once?
    promoted: bool = False
    stage3_state: str = "idle"
    stage3_job_id: str | None = None
    stage3_verdict: str | None = None
    last_fired_ts: datetime | None = None
    last_entry_id: str | None = None

    def break_run(self) -> None:
        """End the current run. Stage-1 telemetry is left alone — it is per chunk."""
        self.run_since = None
        self.run_until = None
        self.promoted = False


class Monitor:
    """The funnel. Feed it chunks; it fires actions through the MCP action server.

    Everything that could reach the network or the filesystem is injected: the embedder,
    the stage-2 confirmer, the stage-3 verifier and the action server. That is not
    ceremony — it is what lets the whole of SPEC §6 be exercised on this box today, with
    no NGC key, no LLM serving and M4 still being written next door.
    """

    def __init__(
        self,
        *,
        registry: TaskRegistry,
        actions: ActionServer,
        confirmer: Stage2Confirmer,
        embedder: Embedder,
        settings: MonitorSettings | None = None,
        verifier: DeepVerifier | None = None,
        verdict_fn: VerdictFn | None = None,
        clock: Callable[[], datetime] = utcnow,
        timezone_override: str | ZoneInfo | None = None,
    ) -> None:
        self.settings = settings or MonitorSettings.from_config()
        self.registry = registry
        self.actions = actions
        self.confirmer = confirmer
        self.embedder = embedder
        self.verifier: DeepVerifier = verifier or NullVerifier()
        self._verdict_fn: VerdictFn = verdict_fn or (
            lambda job: confidence_verdict(
                job, threshold=self.settings.verify_confidence_threshold
            )
        )
        self._clock = clock
        #: Overrides ``ui.display_timezone`` for ``Task.active``. Tests use it; so would a
        #: box watching a camera in another city.
        self._tz = timezone_override
        self._runtime: dict[str, _Runtime] = {}
        #: job_id -> (task_id, entry_id). The bridge between a fired provisional action
        #: and the verdict that will amend it.
        self._pending: dict[str, tuple[str, str]] = {}
        self.stats: dict[str, int] = {
            "chunks_seen": 0,
            "stage1_candidates": 0,
            "stage2_confirms": 0,
            "stage2_matches": 0,
            "promotions": 0,
            "fired": 0,
            "suppressed": 0,
            "verifications_submitted": 0,
        }

    # ----------------------------------------------------------------------------------
    # Registration — SPEC §10 D5. M3's endpoint delegates here.
    # ----------------------------------------------------------------------------------

    def register_task(self, task: Task | dict[str, Any]) -> Task:
        """Register a task at runtime, embedding its description once (SPEC §6.2).

        The Watch pane shows the new card on its next poll with an empty funnel, which is
        the truth: nothing has been evaluated against it yet.
        """
        stored = self.registry.register(task)
        self._runtime.setdefault(stored.task_id, _Runtime())
        return stored

    def tasks(self) -> list[Task]:
        """Every registered task, for ``GET /api/tasks``. M3 owns the route."""
        return self.registry.tasks()

    # ----------------------------------------------------------------------------------
    # The funnel
    # ----------------------------------------------------------------------------------

    def observe(self, chunks: list[ChunkRecord]) -> list[FunnelOutcome]:
        """Evaluate a batch of chunks against every registered task.

        Takes a **list** even though M1 always passes one (CLAUDE.md invariant 9): a batch
        dimension today is a config change later, a single-chunk signature is a refactor.

        Returns one outcome per (chunk, task) pair, in chunk order then registration
        order, so a caller can stream the funnel to the UI without re-deriving it.
        """
        outcomes: list[FunnelOutcome] = []
        for chunk in chunks:
            self.stats["chunks_seen"] += 1
            # Embed once per chunk, not once per task. Stage 1's claim to being free
            # depends on this being a dot product per task, not a model call per task.
            vector = self._chunk_vector(chunk)
            for task in self.registry.tasks():
                outcomes.append(self._evaluate(task, chunk, vector))
        return outcomes

    def _chunk_vector(self, chunk: ChunkRecord) -> list[float]:
        """The caption's vector, reusing the one on the record when it is usable.

        M2's store re-embeds captions rather than trusting whatever arrived on the record,
        for the good reason that a corpus embedded by one model and queried by another is
        a silent recall collapse. The same applies here, one dimension at a time: a
        vector of the wrong width cannot be compared against a task embedding at all, so
        it is re-derived rather than skipped.
        """
        if chunk.gated or not chunk.caption:
            return []
        if chunk.embedding and len(chunk.embedding) == self.embedder.dims:
            return list(chunk.embedding)
        return list(self.embedder.embed_passages([chunk.caption])[0])

    def _rt(self, task_id: str) -> _Runtime:
        return self._runtime.setdefault(task_id, _Runtime())

    def _evaluate(
        self, task: Task, chunk: ChunkRecord, vector: list[float]
    ) -> FunnelOutcome:
        rt = self._rt(task.task_id)

        if not task.enabled:
            rt.break_run()
            return FunnelOutcome(chunk.chunk_id, task.task_id, 0, "task disabled")

        # `active` is LOCAL wall clock and may wrap midnight (SPEC §6.1). Resolved here,
        # against ui.display_timezone, at match time — never stored converted, because a
        # cached UTC copy breaks at the next DST transition.
        window = self.registry.window_for(task.task_id)
        if not window.contains(chunk.t_start, self._tz):
            # Leaving the active hours ends the run. A vehicle that arrived at 05:58 and
            # is still there at 06:01 stops being this task's business at 06:00; carrying
            # the sustain across the boundary would let an out-of-hours minute promote an
            # in-hours event, which is the opposite of what the window is for.
            rt.break_run()
            rt.stage2_verdict = None
            return FunnelOutcome(
                chunk.chunk_id, task.task_id, 0, f"outside active window {task.active}"
            )

        if chunk.gated or not chunk.caption:
            # SPEC §2.3: the detector found nothing and inference was skipped. Nothing is
            # happening, so a run in progress has ended. These null records are still
            # written by M1 and must not be read as "unchanged".
            rt.break_run()
            rt.stage2_verdict = None
            rt.stage1_score = 0.0
            rt.stage1_matched = False
            rt.stage1_chunk_id = chunk.chunk_id
            return FunnelOutcome(chunk.chunk_id, task.task_id, 0, "chunk gated; no caption")

        # ---- stage 1: embedding match. Free, every chunk, deliberately loose. ---------
        score = cosine(task.embedding, vector)
        rt.stage1_score = score
        rt.stage1_chunk_id = chunk.chunk_id
        rt.stage1_matched = score >= self.settings.stage1_cosine_threshold
        if not rt.stage1_matched:
            rt.break_run()
            rt.stage2_verdict = None
            return FunnelOutcome(
                chunk.chunk_id,
                task.task_id,
                1,
                f"stage 1 miss: {score:.3f} < {self.settings.stage1_cosine_threshold:g}",
                stage1_score=score,
            )
        self.stats["stage1_candidates"] += 1

        # ---- stage 2: LLM confirm, on candidates only. -------------------------------
        self.stats["stage2_confirms"] += 1
        verdict = self.confirmer.confirm(chunk.caption, task)
        rt.stage2_verdict = verdict.verdict
        rt.stage2_last_chunk_id = chunk.chunk_id
        if not verdict.match:
            rt.break_run()
            return FunnelOutcome(
                chunk.chunk_id,
                task.task_id,
                2,
                f"stage 2 no match: {verdict.detail}",
                stage1_score=score,
                stage1_matched=True,
                stage2_match=False,
            )
        self.stats["stage2_matches"] += 1

        # ---- the sustain window: `window` seconds of consecutive matches -------------
        # Measured in FOOTAGE time, not wall clock. On the live path they run together;
        # on replayed footage they do not, and the window is a claim about the event
        # ("a vehicle stopped for two minutes"), not about how long we watched.
        if rt.run_since is None:
            rt.run_since = chunk.t_start
        rt.run_until = chunk.t_end
        needed = self.settings.sustain_seconds(task.window)
        held = (rt.run_until - rt.run_since).total_seconds()
        if held < needed:
            return FunnelOutcome(
                chunk.chunk_id,
                task.task_id,
                2,
                f"sustaining {held:.0f}/{needed}s",
                stage1_score=score,
                stage1_matched=True,
                stage2_match=True,
            )

        if not rt.promoted:
            rt.promoted = True
            self.stats["promotions"] += 1
            logger.info(
                "task promoted",
                extra={
                    "fields": {
                        "task_id": task.task_id,
                        "chunk_id": chunk.chunk_id,
                        "stage1_score": round(score, 4),
                        "stage2_model": verdict.model,
                        "sustained_seconds": round(held, 2),
                        "window_seconds": needed,
                        "action": task.action.value,
                    }
                },
            )

        return self._act(task, chunk, rt, score, held, needed)

    # ----------------------------------------------------------------------------------
    # Acting — SPEC §6.3. Fire on stage 2; attach stage 3 afterwards.
    # ----------------------------------------------------------------------------------

    def _act(
        self,
        task: Task,
        chunk: ChunkRecord,
        rt: _Runtime,
        score: float,
        held: float,
        needed: int,
    ) -> FunnelOutcome:
        """Request the task's action for the whole sustained run.

        The range is the **run**, ``run_since .. run_until``, not the single chunk. It is
        what the evidence clip should contain, what the deep worker should re-watch, and —
        because it only ever grows while the run continues — it is what keeps the dedupe
        brake engaged for the rest of the event.

        Called on every sustained chunk, on purpose. Deciding here that we "already fired"
        would be a fourth brake, in the wrong module, with its own bugs.
        """
        assert rt.run_since is not None and rt.run_until is not None  # noqa: S101
        reason = (
            f"task {task.task_id}: {task.describe} | stage1 cosine {score:.3f} | "
            f"sustained {held:.0f}s of {needed}s | caption: {chunk.caption}"
        )
        result = self.actions.fire(
            task.action,
            rt.run_since,
            rt.run_until,
            task=task,
            reason=reason,
        )
        if not result.fired:
            self.stats["suppressed"] += 1
            return FunnelOutcome(
                chunk.chunk_id,
                task.task_id,
                3,
                f"suppressed by {result.brake.value if result.brake else '?'}: {result.detail}",
                stage1_score=score,
                stage1_matched=True,
                stage2_match=True,
                sustained=True,
                action=result,
            )

        self.stats["fired"] += 1
        entry = result.entry
        assert entry is not None  # noqa: S101 - fired implies an entry
        rt.last_fired_ts = entry.ts
        rt.last_entry_id = entry.entry_id

        job_id: str | None = None
        if result.awaits_verification and self.settings.verify_promoted:
            # Reaches a human (SPEC §6.3): the row is UNVERIFIED and we owe it a verdict.
            job_id = self._submit_verification(task, rt, entry.entry_id)
        elif not task.action.reaches_a_human:
            # save_clip is low stakes and complete on arrival. Not "pending" — done.
            rt.stage3_state = "idle"
            rt.stage3_verdict = None

        return FunnelOutcome(
            chunk.chunk_id,
            task.task_id,
            3,
            f"fired {task.action.value} as {entry.entry_id}",
            stage1_score=score,
            stage1_matched=True,
            stage2_match=True,
            sustained=True,
            action=result,
            job_id=job_id,
        )

    def _submit_verification(self, task: Task, rt: _Runtime, entry_id: str) -> str | None:
        """Queue stage 3. **Never blocks** — CLAUDE.md invariant 4, SPEC §6.3.

        A failure to submit is logged and swallowed. The action has already fired and
        cannot be un-fired; letting the worker's absence raise into the chunk loop would
        take down every other standing task to punish this one.
        """
        assert rt.run_since is not None and rt.run_until is not None  # noqa: S101
        question = self.settings.verify_question(task.describe)
        try:
            job = self.verifier.submit(rt.run_since, rt.run_until, question)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            rt.stage3_state = "failed"
            rt.stage3_job_id = None
            logger.error(
                "stage 3 submit failed; action stays unverified",
                extra={
                    "fields": {
                        "task_id": task.task_id,
                        "entry_id": entry_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                },
            )
            return None

        self.stats["verifications_submitted"] += 1
        rt.stage3_state = job.state.value if job.state is not JobState.DONE else "done"
        rt.stage3_job_id = job.job_id
        rt.stage3_verdict = None
        self._pending[job.job_id] = (task.task_id, entry_id)
        logger.info(
            "stage 3 submitted",
            extra={
                "fields": {
                    "task_id": task.task_id,
                    "entry_id": entry_id,
                    "job_id": job.job_id,
                    "t_start": rt.run_since.isoformat(),
                    "t_end": rt.run_until.isoformat(),
                }
            },
        )
        return job.job_id

    # ----------------------------------------------------------------------------------
    # Stage 3 landing — amendments are appends carrying parent_id, never mutations
    # ----------------------------------------------------------------------------------

    def pump_verifications(self) -> list[VerificationOutcome]:
        """Poll every outstanding stage-3 job and apply whatever has landed.

        Call it from the same loop that feeds ``observe``; it does no work when nothing is
        outstanding. A push-based worker can skip this entirely and call
        :meth:`apply_verification` directly.
        """
        outcomes: list[VerificationOutcome] = []
        for job_id in list(self._pending):
            job = self.verifier.poll(job_id)
            if job is None:
                continue
            outcome = self.apply_verification(job)
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    def apply_verification(self, job: DeepJob) -> VerificationOutcome | None:
        """Attach a finished worker verdict to the action it belongs to.

        Returns None while the job is still running, and for a job this monitor never
        submitted — M3 escalates jobs too (SPEC §4.2) and those are not ours to act on.

        The amendment is an **append** carrying ``parent_id`` (SPEC §11.4). The original
        row is never touched, which is what lets the Timeline render a retraction as the
        original struck through with the correction beneath it. Neither brake is released
        by a retraction: "we were wrong, so let us try again immediately" is the
        thirty-alerts failure mode with extra steps.
        """
        pending = self._pending.get(job.job_id)
        if pending is None:
            return None
        task_id, entry_id = pending
        rt = self._rt(task_id)

        if job.state in (JobState.QUEUED, JobState.RUNNING):
            rt.stage3_state = job.state.value
            return None

        verdict = self._verdict_fn(job)
        self._pending.pop(job.job_id, None)
        rt.stage3_state = "failed" if job.state in (JobState.FAILED, JobState.TIMEOUT) else "done"

        if verdict is None:
            # Inconclusive. The row stays exactly as written: UNVERIFIED, which is the
            # honest statement that nothing re-watched the footage successfully.
            rt.stage3_verdict = None
            logger.warning(
                "stage 3 inconclusive; action stays unverified",
                extra={
                    "fields": {
                        "task_id": task_id,
                        "entry_id": entry_id,
                        "job_id": job.job_id,
                        "job_state": job.state.value,
                        "confidence": job.confidence,
                        "error": job.error,
                    }
                },
            )
            return VerificationOutcome(
                task_id, entry_id, job.job_id, None, job.error or "no confidence returned"
            )

        detail = job.answer or job.reasoning
        if verdict:
            self.actions.verify(
                entry_id, reason=detail, clip_path=job.evidence_clip, job_id=job.job_id
            )
            rt.stage3_verdict = "verified"
        else:
            self.actions.retract(
                entry_id, reason=detail, clip_path=job.evidence_clip, job_id=job.job_id
            )
            rt.stage3_verdict = "retracted"

        logger.info(
            "stage 3 landed",
            extra={
                "fields": {
                    "task_id": task_id,
                    "entry_id": entry_id,
                    "job_id": job.job_id,
                    "verdict": rt.stage3_verdict,
                    "confidence": job.confidence,
                    "elapsed_s": round(job.elapsed, 2),
                }
            },
        )
        return VerificationOutcome(task_id, entry_id, job.job_id, rt.stage3_verdict, detail)

    # ----------------------------------------------------------------------------------
    # The Watch pane — SPEC §11.3. M3 owns the HTTP route; this is the payload.
    # ----------------------------------------------------------------------------------

    def registry_adapter(self) -> "_MonitorRegistry":
        """A ``TaskRegistry`` for M3, backed by THIS funnel.

        M3 and M5 share one task registry because they must: a task registered through
        the server has to be the same object the funnel evaluates, or the Watch pane
        offers a form that creates tasks nothing will ever match. This is the adapter
        that lets the server talk to M5's registry in the vocabulary its routes already
        use, without M3 importing M5's internals.
        """
        return _MonitorRegistry(self)

    def state(self) -> MonitorState:
        """Per-task funnel state, shaped per ``ui/mock/monitor_state.json``.

        Absolute timestamps only. ``last_fired_ts`` plus ``cooldown_seconds`` is enough
        for the UI to draw a cooldown that stays honest across a slow poll or a paused
        tab; ``remaining`` seconds computed here would be stale before they arrived.

        ``in_active_window`` is evaluated against *now*, not against the last chunk seen.
        A task falls out of window at 06:00 whether or not a chunk arrives to notice.
        """
        now = self._clock()
        rows: list[TaskFunnelState] = []
        for task in self.registry.tasks():
            rt = self._rt(task.task_id)
            in_window = self.registry.window_for(task.task_id).contains(now, self._tz)
            cooldown = float(task.cooldown)
            cooling = (
                rt.last_fired_ts is not None
                and (now - rt.last_fired_ts).total_seconds() < cooldown
            )
            rows.append(
                TaskFunnelState(
                    task_id=task.task_id,
                    state=_card_state(task, rt, in_window, cooling),
                    in_active_window=in_window,
                    stage1=Stage1State(
                        score=rt.stage1_score,
                        threshold=self.settings.stage1_cosine_threshold,
                        matched=rt.stage1_matched,
                        chunk_id=rt.stage1_chunk_id,
                    ),
                    stage2=Stage2State(
                        verdict=rt.stage2_verdict,
                        since=rt.run_since,
                        sustain_window_s=self.settings.sustain_seconds(task.window),
                        last_chunk_id=rt.stage2_last_chunk_id,
                    ),
                    stage3=Stage3State(
                        state=rt.stage3_state,
                        job_id=rt.stage3_job_id,
                        verdict=rt.stage3_verdict,
                    ),
                    last_fired_ts=rt.last_fired_ts,
                    cooldown_seconds=cooldown,
                    match_range=(
                        TimeRange(rt.run_since, rt.run_until)
                        if rt.run_since is not None and rt.run_until is not None
                        else None
                    ),
                )
            )
        return MonitorState(generated_at=now, tasks=tuple(rows))


def _card_state(task: Task, rt: _Runtime, in_window: bool, cooling: bool) -> str:
    """The one-word card state. Same precedence the Watch pane's badge uses.

    Cooling outranks matching deliberately: a task that is still matching *and* holding
    its brake is the SPEC §6.4 claim being demonstrated, and COOLING is the half of that
    the audience needs to read.
    """
    if not task.enabled:
        return "disabled"
    if not in_window:
        return "out_of_window"
    if cooling:
        return "cooling"
    if rt.stage2_verdict == "match":
        return "matching"
    return "armed"


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def build_monitor(
    *,
    settings: MonitorSettings | None = None,
    actions: ActionServer | None = None,
    verifier: DeepVerifier | None = None,
    load_seed: bool = True,
    clock: Callable[[], datetime] = utcnow,
) -> Monitor:
    """Construct M5 from ``config/settings.yaml``, seeded from ``config/tasks.yaml``.

    The embedder comes from ``index.embed`` so that a task description and the captions it
    is compared against are embedded by the *same* model — a task embedded by one and a
    corpus by another is a monitor that silently never matches (SPEC §3.4).

    Stage 3 defaults to :class:`~services.monitor.verify.NullVerifier` rather than
    reaching for M4: a monitor that refuses to start because the deep worker is not
    wired yet would be the wrong failure. Pass ``WorkerVerifier()`` once M4 is up.
    """
    resolved = settings or MonitorSettings.from_config()
    embedder = build_embedder(IndexSettings.from_config())
    registry = TaskRegistry(embedder)
    if load_seed:
        registry.load_seed(resolved.tasks_file)
    return Monitor(
        registry=registry,
        actions=actions or ActionServer(clock=clock),
        confirmer=build_confirmer(resolved),
        embedder=embedder,
        settings=resolved,
        verifier=verifier,
        clock=clock,
    )


class _MonitorRegistry:
    """``TaskRegistry`` over a live :class:`Monitor` — see ``Monitor.registry_adapter``.

    Every write goes to the funnel's own registry, so registering, editing or deleting a
    task through the UI takes effect on the very next chunk. ``monitor_state`` returns
    the real funnel state rather than the idle placeholder M3 falls back to when M5 is
    not running.
    """

    def __init__(self, monitor: "Monitor") -> None:
        self._monitor = monitor

    @property
    def _registry(self) -> TaskRegistry:
        return self._monitor.registry

    def tasks(self) -> list[Task]:
        return self._registry.tasks()

    def register(self, task: Task) -> Task:
        return self._monitor.register_task(task)

    def remove(self, task_id: str) -> Task:
        return self._registry.remove(task_id)

    def update(self, task_id: str, changes: Mapping[str, Any]) -> Task:
        return self._registry.update(task_id, changes)

    def monitor_state(self) -> dict[str, Any]:
        return self._monitor.state().to_dict()
