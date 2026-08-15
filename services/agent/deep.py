"""Escalation plumbing — SPEC §4.3, CLAUDE.md invariant 4.

    answer provisionally  →  return job_id  →  stream refined answer over WebSocket

**Nothing in this module is ever called on a user's turn except** :meth:`JobRegistry.request`,
which allocates a job, hands it to a worker thread and returns immediately. The blocking
part of deep analysis happens on that thread, behind the registry, which is what makes
invariant 4 structural instead of a rule someone has to remember.

Three backstops from SPEC §4.3, all here:

* **One deep job in flight** (``agent.deep.max_inflight``) — enforced by the number of
  executor threads, so a second escalation queues rather than contending for the single
  VLM process (CLAUDE.md invariant 1).
* **Dedupe identical ranges** — an impatient user clicking twice must not queue the work
  twice. The key is the requested footage range, per ``DeepJob.requested_range``.
* **A 90 s timeout, stated to the user** (``agent.deep.timeout_seconds``). The job is
  marked ``TIMEOUT`` and the refinement says so; a job that quietly never arrives reads
  as a hung UI.

M4 is built concurrently, so :class:`WorkerAnalyzer` binds to ``services.worker`` **at
first use, never at import**. Its contract is SPEC §5's ``deep_analyze(t_start, t_end,
question) -> DeepJob`` plus a non-blocking ``submit(...)`` returning a QUEUED job. Both
shapes are supported and the choice is logged, because "which path did it take" is the
first question when a refinement does not land.
"""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from shared.schema import DeepJob, JobState, to_iso, utcnow

from .settings import AgentSettings
from .telemetry import log_event

__all__ = [
    "DeepAnalyzer",
    "WorkerAnalyzer",
    "UnavailableAnalyzer",
    "JobRegistry",
    "JobUpdate",
    "range_key",
]


def range_key(t_start: datetime, t_end: datetime) -> str:
    """The SPEC §4.3 dedupe key: the requested footage range, to the microsecond."""
    return f"{to_iso(t_start)}|{to_iso(t_end)}"


@runtime_checkable
class DeepAnalyzer(Protocol):
    """M4, as M3 needs it. Injected, so tests never import the worker."""

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        """Queue the work and return a QUEUED job **immediately**."""
        ...

    def result(self, job: DeepJob, timeout_s: float) -> DeepJob:
        """Block on the executor thread until the job resolves. Never called on a turn."""
        ...


@dataclass(frozen=True)
class JobUpdate:
    """One state change, fanned out to whoever is watching (the WS hub, the chat log)."""

    job: DeepJob
    turn_ids: tuple[str, ...]


class UnavailableAnalyzer:
    """What escalation does before M4 exists.

    It fails the job with a sentence rather than raising into the user's turn: the
    provisional answer and the groundedness verdict are still correct and still the
    point, and "the deep worker is not running" is a fact worth rendering rather than a
    500 that loses the turn.
    """

    def __init__(self, detail: str = "the deep worker (M4) is not available") -> None:
        self.detail = detail

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        return DeepJob(
            job_id=uuid.uuid4().hex[:4],
            t_start=t_start,
            t_end=t_end,
            question=question,
            state=JobState.QUEUED,
        )

    def result(self, job: DeepJob, timeout_s: float) -> DeepJob:
        return replace(
            job, state=JobState.FAILED, completed_at=utcnow(), error=self.detail
        )


class WorkerAnalyzer:
    """Lazy binding to ``services/worker`` (M4). Imported at first use, never at import.

    Three shapes are accepted, in this order:

    1. ``default_worker()`` returning an object with ``submit`` and ``wait``/``poll`` —
       M4 owns the queue, the deadline and the events, and we just watch. Preferred:
       one queue in front of the one VLM process is better than two.
    2. Module-level ``submit(...)`` plus a module-level poll — same idea, flatter API.
    3. ``deep_analyze(...)`` alone — we call it from *our* executor thread. Blocking
       there is fine and is never on a user turn; blocking is only forbidden on the turn
       (invariant 4).

    A module that offers none of them raises :class:`RuntimeError` on first use, not on
    import, so M3 comes up and serves the ask surface while M4 is still being written.
    """

    _POLL_NAMES = ("wait", "poll", "get_job", "job", "result")

    def __init__(
        self,
        module: Any | None = None,
        *,
        poll_interval_s: float = 0.25,
    ) -> None:
        self._module = module
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()

    def _module_or_import(self) -> Any:
        with self._lock:
            if self._module is None:
                import services.worker as module  # noqa: PLC0415 — deferred; M4 lands later

                self._module = module
            return self._module

    def _target(self) -> tuple[Any, str | None]:
        """The object M4's jobs live on, and the name of its poll (if it has one).

        ``default_worker()`` is preferred because that is where M4 keeps the queue, the
        90 s deadline and the completion events — polling a flat module function would
        be a second, worse copy of all three.
        """
        module = self._module_or_import()
        factory = getattr(module, "default_worker", None)
        if callable(factory):
            worker = factory()
            poll = self._poll_name(worker)
            if callable(getattr(worker, "submit", None)) and poll:
                return worker, poll
        return module, self._poll_name(module)

    def _poll_name(self, target: Any) -> str | None:
        for name in self._POLL_NAMES:
            if callable(getattr(target, name, None)):
                return name
        return None

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        target, poll_name = self._target()
        submit = getattr(target, "submit", None)
        if callable(submit) and poll_name:
            job = submit(t_start, t_end, question)
            if not isinstance(job, DeepJob):
                raise TypeError(f"worker.submit returned {type(job).__name__}, not DeepJob")
            log_event("agent.deep.bound", path="submit+" + poll_name, job_id=job.job_id)
            return job
        if not callable(getattr(self._module_or_import(), "deep_analyze", None)):
            raise RuntimeError(
                "services.worker exposes neither deep_analyze(t_start, t_end, question) "
                "nor submit(...) with a poll; SPEC §5 requires the former"
            )
        # Path 3: we own the job record until deep_analyze returns one.
        log_event("agent.deep.bound", path="deep_analyze")
        return DeepJob(
            job_id=uuid.uuid4().hex[:4],
            t_start=t_start,
            t_end=t_end,
            question=question,
            state=JobState.QUEUED,
        )

    def result(self, job: DeepJob, timeout_s: float) -> DeepJob:
        target, poll_name = self._target()
        if callable(getattr(target, "submit", None)) and poll_name:
            return self._await(target, poll_name, job, timeout_s)
        finished = self._module_or_import().deep_analyze(job.t_start, job.t_end, job.question)
        return self._adopt(job, finished)

    def _await(self, target: Any, poll_name: str, job: DeepJob, timeout_s: float) -> DeepJob:
        """Wait on M4's own event if it has one; poll only when it does not."""
        if poll_name == "wait":
            current = target.wait(job.job_id, timeout_s)
            return current if isinstance(current, DeepJob) else job
        poll = getattr(target, poll_name)
        deadline = utcnow() + timedelta(seconds=timeout_s)
        while utcnow() < deadline:
            current = poll(job.job_id)
            if isinstance(current, DeepJob) and current.state in (
                JobState.DONE,
                JobState.FAILED,
                JobState.TIMEOUT,
            ):
                return current
            threading.Event().wait(self._poll_interval_s)
        return replace(job, state=JobState.TIMEOUT, completed_at=utcnow())

    @staticmethod
    def _adopt(job: DeepJob, finished: Any) -> DeepJob:
        """Fold a blocking ``deep_analyze`` return onto the job id we already published.

        The user has been shown ``job.job_id``; the worker's own id (if any) must not
        replace it, or the WebSocket refinement addresses a card that does not exist.
        """
        if isinstance(finished, DeepJob):
            return replace(finished, job_id=job.job_id, requested_at=job.requested_at)
        if isinstance(finished, dict):
            # SPEC §5 states the return as ``{answer, evidence_clip, confidence}``.
            return replace(
                job,
                state=JobState.DONE,
                completed_at=utcnow(),
                answer=str(finished.get("answer", "")),
                reasoning=str(finished.get("reasoning", "")),
                confidence=finished.get("confidence"),
                evidence_clip=finished.get("evidence_clip"),
            )
        raise TypeError(f"deep_analyze returned {type(finished).__name__}, not DeepJob or dict")


class JobRegistry:
    """Owns every deep job M3 has requested: dedupe, in-flight cap, timeout, fan-out.

    Thread model: ``request`` runs on the request thread and only ever touches state
    under a lock; ``agent.deep.max_inflight`` executor threads do the waiting. The user's
    turn returns as soon as the job record exists.
    """

    def __init__(
        self,
        analyzer: DeepAnalyzer,
        settings: AgentSettings,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._analyzer = analyzer
        self._s = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, DeepJob] = {}
        self._turns: dict[str, list[str]] = {}
        self._inflight: dict[str, str] = {}  # range key -> job_id
        self._listeners: list[Callable[[JobUpdate], None]] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._started = False

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the executor threads. Idempotent."""
        with self._lock:
            if self._started:
                return
            self._started = True
            for i in range(max(1, self._s.deep_max_inflight)):
                thread = threading.Thread(
                    target=self._run, name=f"deep-executor-{i}", daemon=True
                )
                thread.start()
                self._threads.append(thread)

    def stop(self, timeout_s: float = 5.0) -> None:
        """Ask the executors to finish. A job already running is allowed to land."""
        with self._lock:
            if not self._started:
                return
            self._started = False
            threads = list(self._threads)
            self._threads.clear()
        for _ in threads:
            self._queue.put(None)
        for thread in threads:
            thread.join(timeout=timeout_s)

    def subscribe(self, listener: Callable[[JobUpdate], None]) -> None:
        """Register a fan-out target — the WebSocket hub, the chat log."""
        with self._lock:
            self._listeners.append(listener)

    # -- reads ----------------------------------------------------------------------

    def job(self, job_id: str) -> DeepJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def jobs(self) -> dict[str, DeepJob]:
        with self._lock:
            return dict(self._jobs)

    def turns_for(self, job_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._turns.get(job_id, ()))

    # -- the only method a user turn touches ----------------------------------------

    def request(
        self, t_start: datetime, t_end: datetime, question: str, *, turn_id: str
    ) -> tuple[DeepJob, str | None]:
        """Escalate a range. Returns ``(job, dedupe_of)`` **immediately**.

        ``dedupe_of`` is the job id an identical range is already running under, in which
        case no new work is queued and the same job will refine both turns. SPEC §11.2
        wants that said out loud — a silent no-op reads as a bug in rehearsal.
        """
        key = range_key(t_start, t_end)
        with self._lock:
            existing_id = self._inflight.get(key) if self._s.deep_dedupe_identical_ranges else None
            if existing_id is not None:
                self._turns.setdefault(existing_id, []).append(turn_id)
                existing = self._jobs[existing_id]
                log_event(
                    "agent.deep.dedupe",
                    job_id=existing_id,
                    turn_id=turn_id,
                    range=key,
                    state=existing.state.value,
                )
                return existing, existing_id

        # submit() is M4's non-blocking entry point; it returns a QUEUED job. It is
        # called outside the lock so a slow worker import cannot stall another turn.
        try:
            job = self._analyzer.submit(t_start, t_end, question)
        except Exception as exc:  # noqa: BLE001 — a worker that cannot start is one dead job
            # The provisional answer and the groundedness verdict are already correct and
            # are the point of the turn. Losing them to a 500 because M4 is down would
            # trade the demo's thesis for its footnote.
            failed = DeepJob(
                job_id=uuid.uuid4().hex[:4],
                t_start=t_start,
                t_end=t_end,
                question=question,
                state=JobState.FAILED,
                completed_at=self._clock(),
                error=f"could not queue deep analysis: {type(exc).__name__}: {exc}",
            )
            with self._lock:
                self._jobs[failed.job_id] = failed
                self._turns.setdefault(failed.job_id, []).append(turn_id)
            log_event(
                "agent.deep.submit_failed",
                job_id=failed.job_id,
                turn_id=turn_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return failed, None

        with self._lock:
            self._jobs[job.job_id] = job
            self._turns.setdefault(job.job_id, []).append(turn_id)
            self._inflight[key] = job.job_id
        log_event(
            "agent.deep.requested",
            job_id=job.job_id,
            turn_id=turn_id,
            t_start=to_iso(t_start),
            t_end=to_iso(t_end),
            seconds=round((t_end - t_start).total_seconds(), 2),
            timeout_seconds=self._s.deep_timeout_seconds,
            question=question,
        )
        self.start()
        self._queue.put(job.job_id)
        return job, None

    # -- executor -------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._execute(job_id)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        job = self.job(job_id)
        if job is None:  # pragma: no cover — only reachable if stop() raced a request
            return
        self._publish(replace(job, state=JobState.RUNNING))
        started = self._clock()
        try:
            finished = self._analyzer.result(
                self.job(job_id) or job, self._s.deep_timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 — a worker failure is one failed job
            log_event("agent.deep.failed", job_id=job_id, error=f"{type(exc).__name__}: {exc}")
            self._publish(
                replace(
                    self.job(job_id) or job,
                    state=JobState.FAILED,
                    completed_at=self._clock(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return

        elapsed = (self._clock() - started).total_seconds()
        if elapsed > self._s.deep_timeout_seconds and finished.state is JobState.DONE:
            # The worker answered, but past the deadline the user was shown. Say so
            # rather than pretending the timeout did not happen — SPEC §4.3 requires the
            # timeout be stated, and a 120 s "34 s" answer is how a demo loses trust.
            finished = replace(
                finished,
                state=JobState.TIMEOUT,
                error=(
                    f"deep analysis took {elapsed:.1f}s, past the "
                    f"{self._s.deep_timeout_seconds:.0f}s timeout"
                ),
            )
        if finished.completed_at is None:
            finished = replace(finished, completed_at=self._clock())
        self._publish(finished)

    def _publish(self, job: DeepJob) -> None:
        terminal = job.state in (JobState.DONE, JobState.FAILED, JobState.TIMEOUT)
        with self._lock:
            self._jobs[job.job_id] = job
            turn_ids = tuple(self._turns.get(job.job_id, ()))
            listeners = list(self._listeners)
            if terminal:
                # Release the dedupe slot only when the work is over: while it runs, a
                # second identical ask must ride along rather than queue a twin.
                for key, held in list(self._inflight.items()):
                    if held == job.job_id:
                        del self._inflight[key]
        log_event(
            "agent.deep.state",
            job_id=job.job_id,
            state=job.state.value,
            elapsed_s=round(job.elapsed, 2),
            turn_ids=list(turn_ids),
            error=job.error,
        )
        update = JobUpdate(job=job, turn_ids=turn_ids)
        for listener in listeners:
            try:
                listener(update)
            except Exception as exc:  # noqa: BLE001 — one bad listener must not eat the rest
                log_event(
                    "agent.deep.listener_failed",
                    job_id=job.job_id,
                    error=f"{type(exc).__name__}: {exc}",
                )

    # -- restoring state ------------------------------------------------------------

    def adopt(self, jobs: Iterable[DeepJob], turns: dict[str, list[str]] | None = None) -> None:
        """Load persisted jobs back in (SPEC §11.4) without re-running any of them.

        A refinement that landed before a restart is history, not work: it is served
        from ``/api/chat/history`` and must not re-enter the queue.
        """
        with self._lock:
            for job in jobs:
                self._jobs[job.job_id] = job
            for job_id, turn_ids in (turns or {}).items():
                self._turns.setdefault(job_id, []).extend(turn_ids)
