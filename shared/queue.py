"""The priority queue in front of the VLM (SPEC §7).

There is one VLM process (CLAUDE.md invariant 1) and one camera, so there is exactly one
request in flight (``vlm.queue.max_inflight``). Everything that wants the model queues
here. Priority order, from ``vlm.queue.priorities``:

1. ``interactive``   — M3/M4 working on a user's turn
2. ``verification``  — M5 stage 3, the deep re-watch behind an alert
3. ``ingest``        — M1 captioning

**The bug this module exists to avoid.** A strict-priority queue is the obvious
implementation and it is wrong: under sustained interactive load the ingest lane never
reaches the head, captions stop arriving, and M5 — which matches standing tasks against
captions — goes quietly blind. SPEC §7 words it as *ingest may be paused, never
starved*.

So the rule here is **strict priority with a bounded pause**:

* Normally the highest-priority non-empty lane wins, FIFO within a lane.
* An item whose lane has a pause cap and which has waited at least that long is
  *overdue*, and overdue items win outright — highest-priority-overdue first, and among
  equals the one that has waited longest. ``vlm.queue.max_ingest_pause_seconds`` (5 s)
  is that cap for ingest.

Two consequences worth stating, because they are the whole point:

* An ingest chunk's wait is bounded by ``max_ingest_pause_seconds`` plus the remainder of
  whatever is already in flight. The second term is irreducible with
  ``max_inflight = 1`` and a 20–60 s deep request; it is a property of the hardware, not
  of the policy.
* Promotion is per-item, not per-lane, so a backlogged ingest lane keeps winning until
  its head is younger than the cap — which is exactly "let it catch up" — and then
  yields. No lane is ever handed the queue wholesale.

Aging changes *which* item runs, never *whether* one does, so no timers or polling are
needed: a worker only ever blocks when the queue is genuinely empty.

Usage::

    queue = VLMQueue()
    queue.start()
    job = queue.submit(Priority.INGEST, lambda: client.caption([chunk], prompt=P))
    caption = job.result(timeout=30)

:class:`PriorityPolicy` is the ordering rule on its own — pure, no threads, injected
clock — so the starvation behaviour can be tested as arithmetic rather than as a race.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from shared import config

__all__ = [
    "Priority",
    "JobState",
    "Job",
    "PriorityPolicy",
    "VLMQueue",
    "QueueClosed",
    "QueueTimeout",
]

LOGGER = logging.getLogger("shared.queue")


class Priority(str, Enum):
    """SPEC §7. The *names* are code; the *order* is config (``vlm.queue.priorities``)."""

    INTERACTIVE = "interactive"
    VERIFICATION = "verification"
    INGEST = "ingest"


class JobState(str, Enum):
    """Lifecycle of one queued unit of VLM work."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueueClosed(RuntimeError):
    """Submitted to a queue that has been stopped."""


class QueueTimeout(TimeoutError):
    """A caller waited longer than it was willing to."""


# --------------------------------------------------------------------------------------
# The ordering rule — pure, testable, no threads
# --------------------------------------------------------------------------------------


class Waiting(Protocol):
    """What the policy needs from an item: which lane, and since when."""

    priority: Priority
    enqueued_at: float


_W = TypeVar("_W", bound=Waiting)


class PriorityPolicy(Generic[_W]):
    """Strict priority with a bounded pause per lane. See the module docstring.

    ``pause_caps`` maps a priority to the number of seconds an item in that lane may wait
    before it outranks everything. ``None`` (or absence) means no cap — that lane is
    served by strict priority alone, which is correct for the lanes above ingest because
    starving *them* is the intended behaviour of a priority queue.

    ``enqueued_at`` is supplied by the caller rather than read from the clock here, so a
    simulation can replay a whole minute of arrivals without waiting a minute.
    """

    def __init__(
        self,
        priorities: Sequence[Priority],
        pause_caps: Mapping[Priority, float | None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        order = list(priorities)
        if sorted(p.value for p in order) != sorted(p.value for p in Priority):
            raise config.ConfigError(
                f"vlm.queue.priorities must list every priority exactly once "
                f"{[p.value for p in Priority]}; got {[p.value for p in order]}"
            )
        self._order: tuple[Priority, ...] = tuple(order)
        self._caps = {p: pause_caps.get(p) for p in self._order}
        self._clock = clock
        self._lanes: dict[Priority, deque[_W]] = {p: deque() for p in self._order}

    @property
    def order(self) -> tuple[Priority, ...]:
        return self._order

    def cap(self, priority: Priority) -> float | None:
        return self._caps[priority]

    def enqueue(self, item: _W) -> None:
        self._lanes[item.priority].append(item)

    def depth(self, priority: Priority | None = None) -> int:
        if priority is None:
            return sum(len(lane) for lane in self._lanes.values())
        return len(self._lanes[priority])

    def peek(self, priority: Priority) -> _W | None:
        lane = self._lanes[priority]
        return lane[0] if lane else None

    def overdue(self, now: float | None = None) -> list[Priority]:
        """Lanes whose head has exhausted its pause budget, worst-starved first."""
        now = self._clock() if now is None else now
        starved: list[tuple[float, int, Priority]] = []
        for rank, priority in enumerate(self._order):
            cap = self._caps[priority]
            head = self.peek(priority)
            if cap is None or head is None:
                continue
            waited = now - head.enqueued_at
            if waited >= cap:
                # Sort key: longest wait first, then priority order to break ties.
                starved.append((-waited, rank, priority))
        starved.sort()
        return [priority for _, _, priority in starved]

    def select(self, now: float | None = None) -> _W | None:
        """Pop the item that should run next, or None if nothing is queued."""
        now = self._clock() if now is None else now
        for priority in self.overdue(now):
            return self._lanes[priority].popleft()
        for priority in self._order:
            lane = self._lanes[priority]
            if lane:
                return lane.popleft()
        return None

    def remove(self, item: _W) -> bool:
        """Withdraw a still-queued item. Used by :meth:`Job.cancel`."""
        lane = self._lanes[item.priority]
        try:
            lane.remove(item)
        except ValueError:
            return False
        return True

    def drain(self) -> list[_W]:
        """Empty every lane, in the order the items would have run."""
        out: list[_W] = []
        while (item := self.select()) is not None:
            out.append(item)
        return out


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------

_T = TypeVar("_T")
_JOB_SEQ = itertools.count(1)


@dataclass(eq=False)
class Job(Generic[_T]):
    """One unit of VLM work and its eventual result.

    Compared by identity (``eq=False``): two ingest captions with the same label are
    different jobs, and the policy removes by identity.
    """

    priority: Priority
    fn: Callable[[], _T]
    label: str = ""
    job_id: str = field(default_factory=lambda: f"q{next(_JOB_SEQ):06d}")
    enqueued_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    state: JobState = JobState.QUEUED
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _value: Any = field(default=None, repr=False)
    _error: BaseException | None = field(default=None, repr=False)

    @property
    def waited(self) -> float | None:
        """Seconds spent queued, once it has started. The starvation metric."""
        if self.started_at is None:
            return None
        return self.started_at - self.enqueued_at

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> _T:
        """Block until done. Re-raises whatever the callable raised."""
        if not self._done.wait(timeout):
            raise QueueTimeout(f"job {self.job_id} ({self.label!r}) not finished in {timeout}s")
        if self._error is not None:
            raise self._error
        return self._value  # type: ignore[return-value]

    # -- called by the queue only --------------------------------------------------

    def _finish(self, value: Any, error: BaseException | None, now: float) -> None:
        self._value = value
        self._error = error
        self.finished_at = now
        self.state = JobState.FAILED if error is not None else JobState.DONE
        self._done.set()

    def _cancel(self) -> None:
        self.state = JobState.CANCELLED
        self._error = QueueClosed(f"job {self.job_id} was cancelled")
        self._done.set()


# --------------------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------------------


class VLMQueue:
    """Admission control in front of the single VLM process.

    Holds ``max_inflight`` worker threads (1, per ``vlm.queue.max_inflight``) that pull
    from a :class:`PriorityPolicy`. Callers submit a zero-argument callable — usually a
    ``VLMClient`` call — and get a :class:`Job` back.

    The queue does not know what a VLM is on purpose: it schedules callables, so the
    deep worker, ingest and M5 can all be scheduled against the same one-in-flight budget
    without this module importing any of them.
    """

    def __init__(
        self,
        *,
        priorities: Sequence[Priority] | None = None,
        max_inflight: int | None = None,
        max_ingest_pause_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        order = priorities if priorities is not None else _priorities_from_config()
        cap = (
            max_ingest_pause_seconds
            if max_ingest_pause_seconds is not None
            else float(config.get("vlm.queue.max_ingest_pause_seconds"))
        )
        self._max_inflight = int(
            max_inflight if max_inflight is not None else config.get("vlm.queue.max_inflight")
        )
        if self._max_inflight < 1:
            raise config.ConfigError("vlm.queue.max_inflight must be >= 1")

        self._clock = clock
        self._log = logger or LOGGER
        self._log_json = str(config.get("logging.format", "json")) == "json"
        # Only ingest gets a pause cap. Starving the lanes above it is what a priority
        # queue is *for*; starving ingest is the failure mode (SPEC §7).
        self._policy: PriorityPolicy[Job[Any]] = PriorityPolicy(
            order, {Priority.INGEST: cap}, clock=clock
        )
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._workers: list[threading.Thread] = []
        self._inflight = 0
        self._stopping = False
        self._drain_on_stop = False
        self._promotions = 0
        self._completed: dict[Priority, int] = {p: 0 for p in order}
        self._max_wait: dict[Priority, float] = {p: 0.0 for p in order}

    # -- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        """Spin up the worker threads. Idempotent."""
        with self._lock:
            if self._workers:
                return
            self._stopping = False
            for i in range(self._max_inflight):
                thread = threading.Thread(
                    target=self._serve, name=f"vlm-queue-{i}", daemon=True
                )
                self._workers.append(thread)
                thread.start()

    def stop(self, *, drain: bool = False, timeout: float | None = 5.0) -> None:
        """Stop the workers. ``drain=True`` finishes what is queued first."""
        with self._lock:
            self._stopping = True
            self._drain_on_stop = drain
            if not drain:
                for job in self._policy.drain():
                    job._cancel()
            self._wake.notify_all()
            workers = list(self._workers)
            self._workers.clear()
        for thread in workers:
            thread.join(timeout)

    def __enter__(self) -> VLMQueue:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop(drain=False)

    # -- submission ---------------------------------------------------------------

    def submit(
        self,
        priority: Priority | str,
        fn: Callable[[], _T],
        *,
        label: str = "",
    ) -> Job[_T]:
        """Queue a callable. Returns immediately with a :class:`Job` handle."""
        prio = Priority(priority)
        job: Job[_T] = Job(priority=prio, fn=fn, label=label)
        with self._lock:
            if self._stopping:
                raise QueueClosed("queue is stopping; no new work accepted")
            job.enqueued_at = self._clock()
            self._policy.enqueue(job)  # type: ignore[arg-type]
            self._wake.notify()
        return job

    def cancel(self, job: Job[Any]) -> bool:
        """Withdraw a job that has not started. Returns False if it is already running."""
        with self._lock:
            if job.state is not JobState.QUEUED or not self._policy.remove(job):
                return False
            job._cancel()
            return True

    # -- introspection ------------------------------------------------------------

    def depth(self, priority: Priority | str | None = None) -> int:
        with self._lock:
            return self._policy.depth(None if priority is None else Priority(priority))

    def stats(self) -> dict[str, Any]:
        """Queue health. The promotion count is how you see the brake working."""
        with self._lock:
            return {
                "inflight": self._inflight,
                "depth": {p.value: self._policy.depth(p) for p in self._policy.order},
                "completed": {p.value: n for p, n in self._completed.items()},
                "max_wait_s": {p.value: round(w, 3) for p, w in self._max_wait.items()},
                "ingest_promotions": self._promotions,
                "max_ingest_pause_seconds": self._policy.cap(Priority.INGEST),
            }

    # -- the worker loop ----------------------------------------------------------

    def _serve(self) -> None:
        while True:
            with self._lock:
                while True:
                    if self._stopping and not self._drain_on_stop:
                        return
                    now = self._clock()
                    was_overdue = Priority.INGEST in self._policy.overdue(now)
                    job = self._policy.select(now)
                    if job is not None:
                        break
                    if self._stopping:
                        return
                    # No timeout: aging changes which item wins, never whether one is
                    # available, so there is nothing to wake up *for* until a submit or
                    # a stop — and both notify.
                    self._wake.wait()
                job.state = JobState.RUNNING
                job.started_at = self._clock()
                self._inflight += 1
                waited = job.started_at - job.enqueued_at
                if self._max_wait[job.priority] < waited:
                    self._max_wait[job.priority] = waited
                promoted = was_overdue and job.priority is Priority.INGEST
                if promoted:
                    self._promotions += 1

            if promoted:
                # The brake firing is exactly the thing we cannot tune if we cannot see
                # it: it says ingest hit its pause cap and jumped the queue.
                self._emit(
                    "ingest_promoted",
                    job_id=job.job_id,
                    label=job.label,
                    waited_s=round(waited, 3),
                    cap_s=self._policy.cap(Priority.INGEST),
                )

            value: Any = None
            error: BaseException | None = None
            try:
                value = job.fn()
            except BaseException as exc:  # noqa: BLE001 - the caller owns this failure
                error = exc

            with self._lock:
                self._inflight -= 1
                self._completed[job.priority] += 1
                job._finish(value, error, self._clock())
                self._wake.notify_all()

            if error is not None:
                self._emit(
                    "job_failed",
                    job_id=job.job_id,
                    priority=job.priority.value,
                    label=job.label,
                    error=repr(error),
                )

    def _emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, **fields}
        line = (
            json.dumps(record, sort_keys=True, default=str)
            if self._log_json
            else " ".join(f"{k}={v}" for k, v in record.items())
        )
        self._log.info(line)


def _priorities_from_config() -> list[Priority]:
    """Read ``vlm.queue.priorities``, highest first."""
    raw = config.get("vlm.queue.priorities")
    if not isinstance(raw, list) or not raw:
        raise config.ConfigError("vlm.queue.priorities must be a non-empty list")
    try:
        return [Priority(str(name)) for name in raw]
    except ValueError as exc:
        raise config.ConfigError(
            f"vlm.queue.priorities contains an unknown priority: {exc}. "
            f"Known: {[p.value for p in Priority]}"
        ) from exc
