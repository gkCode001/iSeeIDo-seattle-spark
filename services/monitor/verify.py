"""Stage 3 — worker verify. SPEC §6.2 (20–60 s) and SPEC §6.3 (non-blocking).

Captions are a 1 fps guess with an 80-token budget. Stage 3 re-watches the actual footage
at 4 fps, native resolution, with reasoning on — which is why a ticket is never filed on a
caption alone.

**Stage 3 is not a blocking precondition.** The action already fired on stage-2
confidence; this attaches a verdict afterwards, exactly as M3 attaches a refinement to a
provisional answer. One pattern, both surfaces (CLAUDE.md invariant 4, SPEC §6.3).

M4 (``services/worker``) is being built in parallel and this module deliberately does not
import it at module scope: :class:`WorkerVerifier` resolves it lazily on first use, so M5
and its tests neither need M4's file to exist nor pay for whatever it imports. The
contract we bind to is the fixed one from SPEC §5 — ``deep_analyze(t_start, t_end,
question) -> DeepJob`` plus a non-blocking ``submit`` — narrowed here to the two calls M5
actually makes.

The verdict rule is separate from the transport on purpose. Whether a ``DeepJob`` counts
as agreement is a policy question, and it is the one that decides whether a human-facing
alert gets retracted, so it is a named, injectable function rather than three lines
buried in the funnel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from shared.schema import DeepJob, JobState

__all__ = [
    "DeepVerifier",
    "NullVerifier",
    "WorkerVerifier",
    "VerdictFn",
    "confidence_verdict",
]

logger = logging.getLogger("monitor.verify")


class DeepVerifier(Protocol):
    """The two calls M5 makes against M4. Everything else about M4 is M4's business."""

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        """Queue a deep analysis and return immediately. **Must not block.**

        Blocking here blocks the chunk loop, which stops captions arriving, which stops
        every other standing task — a 60 s worker call would cost the monitor a minute of
        blindness per promotion.
        """
        ...

    def poll(self, job_id: str) -> DeepJob | None:
        """Current state of a submitted job, or None if the worker has forgotten it."""
        ...


#: Job → agreement. True verifies the action, False retracts it, **None is inconclusive**
#: and leaves the row exactly as it was written.
VerdictFn = Callable[[DeepJob], "bool | None"]


def confidence_verdict(job: DeepJob, *, threshold: float) -> bool | None:
    """Default policy: trust the worker's confidence, and only that.

    Three outcomes, and the third is the one that matters:

    * confidence at or above ``threshold`` → verified;
    * confidence below it → retracted;
    * **no confidence at all → inconclusive.** A job that failed, timed out, or came back
      without a number has told us nothing about the footage. Retracting on that would
      turn every worker hiccup into a public "we were wrong" on the Watch pane, and would
      teach a viewer that retractions are noise — which is exactly backwards, because the
      visible retraction is the thesis (SPEC §11.4).

    Deliberately no keyword matching on ``answer``. Reading "no vehicle is visible" out of
    free text is the sort of rule that works on the three sentences it was written against
    and inverts on the fourth, and it would be inverting the decision to retract a human's
    alert.
    """
    if job.state in (JobState.FAILED, JobState.TIMEOUT):
        return None
    if job.state is not JobState.DONE:
        return None
    if job.confidence is None:
        return None
    return float(job.confidence) >= threshold


class NullVerifier:
    """Stage 3 is not wired. Refuses submissions loudly and has no verdicts to give.

    The default when M4 is absent, and what ``monitor.verify_promoted: false`` effectively
    selects. It raises rather than silently accepting, so the log says "stage 3 is not
    wired" once per promotion instead of leaving a row that looks like it is waiting for a
    worker that was never asked. The funnel catches this: actions still fire on stage-2
    confidence (SPEC §6.3) and simply stay ``UNVERIFIED``, which is the honest rendering
    of "nothing re-watched this".
    """

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        raise RuntimeError(
            "stage 3 is not wired: no deep verifier was provided and "
            "monitor.verify_promoted asked for one"
        )

    def poll(self, job_id: str) -> DeepJob | None:
        return None


class WorkerVerifier:
    """Adapter onto M4, resolved lazily so M5 does not depend on its file existing.

    Accepts either a module or any object exposing the SPEC §5 surface. Resolution is
    duck-typed against a small set of names because M4 is in flight; the failure, if it
    comes, is a single readable error naming SPEC §5 rather than an AttributeError three
    frames down inside the chunk loop.
    """

    _SUBMIT_NAMES = ("submit", "submit_deep_analyze", "submit_job")
    _POLL_NAMES = ("poll", "job", "get_job", "job_status")

    def __init__(self, worker: Any | None = None, *, module: str = "services.worker") -> None:
        self._worker = worker
        self._module = module
        self._submit: Callable[..., DeepJob] | None = None
        self._poll: Callable[[str], DeepJob | None] | None = None

    def _resolve(self) -> Any:
        if self._worker is None:
            import importlib  # noqa: PLC0415 - deferred; M4 may not exist yet

            self._worker = importlib.import_module(self._module)
        return self._worker

    def _bind(self) -> None:
        if self._submit is not None:
            return
        target = self._resolve()
        for name in self._SUBMIT_NAMES:
            fn = getattr(target, name, None)
            if callable(fn):
                self._submit = fn
                break
        if self._submit is None:
            raise RuntimeError(
                f"{self._module} exposes no non-blocking submit; SPEC §5 requires "
                f"deep_analyze(t_start, t_end, question) -> DeepJob plus a submit(...) "
                f"that returns immediately. Tried: {', '.join(self._SUBMIT_NAMES)}"
            )
        for name in self._POLL_NAMES:
            fn = getattr(target, name, None)
            if callable(fn):
                self._poll = fn
                break

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        self._bind()
        assert self._submit is not None  # noqa: S101 - _bind raises otherwise
        return self._submit(t_start, t_end, question)

    def poll(self, job_id: str) -> DeepJob | None:
        self._bind()
        if self._poll is None:
            # A worker that only pushes results is fine; the funnel also accepts a job
            # handed to it directly. Polling simply has nothing to say.
            return None
        return self._poll(job_id)
