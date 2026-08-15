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

from shared.queue import Priority
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
    """Adapter onto M4 (SPEC §5), resolved lazily so importing M5 never needs M4's file.

    **Binds to a worker OBJECT, not to the module.** The earlier version duck-typed a
    list of plausible names against ``services.worker`` and happened to find a
    module-level ``submit`` but no ``poll`` — those live on ``DeepWorker``. The result
    was a verifier that submitted jobs and could never collect a verdict: alerts stayed
    UNVERIFIED for ever, the stage-3 spinner never resolved, and retraction — the point
    of stage 3 — could not happen. Nothing errored, so nothing said so.

    So the contract is now explicit and checked once, at bind time: whatever is injected
    must expose ``submit`` and ``poll``. ``services.worker.default_worker()`` does.
    """

    def __init__(self, worker: Any | None = None, *, module: str = "services.worker") -> None:
        self._worker = worker
        self._module = module
        self._bound = False

    def _resolve(self) -> Any:
        """The worker object. A module is accepted and asked for its default worker."""
        if self._worker is None:
            import importlib  # noqa: PLC0415 - deferred; M4 is imported only when used

            module = importlib.import_module(self._module)
            factory = getattr(module, "default_worker", None)
            if not callable(factory):
                raise RuntimeError(
                    f"{self._module} exposes no default_worker(); SPEC §5 requires a "
                    f"worker object with submit(...) and poll(...)."
                )
            self._worker = factory()
        return self._worker

    def _bind(self) -> None:
        if self._bound:
            return
        target = self._resolve()
        missing = [name for name in ("submit", "poll") if not callable(getattr(target, name, None))]
        if missing:
            raise RuntimeError(
                f"{type(target).__name__} is missing {', '.join(missing)}; SPEC §5 "
                f"requires submit(t_start, t_end, question) -> DeepJob and poll(job) -> "
                f"DeepJob. Stage 3 cannot verify without both — an alert that can never "
                f"be verified can never be retracted either."
            )
        self._bound = True

    def submit(self, t_start: datetime, t_end: datetime, question: str) -> DeepJob:
        self._bind()
        # VERIFICATION priority, not INTERACTIVE: a user waiting on an ask outranks a
        # background re-watch (SPEC §7).
        return self._resolve().submit(
            t_start, t_end, question, priority=Priority.VERIFICATION
        )

    def poll(self, job_id: str) -> DeepJob | None:
        self._bind()
        try:
            return self._resolve().poll(job_id)
        except KeyError:
            # A job this worker never heard of — a restart, or somebody else's id.
            return None
