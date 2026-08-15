"""M5 — the standing-task monitor (SPEC §6).

Subscribes to every chunk M1 emits, runs the three-stage funnel over every registered
task, and fires MCP actions through the brakes. The only module that changes the outside
world unprompted.

Typical use::

    from services.monitor import build_monitor

    monitor = build_monitor()                  # seeded from config/tasks.yaml
    for chunk in ingest_stream:
        monitor.observe([chunk])               # a list, always — invariant 9
        monitor.pump_verifications()           # stage 3 verdicts land here

    payload = monitor.state().to_dict()        # GET /api/monitor/state (SPEC §11.3)

and, for SPEC §10 D5, M3's ``register_task`` endpoint delegates straight in::

    task = monitor.register_task(request_json)  # embeds `describe` once, validates `active`

Three things this package deliberately does not own:

* **The brakes.** Cooldown, footage-range dedupe and the append-only log are
  ``services/mcp``'s, and every action goes through :class:`~services.mcp.ActionServer`
  (CLAUDE.md invariant 5). There is no second cooldown here.
* **The Task shape.** ``shared/schema.py`` is the single source of truth.
* **The HTTP route.** ``state()`` returns plain data; M3 serves it.
"""

from services.monitor.active import (
    ALWAYS,
    ActiveWindow,
    ActiveWindowError,
    parse_active_window,
)
from services.monitor.confirm import (
    ConfirmVerdict,
    NIMConfirmer,
    Stage2Confirmer,
    StubConfirmer,
    build_confirmer,
    content_words,
)
from services.monitor.funnel import (
    FunnelOutcome,
    Monitor,
    VerificationOutcome,
    build_monitor,
    cosine,
)
from services.monitor.registry import TaskRegistrationError, TaskRegistry, load_task_seed
from services.monitor.settings import PENDING_SETTINGS, MonitorSettings
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
    WorkerVerifier,
    confidence_verdict,
)

__all__ = [
    # the two things most callers need
    "build_monitor",
    "Monitor",
    # configuration
    "MonitorSettings",
    "PENDING_SETTINGS",
    # tasks — SPEC §6.1 / §10 D5
    "TaskRegistry",
    "TaskRegistrationError",
    "load_task_seed",
    "ActiveWindow",
    "ActiveWindowError",
    "parse_active_window",
    "ALWAYS",
    # stage 2
    "Stage2Confirmer",
    "StubConfirmer",
    "NIMConfirmer",
    "build_confirmer",
    "ConfirmVerdict",
    "content_words",
    # stage 3 — M4's seam, imported lazily
    "DeepVerifier",
    "NullVerifier",
    "WorkerVerifier",
    "VerdictFn",
    "confidence_verdict",
    # the Watch pane — SPEC §11.3
    "MonitorState",
    "TaskFunnelState",
    "Stage1State",
    "Stage2State",
    "Stage3State",
    "TimeRange",
    # values that cross the boundary
    "FunnelOutcome",
    "VerificationOutcome",
    "cosine",
]
