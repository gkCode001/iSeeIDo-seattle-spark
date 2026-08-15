"""M3 — the ask agent (SPEC §4) and the process that serves the §11 UI.

    question → search_index → groundedness gate → provisional answer + job_id
                                                → refinement over the WebSocket

Two things this module exists to make true, both of which are demo-critical:

* **The escalation decision is legible.** SPEC §4.2's two mechanisms — the groundedness
  gate and the model's own tool choice — are both on, both recorded on the turn, and
  both printed. ``ChatTurn.grounded`` is persisted, not recomputed; §11.2 calls that
  badge the most important pixel in the build.
* **A user turn never blocks on deep analysis** (CLAUDE.md invariant 4). ``ask()``
  returns as soon as it has a provisional answer and a ``job_id``; the refinement is
  appended later, never substituted.

Typical use::

    from services.agent import build_app, AskServer

    server = AskServer(build_app())
    server.serve_forever()          # or `python3 -m services.agent`

and the agent alone, without a socket::

    app = build_app()
    result = app.agent.ask("Was the van's rear door open when it backed up?")
    result.turn.grounded            # False -> escalated
    result.job.job_id               # queued, not finished

``agent.backend: stub`` is the shipped default: there is no NGC key on this box and
``agent.model`` is null pending SPEC §10 D3, so the stub is how the whole arc runs today
against the *real* index. It is not a test mock — see ``llm.py``.
"""

from .agent import AskAgent, AskResult, Escalation, GateVerdict
from .deep import DeepAnalyzer, JobRegistry, JobUpdate, UnavailableAnalyzer, WorkerAnalyzer
from .history import ChatHistory, ChatLog
from .llm import (
    ContextChunk,
    LLMBackend,
    LLMRequest,
    LLMResponse,
    OpenAICompatBackend,
    Purpose,
    StubBackend,
    ToolCall,
    build_backend,
)
from .server import AgentApp, AskServer, build_app, main
from .settings import PENDING_SETTINGS, AgentSettings
from .tasks import DuplicateTaskError, SeedTaskRegistry, TaskRegistry
from .tools import TOOL_SCHEMAS, ToolInvocation, Toolbox
from .ws import WebSocketConnection, WebSocketHub, accept_key

__all__ = [
    # the two things most callers need
    "build_app",
    "AskServer",
    # the agent, without a transport
    "AskAgent",
    "AskResult",
    "Escalation",
    "GateVerdict",
    "AgentApp",
    "main",
    # configuration
    "AgentSettings",
    "PENDING_SETTINGS",
    # seams, for tests and for swapping implementations by hand
    "LLMBackend",
    "StubBackend",
    "OpenAICompatBackend",
    "build_backend",
    "DeepAnalyzer",
    "WorkerAnalyzer",
    "UnavailableAnalyzer",
    "JobRegistry",
    "TaskRegistry",
    "SeedTaskRegistry",
    "Toolbox",
    "ChatLog",
    # values that cross the boundary
    "ContextChunk",
    "LLMRequest",
    "LLMResponse",
    "ToolCall",
    "Purpose",
    "ToolInvocation",
    "JobUpdate",
    "ChatHistory",
    "TOOL_SCHEMAS",
    "DuplicateTaskError",
    "WebSocketHub",
    "WebSocketConnection",
    "accept_key",
]
