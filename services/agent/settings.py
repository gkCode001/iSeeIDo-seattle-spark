"""Every dial M3 reads, resolved from ``config/settings.yaml`` in exactly one place.

CLAUDE.md: no magic numbers in service code. Keys that already exist under ``agent:``
are read with :func:`shared.config.get` and no default — a missing one should fail
loudly.

**Keys this module needs that settings.yaml does not have yet** are listed in
``_PENDING`` below and read *with* a default. A default in code is a magic number
wearing a disguise (shared/config.py says so), so each one is temporary, named here
rather than scattered through the agent, and reported to whoever owns settings.yaml.
Every default is chosen so the ask surface runs today on this box: ``agent.backend`` is
``stub``, ``agent.model`` is null (SPEC §10 D3), and there is no NGC key.

The prompts live here for the same reason ``vlm.prompts`` lives in settings.yaml: the
groundedness gate's wording *is* the gate (SPEC §4.2), and a prompt edited in two places
is two gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared import config

__all__ = ["AgentSettings", "PENDING_SETTINGS"]


# --------------------------------------------------------------------------------------
# Settings that belong in config/settings.yaml under ``agent:`` and are not there yet.
# --------------------------------------------------------------------------------------
# Every M3 tunable now lives in config/settings.yaml, where the rationale lives beside
# it. Intentionally EMPTY: a fallback that shadows nothing only confuses the next person
# hunting for the dial, and a default in code is a magic number wearing a disguise.
# Anything added here is a key that has NOT yet reached the YAML.
_PENDING: dict[str, object] = {}

PENDING_SETTINGS: tuple[str, ...] = tuple(_PENDING)


def _pending(dotted: str) -> object:
    """Read a setting, falling back to the table above only if it carries the key.

    The fallback is looked up conditionally: indexing ``_PENDING`` eagerly as
    ``config.get``'s default argument would raise ``KeyError`` for every setting once
    the table empties — which is exactly what happens when the keys land in the YAML.
    """
    if dotted in _PENDING:
        return config.get(dotted, _PENDING[dotted])
    return config.get(dotted)


@dataclass(frozen=True)
class AgentSettings:
    """Resolved configuration for one ask agent and its server."""

    # --- model -------------------------------------------------------------------
    backend: str
    endpoint: str
    max_tokens: int
    temperature: float
    request_timeout: float

    # --- escalation (SPEC §4.2 / §4.3) --------------------------------------------
    groundedness_gate: bool
    deep_timeout_seconds: float
    deep_max_inflight: int
    deep_dedupe_identical_ranges: bool
    deep_range_pad_seconds: float
    deep_max_range_seconds: float
    deep_fallback_window_seconds: float
    deep_poll_interval_seconds: float

    # --- retrieval ----------------------------------------------------------------
    search_lookback_seconds: float

    # --- server -------------------------------------------------------------------
    host: str
    port: int
    ui_dir: Path
    chat_log: Path
    chunks_lookback_seconds: float
    chunks_max: int
    browse_page_size: int
    browse_max_page_size: int
    browse_caption_preview_chars: int
    actions_lookback_seconds: float
    history_max_turns: int
    ws_ping_interval_seconds: float

    # --- prompts ------------------------------------------------------------------
    groundedness_prompt: str
    answer_prompt: str

    # --- stub backend -------------------------------------------------------------
    stub_coverage_threshold: float

    @property
    def model(self) -> str:
        """The ask model — SPEC §10 D3, still UNRESOLVED.

        ``require`` rather than ``get``: an unset model should fail with a sentence
        naming the open decision, not with a 404 from a NIM that was never told what to
        serve. The stub backend never reads this, which is the whole reason the ask
        surface can be exercised today.
        """
        return str(config.require("agent.model"))

    @classmethod
    def from_config(cls) -> AgentSettings:
        """Build from ``config/settings.yaml``. Every existing key is required."""
        return cls(
            backend=str(config.get("agent.backend")),
            endpoint=str(config.get("agent.endpoint")),
            max_tokens=int(config.get("agent.max_tokens")),
            temperature=float(_pending("agent.temperature")),  # type: ignore[arg-type]
            request_timeout=float(_pending("agent.request_timeout_seconds")),  # type: ignore[arg-type]
            groundedness_gate=bool(config.get("agent.groundedness_gate")),
            deep_timeout_seconds=float(config.get("agent.deep.timeout_seconds")),
            deep_max_inflight=int(config.get("agent.deep.max_inflight")),
            deep_dedupe_identical_ranges=bool(config.get("agent.deep.dedupe_identical_ranges")),
            deep_range_pad_seconds=float(_pending("agent.deep.range_pad_seconds")),  # type: ignore[arg-type]
            deep_max_range_seconds=float(_pending("agent.deep.max_range_seconds")),  # type: ignore[arg-type]
            deep_fallback_window_seconds=float(
                _pending("agent.deep.fallback_window_seconds")  # type: ignore[arg-type]
            ),
            deep_poll_interval_seconds=float(_pending("agent.deep.poll_interval_seconds")),  # type: ignore[arg-type]
            search_lookback_seconds=float(
                _pending("agent.search.default_lookback_seconds")  # type: ignore[arg-type]
            ),
            host=str(config.get("agent.host")),
            port=int(config.get("agent.port")),
            ui_dir=(config.REPO_ROOT / str(_pending("agent.ui_dir"))).resolve(),
            chat_log=config.repo_path("paths.chat_log"),
            chunks_lookback_seconds=float(_pending("agent.chunks.lookback_seconds")),  # type: ignore[arg-type]
            chunks_max=int(_pending("agent.chunks.max")),  # type: ignore[arg-type]
            browse_page_size=int(_pending("agent.browse.page_size")),  # type: ignore[arg-type]
            browse_max_page_size=int(
                _pending("agent.browse.max_page_size")  # type: ignore[arg-type]
            ),
            browse_caption_preview_chars=int(
                _pending("agent.browse.caption_preview_chars")  # type: ignore[arg-type]
            ),
            actions_lookback_seconds=float(_pending("agent.actions.lookback_seconds")),  # type: ignore[arg-type]
            history_max_turns=int(_pending("agent.history.max_turns")),  # type: ignore[arg-type]
            ws_ping_interval_seconds=float(_pending("agent.ws.ping_interval_seconds")),  # type: ignore[arg-type]
            groundedness_prompt=str(_pending("agent.prompts.groundedness")),
            answer_prompt=str(_pending("agent.prompts.answer")),
            stub_coverage_threshold=float(_pending("agent.stub.coverage_threshold")),  # type: ignore[arg-type]
        )
