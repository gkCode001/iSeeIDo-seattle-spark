"""Every dial M5 reads, resolved from ``config/settings.yaml`` in exactly one place.

CLAUDE.md: no magic numbers in service code. Keys that already exist under ``monitor:``,
``agent:`` and ``ui:`` are read with :func:`shared.config.get` and no default — a missing
one should fail loudly rather than quietly running the funnel at some number nobody
chose.

**Keys this module needs that settings.yaml does not have yet** are listed in
``_PENDING`` below and read *with* a default. A default in code is a magic number wearing
a disguise (``shared/config.py`` says so), so each one is named here rather than
scattered through the funnel, and each is chosen so that M5 runs end to end **today** on
a box with no NGC key and no LLM serving — see CLAUDE.md's machine-state table. Add them
to settings.yaml and the defaults become dead code.

Deliberately *not* here: the cooldown and dedupe numbers as behaviour. M5 reads them only
to report them to the Watch pane. The brakes themselves live in ``services/mcp`` and are
configured there — one brake, in one place (CLAUDE.md invariant 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared import config

__all__ = ["MonitorSettings", "PENDING_SETTINGS"]


# --------------------------------------------------------------------------------------
# Settings that belong in config/settings.yaml and are not there yet.
# Reported to whoever owns settings.yaml; until then these defaults apply.
# --------------------------------------------------------------------------------------
_PENDING: dict[str, Any] = {
    # Stage 2 needs an LLM and this box has none (no NGC key, nothing serving). `stub`
    # is a deterministic lexical confirmer so the whole three-stage funnel runs today;
    # it reads `agent.backend` so M3 and M5 cannot end up on different backends.
    # OpenAI-compatible chat route for the real thing. Kept configurable because NIM has
    # moved it between releases.
    "monitor.confirm_path": "/chat/completions",
    "monitor.confirm_prompt": (
        "Decide whether the SCENE satisfies the CONDITION. Wording will differ — "
        "judge the meaning, not the words. A 'van' is a vehicle, a 'person' is "
        "someone. Answer with one word: YES or NO."
    ),
    "monitor.confirm_timeout_seconds": 30.0,
    # Stage 2's own budget. It answers one word; anything longer is the model ignoring
    # the instruction, not a better answer. CLAUDE.md invariant 6 in spirit: output
    # tokens are the dial.
    "monitor.confirm_max_tokens": 8,
    # StubConfirmer only. Fraction of the task's content words that must appear in the
    # caption. 0.5 is loose enough to survive caption paraphrase and tight enough that an
    # unrelated caption does not sustain a window. NOT a quality signal — see confirm.py.
    "monitor.stage2_stub_min_overlap": 0.5,
    # Stage 3 verdict threshold. The worker returns a confidence (SPEC §5); below this we
    # retract. A job that failed or timed out returns no confidence at all and is
    # inconclusive rather than a disagreement — see verify.py.
    "monitor.verify_confidence_threshold": 0.5,
    # The question handed to `deep_analyze`. `{describe}` is the task's own words, which
    # is the whole point of re-watching: the caption already said "a van"; we are asking
    # the pixels whether it was *this* van, in *this* place.
    "monitor.verify_question_template": (
        "Does this footage show: {describe}? Answer yes or no, cite the burned-in "
        "timestamps for anything you report, and say so explicitly if the footage does "
        "not show enough to decide."
    ),
}

PENDING_SETTINGS: tuple[str, ...] = tuple(_PENDING)


def _pending(dotted: str) -> Any:
    """Read a not-yet-in-YAML setting, falling back to the documented default above."""
    return config.get(dotted, _PENDING[dotted])


@dataclass(frozen=True)
class MonitorSettings:
    """Resolved configuration for one :class:`~services.monitor.funnel.Monitor`."""

    # --- task registry ------------------------------------------------------------
    tasks_file: Path

    # --- stage 1: embedding match (SPEC §6.2) -------------------------------------
    stage1_cosine_threshold: float

    # --- stage 2: LLM confirm + sustain window ------------------------------------
    stage2_sustain_default: int
    confirm_backend: str
    confirm_model: str | None
    confirm_endpoint: str
    confirm_path: str
    confirm_prompt: str
    confirm_timeout: float
    confirm_max_tokens: int
    confirm_extra_body: dict[str, object]
    stub_min_overlap: float

    # --- stage 3: worker verify ---------------------------------------------------
    verify_promoted: bool
    verify_confidence_threshold: float
    verify_question_template: str

    # --- reported to the Watch pane, enforced in services/mcp ---------------------
    default_cooldown_seconds: float
    dedupe_overlap_seconds: float

    # --- the one place local time is allowed (SPEC §11.5, Task.active) ------------
    display_timezone: str

    def sustain_seconds(self, window: int) -> int:
        """Per-task ``window`` with the configured fallback for an unset one.

        SPEC §6.1 makes ``window`` a required field, so this only matters for a task
        registered at runtime through a form that left it blank (SPEC §11.3).
        """
        return int(window) if window and window > 0 else self.stage2_sustain_default

    def verify_question(self, describe: str) -> str:
        return self.verify_question_template.format(describe=describe)

    @classmethod
    def from_config(cls) -> MonitorSettings:
        """Build from ``config/settings.yaml``. Every existing key is required."""
        return cls(
            tasks_file=config.repo_path("monitor.tasks_file"),
            stage1_cosine_threshold=float(config.get("monitor.stage1_cosine_threshold")),
            stage2_sustain_default=int(config.get("monitor.stage2_sustain_default")),
            # Same key M3 reads. Two surfaces, one LLM decision — a monitor confirming
            # on `nim` while the ask agent is on `stub` is a demo that contradicts itself.
            confirm_backend=str(config.get("agent.backend")),
            # UNRESOLVED — SPEC §10 D3. None is legal here; only NIMConfirmer needs it,
            # and it re-reads the key through `config.require` so the failure names the
            # decision instead of 404ing at an endpoint told to serve nothing.
            confirm_model=(
                None if config.get("agent.model") is None else str(config.get("agent.model"))
            ),
            confirm_endpoint=str(config.get("agent.endpoint")),
            confirm_path=str(_pending("monitor.confirm_path")),
            confirm_prompt=str(_pending("monitor.confirm_prompt")),
            confirm_timeout=float(_pending("monitor.confirm_timeout_seconds")),
            confirm_max_tokens=int(_pending("monitor.confirm_max_tokens")),
            # Same key M3 reads, for the same reason it reads it: on Lightning,
            # chat_template_kwargs.enable_thinking=false is the only switch that keeps the
            # model from writing its reasoning into `content`. Stage 2 asks for 8 tokens
            # and parses the first word, so without this every confirm reads back as a
            # sentence fragment ("Here, I need to judge whether the") and fails closed —
            # a monitor that observes everything and can never promote anything.
            confirm_extra_body=dict(config.get("agent.extra_body", {}) or {}),
            stub_min_overlap=float(_pending("monitor.stage2_stub_min_overlap")),
            verify_promoted=bool(config.get("monitor.verify_promoted")),
            verify_confidence_threshold=float(_pending("monitor.verify_confidence_threshold")),
            verify_question_template=str(_pending("monitor.verify_question_template")),
            default_cooldown_seconds=float(config.get("monitor.default_cooldown_seconds")),
            dedupe_overlap_seconds=float(config.get("monitor.dedupe_overlap_seconds")),
            display_timezone=str(config.get("ui.display_timezone")),
        )
