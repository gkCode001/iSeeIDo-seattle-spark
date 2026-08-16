"""AlertBridge client — the one way an action reaches Discord.

AlertBridge is a separate service on this host (`/opt/alertbridge`, port 8081, reachable
over Tailscale) that owns the Discord webhook and its credentials. We are a caller and
nothing more: no webhook URL in this repo, no discord library, no new dependency —
``requests`` is already one of the two the project has.

**This sits behind the brakes, never beside them.** A Discord message cannot be
un-posted, so the thing that decides whether to send is ``ActionServer.fire`` — cooldown,
time-range dedupe, append-only log — exactly as it is for every other action (CLAUDE.md
invariant 5). This module only knows how to send one, and it is called after the brakes
have already said yes.

Two failure rules, from the caller guide at ``/opt/alertbridge/TRIGGERS.md``:

* **A send failure must not take down the thing it monitors.** Every exception is caught
  and returned as a failed :class:`SendResult`. M5's funnel keeps running; the action log
  records what was attempted. An alerting path that can crash the monitor is worse than
  no alerting path.
* **202 does not mean delivered.** It means AlertBridge accepted the submission. Nothing
  here logs or reports "delivered", because nothing here knows that.

The idempotency key is the part that is easy to get wrong, so it is derived rather than
generated — see :func:`idempotency_key`.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

from shared import config

__all__ = [
    "AlertBridgeClient",
    "AlertBridgeNotConfigured",
    "SendResult",
    "idempotency_key",
]

logger = logging.getLogger("services.mcp.alertbridge")

#: AlertBridge forwards these keys to providers that reject anything else, so the charset
#: is the provider's, not ours: letters, digits, dot, underscore, hyphen. 1–128 chars.
_KEY_ALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_KEY_MAX = 128

_TOKEN_ENV = "ALERTBRIDGE_SERVICE_TOKEN"


class AlertBridgeNotConfigured(RuntimeError):
    """The token is absent. Raised at startup by :meth:`AlertBridgeClient.preflight`.

    Deliberately loud rather than a silent no-op: a standing task configured to notify
    Discord and quietly notifying nobody is the worst of both worlds — it looks armed on
    the Watch pane and it is not.
    """


@dataclass(frozen=True)
class SendResult:
    """What happened to one submission. Never raises; ``ok`` carries the verdict."""

    ok: bool
    status: int | None
    detail: str
    audit_id: str | None = None
    provider_message_id: str | None = None
    replayed: bool = False
    #: True when AlertBridge could not tell us whether delivery happened (503, timeout).
    #: The caller must NOT retry under a fresh key — that is how one alert becomes two.
    outcome_unknown: bool = False

    def summary(self) -> str:
        """One clause for the action log's ``reason``, so the Timeline shows the truth."""
        if self.ok and self.replayed:
            return f"discord replay (audit {self.audit_id})"
        if self.ok:
            return f"discord submitted (audit {self.audit_id})"
        if self.outcome_unknown:
            return f"discord outcome unknown (audit {self.audit_id}): {self.detail}"
        return f"discord send failed: {self.detail}"


def idempotency_key(task_id: str | None, t_start: datetime, t_end: datetime) -> str:
    """A key that is stable across retries of one alert and unique across alerts.

    Derived from the **footage range**, which is the same thing the dedupe brake keys on,
    plus the task. Two properties follow, and both are the point:

    * A retry of the same logical alert — same task, same seconds of video — produces the
      same key, so AlertBridge replays its earlier result instead of posting twice.
    * Two different events cannot collide, because they cannot occupy the same seconds.

    Not random and not derived from the clock: TRIGGERS.md calls both out by name, and
    either would turn every retry into a second message in the channel.
    """
    stamp = f"{t_start:%Y%m%dT%H%M%S}-{t_end:%Y%m%dT%H%M%S}"
    raw = f"spark-{task_id or 'ad-hoc'}-{stamp}-v1-discord"
    cleaned = _KEY_ALLOWED.sub("-", raw).strip("-")
    return cleaned[:_KEY_MAX] or "spark-v1-discord"


class AlertBridgeClient:
    """Posts one message to ``/v1/alerts/discord``. Construct once per process."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        username: str | None = None,
        timeout_seconds: float | None = None,
        token: str | None = None,
        max_text_chars: int | None = None,
    ) -> None:
        self.base_url = str(
            base_url if base_url is not None else config.get("mcp.discord.base_url")
        ).rstrip("/")
        self.username = str(
            username if username is not None else config.get("mcp.discord.username")
        )[:80]
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else config.get("mcp.discord.timeout_seconds")
        )
        self.max_text_chars = int(
            max_text_chars
            if max_text_chars is not None
            else config.get("mcp.discord.max_text_chars")
        )
        #: Read once, from the environment, never from /opt/alertbridge/.env and never
        #: from this repo. None is a legal state — :meth:`preflight` is what refuses.
        self._token = token if token is not None else os.environ.get(_TOKEN_ENV)

    # -- startup -----------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def preflight(self) -> None:
        """Raise unless this client could actually send. Call at startup, not at fire time.

        Discovering a missing token at the moment an alert fires means the alert is lost
        and the operator finds out from the absence of a message.
        """
        if not self._token:
            raise AlertBridgeNotConfigured(
                f"{_TOKEN_ENV} is not set, but a standing task is configured to notify "
                f"Discord. Export the AlertBridge service token in this process's "
                f"environment and restart:\n\n"
                f"    export {_TOKEN_ENV}=...\n\n"
                f"The token is not in this repo and must not be committed to it. "
                f"Check the service is reachable first: "
                f"curl -fsS {self.base_url}/health/live"
            )

    def health(self) -> SendResult:
        """Unauthenticated liveness probe — safe to call at startup, sends no message."""
        import requests  # noqa: PLC0415 — deferred, matching the rest of this project

        try:
            resp = requests.get(f"{self.base_url}/health/live", timeout=self.timeout_seconds)
            return SendResult(
                ok=resp.status_code == 200,
                status=resp.status_code,
                detail=resp.text.strip()[:200],
            )
        except Exception as exc:  # noqa: BLE001 - a probe that raises is a probe that lies
            return SendResult(ok=False, status=None, detail=f"{type(exc).__name__}: {exc}")

    # -- sending -----------------------------------------------------------------------

    def send(self, text: str, key: str) -> SendResult:
        """Submit one Discord message. Never raises.

        ``key`` is the caller's derived idempotency key — see :func:`idempotency_key`.
        Retrying is the caller's decision and must reuse the identical payload AND key;
        this method does not retry on its own, because the one thing worse than a missed
        alert is the same alert posted twice while a human is reading the first.
        """
        import requests  # noqa: PLC0415

        if not self._token:
            return SendResult(ok=False, status=None, detail=f"{_TOKEN_ENV} is not set")

        body = {"text": text[: self.max_text_chars], "username": self.username}
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Idempotency-Key": key,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v1/alerts/discord",
                json=body,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - non-fatal by contract
            # No HTTP response: TRIGGERS.md says retry conservatively with the IDENTICAL
            # payload and key. We do not retry here, but we must not let the caller
            # conclude nothing was sent — the request may well have arrived.
            return SendResult(
                ok=False,
                status=None,
                detail=f"{type(exc).__name__}: {exc}",
                outcome_unknown=True,
            )

        payload: dict[str, object] = {}
        try:
            payload = resp.json() if resp.content else {}
        except ValueError:
            payload = {}
        audit = _as_str(payload.get("audit_id"))
        provider = _as_str(payload.get("provider_message_id"))
        code = str(payload.get("code") or payload.get("error") or "")
        retry_after = resp.headers.get("Retry-After")

        # 202 accepted, 200 terminal replay of an identical earlier request. Both mean
        # "exactly one message exists for this key"; neither means a human has seen it.
        if resp.status_code in (200, 202):
            return SendResult(
                ok=True,
                status=resp.status_code,
                detail="accepted" if resp.status_code == 202 else "replay",
                audit_id=audit,
                provider_message_id=provider,
                replayed=resp.status_code == 200,
            )
        if resp.status_code == 409:
            # request_in_progress -> retryable with the same key after Retry-After.
            # idempotency_key_payload_mismatch -> a bug here, not a transient failure:
            # the same key was reused with different text.
            return SendResult(
                ok=False,
                status=409,
                detail=f"{code or 'conflict'}"
                + (f"; retry after {retry_after}s" if retry_after else ""),
            )
        if resp.status_code == 429:
            return SendResult(
                ok=False,
                status=429,
                detail=f"rate limited{f'; retry after {retry_after}s' if retry_after else ''}",
            )
        if resp.status_code == 503:
            return SendResult(
                ok=False,
                status=503,
                detail=code or "delivery_outcome_unknown",
                audit_id=audit,
                outcome_unknown=True,
            )
        return SendResult(
            ok=False,
            status=resp.status_code,
            detail=(code or resp.text.strip())[:200],
        )


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)
