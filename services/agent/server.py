"""The M3 HTTP surface — SPEC §11.1's "single page served by the M3 process".

``fastapi`` and ``uvicorn`` are not installed and adding a dependency is the fastest way
to lose a day on this box (CLAUDE.md), so this is ``http.server`` plus the small
WebSocket in ``ws.py``. The split is deliberate and is the whole reason a later FastAPI
swap is a new file rather than a rewrite:

* :class:`AgentApp` holds the logic. Its route methods take parsed arguments and return
  ``(status, payload)``. No sockets, no ``self.wfile``, no framework.
* :class:`_Handler` is transport. It parses a query string, calls the app, writes JSON.

The endpoint list is not ours to invent — ``ui/static/data.js`` already declares it, and
that file is the contract:

    GET  /api/config          settings.yaml as JSON
    GET  /api/chunks          ?t_from&t_to  -> {chunks: [ChunkRecord]}
    GET  /api/index           ?offset&limit&q&tier&gated&t_from&t_to&newest_first
                                            -> one page of the index, in time order
    GET  /api/chat/history                  -> {turns: [ChatTurn], jobs: {id: DeepJob}}
    GET  /api/tasks                         -> {tasks: [Task]}
    GET  /api/monitor/state                 -> funnel state (§11.3)
    GET  /api/actions         ?t_from&t_to  -> {entries: [ActionLogEntry]}
    GET  /api/video           ?t_from&t_to  -> stitched footage, NEVER ?file=
    POST /api/ask             {question}    -> ChatTurn.to_dict() + {dedupe_of?, job?}
    POST /api/register_task   Task fields   -> Task
    WS   /ws                                -> refinement | monitor_state | action

Everything else is served from ``ui/`` so the page and its endpoints share an origin.
"""

from __future__ import annotations

import json
import mimetypes
import re
import socketserver
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from services.index import IndexStore, build_index
from services.mcp import ActionServer, ClipCutter, NullClipCutter, build_clip_plan, clip_path_for
from shared import config, timecode
from shared.schema import DeepJob, JobState, Tier, from_iso, to_iso, utcnow

from .agent import AskAgent
from .deep import JobRegistry, JobUpdate, UnavailableAnalyzer, WorkerAnalyzer
from .history import ChatLog
from .llm import build_backend
from .settings import AgentSettings
from .tasks import DuplicateTaskError, SeedTaskRegistry, TaskRegistry, task_from_payload
from .telemetry import log_event
from .tools import Toolbox
from .ws import WebSocketConnection, WebSocketHub, accept_key

__all__ = ["AgentApp", "AskServer", "build_app", "main"]

_JSON = "application/json"

#: Request bodies are a question or six task fields. Anything larger is not ours.
_MAX_BODY_BYTES = 1 << 20

#: ``Range: bytes=0-`` and friends. The <video> element sends one, always.
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


@dataclass
class AgentApp:
    """Everything the routes need, injected. No globals, no import-time construction."""

    agent: AskAgent
    index: IndexStore
    actions: ActionServer
    jobs: JobRegistry
    chat_log: ChatLog
    tasks: TaskRegistry
    hub: WebSocketHub
    settings: AgentSettings
    clip_cutter: ClipCutter

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        """Wire the push channel and reconcile whatever the last process left behind."""
        self.jobs.subscribe(self._on_job_update)
        history = self.chat_log.read()
        self.jobs.adopt(history.jobs.values(), self.chat_log.turn_ids_by_job())
        for job in self.chat_log.unfinished_jobs():
            # The process that was going to finish this is gone. Say so once, in the log,
            # rather than leaving a card counting up against a 90 s timeout forever.
            self.chat_log.append_job(
                DeepJob(
                    job_id=job.job_id,
                    t_start=job.t_start,
                    t_end=job.t_end,
                    question=job.question,
                    state=JobState.FAILED,
                    requested_at=job.requested_at,
                    completed_at=utcnow(),
                    error="the agent restarted while this job was in flight",
                )
            )
        self.jobs.start()

    def stop(self) -> None:
        self.jobs.stop()
        self.hub.close_all()

    # -- push -------------------------------------------------------------------------

    def _on_job_update(self, update: JobUpdate) -> None:
        """Persist and push. SPEC §4.3's third arrow, and §11.4's durability.

        Persisted **before** it is broadcast: a refinement that reached the screen but
        not the file would vanish on reload, which is exactly the failure §11.4 exists to
        prevent.
        """
        self.chat_log.append_job(update.job)
        payload = update.job.to_dict()
        for turn_id in update.turn_ids:
            # One message per turn. A deduped second ask rides the same job and gets its
            # own refinement addressed to its own card (SPEC §4.3).
            self.hub.broadcast({"type": "refinement", "turn_id": turn_id, "job": payload})

    def publish_action(self, entry: dict[str, Any]) -> int:
        """For M5: push an action-log row to the Timeline pane as it is written."""
        return self.hub.broadcast({"type": "action", "entry": entry})

    def publish_monitor_state(self, state: dict[str, Any]) -> int:
        """For M5: push funnel state to the Watch pane."""
        return self.hub.broadcast({"type": "monitor_state", "state": state})

    # -- routes: reads -----------------------------------------------------------------

    def get_config(self) -> tuple[int, dict[str, Any]]:
        """``config/settings.yaml`` as JSON. Read, never written — it is owned elsewhere."""
        return HTTPStatus.OK, dict(config.load())

    def get_chunks(
        self, t_from: datetime | None, t_to: datetime | None
    ) -> tuple[int, dict[str, Any]]:
        """Chunk records overlapping a range, for the player's segment strip.

        Listing is done through the public search API with a wide ``ann_k``/``top_n`` and
        an empty query: with no query terms every candidate scores alike, so the filter
        that survives is the time range — which is the one we want. Reaching past
        ``IndexStore`` into a backend to add a ``list()`` would couple M3 to M2's
        internals for a convenience the UI needs once per paint.
        """
        t_to = t_to or utcnow()
        t_from = t_from or t_to - timedelta(seconds=self.settings.chunks_lookback_seconds)
        limit = self.settings.chunks_max
        hits = self.index.search("", t_from, t_to, ann_k=limit, top_n=limit)
        records = sorted((hit.record for hit in hits), key=lambda r: (r.t_start, r.chunk_id))
        return HTTPStatus.OK, {
            "chunks": [record.to_dict() for record in records],
            "t_from": to_iso(t_from),
            "t_to": to_iso(t_to),
        }

    def get_index(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        t_from: datetime | None = None,
        t_to: datetime | None = None,
        tier: str | None = None,
        gated: bool | None = None,
        contains: str | None = None,
        newest_first: bool = True,
    ) -> tuple[int, dict[str, Any]]:
        """``GET /api/index`` — the index browser's page of captions (ui/browse.html).

        Deliberately **not** :meth:`get_chunks`. That one lists a time range for the
        player strip and reaches the records through ``search("")``, which is fine for a
        strip and wrong for a browser: it is capped at ``agent.chunks.max``, it drops
        gated windows, and it cannot say how many rows exist — so it cannot paginate.
        This route goes through ``IndexStore.browse``, which orders by wall clock and
        returns the total.

        Gated windows are included by default and flagged rather than hidden. They are
        ~78% of the corpus on this box, and a browser that quietly omits them shows a
        reader an index four times smaller than the one the system actually holds — plus
        the skip rate is the health metric SPEC §2.3 asks to keep visible.
        """
        page_size = self.settings.browse_page_size if limit is None else limit
        page_size = max(1, min(int(page_size), self.settings.browse_max_page_size))
        offset = max(0, int(offset))

        resolved_tier: Tier | None = None
        if tier:
            try:
                resolved_tier = Tier(tier)
            except ValueError:
                return HTTPStatus.BAD_REQUEST, {
                    "detail": f"unknown tier {tier!r}; expected one of "
                    + ", ".join(t.value for t in Tier)
                }

        page = self.index.browse(
            offset=offset,
            limit=page_size,
            tier=resolved_tier,
            t_from=t_from,
            t_to=t_to,
            gated=gated,
            contains=contains,
            newest_first=newest_first,
        )
        counts = self.index.stats()
        pages = (page.total + page_size - 1) // page_size
        return HTTPStatus.OK, {
            # ChunkRecord.to_dict(), same shape /api/chunks emits — the browser reads
            # captions and time ranges, and browse() has already dropped the vectors.
            "chunks": [record.to_dict() for record in page.records],
            "total": page.total,
            "offset": page.offset,
            "limit": page_size,
            # 1-based, for a reader. Zero pages when nothing matched, so "page 1 of 0"
            # never appears next to an empty list.
            "page": (offset // page_size) + 1 if page.total else 0,
            "pages": pages,
            "filters": {
                "t_from": to_iso(t_from) if t_from else None,
                "t_to": to_iso(t_to) if t_to else None,
                "tier": resolved_tier.value if resolved_tier else None,
                "gated": gated,
                "contains": contains or None,
                "newest_first": newest_first,
            },
            # Corpus-wide, not page-wide: the denominator a skip rate needs (SPEC §2.3).
            "stats": {
                "total": counts.total,
                "captioned": counts.captioned,
                "gated": counts.gated,
                "skip_rate": round(counts.skip_rate, 4),
                "gate_health": self.index.gate_health(counts),
            },
            "caption_preview_chars": self.settings.browse_caption_preview_chars,
        }

    def get_history(self) -> tuple[int, dict[str, Any]]:
        """SPEC §11.4: the turns *and* their jobs, so a reload keeps the refinement."""
        history = self.chat_log.read(self.settings.history_max_turns)
        payload = history.to_dict()
        # Jobs that landed since the last append (or were adopted at boot) are folded in,
        # so a page loading mid-job sees the live state rather than the last written row.
        jobs = dict(payload["jobs"])  # type: ignore[arg-type]
        for job_id, job in self.jobs.jobs().items():
            jobs[job_id] = job.to_dict()
        payload["jobs"] = jobs
        return HTTPStatus.OK, payload

    def get_tasks(self) -> tuple[int, dict[str, Any]]:
        return HTTPStatus.OK, {"tasks": [task.to_dict() for task in self.tasks.tasks()]}

    def get_monitor_state(self) -> tuple[int, dict[str, Any]]:
        return HTTPStatus.OK, self.tasks.monitor_state()

    def get_actions(
        self, t_from: datetime | None, t_to: datetime | None
    ) -> tuple[int, dict[str, Any]]:
        """The append-only action log — the same rows ``read_action_log`` gives the agent.

        One source, no drift: CLAUDE.md is explicit that a parallel store for the
        Timeline pane is how the agent ends up contradicting the screen.
        """
        t_to = t_to or utcnow()
        t_from = t_from or t_to - timedelta(seconds=self.settings.actions_lookback_seconds)
        entries = self.actions.read_action_log(t_from, t_to)
        return HTTPStatus.OK, {
            "entries": [entry.to_dict() for entry in entries],
            "t_from": to_iso(t_from),
            "t_to": to_iso(t_to),
        }

    # -- routes: writes ----------------------------------------------------------------

    def post_ask(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """SPEC §4. Returns the provisional turn; never awaits the deep job."""
        question = str(body.get("question") or "").strip()
        if not question:
            return HTTPStatus.BAD_REQUEST, {"detail": "question is required"}
        t_from = _parse_iso(body.get("t_from"))
        t_to = _parse_iso(body.get("t_to"))
        result = self.agent.ask(question, t_from=t_from, t_to=t_to)
        return HTTPStatus.OK, result.to_payload()

    def post_register_task(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """SPEC §10 D5 / §11.3. The endpoint is ours; the registry is M5's."""
        try:
            task = task_from_payload(body)
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"detail": str(exc)}
        try:
            registered = self.tasks.register(task)
        except DuplicateTaskError as exc:
            return HTTPStatus.CONFLICT, {"detail": str(exc)}
        return HTTPStatus.OK, registered.to_dict()

    def delete_task(self, task_id: str) -> tuple[int, dict[str, Any]]:
        """DELETE /api/tasks/<id>. The task stops being evaluated immediately.

        Its history is deliberately left alone: anything it already fired stays in the
        append-only action log and on the Timeline, with its reason and its clip. SPEC
        §6.4's "why did you alert at 21:11?" has to keep working for a task that no
        longer exists, and deleting rows to tidy a task list is precisely the rewriting
        an append-only log exists to prevent.
        """
        remove = getattr(self.tasks, "remove", None)
        if remove is None:
            return HTTPStatus.NOT_IMPLEMENTED, {
                "detail": "this task registry does not support deletion"
            }
        try:
            removed = remove(task_id)
        except KeyError:
            return HTTPStatus.NOT_FOUND, {"detail": f"no such task: {task_id}"}
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"detail": str(exc)}
        return HTTPStatus.OK, {"deleted": removed.to_dict()}

    def patch_task(self, task_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """PATCH /api/tasks/<id>. Partial edit; unknown fields are rejected, not ignored.

        ``task_id`` is not editable — it keys the cooldown and the dedupe brake, so
        moving it would orphan the history of the event currently in flight.
        """
        update = getattr(self.tasks, "update", None)
        if update is None:
            return HTTPStatus.NOT_IMPLEMENTED, {
                "detail": "this task registry does not support editing"
            }
        if not isinstance(body, dict) or not body:
            return HTTPStatus.BAD_REQUEST, {"detail": "empty patch; nothing to change"}
        allowed = {"describe", "window", "action", "cooldown", "active", "enabled"}
        unknown = sorted(set(body) - allowed - {"task_id"})
        if unknown:
            return HTTPStatus.BAD_REQUEST, {
                "detail": f"unknown field(s): {', '.join(unknown)}; editable: "
                f"{', '.join(sorted(allowed))}"
            }
        try:
            updated = update(task_id, body)
        except KeyError:
            return HTTPStatus.NOT_FOUND, {"detail": f"no such task: {task_id}"}
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"detail": str(exc)}
        return HTTPStatus.OK, updated.to_dict()

    # -- routes: video ------------------------------------------------------------------

    def video_clip(self, t_from: datetime, t_to: datetime) -> tuple[int, Any]:
        """Resolve a **time range** to footage. Never a filename — CLAUDE.md invariant 3.

        An event at 21:11:58 running 12 s lives in two files, so the range goes through
        ``shared/timecode.py`` and, when it spans a boundary, through the same
        concat-demux plan the MCP evidence clips use. The cut is cached under
        ``paths.clips`` by range, so scrubbing back to a moment does not re-cut it.

        Returns ``(status, Path)`` on success and ``(status, dict)`` on failure — the
        handler streams the one and serialises the other.
        """
        if t_to <= t_from:
            return HTTPStatus.BAD_REQUEST, {"detail": "t_to must be after t_from"}
        try:
            spans = timecode.resolve_range(t_from, t_to)
        except timecode.TimecodeError as exc:
            return HTTPStatus.NOT_FOUND, {"detail": str(exc)}

        slices = [
            {"path": str(span.path), "seek": span.pts_in, "duration": span.duration}
            for span in spans
            if not span.is_gap
        ]
        if not slices:
            return HTTPStatus.NOT_FOUND, {
                "detail": (
                    f"no footage recorded for {to_iso(t_from)} .. {to_iso(t_to)}; the "
                    f"archive has a hole there"
                )
            }

        out_path = clip_path_for(
            t_from,
            t_to,
            clips_dir=Path(config.repo_path("paths.clips")),
            camera_id=str(config.get("camera.id")),
            container=str(config.get("recorder.container")),
        )
        if out_path.is_file():
            return HTTPStatus.OK, out_path

        from services.mcp import SegmentSlice  # noqa: PLC0415 — local, keeps the import list honest

        plan = build_clip_plan(
            [
                SegmentSlice(
                    path=s["path"],
                    seek_seconds=float(s["seek"]),
                    duration_seconds=float(s["duration"]),
                )
                for s in slices
            ],
            out_path,
            ffmpeg_bin=str(config.get("recorder.ffmpeg_bin")),
            copy_codec=bool(config.get("recorder.copy_codec")),
        )
        cut = self.clip_cutter.cut(plan)
        if not cut:
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "detail": "no clip cutter available; footage exists but could not be cut",
                "spans": [span.to_dict() for span in spans],
            }
        return HTTPStatus.OK, Path(cut)


# --------------------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------------------


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return from_iso(value)
    except (ValueError, TypeError):
        return None


class _Handler(BaseHTTPRequestHandler):
    """Thin translation layer. Every decision worth arguing about is in :class:`AgentApp`."""

    server_version = "spark-m3/0.1"
    protocol_version = "HTTP/1.1"

    # Set by AskServer before the server loop starts.
    app: AgentApp

    # -- plumbing -----------------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        log_event("agent.http", peer=self.client_address[0], line=fmt % args)

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def _param(self, name: str) -> datetime | None:
        values = self._query().get(name)
        return _parse_iso(unquote(values[0])) if values else None

    def _text(self, name: str) -> str | None:
        """A query parameter as text. Blank and whitespace-only read as absent — an
        empty search box must mean "no filter", not "captions containing nothing"."""
        values = self._query().get(name)
        if not values:
            return None
        text = unquote(values[0]).strip()
        return text or None

    def _int(self, name: str, default: int | None = None) -> int | None:
        """A query parameter as an int. Garbage falls back rather than 500s: these come
        off a URL a human may have edited, and the browser's answer to ?page=banana
        should be page 1."""
        raw = self._text(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _flag(self, name: str, default: bool | None = None) -> bool | None:
        """Tri-state: absent/``any`` keeps ``default`` (usually None = do not filter)."""
        raw = self._text(name)
        if raw is None:
            return default
        lowered = raw.lower()
        if lowered in {"true", "1", "yes", "only"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return default

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise ValueError(f"request body of {length} bytes refused")
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", _JSON)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routing ------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        route = urlparse(self.path).path
        try:
            if route == "/ws":
                self._upgrade()
                return
            if route == "/api/config":
                self._send_json(*self.app.get_config())
            elif route == "/api/chunks":
                self._send_json(*self.app.get_chunks(self._param("t_from"), self._param("t_to")))
            elif route == "/api/index":
                self._send_json(
                    *self.app.get_index(
                        offset=self._int("offset", 0) or 0,
                        limit=self._int("limit"),
                        t_from=self._param("t_from"),
                        t_to=self._param("t_to"),
                        tier=self._text("tier"),
                        gated=self._flag("gated"),
                        contains=self._text("q"),
                        newest_first=self._flag("newest_first", True) is not False,
                    )
                )
            elif route == "/api/chat/history":
                self._send_json(*self.app.get_history())
            elif route == "/api/tasks":
                self._send_json(*self.app.get_tasks())
            elif route == "/api/monitor/state":
                self._send_json(*self.app.get_monitor_state())
            elif route == "/api/actions":
                self._send_json(*self.app.get_actions(self._param("t_from"), self._param("t_to")))
            elif route == "/api/video":
                self._serve_video()
            elif route == "/api/live.jpg":
                self._serve_live_frame()
            elif route.startswith("/api/"):
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"no such endpoint: {route}"})
            else:
                self._serve_static(route)
        except BrokenPipeError:  # pragma: no cover — a tab closed mid-response
            return
        except Exception as exc:  # noqa: BLE001 — one bad request, not one dead server
            log_event("agent.http.error", route=route, error=f"{type(exc).__name__}: {exc}")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"{type(exc).__name__}: {exc}"}
            )

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        route = urlparse(self.path).path
        try:
            body = self._json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": f"bad request body: {exc}"})
            return
        try:
            if route == "/api/ask":
                self._send_json(*self.app.post_ask(body))
            elif route == "/api/register_task":
                self._send_json(*self.app.post_register_task(body))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"no such endpoint: {route}"})
        except BrokenPipeError:  # pragma: no cover
            return
        except Exception as exc:  # noqa: BLE001
            log_event("agent.http.error", route=route, error=f"{type(exc).__name__}: {exc}")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"{type(exc).__name__}: {exc}"}
            )

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib naming
        """DELETE /api/tasks/<id> — stop evaluating a standing task.

        Deleting a task never touches the action log: what it already fired stays on the
        Timeline with its reason and its clip (SPEC §6.4).
        """
        route = urlparse(self.path).path
        try:
            task_id = self._task_id_from(route)
            if task_id is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"no such endpoint: {route}"})
                return
            self._send_json(*self.app.delete_task(task_id))
        except BrokenPipeError:  # pragma: no cover
            return
        except Exception as exc:  # noqa: BLE001
            log_event("agent.http.error", route=route, error=f"{type(exc).__name__}: {exc}")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"{type(exc).__name__}: {exc}"}
            )

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib naming
        """PATCH /api/tasks/<id> — edit a standing task in place."""
        route = urlparse(self.path).path
        try:
            body = self._json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": f"bad request body: {exc}"})
            return
        try:
            task_id = self._task_id_from(route)
            if task_id is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"no such endpoint: {route}"})
                return
            self._send_json(*self.app.patch_task(task_id, body))
        except BrokenPipeError:  # pragma: no cover
            return
        except Exception as exc:  # noqa: BLE001
            log_event("agent.http.error", route=route, error=f"{type(exc).__name__}: {exc}")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"{type(exc).__name__}: {exc}"}
            )

    @staticmethod
    def _task_id_from(route: str) -> str | None:
        """``/api/tasks/<id>`` -> ``<id>``. Percent-decoded; empty and nested reject."""
        prefix = "/api/tasks/"
        if not route.startswith(prefix):
            return None
        task_id = unquote(route[len(prefix) :]).strip("/")
        return task_id or None

    # -- video ---------------------------------------------------------------------------

    def _serve_live_frame(self) -> None:
        """The camera's current view — what the system is watching *now*.

        A rolling JPEG written by the recorder's own ffmpeg as a second output (see
        `recorder.preview` in settings.yaml). It is emphatically NOT the archive and not
        an analysis input: v4l2 access to /dev/video0 is exclusive, so this is the only
        way to show a live view without taking the camera away from the recorder.

        404 rather than an error when the file is absent — the preview is optional, and
        "the recorder is not running" is a fact the UI should render as a placeholder,
        not an exception.
        """
        path = config.repo_path("recorder.preview.path")
        if not config.get("recorder.preview.enabled", False) or not path.is_file():
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"detail": "no live preview; is the recorder running with "
                           "recorder.preview.enabled?"},
            )
            return
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": str(exc)})
            return
        # ffmpeg rewrites this file in place ~2x/second, so a poll can catch it
        # mid-write and read zero bytes. Report that as "try again", never as an
        # empty image, which browsers cache as a broken one.
        if not body:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "frame mid-write"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        # The whole point is freshness; a cached live view is a still photograph.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self) -> None:
        t_from, t_to = self._param("t_from"), self._param("t_to")
        if t_from is None or t_to is None:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"detail": "t_from and t_to are required; video is fetched by range, never by file"},
            )
            return
        status, payload = self.app.video_clip(t_from, t_to)
        if not isinstance(payload, Path):
            self._send_json(status, payload)
            return
        self._send_file(payload)

    def _send_file(self, path: Path) -> None:
        """Serve a file, honouring a single ``Range`` header.

        The <video> element always sends one, and a server that answers 200 to a range
        request gets a player that will not seek.
        """
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = HTTPStatus.OK
        match = _RANGE_RE.fullmatch(self.headers.get("Range", "") or "")
        if match and size:
            first, last = match.group(1), match.group(2)
            if first:
                start = min(int(first), size - 1)
                end = min(int(last), size - 1) if last else size - 1
            elif last:  # suffix range: the last N bytes
                start = max(size - int(last), 0)
            status = HTTPStatus.PARTIAL_CONTENT

        length = max(end - start + 1, 0)
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # -- static -------------------------------------------------------------------------

    def _serve_static(self, route: str) -> None:
        """Serve ``ui/`` so the page and its endpoints share an origin (SPEC §11.1).

        ``ui/`` is owned elsewhere and is read-only from here. Paths are resolved and
        checked against the root, because ``..`` in a URL is not a routing question.
        """
        root = self.app.settings.ui_dir
        relative = unquote(route).lstrip("/") or "index.html"
        target = (root / relative).resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"not found: {route}"})
            return
        self._send_file(target)

    # -- websocket ------------------------------------------------------------------------

    def _upgrade(self) -> None:
        """RFC 6455 handshake, then hand the socket to the hub for the rest of its life."""
        key = self.headers.get("Sec-WebSocket-Key")
        upgrade = (self.headers.get("Upgrade") or "").lower()
        if not key or upgrade != "websocket":
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"detail": "/ws expects a WebSocket upgrade"}
            )
            return
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key(key))
        self.end_headers()

        connection = WebSocketConnection(
            self.rfile, self.wfile, peer=f"{self.client_address[0]}:{self.client_address[1]}"
        )
        self.app.hub.add(connection)
        try:
            connection.serve()
        finally:
            self.app.hub.remove(connection)
            self.close_connection = True


class _Server(ThreadingHTTPServer):
    """Threaded so a 90 s WebSocket read cannot hold up the next question."""

    daemon_threads = True
    allow_reuse_address = True


class AskServer:
    """Owns the socket and the app's lifecycle. ``port=0`` picks a free port (tests)."""

    def __init__(self, app: AgentApp, host: str | None = None, port: int | None = None) -> None:
        self.app = app
        handler = type("_BoundHandler", (_Handler,), {"app": app})
        socketserver.TCPServer.allow_reuse_address = True
        self._httpd = _Server(
            (host if host is not None else app.settings.host,
             port if port is not None else app.settings.port),
            handler,
        )
        self._thread: threading.Thread | None = None
        self._pinger: threading.Timer | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    def start(self) -> None:
        """Serve in a background thread. Returns once the socket is accepting."""
        self.app.start()
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._schedule_ping()
        log_event(
            "agent.server.started",
            host=self.host,
            port=self.port,
            backend=self.app.settings.backend,
            ui_dir=str(self.app.settings.ui_dir),
        )

    def serve_forever(self) -> None:
        """Blocking form, for ``python3 -m services.agent``."""
        self.app.start()
        self._schedule_ping()
        log_event(
            "agent.server.started",
            host=self.host,
            port=self.port,
            backend=self.app.settings.backend,
            ui_dir=str(self.app.settings.ui_dir),
        )
        try:
            self._httpd.serve_forever()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._pinger is not None:
            self._pinger.cancel()
            self._pinger = None
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.app.stop()
        log_event("agent.server.stopped")

    def _schedule_ping(self) -> None:
        """Keepalive. A silent socket behind a proxy dies without a word."""
        interval = self.app.settings.ws_ping_interval_seconds
        if interval <= 0:
            return

        def tick() -> None:
            self.app.hub.ping_all()
            self._schedule_ping()

        self._pinger = threading.Timer(interval, tick)
        self._pinger.daemon = True
        self._pinger.start()


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def build_app(
    *,
    settings: AgentSettings | None = None,
    index: IndexStore | None = None,
    actions: ActionServer | None = None,
    analyzer: Any | None = None,
    clip_cutter: ClipCutter | None = None,
    tasks: TaskRegistry | None = None,
) -> AgentApp:
    """Wire M3 from config. Every dependency is overridable, which is how tests work.

    The deep analyzer defaults to a lazy binding to ``services/worker`` that resolves at
    first escalation, so M3 starts and serves whether or not M4 exists yet.
    """
    resolved = settings or AgentSettings.from_config()
    store = index if index is not None else build_index()
    store.ensure_ready()
    action_server = actions if actions is not None else ActionServer()
    deep = analyzer if analyzer is not None else _default_analyzer(resolved)
    registry = JobRegistry(deep, resolved)
    chat_log = ChatLog(resolved.chat_log)
    toolbox = Toolbox(store, action_server, registry, resolved)
    agent = AskAgent(build_backend(resolved), toolbox, chat_log, resolved)
    return AgentApp(
        agent=agent,
        index=store,
        actions=action_server,
        jobs=registry,
        chat_log=chat_log,
        tasks=tasks if tasks is not None else SeedTaskRegistry(actions=action_server),
        hub=WebSocketHub(),
        settings=resolved,
        clip_cutter=clip_cutter if clip_cutter is not None else _default_cutter(),
    )


def _default_analyzer(settings: AgentSettings) -> Any:
    """M4 if it is importable, an honest failure if it is not.

    The import is attempted once, here, so the log says at startup whether escalation
    will reach a worker — better than discovering it on stage, inside a turn.
    """
    try:
        import services.worker as worker  # noqa: PLC0415 — probe, deliberately

        if callable(getattr(worker, "deep_analyze", None)) or callable(
            getattr(worker, "submit", None)
        ):
            return WorkerAnalyzer(worker, poll_interval_s=settings.deep_poll_interval_seconds)
        detail = "services.worker exposes no deep_analyze/submit yet (M4 in progress)"
    except ImportError as exc:  # pragma: no cover — M4 lands before the demo
        detail = f"services.worker is not importable: {exc}"
    log_event("agent.deep.unavailable", detail=detail)
    return UnavailableAnalyzer(detail)


def _default_cutter() -> ClipCutter:
    """ffmpeg if it is on PATH, otherwise a cutter that admits it cut nothing."""
    from services.mcp import ffmpeg_cutter_from_config  # noqa: PLC0415 — construction only

    cutter = ffmpeg_cutter_from_config()
    return cutter if cutter.available() else NullClipCutter()


def main(argv: list[str] | None = None) -> int:
    """``python3 -m services.agent`` — the M3 process (SPEC §11.1)."""
    import argparse  # noqa: PLC0415 — CLI only
    import logging  # noqa: PLC0415

    settings = AgentSettings.from_config()
    parser = argparse.ArgumentParser(description="M3 — ask agent and its server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=str(config.get("logging.level", "INFO")), format="%(message)s"
    )
    server = AskServer(build_app(settings=settings), host=args.host, port=args.port)
    print(f"[m3] ask agent on http://{args.host}:{args.port}  (backend={settings.backend})")
    print(f"[m3] ui from {settings.ui_dir} — flip MODE to 'live' in ui/static/data.js")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[m3] stopped")
    return 0
