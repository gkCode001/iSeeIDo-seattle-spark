#!/usr/bin/env python3
"""Preview server for the SPEC §11 UI. Standard library only, no dependencies.

This is NOT M3. It exists so the page can be opened, demoed and rehearsed before the
FastAPI process exists — and so the fixtures can be loaded at all, since ``file://``
blocks ``fetch`` and the page would otherwise come up empty.

    python3 ui/serve.py                 # http://127.0.0.1:8090
    python3 ui/serve.py --port 9000

It serves ``ui/`` and one live endpoint:

    GET /api/config   ->  config/settings.yaml, as JSON

That endpoint is the reason the preview shows the operator's real display timezone and
real thresholds rather than the fixture copy: the client asks for it first and only
falls back to ``ui/mock/config.json`` when it is absent (SPEC §11.5 / no magic numbers).
``config/settings.yaml`` is read, never written — it is owned elsewhere.

When M3 comes up, this file stops being needed: point the browser at the FastAPI process
and flip ``MODE`` in ``ui/static/data.js``.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
REPO_ROOT = UI_DIR.parent
SETTINGS = REPO_ROOT / "config" / "settings.yaml"


def load_settings() -> dict | None:
    """Best-effort read of config/settings.yaml.

    PyYAML is already a project dependency (shared/config.py imports it), but this
    previewer must run even in a bare interpreter, so a missing import is not fatal —
    the client falls back to ui/mock/config.json and says so in the console.
    """
    try:
        import yaml  # noqa: PLC0415 - optional at preview time, by design
    except ImportError:
        return None
    if not SETTINGS.is_file():
        return None
    with SETTINGS.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?")[0] == "/api/config":
            settings = load_settings()
            if settings is None:
                self.send_error(404, "settings.yaml unavailable; client will use ui/mock/config.json")
                return
            body = json.dumps(settings, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        # Rehearsal aid: never serve a stale pane out of the browser cache.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[ui] " + (fmt % args) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        settings_state = "config/settings.yaml" if load_settings() else "ui/mock/config.json (fallback)"
        print(f"[ui] serving {UI_DIR} on http://{args.host}:{args.port}")
        print(f"[ui] tunables from {settings_state}")
        print("[ui] mock fixtures; flip MODE in ui/static/data.js for the real endpoints")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[ui] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
