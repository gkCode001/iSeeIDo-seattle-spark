"""``python3 -m services.agent`` — run M3.

Serves the SPEC §11 UI and the endpoints ``ui/static/data.js`` declares, on
``agent.host``/``agent.port``. Stdlib only: fastapi and uvicorn are not installed on
this box (CLAUDE.md — do not add dependencies).
"""

from __future__ import annotations

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
