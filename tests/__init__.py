"""Test package init — runs before any test module is imported.

**Its whole job is to keep the suite out of production data.**

`config/settings.yaml` sets `index.store.memory_path` so the in-memory index survives
across processes (M1, M3 and M5 are separate processes; without it ingest's captions die
with the ingest process). The side effect is that every bare `build_index()` — including
the dozens in this suite — persists to that same file. Running the tests then writes
fixture chunks into the real index, where they are indistinguishable from footage:
retrieval starts citing "a white panel van reverses toward the loading door" and the
archive player is asked for a segment that was never recorded.

That is not hypothetical; it happened, and the symptom was a UI stitching 20 files for a
5-second chunk. So the redirect lives here rather than in each TestCase: a test that
forgets to opt out of persistence should be impossible, not merely discouraged.

It is done by writing a temporary settings file and pointing ``SPARK_SETTINGS`` at it,
rather than by mutating the loaded config dict — because the suite calls
``config.load.cache_clear()`` in places, which re-reads from disk and would silently
discard an in-memory override. Redirecting the *file* survives that.

Add any other path the suite must not write to here, not in individual tests.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import yaml

_TMP = Path(tempfile.mkdtemp(prefix="spark-tests-"))
_REPO = Path(__file__).resolve().parent.parent

_settings = yaml.safe_load((_REPO / "config" / "settings.yaml").read_text()) or {}
_settings.setdefault("index", {}).setdefault("store", {})["memory_path"] = str(
    _TMP / "index.jsonl"
)
_paths = _settings.setdefault("paths", {})
_paths["action_log"] = str(_TMP / "actions.jsonl")
_paths["chat_log"] = str(_TMP / "chats.jsonl")
_paths["clips"] = str(_TMP / "clips")
# The archive, for the same reason as the rest — plus one that is worse than writing to
# it. `services/retention.py` DELETES segment files, and a route test that reached the
# real `data/archive` would silently unlink hours of recorded footage. It is created
# rather than merely renamed so `timecode.list_segments` sees an empty archive instead of
# raising at a directory that is not there.
_paths["archive"] = str(_TMP / "archive")
(_TMP / "archive").mkdir(parents=True, exist_ok=True)

_SETTINGS_FILE = _TMP / "settings.yaml"
_SETTINGS_FILE.write_text(yaml.safe_dump(_settings, sort_keys=False))

# Set unconditionally: an ambient SPARK_SETTINGS from the shell would otherwise let the
# suite write to whatever that file points at, which is the failure this module exists
# to prevent.
os.environ["SPARK_SETTINGS"] = str(_SETTINGS_FILE)


@atexit.register
def _cleanup() -> None:
    shutil.rmtree(_TMP, ignore_errors=True)
