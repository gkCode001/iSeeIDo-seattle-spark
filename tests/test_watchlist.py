"""``services/ingest/watchlist.py`` — the task checklist M1 appends to its caption prompt.

The failure mode being guarded against is silence: a watchlist that stops reaching the
captioner does not raise, it just means standing tasks quietly go back to matching on
whatever the caption happened to mention.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.ingest.watchlist import Watchlist, write_watchlist

PREAMBLE = "Then add a final line starting \"WATCHING:\". Items:"
PROMPT = "Describe this moment."

SEED_YAML = """
tasks:
  - task_id: fire-door-blocked
    describe: a vehicle stopped in front of the fire door
    enabled: true
  - task_id: retired-task
    describe: something nobody watches for any more
    enabled: false
"""


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)

    def _watchlist(self, path: Path | None, seed: Path | None = None, **kw) -> Watchlist:
        return Watchlist(path, seed_path=seed, preamble=PREAMBLE, **kw)

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text(json.dumps({"tasks": rows}), encoding="utf-8")


class RenderTests(_Tmp):
    def test_items_are_appended_after_the_description_prompt(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": "a", "describe": "a red van"}])
        out = self._watchlist(path).apply(PROMPT)
        self.assertTrue(out.startswith(PROMPT), "the general description must come first")
        self.assertIn(PREAMBLE, out)
        self.assertIn("- a red van", out)

    def test_no_tasks_leaves_the_prompt_untouched(self) -> None:
        path = self.root / "w.json"
        self._write(path, [])
        self.assertEqual(self._watchlist(path).apply(PROMPT), PROMPT)

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(self._watchlist(self.root / "nope.json").apply(PROMPT), PROMPT)

    def test_unparseable_file_degrades_to_no_checklist(self) -> None:
        path = self.root / "w.json"
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self._watchlist(path).apply(PROMPT), PROMPT)

    def test_disabled_watchlist_never_touches_the_prompt(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": "a", "describe": "a red van"}])
        self.assertEqual(self._watchlist(path, enabled=False).apply(PROMPT), PROMPT)

    def test_disabled_tasks_do_not_steer_captions(self) -> None:
        path = self.root / "w.json"
        self._write(
            path,
            [
                {"task_id": "a", "describe": "a red van", "enabled": True},
                {"task_id": "b", "describe": "a blue van", "enabled": False},
            ],
        )
        out = self._watchlist(path).apply(PROMPT)
        self.assertIn("a red van", out)
        self.assertNotIn("a blue van", out)

    def test_rows_missing_a_field_are_skipped_not_rendered_blank(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": "a"}, {"describe": "orphan"}, {"task_id": "b", "describe": "ok"}])
        out = self._watchlist(path).apply(PROMPT)
        self.assertIn("- ok", out)
        self.assertNotIn("- \n", out)


class CapTests(_Tmp):
    def test_max_items_truncates(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": f"t{i}", "describe": f"item {i}"} for i in range(10)])
        out = self._watchlist(path, max_items=3).apply(PROMPT)
        self.assertIn("item 2", out)
        self.assertNotIn("item 3", out)

    def test_truncation_is_logged_because_a_silent_cap_reads_as_a_detection_bug(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": f"t{i}", "describe": f"item {i}"} for i in range(5)])
        with self.assertLogs("services.ingest.watchlist", level="WARNING") as caught:
            self._watchlist(path, max_items=2).suffix()
        self.assertTrue(any("truncated" in line for line in caught.output))


class SeedFallbackTests(_Tmp):
    def test_seed_is_used_when_the_runtime_file_is_absent(self) -> None:
        seed = self.root / "tasks.yaml"
        seed.write_text(SEED_YAML, encoding="utf-8")
        out = self._watchlist(self.root / "nope.json", seed=seed).apply(PROMPT)
        self.assertIn("a vehicle stopped in front of the fire door", out)

    def test_seed_respects_enabled_false(self) -> None:
        seed = self.root / "tasks.yaml"
        seed.write_text(SEED_YAML, encoding="utf-8")
        out = self._watchlist(self.root / "nope.json", seed=seed).apply(PROMPT)
        self.assertNotIn("nobody watches", out)

    def test_runtime_file_wins_over_the_seed(self) -> None:
        seed = self.root / "tasks.yaml"
        seed.write_text(SEED_YAML, encoding="utf-8")
        path = self.root / "w.json"
        self._write(path, [{"task_id": "live", "describe": "a runtime task"}])
        out = self._watchlist(path, seed=seed).apply(PROMPT)
        self.assertIn("a runtime task", out)
        self.assertNotIn("fire door", out)


class RefreshTests(_Tmp):
    def test_a_task_added_while_ingest_runs_reaches_the_next_caption(self) -> None:
        """The whole point of polling: CRUD happens in another process, mid-run."""
        path = self.root / "w.json"
        self._write(path, [{"task_id": "a", "describe": "a red van"}])
        w = self._watchlist(path)
        self.assertNotIn("a blue bicycle", w.apply(PROMPT))

        self._write(path, [{"task_id": "a", "describe": "a red van"},
                           {"task_id": "b", "describe": "a blue bicycle"}])
        self.assertIn("a blue bicycle", w.apply(PROMPT))

    def test_a_deleted_task_stops_steering_captions(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": "a", "describe": "a red van"}])
        w = self._watchlist(path)
        self.assertIn("a red van", w.apply(PROMPT))
        self._write(path, [])
        self.assertNotIn("a red van", w.apply(PROMPT))

    def test_refresh_is_a_no_op_when_nothing_changed(self) -> None:
        path = self.root / "w.json"
        self._write(path, [{"task_id": "a", "describe": "a red van"}])
        w = self._watchlist(path)
        w.refresh()
        self.assertFalse(w.refresh())


class WriteWatchlistTests(_Tmp):
    def test_round_trips_through_the_reader(self) -> None:
        path = self.root / "w.json"

        class _T:
            def __init__(self, tid, d, e=True):
                self.task_id, self.describe, self.enabled = tid, d, e

        write_watchlist(path, [_T("a", "a red van"), _T("b", "a blue van", False)])
        out = self._watchlist(path).apply(PROMPT)
        self.assertIn("a red van", out)
        self.assertNotIn("a blue van", out)

    def test_accepts_mappings_as_well_as_objects(self) -> None:
        path = self.root / "w.json"
        write_watchlist(path, [{"task_id": "a", "describe": "a red van"}])
        self.assertIn("a red van", self._watchlist(path).apply(PROMPT))

    def test_creates_the_parent_directory(self) -> None:
        path = self.root / "nested" / "deep" / "w.json"
        write_watchlist(path, [{"task_id": "a", "describe": "a red van"}])
        self.assertTrue(path.is_file())

    def test_no_temp_file_is_left_behind(self) -> None:
        """M1 polls this path; a torn read drops the checklist for a whole window."""
        path = self.root / "w.json"
        write_watchlist(path, [{"task_id": "a", "describe": "a red van"}])
        self.assertEqual([p.name for p in self.root.iterdir()], ["w.json"])


if __name__ == "__main__":
    unittest.main()
