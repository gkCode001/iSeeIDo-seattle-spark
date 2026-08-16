"""Tests for the SPEC §11 UI's contract with the rest of the system.

There is no JS test runner here and a hackathon does not have room for one, so these
tests cover the part of the UI that can silently rot without anyone noticing: the mock
fixtures. They are the handshake with M3 — if they drift from ``shared/schema.py``, the
page keeps rendering beautifully against data the real endpoints will never send, and
the failure surfaces on stage.

Every assertion below is a rule from CLAUDE.md or SPEC §11:

* fixtures are exact ``to_dict()`` output (schema is the single source of truth)
* the UI ships no remote assets (invariant 10)
* the fixture config subset still matches ``config/settings.yaml`` (no magic numbers)
* amendments reference a real parent and sort *after* it in UTC (§6.4 / §11.4)
* cited chunk ids resolve, or the "clickable cited range" is a dead chip (§11.2)

Stdlib ``unittest`` on purpose — see pyproject.toml:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from shared.schema import (
    ActionLogEntry,
    ChatTurn,
    ChunkRecord,
    DeepJob,
    Task,
    from_iso,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "ui"
MOCK_DIR = UI_DIR / "mock"


def load(name: str) -> dict:
    with (MOCK_DIR / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


class TestFixturesMatchSchema(unittest.TestCase):
    """Round-trip every fixture row through shared/schema.py.

    ``from_dict`` then ``to_dict`` must reproduce the row byte for byte. That catches a
    renamed field, a dropped field, a naive timestamp and a bad enum value in one go.
    """

    def assert_roundtrip(self, cls, rows, label):
        for row in rows:
            with self.subTest(label=label, row=row.get(next(iter(row)))):
                self.assertEqual(cls.from_dict(row).to_dict(), row)

    def test_chunks(self):
        self.assert_roundtrip(ChunkRecord, load("chunks.json")["chunks"], "chunk")

    def test_tasks(self):
        self.assert_roundtrip(Task, load("tasks.json")["tasks"], "task")

    def test_actions(self):
        self.assert_roundtrip(ActionLogEntry, load("actions.json")["entries"], "action")

    def test_chat_turns(self):
        self.assert_roundtrip(ChatTurn, load("chat_turns.json")["turns"], "turn")

    def test_jobs(self):
        self.assert_roundtrip(DeepJob, list(load("jobs.json")["jobs"].values()), "job")

    def test_timestamps_are_z_suffixed(self):
        """SPEC §11.5: every payload underneath stays UTC. Local time is render-only."""
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
        for name in ("chunks.json", "actions.json", "chat_turns.json", "jobs.json", "monitor_state.json"):
            blob = json.dumps(load(name))
            for value in re.findall(r'"(\d{4}-\d{2}-\d{2}T[^"]+)"', blob):
                with self.subTest(file=name, value=value):
                    self.assertRegex(value, pattern)


class TestActionLogShape(unittest.TestCase):
    """SPEC §6.4 / §11.4 — the append-only log and its amendments."""

    def setUp(self):
        self.entries = [ActionLogEntry.from_dict(d) for d in load("actions.json")["entries"]]
        self.by_id = {e.entry_id: e for e in self.entries}

    def test_amendments_reference_a_real_parent(self):
        for e in self.entries:
            if e.parent_id:
                self.assertIn(e.parent_id, self.by_id, f"{e.entry_id} points at a missing parent")

    def test_amendments_sort_after_their_parent_in_utc(self):
        """The fold in the Timeline pane assumes it, and UTC is the only safe key:
        two instants inside a DST fall-back hour compare EQUAL as local wall clock."""
        for e in self.entries:
            if e.parent_id:
                self.assertGreater(e.ts, self.by_id[e.parent_id].ts)

    def test_a_retraction_exists_to_demo(self):
        """SPEC §11.4 says show retractions on stage. If the fixture loses its only
        retraction, the Timeline pane silently stops making the point."""
        statuses = {e.status.value for e in self.entries if e.parent_id}
        self.assertIn("retracted", statuses)
        self.assertIn("verified", statuses)

    def test_footage_range_is_not_the_row_timestamp(self):
        """t_start/t_end are the footage; ts is when the row was appended. Conflating
        them is what makes 'why did you alert at 21:11?' unanswerable."""
        for e in self.entries:
            self.assertLessEqual(e.t_start, e.t_end)


class TestCrossReferences(unittest.TestCase):
    """A citation that does not resolve renders as a chip that scrubs nowhere."""

    def setUp(self):
        self.chunk_ids = {c["chunk_id"] for c in load("chunks.json")["chunks"]}
        self.job_ids = set(load("jobs.json")["jobs"])
        self.task_ids = {t["task_id"] for t in load("tasks.json")["tasks"]}

    def test_chat_turn_citations_resolve(self):
        for turn in load("chat_turns.json")["turns"]:
            for cid in turn["cited_chunk_ids"]:
                self.assertIn(cid, self.chunk_ids, f"{turn['turn_id']} cites a missing chunk")

    def test_chat_turn_jobs_resolve(self):
        """SPEC §11.4: the turn persists the job, so a reload can rebuild the refinement."""
        for turn in load("chat_turns.json")["turns"]:
            if turn["job_id"]:
                self.assertIn(turn["job_id"], self.job_ids)

    def test_action_jobs_resolve(self):
        for entry in load("actions.json")["entries"]:
            if entry["job_id"]:
                self.assertIn(entry["job_id"], self.job_ids)

    def test_monitor_state_covers_registered_tasks(self):
        state_ids = {t["task_id"] for t in load("monitor_state.json")["tasks"]}
        self.assertEqual(state_ids, self.task_ids)

    def test_ask_script_citations_resolve(self):
        doc = load("ask_script.json")
        ids = {s["id"] for s in doc["scripts"]}
        self.assertIn(doc["default_script"], ids)
        for script in doc["scripts"]:
            for cid in script["turn"]["cited_chunk_ids"]:
                self.assertIn(cid, self.chunk_ids)
            if script["job"]:
                for cid in script["job"].get("cited_chunk_ids", []):
                    self.assertIn(cid, self.chunk_ids)

    def test_escalation_arc_is_rehearsable(self):
        """The one sequence the demo cannot lose: provisional, then a refinement that
        lands later and well inside the stated timeout."""
        import yaml

        with (REPO_ROOT / "config" / "settings.yaml").open("r", encoding="utf-8") as fh:
            timeout = yaml.safe_load(fh)["agent"]["deep"]["timeout_seconds"]

        doc = load("ask_script.json")
        escalations = [s for s in doc["scripts"] if s["turn"]["grounded"] is False]
        self.assertTrue(escalations, "no escalating script left to rehearse")
        for script in escalations:
            self.assertIsNotNone(script["job"], "an escalated turn must carry a job")
            self.assertGreater(script["refined_ms"], script["provisional_ms"])
            self.assertLess(script["refined_ms"] / 1000, timeout)

        self.assertTrue(
            [s for s in doc["scripts"] if s["turn"]["grounded"] is True],
            "SPEC §10 D6 wants a question the index genuinely answers, too",
        )


class TestConfigSubsetMatchesSettings(unittest.TestCase):
    """ui/mock/config.json is the offline fallback for GET /api/config. Every value in
    it must still equal config/settings.yaml, or the UI demos numbers nobody tuned."""

    def test_values_match(self):
        import yaml

        with (REPO_ROOT / "config" / "settings.yaml").open("r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)
        fixture = load("config.json")

        def walk(fix, real, path=""):
            for key, value in fix.items():
                if key.startswith("_"):
                    continue
                here = f"{path}.{key}".lstrip(".")
                self.assertIn(key, real, f"{here} is not in config/settings.yaml")
                if isinstance(value, dict):
                    walk(value, real[key], here)
                else:
                    self.assertEqual(value, real[key], f"{here} drifted from settings.yaml")

        walk(fixture, settings)


class TestNoRemoteAssets(unittest.TestCase):
    """CLAUDE.md invariant 10. The demo is rehearsed and run with the network off."""

    SHIPPED = ["index.html", "browse.html", "static", "mock"]

    def shipped_files(self):
        for entry in self.SHIPPED:
            path = UI_DIR / entry
            if path.is_file():
                yield path
            else:
                yield from (p for p in path.rglob("*") if p.is_file())

    def test_no_absolute_urls(self):
        """Anything served to the browser must resolve inside ui/. serve.py is excluded
        deliberately: it is a dev-time previewer, not part of the shipped page."""
        offenders = []
        for path in self.shipped_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"https?://", line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "remote reference in a shipped UI file:\n" + "\n".join(offenders))

    def test_no_remote_loading_syntax(self):
        banned = re.compile(r"@import|integrity=|crossorigin|//cdn\.|unpkg|jsdelivr|googleapis")
        for path in self.shipped_files():
            text = path.read_text(encoding="utf-8")
            for match in banned.finditer(text):
                # Prose in a comment is fine; a real directive is not.
                line = text[: match.start()].count("\n") + 1
                snippet = text.splitlines()[line - 1].strip()
                if snippet.startswith(("*", "//", "<!--")) or " no " in snippet.lower():
                    continue
                self.fail(f"{path.relative_to(REPO_ROOT)}:{line}: {snippet}")

    def test_page_references_only_local_files(self):
        for page in ("index.html", "browse.html"):
            html = (UI_DIR / page).read_text(encoding="utf-8")
            for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
                if ref.startswith("data:"):
                    continue
                with self.subTest(page=page, ref=ref):
                    self.assertFalse(ref.startswith(("http:", "https:", "//")))
                    self.assertTrue((UI_DIR / ref).is_file(), f"{ref} is missing from ui/")

    def test_the_two_pages_link_to_each_other(self):
        """The index browser is only discoverable if the console points at it, and a
        reader who lands on the browser needs a way back. A dead-end page is one nobody
        finds at hour 39."""
        console = (UI_DIR / "index.html").read_text(encoding="utf-8")
        browser = (UI_DIR / "browse.html").read_text(encoding="utf-8")
        self.assertIn('href="browse.html"', console)
        self.assertIn('href="index.html"', browser)


class TestTimeDiscipline(unittest.TestCase):
    """SPEC §11.5: conversion to local happens in exactly one helper, at render."""

    def js_files(self):
        return sorted((UI_DIR / "static").glob("*.js"))

    def test_only_time_js_converts_to_local(self):
        offenders = []
        for path in self.js_files():
            if path.name == "time.js":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"toLocale\w*|getHours\(|getMinutes\(|Intl\.DateTimeFormat", line):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "local-time conversion outside ui/static/time.js:\n" + "\n".join(offenders),
        )

    def test_fixture_times_parse_as_utc(self):
        for chunk in load("chunks.json")["chunks"]:
            self.assertEqual(from_iso(chunk["t_start"]).utcoffset().total_seconds(), 0)


if __name__ == "__main__":
    unittest.main()


class TestArchivePlayerDrivesTheVideoElement(unittest.TestCase):
    """In live mode the picture is a real <video>, and the element has to be *driven*.

    The mock-mode canvas is painted frame-by-frame from `state.posMs`, so it animates
    purely from the rAF loop. A <video> does the opposite: it owns its own clock and has
    to be started, paused and seeked explicitly. Wiring only half of that is what left
    the archive player showing a single frame — the element had a src from the time
    range and was simply never started.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = (UI_DIR / "static" / "player.js").read_text(encoding="utf-8")

    def test_play_starts_the_element(self) -> None:
        self.assertIn("els.video.play()", self.src)

    def test_pause_stops_the_element(self) -> None:
        self.assertIn("els.video.pause()", self.src)

    def test_scrubbing_seeks_the_element(self) -> None:
        """Otherwise the slider moves the digits while the picture sits still."""
        self.assertIn("els.video.currentTime =", self.src)

    def test_the_playhead_follows_the_element_not_a_separate_timer(self) -> None:
        """The burned-in wall clock, the segment strip and the digits must agree with
        the pixels on screen; a separately-ticking playhead drifts away from the frame
        it claims to describe."""
        self.assertIn("els.video.currentTime * 1000", self.src)

    def test_a_rejected_autoplay_is_reported_not_swallowed(self) -> None:
        """'Nothing happens' is indistinguishable from the bug this replaces."""
        self.assertIn("autoplay blocked", self.src)

    def test_the_video_can_autoplay_at_all(self) -> None:
        """Browsers only permit programmatic playback when the element is muted, so the
        markup has to carry it — play() alone is not enough."""
        markup = (UI_DIR / "index.html").read_text(encoding="utf-8")
        tag = markup[markup.index("data-player-video") : markup.index("data-player-video") + 120]
        self.assertIn("muted", tag)
        self.assertIn("playsinline", tag)


class TestStandingTaskForm(unittest.TestCase):
    """SPEC §11.3's task form has to be escapable.

    Measured at 1366x768: opened, the form was 104 px taller than the pane, so the
    register button sat below it and the only visible way out was to submit. Combined
    with `list-style: none` hiding the disclosure marker, the panel read as a one-way
    door.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.css = (UI_DIR / "static" / "app.css").read_text(encoding="utf-8")
        cls.html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (UI_DIR / "static" / "watch.js").read_text(encoding="utf-8")

    @staticmethod
    def _rule(css: str, selector: str) -> str:
        """The declarations of one rule. Sliced to the closing brace rather than a fixed
        character count, which a comment inside the rule would silently push past."""
        start = css.index(selector + " {")
        return css[start : css.index("}", start) + 1]

    def test_the_form_is_bounded_and_scrolls(self) -> None:
        """A definite max-height plus overflow keeps the foot of the form reachable;
        without it the content paints past the pane."""
        block = self._rule(self.css, ".task-form")
        self.assertIn("max-height", block)
        self.assertIn("overflow-y: auto", block)
        self.assertIn("vh", block, "a percentage max-height does not resolve here")

    def test_the_heading_stays_visible_while_the_form_scrolls(self) -> None:
        head = self._rule(self.css, ".task-form > summary")
        self.assertIn("position: sticky", head)

    def test_there_is_a_disclosure_affordance(self) -> None:
        """list-style: none removes the native marker, so one has to be drawn."""
        self.assertIn(".task-form > summary::before", self.css)
        self.assertIn(".task-form[open] > summary::before", self.css)

    def test_there_is_an_explicit_way_out(self) -> None:
        self.assertIn("data-watch-cancel", self.html)
        self.assertIn("data-watch-cancel", self.js)
        self.assertIn("details.open = false", self.js)

    def test_cancelling_clears_the_half_typed_task(self) -> None:
        """A partly-filled task left behind would reappear later looking real.

        Asserts the BEHAVIOUR through whatever helper cancel delegates to, rather than
        scanning a fixed slice after the listener — that slice stopped seeing the reset
        the moment the handler was extracted into closeForm().
        """
        self.assertIn("closeForm()", self.js[self.js.index("data-watch-cancel") :][:400])
        close = self.js[self.js.index("function closeForm()") :]
        close = close[: close.index("\n  }") + 4]
        self.assertIn("open = false", close)
        self.assertIn("reset()", close)


class TestRetentionControl(unittest.TestCase):
    """The delete-old-footage button — the only control on this page that destroys data.

    Structural assertions, like the standing-task form's above: there is no JS runner
    here, so what is pinned is the shape that makes the control safe. Each of these has
    failed silently in some other UI at some point — a confirm that got refactored into a
    one-click action, a destructive control left live against fixtures.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (UI_DIR / "static" / "retention.js").read_text(encoding="utf-8")
        cls.data = (UI_DIR / "static" / "data.js").read_text(encoding="utf-8")

    def test_the_button_is_wired_and_shipped(self) -> None:
        self.assertIn("data-purge", self.html)
        self.assertIn('src="static/retention.js"', self.html)

    def test_clicking_it_does_not_delete(self) -> None:
        """The click opens the panel and asks for a plan. Deleting is a second click on a
        separate button, against counts the operator has now seen."""
        opener = self.js[self.js.index('refs.button.addEventListener') :][:120]
        self.assertIn("open", opener)
        self.assertIn("retentionPlan()", self.js)
        confirm = self.js[self.js.index("data-purge-confirm") :]
        self.assertIn("applyRetention", confirm)

    def test_the_confirm_and_the_apply_are_different_buttons(self) -> None:
        self.assertIn("data-purge-confirm", self.html)
        self.assertIn("data-purge-cancel", self.html)

    def test_mock_mode_cannot_reach_the_destructive_call(self) -> None:
        """Every other pane degrades to fixtures. A destructive control that degraded to
        a scripted success would report deleting footage that was never touched — and the
        operator would believe it."""
        self.assertIn("isMock()", self.js)
        apply_fn = self.data[self.data.index("function applyRetention") :][:400]
        self.assertIn("isMock()", apply_fn)
        self.assertIn("reject", apply_fn)

    def test_the_age_is_read_from_config_not_hard_coded(self) -> None:
        """CLAUDE.md: no magic numbers. The label says what settings.yaml says."""
        self.assertIn("retention.max_age_seconds", self.js)


class TestIndexNavLink(unittest.TestCase):
    """The console links to the index browser, and must not navigate away to get there.

    Both live panes are stateful: the Ask log, an in-flight deep job and the live camera
    poll all live in the console document. Following a same-tab link mid-escalation
    throws away the provisional/refined pair that is the point of the demo.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.console = (UI_DIR / "index.html").read_text(encoding="utf-8")
        cls.browse = (UI_DIR / "browse.html").read_text(encoding="utf-8")

    def _anchor(self, html: str, href: str) -> str:
        start = html.index(f'href="{href}"')
        return html[html.rindex("<a", 0, start) : html.index(">", start) + 1]

    def test_the_index_link_opens_a_new_tab(self) -> None:
        anchor = self._anchor(self.console, "browse.html")
        self.assertIn('target="_blank"', anchor)

    def test_the_new_tab_cannot_reach_back_through_window_opener(self) -> None:
        anchor = self._anchor(self.console, "browse.html")
        self.assertIn('rel="noopener"', anchor)

    def test_the_browse_page_still_links_back_in_place(self) -> None:
        """Opened directly or bookmarked, the browser page must be able to navigate to
        the console normally rather than spawning yet another tab."""
        self.assertIn('href="index.html"', self.browse)
        anchor = self._anchor(self.browse, "index.html")
        self.assertNotIn("target=", anchor)

    def test_the_browse_page_ships_no_remote_assets_either(self) -> None:
        """Invariant 10 applies to every page, not just the console."""
        for match in re.findall(r"https?://[^\"' )]*", self.browse):
            self.fail(f"remote reference in browse.html: {match}")


class TestModeDetection(unittest.TestCase):
    """The console must not show convincing fake data by default.

    MODE used to default to "mock", from when there was no backend. Once M3 served the
    page, opening it without ?mode=live produced a fully populated console — a live
    camera pane reading "unavailable", a Timeline of six invented alerts, a funnel
    mid-cooldown — all fixtures, with one small pill as the only warning.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = (UI_DIR / "static" / "data.js").read_text(encoding="utf-8")
        cls.loader = cls.js[cls.js.index("function loadConfig()") :][:1600]

    def test_mode_is_detected_rather_than_assumed(self) -> None:
        self.assertIn('MODE = "live"', self.loader)
        self.assertIn('MODE = "mock"', self.loader)

    def test_liveness_is_probed_on_an_m3_only_endpoint(self) -> None:
        """ui/serve.py also serves /api/config, so probing THAT would call the mock
        previewer live and then 404 every other pane."""
        self.assertIn("ENDPOINTS.tasks", self.loader)
        probe = self.loader[: self.loader.index("ENDPOINTS.tasks")]
        self.assertNotIn("getJSON(ENDPOINTS.config)", probe)

    def test_an_explicit_mode_always_wins(self) -> None:
        """?mode=mock on a live box is how the demo is rehearsed."""
        self.assertIn("MODE_PINNED = true", self.js)
        self.assertIn("MODE_PINNED", self.loader)

    def test_real_settings_are_used_even_in_mock_mode(self) -> None:
        """ui/serve.py serves /api/config precisely so a mock preview still renders the
        true timezone and thresholds."""
        self.assertIn("getJSON(ENDPOINTS.config)", self.loader)
        self.assertIn("config.json", self.loader)
