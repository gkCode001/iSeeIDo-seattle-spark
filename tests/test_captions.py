"""``shared/captions.py`` — splitting a task-aware caption.

This module is load-bearing in a quiet way: get it wrong and standing tasks either stop
firing or fire on everything, with nothing in the logs pointing here. The regression that
matters most is at the bottom — embedding the whole caption *inverts* stage 1's ranking.
"""

from __future__ import annotations

import unittest

from shared.captions import ABSENT, PRESENT, split_caption

DESC = (
    "Four individuals are seated around a table. On the left, a woman is looking down "
    "at a laptop screen. A doorway is visible in the background on the right."
)
BLOCK = (
    "WATCHING:\n"
    "- a vehicle stopped in front of the fire door: absent\n"
    "- a person unloading boxes from a vehicle at the loading bay: absent\n"
    "- a person wearing a red wristband: present"
)
FULL = f"{DESC}\n\n{BLOCK}"

RED = "a person wearing a red wristband"
FIRE = "a vehicle stopped in front of the fire door"


class SplitCaptionTests(unittest.TestCase):
    def test_description_excludes_the_block(self) -> None:
        parts = split_caption(FULL)
        self.assertEqual(parts.description, DESC)
        self.assertNotIn("WATCHING", parts.description)
        self.assertNotIn("fire door", parts.description)

    def test_verdicts_are_parsed_per_item(self) -> None:
        parts = split_caption(FULL)
        self.assertTrue(parts.has_watchlist)
        self.assertEqual(len(parts.watching), 3)
        self.assertEqual(sorted(set(parts.watching.values())), [ABSENT, PRESENT])

    def test_verdict_for_matches_the_task_description(self) -> None:
        parts = split_caption(FULL)
        self.assertIs(parts.verdict_for(RED), True)
        self.assertIs(parts.verdict_for(FIRE), False)

    def test_absent_is_false_not_none(self) -> None:
        """The distinction gates the fallback: None means "unanswered", not "no"."""
        parts = split_caption(FULL)
        self.assertIsNotNone(parts.verdict_for(FIRE))
        self.assertIs(parts.verdict_for(FIRE), False)

    def test_unanswered_task_returns_none_so_the_caller_falls_back(self) -> None:
        parts = split_caption(f"{DESC}\n\nWATCHING:\n- {FIRE}: absent")
        self.assertIsNone(parts.verdict_for("a dog running across the yard"))

    def test_plain_caption_has_no_watchlist(self) -> None:
        parts = split_caption(DESC)
        self.assertFalse(parts.has_watchlist)
        self.assertEqual(parts.description, DESC)
        self.assertIsNone(parts.verdict_for(RED))

    def test_empty_and_none_are_safe(self) -> None:
        for value in (None, "", "   "):
            parts = split_caption(value)
            self.assertFalse(parts.has_watchlist)
            self.assertIsNone(parts.verdict_for(RED))


class FormattingToleranceTests(unittest.TestCase):
    """A VLM writes the same block five ways. Losing it to an asterisk disarms every task."""

    def test_bold_header_and_em_dash(self) -> None:
        parts = split_caption(f"{DESC}\n\n**WATCHING:**\n* {RED} — present")
        self.assertIs(parts.verdict_for(RED), True)

    def test_yes_no_answers(self) -> None:
        parts = split_caption(f"{DESC}\n\nWATCHING:\n1) {RED}: yes\n2) {FIRE}: no")
        self.assertIs(parts.verdict_for(RED), True)
        self.assertIs(parts.verdict_for(FIRE), False)

    def test_not_present_reads_as_absent(self) -> None:
        parts = split_caption(f"{DESC}\n\nWATCHING:\n- {RED}: not present")
        self.assertIs(parts.verdict_for(RED), False)

    def test_lowercase_header(self) -> None:
        parts = split_caption(f"{DESC}\n\nwatching:\n- {RED}: present")
        self.assertIs(parts.verdict_for(RED), True)

    def test_item_reworded_without_articles_still_matches(self) -> None:
        parts = split_caption(f"{DESC}\n\nWATCHING:\n- person wearing red wristband: present")
        self.assertIs(parts.verdict_for(RED), True)

    def test_header_with_nothing_parseable_keeps_the_text(self) -> None:
        """Malformed is not empty: the text must still reach search and stage 2."""
        raw = f"{DESC}\n\nWATCHING:\nnothing to report today"
        parts = split_caption(raw)
        self.assertFalse(parts.has_watchlist)
        self.assertEqual(parts.description, raw.strip())

    def test_prose_containing_the_word_present_does_not_become_a_verdict(self) -> None:
        parts = split_caption("Two people are present in the room. Nobody moves.")
        self.assertFalse(parts.has_watchlist)

    def test_a_different_task_sharing_one_noun_does_not_match(self) -> None:
        parts = split_caption(f"{DESC}\n\nWATCHING:\n- {FIRE}: present")
        self.assertIsNone(parts.verdict_for("a person carrying a ladder"))


class Stage1RegressionTests(unittest.TestCase):
    """The measurement that forced the verdict path to exist.

    The WATCHING block quotes every task verbatim, absent ones included. Embedding the
    whole caption therefore ranks the two ABSENT tasks *above* the one PRESENT task,
    because their descriptions are longer and carry more distinctive words. Stage 1 is
    not merely blurred by this, it is inverted — so the block must never reach the
    embedder, and the verdict must be read directly.
    """

    def test_full_caption_ranks_absent_tasks_above_the_present_one(self) -> None:
        from services.index.embedding import HashingEmbedder
        from services.monitor.funnel import cosine

        emb = HashingEmbedder(dims=256)
        full = emb.embed_passages([FULL])[0]
        fire_v, red_v = emb.embed_query(FIRE), emb.embed_query(RED)
        self.assertGreater(
            cosine(fire_v, full),
            cosine(red_v, full),
            "the inversion this module exists to avoid has stopped reproducing; if the "
            "embedder changed, re-derive the numbers in shared/captions.py rather than "
            "deleting this test",
        )

    def test_the_verdict_path_gets_it_right_where_cosine_cannot(self) -> None:
        parts = split_caption(FULL)
        self.assertIs(parts.verdict_for(RED), True)
        self.assertIs(parts.verdict_for(FIRE), False)


if __name__ == "__main__":
    unittest.main()
