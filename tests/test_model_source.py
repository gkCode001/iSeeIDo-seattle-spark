"""The topbar model selector: ``GET/POST /api/model`` and the switch behind it.

The contract these lock down is the one ui/static/model.js renders:

* both options are described whether or not they are reachable, with the *reason* an
  unreachable one is unreachable — the page greys the row and prints that sentence;
* a failed switch leaves the surface answering with the model it already had, rather
  than with nothing;
* switching rebinds later turns only.

LM Studio is never contacted. Every probe here goes through an injected fetcher, and the
"LM Studio is running" cases are arranged by handing the builder a fake — which is the
only way to test them, since a passing CI box has no LM Studio on it.
"""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

from services.agent.agent import AskAgent
from services.agent.llm import (
    DEFAULT_SOURCE,
    LLMRequest,
    LLMResponse,
    StubBackend,
    build_source_backend,
    describe_sources,
)
from services.agent.server import AgentApp
from services.agent.settings import AgentSettings
from services.agent.ws import WebSocketHub
from services.mcp import NullClipCutter
from shared import lmstudio

from .test_agent import AgentCase, SeedTaskRegistry


def _dead_fetch(url: str, timeout: float) -> object:
    raise OSError("connection refused")


def _live_fetch(model_id: str = "gemma-4-26b-a4b", **entry: object):
    row = {
        "id": model_id,
        "type": "vlm",
        "state": "loaded",
        "loaded_context_length": 32768,
    }
    row.update(entry)

    def fetch(url: str, timeout: float) -> object:
        if url.endswith("/api/v0/models"):
            return {"data": [row]}
        raise AssertionError("native route should have answered")

    return fetch


class _RecordingBackend:
    """Stands in for a model so a switch is observable without an endpoint."""

    def __init__(self, model: str) -> None:
        self._model = model
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text="ok", model=self._model, backend=self.name)


# --------------------------------------------------------------------------------------
# describe_sources — what the selector renders before anything is chosen
# --------------------------------------------------------------------------------------


class DescribeSourcesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AgentSettings.from_config()

    def test_both_sources_are_always_offered(self) -> None:
        sources = describe_sources(self.settings, fetch=_dead_fetch)
        self.assertEqual([s.id for s in sources], [DEFAULT_SOURCE, lmstudio.BACKEND])

    def test_unreachable_lmstudio_carries_the_reason(self) -> None:
        lm = describe_sources(self.settings, fetch=_dead_fetch)[1]
        self.assertFalse(lm.available)
        self.assertIn("connection refused", lm.detail)

    def test_reachable_lmstudio_reports_the_loaded_model(self) -> None:
        lm = describe_sources(self.settings, fetch=_live_fetch())[1]
        self.assertTrue(lm.available)
        self.assertEqual(lm.model, "gemma-4-26b-a4b")
        # Both servers up is the invariant-1 case the page must say out loud.
        self.assertIn("invariant 1", lm.note)

    def test_default_is_labelled_with_the_model_not_the_filename(self) -> None:
        default = describe_sources(self.settings, fetch=_dead_fetch)[0]
        self.assertEqual(default.label, "gemma-4-E2B-it")
        self.assertTrue(default.available)

    def test_stub_backend_says_no_model_is_being_called(self) -> None:
        stub = dataclasses.replace(self.settings, backend="stub")
        default = describe_sources(stub, fetch=_dead_fetch)[0]
        self.assertIn("stub", default.note)


class BuildSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = dataclasses.replace(AgentSettings.from_config(), backend="stub")

    def test_unknown_source_is_a_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_source_backend("gpt-9", self.settings)

    def test_default_source_builds_the_configured_backend(self) -> None:
        backend = build_source_backend(DEFAULT_SOURCE, self.settings)
        self.assertIsInstance(backend, StubBackend)

    def test_lmstudio_source_resolves_the_loaded_model(self) -> None:
        backend = build_source_backend(
            lmstudio.BACKEND, self.settings, fetch=_live_fetch("some-loaded-model")
        )
        self.assertEqual(backend.model, "some-loaded-model")
        self.assertEqual(backend.name, "lmstudio")

    def test_lmstudio_source_carries_the_reasoning_switch(self) -> None:
        """`--reasoning off` is a launch flag we cannot pass, so it must ride in the body."""
        backend = build_source_backend(lmstudio.BACKEND, self.settings, fetch=_live_fetch())
        extra = backend._extra  # noqa: SLF001 - the payload is the point of the test
        self.assertTrue(extra, "reasoning_off_payload did not reach the backend")
        self.assertEqual(extra.get("reasoning_format"), "none")

    def test_lmstudio_source_points_at_lmstudios_endpoint(self) -> None:
        backend = build_source_backend(lmstudio.BACKEND, self.settings, fetch=_live_fetch())
        self.assertIn("1234", backend.endpoint)


# --------------------------------------------------------------------------------------
# The routes
# --------------------------------------------------------------------------------------


class ModelRouteTests(AgentCase):
    def setUp(self) -> None:
        super().setUp()
        self.app = AgentApp(
            agent=self.agent,
            index=self.index,
            actions=self.actions,
            jobs=self.jobs,
            chat_log=self.chat_log,
            tasks=SeedTaskRegistry(actions=self.actions),
            hub=WebSocketHub(),
            settings=self.settings,
            clip_cutter=NullClipCutter(),
        )

    def test_get_reports_the_active_model(self) -> None:
        status, payload = self.app.get_model()
        self.assertEqual(status, 200)
        self.assertEqual(payload["active"], DEFAULT_SOURCE)
        self.assertEqual(payload["model"], self.agent.backend.model)
        self.assertEqual(len(payload["sources"]), 2)

    def test_get_states_the_scope_of_the_switch(self) -> None:
        """The captioner is another process; the page must not imply otherwise."""
        _, payload = self.app.get_model()
        self.assertIn("captioner", payload["scope"])

    def test_switch_to_an_unknown_source_is_a_400(self) -> None:
        status, payload = self.app.post_model({"source": "gpt-9"})
        self.assertEqual(status, 400)
        self.assertIn("gpt-9", payload["detail"])

    def test_switch_with_no_source_is_a_400(self) -> None:
        status, _ = self.app.post_model({})
        self.assertEqual(status, 400)

    def test_switch_to_a_dead_lmstudio_is_a_409_and_changes_nothing(self) -> None:
        # The route builds its own backend, so there is no fetcher to inject — patch the
        # module default instead. Without this the test reads the DEVELOPER'S box: it
        # passed while LM Studio was closed and started failing the moment it was opened,
        # which is the test telling you about the environment rather than the code.
        before = self.agent.backend
        with mock.patch.object(lmstudio, "urllib_fetch", _dead_fetch):
            status, payload = self.app.post_model({"source": lmstudio.BACKEND})
        self.assertEqual(status, 409)
        self.assertIn("LM Studio", payload["detail"])
        self.assertIs(self.agent.backend, before, "a failed switch dropped the model")
        self.assertEqual(self.app.model_source, DEFAULT_SOURCE)

    def test_switching_to_the_active_source_is_a_no_op(self) -> None:
        before = self.agent.backend
        status, payload = self.app.post_model({"source": DEFAULT_SOURCE})
        self.assertEqual(status, 200)
        self.assertIs(self.agent.backend, before)
        self.assertNotIn("switched_from", payload)

    def test_a_successful_switch_rebinds_later_turns(self) -> None:
        replacement = _RecordingBackend("the-other-model")
        previous = self.agent.backend.model

        self.agent.use_backend(replacement)
        self.app.model_source = lmstudio.BACKEND

        status, payload = self.app.get_model()
        self.assertEqual(status, 200)
        self.assertEqual(payload["active"], lmstudio.BACKEND)
        self.assertEqual(payload["model"], "the-other-model")
        self.assertNotEqual(payload["model"], previous)

    def test_warning_when_both_servers_are_up(self) -> None:
        """Two model processes, one 128 GB pool. No toggle can fix it; saying so can."""
        app_sources = describe_sources(self.settings, fetch=_live_fetch())
        self.assertTrue(app_sources[1].available)
        # get_model probes for real, so assert on the branch it feeds rather than
        # reaching a live LM Studio from a test.
        both_up = app_sources[1].available and self.app.model_source == DEFAULT_SOURCE
        self.assertTrue(both_up)


class UseBackendTests(AgentCase):
    def test_later_turns_go_to_the_new_backend(self) -> None:
        replacement = _RecordingBackend("the-other-model")
        self.agent.ask("what is happening right now?")
        self.assertEqual(replacement.calls, 0, "the new backend answered a turn before it was bound")

        self.agent.use_backend(replacement)
        self.agent.ask("and now?")
        self.assertGreater(replacement.calls, 0, "the switch did not take effect")
        self.assertEqual(self.agent.backend.model, "the-other-model")

    def test_rebinding_keeps_the_chat_log(self) -> None:
        self.agent.ask("what is happening right now?")
        before = len(self.chat_log.read().turns)
        self.agent.use_backend(StubBackend(self.settings))
        self.agent.ask("and now?")
        self.assertEqual(len(self.chat_log.read().turns), before + 1)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------------------
# Conversational context — prior turns folded into both prompts
# --------------------------------------------------------------------------------------


class _CapturingBackend:
    """Records every prompt it is handed, so the tests can assert on what the model sees."""

    def __init__(self, reply: str = "One person is standing near the door.") -> None:
        self.prompts: list[str] = []
        self._reply = reply

    @property
    def name(self) -> str:
        return "capture"

    @property
    def model(self) -> str:
        return "capture"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.prompts.append(request.user)
        return LLMResponse(text=self._reply, model=self.model, backend=self.name)

    def coverage(self, question, context) -> float:  # StubBackend duck-type, unused here
        return 1.0


class HistoryContextTests(AgentCase):
    """A follow-up must be able to resolve what it refers to (agent.history.context_turns)."""

    def setUp(self) -> None:
        super().setUp()
        self.capture = _CapturingBackend()
        self.agent.use_backend(self.capture)

    def test_the_first_turn_carries_no_conversation(self) -> None:
        self.agent.ask("is anyone at the door?")
        for prompt in self.capture.prompts:
            self.assertNotIn("CONVERSATION SO FAR", prompt)

    def test_a_follow_up_sees_the_previous_exchange(self) -> None:
        self.agent.ask("is anyone at the door?")
        self.capture.prompts.clear()
        self.agent.ask("what were they wearing?")

        self.assertTrue(self.capture.prompts, "the follow-up made no model call")
        for prompt in self.capture.prompts:
            self.assertIn("CONVERSATION SO FAR", prompt)
            self.assertIn("is anyone at the door?", prompt)
            self.assertIn("One person is standing near the door.", prompt)

    def test_both_the_gate_and_the_answer_get_it(self) -> None:
        """The gate judges the follow-up too; blind to history it would escalate every one."""
        self.agent.ask("is anyone at the door?")
        self.capture.prompts.clear()
        self.agent.ask("what were they wearing?")
        self.assertEqual(len(self.capture.prompts), 2, "expected a gate call and an answer call")

    def test_history_is_labelled_as_not_evidence(self) -> None:
        """Otherwise the gate calls a follow-up grounded because IT said something earlier
        — the confident-answer-from-absent-evidence failure the gate exists to catch."""
        self.agent.ask("is anyone at the door?")
        self.capture.prompts.clear()
        self.agent.ask("what were they wearing?")
        for prompt in self.capture.prompts:
            self.assertIn("NOT evidence", prompt)
            self.assertIn("RETRIEVED CONTEXT", prompt)

    def test_only_the_configured_number_of_turns_is_carried(self) -> None:
        for i in range(6):
            self.agent.ask(f"question number {i}?")
        self.capture.prompts.clear()
        self.agent.ask("and finally?")

        prompt = self.capture.prompts[0]
        carried = [i for i in range(6) if f"question number {i}?" in prompt]
        self.assertEqual(len(carried), self.settings.history_context_turns)
        # The most recent ones, not the oldest.
        self.assertEqual(carried, sorted(carried)[-self.settings.history_context_turns :])
        self.assertNotIn("question number 0?", prompt)

    def test_long_answers_are_clipped(self) -> None:
        """Prior answers are prefill on every later turn; a 512-token one would crowd out
        the captions, which are the actual evidence."""
        self.agent.use_backend(_CapturingBackend("x" * 5000))
        self.agent.ask("first?")
        capture = _CapturingBackend()
        self.agent.use_backend(capture)
        self.agent.ask("second?")
        self.assertLess(
            len(capture.prompts[0]),
            2000,
            "an unclipped prior answer reached the prompt",
        )

    def test_zero_turns_restores_the_stateless_behaviour(self) -> None:
        agent = AskAgent(
            self.capture,
            self.tools,
            self.chat_log,
            dataclasses.replace(self.settings, history_context_turns=0),
        )
        agent.ask("is anyone at the door?")
        self.capture.prompts.clear()
        agent.ask("what were they wearing?")
        for prompt in self.capture.prompts:
            self.assertNotIn("CONVERSATION SO FAR", prompt)
