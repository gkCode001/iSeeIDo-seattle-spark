"""Tests for ``shared/vlm_client.py``.

Every test injects a fake transport. CLAUDE.md is explicit that the real endpoint is
never called from a test — it contends with ingest, and a test suite that needs a GPU
up is a test suite nobody runs.

Stdlib ``unittest``:  ``python3 -m unittest discover -s tests -t . -v``
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shared import config
from shared.vlm_client import (
    Profile,
    ProfileViolation,
    RequestsTransport,
    VLMChunk,
    VLMClient,
    VLMResponseError,
    VLMTransportError,
    encode_frame,
)

# The model is null in settings.yaml (UNRESOLVED, SPEC §10 D1). Tests pass one
# explicitly — they are testing the client, not settling the decision.
TEST_MODEL = "test/cosmos-reason-stub"
PROMPT = "Describe what happens in these frames."


def make_chunk(chunk_id: str = "cam01_20260814T211107_211112", frames: int = 5) -> VLMChunk:
    return VLMChunk(
        chunk_id=chunk_id, frames=[f"data:image/jpeg;base64,f{i}" for i in range(frames)]
    )


def completion(
    content: str = "A white panel van reverses toward the loading door.",
    *,
    reasoning: str | None = None,
    prompt_tokens: int = 1234,
    completion_tokens: int = 42,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "cmpl-1",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class FakeTransport:
    """Records every POST and replays canned bodies. The seam that keeps tests offline."""

    def __init__(self, *bodies: dict[str, Any] | Exception) -> None:
        self.bodies = list(bodies) or [completion()]
        self.calls: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.timeouts: list[float | None] = []

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout: float | None
    ) -> dict[str, Any]:
        self.calls.append(dict(payload))
        self.urls.append(url)
        self.timeouts.append(timeout)
        body = self.bodies[min(len(self.calls) - 1, len(self.bodies) - 1)]
        if isinstance(body, Exception):
            raise body
        return body

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


class FakeClock:
    """Monotonic-ish clock we advance by hand, so wall_time_ms is an assertion."""

    def __init__(self, step: float = 2.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


_QUIET = logging.getLogger("test.vlm.quiet")
_QUIET.addHandler(logging.NullHandler())
_QUIET.propagate = False


def client(transport: FakeTransport, **kwargs: Any) -> VLMClient:
    kwargs.setdefault("model", TEST_MODEL)
    kwargs.setdefault("logger", _QUIET)  # keep call logs out of the test output
    return VLMClient(transport, **kwargs)


# ======================================================================================
# Profiles — CLAUDE.md invariant 6
# ======================================================================================


class TestProfiles(unittest.TestCase):
    def test_live_profile_matches_the_invariant(self) -> None:
        """Invariant 6's two halves, and only one of them is a fixed number.

        ``enable_reasoning`` is absolute: this is a thinking model, and left on it spends
        the whole budget inside ``reasoning_content`` and returns an empty caption.

        ``max_tokens`` is a measured setting, not a constant. It was 80 while the caption
        was one line; it is 320 now that the caption also answers the standing-task
        checklist (services/ingest/watchlist.py). Asserted against the config rather than
        a literal so that re-tuning it is a settings edit, not a test edit — what must not
        drift is the client honouring the profile.
        """
        c = client(FakeTransport())
        self.assertFalse(c.live.enable_reasoning)
        self.assertEqual(c.live.max_tokens, int(config.get("vlm.profiles.live.max_tokens")))
        self.assertGreater(c.live.max_tokens, 0)

    def test_deep_profile_matches_the_spec(self) -> None:
        c = client(FakeTransport())
        self.assertTrue(c.deep.enable_reasoning)
        self.assertEqual(c.deep.max_tokens, 600)
        self.assertEqual(c.deep.sample_fps, 4.0)
        self.assertTrue(c.deep.native_resolution)

    def test_caption_sends_the_live_profile(self) -> None:
        t = FakeTransport()
        client(t).caption([make_chunk()], prompt=PROMPT)
        self.assertEqual(t.last["max_tokens"], int(config.get("vlm.profiles.live.max_tokens")))
        self.assertEqual(t.last["temperature"], 0.0)
        self.assertIs(t.last["chat_template_kwargs"]["enable_reasoning"], False)

    def test_analyze_sends_the_deep_profile(self) -> None:
        t = FakeTransport()
        client(t).analyze([make_chunk()], prompt="Was the rear door open?")
        self.assertEqual(t.last["max_tokens"], 600)
        self.assertIs(t.last["chat_template_kwargs"]["enable_reasoning"], True)

    def test_caller_cannot_raise_max_tokens(self) -> None:
        t = FakeTransport()
        c = client(t)
        with self.assertRaises(ProfileViolation):
            c.caption([make_chunk()], prompt=PROMPT, max_tokens=400)
        with self.assertRaises(ProfileViolation):
            c.analyze([make_chunk()], prompt=PROMPT, max_tokens=2000)
        self.assertEqual(t.calls, [], "a rejected budget must not reach the endpoint")

    def test_caller_may_lower_max_tokens(self) -> None:
        t = FakeTransport()
        client(t).caption([make_chunk()], prompt=PROMPT, max_tokens=40)
        self.assertEqual(t.last["max_tokens"], 40)

    def test_zero_or_negative_max_tokens_is_rejected(self) -> None:
        with self.assertRaises(ProfileViolation):
            client(FakeTransport()).caption([make_chunk()], prompt=PROMPT, max_tokens=0)

    def test_there_is_no_reasoning_parameter_to_pass(self) -> None:
        # Reasoning is a property of the profile, not an argument. If this ever becomes
        # a kwarg, the live path can be turned into a 60 s/chunk path by a caller.
        with self.assertRaises(TypeError):
            client(FakeTransport()).caption(  # type: ignore[call-arg]
                [make_chunk()], prompt=PROMPT, enable_reasoning=True
            )


# ======================================================================================
# The list signature — CLAUDE.md invariant 9
# ======================================================================================


class TestBatchDimension(unittest.TestCase):
    def test_caption_takes_a_list_and_returns_one_result_per_chunk(self) -> None:
        t = FakeTransport(completion("first"), completion("second"), completion("third"))
        results = client(t).caption(
            [make_chunk("a"), make_chunk("b"), make_chunk("c")], prompt=PROMPT
        )
        self.assertEqual([r.chunk_id for r in results], ["a", "b", "c"])
        self.assertEqual([r.text for r in results], ["first", "second", "third"])
        self.assertEqual(len(t.calls), 3, "one camera, one request in flight: serviced serially")

    def test_the_single_chunk_case_we_actually_use(self) -> None:
        results = client(FakeTransport()).caption([make_chunk()], prompt=PROMPT)
        self.assertEqual(len(results), 1)

    def test_empty_list_makes_no_calls(self) -> None:
        t = FakeTransport()
        self.assertEqual(client(t).caption([], prompt=PROMPT), [])
        self.assertEqual(t.calls, [])

    def test_a_chunk_with_no_frames_is_a_bug_not_an_empty_caption(self) -> None:
        t = FakeTransport()
        with self.assertRaises(ValueError):
            client(t).caption([VLMChunk(chunk_id="x", frames=[])], prompt=PROMPT)
        self.assertEqual(t.calls, [])

    def test_empty_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            client(FakeTransport()).caption([make_chunk()], prompt="   ")


# ======================================================================================
# Request shape
# ======================================================================================


class TestRequestShape(unittest.TestCase):
    def test_frames_are_sent_as_image_url_parts_with_the_prompt_last(self) -> None:
        t = FakeTransport()
        client(t).caption([make_chunk(frames=5)], prompt=PROMPT)
        content = t.last["messages"][0]["content"]
        self.assertEqual([p["type"] for p in content], ["image_url"] * 5 + ["text"])
        self.assertEqual(content[0]["image_url"]["url"], "data:image/jpeg;base64,f0")
        self.assertEqual(content[-1]["text"], PROMPT)

    def test_extra_text_is_appended_to_the_prompt(self) -> None:
        t = FakeTransport()
        chunk = VLMChunk(chunk_id="x", frames=["data:image/jpeg;base64,f0"], extra_text="21:11 UTC")
        client(t).caption([chunk], prompt=PROMPT)
        self.assertIn("21:11 UTC", t.last["messages"][0]["content"][-1]["text"])

    def test_endpoint_comes_from_config_and_targets_chat_completions(self) -> None:
        t = FakeTransport()
        client(t).caption([make_chunk()], prompt=PROMPT)
        self.assertEqual(t.urls[-1], "http://localhost:8000/v1/chat/completions")

    def test_encode_frame_builds_a_data_uri(self) -> None:
        self.assertEqual(encode_frame(b"\x00\x01"), "data:image/jpeg;base64,AAE=")


# ======================================================================================
# The model name — SPEC §10 D1
# ======================================================================================


class TestModelResolution(unittest.TestCase):
    def test_unresolved_model_fails_with_a_readable_message(self) -> None:
        """SPEC §10 D1 is resolved now, so this pins the GUARD, not the config value.

        A null vlm.model must still fail with a sentence naming the decision rather than
        404-ing later against a server that was never told which model to serve. Testing
        the mechanism means this keeps working whichever way D1 is revisited.
        """
        root = config.load()
        previous = root["vlm"]["model"]
        root["vlm"]["model"] = None
        try:
            with self.assertRaises(config.ConfigError) as ctx:
                VLMClient(FakeTransport())
            self.assertIn("vlm.model", str(ctx.exception))
            self.assertIn("SPEC", str(ctx.exception))
        finally:
            root["vlm"]["model"] = previous

    def test_the_resolved_model_is_used_when_no_override_is_given(self) -> None:
        """D1 landed: a client built with no explicit model picks up settings.yaml."""
        t = FakeTransport()
        VLMClient(t).caption([make_chunk()], prompt=PROMPT)
        self.assertEqual(t.last["model"], str(config.get("vlm.model")))

    def test_explicit_model_is_sent_on_every_request(self) -> None:
        t = FakeTransport()
        c = client(t, model="cosmos-reason-2b")
        c.caption([make_chunk()], prompt=PROMPT)
        self.assertEqual(t.last["model"], "cosmos-reason-2b")
        self.assertEqual(c.model, "cosmos-reason-2b")


# ======================================================================================
# Response parsing
# ======================================================================================


class TestResponseParsing(unittest.TestCase):
    def test_caption_text_and_usage(self) -> None:
        t = FakeTransport(completion("  a van reverses  ", prompt_tokens=900, completion_tokens=31))
        result = client(t).caption([make_chunk()], prompt=PROMPT)[0]
        self.assertEqual(result.text, "a van reverses")
        self.assertEqual(result.prompt_tokens, 900)
        self.assertEqual(result.completion_tokens, 31)
        self.assertEqual(result.profile, Profile.LIVE.value)

    def test_deep_reasoning_from_reasoning_content(self) -> None:
        t = FakeTransport(completion("Yes.", reasoning="The doors are visible at 21:11:19."))
        result = client(t).analyze([make_chunk()], prompt=PROMPT)[0]
        self.assertEqual(result.text, "Yes.")
        self.assertEqual(result.reasoning, "The doors are visible at 21:11:19.")

    def test_deep_reasoning_from_inline_think_tags(self) -> None:
        # vLLM inlines the trace when no reasoning parser is configured. Which we get
        # depends on the D1 variant, so both are handled.
        t = FakeTransport(completion("<think>frame 3 shows the door</think>Yes, it was open."))
        result = client(t).analyze([make_chunk()], prompt=PROMPT)[0]
        self.assertEqual(result.text, "Yes, it was open.")
        self.assertEqual(result.reasoning, "frame 3 shows the door")

    def test_live_results_never_carry_reasoning(self) -> None:
        t = FakeTransport(completion("a van", reasoning="this should not be here"))
        self.assertEqual(client(t).caption([make_chunk()], prompt=PROMPT)[0].reasoning, "")

    def test_malformed_responses_raise(self) -> None:
        for body in ({}, {"choices": []}, {"choices": [{"index": 0}]}):
            with self.subTest(body=body):
                with self.assertRaises(VLMResponseError):
                    client(FakeTransport(body)).caption([make_chunk()], prompt=PROMPT)

    def test_transport_errors_propagate_as_vlm_errors(self) -> None:
        t = FakeTransport(VLMTransportError("connection refused"))
        with self.assertRaises(VLMTransportError):
            client(t).caption([make_chunk()], prompt=PROMPT)

    def test_wall_time_is_measured_with_the_injected_clock(self) -> None:
        t = FakeTransport()
        result = client(t, clock=FakeClock(step=2.0)).caption([make_chunk()], prompt=PROMPT)[0]
        self.assertEqual(result.wall_time_ms, 2000.0)


# ======================================================================================
# Structured logging — "we cannot tune what we cannot see"
# ======================================================================================


class TestLogging(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(f"test.vlm.{self.id()}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def _capture(self, fn: Any) -> list[logging.LogRecord]:
        with self.assertLogs(self.logger, level=logging.DEBUG) as captured:
            try:
                fn()
            except Exception:  # noqa: BLE001 - failure paths must log too
                pass
        return captured.records

    def test_every_configured_field_is_logged_on_success(self) -> None:
        t = FakeTransport(completion(prompt_tokens=900, completion_tokens=31))
        c = client(t, logger=self.logger, clock=FakeClock(step=2.0))
        records = self._capture(lambda: c.caption([make_chunk()], prompt=PROMPT))
        self.assertEqual(len(records), 1)
        entry = json.loads(records[0].getMessage())
        self.assertEqual(
            set(entry), set(config.get("logging.vlm_calls")),
            "the logged fields are exactly the ones settings.yaml asks for",
        )
        self.assertEqual(entry["model"], TEST_MODEL)
        self.assertEqual(entry["profile"], "live")
        self.assertEqual(entry["prompt_tokens"], 900)
        self.assertEqual(entry["completion_tokens"], 31)
        self.assertEqual(entry["wall_time_ms"], 2000.0)

    def test_deep_calls_log_their_own_profile(self) -> None:
        c = client(FakeTransport(), logger=self.logger)
        records = self._capture(lambda: c.analyze([make_chunk()], prompt=PROMPT))
        self.assertEqual(json.loads(records[0].getMessage())["profile"], "deep")

    def test_failures_are_logged_too(self) -> None:
        t = FakeTransport(VLMTransportError("connection refused"))
        c = client(t, logger=self.logger)
        records = self._capture(lambda: c.caption([make_chunk()], prompt=PROMPT))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].levelno, logging.WARNING)
        self.assertIn("connection refused", json.loads(records[0].getMessage())["error"])

    def test_one_log_line_per_chunk_in_a_batch(self) -> None:
        c = client(FakeTransport(), logger=self.logger)
        chunks = [make_chunk("a"), make_chunk("b")]
        records = self._capture(lambda: c.caption(chunks, prompt=PROMPT))
        self.assertEqual(len(records), 2)


class TestLoggingConfigValidation(unittest.TestCase):
    """A log field settings.yaml asks for but the client cannot produce is a config bug."""

    def setUp(self) -> None:
        self.previous = os.environ.get("SPARK_SETTINGS")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self.previous is None:
            os.environ.pop("SPARK_SETTINGS", None)
        else:
            os.environ["SPARK_SETTINGS"] = self.previous
        config.load.cache_clear()

    def _use(self, mutate: Any) -> None:
        """Point ``SPARK_SETTINGS`` at a copy of the real file with one edit applied."""
        import yaml

        data = yaml.safe_load(
            (config.REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")
        )
        mutate(data)
        path = Path(self.tmp.name) / "settings.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        os.environ["SPARK_SETTINGS"] = str(path)
        config.load.cache_clear()

    def test_unknown_log_field_fails_loudly_at_construction(self) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data["logging"]["vlm_calls"] = ["model", "gpu_temperature"]

        self._use(mutate)
        with self.assertRaises(config.ConfigError) as ctx:
            VLMClient(FakeTransport(), model=TEST_MODEL)
        self.assertIn("gpu_temperature", str(ctx.exception))

    def test_endpoint_model_and_budget_all_come_from_the_active_settings_file(self) -> None:
        def mutate(data: dict[str, Any]) -> None:
            data["vlm"]["endpoint"] = "http://elsewhere:9000/v1"
            data["vlm"]["model"] = "pinned-model"
            data["vlm"]["profiles"]["live"]["max_tokens"] = 24

        self._use(mutate)
        t = FakeTransport()
        c = VLMClient(t)  # model resolves from config now, no override needed
        self.assertEqual(c.model, "pinned-model")
        self.assertEqual(c.live.max_tokens, 24)
        c.caption([make_chunk()], prompt=PROMPT)
        self.assertEqual(t.urls[-1], "http://elsewhere:9000/v1/chat/completions")
        self.assertEqual(t.last["max_tokens"], 24)


# ======================================================================================
# Default transport — still never touches a socket here
# ======================================================================================


class TestRequestsTransport(unittest.TestCase):
    """Exercised against a fake session object. No sockets, no endpoint, no GPU."""

    class _Response:
        def __init__(self, status: int, body: Any, text: str = "") -> None:
            self.status_code = status
            self._body = body
            self.text = text

        def json(self) -> Any:
            if isinstance(self._body, Exception):
                raise self._body
            return self._body

    class _Session:
        def __init__(self, response: Any) -> None:
            self.response = response
            self.kwargs: dict[str, Any] = {}

        def post(self, url: str, **kwargs: Any) -> Any:
            self.kwargs = {"url": url, **kwargs}
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    def test_success(self) -> None:
        session = self._Session(self._Response(200, {"ok": True}))
        body = RequestsTransport(session).post("http://x/v1", {"a": 1}, timeout=3.0)
        self.assertEqual(body, {"ok": True})
        self.assertEqual(session.kwargs["json"], {"a": 1})
        self.assertEqual(session.kwargs["timeout"], 3.0)

    def test_http_error_becomes_a_transport_error(self) -> None:
        session = self._Session(self._Response(503, None, text="no model loaded"))
        with self.assertRaises(VLMTransportError) as ctx:
            RequestsTransport(session).post("http://x/v1", {}, timeout=None)
        self.assertIn("503", str(ctx.exception))

    def test_connection_failure_becomes_a_transport_error(self) -> None:
        session = self._Session(OSError("connection refused"))
        with self.assertRaises(VLMTransportError):
            RequestsTransport(session).post("http://x/v1", {}, timeout=None)

    def test_non_json_body_becomes_a_response_error(self) -> None:
        session = self._Session(self._Response(200, ValueError("not json")))
        with self.assertRaises(VLMResponseError):
            RequestsTransport(session).post("http://x/v1", {}, timeout=None)


if __name__ == "__main__":
    unittest.main()
