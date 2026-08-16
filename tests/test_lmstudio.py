"""shared/lmstudio.py + the model-source selector behind ``/api/model``.

Every test injects a fetcher. CLAUDE.md forbids tests touching a real endpoint, and this
module's whole job is asking a server what it is serving — mocking it is the only way to
exercise "nothing loaded", "loaded too small" and "two loaded at once" at all, since each
needs LM Studio to be in a state you cannot arrange from pytest.
"""

from __future__ import annotations

import unittest

from shared import lmstudio


def _settings(**overrides: object) -> lmstudio.LMStudioSettings:
    base = {
        "endpoint": "http://localhost:1234/v1",
        "model": None,
        "require_vision": False,
        "min_context_tokens": 32768,
        "resolve_timeout_seconds": 1.0,
        "reasoning_off_payload": {},
    }
    base.update(overrides)
    return lmstudio.LMStudioSettings(**base)  # type: ignore[arg-type]


def _native(*entries: dict) -> lmstudio.Fetch:
    """A fetcher whose native route answers and whose OpenAI route is never reached."""

    def fetch(url: str, timeout: float) -> object:
        if url.endswith("/api/v0/models"):
            return {"data": list(entries)}
        raise AssertionError(f"native route should have answered; got a call to {url}")

    return fetch


def _entry(model_id: str, **overrides: object) -> dict:
    entry = {
        "id": model_id,
        "type": "vlm",
        "state": "loaded",
        "loaded_context_length": 32768,
    }
    entry.update(overrides)
    return entry


class ResolveTests(unittest.TestCase):
    def test_takes_the_loaded_model(self) -> None:
        model = lmstudio.resolve(_settings(), fetch=_native(_entry("gemma-4-26b-a4b")))
        self.assertEqual(model.id, "gemma-4-26b-a4b")
        self.assertEqual(model.context_length, 32768)
        self.assertTrue(model.is_vision)

    def test_ignores_models_that_are_not_loaded(self) -> None:
        fetch = _native(
            _entry("downloaded-but-cold", state="not-loaded"),
            _entry("actually-loaded"),
        )
        self.assertEqual(lmstudio.resolve(_settings(), fetch=fetch).id, "actually-loaded")

    def test_nothing_loaded_names_the_fix(self) -> None:
        fetch = _native(_entry("cold", state="not-loaded"))
        with self.assertRaises(lmstudio.NoModelLoaded) as caught:
            lmstudio.resolve(_settings(), fetch=fetch)
        self.assertIn("no model loaded", str(caught.exception))

    def test_empty_list_is_not_a_crash(self) -> None:
        with self.assertRaises(lmstudio.NoModelLoaded):
            lmstudio.resolve(_settings(), fetch=_native())

    def test_two_chat_models_loaded_refuses_to_guess(self) -> None:
        """Invariant 1 territory — and picking one silently would hide it."""
        fetch = _native(_entry("model-a"), _entry("model-b"))
        with self.assertRaises(lmstudio.NoModelLoaded) as caught:
            lmstudio.resolve(_settings(), fetch=fetch)
        self.assertIn("lmstudio.model", str(caught.exception))

    def test_an_embedding_model_alongside_the_chat_one_is_fine(self) -> None:
        fetch = _native(_entry("text-embed", type="embeddings"), _entry("the-chat-one"))
        self.assertEqual(lmstudio.resolve(_settings(), fetch=fetch).id, "the-chat-one")

    def test_pinned_model_must_be_listed(self) -> None:
        fetch = _native(_entry("something-else"))
        with self.assertRaises(lmstudio.NoModelLoaded) as caught:
            lmstudio.resolve(_settings(model="the-one-i-asked-for"), fetch=fetch)
        self.assertIn("something-else", str(caught.exception))

    def test_pinned_model_wins_over_the_loaded_one(self) -> None:
        fetch = _native(_entry("pinned", state="not-loaded"), _entry("loaded"))
        self.assertEqual(lmstudio.resolve(_settings(model="pinned"), fetch=fetch).id, "pinned")


class UsabilityTests(unittest.TestCase):
    def test_short_context_is_refused_with_the_reason(self) -> None:
        """A 4k context does not fail here — it fails as a deep job that never finishes."""
        fetch = _native(_entry("tiny-ctx", loaded_context_length=4096))
        with self.assertRaises(lmstudio.UnusableModel) as caught:
            lmstudio.resolve(_settings(), fetch=fetch)
        message = str(caught.exception)
        self.assertIn("4096", message)
        self.assertIn("32768", message)

    def test_context_check_can_be_switched_off(self) -> None:
        fetch = _native(_entry("tiny-ctx", loaded_context_length=4096))
        self.assertEqual(
            lmstudio.resolve(_settings(min_context_tokens=None), fetch=fetch).id, "tiny-ctx"
        )

    def test_text_only_model_refused_when_vision_is_required(self) -> None:
        fetch = _native(_entry("text-only", type="llm"))
        with self.assertRaises(lmstudio.UnusableModel) as caught:
            lmstudio.resolve(_settings(require_vision=True), fetch=fetch)
        self.assertIn("vision", str(caught.exception))

    def test_text_only_model_allowed_for_the_ask_surface(self) -> None:
        fetch = _native(_entry("text-only", type="llm"))
        self.assertEqual(lmstudio.resolve(_settings(), fetch=fetch).id, "text-only")

    def test_an_embedding_model_alone_is_refused(self) -> None:
        fetch = _native(_entry("text-embed", type="embeddings"))
        with self.assertRaises(lmstudio.UnusableModel):
            lmstudio.resolve(_settings(), fetch=fetch)


class FallbackRouteTests(unittest.TestCase):
    """The OpenAI route reports an id and nothing else — so the extra checks must not
    fire on data it never supplied."""

    @staticmethod
    def _openai_only(*ids: str) -> lmstudio.Fetch:
        def fetch(url: str, timeout: float) -> object:
            if url.endswith("/api/v0/models"):
                raise OSError("404 — older LM Studio, no native API")
            return {"data": [{"id": model_id, "object": "model"} for model_id in ids]}

        return fetch

    def test_falls_back_and_still_finds_the_model(self) -> None:
        model = lmstudio.resolve(_settings(), fetch=self._openai_only("some-model"))
        self.assertEqual(model.id, "some-model")
        self.assertIsNone(model.context_length)
        self.assertFalse(model.kind_known)

    def test_unknown_context_does_not_trip_the_minimum(self) -> None:
        model = lmstudio.resolve(
            _settings(min_context_tokens=32768), fetch=self._openai_only("some-model")
        )
        self.assertEqual(model.id, "some-model")

    def test_unknown_type_does_not_trip_require_vision(self) -> None:
        model = lmstudio.resolve(
            _settings(require_vision=True), fetch=self._openai_only("some-model")
        )
        self.assertEqual(model.id, "some-model")

    def test_both_routes_dead_names_both(self) -> None:
        def fetch(url: str, timeout: float) -> object:
            raise OSError("connection refused")

        with self.assertRaises(lmstudio.LMStudioUnreachable) as caught:
            lmstudio.resolve(_settings(), fetch=fetch)
        self.assertIn("/api/v0/models", str(caught.exception))
        self.assertIn("/v1/models", str(caught.exception))


class ProbeTests(unittest.TestCase):
    def test_probe_reports_success(self) -> None:
        result = lmstudio.probe(_settings(), fetch=_native(_entry("loaded-one")))
        self.assertTrue(result.available)
        self.assertEqual(result.model, "loaded-one")

    def test_probe_turns_failure_into_a_message(self) -> None:
        def fetch(url: str, timeout: float) -> object:
            raise OSError("connection refused")

        result = lmstudio.probe(_settings(), fetch=fetch)
        self.assertFalse(result.available)
        self.assertIn("connection refused", result.detail)
        self.assertIsNone(result.model)


class MergePayloadTests(unittest.TestCase):
    def test_merges_nested_chat_template_kwargs(self) -> None:
        payload = {"model": "m", "chat_template_kwargs": {"enable_reasoning": False}}
        lmstudio.merge_payload(payload, {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_reasoning": False, "enable_thinking": False},
        )

    def test_top_level_keys_are_set(self) -> None:
        payload: dict = {"model": "m"}
        lmstudio.merge_payload(payload, {"reasoning_format": "none"})
        self.assertEqual(payload["reasoning_format"], "none")

    def test_scalar_overwrites_a_dict(self) -> None:
        payload = {"chat_template_kwargs": {"a": 1}}
        lmstudio.merge_payload(payload, {"chat_template_kwargs": None})
        self.assertIsNone(payload["chat_template_kwargs"])


class BaseUrlTests(unittest.TestCase):
    def test_strips_the_v1_suffix(self) -> None:
        self.assertEqual(_settings().base_url, "http://localhost:1234")

    def test_leaves_a_bare_host_alone(self) -> None:
        self.assertEqual(
            _settings(endpoint="http://localhost:1234").base_url, "http://localhost:1234"
        )


if __name__ == "__main__":
    unittest.main()
