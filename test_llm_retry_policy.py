import unittest
from unittest.mock import patch

from app.llm import (
    HybridLLM,
    LLMRateLimitExceeded,
    get_llm_call_count,
    reset_llm_call_counter,
    restore_llm_call_counter,
)


class _FakeCompletions:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        raise self.exc


class _FakeChat:
    def __init__(self, exc):
        self.completions = _FakeCompletions(exc)


class _FakeClient:
    def __init__(self, exc):
        self.chat = _FakeChat(exc)


class _StatusError(Exception):
    def __init__(self, status_code, code=""):
        super().__init__(code or f"status {status_code}")
        self.status_code = status_code
        self.code = code


class LLMRetryPolicyTests(unittest.TestCase):
    def test_429_is_not_retried_by_app_loop(self):
        token = reset_llm_call_counter(limit=3)
        fake_client = _FakeClient(_StatusError(429, "rate_limit_exceeded"))
        try:
            with patch("app.llm.openai_client", fake_client):
                with self.assertRaises(LLMRateLimitExceeded):
                    HybridLLM()._invoke_with_client(
                        client=fake_client,
                        model="test-model",
                        user_prompt="hello",
                        system_prompt="system",
                        session_id="test-session",
                        call_type="generation",
                    )

            self.assertEqual(fake_client.chat.completions.calls, 1)
            self.assertEqual(get_llm_call_count(), 1)
        finally:
            restore_llm_call_counter(token)


if __name__ == "__main__":
    unittest.main()
