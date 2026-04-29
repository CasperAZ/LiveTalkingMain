import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

from llm import llm_response


class DummyAvatar:
    def __init__(self, opt):
        self.opt = opt
        self.sessionid = opt.sessionid
        self.parts = []

    def put_msg_txt(self, text, datainfo=None):
        self.parts.append(text)


class LLMFilterTest(unittest.TestCase):
    def test_blocked_input_is_ignored_without_model_call(self):
        with tempfile.TemporaryDirectory() as tempdir:
            words_path = Path(tempdir) / "sensitive_words.txt"
            words_path.write_text("测试违禁词\n", encoding="utf-8")
            opt = SimpleNamespace(
                sessionid="filter-test-session",
                llm_base_url="http://127.0.0.1:1/v1",
                llm_model="gemma-local",
                llm_api_key="local-not-needed",
                llm_memory_rounds=10,
                llm_system_prompt="system",
                llm_persona_prompt="persona",
                llm_live_context="",
                llm_sensitive_words_path=str(words_path),
                llm_filter_reply="",
            )
            avatar = DummyAvatar(opt)

            llm_response("这个问题包含测试违禁词", avatar, {})

        self.assertEqual(avatar.parts, [])

    def test_blocked_input_can_return_optional_filter_reply(self):
        with tempfile.TemporaryDirectory() as tempdir:
            words_path = Path(tempdir) / "sensitive_words.txt"
            words_path.write_text("测试违禁词\n", encoding="utf-8")
            opt = SimpleNamespace(
                sessionid="filter-test-session-with-reply",
                llm_base_url="http://127.0.0.1:1/v1",
                llm_model="gemma-local",
                llm_api_key="local-not-needed",
                llm_memory_rounds=10,
                llm_system_prompt="system",
                llm_persona_prompt="persona",
                llm_live_context="",
                llm_sensitive_words_path=str(words_path),
                llm_filter_reply="安全兜底话术",
            )
            avatar = DummyAvatar(opt)

            llm_response("这个问题包含测试违禁词", avatar, {})

        self.assertEqual(avatar.parts, ["安全兜底话术"])

    def test_blocked_model_output_returns_filter_reply(self):
        with tempfile.TemporaryDirectory() as tempdir, fake_openai_module(["测试违禁词。"]):
            words_path = Path(tempdir) / "sensitive_words.txt"
            words_path.write_text("测试违禁词\n", encoding="utf-8")
            opt = SimpleNamespace(
                sessionid="filter-output-test-session",
                llm_base_url="http://127.0.0.1:8020/v1",
                llm_model="gemma-local",
                llm_api_key="local-not-needed",
                llm_memory_rounds=10,
                llm_system_prompt="system",
                llm_persona_prompt="persona",
                llm_live_context="",
                llm_sensitive_words_path=str(words_path),
                llm_filter_reply="安全兜底话术",
            )
            avatar = DummyAvatar(opt)

            llm_response("正常问题", avatar, {})

        self.assertEqual(avatar.parts, ["安全兜底话术"])


@contextmanager
def fake_openai_module(chunks):
    previous = sys.modules.get("openai")
    module = ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_stream(chunks))
            )

    module.OpenAI = FakeOpenAI
    sys.modules["openai"] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous


def fake_stream(chunks):
    for content in chunks:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=content))
            ]
        )


if __name__ == "__main__":
    unittest.main()
