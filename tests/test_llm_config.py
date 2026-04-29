import sys
import unittest
from unittest.mock import patch

from config import parse_args


class LLMConfigTest(unittest.TestCase):
    def test_defaults_point_to_local_gemma_service(self):
        with patch.object(sys, "argv", ["app.py"]):
            opt = parse_args()

        self.assertEqual(opt.llm_base_url, "http://127.0.0.1:8020/v1")
        self.assertEqual(opt.llm_model, "gemma-local")
        self.assertEqual(opt.llm_api_key, "local-not-needed")
        self.assertIn("只输出", opt.llm_system_prompt)
        self.assertIn("直播数字人助手", opt.llm_persona_prompt)
        self.assertEqual(opt.llm_live_context, "")
        self.assertEqual(opt.llm_memory_rounds, 10)
        self.assertEqual(opt.llm_sensitive_words_path, "data/sensitive_words")
        self.assertEqual(opt.llm_filter_reply, "")

    def test_llm_options_can_be_overridden_from_cli(self):
        with patch.object(
            sys,
            "argv",
            [
                "app.py",
                "--llm_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "--llm_model",
                "qwen-plus",
                "--llm_api_key",
                "test-key",
                "--llm_system_prompt",
                "test system prompt",
                "--llm_persona_prompt",
                "test persona",
                "--llm_live_context",
                "test context",
                "--llm_memory_rounds",
                "3",
                "--llm_sensitive_words_path",
                "test_words.txt",
                "--llm_filter_reply",
                "test filter reply",
            ],
        ):
            opt = parse_args()

        self.assertEqual(opt.llm_base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(opt.llm_model, "qwen-plus")
        self.assertEqual(opt.llm_api_key, "test-key")
        self.assertEqual(opt.llm_system_prompt, "test system prompt")
        self.assertEqual(opt.llm_persona_prompt, "test persona")
        self.assertEqual(opt.llm_live_context, "test context")
        self.assertEqual(opt.llm_memory_rounds, 3)
        self.assertEqual(opt.llm_sensitive_words_path, "test_words.txt")
        self.assertEqual(opt.llm_filter_reply, "test filter reply")


if __name__ == "__main__":
    unittest.main()
