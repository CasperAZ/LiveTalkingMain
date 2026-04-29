import unittest

from llm import _build_chat_messages, _clear_llm_memory, _record_chat_turn


class LLMSessionMemoryTest(unittest.TestCase):
    def setUp(self):
        _clear_llm_memory()

    def test_memory_is_isolated_by_session_id(self):
        _record_chat_turn("session-a", "你好", "你好，欢迎来到直播间。", 10)

        session_a_messages = _build_chat_messages("system prompt", "session-a", "刚才我说了什么？", 10)
        session_b_messages = _build_chat_messages("system prompt", "session-b", "刚才我说了什么？", 10)

        self.assertEqual(
            session_a_messages,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，欢迎来到直播间。"},
                {"role": "user", "content": "刚才我说了什么？"},
            ],
        )
        self.assertEqual(
            session_b_messages,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "刚才我说了什么？"},
            ],
        )

    def test_memory_keeps_latest_ten_rounds(self):
        for index in range(12):
            _record_chat_turn("session-a", f"user-{index}", f"assistant-{index}", 10)

        messages = _build_chat_messages("system prompt", "session-a", "current", 10)
        contents = [message["content"] for message in messages]

        self.assertNotIn("user-0", contents)
        self.assertNotIn("assistant-0", contents)
        self.assertNotIn("user-1", contents)
        self.assertNotIn("assistant-1", contents)
        self.assertIn("user-2", contents)
        self.assertIn("assistant-11", contents)
        self.assertEqual(len(messages), 22)

    def test_zero_rounds_disables_memory(self):
        _record_chat_turn("session-a", "你好", "你好，欢迎来到直播间。", 0)

        messages = _build_chat_messages("system prompt", "session-a", "还有记忆吗？", 0)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "还有记忆吗？"},
            ],
        )

    def test_persona_and_live_context_are_inserted_before_history(self):
        _record_chat_turn("session-a", "上一句", "上一句回复", 10)

        messages = _build_chat_messages(
            "system prompt",
            "session-a",
            "当前问题",
            10,
            persona_prompt="人设：专业但亲和的主播",
            live_context="当前口播内容：空气炸锅限时优惠",
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "system", "content": "人设：专业但亲和的主播"},
                {"role": "system", "content": "当前口播内容：空气炸锅限时优惠"},
                {"role": "user", "content": "上一句"},
                {"role": "assistant", "content": "上一句回复"},
                {"role": "user", "content": "当前问题"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
