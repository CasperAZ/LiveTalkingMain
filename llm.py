import time
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    """
    大模型响应链路。

    位置：`/human` 接口在 type=chat 分支里异步调用，最终会把结果文字
    再喂回 avatar_session.put_msg_txt()，交由 TTS 合成。

    设计是“事件流”：
    - 先拿到 chunk stream；
    - 拼接为自然断句（以标点为边界）；
    - 每到一个小片段就投递给 TTS（让数字人更快出现语音反馈）。
    """
    try:
        opt = avatar_session.opt
        start = time.perf_counter()

        # 这里使用 OpenAI 兼容接口（DashScope qwen-plus）
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")

        completion = client.chat.completions.create(
            model="qwen-plus",
            # system 设定角色，下面 messages 的 user 段是你传进来的原始文本
            messages=[
                {
                    "role": "system",
                    "content": "你是一个善于表达、语速适中的中文口播数字人。",
                },
                {"role": "user", "content": message},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        result = ""
        first = True
        for chunk in completion:
            if len(chunk.choices) > 0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False

                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue

                # 基于标点打断，避免等到完整段落才开始TTS，提升实时性。
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。；：！？":
                        result = result + msg[lastpos : i + 1]
                        lastpos = i + 1
                        if len(result) > 10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        # 剩下的碎片也要补推一遍，避免丢尾巴。
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception as e:
        logger.exception("llm exception:")
        return
