import time
from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger
from content_filter import get_sensitive_word_filter


_LLM_MEMORY_BY_SESSION = defaultdict(list)
_LLM_MEMORY_LOCK = Lock()


def _normalize_memory_rounds(memory_rounds):
    try:
        return max(0, int(memory_rounds))
    except (TypeError, ValueError):
        return 10


def _trim_memory(messages, memory_rounds):
    memory_rounds = _normalize_memory_rounds(memory_rounds)
    if memory_rounds <= 0:
        return []
    return messages[-memory_rounds * 2:]


def _record_chat_turn(sessionid: str, user_message: str, assistant_message: str, memory_rounds: int = 10):
    if not sessionid:
        sessionid = "default"

    memory_rounds = _normalize_memory_rounds(memory_rounds)
    if memory_rounds <= 0:
        return

    with _LLM_MEMORY_LOCK:
        messages = _LLM_MEMORY_BY_SESSION[sessionid]
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])
        _LLM_MEMORY_BY_SESSION[sessionid] = _trim_memory(messages, memory_rounds)


def _build_chat_messages(
    system_prompt: str,
    sessionid: str,
    user_message: str,
    memory_rounds: int = 10,
    persona_prompt: str = "",
    live_context: str = "",
):
    if not sessionid:
        sessionid = "default"

    with _LLM_MEMORY_LOCK:
        history = list(_trim_memory(_LLM_MEMORY_BY_SESSION.get(sessionid, []), memory_rounds))

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if persona_prompt and persona_prompt.strip():
        messages.append({"role": "system", "content": persona_prompt.strip()})
    if live_context and live_context.strip():
        messages.append({"role": "system", "content": live_context.strip()})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def _clear_llm_memory(sessionid: str | None = None):
    with _LLM_MEMORY_LOCK:
        if sessionid is None:
            _LLM_MEMORY_BY_SESSION.clear()
        else:
            _LLM_MEMORY_BY_SESSION.pop(sessionid, None)


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

        llm_base_url = getattr(opt, "llm_base_url", "http://127.0.0.1:8020/v1")
        llm_model = getattr(opt, "llm_model", "gemma-local")
        llm_api_key = getattr(opt, "llm_api_key", "local-not-needed") or "local-not-needed"
        llm_memory_rounds = _normalize_memory_rounds(getattr(opt, "llm_memory_rounds", 10))
        llm_sensitive_words_path = getattr(opt, "llm_sensitive_words_path", "data/sensitive_words.txt")
        llm_filter_reply = getattr(
            opt,
            "llm_filter_reply",
            "",
        )
        llm_system_prompt = getattr(
            opt,
            "llm_system_prompt",
            "你在为数字人直播生成口播回复。请优先根据当前会话历史和当前直播上下文回答。只输出可以直接朗读的简短中文话术，不要输出动作描述、括号说明、表情符号或 Markdown。",
        )
        llm_persona_prompt = getattr(
            opt,
            "llm_persona_prompt",
            "人设：你是一个中文直播数字人助手，表达亲和、专业，语速适中。",
        )
        llm_live_context = getattr(opt, "llm_live_context", "")
        llm_datainfo = datainfo.get("llm", {}) if isinstance(datainfo, dict) else {}
        if isinstance(llm_datainfo, dict):
            llm_persona_prompt = (
                llm_datainfo.get("persona_prompt")
                or llm_datainfo.get("persona")
                or llm_persona_prompt
            )
            llm_live_context = (
                llm_datainfo.get("live_context")
                or llm_datainfo.get("context")
                or llm_live_context
            )
        sessionid = getattr(opt, "sessionid", None) or getattr(avatar_session, "sessionid", None) or "default"
        sensitive_filter = get_sensitive_word_filter(llm_sensitive_words_path)

        input_filter_result = sensitive_filter.check(message)
        if input_filter_result.blocked:
            logger.warning(
                f"llm input blocked by sensitive word: {input_filter_result.word},"
                f"category={input_filter_result.category}"
            )
            if llm_filter_reply:
                avatar_session.put_msg_txt(llm_filter_reply, datainfo)
            return

        # 这里使用 OpenAI 兼容接口。默认连接本机 llama.cpp Gemma 服务；
        # 如需切换远程模型，只需要改启动参数，不要在这里写死服务商。
        from openai import OpenAI

        client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
        )
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,model={llm_model},base_url={llm_base_url},{message}")

        completion = client.chat.completions.create(
            model=llm_model,
            # system + 当前 session 的短期历史 + 当前用户输入。
            messages=_build_chat_messages(
                llm_system_prompt,
                sessionid,
                message,
                llm_memory_rounds,
                persona_prompt=llm_persona_prompt,
                live_context=llm_live_context,
            ),
            stream=True,
            stream_options={"include_usage": True},
        )

        result = ""
        assistant_response = ""
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
                assistant_response += msg

                # 基于标点打断，避免等到完整段落才开始TTS，提升实时性。
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。；：！？":
                        result = result + msg[lastpos : i + 1]
                        lastpos = i + 1
                        if len(result) > 10:
                            output_filter_result = sensitive_filter.check(result)
                            if output_filter_result.blocked:
                                logger.warning(
                                    f"llm output blocked by sensitive word: {output_filter_result.word},"
                                    f"category={output_filter_result.category}"
                                )
                                if llm_filter_reply:
                                    avatar_session.put_msg_txt(llm_filter_reply, datainfo)
                                return
                            logger.info(result)
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        # 剩下的碎片也要补推一遍，避免丢尾巴。
        if result:
            output_filter_result = sensitive_filter.check(result)
            if output_filter_result.blocked:
                logger.warning(
                    f"llm output blocked by sensitive word: {output_filter_result.word},"
                    f"category={output_filter_result.category}"
                )
                if llm_filter_reply:
                    avatar_session.put_msg_txt(llm_filter_reply, datainfo)
                return
            avatar_session.put_msg_txt(result, datainfo)
        if assistant_response:
            _record_chat_turn(sessionid, message, assistant_response, llm_memory_rounds)

    except Exception as e:
        logger.exception("llm exception:")
        return
