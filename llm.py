import time
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

def llm_response(message,avatar_session:'BaseAvatar',datainfo:dict={}):
    # 这是“文本 -> 大模型 -> 分段文本 -> TTS”的桥接函数。
    # 它的目标不是一次性生成完整答案，而是尽快把第一批可播报文本送出去，
    # 从而降低直播场景里的主观等待时间。
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        from openai import OpenAI
        client = OpenAI(
            # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            # 填写DashScope SDK的base_url
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{'role': 'system', 'content': '你是一个知识助手，尽量以简短、口语化的方式输出'},
                    {'role': 'user', 'content': message}],
            stream=True,
            # 通过以下设置，在流式输出的最后一行展示token使用信息
            stream_options={"include_usage": True}
        )
        result=""
        first = True
        for chunk in completion:
            if len(chunk.choices)>0:
                #print(chunk.choices[0].delta.content)
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos=0
                # 按标点拆分，是为了把流式结果切成适合立即播报的短句。
                # 如果整段文字攒太久才发给 TTS，数字人会显得“思考很久才开口”。
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；" :
                        result = result+msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result)>10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result,datainfo)
                            result=""
                result = result+msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            # 最后可能会剩下一段没有标点的尾巴，这里补发，避免漏播。
            avatar_session.put_msg_txt(result,datainfo)
        
    except Exception as e:
        logger.exception('llm exceptiopn:')
        return   
