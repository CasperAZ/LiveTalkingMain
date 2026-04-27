###############################################################################
#  服务器路由层（Route / Controller）
#
#  Go 对照理解：
#  - 这一层相当于 gin/echo/net-http 里的 handler 层（HTTP 入口层）
#  - 负责：解析请求 -> 找会话 -> 调业务方法 -> 返回 JSON
#  - 不负责：模型推理细节、会话创建细节、底层音视频处理
#
#  换句话说，本文件是“薄控制层”，目标是让业务入口集中、可维护、可替换。
###############################################################################

import json
import numpy as np
import asyncio
from aiohttp import web

from utils.logger import logger


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """
    返回统一成功响应。

    约定响应格式：
    {"code": 0, "msg": "ok", "data": ...可选...}

    Go 对照：
    类似在 gin 里统一返回 c.JSON(200, gin.H{...}) 的 helper。
    """
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """
    返回统一错误响应。

    约定响应格式：
    {"code": 非0, "msg": "错误说明"}

    说明：
    - 这里仍然用 HTTP 200 + 业务码的风格（而不是 4xx/5xx），
      是这套项目当前的接口约定。
    - 如果你后续做网关/平台接入，可在这一层统一改返回规范。
    """
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager

def get_session(request, sessionid: str):
    """
    获取会话实例（薄封装）。

    参数说明：
    - request：按 aiohttp 约定，这里通常是 web.Request 实例；
      但 Python 语法层面不强制写类型注解，所以函数签名也可以先不标注类型。
    - sessionid：业务侧会话 ID。

    为什么封装一层：
    - 当前只是直接读 session_manager；
    - 未来可在这里统一加“按租户/平台路由不同会话仓库、鉴权、埋点、灰度”等逻辑，
      调用方无需改动。

    Go 对照：
    类似先写一个 getSession(ctx, id) helper，避免 handler 到处直接碰全局变量。
    """
    # 这里单独包一层，是为了以后你如果想切换成
    # “按平台/业务线路由到不同 session 仓库”，改动面更小。
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """
    文本入口：支持 echo/chat 两种模式，并透传 TTS 细分参数。

    请求体（示例）：
    {
      "sessionid": "...",
      "type": "echo" | "chat",
      "text": "...",
      "interrupt": true/false,
      "tts": {"voice": "...", "emotion": "..."}  # 可选
    }

    处理逻辑：
    - echo：直接把 text 放入播报队列，不走 LLM；
    - chat：把 text 交给 llm_response，再把生成结果持续喂给播报链路。

    Go 对照：
    - 这相当于一个 POST handler，先 BindJSON，再调 service。
    - 这里的 asyncio + run_in_executor 类似“主协程不阻塞，把重活交 worker”。
    """
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            # 新消息到来时先打断旧播报，适合直播互动场景。
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            # tts 子参数不会在这里解析，而是一路透传给具体 TTS 插件。
            # 这样路由层保持“协议转发”职责，不侵入具体引擎细节。
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            # echo 模式：直接朗读用户输入，不调用大模型。
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            # chat 模式：把输入交给大模型，后续边生成边播报。
            # request.app 类似 Go 里从全局容器/依赖注入容器取 service。
            llm_response = request.app.get("llm_response")
            if llm_response:
                # 放到线程池，避免 LLM 处理阻塞 aiohttp 事件循环。
                # 语义上类似“异步提交任务”，不是同步卡住当前请求线程。
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """
    立即打断当前播报。

    用途：
    - 用户发送新问题时抢占旧播报；
    - 直播场景里做“高优先级消息插队”。

    Go 对照：
    类似向播放器/worker 发送一个 flush/interrupt 控制命令。
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """
    上传整段音频文件，交给会话侧进行后续切块与驱动。

    输入形态：
    - multipart/form-data
    - sessionid: 文本字段
    - file: 音频文件字段

    说明：
    - 这里仅做上传接入，不在路由层做重采样/解码细节处理；
    - 具体音频拆帧与驱动逻辑下沉到 avatar_session。
    """
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        # 上传的是整段音频文件，后续会被切成 20ms 一块的 PCM 数据进入驱动链路。
        # Go 对照：相当于 handler 只负责收 multipart，真实处理交给 service 层。
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """
    设置会话“自定义状态码”（通常用于动作/表情编排）。

    输入示例：
    {"sessionid":"...", "audiotype":2}

    这一层只传递状态，不定义状态语义；
    具体“数字代表什么动作”由业务方在上层约定。
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        # audiotype 是动作编排状态号。
        # 比如业务上可以定义：
        # 1 = 静默待机
        # 2 = 欢迎新观众
        # 3 = 礼物反馈动作
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """
    录制控制入口：开始录制 / 停止录制。

    输入示例：
    {"sessionid":"...", "type":"start_record"}
    {"sessionid":"...", "type":"end_record"}

    Go 对照：
    这是典型“命令式接口”（command-style endpoint），
    handler 根据 type 分发到不同方法。
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """
    查询当前会话是否处于“正在说话”状态。

    返回 data 通常是布尔值：
    - true：当前仍在播报
    - false：当前空闲

    说明：
    - 该接口常用于前端轮询 UI 状态（按钮禁用、转圈提示等）。
    """
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """
    把所有 HTTP 入口注册到 aiohttp app。

    这里可以视为“路由总表”，便于你做平台适配时统一查看：
    - 哪些接口对外暴露
    - URL 与处理函数的映射关系

    Go 对照：
    类似 gin.Engine 上集中写 router.POST(...) / router.Static(...)。
    """
    # 这里就是项目当前所有对外业务入口的总表。
    # 快手适配时很可能会新增一层“平台 webhook -> 内部统一接口”的转换。
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static('/', path='web')
