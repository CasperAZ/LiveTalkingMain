###############################################################################
# 路由层（Routes / Controller）
#
# 这里可以当成 Go 的 handler 层理解：
#   - 每个 @POST endpoint 对应一个处理函数
#   - 请求进来后只做参数适配，不直接负责推理细节
#   - 真正逻辑在 BaseAvatar / 模块层
###############################################################################

import asyncio
import json
from aiohttp import web

import numpy as np  # 兼容历史代码保留

from utils.logger import logger
from server.session_manager import session_manager


def json_ok(data=None):
    """
    统一成功响应。
    与 Go 中经常写的 c.JSON(200, gin.H{...}) 类似。
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
    统一错误响应。
    当前项目沿用了“HTTP 200 + 业务码”的风格。
    """
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


def get_session(request, sessionid: str):
    """
    从会话仓库里取会话。

    request 是 aiohttp 的 web.Request；这里保留 request 是为了统一入口参数签名，
    实际上在当前文件里没有使用这个对象，目的是未来扩展更清晰。
    """
    return session_manager.get_session(sessionid)


async def human(request):
    """
    统一处理聊天/发音输入。

    请求 body:
    {
      "sessionid": "...",
      "type": "echo" | "chat",
      "text": "...",
      "interrupt": true/false,
      "tts": {"voice": "...", "emotion": "..."}  # 可选
    }
    """
    try:
        # JSON 反序列化：aiohttp 里的 request.json() 是 awaitable
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        # 客户端想打断当前发音时，清空排队状态
        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):
            # datainfo 会被透传到 TTS 模块（用于 voice/emotion 等扩展）
            datainfo['tts'] = params.get('tts')
        if params.get('llm'):
            # 预留给直播人设/当前口播内容/商品上下文等 LLM 编排信息。
            # 示例：{"llm": {"persona": "...", "context": "..."}}
            datainfo['llm'] = params.get('llm')

        # echo：把文本直接当字幕/一句话输入
        if params['type'] == 'echo':
            avatar_session.put_msg_txt(params['text'], datainfo)
        # chat：调用 llm_response 生成更完整回复，再继续进入 TTS
        elif params['type'] == 'chat':
            llm_response = request.app.get("llm_response")
            if llm_response:
                # llm_response 可能是较慢操作，扔给 executor 防止阻塞事件循环
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params['text'], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """
    手动中断当前会话正在进行的播报/语音排队。
    与 UI 上“打断”按钮语义一致。
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
    上传音频文件进行播报。

    表单字段：
    - sessionid: 会话 ID
    - file: 文件流
    """
    try:
        # multipart/form-data 的读取方式和 JSON 不同
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        # 交给会话层处理切片和识别/喂声道
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """
    设置“动作类型”/自定义状态码（custom state）。

    语义在上层业务决定，例如：
    1=静默、2=欢迎、3=礼物反馈...
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """
    通用录制控制：
    {"sessionid":"...", "type":"start_record"} / {"type":"end_record"}
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
    查询当前会话是否正在播放（bool）
    """
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


def setup_routes(app):
    """
    注册对外控制面路由。
    这里是服务稳定层，后续可在这里把平台回调 webhook 映射进同一套 handler。
    """
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static('/', path='web')
