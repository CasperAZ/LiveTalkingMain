###############################################################################
# WebRTC 会话管理
# 这是服务端“RTC 连接层”的控制器：只负责建立/销毁连接，真正口型/音画生成在 avatar session 内。
###############################################################################

import asyncio
import copy
import json
from typing import Dict, Optional

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger
from server.session_manager import session_manager


class RTCManager:
    """
    管理当前进程中的 WebRTC 连接：
    - 前端来 /offer（入栈场景）时负责和 session_manager 协作创建会话；
    - 也支持 rtcpush 主动推流场景；
    - 统一收集 RTCPeerConnection，用于关闭时统一回收。

    可以把它理解成“信令层的协调员”。
    """

    def __init__(self, opt):
        # 学习注释：把配置项保存下来。当前类主要用不到 opt 的详细字段，但保留是为了后续扩展统一访问。
        self.opt = opt
        # 学习注释：当前进程内所有已创建 PC 的集合，避免漏掉连接关闭导致资源悬挂。
        self.pcs: set = set()

    async def handle_offer(self, request):
        """
        浏览器端标准 webrtc 握手入口（offer 接收者）。
        流程：解析前端 JSON -> 创建数字人会话 -> 创建 pc -> 设置 tracks -> 本地生成 answer -> 返回给前端。
        """
        # 学习注释：aiohttp 的请求体是异步读取，这里解析 JSON。
        # 返回中必须有 sdp / type（通常 type='offer'）。
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # 学习注释：先在 session_manager 创建一个会话。
        # 这一步主要发生在服务逻辑层，把“会话 ID + avatar 会话实体”固定下来。
        # params 中可带 avatar/refaudio/reftext/custom_config 等参数，由 session_builder 统一消费。
        sessionid = await session_manager.create_session(params)
        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 学习注释：建立 pc + STUN 地址。
        # STUN 用于补齐外网候选地址（NAT 穿透），使连接更容易打通。
        # 这里只配了一个公开 STUN，未配 TURN；在严格网络下可能不够用。
        ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[ice_server])
        )
        # 学习注释：把新建连接入池。后面 shutdown 时可一口气关闭全部。
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            # 学习注释：pc 生命周期里会触发状态机变化（connecting -> connected -> failed/closed）。
            # failed/closed 时清理是为了避免“空会话”占用资源。
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid)

        # 学习注释：把 avatar session 的渲染输出，映射成 aiortc 可消费的两条媒体轨道。
        # HumanPlayer 内部会把数字人引擎产出的音视频帧推进到 track。
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 学习注释：显式设置视频编解码偏好。
        # 一般建议 H264 优先，VP8 备选；rtx 一般用于重传辅助。
        # 这样协商器有更明确的优先级。
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        # 学习注释：标准 SDP 协商流程：
        # 1) setRemoteDescription(offer) 接收对端 offer；
        # 2) createAnswer 生成 answer；
        # 3) setLocalDescription(answer) 后把本端应答入状态机。
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # 学习注释：返回给前端的是 answer 的 SDP + sessionid，前端继续进行 setRemoteDescription 完成通话。
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """
        rtcpush 场景：
        这类是服务端“主动推送”到远端 URL，不是浏览器发起 /offer。
        常见于推流到中转服务/平台集群。
        """
        import aiohttp

        # 学习注释：rtcpush 不走浏览器 offer，而是服务端先 create_session 再发起本地 offer。
        # 这里用空参数创建会话，表示使用默认配置/默认 avatar。
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 学习注释：同样建立一个 RTCPeerConnection 并纳入统一管理。
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            # 学习注释：rtcpush 只做连接层清理，不触发 session_manager.remove_session（可按你的业务再加）。
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        # 学习注释：此处和 handle_offer 对称：都把同一个 HumanPlayer 注入为音/视频 track。
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 学习注释：服务端先建立本地 offer 并变成 SDP。
        await pc.setLocalDescription(await pc.createOffer())

        # 学习注释：把 SDP 发给推流端提供的 signaling 地址，拿回 answer SDP 后再设置为远端描述。
        # 这是标准的主动发起方（ClientOfferer）流程。
        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """服务关闭时，统一关闭所有 PC 连接。"""
        # 学习注释：先拼出全部 close 协程，再并发等待，避免逐个等待带来额外阻塞。
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        # 学习注释：set 清空仅是 Python 容器层释放；底层资源应在 pc.close() 后关闭。
        self.pcs.clear()
