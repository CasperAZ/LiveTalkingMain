###############################################################################
# WebRTC 会话管理（PeerConnection 的创建与收口）
#
# 这层和 Go 的 signaling handler 类似：
# - 收到客户端 offer
# - 创建 RTCPeerConnection
# - 绑定音频/视频 track
# - 生成 answer / 透传给客户端
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
    管理当前进程里的 WebRTC 连接（pc 集合）以及 offer/push 两种入口。
    """

    def __init__(self, opt):
        self.opt = opt
        self.pcs: set = set()

    async def handle_offer(self, request):
        """
        典型 webrtc 推流入口（前端 -> /offer）。
        返回 answer + sessionid。
        """
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # 预留位：可在这里加 max_session 限流。当前默认走 session_manager 限制/外部限制。
        sessionid = await session_manager.create_session(params)
        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 建立 pc + STUN 地址
        ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[ice_server])
        )
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                # 连接失败/关闭时，清理 pc 并移除会话
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid)

        # 用 HumanPlayer 把 avatar 的 output 封装成 aiortc tracks
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 优先选 h264（可用时），再兜底 VP8/rtx
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

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
        RTCPush 出口（主动发起 offer 给平台推流端）。
        这里同样创建 pc，但是由服务端先呼叫目标 URL。
        """
        import aiohttp

        # 与 handle_offer 区别：这里没有“客户端 offer”，只构造服务端 offer 并等待 answer。
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        # 向第三方推流服务发布本地 SDP，拿到 answer 后回填
        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """服务关闭时，统一关闭已有 pc 连接。"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

