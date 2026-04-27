###############################################################################
# 输出插件：WebRTC
#
# 这里不直接创建 WebRTC 连接；RTCManager 已经创建了 RTCPeerConnection，
# HumanPlayer 负责给 aiortc 提供音视频 Track。
# 本类只是“桥接点”：接收 BaseAvatar 推出的帧，转发给 HumanPlayer。
###############################################################################

from streamout.base_output import BaseOutput
from registry import register
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "webrtc")
class WebRTCOutput(BaseOutput):
    """
    WebRTC 输出插件。

    真实的数据输送在 HumanPlayer 内：
    - VideoFrame/AudioFrame -> aiortc track -> 对端浏览器/WebRTC 推流。
    这里持有 _player 引用，避免每帧重复查找。
    """

    def __init__(self, opt=None, parent: Optional["BaseAvatar"] = None, **kwargs):
        super().__init__(opt, parent)
        self._player = None

    def start(self) -> None:
        """
        WebRTC 分支时无需在这里创建连接。

        RTCManager 负责信令/连接建立，
        BaseAvatar 的 render 流程会在会话准备好后调用 HumanPlayer，
        它会把 self 注入到 _player 再消费 push_*。
        """
        # 有意为空；保留是为了和 BaseOutput 契约一致。
        pass

    def push_video_frame(self, frame) -> None:
        # 由 HumanPlayer 转成 av.VideoFrame 并 push 到内核队列。
        if self._player:
            self._player.push_video(frame)

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        # eventpoint 一般用于通知（开始/结束/自定义状态）给上层动画状态机。
        if self._player:
            self._player.push_audio(frame, eventpoint)

    def get_buffer_size(self) -> int:
        # 对 WebRTC 来说，buffer 大小在 HumanPlayer 内部 track 队列里。
        if self._player and hasattr(self._player, "get_buffer_size"):
            return self._player.get_buffer_size()
        return 0

    def stop(self) -> None:
        # WebRTC 停止动作由 session/rpc 与 rtc_manager 统一回收；
        # 这里留空保证接口一致。
        pass
