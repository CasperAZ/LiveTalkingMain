###############################################################################
# 输出插件：RTMP
#
# 用于把数字人内容推到流媒体服务器（nginx-rtmp / srs / ...），
# 常见接入方式是直接把单个 URL 推给对应平台或中转服务。
###############################################################################

import subprocess  # 保留历史导入（兼容旧逻辑中可能用到的调用）
import time
import numpy as np
from streamout.base_output import BaseOutput
from registry import register
from utils.logger import logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "rtmp")
class RTMPOutput(BaseOutput):
    """
    RTMP 推流实现。

    核心逻辑：
    - 第一次收到视频时初始化 StreamerConfig 与推流器。
    - video 先转 RGB 送给底层 rtmp_streaming。
    - audio 先入队或直接推送，避免视频/音频顺序错乱。
    """

    def __init__(self, opt=None, parent: Optional["BaseAvatar"] = None, **kwargs):
        super().__init__(opt, parent)
        self.push_url = getattr(
            opt, "push_url", "rtmp://localhost/live/livestream"
        )
        self.width = getattr(opt, "W", 450)
        self.height = getattr(opt, "H", 450)
        self.fps = getattr(opt, "fps", 25)
        self.bitrate = getattr(opt, "bitrate", 1000000)
        self._streamer = None

        # 统计与节奏校正
        self.framecount = 0
        self.lasttime = time.perf_counter()
        self.totaltime = 0

    def start(self) -> None:
        """
        初始化音频队列和退出标志。

        注意：真实 Streamer 要等到第一帧视频拿到尺寸后再创建，
        这样可以拿到准确的宽高信息。
        """
        import queue

        self._audio_queue = queue.Queue()
        self._quit_event = False

    def _init_streamer(self, frame_height, frame_width):
        """
        延迟初始化推流器（懒加载），避免没有有效帧时浪费资源。
        """
        try:
            from rtmp_streaming import StreamerConfig, Streamer
        except ImportError:
            logger.error(
                "rtmp_streaming is not installed. Please install python_rtmpstream."
            )
            raise

        sc = StreamerConfig()
        sc.source_width = frame_width
        sc.source_height = frame_height
        sc.stream_width = frame_width
        sc.stream_height = frame_height
        sc.stream_fps = self.fps
        sc.stream_bitrate = self.bitrate
        sc.stream_profile = "main"
        sc.audio_channel = 1

        # 默认按父 session 里真实采样率发送音频，避免和系统假设不一致。
        sc.sample_rate = getattr(self.opt, "sample_rate", 16000)
        if self.parent:
            sc.sample_rate = self.parent.sample_rate

        sc.stream_server = self.push_url

        self._streamer = Streamer()
        self._streamer.init(sc)

        self._starttime = time.perf_counter()
        self._totalframe = 0
        logger.info(
            f"RTMP output started via python_rtmpstream: "
            f"{self.push_url} with resolution {frame_width}x{frame_height}"
        )

    def push_video_frame(self, frame) -> None:
        if isinstance(frame, np.ndarray):
            if self._streamer is None:
                # 首帧到来时才能确定分辨率，才可创建推流器。
                self.height, self.width = frame.shape[:2]
                self._init_streamer(self.height, self.width)

                # 如果音频先到先积压，这里把队列里的音频先放出来保持同步。
                while not self._audio_queue.empty():
                    buffered_audio = self._audio_queue.get()
                    self._streamer.stream_frame_audio(buffered_audio)

            import cv2

            # rtmp_streaming 期望 RGB，OpenCV 默认 BGR。
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._streamer.stream_frame(rgb_frame)

            # 40ms 一帧（25fps）做简单节奏对齐，防止送得太快。
            delay = self._starttime + self._totalframe * 0.04 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self._totalframe += 1

            # 打印窗口内的平均输出 fps，便于判断吞吐。
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                logger.info(
                    f"------actual avg final fps:{self.framecount / self.totaltime:.4f}"
                )
                self.framecount = 0
                self.totaltime = 0

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        if isinstance(frame, np.ndarray):
            # rtmp_streaming 接口接收 float32，这里把 int16 转换为 [-1,1]。
            if frame.dtype == np.int16:
                frame = frame.astype(np.float32) / 32767.0

            if self._streamer:
                self._streamer.stream_frame_audio(frame)
                self.parent.notify(eventpoint)
            else:
                # 推流器还没初始化时先缓存音频，避免丢包。
                self._audio_queue.put(frame)

    def stop(self) -> None:
        # 这里仅置空实例；上层生命周期会回收对象。
        self._quit_event = True
        self._streamer = None
        logger.info("RTMP output stopped")
