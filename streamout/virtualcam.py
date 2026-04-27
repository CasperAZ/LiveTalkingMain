###############################################################################
# 输出插件：VirtualCam（桌面层摄像头输入）
#
# 目标：把数字人作为一台“本地摄像头”暴露给外部软件（OBS、会议软件等）。
# 一般流程：
# 1) 创建 pyvirtualcam.Camera，把每帧图像发送给虚拟摄像头。
# 2) 使用 PyAudio 在独立线程里播放音频，和视频时钟解耦。
###############################################################################

import numpy as np
from streamout.base_output import BaseOutput
from registry import register
from utils.logger import logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "virtualcam")
class VirtualCamOutput(BaseOutput):
    """
    虚拟摄像头输出。

    这是“把数字人当摄像头用”的典型方式：
    - 画面 -> 送给 pyvirtualcam.
    - 声音 -> 通过本地 PyAudio 线放。
    """

    def __init__(self, opt=None, parent: Optional["BaseAvatar"] = None, **kwargs):
        super().__init__(opt, parent)
        self.width = getattr(opt, "W", 450)
        self.height = getattr(opt, "H", 450)
        self.fps = getattr(opt, "fps", 25)
        self._cam = None
        self._audio_queue = None
        self._audio_thread = None
        self._quit_event = None

    def _play_audio_loop(self):
        """
        音频播放线程。

        注意：pyvirtualcam 只负责视频；音频不能直接挂到它上面。
        所以这里用 PyAudio 将音频帧在后台持续播放，避免阻塞主渲染线程。
        """
        import pyaudio
        import queue

        p = pyaudio.PyAudio()
        stream = p.open(
            rate=16000,
            channels=1,
            format=8,  # PyAudio int16
            output=True,
            output_device_index=1,
        )
        stream.start_stream()
        while not self._quit_event.is_set():
            try:
                data = self._audio_queue.get(block=True, timeout=1)
                stream.write(data)
            except queue.Empty:
                continue
        stream.close()
        p.terminate()

    def start(self) -> None:
        """
        启动音频播放线程。

        视频侧是惰性初始化（第一帧到来后创建 Camera），
        音频侧可提前启动线程，避免第一帧时卡顿。
        """
        try:
            import pyvirtualcam

            import queue
            from threading import Thread, Event

            self._audio_queue = queue.Queue(maxsize=3000)
            self._quit_event = Event()
            self._audio_thread = Thread(
                target=self._play_audio_loop, daemon=True, name="pyaudio_stream"
            )
            self._audio_thread.start()
        except ImportError:
            logger.error("pyvirtualcam not installed. pip install pyvirtualcam")
            raise

    def push_video_frame(self, frame) -> None:
        # frame 约定为 BGR ndarray（OpenCV 格式），第一次收到时按尺寸创建 Camera。
        if isinstance(frame, np.ndarray):
            if self._cam is None:
                import pyvirtualcam

                self.height, self.width = frame.shape[:2]
                self._cam = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                )
                logger.info(
                    f"VirtualCam output started: {self._cam.device} "
                    f"with resolution {self.width}x{self.height}"
                )

            self._cam.send(frame)
            self._cam.sleep_until_next_frame()

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        # 这里只负责把 int16 数组转成 bytes 放进音频队列。
        if self._audio_queue:
            self._audio_queue.put(frame.tobytes())
            # datainfo 中的事件会同步回调给 avatar session（如动作同步点）。
            self.parent.notify(eventpoint)

    def stop(self) -> None:
        # 先发退出信号，等待播放线程退出，再关闭虚拟摄像头。
        if self._quit_event:
            self._quit_event.set()
        if self._audio_thread:
            self._audio_thread.join()
        if self._cam:
            self._cam.close()
            self._cam = None
            logger.info("VirtualCam output stopped")
