from threading import Thread
import queue
from queue import Queue
from io import BytesIO
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger


class State(Enum):
    # 供子类/上层理解：TTS 工作态 or 暂停态。
    RUNNING = 0
    PAUSE = 1


class BaseTTS:
    """
    所有 TTS 插件的基类。

    关键职责：
    - 接收文本消息（put_msg_txt），放入内部队列。
    - 在独立线程渲染文本到 PCM（子类实现 txt_to_audio）。
    - 把音频切片推送到 BaseAvatar 通道。

    这里不会直接“讲”一次文本，而是持续消费队列，
    所以你做成“可中断、可并发”的行为会更容易。
    """

    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        # 音频统一采样率 16kHz，和前后端音频处理保持一致
        self.sample_rate = 16000
        # 每块 20ms 大小：16000 * 20ms = 320 样点
        self.chunk = self.sample_rate // (opt.fps * 2)
        self.input_stream = BytesIO()

        # 文本任务队列 + 状态
        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        """
        清空当前未处理文本，状态置为 pause。
        在上游触发“打断/立即结束”时会被调用。
        """
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        """
        上游只要拿到文本就进队列，具体合成时机由后台线程决定。
        datainfo 可携带动作/音色等控制元信息。
        """
        if len(msg) > 0:
            self.msgqueue.put((msg, datainfo))

    def render(self, quit_event):
        """
        启动独立渲染线程。此线程会不断从 msgqueue 拉任务。
        quit_event 由会话销毁时置位，用于优雅退出。
        """
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()

    def process_tts(self, quit_event):
        """
        子类通常不覆写这层逻辑，只覆写 txt_to_audio。
        这样不同 TTS 厂商可复用同一套队列和生命周期行为。
        """
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            # 到这里就是“拿到一段文字”，交给子类实现。
            self.txt_to_audio(msg)
        self.stop_tts()
        logger.info("ttsreal thread stop")

    def txt_to_audio(self, msg: tuple[str, dict]):
        """
        子类必须实现：把文本变成音频帧并投递给上游输出。
        例如 edge/xtts/gpt-sovits ...
        """
        pass

    def stop_tts(self):
        """
        子类可实现：关闭底层客户端、清理流、释放模型句柄。
        """
        pass
