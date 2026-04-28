###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import asyncio
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import queue
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

"""
这里是流媒体时间戳桥接层。
BaseAvatar 产出的是逻辑帧，aiortc 需要的是带时间基的媒体帧；
这层负责把队列中的逻辑帧转成 WebRTC 可推帧，并按时间戳推进。
"""
AUDIO_PTIME = 0.020  # 20ms 一个音频包
VIDEO_CLOCK_RATE = 90000  # RTP 时钟基准
VIDEO_PTIME = 0.040  # 1/25 秒（默认 25fps）
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

# 这些常量决定 WebRTC 时间轴的推进方式：
# - AUDIO_PTIME=20ms：每次送一个 20ms 音频包
# - VIDEO_PTIME=40ms：对应 25fps 的视频输出
# - *_TIME_BASE：告诉播放器“pts 的单位是什么”

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger

# 学习注释：`logger` 是标准库日志实例，`mylogger` 是项目里的统一日志封装。

# 这个模块的作用是把“数字人内部生产出的音视频帧”包装成 aiortc 能识别的媒体轨。
# 你可以把它理解成：
# 上游是项目自己的渲染队列，下游是 WebRTC 浏览器播放器，中间靠这里做桥接。
class PlayerStreamTrack(MediaStreamTrack):
    """
    通用媒体 track：
    - 将 avatar 的帧从队列取出
    - 计算并设置 pts/time_base
    - 输出给 aiortc
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        # 学习注释：每个 Track 只服务一种类型（音频或视频），并按各自节奏管理时间戳。
        self.kind = kind
        # 学习注释：保留 player 引用，后续回调可通过它转发控制/通知/事件信息。
        self._player = player
        # 学习注释：固定长度队列把“生产者线程”和“消费端协程”解耦，防止无限堆积。
        self._queue = queue.Queue(maxsize=100)
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        """
        按 kind 推进时间戳：video 40ms 一帧，audio 20ms 一包。
        如果发送过快，会简单 sleep 到目标 wall clock 时间。
        """
        # 学习注释：aiortc 要求每个输出帧都带 pts 与 time_base。
        # 固定步长推进是低延迟流式播放里常见的节奏控制方式。
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                # 用固定步长递增时间戳，比直接取当前 wall clock 更稳定。
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                # 音频同理，按固定 20ms 的节奏推进采样时间轴。
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # aiortc 需要一帧媒体数据时会调用这里。
        # 推流消费者：第一次 recv 时启动后台 worker，持续从 BaseAvatar output 拉帧。
        # 学习注释：aiortc 会反复调用这个协程，每次取一帧并把时间戳信息补齐后返回。
        self._player._start(self)
        # if self.kind == 'video':
        #     frame = await self._queue.get()
        # else: #audio
        #     if hasattr(self, "_timestamp"):
        #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
        #         if wait>0:
        #             await asyncio.sleep(wait)
        #         if self._queue.qsize()<1:
        #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
        #             audio = np.zeros((1, 320), dtype=np.int16)
        #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
        #             frame.sample_rate=16000
        #         else:
        #             frame = await self._queue.get()
        #     else:
        while True:
            try:
                frame, eventpoint = self._queue.get_nowait()
                break
            # 学习注释：queue.Empty 表示本轮还没新包入队。
            # 短暂 sleep 5ms，避免空转导致 CPU 占满。
            except queue.Empty:
                # 队列暂时没数据时短暂休眠，避免 CPU 空转。
                # 队列为空时，短暂等待避免 CPU 空转。
                await asyncio.sleep(0.005)
                
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint and self._player is not None:
            # 音频事件（开始/结束/文本等）随帧同步回上层。
            self._player.notify(eventpoint)
        # 学习注释：eventpoint 跟随当前帧时钟同步，动作回调可按播放时间线对齐。
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        # 学习注释：停止 Track 生命周期并清空已缓存帧，避免停流后继续泄漏播放。
        super().stop()
        # Drain & delete remaining frames
        while not self._queue.empty():
            item = self._queue.get_nowait()
            del item
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    container
):
    # 学习注释：这是独立渲染线程入口，负责持续调用 container.render。
    container.render(quit_event)

class HumanPlayer:

    def __init__(
        self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        # 学习注释：这个类负责包装两个媒体轨道（音频/视频），
        # 并把渲染出的帧喂给 aiortc。
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        self.__container = avatar_session
        if hasattr(self.__container, 'output'):
            # 让 output 层反向持有 player，这样 output.push_* 可以直接喂给 WebRTC。
            self.__container.output._player = self

    def push_video(self, frame):
        # 学习注释：渲染线程调用此方法，把一帧图像入队给 WebRTC 发送。
        """把 BGR ndarray 转成 av.VideoFrame 并放进视频队列。"""
        from av import VideoFrame
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        self.__video._queue.put((new_frame, None))

    def push_audio(self, frame, eventpoint=None):
        # 学习注释：将 int16 PCM 音频片段和可选事件点一起入队，和音画同步。
        """把 int16 PCM 转成 av.AudioFrame 并放进音频队列。"""
        from av import AudioFrame
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = 16000
        self.__audio._queue.put((new_frame, eventpoint))

    def get_buffer_size(self) -> int:
        # 学习注释：输出队列长度监控指标，便于观察是否有卡顿堆积。
        return self.__video._queue.qsize()

    def notify(self,eventpoint):
        # 学习注释：只在容器存在时转发事件，避免销毁后空指针访问。
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        # 学习注释：惰性启动。仅当第一条轨道订阅时才创建渲染线程。
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            # 第一次真正开始消费媒体轨时，才启动底层渲染线程，属于懒启动。
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        # 学习注释：当没有活动轨道时关闭渲染线程，并释放容器引用。
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
