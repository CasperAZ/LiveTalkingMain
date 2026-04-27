###############################################################################
#  Output 抽象层
#
#  这是整个项目“输出网关”的统一入口定义。
#  它不负责“怎么推流”，只定义“推什么东西”。
#  具体实现（webrtc/rtmp/virtualcam）只要按接口实现即可。
#
#  使用场景回顾：
#  - avatar 在推理后会拿到一张视频帧 + 一帧音频。
#  - 同一套输入可以被不同 transport 使用（同一模型，多种输出方式）。
###############################################################################

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from numpy.typing import NDArray
import numpy as np

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


class BaseOutput(ABC):
    """
    输出基类（抽象类）。

    任何输出插件都要遵循这四个最小能力：
    1. start(): 启动本输出链路。
    2. push_video_frame(): 推送一帧视频。
    3. push_audio_frame(): 推送一帧音频。
    4. stop(): 停止并回收资源。

    这样上层代码只依赖“能力”，不会关心你是推到虚拟摄像头、
    还是推到 RTMP、还是给 aiortc 提供 RTP Track。
    """

    def __init__(self, opt=None, parent: Optional["BaseAvatar"] = None, **kwargs):
        # opt: 命令行配置（transport 细节、fps 等）
        # parent: 当前会话对应的 avatar 实例（可回调通知）
        self.opt = opt
        self.parent = parent

    @abstractmethod
    def start(self) -> None:
        """启动输出链路（必须由子类实现）"""
        ...

    @abstractmethod
    def push_video_frame(self, frame) -> None:
        """推送视频帧（必须由子类实现）"""
        ...

    @abstractmethod
    def push_audio_frame(self, frame: NDArray[np.int16], eventpoint=None) -> None:
        """推送音频帧（必须由子类实现）"""
        ...

    def get_buffer_size(self) -> int:
        """
        可选能力：返回当前输出端积压帧数量（默认 0）。
        一些输出端会缓冲队列，值可以用于监控“卡顿积压”。
        """
        return 0

    @abstractmethod
    def stop(self) -> None:
        """停止输出链路并释放资源（必须由子类实现）"""
        ...
