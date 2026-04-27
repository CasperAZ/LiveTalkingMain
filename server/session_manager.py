###############################################################################
#  全局会话管理器 (Session Manager)
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar

def _rand_session_id() -> str:
    """
    生成随机 session_id。

    这里使用 UUID4，目标是尽量降低冲突概率，
    便于在并发创建会话时直接拿来作为外部可见会话标识。
    """
    return str(uuid.uuid4())


class SessionManager:
    """
    全局数字人会话管理器（Singleton + Factory Injection）。

    这个类做两件事：
    1) 管理会话容器：增、查、删，维护 session_id -> BaseAvatar 实例映射。
    2) 统一创建入口：通过外部注入的 build_session_fn 构建会话，避免业务层到处散落“如何创建会话”的细节。

    它刻意不做的事：
    - 不关心具体是 wav2lip / musetalk / ultralight；
    - 不直接写死 TTS/ASR/渲染初始化逻辑；
    这些都交给 build_session_fn，SessionManager 只做编排与生命周期管理。
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        # 单例模式：整个进程只保留一个 SessionManager 实例。
        # 好处是所有模块都读写同一份 sessions 状态，避免多份状态带来的错乱。
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # __new__ 虽然保证了单例实例，但 __init__ 可能被多次调用，
        # 因此用 initialized 做“仅首次初始化”保护。
        if not hasattr(self, "initialized"):
            # 会话注册表：key 是 session_id，value 是 BaseAvatar 实例。
            # 注意：创建过程中的“占位态”会临时写入 None（见 create_session）。
            self.sessions: Dict[str, BaseAvatar] = {}
            # 会话工厂函数，由外部注入。
            # 期望签名：build_session_fn(sessionid: str, params: dict) -> BaseAvatar
            self.build_session_fn = None
            self.initialized = True

    def init_builder(self, build_session_fn):
        """
        注入“会话工厂函数”。

        这就是本文件里最核心的“工厂模式落点”：
        - SessionManager 负责“何时创建、如何登记、如何并发安全地创建”；
        - build_session_fn 负责“创建什么对象、里面初始化哪些组件”。
        通过依赖注入，两者解耦，后续替换会话实现时无需改 SessionManager。
        """
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """
        通过 session_id 获取会话对象。

        返回值可能是：
        - BaseAvatar 实例：会话已创建完成；
        - None：不存在，或正处于“占位创建中”（取决于调用时机）。
        """
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """
        判断会话是否“可用”。

        这里不仅检查 key 是否存在，还要求 value 不是 None，
        从而把“创建中占位”与“可实际使用”区分开。
        """
        return sessionid in self.sessions and self.sessions[sessionid] is not None
        
    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        在异步上下文中创建并注册一个新会话。

        调用流程：
        1) 检查是否已经注入 build_session_fn；
        2) 若未指定 session_id，则自动生成 UUID；
        3) 先写入 self.sessions[sessionid] = None 作为“创建中占位”；
        4) 在线程池执行 build_session_fn（避免阻塞 asyncio 事件循环）；
        5) 创建成功后回填真实会话对象并返回 session_id。

        为什么要 run_in_executor（给 Go 背景同学的对照）：
        - 可以把它理解成：把“重活函数”丢到线程池 worker 去跑，
          主事件循环继续处理网络请求，不被阻塞。
        - 行为上接近“异步提交任务 + 等结果”，而不是裸开 goroutine 后不管结果。
        - 真正拿结果发生在 await 这一行；如果子任务失败，异常也会在 await 时抛回当前协程。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")
            
        if sessionid is None:
            sessionid = _rand_session_id()
            
        logger.info('Creating sessionid=%s, current session num=%d', sessionid, len(self.sessions))
        # 先占位：防止并发请求在“尚未建好”窗口重复触发同一 session 的构建。
        # 配合 has_session()，外层可以判断它是否已经真正可用。
        self.sessions[sessionid] = None

        # Go 对照理解：
        # 1) build_session_fn 在线程池里执行；其异常会先被 Future 持有。
        # 2) 执行到 await 时，Future 的异常会在“当前协程”重新抛出，
        #    这就是这里说的“向上冒泡”（更像 Go 的 err 逐层 return）。
        # 3) 一旦这里抛异常，下面“回填真实对象”不会执行，session 会保持 None 占位；
        #    当前实现不自动清理，占位可由上层在失败路径里 remove_session 补偿。
        avatar_session = await asyncio.get_event_loop().run_in_executor(
            None, self.build_session_fn, sessionid, params
        )
        # 构建完成后，替换占位为真实会话实例。
        self.sessions[sessionid] = avatar_session
        return sessionid
        
    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """
        同步注册一个已构建好的会话对象。

        常见场景：
        - 非 WebRTC 模式（如 virtualcam/rtmp）启动时手工创建固定会话；
        - 外部流程已经构建好会话，只需要接入统一管理。
        """
        self.sessions[sessionid] = avatar_session
        
    def remove_session(self, sessionid: str):
        """
        从注册表移除会话。

        当前行为是“管理层解绑引用”，并不等价于完整资源回收。
        若要做彻底清理，可在此扩展：
        - 停止渲染/推流线程；
        - 关闭网络连接与媒体输出；
        - 释放临时缓存或文件句柄。
        """
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            self.sessions.pop(sessionid, None)

# 模块级单例导出：其他模块直接 from ... import session_manager 即可共享同一实例。
session_manager = SessionManager()
