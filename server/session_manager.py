###############################################################################
# 全局会话管理器（Session Manager）
#
# 这是项目里“会话编排”层，和 Go 的 singleton + registry 思路类似：
# - SessionManager 本身只管理会话对象（sessionid -> BaseAvatar）
# - 真正怎么创建会话由外部注册的 build_session_fn 决定（工厂注入）
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional

from utils.logger import logger
from avatars.base_avatar import BaseAvatar


def _rand_session_id() -> str:
    """生成随机 session id，默认用 uuid4（Go 场景下可理解为全局唯一 key）。"""
    return str(uuid.uuid4())


class SessionManager:
    """
    会话管理器（单例）。

    职责是：
    1. 维护 sessionid 到会话实例的映射；
    2. 提供统一入口创建/查询/注销会话；
    3. 不关心每个会话具体是哪个模型。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 单例：保证整个进程里只有一份会话总表。
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 防重复初始化：避免单例在第一次后多次重置状态。
        if not hasattr(self, "initialized"):
            # sessionid -> BaseAvatar（也可能临时是 None）
            self.sessions: Dict[str, BaseAvatar] = {}
            # build_session_fn 由 app.py 在启动时注入（工厂函数）
            self.build_session_fn = None
            self.initialized = True

    def init_builder(self, build_session_fn):
        """
        注入会话工厂。

        这里是核心解耦点：
        - SessionManager 不直接 import 具体 avatar 模块；
        - 需要创建会话时，只调用 build_session_fn(sessionid, params)。
        """
        self.build_session_fn = build_session_fn

    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """根据 id 获取会话实例；找不到返回 None。"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """判断会话是否存在且不为 None（创建中占位时会是 None）。"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None

    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        在异步上下文里创建新会话。

        执行流程：
        1) 检查 builder 是否已注入；
        2) 若没有 sessionid 就生成 uuid；
        3) 先塞一个 None 占位，防止并发重复创建；
        4) 用 run_in_executor 在默认线程池里执行 build_session_fn（CPU/模型加载）；
        5) 将完成后的实例写回 sessions 并返回 sessionid。

        重要：
        - 这里是 await 调用，在 build_session_fn 内抛错会回到上层 await caller（不会吞掉）；
        - 目前未在这里清理 None 占位，调用方需要按需补偿 remove_session。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")

        if sessionid is None:
            sessionid = _rand_session_id()

        logger.info('Creating sessionid=%s, current session num=%d', sessionid, len(self.sessions))
        # 并发保护：先放占位，避免两个并发请求同时创建同一个 id。
        self.sessions[sessionid] = None

        # 在事件循环中开线程执行，避免阻塞 aiohttp 的处理协程；
        # 异常会直接从 run_in_executor 的 await 上抛回调用链。
        avatar_session = await asyncio.get_event_loop().run_in_executor(
            None, self.build_session_fn, sessionid, params
        )
        self.sessions[sessionid] = avatar_session
        return sessionid

    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """
        直接挂接一个已构建会话（用于非标准入口，例如 virtualcam/rtmp 的主线程启动）。
        """
        self.sessions[sessionid] = avatar_session

    def remove_session(self, sessionid: str):
        """
        从管理表里移除会话。

        当前只做字典移除；更完整的清理（停止线程、关闭资源）可在这里扩展。
        """
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            self.sessions.pop(sessionid, None)


# 全局单例，其他模块通过 from server.session_manager import session_manager 引用。
session_manager = SessionManager()

