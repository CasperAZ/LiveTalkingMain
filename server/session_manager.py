###############################################################################
#  全局会话管理器 (Session Manager)
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar

def _rand_session_id() -> str:
    """生成 UUID session ID"""
    return str(uuid.uuid4())

class SessionManager:
    """
    全局数字人会话管理器。
    
    统一管理 avatar_sessions 生命周期，并在脱离 WebRTC 时依然保持服务可用。
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.sessions: Dict[str, BaseAvatar] = {}
            self.build_session_fn = None
            self.initialized = True

    def init_builder(self, build_session_fn):
        """配置用于构建 avatar_session 的工厂函数"""
        # SessionManager 只负责“管理”，不负责“知道怎么造”。
        # 真正造会话的逻辑从外部注入，方便解耦。
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """获取已存活的会话"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """检查会话是否存在"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None
        
    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        在异步环境中创建一个新会话
        如果 sessionid 为 None，则自动生成。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")
            
        if sessionid is None:
            sessionid = _rand_session_id()
            
        logger.info('Creating sessionid=%s, current session num=%d', sessionid, len(self.sessions))
        # 先占位，避免并发情况下重复创建同一个 session。
        self.sessions[sessionid] = None

        # 构建会话可能会做很多重活，比如加载素材、初始化 TTS/ASR 等，
        # 所以丢到线程池里执行，避免阻塞 aiohttp 的事件循环。
        avatar_session = await asyncio.get_event_loop().run_in_executor(
            None, self.build_session_fn, sessionid, params
        )
        self.sessions[sessionid] = avatar_session
        return sessionid
        
    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """同步添加静态或外部管理的会话（供非服务端入口调用）"""
        self.sessions[sessionid] = avatar_session
        
    def remove_session(self, sessionid: str):
        """销毁会话资源"""
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            # 当前版本先把引用从管理表中移除。
            # 如果后续你想做更完整的资源回收，可以在这里继续补：
            # - 停线程
            # - 关输出
            # - 清临时文件
            self.sessions.pop(sessionid, None)

# 单例抛出
session_manager = SessionManager()
