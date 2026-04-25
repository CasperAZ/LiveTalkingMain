###############################################################################
#  插件注册表
#
#  这是项目实现“可插拔架构”的关键文件。
#  可以把它理解成一个简化版插件工厂：
#  - `@register(...)` 负责登记“名字 -> 类”
#  - `create(...)` 负责按名字创建实例
#
#  这也是你后续扩展平台能力时最值得保留的设计之一：
#  新增一个 TTS、一个输出器、甚至一个新数字人模型，不需要大改主流程。
###############################################################################

from typing import Dict, Type, Any
from utils.logger import logger

_REGISTRY: Dict[str, Dict[str, Type]] = {
    "stt": {},
    "llm": {},
    "tts": {},
    "avatar": {},
    "output": {},
}


def register(category: str, name: str):
    """
    装饰器：注册插件类到全局注册表。

    用法::

        @register("tts", "edgetts")
        class EdgeTTS(BaseTTS): ...
    """
    def decorator(cls):
        # category 例如 tts / avatar / output。
        # name 是配置里真正填写的插件名。
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        _REGISTRY[category][name] = cls
        logger.info(f"Registered {category}/{name}: {cls.__name__}")
        return cls
    return decorator


def create(category: str, name: str, **kwargs) -> Any:
    """
    按名称创建插件实例。

    Usage::

        tts = registry.create("tts", "edgetts", opt=opt)
    """
    # 如果这里报“Plugin not found”，常见原因有两个：
    # 1. 对应模块还没 import，装饰器还没执行；
    # 2. 配置名和 @register 里的名字不一致。
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list(_REGISTRY.get(category, {}).keys())
        raise ValueError(
            f"Plugin '{name}' not found in category '{category}'. "
            f"Available: {available}"
        )
    cls = _REGISTRY[category][name]
    return cls(**kwargs)


def list_plugins(category: str = None) -> Dict[str, list]:
    """列出已注册的插件"""
    if category:
        return {category: list(_REGISTRY.get(category, {}).keys())}
    return {cat: list(plugins.keys()) for cat, plugins in _REGISTRY.items()}
