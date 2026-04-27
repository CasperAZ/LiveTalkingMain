###############################################################################
# 注册中心（Registry）
#
# 作用：把“模块名 -> 实现类”注册到全局表，运行时按名字创建实例。
# 这在项目里用得很广，比如：
# - avatar 插件：musetalk / wav2lip / ultralight / wav2lipls
# - tts 插件：edgetts / qwen tts / ... 等
# - output 插件：webrtc / rtmp / virtualcam
#
# 设计动机：解耦依赖。你新增插件时只改配置，不改主业务流程。
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
    注册装饰器。

    使用方式：

    @register("tts", "edgetts")
    class EdgeTTS(BaseTTS): ...

    运行后会记录到 _REGISTRY["tts"]["edgetts"]。
    """
    def decorator(cls):
        # category 例如：tts / avatar / output
        # name 是业务层统一使用的字符串键，例如 "webrtc"、"wav2lip"。
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        _REGISTRY[category][name] = cls
        logger.info(f"Registered {category}/{name}: {cls.__name__}")
        return cls

    return decorator


def create(category: str, name: str, **kwargs) -> Any:
    """
    按类别和名称创建实例。

    - category：要创建的分组（"tts"/"avatar"/"output"...）
    - name：映射键（"edgetts"、"wav2lip"、"webrtc"...）
    - kwargs：透传给构造函数（通常是 opt / parent / model 等）
    """
    # 常见失败原因：
    # 1) 没 import 到对应模块（类未注册）
    # 2) 类名写错，和 register 的 name 不一致
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list(_REGISTRY.get(category, {}).keys())
        raise ValueError(
            f"Plugin '{name}' not found in category '{category}'. "
            f"Available: {available}"
        )

    cls = _REGISTRY[category][name]
    return cls(**kwargs)


def list_plugins(category: str = None) -> Dict[str, list]:
    """
    查询当前注册表。用于调试或管理后台展示可用插件。
    """
    if category:
        return {category: list(_REGISTRY.get(category, {}).keys())}
    return {cat: list(plugins.keys()) for cat, plugins in _REGISTRY.items()}
