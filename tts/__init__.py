# TTS 子包只导出最基础的抽象类和状态枚举。
# 具体实现（EdgeTTS、GPT-SoVITS、QwenTTS 等）在各自模块中通过 registry 注册。
from .base_tts import BaseTTS, State
