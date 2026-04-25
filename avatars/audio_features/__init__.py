# audio_features 子包负责“音频 -> 模型可用特征”的转换。
# 不同数字人模型依赖的特征类型不同：
# - Wav2Lip 常用 Mel
# - MuseTalk 常用 Whisper 特征
# - UltraLight 常用 HuBERT 特征
