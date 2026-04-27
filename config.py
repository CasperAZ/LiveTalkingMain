###############################################################################
# 配置解析（Config）
#
# 这个文件只做一件事：
#   把命令行参数解析成 opt 对象，交给 app.py 和各个 avatar/ASR/TTS 模块。
#
# 你可以把它理解成 Go 的 config struct：定义“系统启动时的开关”，
# 大多数参数在服务运行期间几乎不会再变更。
###############################################################################

import argparse
import json


def str_or_int(value):
    """尝试把字符串转成 int，失败则保留原字符串。"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数，返回统一的运行时配置对象。"""
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 实时链路时序参数（fps / ASR context）────────────────────────────
    # fps 决定“每秒画面帧数”，也决定音频 chunk 大小（base_avatar 内 chunk = 16000 // (fps*2)）。
    # 音频 chunk 里每一帧默认按 20ms 计算，因此 fps=25 时 chunk=320 样本。
    # l/r 是 ASR 左右上下文窗口大小（按“约20ms音频帧”为单位），影响识别延迟/准确性。
    # m 在当前主干中保留但未实际消费（legacy 参数，兼容旧脚本）。
    parser.add_argument('--fps', type=int, default=25,
                        help="video fps (also controls audio chunk size); current pipeline expects 25")
    parser.add_argument('-l', type=int, default=10,
                        help="left context size for ASR feature window (in ~20ms audio frames)")
    parser.add_argument('-m', type=int, default=8,
                        help="legacy reserved arg; currently unused by runtime")
    parser.add_argument('-r', type=int, default=10,
                        help="right context size for ASR feature window (in ~20ms audio frames)")

    # ─── GUI 参数（暂时注释）──────────────────────────────────────────
    # 当前默认走前后端 HTTP/WebRTC，不需要从命令行传 W/H；代码保留是为了兼容历史入口。
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型参数（主模型选择）────────────────────────────────
    # model: avatar 推理链路
    #   - musetalk / wav2lip / ultralight / wav2lipls
    # avatar_id: 对应 data/avatars/<avatar_id> 下的素材目录
    # batch_size: 推理 batch，值越大吞吐越高、显存压力越高
    # modelres: 模型输入分辨率（如 160/192/256/384，当前部分链路未生效）
    # modelfile: 给 wav2lip/wav2lipls 直接指定 ckpt 路径文件名（如 --modelfile my.pth）
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 自定义动作编排──────────────────────────────────────────────
    # customvideo_config 是 JSON 文件路径，文件内容在 BaseAvatar.__loadcustom 里读（支持预先定义动作序列）。
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS 参数──────────────────────────────────────────────────
    # REF_FILE / REF_TEXT：当前会话参考音色参数，部分 TTS 实现会使用到它们。
    # 不同 TTS 插件会解释方式不同（EdgeTTS 用 voice，gpt-sovits 通常组合 text+voice）。
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="reference audio / voice id")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')

    # ─── 输入输出与网络──────────────────────────────────────────────
    # transport 决定服务出口：
    #   webrtc / rtcpush / rtmp / virtualcam
    # max_session 是会话上限；listenport 是 HTTP 控制面端口。
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    opt = parser.parse_args()

    # 解析 customvideo_config：把 JSON 文件内容注入到 opt.customopt，
    # 供 BaseAvatar.__loadcustom 使用，形成会话可复用的“动作模板”。
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r', encoding='utf-8') as f:
            opt.customopt = json.load(f)

    return opt

