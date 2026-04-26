###############################################################################
#  配置解析
#
#  这个文件把命令行参数整理成统一的 `opt` 配置对象。
#  几乎所有模块都会依赖它：
#  - app.py 用它决定启动哪条主链路；
#  - avatar / ASR / TTS / output 都会读取其中的运行参数；
#  - 后续如果接快手/抖音/视频号，平台级配置也很适合从这里进入。
###############################################################################

import argparse
import json
import os


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 时序/特征窗口参数 ───────────────────────────────────────────────
    # 这里不只是“音频提取参数”，而是整个实时链路的节拍参数：
    # 1) fps 决定视频主时钟，也间接决定每个音频块时长。
    #    在当前实现中：chunk_samples = 16000 / (fps * 2)
    #    当 fps=25 时，每个音频块是 320 采样点（约 20ms）。
    # 2) l/r 是 ASR 特征窗口的左右上下文长度（单位：音频帧，约 20ms/帧）。
    #    可理解为围绕当前时刻的滑动窗口：窗口总长度约为 (l + r) 帧。
    # 3) m 是历史遗留参数（legacy）。在当前代码里没有被实际消费，
    #    保留它主要是为了兼容旧脚本/旧命令行，不参与当前推理逻辑。
    parser.add_argument('--fps', type=int, default=25,
                        help="video fps (also controls audio chunk size); current pipeline expects 25")
    parser.add_argument('-l', type=int, default=10,
                        help="left context size for ASR feature window (in ~20ms audio frames)")
    parser.add_argument('-m', type=int, default=8,
                        help="legacy reserved arg; currently unused by runtime")
    parser.add_argument('-r', type=int, default=10,
                        help="right context size for ASR feature window (in ~20ms audio frames)")

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型参数 ────────────────────────────────────────────────
    # model 决定走哪套口型推理链路。
    # avatar_id 对应 data/avatars/ 下的一套素材目录。
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 自定义动作/静默动画配置 ───────────────────────────────────────
    # customvideo_config 对应一个 JSON 文件，用来定义静默时的动作编排。
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS 参数 ──────────────────────────────────────────────────────
    # REF_FILE / REF_TEXT 的语义由具体 TTS 插件决定。
    # 例如：
    # - EdgeTTS 中 REF_FILE 常当 voice 名称使用；
    # - GPT-SoVITS 中 REF_FILE/REF_TEXT 常当参考音频和参考文本使用。
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')

    # ─── 输出/传输参数 ─────────────────────────────────────────────────
    # transport 决定最后结果发往哪里：
    # - webrtc: 浏览器实时播放
    # - rtcpush: 主动推 WebRTC 到远端
    # - rtmp: 推流到直播平台或中转服务器
    # - virtualcam: 输出到本机虚拟摄像头
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    opt = parser.parse_args()

    # ─── 配置后处理 ────────────────────────────────────────────────────
    # 这里把 JSON 文件读成 Python 对象，方便后续 BaseAvatar 直接使用。
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
