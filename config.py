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
    # 这一组参数控制“选哪个模型 + 选哪套素材 + 推理吞吐/预热行为”。
    # 对入门同学可以先这样理解：
    # 1) model：决定主推理链路（musetalk / wav2lip / ultralight）。
    # 2) avatar_id：决定从 data/avatars/<avatar_id>/ 读取哪套人物素材。
    # 3) batch_size：决定一次并行推理多少帧（吞吐和显存占用强相关）。
    # 4) modelres / modelfile：属于“可配置入口参数”，但当前主流程中尚未完全接线。
    #
    # model（模型类型）
    # - musetalk：通常口型更自然，但链路更重，对显存/算力要求更高。
    # - wav2lip：工程成熟度高、兼容性好，很多场景优先用它。
    # - ultralight：资源占用更轻，适合算力有限场景。
    # 注意：这里只是“选择代码分支”，真正能否跑通还取决于对应模型权重和 avatar 资源是否齐全。
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")

    # avatar_id（素材目录 ID）
    # - 例如默认值 wav2lip256_avatar1，对应目录：data/avatars/wav2lip256_avatar1/
    # - 目录里一般包含 full_imgs / face_imgs / coords.pkl（不同模型会有少量差异）
    # - 如果 avatar_id 写错、目录缺文件，启动后通常会在加载头像阶段报错。
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")

    # batch_size（推理批大小）
    # - 含义：一次送进模型并行处理的样本数（通常可理解为“并行帧数”）。
    # - 默认 16：在速度与显存之间做的折中。
    # - 调大：吞吐可能更高，但显存压力上升，可能 OOM（显存不足）。
    # - 调小：更稳、更省显存，但吞吐下降，端到端延迟可能上升。
    # - 实操建议：先从 8/16 起步，再按显存余量逐步上调。
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    # modelres（模型分辨率参数，当前版本主流程未完整使用）
    # - 设计意图：控制模型输入/预热时使用的分辨率（常见是方形边长，如 160/192/256）。
    # - 现实状态：当前 app 主流程里 warm_up 分辨率是按模型分支写死的（例如 wav2lip=256），
    #   不是直接读取这个参数；所以你改这里，通常不会立即改变实际推理分辨率行为。
    # - 保留原因：便于后续把分辨率配置“接线”到统一入口。
    parser.add_argument('--modelres', type=int, default=192)

    # modelfile（自定义模型权重路径，当前版本主流程未完整使用）
    # - 设计意图：允许通过命令行指定权重文件，例如：
    #   --modelfile ./models/your_wav2lip.pth
    # - 现实状态：当前主流程（尤其 wav2lip 分支）仍使用固定路径加载权重，
    #   因此这个参数现在更像“预留扩展位”。
    # - 对学习阶段的建议：先用项目默认权重跑通，再考虑把该参数接入主流程。
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
