###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Avatar 基类
#
#  这是每一路“数字人会话”的总控骨架。可以把它理解成一个流水线调度器：
#  文本/音频输入 -> TTS/ASR 特征 -> 口型模型推理 -> 回贴到底图 -> 输出到 WebRTC/RTMP/虚拟摄像头。
#  以后你要做快手适配时，这个文件非常关键，因为它决定了：
#  1. 输入是怎样进入数字人会话的；
#  2. 会话内部怎样并行跑 TTS、推理和输出；
#  3. 最终媒体帧怎样统一发给不同输出通道。
#

import math
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import resampy
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import read_imgs,mirror_index

###############################################################################
# 这份文件建议按“运行时总线”来看：  
# - 输入层：收集文本/音频，交给 TTS 与 ASR；  
# - 推理层：调用子类实现的模型推理；  
# - 渲染层：把推理结果和素材合成一帧，走统一输出接口；  
# - 输出层：按 transport 决定 webrtc/rtmp/virtualcam 的接收方。  
###############################################################################

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

@dataclass
class AudioFrameData:
    # 一个 audio_frame = 1 个“音频小块”对应的标准结构（Go 可以理解为一个 struct）。
    # - data: float32 音频样本数组  
    # - type: 0 正常、1 静音、>1 自定义动作标签
    # - userdata: 每块音频附带事件元数据，比如 {"status": "start"/"end"}
    # data:
    #   一个 20ms 左右的音频块，通常是 float32 单声道 PCM。
    # type:
    #   0 = 正常说话音频
    #   1 = 静音占位音频
    #   >1 = 自定义动作音频
    # userdata:
    #   附带的业务数据，比如 start/end、文本内容、TTS 参数等。
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)

class BaseAvatar:
    """
    数字人会话核心基类（会话级总控）。

    这一层不直接绑定某一个具体模型，而是定义“统一流水线协议”：
    1) 输入层：文本输入 / 音频输入（TTS、上传文件、自定义动作音频）。
    2) 特征层：ASR/特征提取（mel、whisper、hubert 等）。
    3) 推理层：子类实现 inference_batch() 产出口型区域结果。
    4) 画面层：子类实现 paste_back_frame() 回贴到原始底图。
    5) 输出层：统一推给 streamout（WebRTC/RTMP/虚拟摄像头）。

    线程模型（关键）：
    - 线程A: TTS（把文本变成音频块）
    - 线程B: inference（吃特征 -> 产出预测帧）
    - 线程C: process_frames（后处理画面 + 音频/视频统一输出）
    """
    def __init__(self, opt):
        # BaseAvatar 的关键职责：统一管理一个会话的音视频推理链路，而不绑定某一个具体模型。
        self.opt = opt
        # 音频采样率，单位 Hz。16000 表示每秒 16000 个采样点。
        self.sample_rate = 16000
        # 音频分块长度（采样点数）：
        # chunk = sample_rate // (fps * 2)
        #
        # 设计动机：
        # - 视频 1 帧时长大约对应 2 个音频块（默认 25fps -> 40ms/每帧）  
        # - 这样 ASR 特征和口型推理天然对齐到视频帧节奏
        # - 默认 25fps 时，一个视频帧周期是 40ms；
        # - 系统按“1个视频帧对应2个音频块”推进；
        # - 所以每个音频块约 20ms -> 16k * 0.02 = 320 点。
        self.chunk = self.sample_rate // (opt.fps*2)
        self.sessionid = self.opt.sessionid

        self.speaking = False
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        # 自定义动作状态：
        # 0 表示正常说话驱动；
        # 1 表示静音占位；
        # >1 表示某种业务自定义动作编号。
        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        # 对每种自定义动作，缓存图片序列/音频序列和播放进度索引，切换动作时只改状态机即可。
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        # self.custom_opt = {}
        self.__loadcustom()

        # 推理结果队列：线程B(inference) -> 线程C(process_frames) 的桥梁。
        # 队列元素结构：(pred_frame_or_none, [audio_frame_1, audio_frame_2], frame_idx)
        # 结构里固定 2 个音频帧 = 1 个“画面时隙”，batch_size 决定一次处理批次大小。
        self.batch_size = opt.batch_size
        self.res_frame_queue = Queue(self.batch_size*2)
        self.render_event = Event()

        # TTS 插件映射：通过 registry 反射装配，可在配置里切换，不改会话主逻辑。
        _tts_modules = {
            'edgetts': 'tts.edge',
            'gpt-sovits': 'tts.sovits',
            'xtts': 'tts.xtts',
            'cosyvoice': 'tts.cosyvoice',
            'fishtts': 'tts.fish',
            'tencent': 'tts.tencent',
            'doubao': 'tts.doubao',
            'indextts2': 'tts.indextts2',
            'azuretts': 'tts.azure',
            'qwentts': 'tts.qwentts'
        }

        if opt.tts in _tts_modules:
            # 先 import，再 create，是为了触发对应模块里的 @register 装饰器。
            importlib.import_module(_tts_modules[opt.tts])
            self.tts = registry.create("tts", opt.tts, opt=opt, parent=self)
        else:
            logger.error(f"TTS module {opt.tts} not found.")

        # 输出插件映射：把“推送方式”从配置里解耦，不同 transport 走不同实现。
        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # Output 决定最终把媒体流送到哪里。
        # 这一层是平台适配时非常适合扩展的地方。
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    # 统一的“喂文本”入口。
    # 外部通常不需要知道底层是如何经过 TTS / ASR / 推理的。
    # put_msg_txt: 文本入口，驱动 TTS 产出音频帧；上层不需要关心后续走哪种 TTS。
    def put_msg_txt(self, msg, datainfo:dict={}):
        if hasattr(self, 'tts'):
            self.tts.put_msg_txt(msg, datainfo)
    
    # put_audio_frame: 外部任意来源的音频块入口（16kHz、float32，通常 20ms）。
    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict={}): # 16khz 20ms pcm
        # 统一的“喂音频块”入口。
        # 不管音频来自 TTS、上传文件还是自定义动作，最后都走这里。
        if hasattr(self, 'asr'):
            self.asr.put_audio_frame(audio_chunk, datainfo)

    # put_audio_file: 上传文件字节数组 -> decode -> 分块 -> 入 ASR。
    def put_audio_file(self, filebyte, datainfo:dict={}): 
        # 上传整段音频文件时，流程是：
        # 文件字节 -> 解码成波形 -> 重采样/归一化 -> 按 chunk 切块 -> 逐块入队。
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk

    # put_audio_filepath: 按文件路径读取同上，不必先手动转成 bytes。
    def put_audio_filepath(self, filepath, datainfo:dict={}): 
        # 与 put_audio_file 逻辑相同，只是输入改成了文件路径。
        stream = self.__create_bytes_stream(filepath)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
    
    # 统一音频预处理：decode + 单声道 + 重采样，返回 float32 ndarray。
    def __create_bytes_stream(self, byte_stream):
        # 标准化音频输入规格：
        # - 统一转成 float32
        # - 多声道只取第一轨
        # - 统一重采样到 16k
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    # flush_talk: 打断当前会话片段，清掉未播/未推队列，重置动作状态。
    def flush_talk(self):
        # 打断说话时，需要同时清掉：
        # 1. TTS 还没合成的文本；
        # 2. ASR 还没消费的音频块；
        # 3. 自定义动作状态。
        if hasattr(self, 'tts') and hasattr(self.tts, 'flush_talk'):
            self.tts.flush_talk()
        if hasattr(self, 'asr') and hasattr(self.asr, 'flush_talk'):
            self.asr.flush_talk()
        self.custom_audiotype = 0  

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    # __loadcustom: 加载配置里的 customopt（动作分支）资源索引。
    def __loadcustom(self):
        if not hasattr(self.opt, 'customopt') or not self.opt.customopt:
            return
        # customopt 描述“静默或特定状态下播放什么动画/音频”。
        # 典型结构（每项）：
        # {
        #   "audiotype": 2,
        #   "imgpath": "xxx/frames",
        #   "audiopath": "xxx/audio.wav"
        # }
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            # self.custom_opt[item['audiotype']] = item

    # init_customindex: 每次会话启动或重置动作时，把自定义序列索引归零。
    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    # notify: 当前主要处理 frame event 的状态点（start/end），便于日志和埋点。
    def notify(self, eventpoint:dict):
        # 这里可以理解成“音频事件同步点”。
        # 当前只是打日志，但很适合后续扩展成字幕、状态回调、平台通知等。
        if eventpoint and eventpoint.get('status'):
            logger.info("notify:%s", eventpoint)

    # start_recording: 调试能力，启动本地 ffmpeg 管道记录 raw 音/视频。
    def start_recording(self):
        # 录制功能是通过两个 ffmpeg 进程分别写视频裸流和音频裸流实现的。
        if self.recording:
            return
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio)

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    # get_custom_audio_stream: 为某个自定义动作取一段 chunk 音频并推进指针。
    def get_custom_audio_stream(self, audiotype):
        # 每次只取出一个 chunk 长度的音频，保证自定义动作也遵守同一节拍。
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    # set_custom_state: 动作编排入口，切换到指定 audiotype 并可重置播放进度。
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        # 切换到某个动作状态，本质上就是切换后续静音阶段所使用的音频/图像序列。
        self.custom_audiotype = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    # get_avatar_length: 兼容不同 avatar 素材，没有循环素材时给 1 防止除零/越界。
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1
        
    # inference: 后台推理线程。取 ASR 特征 + 音频帧对，生成推理结果写入 res_frame_queue。
    def inference(self, quit_event):
        # 这个线程负责“音频特征 -> 口型推理结果”。
        length = self.get_avatar_length()
        index = 0
        count = 0
        counttime = 0
        last_speaking = False

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            starttime = time.perf_counter()
            audiofeat_batch = []
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            # 这一批次是否全静音：用于决定是否跳过模型推理（提升吞吐）。
            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            # 关键对齐策略：
            # - 每个视频帧对应 2 个音频块
            # - 当前 batch 推理 batch_size 个视频帧
            # - 因此需要准备 batch_size * 2 个音频块
            for _ in range(self.batch_size * 2):
                audioframe:AudioFrameData = self.asr.output_queue.get()
                if audioframe.type == 0:
                    is_all_silence = False               
                audio_frames.append(audioframe)

             # 检测状态变化
            current_speaking = not is_all_silence

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                # 没有说话时不跑模型，直接走静默帧/原始帧逻辑。
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    # i*2:i*2+2：把两块音频打包给当前这个视频帧
                    self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], idx))
                    index = index + 1
            else:
                # 只有检测到非静音时，才真的调用子类模型做推理。
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                t = time.perf_counter()

                pred = self.inference_batch(index, audiofeat_batch)

                counttime += (time.perf_counter() - t)
                count += self.batch_size
                if count >= 100:
                    logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(pred):
                    # 推理结果帧 + 对应两块音频，一起入队给后处理线程
                    self.res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2], mirror_index(length, index)))
                    index = index + 1
                    
            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking         
        logger.info('baseavatar inference thread stop')

    # process_frames: 消费推理队列，把音画数据同步到 output（webrtc/rtmp/virtualcam）。
    def process_frames(self,quit_event):
        # 这个线程负责把推理结果做成“最终可播放的完整画面”，然后送去输出层。
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        _last_speaking = False
        _transition_start = time.time()
        if enable_transition:
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存

        self.output.start()
        
        while not quit_event.is_set():
            try:
                audio_frames: list[AudioFrameData]
                res_frame,audio_frames,idx = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            if current_speaking != _last_speaking:
                logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
            _last_speaking = current_speaking

            # audio_frames 长度固定是 2（和“1视频帧=2音频块”策略一致）。
            if audio_frames[0].type!=0 and audio_frames[1].type!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    # 子类会实现 paste_back_frame，把预测出来的嘴部区域贴回到底图。
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            
            # 所有输出方式都统一走 output 接口，这样主流程不关心“发到哪里”。
            self.output.push_video_frame(combine_frame)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                #frame,type,eventpoint = audio_frame
                frame = (audio_frame.data * 32767).astype(np.int16)

                # 音频帧与视频帧在同一层往外推，便于保持时序一致。
                self.output.push_audio_frame(frame, audio_frame.userdata)
                self.record_audio_data(frame)
                
            # if self.opt.transport == 'virtualcam' and hasattr(self.output, '_cam') and self.output._cam:
            #     self.output._cam.sleep_until_next_frame()

        self.output.stop()
        logger.info('baseavatar process_frames thread stop') 

    # render: 会话启动入口；启动 TTS、inference、process_frames 三条执行链并做背压控制。
    def render(self,quit_event):
        # render 是会话真正启动的入口，会同时拉起：
        # - TTS 线程
        # - inference 线程
        # - process_frames 线程
        self.quit_event = quit_event
        
        self.init_customindex()
        self.tts.render(quit_event)

        infer_quit_event = mp.Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0
        while not quit_event.is_set(): 
            t = time.perf_counter()
            # run_step() 每次推进一点音频驱动流程，相当于整个会话的节拍器。
            # 子类 ASR 会在这里：
            # 1) 从输入队列取音频块；
            # 2) 维护上下文窗口；
            # 3) 产出可供推理线程消费的特征 batch。
            self.asr.run_step()

            buffer_size = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0
            if buffer_size >= 5:
                # 下游积压时主动降速，避免延迟持续放大。
                logger.debug('sleep qsize=%d', buffer_size)
                time.sleep(0.04 * buffer_size * 0.8)
        logger.info('baseavatar render thread stop')

        infer_quit_event.set()
        infer_thread.join()

        process_quit_event.set()
        process_thread.join()

