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

import time
import numpy as np

import queue
from queue import Queue
from numpy.typing import NDArray
import torch.multiprocessing as mp

from avatars.base_avatar import BaseAvatar,AudioFrameData


class BaseASR:
    """
    ASR/音频特征层的统一基类（这里的 ASR 更偏“音频驱动与特征切片”，不只是语音识别）。

    你可以把它理解成“音频节拍器 + 特征窗口管理器”：
    1) 接收上游音频（TTS 合成音频、用户上传音频、自定义动作音频）。
    2) 按固定节拍切成小块（chunk），维持实时流水线稳定推进。
    3) 维护左右上下文窗口（l/r），把连续特征切成和视频帧对齐的小片段。
    4) 把“可用于模型推理”的特征批次放入 feat_queue，交给口型推理线程消费。
    """
    def __init__(self, opt, parent:BaseAvatar = None):
        self.opt = opt
        self.parent = parent

        # fps 是全链路节拍参数（视频主时钟），不是“只管视频不管音频”。
        self.fps = opt.fps
        # 采样率单位是 Hz（每秒采样点数）。16000 Hz = 每秒 16000 个采样点。
        self.sample_rate = 16000
        # 关键公式：
        # chunk_samples = sample_rate // (fps * 2)
        #
        # 为什么是 fps * 2：
        # - 当前默认视频 25fps -> 每帧 40ms
        # - 工程里让“每个视频帧对应 2 个音频块”
        # - 所以每个音频块是 20ms
        # - 20ms * 16000 = 320 采样点
        self.chunk = self.sample_rate // (opt.fps*2)

        # 输入音频队列：上游喂进来的 AudioFrameData 先放这里。
        self.queue:Queue[AudioFrameData] = Queue()
        # 输出音频队列：与特征同步后的音频帧，供渲染/输出线程发送。
        self.output_queue:Queue[AudioFrameData] = Queue()

        self.batch_size = opt.batch_size

        # 连续音频帧缓存（滑动窗口底座），用于构建特征上下文。
        self.frames: list[NDArray[np.float32]] = []
        # 左/右上下文长度（单位：音频帧，默认约 20ms 一帧）。
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        # 特征队列：子类 run_step() 提取好的特征批次放进去，推理线程会消费。
        self.feat_queue = Queue(maxsize=2)

        #self.warm_up()

    def flush_talk(self):
        # 清掉还没消费的音频块，用于中断当前播报。
        self.queue.queue.clear()

    def put_audio_frame(self,audio_chunk:NDArray[np.float32],datainfo:dict): #16khz 20ms pcm
        # type=0 表示正常说话音频。
        self.queue.put(AudioFrameData(data=audio_chunk,type=0,userdata=datainfo))

    #return frame:audio pcm; type: 0-normal speak, 1-silence; eventpoint:custom event sync with audio
    def get_audio_frame(self)->AudioFrameData:        
        try:
            if self.parent and self.parent.custom_audiotype>1: #播放自定义音频,优先播放完自定义动作,可以通过interrupt打断动作播放
                # 自定义动作优先级更高，会直接覆盖普通 TTS 音频输入。
                frame = self.parent.get_custom_audio_stream(self.parent.custom_audiotype)
                type = self.parent.custom_audiotype
                return AudioFrameData(data=frame, type=type, userdata={})
            else:
                frame = self.queue.get(block=True,timeout=0.01)
                return frame
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            # 上游没音频时，补静音帧，保持节拍连续。
            frame = np.zeros(self.chunk, dtype=np.float32)
            return AudioFrameData(data=frame, type=1, userdata={})


    #return frame:audio pcm; type: 0-normal speak, 1-silence; eventpoint:custom event sync with audio
    def get_audio_out(self)->AudioFrameData: 
        return self.output_queue.get()
    
    def warm_up(self):
        # 预热阶段先填充 (left + right) 个音频块，避免刚启动时窗口不完整。
        # 然后再弹出 left 个，让“当前时间点”位于窗口中间附近。
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame=self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        pass

    def get_next_feat(self,block,timeout):        
        return self.feat_queue.get(block,timeout)

    #分割音频特征，子类调用
    def _get_sliced_feature(self, feature_array, 
                        vid_idx,  
                        audio_feat_win,  
                        feature_idx_multiplier=1.0):
        """
        按“视频帧索引”从长音频特征序列中切一段窗口。

        参数说明：
        - feature_array: 整段音频的特征序列（例如 mel / whisper 特征）。
        - vid_idx:       目标视频帧在 batch 内的帧号。
        - audio_feat_win:窗口大小 [left_win, right_win]，单位是“视频帧数语义”。
        - feature_idx_multiplier:
            由于“音频特征步长”和“视频帧步长”通常不同，
            需要这个倍率把视频索引映射到特征索引。

        返回：
        - selected_feature: 切出来的特征窗口（numpy array）
        - selected_idx:     实际使用到的特征索引（便于调试对齐）
        """
        length = feature_array.shape[0] #len(feature_array)
        selected_feature = []
        selected_idx = []
        
        # 因为音频特征和视频帧的时间分辨率不同，这里要先做索引换算。
        center_idx = int(vid_idx * feature_idx_multiplier) 
        left = int(center_idx - audio_feat_win[0]*feature_idx_multiplier)
        right = int(center_idx + audio_feat_win[1]*feature_idx_multiplier)
        # pad_left = 0
        # pad_right = 0
        # if left < 0:
        #     pad_left = -left
        #     left = 0
        # if right > feature_array.shape[0]:
        #     pad_right = right - feature_array.shape[0]
        #     right = feature_array.shape[0]
        # auds = feature_array[left:right]
        # if pad_left > 0:
        #     auds = np.concatenate([feature_array[left]*pad_left, auds], axis=0)
        # if pad_right > 0:
        #     auds = np.concatenate([auds, feature_array[right-1]*pad_right], axis=0) # [8, 16]
        
        for idx in range(left,right):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        # selected_feature = np.concatenate(selected_feature, axis=0)
        # selected_feature = selected_feature.reshape(-1, 256)# 20*256
        return np.asarray(selected_feature),selected_idx

    # 参数定义
    def _feature2chunks(self,feature_array,batch_size,audio_feat_win=[8,8],start=0,feature_idx_multiplier=1.0):
        """
        把连续特征序列切成 batch 份“逐帧特征窗口”。

        常见用途：
        - 当前循环要推理 batch_size 个视频帧，
          那就为这 batch_size 个帧各切一份特征窗口。

        参数说明：
        - start: 这一批帧的起始视频索引（常取 stride_left_size/2 一类偏移）。
        - audio_feat_win: 每个视频帧需要看多大左右音频上下文。
        """
        # 把长特征序列切成与视频帧对齐的小窗口。
        feature_chunks = []
        #start += 10
        #feature_idx_multiplier = 50./fps 
        for i in range(batch_size):
            # start_idx = int(i * whisper_idx_multiplier)
            # if start_idx>=len(feature_array):
            #     break
            selected_feature,selected_idx = self._get_sliced_feature(
                feature_array=feature_array, vid_idx=i+start,
                audio_feat_win=audio_feat_win, feature_idx_multiplier=feature_idx_multiplier)
            #print(f"i:{i},selected_idx {selected_idx},feature_idx_multiplier:{feature_idx_multiplier}")
            feature_chunks.append(selected_feature)
        return feature_chunks
