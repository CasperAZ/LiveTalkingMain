###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku.foxmail.com
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
#  Wav2LipLS 口型模型实现（继承 BaseAvatar）
#
#  这份文件是整个项目中“数字人会话”在 wav2lipls 方案下的具体实现：
#  - 加载 wav2lipls 模型
#  - 加载 avatar 素材（底图、嘴部图、坐标）
#  - 把音频特征交给模型推理
#  - 把推理结果回贴到底图并输出

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.hubert import HubertASR
import asyncio
from av import AudioFrame, VideoFrame

from avatars.base_avatar import BaseAvatar
from avatars.ultralight.audio2feature import Audio2Feature
from avatars.wav2lipls.models import Human

from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register


device = initialize_device()
logger.info('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    """
    加载 checkpoint 文件。
    GPU 下直接 torch.load，CPU 下加 map_location 避免跨设备失败。
    """
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path, modelres=192):
    """
    加载 wav2lipls 模型 + Hubert 特征提取器。

    返回:
    - model.eval() 后的 torch model
    - audio_processor: Hubert 特征处理器实例

    注意：
    - `modelres` 是模型输入边长（例如 192 / 384）
    """
    audio_processor = Audio2Feature()
    model = Human(sr=False, face_size=modelres)  # 初始化网络，face_size 用于输入尺寸
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        # 去掉并行训练时加入的前缀，避免 load_state_dict 失败
        new_s[k.replace('module.', '').replace('_orig_mod.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device, dtype=torch.float16)
    return model.eval(), audio_processor


def load_avatar(avatar_id: str, imgcache_num: int = 0):
    """
    读取 avatar 素材包：
    - full_imgs:  每帧底图
    - face_imgs:  每帧嘴部裁剪图
    - coords.pkl: 每帧坐标

    imgcache_num=0 表示直接把全部图片一次读到内存；
    >0 时走 ImgCache 缓存读取（避免大规模素材占满内存）。
    """
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"

    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)

    if imgcache_num == 0:
        input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        frame_list_cycle = read_imgs(input_img_list)

        input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        face_list_cycle = read_imgs(input_face_list)
    else:
        # 大量素材时可切换到磁盘缓存读取
        from imgcache import ImgCache
        frame_list_cycle = ImgCache(full_imgs_path, imgcache_num)
        face_list_cycle = ImgCache(face_imgs_path, imgcache_num)

    return frame_list_cycle, face_list_cycle, coord_list_cycle


@torch.no_grad()
def warm_up(batch_size, model, modelres):
    """
    推理预热：
    先用全 1 的伪输入跑一遍，触发 cudnn/cuda kernel 初始化，
    降低第一个真实请求时的耗时尖峰。
    """
    logger.info('warmup model...')
    human, _ = model
    weight_dtype = torch.float16
    hn = torch.zeros(2, batch_size, 512).to(device, dtype=weight_dtype)
    cn = torch.zeros(2, batch_size, 512).to(device, dtype=weight_dtype)
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device, dtype=weight_dtype)
    mel_batch = torch.ones(batch_size, 10, 1024).to(device, dtype=weight_dtype)
    human(mel_batch, img_batch, hn, cn)


def _correction(frame_to_save_, pred):
    """
    边缘修正：
    对生成的嘴部预测结果做过渡平滑，减少接缝抖动。
    """
    for j in range(20):
        pred[-1 - j] = (j / 20. * pred[-1 - j] + (20 - j) / 20. * frame_to_save_[-1 - j]).astype('uint8')
    for j in range(20):
        pred[:, 20 - j] = ((20 - j) / 20. * pred[:, 20 - j] + j / 20. * frame_to_save_[:, 20 - j]).astype('uint8')
    for j in range(20):
        pred[:, -1 - j] = (j / 20. * pred[:, -1 - j] + (20 - j) / 20. * frame_to_save_[:, -1 - j]).astype('uint8')

    # 绿色通道和蓝色通道差异较大位置，优先使用原底图像素，避免明显边缘伪影
    frame_to_save_bool = np.expand_dims(
        (frame_to_save_[:, :, 1].astype('int32') - frame_to_save_[:, :, 2].astype('int32')) > 60, -1)
    pred = pred * (1 - frame_to_save_bool) + frame_to_save_ * frame_to_save_bool

    return pred


@register("avatar", "wav2lipls")
class LipLsReal(BaseAvatar):
    """
    wav2lipls 会话类。
    BaseAvatar 已经提供了：
    - TTS 渲染入口
    - ASR 输入队列管理
    - 推理线程和输出线程编排

    本类只补齐三件事：
    - 绑定 wav2lipls + Hubert 特征提取器
    - 实现 inference_batch（批量嘴型推理）
    - 实现 paste_back_frame（把结果贴回底图）
    """

    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        # 从 app.py 注入的 model 打包参数里取回 (net, hubert_processor)
        self.model, self.audio_processor = model
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar

        # Wav2lipLS 使用 HubertASR 做音频特征，窗口设置 [0, 5]
        self.asr = HubertASR(opt, self, self.audio_processor, [0, 5])
        self.asr.warm_up()

    def inference_batch(self, index, audiofeat_batch):
        """
        关键推理函数：一次处理一个 batch 的视频帧。

        参数:
        - index: 这批开始的帧序号（配合 mirror_index 做循环取帧）
        - audiofeat_batch: 当前 batch 的音频特征（shape 与 batch_size 对齐）

        返回:
        - pred: 预测后的嘴部区域序列，形状通常 [batch, H, W, C]
        """
        length = len(self.face_list_cycle)
        weight_dtype = torch.float16
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)

        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        # 组装输入：前 3 通道是可见图 + 3 通道做占位 mask
        img_masked = img_batch.copy()
        img_masked[:, face.shape[0] // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
            device, dtype=weight_dtype, non_blocking=True
        )
        audiofeat_batch = torch.FloatTensor(audiofeat_batch).to(
            device, dtype=weight_dtype, non_blocking=True
        )

        with torch.no_grad():
            hn = torch.zeros(2, self.batch_size, 512).to(device, dtype=weight_dtype)
            cn = torch.zeros(2, self.batch_size, 512).to(device, dtype=weight_dtype)
            pred, hn, cn = self.model(audiofeat_batch, img_batch, hn, cn)

        # 变回 HWC，便于后续 cv2 合成
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self, pred_frame, idx: int):
        """
        把模型输出的嘴部区域贴回底图对应帧。

        1) 按 idx 读当前底图与坐标
        2) 缩放 pred_frame 到真实人脸框尺寸
        3) 边缘修正后写回
        """
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        res_frame = _correction(combine_frame[y1:y2, x1:x2], res_frame)
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame
