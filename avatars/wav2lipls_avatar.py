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
#  Wav2LipLS 数字人 — 迁移自 liplsreal.py
#  使用 HubertASR 音频特征提取（与 ultralight 共享）
#

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
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path, modelres=192):
    audio_processor = Audio2Feature()
    model = Human(sr=False, face_size=modelres) # 是否超分辨率
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '').replace('_orig_mod.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device, dtype=torch.float16)
    return model.eval(),audio_processor

def load_avatar(avatar_id:str, imgcache_num:int=0):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    if imgcache_num==0:
        input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        frame_list_cycle = read_imgs(input_img_list)
        input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        face_list_cycle = read_imgs(input_face_list)
    else:
        from imgcache import ImgCache
        frame_list_cycle = ImgCache(full_imgs_path,imgcache_num)
        face_list_cycle = ImgCache(face_imgs_path,imgcache_num)
    
    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    human,_ = model
    weight_dtype = torch.float16
    hn = torch.zeros(2, batch_size, 512).to(device, dtype=weight_dtype)
    cn = torch.zeros(2, batch_size, 512).to(device, dtype=weight_dtype)
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device, dtype=weight_dtype)
    mel_batch = torch.ones(batch_size, 10, 1024).to(device, dtype=weight_dtype)
    human(mel_batch, img_batch, hn, cn)

 ### 精修过渡优化
def _correction(frame_to_save_, pred):
    for j in range(20):
        pred[-1 - j] = (j / 20. * pred[-1 - j] + (20 - j) / 20. * frame_to_save_[-1 - j]).astype('uint8')
    for j in range(20):
        pred[:, 20 - j] = ((20 - j) / 20. * pred[:, 20 - j] + j / 20. * frame_to_save_[:, 20 - j]).astype('uint8')
    for j in range(20):
        pred[:, -1 - j] = (j / 20. * pred[:, -1 - j] + (20 - j) / 20. * frame_to_save_[:, -1 - j]).astype('uint8')
    frame_to_save_bool = np.expand_dims(
        (frame_to_save_[:, :, 1].astype('int32') - frame_to_save_[:, :, 2].astype('int32')) > 60, -1)
    pred = pred * (1 - frame_to_save_bool) + frame_to_save_ * frame_to_save_bool

    return pred

@register("avatar", "wav2lipls")
class LipLsReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = mp.Queue(self.batch_size*2)
        self.model,self.audio_processor = model
        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = HubertASR(opt,self,self.audio_processor,[0,5])
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        weight_dtype = torch.float16  # 数据类型
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device, dtype=weight_dtype, non_blocking=True)
        audiofeat_batch = torch.FloatTensor(audiofeat_batch).to(device, dtype=weight_dtype, non_blocking=True)

        with torch.no_grad():
            hn = torch.zeros(2, self.batch_size, 512).to(device, dtype=weight_dtype)
            cn = torch.zeros(2, self.batch_size, 512).to(device, dtype=weight_dtype)
            pred, hn, cn = self.model(audiofeat_batch, img_batch, hn, cn)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred
    
    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        res_frame = _correction(combine_frame[y1:y2, x1:x2], res_frame)
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame
        
            
