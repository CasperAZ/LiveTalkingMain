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

# server.py
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_sockets import Sockets
import base64
import json
# import gevent
# from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread, Event
# import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import llm_response
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import asyncio
import torch
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc


app = Flask(__name__)
opt = None
model = None
# key: avatar_id, value: 由当前模型加载出的 avatar 数据（底图/坐标）
global_avatars = {}


##### webrtc ################################
# rtc_manager 负责 WebRTC 连接生命周期管理。
rtc_manager = None


def randN(N)->int:
    """生成长度为 N 的随机数字码（用于简单业务标识）。"""
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)


def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    """
    会话工厂：把会话参数 + 运行参数 + 模型 + 素材组装为一个 avatar 会话实例。

    任何外部入口（/offer、rtmp、web 等）都应通过它创建会话，
    这样可保证会话级参数不会互相污染。
    """
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    # 会话是否指定了别的 avatar_id（用户动态切换角色）
    avatar_id = params.get('avatar',opt.avatar_id)
    # 用于参考音色/音频克隆
    ref_audio = params.get('refaudio', '')
    ref_text = params.get('reftext', '')

    if (avatar_id and avatar_id != opt.avatar_id):
        # 全局 avatar 缓存可避免同一素材重复读盘
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    else:
        # 使用启动时已加载的默认 avatar
        avatar_this = global_avatars.get(opt.avatar_id)

    # 参考音色参数只影响“当前会话”，不会改写全局默认
    if ref_audio:
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text

    # 动作编排配置（可选）：将 JSON 写入会话级 customopt
    custom_config = params.get('custom_config', '')
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    return avatar_session


async def offer(request):
    """WebRTC 握手入口：所有 SDP offer 都走这里。"""
    return await rtc_manager.handle_offer(request)


async def on_shutdown(app):
    """进程关闭时，把所有 WebRTC 资源释放。"""
    await rtc_manager.shutdown()


def main():
    global rtc_manager, opt, model, load_avatar

    # 1) 解析启动参数（参数定义与默认值在 config.py）
    from config import parse_args
    opt = parse_args()

    # 2) 按 opt.model 动态 import 对应 avatar 实现
    #   约定模块内要有 load_model/load_avatar/warm_up 三个函数。
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
        'wav2lipls':  'avatars.wav2lipls_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    # 3) 根据模型分支加载对应模型和 avatar
    if opt.model == 'musetalk':
        model = load_model()
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, 256)
    elif opt.model == 'ultralight':
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, global_avatars[opt.avatar_id], 160)
    elif opt.model == 'wav2lipls':
        # 支持用户通过 --modelfile 指定权重；否则按 modelres 选择内置 ckpt。
        if opt.modelfile != '':
            model = load_model("./models/" + opt.modelfile, opt.modelres)
        else:
            if opt.modelres == 384:
                model = load_model("./models/checkpoint_step002130000.pth", opt.modelres)
            else:
                model = load_model("./models/checkpoint_step001430000.pth", opt.modelres)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, opt.modelres)

    # 4) 注册会话工厂（SessionManager 只管管理，不关心模型细节）
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)

    # 5) 非 WebRTC 模式：创建固定会话 0，并手动启动渲染线程（虚拟摄像头/RTMP）
    if opt.transport == 'virtualcam' or opt.transport == 'rtmp':
        thread_quit = Event()
        params = {}
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render, args=(thread_quit,))
        rendthrd.start()

    # 6) 启动 aiohttp 控制面（包括 /offer + 其他 API）
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    setup_routes(appasync)

    # 7) 跨域设置（开发默认放开，生产请收敛）
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    for route in list(appasync.router.routes()):
        cors.add(route)

    # 8) 启动地址提示
    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp':
        pagename = 'rtmpapi.html'
    elif opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    logger.info('start http server; http://<serverip>:' + str(opt.listenport) + '/' + pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:' + str(opt.listenport) + '/dashboard.html')

    def run_server(runner):
        """
        单独创建 event loop 运行 aiohttp server；这样推理线程不受影响。
        rtcpush 模式下还会在这里主动发起推流连接（handle_rtcpush）。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k != 0:
                    push_url = opt.push_url + str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()

    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    # 以下是历史遗留的兼容启动方式注释，当前未启用
    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)
    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
