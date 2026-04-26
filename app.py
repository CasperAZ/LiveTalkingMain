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

# 这个文件是整个项目的总启动入口，可以把它理解成“装配车间”。
# 它自己不生成口型、不合成语音，也不直接推流，而是负责把各个模块按配置拼起来。
# 如果你后面要把项目改造成“快手数字人直播”，先重点看这里的装配关系，再看：
# 1. `server/routes.py`：接收外部消息/指令；
# 2. `streamout/`：把生成结果输出到具体平台；
# 3. `build_avatar_session()`：给不同平台会话注入不同参数。
# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
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
#sockets = Sockets(app)
opt = None
model = None
# 全局 avatar 缓存，避免多个会话重复加载同一套数字人素材。
# key 是 avatar_id，value 是 load_avatar() 返回的素材对象。
global_avatars = {} # avatar_id: payload
        

#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    # 每个用户/连接都需要一份独立的 opt 副本。
    # 否则像 sessionid、参考音色、动作编排这类“会话级配置”会互相污染。
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id) 
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    if (avatar_id and avatar_id != opt.avatar_id):
        # 如果请求里指定了一个新的 avatar，就优先使用该 avatar。
        # 这里先查缓存，避免每个新会话都重复读磁盘、反序列化素材。
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    else:
        # 没指定时使用启动时默认加载的 avatar。
        avatar_this = global_avatars.get(opt.avatar_id)
    if ref_audio: #请求参数配置了参考音频
        # 这里通常用于语音克隆或切换音色。
        # 注意：只影响当前会话，不会改全局默认值。
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        # 前端如果动态传了动作编排 JSON，这里会覆盖进本会话。
        opt_this.customopt = json.loads(custom_config)

    # 根据配置里的模型名，实例化具体数字人实现。
    # 例如 opt.model=wav2lip 时，这里会创建 LipReal。
    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    return avatar_session

async def offer(request):
    # WebRTC SDP offer 的处理下沉给 RTCManager，
    # app.py 只保留“转发入口”，避免启动文件掺杂太多连接细节。
    return await rtc_manager.handle_offer(request)

async def on_shutdown(app):
    # 服务关闭时统一回收所有 RTC 连接。
    await rtc_manager.shutdown()



def main():
    global rtc_manager, opt, model,load_avatar
    # 运行时解析命令行参数（来自 sys.argv）：
    # - 如果启动命令里传了参数（例如 --model musetalk），这里会读到传入值；
    # - 如果没传，则自动回退到 config.py 中定义的 default。
    # 所以 config.py 不是“写死配置”，而是“参数定义 + 默认值”。
    from config import parse_args
    opt = parse_args()

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    # 约定每个 avatar 模块都暴露以下三个函数，方便统一装配：
    # - load_model: 加载推理模型
    # - load_avatar: 加载 avatar 素材
    # - warm_up: 预热，减少第一次推理卡顿
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    if opt.model == 'musetalk':
        # MuseTalk：模型和 avatar 素材是两块独立资源。
        model = load_model()
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        # Wav2Lip：加载统一权重文件，再加载 avatar 对应的人脸素材。
        model = load_model("./models/wav2lip.pth")
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        # UltraLight：模型参数与 opt 绑定更紧，所以直接把 opt 传进去。
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,global_avatars[opt.avatar_id],160)

    # 把“如何创建会话”的工厂函数交给 SessionManager。
    # 以后即使不是 WebRTC 场景，只要要创建一个数字人会话，也能复用这条入口。
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
    
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # virtualcam / rtmp 模式没有浏览器来触发会话创建，
        # 所以这里手工创建一个固定 session 0，并直接启动渲染线程。
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    
    # 注册所有通用业务接口。
    # 对外部平台适配来说，这里相当于项目的“控制面 API”。
    setup_routes(appasync) 

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        # 单独创建事件循环，避免和模型线程、推流线程混在一起。
        # 这是这个项目把“异步网络层”和“同步推理层”拼在一起的关键做法之一。
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            # rtcpush 模式下，服务端会主动向远端 RTC 地址发起推流。
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

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
    
    
    
