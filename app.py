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

    # ─── 会话装配入口 ────────────────────────────────────────────────────
    # 这里把“创建一个数字人会话”的方法（build_avatar_session）注册给 SessionManager。
    # 可以把它理解为“会话工厂注册”：
    # - SessionManager 负责管理会话生命周期（创建、查询、移除）；
    # - 但具体怎么创建会话实例，由这个工厂函数决定。
    # 这样做的好处是：后面不管是 WebRTC、RTMP、virtualcam，还是其它接入方式，
    # 只要需要会话，都会走同一条创建逻辑，避免每种传输方式各写一份。
    session_manager.init_builder(build_avatar_session)
    # RTCManager 负责 WebRTC 连接管理（offer/answer、PeerConnection 生命周期等）。
    # 注意：它管理的是“连接”，不是“模型推理本身”。
    rtc_manager = RTCManager(opt)
    
    # ─── 非 WebRTC 模式的会话启动 ──────────────────────────────────────
    # virtualcam / rtmp 场景下，没有浏览器去触发 /offer，
    # 所以不会自动创建会话。这里手工创建一个固定会话 '0' 并启动渲染线程。
    # 你可以理解成“服务端自启动一个无人值守会话”。
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        # thread_quit 用来通知渲染线程停止（当前主流程里常驻运行，通常不主动置位）。
        thread_quit = Event()
        # 预留参数字典：如果后续要给 session 0 注入特定配置，可从这里传。
        params = {}
        session_manager.add_session('0', build_avatar_session('0', params))
        # render() 是 Avatar 会话内部的主循环（TTS/ASR/推理/输出联动）。
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # ─── aiohttp 应用层初始化 ────────────────────────────────────────────
    # 这里是“网络控制面”应用，不是推理线程。主要负责 API 接口、WebRTC 信令等。
    # client_max_size=100MB：允许较大的上传请求（例如音频文件上传场景）。
    appasync = web.Application(client_max_size=1024**2*100)
    # 把 llm_response 函数挂到 app 上，路由处理器会从 app 中取出并调用。
    # 这样做可以减少全局变量耦合，路由层更容易测试和替换。
    appasync["llm_response"] = llm_response

    # 服务关闭时统一执行 on_shutdown，回收 RTC 连接等资源。
    appasync.on_shutdown.append(on_shutdown)
    # WebRTC 信令入口：浏览器/客户端通过 /offer 发起会话协商。
    appasync.router.add_post("/offer", offer)
    
    # 注册业务 API（/human、/humanaudio、/interrupt_talk 等）。
    # 对外部平台适配来说，这里相当于“控制面接口层”。
    setup_routes(appasync) 

    # ─── CORS 配置 ──────────────────────────────────────────────────────
    # 允许跨域调用这些接口，方便浏览器前端或其它域名下的控制面访问。
    # 当前配置是较宽松的“全开放”策略（生产环境可按需收紧来源域名）。
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # 把 CORS 规则应用到当前 app 的所有路由。
    for route in list(appasync.router.routes()):
        cors.add(route)

    # ─── 启动信息展示 ────────────────────────────────────────────────────
    # 根据 transport 选择推荐访问的页面，方便本地调试时直接打开对应前端。
    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')

    # ─── 网络服务主循环 ──────────────────────────────────────────────────
    # 这里显式创建一个独立事件循环，和推理线程分离，避免互相阻塞。
    # 核心思想：网络 I/O 归 asyncio 管，模型推理归渲染线程管。
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 1) 初始化 runner（注册路由、准备站点）
        loop.run_until_complete(runner.setup())
        # 2) 绑定监听地址与端口
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        # 3) 启动 HTTP 服务
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            # rtcpush 模式下，服务端主动向远端 WHIP/RTC 地址发起推流。
            # max_session 控制发起几路推流；第 0 路用原始 push_url，
            # 后续路数在末尾拼接序号（例如 ...livestream1、...livestream2）。
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        # 4) 常驻事件循环，持续响应 API / WebRTC 信令
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    # 当前实现直接在主线程运行 run_server（阻塞式常驻）。
    # 上面注释掉的是“另起线程跑服务”的旧尝试。
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    # Windows/macOS 上使用 spawn 更稳妥，能避免 fork 带来的某些资源继承问题。
    # 必须放在入口保护里调用，避免子进程重复执行入口逻辑。
    mp.set_start_method('spawn')
    main()
    
    
    
