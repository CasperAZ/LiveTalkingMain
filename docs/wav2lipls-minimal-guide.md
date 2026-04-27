# LiveTalkingMain + wav2lipls 最小接入说明（学习版）

这个文件按“先能跑起来、再逐步优化”来写，建议按顺序执行，避免参数和文件名错位。

## 1) 放置权重文件

```bash
cp checkpoint_step001430000.pth ./models/
```

说明：
- `checkpoint_step001430000.pth`：默认用于 `--modelres` 非 384 的场景（例如 192）。
- 项目里也可以放 `checkpoint_step002130000.pth`，这是一个 384 分辨率版本（后面有示例）。
- 文件一定要放在 `models/` 根目录，否则启动时报找不到路径。

## 2) 生成 avatar 素材（关键）

```bash
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m avatars.wav2lipls.genavatar_yolo \
  --video_path xxx.mp4 \
  --img_size 192 \
  --avatar_id wav2lipls_avatar1
```

说明：
- `xxx.mp4` = 你准备好的数字人视频（需要脸部在帧里可见）。
- 脚本会输出：
  - `data/avatars/<avatar_id>/full_imgs/`：每帧底图
  - `data/avatars/<avatar_id>/face_imgs/`：切出的人脸裁剪图
  - `data/avatars/<avatar_id>/coords.pkl`：每帧人脸框坐标
- `img_size` 先用 192 起步，后续如果显存和效果允许可以尝试 384。

## 3) 下载 Hubert 模型（音频特征）

```python
from huggingface_hub import snapshot_download
snapshot_download('facebook/hubert-large-ls960-ft', local_dir='models/hubert-large-ls960-ft')
```

说明：
- 这个目录用于 `wav2lipls` 的 Hubert 特征提取。
- 确认目录存在：`models/hubert-large-ls960-ft/`。

## 4) 启动服务（默认 192 分辨率）

```bash
python app.py --transport webrtc --model wav2lipls --avatar_id wav2lipls_avatar1 --max_session 10
```

参数说明：
- `--transport webrtc`：本地测试常用（配套网页可发起/webrtc）
- `--model wav2lipls`：启用你刚加入的模型链路
- `--avatar_id wav2lipls_avatar1`：与你生成素材目录一致
- `--max_session 10`：并发会话上限（这个分支里常见的示例值）
