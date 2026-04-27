1，将checkpoint_step001430000.pth拷到models下，其他代码拷到livetalking下覆盖

2，生成avatar
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple 
python -m avatars.wav2lipls.genavatar_yolo  --video_path xxx.mp4  --img_size 192 --avatar_id wav2lipls_avatar1

3,下载hubert模型放到models目录下
```python
import huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download('facebook/hubert-large-ls960-ft', local_dir='models/hubert-large-ls960-ft')
```
4，运行
python app.py --transport webrtc --model wav2lipls --avatar_id wav2lipls_avatar1 --max_session 10



384模型
将checkpoint_step002130000.pth拷到models下


python -m avatars.wav2lipls.genavatar_yolo  --video_path xxx.mp4  --img_size 384 --avatar_id wav2lipls_avatar1


python app.py --transport webrtc --model wav2lipls --modelres 384 --avatar_id wav2lipls_avatar1 --max_session 10
