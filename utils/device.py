import torch
import warnings

def initialize_device():
    # 设备选择优先级：
    # CUDA > Apple MPS > CPU
    # 这样同一套代码可以尽量在不同硬件环境下自动工作。
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
