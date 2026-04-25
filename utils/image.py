import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 这个文件主要处理两类事情：
# 1. 批量读取素材图片；
# 2. 把循环帧序列做“镜像往返索引”，避免播放到末尾时突兀跳回第一帧。

# def read_imgs(img_list):
#     frames = []
#     logger.info('reading images...')
#     for img_path in tqdm(img_list):
#         frame = cv2.imread(img_path)
#         frames.append(frame)
#     return frames

def read_imgs(img_list):
    # 使用线程池并发读取图片，能显著降低大量 avatar 素材的加载时间。
    def load_image(index, img_path):
        return index, cv2.imread(img_path)

    frames = [None] * len(img_list)  # Initialize a list with the same length as img_list
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, idx, img_path): idx for idx, img_path in enumerate(img_list)}
        for future in tqdm(as_completed(futures), total=len(img_list)):
            idx, img = future.result()
            frames[idx] = img
    return frames

def mirror_index(size, index):
    # 例子：
    # size=4 时，输出索引序列会像 0,1,2,3,3,2,1,0,0,1...
    # 这样比单纯 0,1,2,3,0,1,2,3... 更自然，适合循环动作。
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 
