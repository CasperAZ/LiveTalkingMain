from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--avatar_id', default='wav2lipls_avatar1', type=str)
parser.add_argument('--video_path', default='', type=str)
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

# 初始化YOLO模型
face_det = YOLO('./avatars/wav2lipls/models/yolov8n-face.pt')

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    print(f"即将使用OpenCV将视频: {vid_path} 转换为图片")
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break
    print("视频转换完成")

def read_imgs(img_list):
    frames = []
    print('读取图片到内存...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def detect_face(face_img):
    boxes = face_det(face_img,
                     imgsz=640,
                     conf=0.01,
                     iou=0.5,
                     half=False,  # cpu必须是False
                     augment=False,
                     verbose=False,
                     device=device)[0].boxes
    bboxes = boxes.xyxy.cpu().numpy()
    return bboxes

def face_detect(images):
    print('即将开始人脸检测...')
    predictions = []
    
    for i in tqdm(range(len(images))):
        try:
            boxes = detect_face(images[i])
            if len(boxes) > 0:
                predictions.append(boxes[0])  # 取第一个检测到的人脸
            else:
                predictions.append(None)
        except RuntimeError as e:
            if "Couldn't load custom C++ ops" in str(e):
                print("检测到PyTorch版本兼容性问题，尝试使用CPU进行检测...")
                boxes = face_det(images[i],
                               imgsz=640,
                               conf=0.01,
                               iou=0.5,
                               half=False,
                               augment=False,
                               verbose=False,
                               device='cpu')[0].boxes
                bboxes = boxes.xyxy.cpu().numpy()
                if len(bboxes) > 0:
                    predictions.append(bboxes[0])
                else:
                    predictions.append(None)
            else:
                raise e

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: 
        print('正在平滑处理人脸框...')
        boxes = get_smoothened_boxes(boxes, T=5)
    
    print('正在生成最终结果...')
    results = [[image[int(y1):int(y2), int(x1):int(x2)], (int(y1), int(y2), int(x1), int(x2))] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    print('人脸检测完成')
    return results

if __name__ == "__main__":
    avatar_path = f"./data/avatars/{args.avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    osmakedirs([avatar_path,full_imgs_path,face_imgs_path])
    print(args)

    video2imgs(args.video_path, full_imgs_path, ext = 'png')
    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

    frames = read_imgs(input_img_list)
    face_det_results = face_detect(frames) 
    coord_list = []
    idx = 0
    print(f"共检测到{len(face_det_results)}张人脸")
    for frame,coords in face_det_results:        
        resized_crop_frame = cv2.resize(frame,(args.img_size, args.img_size))
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append(coords)
        idx = idx + 1
	
    print(f"写入数据到坐标文件:{coords_path}")
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
