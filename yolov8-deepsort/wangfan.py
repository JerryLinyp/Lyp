import cv2
import numpy as np
import torch  # 导入torch库
from ultralytics import YOLO  # 替换为你的YOLOv8实现
from deep_sort import deep_sort  # 替换为你的DeepSORT实现

# 初始化YOLOv8模型
model_path = 'C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/best.pt'  # 替换为你的模型路径
conf_thres = 0.7  # 置信度阈值
iou_thres = 0.45  # IoU阈值
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path, conf=conf_thres, iou=iou_thres, device=device)

# 初始化DeepSORT跟踪器
#deepsort = DeepSORT(max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0)
# 初始化DeepSORT跟踪器
deepsort_tracker = deep_sort(max_age=30, min_hits=3, iou_threshold=0.3)

# 读取视频文件
video_path = 'C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/test.mp4'
output_path = 'C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/runs/result6.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码格式
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 变量初始化
track_ids = []
bbox_history = {}  # 使用字典来存储每个物体的边界框历史
motion_directions = {}  # 使用字典来存储每个物体的运动方向
return_count = 0  # 往返运动次数计数器

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8进行目标检测
    detections = model(frame)  # 注意这里应该是model而不是yolov8_model

    # 将检测结果转换为DeepSORT所需的格式
    detections = [[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], conf, cls] for bbox, conf, cls in detections]

    # 使用DeepSORT进行目标跟踪
    tracks = deepsort_tracker.update(detections, frame)

    # 遍历跟踪结果并更新状态
    for track in tracks:
        track_id = track.track_id
        bbox = track.bbox.tolist()  # 转换为列表格式，以便后续处理
        x_center = (bbox[0] + bbox[2]) / 2

        if track_id not in bbox_history:
            bbox_history[track_id] = [[x_center]]  # 初始化边界框历史
            motion_directions[track_id] = None  # 初始化为None，表示方向未知
        else:
            # 获取物体的历史中心位置
            prev_x_centers = bbox_history[track_id]
            prev_x_center = prev_x_centers[-1]

            # 计算物体的运动方向（这里简单假设物体是水平移动的）
            direction = 'right' if x_center > prev_x_center else 'left'

            # 检查物体的运动方向是否改变（并且已经有一个已知方向）
            if motion_directions[track_id] is not None and motion_directions[track_id] != direction:
                # 如果方向改变，则增加往返运动次数（这里只计算相邻两次方向改变的情况）
                if len(prev_x_centers) > 1 and (prev_x_centers[-2] < x_center if direction == 'right' else prev_x_centers[-2] > x_center):
                    return_count += 1

            # 更新物体的运动方向和边界框历史
            motion_directions[track_id] = direction
            bbox_history[track_id].append(x_center)

    # 在图像上绘制跟踪结果和计数结果
    for track in tracks:
        track_id = track.track_id
        bbox = track.bbox.astype(int).tolist()
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        text = f'ID: {track_id} Count: {return_count}'
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 将结果帧写入输出视频文件
    out.write(frame)

# 释放资源
cap.release()
out.release()

print(f'Total return trips counted: {return_count}')
