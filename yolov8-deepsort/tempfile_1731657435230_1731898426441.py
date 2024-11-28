import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from deep_sort.deep_sort.detection import Detection

# 定义类别列表
class_names = ['drill bit']  # 根据你的需求定义类别

# 初始化 YOLOv8 模型
model = YOLO("C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/best.pt")  # 确保路径正确
model.conf = 0.5  # 设置置信度阈值

# 初始化 DeepSORT
deepsort = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")  # 确保路径正确

# 定义视频源
video_source = "C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/test.mp4"
cap = cv2.VideoCapture(video_source)

# 获取视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义输出视频
output_video_path = "C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 初始化计数器和其他变量
counter = 0
direction = None
start_line_y = 300  # 假设的起点线 y 坐标
end_line_y = 400    # 假设的终点线 y 坐标

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 目标检测
    results = model(frame, size=640)  # 根据需要调整输入大小
    detections = []

    # 解析 YOLOv8 输出
    for *xyxy, conf, cls in results.xyxy[0]:  # 解析边界框
        x1, y1, x2, y2 = map(int, xyxy)
        if class_names[int(cls)] in class_names and conf >= model.conf:
            detections.append(Detection(bbox=[x1, y1, x2, y2], confidence=conf, class_id=cls))

    # 转换为 DeepSORT 格式
    detections = [d.__dict__ for d in detections]

    # DeepSORT 跟踪
    tracked_objects = deepsort.update(detections, frame.shape[:2])

    # 绘制和计数逻辑
    for track in tracked_objects:
        bbox = track['bbox']
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        # 判断是否穿过线
        if direction is None and y_center < start_line_y:
            direction = "forward"
        elif direction == "forward" and y_center > end_line_y:
            direction = "backward"
        elif direction == "backward" and y_center < start_line_y:
            counter += 1
            direction = None

        # 绘制边界框和 ID
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, str(track['track_id']), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示计数
    cv2.putText(frame, f"Count: {counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 写入和显示视频帧
    out.write(frame)
    cv2.imshow("Frame", frame)

    # 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
