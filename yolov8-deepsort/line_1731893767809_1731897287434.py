import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# 初始化YOLOv8模型
model = YOLO('yolov8n.pt')

# 初始化DeepSort
tracker = DeepSort(max_age=30, n_init=2)

# 读取视频
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# 定义计数线
line_start = (300, 300)
line_end = (600, 300)

# 计数器
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8进行目标检测
    results = model(frame)

    # 获取检测结果
    detections = []
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.5:  # 置信度阈值
            detections.append([x1, y1, x2, y2, conf])

    # 使用DeepSort进行跟踪
    tracks = tracker.update_tracks(detections, frame=frame)

    # 绘制计数线
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # 处理跟踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = [int(i) for i in ltrb]

        # 检查是否穿过计数线
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        if center_y < line_start[1]:
            count += 1

        # 绘制跟踪框和ID
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示计数
    cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
