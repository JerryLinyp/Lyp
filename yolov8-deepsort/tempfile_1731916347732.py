from ultralytics import YOLO
import cv2
import numpy as np

# 定义全局变量用于计数
count = 0
# 定义起始线和终止线的x坐标
start_line_x = 350
end_line_x = 580

# 记录每个目标上一帧的中心点横坐标
last_center_x = {}

# 检查目标是否越过计数线的函数
def check_crossed_line(bbox, class_id, conf):
    global count, last_center_x
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 如果目标在上一帧已经存在，检查是否从起始线到终止线或从终止线到起始线
    if class_id in last_center_x:
        if (last_center_x[class_id] < start_line_x and center_x > end_line_x) or \
           (last_center_x[class_id] > end_line_x and center_x < start_line_x):
            count += 1

    # 更新目标的中心点横坐标
    last_center_x[class_id] = center_x
    return count

# 加载模型
model = YOLO('C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/best.pt')

# 打开视频文件
video_path = "C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/test.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧的维度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/runs/result.mp4', fourcc, 20.0,
                      (frame_width, frame_height))

# 循环视频帧
while cap.isOpened():
    # 读取某一帧
    success, frame = cap.read()
    if not success:
        # 最后结尾中断视频帧循环
        break

    # 使用YOLO进行预测
    results = model(frame)

    # 可视化结果
    annotated_frame = results[0].plot()

    # 获取检测到的目标信息（边界框、类别等）
    detections = results[0].boxes.data.cpu().numpy()
    valid_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        if conf >= 0.6:  # 过滤掉置信度小于0.6的目标
            valid_detections.append(detection)

    for detection in valid_detections:
        x1, y1, x2, y2, conf, class_id = detection
        # 检查目标是否越过计数线并更新计数
        current_count = check_crossed_line([x1, y1, x2, y2], int(class_id), conf)

        # 在画面上显示计数结果
        cv2.putText(annotated_frame, f"Count: {current_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 绘制起始线和终止线
        cv2.line(annotated_frame, (start_line_x, 0), (start_line_x, frame_height), (0, 255, 0), 2)
        cv2.line(annotated_frame, (end_line_x, 0), (end_line_x, frame_height), (0, 0, 255), 2)

    # 将带注释的帧写入视频文件
    out.write(annotated_frame)

# 释放读取和写入对象
cap.release()
out.release()
