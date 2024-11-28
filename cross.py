from ultralytics import YOLO
import cv2
import numpy as np

# 定义全局变量用于计数
count = 0
# 定义计数线的x坐标
count_line_x = 520

# 记录每个目标上一帧的中心点横坐标
last_center_x = {}

# 检查目标是否越过计数线的函数
def check_crossed_line(bbox, class_id):
    global count, last_center_x
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 如果目标在上一帧已经存在，检查是否从左到右越过了计数线
    if class_id in last_center_x:
        if last_center_x[class_id] < count_line_x and center_x > count_line_x:
            count += 1
    
    # 更新目标的中心点横坐标
    last_center_x[class_id] = center_x
    return count

# Load a model
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
out = cv2.VideoWriter('C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/runs/6.mp4', fourcc, 20.0, (frame_width, frame_height))
# 循环视频帧
while cap.isOpened():
    # 读取某一帧
    success, frame = cap.read()
    if success:
        # 使用yolov8进行预测
        results = model(frame)
        # 可视化结果
        annotated_frame = results[0].plot()
        # 获取检测到的目标信息（边界框、类别等）
        detections = results[0].boxes.data.cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            # 检查目标是否越过计数线并更新计数
            current_count = check_crossed_line([x1, y1, x2, y2], int(class_id))
            # 在画面上显示计数结果（位置可以根据实际需求调整）
            cv2.putText(annotated_frame, f"Count: {current_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 绘制计数线（这里绘制垂直的红线作为示例，可根据需求调整颜色、线宽等）
            cv2.line(annotated_frame, (count_line_x, 0), (count_line_x, frame_height), (0, 0, 255), 2)
        # 将带注释的帧写入视频文件
        out.write(annotated_frame)
    else:
        # 最后结尾中断视频帧循环
        break
# 释放读取和写入对象
cap.release()
out.release()
