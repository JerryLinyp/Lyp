import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import deep_sort.deep_sort as ds

def validate_input_path(input_path: str) -> bool:
    """验证输入路径是否存在且为有效的视频文件。"""
    if not os.path.isfile(input_path):
        print(f"Input path {input_path} does not exist or is not a file.")
        return False
    return True

def draw_line(frame, line):
    """在帧中绘制线。"""
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

def is_crossing_line(line, center):
    """判断中心点是否跨越了线。"""
    x1, y1, x2, y2 = line
    x, y = center
    return (y1 - y2) * x + (x2 - x1) * y + (x1 * y2 - x2 * y1) > 0

def putTextWithBackground(img, text, origin, font_scale, text_color, bg_color):
    """在图像上绘制带有背景的文本。"""
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_width, text_height = text_size
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, lineType=cv2.LINE_AA)

def extract_detections(results, detect_class):
    """从模型结果中提取和处理检测信息。"""
    detections = np.empty((0, 4))
    confarray = []
    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                confarray.append(conf)
    return detections, confarray

def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker, lines) -> Path:
    """处理视频，检测并跟踪目标，并添加划线计数功能。"""
    if not validate_input_path(input_path):
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_video_path = Path(output_path) / "shuchu.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True)

    line_counts = {i: 0 for i in range(len(lines))}

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)
        detections, confarray = extract_detections(results, detect_class)
        resultsTracker = tracker.update(detections, confarray, frame)

        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            for i, line in enumerate(lines):
                if is_crossing_line(line, center):
                    line_counts[i] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))

        for i, line in enumerate(lines):
            draw_line(frame, line)
            putTextWithBackground(frame, f"Count: {line_counts[i]}", (line[0], line[1] - 20), font_scale=1.0, text_color=(255, 255, 255), bg_color=(0, 0, 255))

        output_video.write(frame)

    output_video.release()
    cap.release()
    print(f'Output video saved to: {output_video_path}')
    return output_video_path

if __name__ == "__main__":
    input_path = "C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/test.mp4"
    output_path = "C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/runs"
    model = YOLO("C:/Users/38917/Downloads/ultralytics-main/yolov8-deepsort/best.pt")
    detect_class = 0
    print(f"Detecting {model.names[detect_class]}")
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    # 定义划线
    lines = [
        #(100, 200, 500, 200),  # 水平线
        (520, 0, 520, 768)   # 垂直线
    ]

    detect_and_track(input_path, output_path, detect_class, model, tracker, lines)
