import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from deep_sort.deep_sort.detection import Detection

# 定义类别列表
class_names = ['drill bit']  # 根据你的需求定义类别

# 初始化 YOLOv8 模型
model = YOLO("C:/Users/38917/Downloads/ultralytics-main/yolov8-d