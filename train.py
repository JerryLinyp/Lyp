from ultralytics import YOLO
 
if __name__ == '__main__':
    # 加载一个模型
    model = YOLO('yolov8n.yaml')  # 从YAML建立一个新模型
    # 训练模型
    results = model.train(
        data='C:/Users/38917/Downloads/ultralytics-main/drillpipe.yaml',
        device='0',
        epochs=50,
        batch=4,
        verbose=False,
        imgsz=1280)
