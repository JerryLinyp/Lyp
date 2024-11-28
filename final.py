import mysql.connector
from ultralytics import YOLO
import cv2
import numpy as np
import hashlib
import boto3
from botocore.exceptions import NoCredentialsError

# MySQL 数据库连接配置
config = {
    'user': 'root',
    'password': 'Linyipeng123',
    'host': 'localhost',
    'database': 'video',
    'port': '3306'
}

# MinIO 配置
minio_config = {
    'endpoint_url': 'http://127.0.0.1:9000',  # MinIO 服务器地址
    'aws_access_key_id': 'minioadmin',
    'aws_secret_access_key': 'minioadmin',
    'region_name': 'us-east-1',  # 可选
    'bucket_name': 'drillpipe'  # 存储桶名称
}

# 连接到 MySQL 数据库
try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
except mysql.connector.Error as err:
    print(f"数据库连接失败: {err}")
    exit(1)

# 连接到 MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=minio_config['endpoint_url'],
    aws_access_key_id=minio_config['aws_access_key_id'],
    aws_secret_access_key=minio_config['aws_secret_access_key'],
    region_name=minio_config['region_name']
)

# 执行查询以获取视频地址
query = "SELECT path FROM video_path WHERE status='0'"
try:
    cursor.execute(query)
    results = cursor.fetchall()  # 获取所有结果
except mysql.connector.Error as err:
    print(f"执行查询失败: {err}")
    cursor.close()
    conn.close()
    exit(1)

# 加载模型（通常模型只需要加载一次）
model = YOLO('C:/Users/38917/Desktop/improve/runs/detect/train29/weights/best.pt')

for result in results:
    video_path = result[0]

    # 为每个视频创建一个唯一的输出路径
    output_video_path = f'C:/Users/38917/Desktop/improve/runs/output_video_{hashlib.md5(video_path.encode()).hexdigest()}.mp4'

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue  # 跳过当前视频，处理下一个

    # 获取视频帧的维度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    # 循环视频帧
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov8进行预测
            results = model(frame)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
        else:
            # 最后结尾中断视频帧循环
            break

    # 释放资源
    cap.release()
    out.release()

    # 上传视频到 MinIO
    try:
        s3_client.upload_file(output_video_path, minio_config['bucket_name'], output_video_path.split('/')[-1])
        print(f"Video uploaded to MinIO: {output_video_path.split('/')[-1]}")
    except NoCredentialsError:
        print("Credentials not available")
    except Exception as e:
        print(f"Failed to upload video: {e}")

    # 更新数据库状态
    update_query = "UPDATE video_path SET status='1' WHERE path=%s"
    try:
        cursor.execute(update_query, (video_path,))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"更新数据库状态失败: {err}")

# 关闭数据库连接
cursor.close()
conn.close()
