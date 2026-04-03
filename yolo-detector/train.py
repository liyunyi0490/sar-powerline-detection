from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO('yolov8m.pt')

    # 训练配置
    train_config = {
        'data': 'data.yaml',
        'epochs': 100,
        'patience': 20,  # 早停轮数
        'imgsz': 640,
        'augment': True,  # 启用数据增广
        'device': 0,  # GPU设备
        'batch': 8,  # 批量大小，根据显存调整
        'name': 'pylon_powerline_detection3',
        # 详细的增广配置
        'hsv_h': 0.015,  # 色调增强
        'hsv_s': 0.7,  # 饱和度增强
        'hsv_v': 0.4,  # 亮度增强
        'degrees': 10,  # 旋转角度
        'translate': 0.1,  # 平移
        'scale': 0.5,  # 缩放
        'shear': 0.0,  # 剪切
        'flipud': 0.5,  # 上下翻转
        'fliplr': 0.5,  # 左右翻转
        'mosaic': 1.0,  # 马赛克增强
        'mixup': 0.0  # 混合增强
    }

    # 开始训练
    print("开始训练模型...")
    results = model.train(**train_config)
    print("训练完成！")