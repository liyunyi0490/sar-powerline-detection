# 电塔电线检测项目

## 项目结构

```
├── data/             # 数据集目录
│   ├── train/        # 训练集
│   │   ├── images/   # 训练图片
│   │   └── labels/   # 训练标签
│   ├── val/          # 验证集
│   │   ├── images/   # 验证图片
│   │   └── labels/   # 验证标签
│   └── output/       # 推理结果输出目录
├── runs/             # 运行结果目录
│   └── detect/       # 检测模型训练结果
├── yolov8m.pt        # 预训练模型
├── data.yaml         # 数据集配置文件
├── train.py          # 训练脚本
├── evaluate.py       # 模型评价脚本
└── inference.py      # 推理脚本
```

## 项目作用

本项目主要用于对光学图片和SAR图片融合后的图像进行YOLO目标检测，检测目标为电塔（pylon）和电线（powerline）。

## 环境配置

### 依赖项
- Python 3.8+
- PyTorch 1.8+
- Ultralytics YOLOv8
- OpenCV
- NumPy

### 推荐环境
使用conda创建并激活环境：

```bash
conda create -n sar39 python=3.9
conda activate sar39
pip install ultralytics opencv-python numpy torch torchvision
```

## 使用方法

### 1. 数据准备
确保数据集按照上述结构组织，包含训练集和验证集，每个集合下有images和labels文件夹。

标签格式为YOLO格式：
```
<class_id> <x_center> <y_center> <width> <height>
```
其中：
- `class_id`：0表示pylon（电塔），1表示powerline（电线）
- 坐标为归一化值（0-1）

### 2. 模型训练

运行训练脚本：

```bash
python train.py
```

训练配置参数：
- 预训练模型：yolov8m.pt
- 训练轮数：100
- 早停设置：20轮无增益暂停
- 批量大小：8（适合8GB显存）
- 图像大小：640x640
- 数据增广：启用（包含色调、饱和度、亮度调整，旋转、平移、缩放，翻转等）

训练结果会保存在 `runs/detect/pylon_powerline_detection3/` 目录中。

### 3. 模型评价

运行评价脚本：

```bash
python evaluate.py
```

评价指标：
- 精确率（Precision）
- 召回率（Recall）
- F1分数

评价参数：
- 置信度阈值：0.3
- IOU阈值：0.3
- 去重IOU阈值：0.4

### 4. 模型推理

运行推理脚本：

```bash
python inference.py
```

推理参数：
- 置信度阈值：0.3
- 去重重叠阈值：0.4

推理结果会保存在 `runs/data/` 目录中，包含带有检测框的图片。

## 模型配置

### data.yaml

```yaml
train: data/train/images
val: data/val/images

nc: 2
names: ['pylon', 'powerline']
```

### 训练配置（train.py）

- `data`：数据集配置文件路径
- `epochs`：训练轮数
- `patience`：早停轮数
- `imgsz`：图像大小
- `augment`：是否启用数据增广
- `device`：训练设备（0表示第一个GPU）
- `batch`：批量大小
- `name`：训练结果保存名称
- 增广参数：hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, flipud, fliplr, mosaic, mixup

## 注意事项

1. 确保GPU显存足够（推荐8GB以上）
2. 训练前确保数据集标签格式正确
3. 推理时会自动创建输出目录
4. 评价脚本会计算模型在验证集上的性能指标

## 结果说明

- 训练完成后，最佳模型权重会保存在 `runs/detect/pylon_powerline_detection3/weights/best.pt`
- 推理结果图片会保存在 `data/output/` 目录
- 评价结果会在终端显示，包含精确率、召回率和F1分数
