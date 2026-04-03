from ultralytics import YOLO
import os
import glob
import cv2
import numpy as np
from pathlib import Path

def nms(boxes, scores, overlap_threshold=0.4):
    """基于重叠面积比例的去重：当两个框重叠面积占任意一个框的0.4以上时，删除置信度低的框"""
    if len(boxes) == 0:
        return []
    
    # 按置信度降序排序
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 获取当前框和其他框
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # 计算当前框面积
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        
        # 计算重叠面积
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算其他框面积
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        
        # 计算重叠面积占当前框的比例和占其他框的比例
        overlap_current = intersection / (current_area + 1e-6)
        overlap_other = intersection / (other_areas + 1e-6)
        
        # 保留重叠面积占比小于阈值的框
        mask = (overlap_current < overlap_threshold) & (overlap_other < overlap_threshold)
        indices = indices[1:][mask]
    
    return keep

# 加载训练好的模型
def find_latest_model():
    """查找最新的训练模型"""
    detect_dir = 'runs/detect'
    if not os.path.exists(detect_dir):
        return None
    
    # 获取所有训练结果文件夹
    train_dirs = [d for d in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, d))]
    if not train_dirs:
        return None
    
    # 按修改时间排序，选择最新的
    train_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(detect_dir, x)), reverse=True)
    latest_dir = train_dirs[0]
    model_path = os.path.join(detect_dir, latest_dir, 'weights', 'best.pt')
    
    if os.path.exists(model_path):
        return model_path
    return None

# 查找最新模型
model_path = find_latest_model()
if not model_path:
    print("错误：未找到训练模型！")
    print("请先运行 train.py 训练模型")
    exit(1)

print(f"使用模型: {model_path}")
model = YOLO(model_path)

# 验证集图片路径
val_images_path = 'data/val/images'

# 输出结果路径
output_path = 'data/output'
os.makedirs(output_path, exist_ok=True)

# 获取所有验证集图片
image_files = glob.glob(os.path.join(val_images_path, '*.jpg'))

print(f"开始对验证集的 {len(image_files)} 张图片进行检测...")

# 对每张图片进行检测
for image_file in image_files:
    # 读取图片
    img = cv2.imread(image_file)
    if img is None:
        print(f"无法读取图片: {image_file}")
        continue
    
    # 执行检测
    results = model(image_file, conf=0.3)
    
    # 获取检测结果
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print(f"未检测到目标: {os.path.basename(image_file)}")
        # 保存原图
        output_file = os.path.join(output_path, os.path.basename(image_file))
        cv2.imwrite(output_file, img)
        continue
    
    # 提取检测框信息
    xyxy = boxes.xyxy.cpu().numpy()  # 检测框坐标
    confs = boxes.conf.cpu().numpy()  # 置信度
    cls_ids = boxes.cls.cpu().numpy().astype(int)  # 类别ID
    
    # 置信度过滤（0.3以上）
    mask = confs >= 0.3
    xyxy = xyxy[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]
    
    if len(xyxy) == 0:
        print(f"过滤后无目标: {os.path.basename(image_file)}")
        output_file = os.path.join(output_path, os.path.basename(image_file))
        cv2.imwrite(output_file, img)
        continue
    
    # 按类别分别进行NMS去重
    final_boxes = []
    final_confs = []
    final_cls_ids = []
    
    unique_classes = np.unique(cls_ids)
    for cls_id in unique_classes:
        cls_mask = cls_ids == cls_id
        cls_boxes = xyxy[cls_mask]
        cls_confs = confs[cls_mask]
        
        # 对该类别的框进行NMS
        keep_indices = nms(cls_boxes, cls_confs, overlap_threshold=0.4)
        
        final_boxes.extend(cls_boxes[keep_indices])
        final_confs.extend(cls_confs[keep_indices])
        final_cls_ids.extend([cls_id] * len(keep_indices))
    
    # 使用YOLO自带的plot方法进行标注
    # 创建一个新的Results对象用于绘图
    from ultralytics.engine.results import Results
    
    # 构建新的boxes tensor
    import torch
    if len(final_boxes) > 0:
        final_boxes_tensor = torch.tensor(final_boxes)
        final_confs_tensor = torch.tensor(final_confs)
        final_cls_ids_tensor = torch.tensor(final_cls_ids)
        
        # 合并为YOLO格式的boxes (xyxy, conf, cls)
        new_boxes = torch.cat([
            final_boxes_tensor,
            final_confs_tensor.unsqueeze(1),
            final_cls_ids_tensor.unsqueeze(1)
        ], dim=1)
        
        # 创建新的Results对象
        result = Results(
            orig_img=img,
            path=image_file,
            names={0: 'pylon', 1: 'powerline'},
            boxes=new_boxes
        )
        
        # 使用YOLO自带的plot方法绘制
        annotated_img = result.plot()
        
        # 保存结果图片
        output_file = os.path.join(output_path, os.path.basename(image_file))
        cv2.imwrite(output_file, annotated_img)
        
        print(f"已处理: {os.path.basename(image_file)}, 检测到 {len(final_boxes)} 个目标")
    else:
        output_file = os.path.join(output_path, os.path.basename(image_file))
        cv2.imwrite(output_file, img)
        print(f"已处理: {os.path.basename(image_file)}, 去重后无目标")

print(f"检测完成！结果保存在: {output_path}")