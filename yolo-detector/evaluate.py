from ultralytics import YOLO
import os
import glob
import numpy as np

def load_labels(label_file):
    """加载标签文件"""
    labels = []
    if not os.path.exists(label_file):
        return labels
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                # YOLO格式: x_center, y_center, width, height (归一化)
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                # 转换为xyxy格式 (归一化)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                labels.append([cls_id, x1, y1, x2, y2])
    return labels

def compute_iou(box1, box2):
    """计算两个框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制（NMS）去重"""
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
        
        # 计算当前框与其余框的IOU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = []
        for box in other_boxes:
            ious.append(compute_iou(current_box, box))
        ious = np.array(ious)
        
        # 保留IOU小于阈值的框
        mask = ious < iou_threshold
        indices = indices[1:][mask]
    
    return keep

def evaluate_model(model_path, val_images_path, val_labels_path, conf_threshold=0.4, iou_threshold=0.3):
    """评价模型"""
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有验证集图片
    image_files = glob.glob(os.path.join(val_images_path, '*.jpg'))
    
    total_true_positives = 0
    total_predicted = 0
    total_ground_truth = 0
    
    print(f"开始评价模型，共 {len(image_files)} 张验证图片...")
    
    for image_file in image_files:
        # 获取对应标签文件
        image_name = os.path.basename(image_file)
        label_file = os.path.join(val_labels_path, image_name.replace('.jpg', '.txt'))
        
        # 加载真实标签
        ground_truths = load_labels(label_file)
        total_ground_truth += len(ground_truths)
        
        # 执行检测
        results = model(image_file, conf=conf_threshold)
        
        # 获取检测结果
        boxes = results[0].boxes
        predictions = []
        
        if len(boxes) > 0:
            # 提取检测框信息
            xyxy = boxes.xyxy.cpu().numpy()  # 检测框坐标 (像素值)
            confs = boxes.conf.cpu().numpy()  # 置信度
            cls_ids = boxes.cls.cpu().numpy().astype(int)  # 类别ID
            
            # 获取图片尺寸
            img_h = results[0].orig_shape[0]
            img_w = results[0].orig_shape[1]
            
            # 转换为归一化坐标
            normalized_boxes = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                # 归一化
                x1_norm = x1 / img_w
                y1_norm = y1 / img_h
                x2_norm = x2 / img_w
                y2_norm = y2 / img_h
                normalized_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            # 按类别分别进行NMS去重
            unique_classes = np.unique(cls_ids)
            for cls_id in unique_classes:
                cls_mask = cls_ids == cls_id
                cls_boxes = np.array(normalized_boxes)[cls_mask]
                cls_confs = confs[cls_mask]
                
                # 对该类别的框进行NMS
                keep_indices = nms(cls_boxes, cls_confs, iou_threshold=0.4)
                
                # 保留去重后的框
                for idx in keep_indices:
                    box = cls_boxes[idx]
                    conf = cls_confs[idx]
                    predictions.append([cls_id, box[0], box[1], box[2], box[3], conf])
        
        total_predicted += len(predictions)
        
        # 匹配预测框和真实框
        matched_gt = set()
        for pred in predictions:
            pred_cls, pred_x1, pred_y1, pred_x2, pred_y2, _ = pred
            pred_box = [pred_x1, pred_y1, pred_x2, pred_y2]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                if pred_cls != gt_cls:
                    continue
                
                gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]
                iou = compute_iou(pred_box, gt_box)
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                total_true_positives += 1
                matched_gt.add(best_gt_idx)
    
    # 计算精确率和召回率
    if total_predicted > 0:
        precision = total_true_positives / total_predicted
    else:
        precision = 0.0
    
    if total_ground_truth > 0:
        recall = total_true_positives / total_ground_truth
    else:
        recall = 0.0
    
    # 计算F1分数
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    print(f"\n评价结果:")
    print(f"真实目标数: {total_ground_truth}")
    print(f"预测目标数: {total_predicted}")
    print(f"正确检测数: {total_true_positives}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    
    return precision, recall, f1_score

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

if __name__ == '__main__':
    # 查找最新模型
    model_path = find_latest_model()
    if not model_path:
        print("错误：未找到训练模型！")
        print("请先运行 train.py 训练模型")
        exit(1)
    
    print(f"使用模型: {model_path}")
    
    # 验证集路径
    val_images_path = 'data/val/images'
    val_labels_path = 'data/val/labels'
    
    # 评价模型
    evaluate_model(model_path, val_images_path, val_labels_path, 
                  conf_threshold=0.3, iou_threshold=0.3)