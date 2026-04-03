"""
多模态融合网络 - 优化版本
针对电塔和电线检测优化的光学+SAR融合
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2


class SpatialAttention(nn.Module):
    """空间注意力模块 - 关注电塔和电线的空间位置"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2, 1),
        )
        
    def forward(self, optical_feat, sar_feat):
        concat = torch.cat([optical_feat, sar_feat], dim=1)
        attention = self.conv(concat)
        attention = F.softmax(attention, dim=1)
        return attention[:, 0:1], attention[:, 1:2]


class ChannelAttention(nn.Module):
    """通道注意力模块 - 关注不同特征通道的重要性"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 4, channels // 2, 1),  # 输入是channels*2（光学+SAR）*2（avg+max）=channels*4
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels * 2, 1),  # 输出是channels*2（光学+SAR）
            nn.Sigmoid()
        )
        
    def forward(self, optical_feat, sar_feat):
        concat = torch.cat([optical_feat, sar_feat], dim=1)
        avg_out = self.avg_pool(concat)
        max_out = self.max_pool(concat)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.mlp(out)
        return attention


class EdgePreserveBlock(nn.Module):
    """边缘保持模块 - 保留电塔和电线的边缘信息"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return self.conv3(x)


class AttentionFusion(nn.Module):
    """
    优化的多模态融合网络
    针对电塔和电线检测优化
    """
    
    def __init__(self, feature_dim=96):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 光学特征提取器 - 更深的网络
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # SAR特征提取器 - 更深的网络
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 注意力机制
        self.spatial_attention = SpatialAttention(feature_dim * 2)
        self.channel_attention = ChannelAttention(feature_dim)
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 边缘保持模块
        self.edge_preserve = EdgePreserveBlock(feature_dim, feature_dim)
        
        # 投影到3通道 - 使用渐进式上采样
        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, optical, sar):
        """
        Args:
            optical: [B, 3, H, W] 光学图像
            sar: [B, 1, H, W] SAR图像
        Returns:
            fused_rgb: [B, 3, H, W] 融合后的伪RGB图像
        """
        # 提取特征
        optical_feat = self.optical_encoder(optical)
        sar_feat = self.sar_encoder(sar)
        
        # 空间注意力
        w_opt, w_sar = self.spatial_attention(optical_feat, sar_feat)
        weighted_opt = optical_feat * w_opt
        weighted_sar = sar_feat * w_sar
        
        # 通道注意力
        channel_weights = self.channel_attention(optical_feat, sar_feat)
        channel_weights = channel_weights.view(-1, self.feature_dim * 2, 1, 1)
        
        # 融合
        fused = torch.cat([weighted_opt, weighted_sar], dim=1)
        fused = fused * channel_weights
        fused = self.fusion_conv(fused)
        
        # 边缘保持
        fused = self.edge_preserve(fused)
        
        # 投影到3通道
        output = self.projection(fused)
        return output


class MultimodalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, optical_dir, sar_dir, img_size=640):
        self.optical_dir = optical_dir
        self.sar_dir = sar_dir
        self.img_size = img_size
        
        self.image_files = [f for f in os.listdir(optical_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 读取光学图像
        optical_path = os.path.join(self.optical_dir, img_name)
        optical = Image.open(optical_path).convert('RGB')
        optical = optical.resize((self.img_size, self.img_size))
        optical = np.array(optical).astype(np.float32) / 255.0
        optical = torch.from_numpy(optical).permute(2, 0, 1)
        
        # 读取SAR图像
        sar_path = os.path.join(self.sar_dir, img_name)
        sar = Image.open(sar_path).convert('L')
        sar = sar.resize((self.img_size, self.img_size))
        sar = np.array(sar).astype(np.float32) / 255.0
        sar = torch.from_numpy(sar).unsqueeze(0)
        
        return {
            'optical': optical,
            'sar': sar,
            'name': img_name
        }


def gradient_loss(pred, target):
    """梯度损失 - 保持边缘清晰度"""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)
    
    return loss_dx + loss_dy


def ssim_loss(pred, target, window_size=11):
    """简化版SSIM损失 - 保持结构相似性"""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    mu_pred = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu_target = F.avg_pool2d(target, window_size, 1, window_size//2)
    
    sigma_pred = F.avg_pool2d(pred ** 2, window_size, 1, window_size//2) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target ** 2, window_size, 1, window_size//2) - mu_target ** 2
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu_pred * mu_target
    
    ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
    
    return 1 - ssim.mean()


def train_fusion_model(optical_dir, sar_dir, epochs=100, batch_size=2, lr=0.0001, patience=20):
    """
    训练融合网络 - 优化版本
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 创建模型 - 使用更大的特征维度
    model = AttentionFusion(feature_dim=96).to(device)
    
    # 创建数据集 - 统一使用640尺寸
    dataset = MultimodalDataset(optical_dir, sar_dir, img_size=640)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 启用混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')
    
    # 早停相关变量
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optical = batch['optical'].to(device)
            sar = batch['sar'].to(device)
            
            # 前向传播 - 使用混合精度
            with torch.cuda.amp.autocast(enabled=device == 'cuda'):
                fused = model(optical, sar)
                
                # 多损失组合
                # 1. L1损失 - 像素级相似
                l1_loss = F.l1_loss(fused, optical * 2 - 1)
                
                # 2. 梯度损失 - 边缘保持
                grad_loss = gradient_loss(fused, optical * 2 - 1)
                
                # 3. SSIM损失 - 结构相似
                ssim = ssim_loss((fused + 1) / 2, optical)
                
                # 组合损失
                loss = l1_loss + 0.5 * grad_loss + 0.3 * ssim
            
            # 反向传播 - 使用混合精度
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, '
                      f'Grad: {grad_loss.item():.4f}, SSIM: {ssim.item():.4f})')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/fusion_model_best.pt')
            print(f"Saved best model at epoch {epoch+1}, loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # 加载最佳模型
    if os.path.exists('models/fusion_model_best.pt'):
        model.load_state_dict(torch.load('models/fusion_model_best.pt', map_location=device, weights_only=True))
        print(f"Loaded best model from epoch {best_epoch+1}")
    
    # 保存最终模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/fusion_model.pt')
    print("Model saved to models/fusion_model.pt")
    
    return model


def generate_fused_images(optical_dir, sar_dir, output_dir, model_path='models/fusion_model.pt', img_size=640):
    """
    使用训练好的模型生成融合图像
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AttentionFusion(feature_dim=96).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random weights")
    
    dataset = MultimodalDataset(optical_dir, sar_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f'Processing batch {batch_idx+1}/{total_batches}...')
            
            optical = batch['optical'].to(device)
            sar = batch['sar'].to(device)
            names = batch['name']
            
            fused = model(optical, sar)
            
            # 保存 - 将[-1, 1]映射到[0, 255]
            fused_np = ((fused.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            for i, name in enumerate(names):
                img = fused_np[i].transpose(1, 2, 0)
                output_path = os.path.join(output_dir, name)
                cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Fused images saved to {output_dir}")
    print(f"Total images: {len(dataset)}")


def main():
    """主函数"""
    
    config = {
        'optical': 'data/optical',
        'sar': 'data/sar',
        'output': 'data/fused',
        'img_size': 640
    }
    
    print("=" * 60)
    print("Multimodal Fusion - Optimized Version")
    print("=" * 60)
    
    model_path = 'models/fusion_model.pt'
    if os.path.exists(model_path):
        print("\n[Step 1/2] Found existing fusion model, skipping training...")
        print(f"Using existing model: {model_path}")
    else:
        print("\n[Step 1/2] Training fusion network...")
        print("-" * 60)
        print("Training with:")
        print("  - Feature dim: 96")
        print("  - Image size: 640")
        print("  - Batch size: 2")
        print("  - Epochs: 100")
        print("  - Loss: L1 + Gradient + SSIM")
        print("  - Mixed precision: Enabled")
        print("-" * 60)
        model = train_fusion_model(
            config['optical'],
            config['sar'],
            epochs=100,
            batch_size=2
        )
    
    print("\n[Step 2/2] Generating fused images...")
    print("-" * 60)
    generate_fused_images(
        config['optical'],
        config['sar'],
        config['output'],
        model_path='models/fusion_model.pt',
        img_size=config['img_size']
    )
    
    print("\n" + "=" * 60)
    print("Fusion completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
