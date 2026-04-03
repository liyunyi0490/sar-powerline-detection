# 光学图片转SAR图片工具

## 项目简介

本项目旨在将光学图片转换为模拟SAR（合成孔径雷达）图片，通过添加Speckle噪声来模拟真实SAR图像的特性。

## 项目结构

```
ACD(光学加噪转sar)/
├── convert_to_sar_simple.py  # SAR转换工具
├── README.md                 # 项目说明文件
├── images/                   # 输入光学图片文件夹
└── output/                   # 输出SAR图片文件夹
```

## 核心功能

- 将彩色光学图片转换为灰度图像
- 添加Speckle噪声模拟真实SAR图像特性
- 支持批量处理多个图片
- 通过调整`L`参数控制噪声强度

## 环境配置

### 依赖库

- Python 3.x
- Pillow (PIL) - 图像处理

### 安装依赖

```bash
pip install Pillow
```

## 使用方法

1. 将需要转换的光学图片放入`images/`文件夹中

2. 运行脚本：

```bash
python convert_to_sar_simple.py
```

3. 转换后的SAR图片将保存在`output/`文件夹中

## 噪声参数说明

- `L`参数：多视处理次数，控制噪声强度
  - `L=4` - 噪声较大，更接近真实SAR图像
  - `L=10` - 噪声中等
  - `L=20` - 噪声较小

## 注意事项

1. 确保`images/`文件夹中只包含图片文件（.jpg, .jpeg, .png）
2. `output/`文件夹会自动创建，无需手动创建
3. 转换过程可能需要一定时间，取决于图片数量和大小
4. 转换后的图片为灰度图像，模拟SAR图像的特性

## 示例

### 输入：光学图片
![光学图片示例](images/NZ_Dunedin_103x01.jpg)

### 输出：模拟SAR图片
![SAR图片示例](output/NZ_Dunedin_103x01.jpg)

## 许可证

本项目为开源工具，仅供研究和学习使用。