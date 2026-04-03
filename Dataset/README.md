# sar-powerline-detection
SAR-based Powerline Damage Detection System, integrating multimodal fusion technology for automatic powerline detection and damage assessment.

## Project Introduction
A SAR-based powerline damage detection system utilizing multimodal fusion technology for automatic powerline detection and damage assessment. The project focuses on powerline detection, with damage assessment under development.

## Project Flow
1. **Data Processing**: Obtain optical images and tower annotations from LSKF-YOLO, add powerline annotations, modify existing tower annotations, convert to grayscale with noise to simulate SAR, then denoise using SAR-BM3D
2. **Multimodal Fusion**: Extract high-dimensional features from optical and SAR images, fuse through attention mechanisms, generate fused pseudo-optical images
3. **Object Detection**: Train YOLO model on fused images for powerline and tower detection
4. **Damage Assessment** (In development): Evaluate powerline damage based on detection results

## Technical Features
- **Multimodal Fusion**: Combines optical texture and SAR structural information
- **Attention Mechanism**: Automatically learns modality importance weights
- **Edge Preservation**: Retains powerline edge information for improved detection
- **Cloud Penetration**: Leverages SAR's cloud-free characteristics for adverse weather detection

## Dataset & Citation
Uses SRSPTD dataset from [LSKF-YOLO](https://github.com/ZX815/LSKF-YOLO), a subset of Electric Transmission and Distribution Infrastructure Imagery Dataset.

Citation:C. Shi, X. Zheng, Z. Zhao, K. Zhang, Z. Su and Q. Lu, "LSKF-YOLO: Large Selective Kernel Feature Fusion Network for Power Tower Detection in High-Resolution Satellite Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024, Art no. 5620116, doi: 10.1109/TGRS.2024.3389056.
