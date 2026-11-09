# 基于 U-Net 的结肠镜息肉分割

## 1. 项目概述

本项目实现了一个基于 U-Net 的结肠镜息肉语义分割模型，给定结肠镜图像，模型输出像素级二值掩膜，区分"息肉区域"和"背景"。项目完整覆盖了数据划分、模型训练、评估和可视化全流程，可作为图像算法/医学影像算法岗位的求职作品。

## 2. 问题定义

**任务类型：** 医学图像语义分割（二值分割）

**输入：** 结肠镜图像（RGB，原始分辨率）

**输出：** 像素级二值掩膜（息肉区域为前景 1，背景为 0）

**评估指标：** Dice 系数、IoU（交并比）、准确率、召回率、特异性

## 3. 数据集

- **数据集名称：** CVC-ClinicDB
- **来源：** MICCAI 2015 自动息肉检测挑战赛训练阶段官方数据集
- **规模：** 共 612 张结肠镜图像及对应息肉分割掩膜
- **格式：** .tif 格式（原始图像和掩膜均为 TIFF 格式）
- **下载：** [OpenDataLab](https://opendatalab.org.cn/OpenDataLab/CVC-ClinicDB) 或 [阿里云天池](https://tianchi.aliyun.com/dataset/93690)

### 数据划分

| 集合 | 样本数 | 说明 |
|------|--------|------|
| 训练集 | 428 | 用于模型参数更新 |
| 验证集 | 91 | 用于训练过程监控和早停判断 |
| 测试集 | 93 | 仅用于最终评估，不参与任何训练决策 |

划分比例约为 7 : 1.5 : 1.5，随机种子固定以保证可复现。划分文件保存在 `splits/` 目录下。

## 4. 环境依赖

| 依赖 | 版本要求 | 作用 |
|------|----------|------|
| Python | >= 3.8 | 项目主语言 |
| PyTorch | >= 1.9 | 深度学习框架，构建 U-Net、训练模型 |
| CUDA / cuDNN | 与 PyTorch GPU 版匹配 | GPU 加速训练（无 GPU 也可用 CPU） |
| tifffile | >= 2020 | 读取 .tif/.tiff 格式医学图像 |
| NumPy | >= 1.19 | 数值计算、数组操作 |
| Matplotlib | >= 3.3 | 绘制训练曲线、可视化 |
| scikit-learn | >= 0.24 | 数据集划分、辅助评估 |
| Pillow | >= 8.0 | 基础图像 IO（仅用于非 tif 格式支持） |
| tqdm | >= 4.50 | 训练进度条显示 |
| MATLAB 2019 | 可选 | 数据校验、指标交叉验证（需 Image Processing Toolbox） |

> **关于图像读取库的选择说明：**
>
> 本项目统一使用 `tifffile` 作为图像读取库，而非常用的 `PIL`（Pillow）或 `OpenCV`，原因如下：
>
> 1. **PIL 无法正常读取 .tif 图像：** CVC-ClinicDB 数据集的原始图像和掩膜均为 .tif 格式，PIL 在部分系统环境下读取 .tif 文件会出现解码错误或返回异常数据。
> 2. **OpenCV 读取彩色图像会转为灰度：** OpenCV 的 `cv2.imread` 默认将彩色图像转换为 BGR 三通道格式，在某些使用场景下会导致颜色信息丢失或通道顺序不一致的问题。
> 3. **tifffile 的优势：** `tifffile` 专门针对 TIFF 格式优化，能够稳定读取多通道、高精度（如 16-bit）的医学图像，且返回的数组维度与预期一致，适合医学图像分割项目的需求。
>
> 因此，本项目中所有涉及图像文件读取的代码（`dataset.py`、`visualize.py`、`evaluate.py` 等）均使用 `tifffile.imread` / `tifffile.imwrite`，不再依赖 PIL 或 OpenCV 进行图像读写。

### 环境安装

```bash
# 创建虚拟环境
conda create -n polyp_seg python=3.8 -y
conda activate polyp_seg

# 安装 PyTorch（CUDA 11.8 示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install tifffile numpy matplotlib scikit-learn Pillow tqdm

# 或使用 requirements.txt
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
torch>=1.9
torchvision
tifffile
numpy
matplotlib
scikit-learn
Pillow
tqdm
```

## 5. 模型架构

**模型名称：** U-Net

**网络结构：**

- **输入：** 3×256×256（RGB 三通道，256×256 分辨率）
- **编码器（Contracting Path）：** 4 层下采样，通道数 64→128→256→512
- **瓶颈层（Bottleneck）：** 1024 通道
- **解码器（Expansive Path）：** 4 层上采样，通道数 512→256→128→64
- **跳跃连接（Skip Connections）：** 编码器每层特征图通过卷积调整后与解码器对应层拼接
- **输出层：** 1×1 卷积 + Sigmoid 激活，输出单通道概率图

## 6. 训练配置

| 超参数 | 取值 |
|--------|------|
| 损失函数 | BCELoss + DiceLoss |
| 优化器 | Adam |
| 学习率 | 1e-3 |
| Batch Size | 8 |
| 总 Epoch 数 | 50 |
| 输入尺寸 | 256×256 |
| 数据增强 | 随机翻转、旋转、缩放 |
| 随机种子 | 固定（可复现） |
| 模型保存策略 | 保存验证集 Dice 最高的模型权重 |
| 计算设备 | CUDA（GPU） |

### 训练过程记录

训练共 50 个 Epoch，验证集最佳指标出现在第 49 轮：

| 指标 | 最佳值 | 出现轮次 |
|------|--------|----------|
| Val Dice | 0.8661 | Epoch 49 |
| Val IoU | 0.7700 | Epoch 49 |
| Train Loss（最终） | 0.0656 | Epoch 50 |

最佳模型权重保存在 `outputs/models/best_unet.pth`。

## 7. 评估结果

### 测试集评估

在未见过的测试集（93 张图像）上评估最佳模型：

| 指标 | 结果 |
|------|------|
| **Test Dice** | **0.8812** |
| **Test IoU** | **0.7938** |

> 注：测试集指标高于验证集最佳指标，说明模型在未见数据上仍保持了良好的泛化能力。

### 可视化结果

从测试集中按四类场景各挑选了典型样本进行可视化展示（原图、真实掩膜、预测结果、叠加对比图）：

| 场景类型 | 样本文件 | Dice 值 | 说明 |
|----------|----------|---------|------|
| 正常分割成功 | 494.tif | 0.9735 | 息肉区域完整分割，边界清晰 |
| 边界模糊样本 | 300.tif | 0.6994 | 息肉边界模糊导致分割不完整 |
| 小目标样本 | 608.tif | 0.7910 | 小息肉目标分割效果尚可 |
| 失败案例 | 25.tif | 0.1160 | 分割严重失败，需进一步分析原因 |

可视化结果保存在 `outputs/predictions/` 目录下。

## 8. 项目结构

```
polyp-segmentation/
├── data/
│   ├── images/          # 原始结肠镜图像
│   └── masks/           # 真实分割掩膜
├── splits/
│   ├── train.txt        # 训练集文件列表
│   ├── val.txt          # 验证集文件列表
│   └── test.txt         # 测试集文件列表
├── src/
│   ├── dataset.py       # 数据集加载类
│   ├── unet.py          # U-Net 模型定义
│   ├── split_dataset.py # 数据集划分脚本
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── visualize.py     # 可视化脚本
├── outputs/
│   ├── models/          # 模型权重
│   ├── logs/            # 训练日志
│   └── predictions/     # 可视化结果
├── matlab/              # MATLAB 辅助脚本（可选）
├── requirements.txt     # Python 依赖
└── README.md            # 项目说明
```

## 9. 运行方式

### 一键执行全流程

```bash
cd D:\PythonProgram\polyp-segmentation
python main.py
```

依次执行：数据划分 → 模型训练 → 模型评估 → 结果可视化

### 单独执行各步骤

```bash
# 数据划分
python src/split_dataset.py

# 模型训练
python src/train.py

# 模型评估
python src/evaluate.py

# 结果可视化
python src/visualize.py
```

## 10. 不足与改进方向

### 当前不足

1. **数据量较小：** CVC-ClinicDB 仅 612 张图像，模型泛化能力受限。
2. **单中心数据：** 数据来自单一来源，未覆盖不同内镜设备、不同医院的光照和成像差异。
3. **未做临床部署验证：** 模型仅在离线测试集上评估，未在实际内镜系统中实时推理验证。
4. **未与更强基线充分对比：** 如 nnU-Net、Swin-UNet、TransUNet 等先进分割模型未做横向对比。
5. **小目标和边界问题：** 小息肉分割精度有限，边界模糊样本的 Dice 较低。

### 改进方向

1. 引入更多公开数据集（如 Kvasir、CVC-ColonDB）进行多数据集训练。
2. 尝试数据增强策略升级（Mixup、CutOut、Coarse Dropout 等）。
3. 引入注意力机制（如 Attention U-Net）或 Transformer 模块增强特征提取。
4. 使用更先进的损失函数（如 Focal Loss、Boundary Loss）处理类别不平衡。
5. 模型量化与 ONNX/TensorRT 部署，实现实时推理。

## 11. 面试准备要点

- U-Net 为什么适合医学图像分割？（跳跃连接保留空间信息、编码器-解码器结构）
- Dice Loss 和 BCE Loss 各自的作用？（Dice 处理类别不平衡，BCE 提供像素级交叉熵约束）
- 数据增强如何保证图像和掩膜同步变换？（使用相同的随机参数对图像和掩膜同时变换）
- Dice 和 IoU 的关系？（Dice = 2TP/(2TP+FP+FN)，IoU = TP/(TP+FP+FN)，Dice 略大于 IoU）
- 小样本情况下如何防止过拟合？（数据增强、权重衰减、早停、Dropout）
- 医学图像分割中类别不平衡怎么处理？（Dice Loss、Focal Loss、加权交叉熵）
- 本项目中 tifffile 替代 PIL/OpenCV 的原因？（PIL 无法稳定读取 .tif，OpenCV 彩色转灰度问题）
