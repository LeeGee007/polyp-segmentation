# ---------- 导入依赖 ----------
import os  # 文件路径、目录操作
import random  # 随机数工具，用于数据增强
from pathlib import Path

import cv2  # OpenCV，用于图像缩放、颜色转换、掩膜读取
import numpy as np  # NumPy 数组/矩阵运算
import tifffile  # 关键新增：用于正确读取 RGB TIFF 原图
import torch
from torch.utils.data import Dataset

# ---------- 计算项目根目录 ----------
# __file__ 是当前脚本的绝对路径，例如 D:\PythonProgram\polyp-segmentation\src\dataset.py
# .resolve() 解析成绝对路径，.parent 取上一级目录
# 最终得到项目根目录：D:\PythonProgram\polyp-segmentation
BASE_DIR = Path(__file__).resolve().parent.parent


# ---------- 定义数据集类 ----------
class PolypDataset(Dataset):
    """CVC-ClinicDB 息肉分割数据集

    关键改动：
    - 原图使用 tifffile.imread 读取，避免 OpenCV 把 RGB TIFF 误读成灰度。
    - tifffile 读出的是 RGB 顺序，不需要再做 BGR->RGB 转换。
    - 掩膜仍用 cv2.imread 读取（掩膜通常是单通道 PNG/TIF，OpenCV 读取可靠）。
    """

    def __init__(self, image_dir, mask_dir, split_file=None, image_size=256, augment=False):
        """
        image_dir:  原始图像目录，例如 data/CVC-ClinicDB/images
        mask_dir:   掩膜目录，例如 data/CVC-ClinicDB/masks
        split_file: 数据划分文件，例如 splits/train.txt；为 None 则使用目录下全部文件
        image_size: 图像统一 resize 到的尺寸（默认 256）
        augment:    是否启用数据增强（随机翻转、亮度扰动）
        """
        self.image_dir = str(image_dir)
        self.mask_dir = str(mask_dir)
        self.split_file = split_file
        self.image_size = image_size
        self.augment = augment

        # 读取样本文件名列表
        if split_file is not None and os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                self.file_names = [line.strip() for line in f if line.strip()]
        else:
            self.file_names = sorted(os.listdir(self.image_dir))

        # 初始化后校验图像与掩膜是否一一对应
        self._check_files()

    def _check_files(self):
        """检查 file_names 中的每个样本在掩膜目录中是否存在同名文件"""
        for name in self.file_names:
            mask_path = os.path.join(self.mask_dir, name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"掩膜文件不存在: {mask_path}")

    def __len__(self):
        """返回数据集样本总数，DataLoader 据此判断一个 epoch 的迭代次数"""
        return len(self.file_names)

    def __getitem__(self, idx):
        """根据索引 idx 返回单个样本：(image_tensor, mask_tensor)

        image: (3, H, W) float32，像素值归一化到 [0, 1]
        mask:  (1, H, W) float32，二值掩膜，只包含 0 和 1
        """
        file_name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        # ===== 关键修改：用 tifffile 读取原图 =====
        # tifffile 能正确解析 RGB TIFF，返回 (H, W, 3) uint8，RGB 顺序
        # OpenCV 的 cv2.imread 对某些 TIFF 会把 RGB 误判为灰度并复制成 3 通道
        image = tifffile.imread(image_path)

        # 统一通道处理：确保最终是 RGB 三通道
        if image.ndim == 3 and image.shape[2] == 3:
            pass  # 已经是 RGB，无需处理
        elif image.ndim == 2:
            # 极少数情况下是灰度 TIFF，补成三通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            # 带 alpha 通道的 RGBA，丢弃 alpha
            image = image[:, :, :3]

        # 统一转为 uint8（tifffile 通常已返回 uint8，这里做防御性处理）
        if image.dtype != np.uint8:
            if image.dtype == np.uint16:
                image = (image / 256.0).astype(np.uint8)
            elif np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        # ==========================================

        # 掩膜仍用 OpenCV 读取（单通道二值图，OpenCV 足够可靠）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩膜: {mask_path}")

        # 统一缩放到 image_size x image_size
        # 图像用双线性插值（INTER_LINEAR），掩膜用最近邻（INTER_NEAREST）避免产生中间值
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 图像归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0

        # 掩膜二值化：大于 127 的像素设为 1，否则为 0
        mask = (mask > 127).astype(np.float32)

        # 数据增强（随机翻转 + 亮度扰动）
        if self.augment:
            image, mask = self._augment(image, mask)

        # 通道维度转换：(H, W, C) -> (C, H, W)，适配 PyTorch 模型输入格式
        image = image.transpose(2, 0, 1)
        mask = mask[np.newaxis, :, :]  # (H, W) -> (1, H, W)

        # 转成 torch.Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask

    def _augment(self, image, mask):
        """随机水平/垂直翻转 + 随机亮度扰动"""
        # 随机水平翻转（概率 50%）
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # 随机垂直翻转（概率 50%）
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # 随机亮度扰动：像素值在 0.8~1.2 倍之间随机缩放
        brightness = random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 1)

        return image, mask


# ---------- 主程序入口 ----------
if __name__ == "__main__":
    # 创建数据集对象
    dataset = PolypDataset(
        image_dir=BASE_DIR / "data/images",
        mask_dir=BASE_DIR / "data/masks",
        split_file=None,
        image_size=256,
        augment=False,
    )

    print(f"数据集长度: {len(dataset)}")

    # 取第 0 个样本，验证读取是否正常
    image, mask = dataset[0]
    print(f"图像形状: {image.shape}")  # 期望 (3, 256, 256)
    print(f"掩膜形状: {mask.shape}")  # 期望 (1, 256, 256)
    print(f"掩膜最小值: {mask.min()}")  # 期望 0.0
    print(f"掩膜最大值: {mask.max()}")  # 期望 1.0
