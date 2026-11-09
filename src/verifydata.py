import os
from pathlib import Path

import tifffile

BASE_DIR = Path(__file__).resolve().parent.parent

image_dir = BASE_DIR / "data/images"
mask_dir = BASE_DIR / "data/masks"

image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

assert len(image_files) == len(mask_files), "图像和掩膜数量不一致"
assert image_files == mask_files, "图像和掩膜文件名不一致"

print(f"图像数量: {len(image_files)}")
print(f"掩膜数量: {len(mask_files)}")

for name in image_files[:5]:
    image_path = os.path.join(image_dir, name)
    mask_path = os.path.join(mask_dir, name)

    # 使用 tifffile 读取图像和掩膜
    img = tifffile.imread(image_path)
    msk = tifffile.imread(mask_path)

    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    if msk is None:
        raise FileNotFoundError(f"无法读取掩膜: {mask_path}")

    # 如果是彩色图像，简单计算均值转为灰度（或者直接打印 shape 观察）
    # 注意：tifffile 读取的彩色图是 RGB 格式，通道在最后 (H, W, C)
    if len(img.shape) == 3:
        gray_img = img.mean(axis=-1)
        print(name, img.shape, gray_img.shape, msk.shape, msk.min(), msk.max())
    else:
        print(name, img.shape, msk.shape, msk.min(), msk.max())
