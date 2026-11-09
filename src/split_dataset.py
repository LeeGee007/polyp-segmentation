# 数据划分脚本：把 data/images 下的图像按 7:1.5:1.5 划分为 train/val/test
import random
from pathlib import Path

# 当前脚本位于 src/，parent.parent 即项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

image_dir = BASE_DIR / "data/images"
mask_dir = BASE_DIR / "data/masks"
split_dir = BASE_DIR / "splits"

# 创建 splits 目录，exist_ok=True 表示目录已存在时不报错
split_dir.mkdir(exist_ok=True)

# 只保留常见图像后缀的文件名
files = sorted([
    f.name for f in image_dir.iterdir()
    if f.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
])

# 固定随机种子，保证每次划分结果一致
random.seed(42)
random.shuffle(files)

train_ratio = 0.7
val_ratio = 0.15

n = len(files)
n_train = int(n * train_ratio)
n_val = int(n * val_ratio)

train_files = files[:n_train]
val_files = files[n_train: n_train + n_val]
test_files = files[n_train + n_val:]

for name, split in [("train", train_files), ("val", val_files), ("test", test_files)]:
    path = split_dir / f"{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for file_name in split:
            f.write(file_name + "\n")
    print(f"{name}: {len(split)} samples -> {path}")
