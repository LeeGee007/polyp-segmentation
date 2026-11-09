# visualize.py
# 可视化脚本：挑选四类典型样本，生成原始图像、真实掩膜、预测结果和叠加对比图
# 图像读写使用 tifffile，不使用 PIL 和 OpenCV

from pathlib import Path

# ---------- 导入依赖库 ----------
import numpy as np
import tifffile  # 用于读写 .tif / .tiff 格式的医学图像
import torch

# 从同目录下的 dataset.py 和 unet.py 导入数据集类和模型
from dataset import PolypDataset
from unet import UNet

# ---------- 路径配置 ----------
# __file__ 是当前脚本文件的路径，.parent.parent 回到项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

image_dir = BASE_DIR / "data/images"  # 原始图像目录
mask_dir = BASE_DIR / "data/masks"  # 真实掩膜目录
split_dir = BASE_DIR / "splits"  # 数据划分文件目录
model_dir = BASE_DIR / "outputs/models"  # 模型权重保存目录
pred_dir = BASE_DIR / "outputs/predictions"  # 可视化结果保存目录

# 创建输出目录，exist_ok=True 表示目录已存在时不会报错
pred_dir.mkdir(parents=True, exist_ok=True)


# ---------- 辅助函数 ----------

def dice_coeff(pred, target, smooth=1e-6):
    """
    计算 Dice 系数（重叠度指标）
    pred:   预测掩膜，numpy 数组，值为 0 或 1
    target: 真实掩膜，numpy 数组，值为 0 或 1
    smooth: 平滑项，防止分母为 0
    返回值范围 [0, 1]，越接近 1 表示分割越准确
    """
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def classify_sample(dice, mask_ratio):
    """
    根据 Dice 系数和真实掩膜占比，将样本归入四类场景之一
    dice:       当前样本的 Dice 值
    mask_ratio: 真实掩膜中前景像素的占比（前景像素数 / 总像素数）
    返回类别字符串，若不属于任何一类则返回 None
    """
    # Dice >= 0.85：分割效果很好，属于成功样本
    if dice >= 0.85:
        return "success"

    # 真实掩膜占比 < 3%：目标非常小，属于小目标样本
    if mask_ratio < 0.03:
        return "small_target"

    # Dice 在 0.50~0.70 之间：分割有一定效果但不完美，通常是边界模糊导致
    if 0.50 <= dice < 0.70:
        return "blurry_boundary"

    # Dice < 0.50：分割效果很差，属于失败案例
    if dice < 0.50:
        return "failure"

    return None


def make_visualization(orig_image, gt_mask, pred_mask):
    """
    生成 2x2 可视化对比图（numpy 数组，HWC 格式，uint8）
    布局：
        左上：原始图像
        右上：真实掩膜（白色前景）
        左下：预测结果（白色前景）
        右下：叠加图（原图 + 绿色预测区域）

    orig_image: 原始图像，(H, W, C)，uint8
    gt_mask:    真实掩膜，(H, W)，值为 0 或 255
    pred_mask:  预测掩膜，(H, W)，值为 0 或 255
    """
    h, w = orig_image.shape[:2]

    # 将一通道掩膜复制成三通道，方便与原图拼接显示
    gt_gray = np.stack([gt_mask] * 3, axis=-1)  # (H, W, 3)
    pred_gray = np.stack([pred_mask] * 3, axis=-1)  # (H, W, 3)

    # 叠加图：复制原图，把预测为前景的像素染成绿色 (R=0, G=255, B=0)
    overlay = orig_image.copy()
    overlay[pred_mask > 0] = [0, 255, 0]

    # 水平拼接：第一行 [原图, 真实掩膜]，第二行 [预测结果, 叠加图]
    top = np.hstack([orig_image, gt_gray])  # (H, 2W, 3)
    bottom = np.hstack([pred_gray, overlay])  # (H, 2W, 3)
    viz = np.vstack([top, bottom])  # (2H, 2W, 3)

    return viz


# ---------- 主函数 ----------

def main():
    # 1. 选择计算设备：有 GPU 用 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载测试集（使用与训练时相同的数据集类，保证预处理一致）
    test_dataset = PolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=split_dir / "test.txt",  # 只加载测试集的文件名
        image_size=256,
        augment=False,  # 测试阶段不做数据增强
    )

    # 3. 创建模型并加载训练好的权重
    model = UNet(in_channels=3, out_channels=1, base=64).to(device)
    model.load_state_dict(torch.load(model_dir / "best_unet.pth", map_location=device))
    model.eval()  # 切换到评估模式，关闭 BatchNorm 等训练特有的行为

    # 4. 遍历测试集，按四类场景收集候选样本
    #    每类最多收集 3 个候选，最后从中挑选最典型的一张
    buckets = {
        "success": [],
        "blurry_boundary": [],
        "small_target": [],
        "failure": [],
    }

    with torch.no_grad():  # 推理阶段不需要计算梯度，节省显存
        for idx in range(len(test_dataset)):
            file_name = test_dataset.file_names[idx]
            image, mask = test_dataset[idx]  # image: (C,H,W), mask: (1,H,W)

            # 增加 batch 维度后送入模型: (1, C, H, W)
            x = image.unsqueeze(0).to(device)
            out = model(x)  # 输出 logits: (1, 1, H, W)
            prob = torch.sigmoid(out).squeeze().cpu().numpy()  # 概率图: (H, W)
            pred = (prob > 0.5).astype(np.float32)  # 二值预测: (H, W)

            # 真实掩膜去掉通道维度: (H, W)
            mask_np = mask.squeeze().numpy()

            # 计算当前样本的 Dice 和掩膜占比
            dice = dice_coeff(pred, mask_np)
            mask_ratio = mask_np.mean()  # 前景像素占比

            # 分类并加入候选池（每类最多 3 个）
            category = classify_sample(dice, mask_ratio)
            if category and len(buckets[category]) < 3:
                buckets[category].append({
                    "file_name": file_name,
                    "dice": dice,
                    "mask_ratio": mask_ratio,
                    # 保存模型输入尺寸的预测结果，后续会 resize 回原图
                    "pred_256": pred,
                    "mask_256": mask_np,
                })

    # 5. 为每个场景挑选最佳样本，生成可视化图并用 tifffile 保存
    labels = {
        "success": "正常分割成功",
        "blurry_boundary": "边界模糊样本",
        "small_target": "小目标样本",
        "failure": "失败案例",
    }

    for category, label in labels.items():
        samples = buckets[category]

        # 如果某类没有符合条件的样本，打印提示并跳过
        if not samples:
            print(f"️  未找到符合条件的「{label}」样本")
            continue

        # 排序挑选最典型的样本：
        #   失败案例选 Dice 最低的；其他类别选 Dice 最高的
        if category == "failure":
            samples.sort(key=lambda s: s["dice"])
        else:
            samples.sort(key=lambda s: -s["dice"])

        sample = samples[0]
        file_name = sample["file_name"]

        # 用 tifffile 读取原始分辨率的图像和掩膜（tifffile 返回 (H, W, C) 或 (H, W)）
        orig_path = str(image_dir / file_name)
        mask_path = str(mask_dir / file_name)

        orig_image = tifffile.imread(orig_path)  # (H, W, C) 或 (H, W)
        gt_mask_raw = tifffile.imread(mask_path)  # (H, W)

        # 确保原始图像是三通道 uint8
        if orig_image.ndim == 2:
            orig_image = np.stack([orig_image] * 3, axis=-1)
        if orig_image.dtype != np.uint8:
            # tifffile 读出的可能是 float，需要归一化后转 uint8
            if orig_image.max() > 1.0:
                orig_image = (orig_image / 255.0).clip(0, 1)
            orig_image = (orig_image * 255).astype(np.uint8)

        # 确保真实掩膜是二值 uint8（0 或 255）
        gt_mask_raw = (gt_mask_raw > 127).astype(np.uint8) * 255

        # 将模型输出的 256x256 预测结果 resize 回原图尺寸（最近邻插值保持二值）
        h, w = orig_image.shape[:2]
        pred_256 = sample["pred_256"]
        mask_256 = sample["mask_256"]

        # 手动最近邻 resize：用 numpy 索引实现，不依赖 OpenCV / PIL
        row_scale = pred_256.shape[0] / h
        col_scale = pred_256.shape[1] / w
        row_idx = np.floor(np.arange(h) * row_scale).astype(int).clip(0, pred_256.shape[0] - 1)
        col_idx = np.floor(np.arange(w) * col_scale).astype(int).clip(0, pred_256.shape[1] - 1)
        pred_full = pred_256[np.ix_(row_idx, col_idx)]  # (H, W)
        pred_full = (pred_full > 0.5).astype(np.uint8) * 255

        # 真实掩膜也 resize 到原图尺寸（用于可视化展示，可能与原标注略有差异）
        row_scale_m = mask_256.shape[0] / h
        col_scale_m = mask_256.shape[1] / w
        row_idx_m = np.floor(np.arange(h) * row_scale_m).astype(int).clip(0, mask_256.shape[0] - 1)
        col_idx_m = np.floor(np.arange(w) * col_scale_m).astype(int).clip(0, mask_256.shape[1] - 1)
        gt_mask_full = mask_256[np.ix_(row_idx_m, col_idx_m)]
        gt_mask_full = (gt_mask_full > 0.5).astype(np.uint8) * 255

        # 生成 2x2 可视化对比图
        viz = make_visualization(orig_image, gt_mask_full, pred_full)

        # 用 tifffile 保存为 .tif 文件
        save_name = f"{category}_{Path(file_name).stem}.tif"
        save_path = pred_dir / save_name
        tifffile.imwrite(str(save_path), viz)

        print(f" [{label}] {file_name}  Dice={sample['dice']:.4f}  ->  {save_path}")

    print("\n 可视化完成，结果保存在 outputs/predictions/ 目录下")


# ---------- 程序入口 ----------
if __name__ == "__main__":
    main()

