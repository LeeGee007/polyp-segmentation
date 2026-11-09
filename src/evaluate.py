# 评估脚本：在测试集上计算 Dice 和 IoU
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PolypDataset
from unet import UNet

BASE_DIR = Path(__file__).resolve().parent.parent

image_dir = BASE_DIR / "data/images"
mask_dir = BASE_DIR / "data/masks"
split_dir = BASE_DIR / "splits"
model_dir = BASE_DIR / "outputs/models"


def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = PolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=split_dir / "test.txt",
        image_size=256,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1, base=64).to(device)

    # 加载训练阶段保存的最佳模型
    model.load_state_dict(torch.load(model_dir / "best_unet.pth", map_location=device))
    model.eval()

    dice_list = []
    iou_list = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            dice_list.append(dice_coeff(probs, masks).item())
            iou_list.append(iou_score(probs, masks).item())

    mean_dice = sum(dice_list) / len(dice_list)
    mean_iou = sum(iou_list) / len(iou_list)

    print(f"Test Dice: {mean_dice:.4f}")
    print(f"Test IoU:  {mean_iou:.4f}")


if __name__ == "__main__":
    main()
