# 训练脚本：训练 U-Net 并保存最佳模型和日志
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PolypDataset
from unet import UNet

BASE_DIR = Path(__file__).resolve().parent.parent

pretrained_path = BASE_DIR / "outputs/models/best_unet.pth"

image_dir = BASE_DIR / "data/images"
mask_dir = BASE_DIR / "data/masks"
split_dir = BASE_DIR / "splits"
model_dir = BASE_DIR / "outputs/models"
log_dir = BASE_DIR / "outputs/logs"

model_dir.mkdir(exist_ok=True)
log_dir.mkdir(exist_ok=True)


def dice_coeff(pred, target, smooth=1e-6):
    """Dice 系数：衡量预测与真实掩膜的重叠度"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    """IoU：交并比，分割任务常用指标"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 训练集启用数据增强
    train_dataset = PolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=split_dir / "train.txt",
        image_size=256,
        augment=True,
    )
    # 验证集不增强
    val_dataset = PolypDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=split_dir / "val.txt",
        image_size=256,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1, base=64).to(device)

    # 二值分割使用 BCEWithLogitsLoss，内部已包含 Sigmoid
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 如果存在上一轮最佳模型，就加载作为初始权重
    if pretrained_path.exists():
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("加载上一轮最佳模型作为初始权重：", pretrained_path)
    else:
        print("未找到预训练权重，从随机初始化开始训练")

    epochs = 50
    best_dice = 0.0
    patience = 6  # 连续 6 轮不提升就停止
    no_improve = 0  # 记录连续未提升轮数
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段必须切换到 eval 模式，关闭 BatchNorm/Dropout 的训练行为
        model.eval()
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                val_dice += dice_coeff(probs, masks).item()
                val_iou += iou_score(probs, masks).item()

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        log_line = (
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} "
            f"| Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
        )
        print(log_line)

        with open(log_dir / "train_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_dir / "best_unet.pth")
            print(f"  -> 保存最佳模型，Dice={best_dice:.4f}")
        else:
            no_improve += 1
            print(f"  -> 未提升，连续 {no_improve}/{patience} 轮")

        if no_improve >= patience:
            print(f"早停触发：验证 Dice 连续 {patience} 轮未提升，停止训练。")
            print(f"最佳模型来自第 {best_epoch} 轮，Dice={best_dice:.4f}")
            break

    print("训练完成，最佳模型已保存至 outputs/models/best_unet.pth")


if __name__ == "__main__":
    main()
