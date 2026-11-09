# 主程序入口：一键执行数据划分、训练、评估和可视化
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# 按顺序执行的脚本列表
STEPS = [
    ("数据划分", SRC_DIR / "split_dataset.py"),
    ("模型训练", SRC_DIR / "train.py"),
    ("模型评估", SRC_DIR / "evaluate.py"),
    ("结果可视化", SRC_DIR / "visualize.py"),
]


def run_step(name, script_path):
    """执行单个脚本，失败时立即停止"""
    print(f"\n{'=' * 60}")
    print(f"开始执行: {name}")
    print(f"脚本路径: {script_path}")
    print(f"{'=' * 60}\n")

    result = subprocess.run([sys.executable, str(script_path)], check=True)

    if result.returncode != 0:
        print(f"\n❌ {name} 执行失败，程序终止。")
        sys.exit(1)

    print(f"\n✅ {name} 完成。")


def main():
    print("🚀 息肉分割全流程启动")
    print(f"项目根目录: {BASE_DIR}")

    for name, script in STEPS:
        if not script.exists():
            print(f"\n❌ 找不到脚本: {script}")
            sys.exit(1)
        run_step(name, script)

    print(f"\n{'=' * 60}")
    print("🎉 全部流程执行完毕！")
    print(f"{'=' * 60}")
    print("\n输出文件位置:")
    print(f"  模型权重: {BASE_DIR / 'outputs/models/best_unet.pth'}")
    print(f"  训练日志: {BASE_DIR / 'outputs/logs/train_log.txt'}")
    print(f"  可视化结果: {BASE_DIR / 'outputs/predictions/'}")


if __name__ == "__main__":
    main()
