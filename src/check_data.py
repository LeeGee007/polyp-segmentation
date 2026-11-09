import os

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_path = "../data/images/1.tif"
mask_path = "../data/masks/1.tif"

print("当前目录:", os.getcwd())
print("图像是否存在:", os.path.exists(image_path))
print("掩膜是否存在:", os.path.exists(mask_path))

if os.path.exists(image_path):
    print("图像文件大小:", os.path.getsize(image_path), "bytes")

if os.path.exists(mask_path):
    print("掩膜文件大小:", os.path.getsize(mask_path), "bytes")

if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
    try:
        img = Image.open(image_path)
        print("图像格式:", img.format)
        print("图像尺寸:", img.size)
        print("图像模式:", img.mode)
    except Exception as e:
        print("Pillow 读取失败:", e)
