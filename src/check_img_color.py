import cv2
import tifffile

path = r"D:\PythonProgram\polyp-segmentation\data\images\1.tif"

img = tifffile.imread(path)
print("tifffile shape:", img.shape)
print("tifffile dtype:", img.dtype)
print("tifffile min/max:", img.min(), img.max())

if img.ndim == 3 and img.shape[2] == 3:
    print("tifffile 读出了 3 通道彩色图")
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
elif img.ndim == 2:
    print("tifffile 读出了单通道灰度图")
    img_bgr = img
else:
    print("tifffile 读出形状:", img.shape)
    img_bgr = img

cv2.imwrite("check_tifffile.png", img_bgr)
print("已保存 check_tifffile.png")
