"""训练和部署共用的图像处理：录制时存的是 640×480 BGR 的 JPEG，送进模型前都按这里的办法裁剪、缩放。
转换数据（convert_to_lerobot.py）和以后真机部署（Orin 上把相机画面发给模型）必须用同一套，不然模型看到的画面对不上。

- 第三视角：按工作区裁一个正方形，再缩到 224×224。
  2026-11-06 相机挪到工作区对面（底座坐标 (−0.14, 0.92, 0.83) m、朝底座、俯 51°，11-06 18:56 标定）、工作区改成
  58×36 cm（work_area.json：x −0.29~+0.29、y 0.21~0.57 m），裁剪框按工作区四角在桌面和上方 15 cm 的投影＋四周 5% 余量算：
  x 43~436、y 80~473（393 px），3.3 cm 方块缩完 8~10 像素。之前（11-03 相机、50×30 cm 工作区）是 (10, 100, 380)，
  用旧相机录的数据和训的模型（red19_v1）都对不上新画面了。
  相机挪了、或者工作区换了，要重新算并且重新转换全部数据。
- 腕部：整幅等比缩到 224 宽，上下补黑边（和 openpi 自己的 resize_with_pad 一样）。
- 颜色：OpenCV 解出来是 BGR，模型要 RGB。
"""
import cv2
import numpy as np
from openpi_client import image_tools

IMG = 224                                 # 送进模型的边长（π0.5 的图像输入）
SCENE_CROP = (43, 80, 393)                # 第三视角裁剪：左上角 x、y 和边长（像素，640×480 原图）；2026-11-03~06 用的是 (10, 100, 380)


def decode_bgr(jpeg_bytes):
    return cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)


def scene_to_model(bgr, crop=SCENE_CROP):
    x0, y0, s = crop
    rgb = cv2.cvtColor(bgr[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB)
    return image_tools.resize_with_pad(rgb[None], IMG, IMG)[0]


def wrist_to_model(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return image_tools.resize_with_pad(rgb[None], IMG, IMG)[0]
