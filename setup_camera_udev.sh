#!/bin/bash
# 让普通用户能打开两台奥比中光相机（Gemini 335 / 305）。需要 root：sudo bash ~/teleop/setup_camera_udev.sh
set -e
SRC=/home/whut/miniforge3/envs/teleop/lib/python3.10/site-packages/pyorbbecsdk/shared/99-obsensor-libusb.rules
install -m 644 "$SRC" /etc/udev/rules.d/99-obsensor-libusb.rules
udevadm control --reload-rules
udevadm trigger
echo "相机规则装好了：拔插一下相机 USB 线，或直接重启电脑。"
