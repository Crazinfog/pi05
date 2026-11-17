#!/bin/bash
# 恢复时要 root 的几项（用户自己输密码运行）：
#   sudo bash restore_root.sh            相机 udev 规则 + 机械臂网口 eth0 固定 IP
#   sudo bash restore_root.sh --firefox  另外把 Firefox 升级到新版（出厂的 Firefox 113 打不开三维网页，只升这一个包）
# 不会 apt upgrade 整机（会覆盖厂家的内核和设备树）。
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$(id -u)" != 0 ]; then echo "要用 sudo 运行：sudo bash $0 $*"; exit 1; fi

echo "1. 奥比中光相机的 udev 规则（不装的话普通用户打不开相机）"
cp "$HERE/system/99-obsensor-libusb.rules" /etc/udev/rules.d/
udevadm control --reload-rules && udevadm trigger
echo "   装好了；相机拔插一下生效"

echo "2. 机械臂网口：eth0 固定 192.168.1.101/24，不当默认路由（上网照样走 Wi-Fi）"
if nmcli -t -f NAME con show | grep -qx "Wired connection 1"; then
  CON="Wired connection 1"
elif nmcli -t -f NAME con show | grep -qx "eco65-arm"; then
  CON="eco65-arm"
else
  nmcli con add type ethernet ifname eth0 con-name eco65-arm >/dev/null
  CON="eco65-arm"
fi
nmcli con mod "$CON" connection.interface-name eth0 ipv4.method manual ipv4.addresses 192.168.1.101/24 \
  ipv4.gateway "" ipv4.never-default yes ipv6.method ignore
nmcli con up "$CON" >/dev/null 2>&1 && echo "   设好了，已连上" || echo "   设好了（网线没插或机械臂没开时连不上，插上会自动连）"

V=$(firefox --version 2>/dev/null | grep -o '[0-9]\+' | head -1)
if [ "$1" = "--firefox" ]; then
  echo "3. 升级 Firefox（只装 firefox 这一个包）"
  apt-get install -y firefox
elif [ -n "$V" ] && [ "$V" -lt 120 ]; then
  echo "3. 注意：Firefox 是 $V 版，太旧打不开三维网页。要升级就再跑：sudo bash $0 --firefox"
fi
echo "完成。"
