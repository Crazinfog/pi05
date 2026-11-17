#!/bin/bash
# 在 Orin 的应用菜单里放两个图标：“ECO65 三维示教”和“ECO65 三维示教（离线模拟）”。普通用户运行，不需要 sudo。
# 2026-11-04 用户：桌面不要快捷方式——启动入口是 ~/teleop 里的“启动三维示教.sh”“启动三维示教_离线模拟.sh”，
# 这个脚本只放应用菜单（按 Super 键搜 ECO65），并删掉以前放在桌面上的两个图标。
# 点图标会开一个终端窗口跑服务（看日志、Ctrl+C 关服务），网页用 Firefox 打开并自动全屏（右上角按钮切窗口 / 最小化，F11 也行）。
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
PYV=${TELEOP_PY:-$HOME/miniforge3/envs/teleop/bin/python}
chmod +x "$DIR/start_twin.sh"
APPS=$HOME/.local/share/applications
DESK=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
mkdir -p "$APPS"
chmod +x "$DIR/启动三维示教.sh" "$DIR/启动三维示教_离线模拟.sh" 2>/dev/null || true
make() {   # 文件名 显示名 参数
  cat > "$APPS/$1.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$2
Comment=ECO65 三维示教网页（采数据用）
Exec=gnome-terminal --title="$2" -- bash -c 'TELEOP_PY="$PYV" "$DIR/start_twin.sh" $3; echo; read -p "服务已经停了，按回车关闭这个窗口"'
Icon=applications-science
Terminal=false
Categories=Science;Robotics;
EOF
  chmod +x "$APPS/$1.desktop"
  rm -f "$DESK/$1.desktop"                 # 以前放在桌面上的去掉
}
make eco65-twin "ECO65 三维示教" ""
make eco65-twin-offline "ECO65 三维示教（离线模拟）" "--offline"
echo "应用菜单里加好了两个图标（按 Super 键搜 ECO65）；桌面上的去掉了。启动入口：$DIR/启动三维示教.sh"
