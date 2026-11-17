#!/bin/bash
# 启动 ECO65 三维示教（连机械臂）。在 ~/teleop 文件夹里右键这个文件 →“作为程序运行”，或者在终端里 bash 它。
# 2026-11-04 用户：桌面不要快捷方式，启动入口放在项目文件夹里。
# 不在终端里运行时（文件管理器里点的）会自己开一个终端窗口跑服务：看日志、Ctrl+C 关服务；网页用 Firefox 打开并自动全屏。
# 加 --offline 是离线模拟（不连机械臂），见旁边的“启动三维示教_离线模拟.sh”。
DIR=$(cd "$(dirname "$0")" && pwd)
export TELEOP_PY=${TELEOP_PY:-$HOME/miniforge3/envs/teleop/bin/python}
if [ -t 1 ]; then
  exec "$DIR/start_twin.sh" "$@"
fi
TITLE="ECO65 三维示教"; [ "$1" = "--offline" ] && TITLE="ECO65 三维示教（离线模拟）"
exec gnome-terminal --title="$TITLE" -- bash -c "'$DIR/start_twin.sh' $*; echo; read -p '服务已经停了，按回车关闭这个窗口'"
