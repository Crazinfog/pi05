#!/bin/bash
# 从备份恢复三维示教程序和 Python 环境（普通用户运行，不需要 sudo，不需要联网）：
#   bash restore.sh
# 环境解压到 ~/miniforge3/envs/teleop，程序解压到 ~/teleop，再在应用菜单放启动图标（启动入口是 ~/teleop/启动三维示教.sh）。
# 只能恢复到同类机器上：ARM（aarch64）的 Orin、Ubuntu 20.04（JetPack 5）。已经有同名目录时不覆盖，先改名再恢复。
# 测试用的变量：ENV_DIR（环境放哪）、CODE_PARENT（程序放在哪个目录下面）、SKIP_LAUNCHER=1（不放应用菜单图标）
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ENV_DIR=${ENV_DIR:-$HOME/miniforge3/envs/teleop}
CODE_PARENT=${CODE_PARENT:-$HOME}

echo "检查备份文件是不是完整……"
(cd "$HERE" && sha256sum -c SHA256SUMS)
for d in "$ENV_DIR" "$CODE_PARENT/teleop"; do
  if [ -e "$d" ]; then
    echo "$d 已经存在。为了不覆盖，先改个名再恢复，比如：mv $d ${d}_old"
    exit 1
  fi
done

echo "解压 Python 环境到 $ENV_DIR ……"
mkdir -p "$ENV_DIR"
tar -xzf "$HERE/teleop_env.tar.gz" -C "$ENV_DIR"
"$ENV_DIR/bin/conda-unpack"                  # 把环境里写死的路径改成现在的位置

echo "解压程序到 $CODE_PARENT/teleop ……"
mkdir -p "$CODE_PARENT"
tar -xzf "$HERE/teleop_code.tar.gz" -C "$CODE_PARENT"

echo "检查环境能不能用……"
"$ENV_DIR/bin/python" - <<'EOF'
import numpy, scipy, cv2, h5py, aiohttp, pyorbbecsdk, Robotic_Arm
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
print("环境能用：numpy", numpy.__version__, "· opencv", cv2.__version__, "· scipy", scipy.__version__)
EOF

if [ -z "$SKIP_LAUNCHER" ]; then
  TELEOP_PY="$ENV_DIR/bin/python" bash "$CODE_PARENT/teleop/install_launcher.sh"
fi
echo
echo "恢复好了。还差两项要 root（相机权限、机械臂网口固定 IP）："
echo "  sudo bash $HERE/restore_root.sh"
