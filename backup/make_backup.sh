#!/bin/bash
# 把三维示教程序连同 Python 环境打包成一份备份（在 Orin 上运行，不需要 sudo）：
#   bash ~/teleop/backup/make_backup.sh [输出目录，默认 ~/teleop_backup]
# 生成 eco65_teleop_日期时间/：
#   teleop_env.tar.gz   conda 环境 teleop（conda-pack 打的，解压后不联网就能用）
#   teleop_code.tar.gz  ~/teleop 整个程序，含相机标定、初始位姿、轨迹模板、录制设置、日志；不含录的示范数据 data/
#   system/             相机 udev 规则、机械臂网口设置、版本清单、包清单（给 restore_root.sh 和人看）
#   restore.sh / restore_root.sh / 恢复说明.txt / SHA256SUMS
set -e
OUT_ROOT=${1:-$HOME/teleop_backup}
CONDA=${CONDA:-$HOME/miniforge3}
ENV=$CONDA/envs/teleop
OUT=$OUT_ROOT/eco65_teleop_$(date +%Y%m%d_%H%M)
HERE=$(cd "$(dirname "$0")" && pwd)
[ -x "$CONDA/bin/conda-pack" ] || "$CONDA/bin/python" -m pip install conda-pack -i https://pypi.tuna.tsinghua.edu.cn/simple -q
mkdir -p "$OUT/system"

echo "1/3 打包 Python 环境（约 600 MB，压缩要一两分钟）……"
"$CONDA/bin/conda-pack" -p "$ENV" -o "$OUT/teleop_env.tar.gz" -j -1 -q

echo "2/3 打包程序 ~/teleop（不含 data/ 里录的示范）……"
tar -czf "$OUT/teleop_code.tar.gz" -C "$HOME" --exclude='teleop/data' --exclude='__pycache__' --exclude='*.pyc' teleop

echo "3/3 记下系统里的设置和版本……"
cp /etc/udev/rules.d/99-obsensor-libusb.rules "$OUT/system/" 2>/dev/null || echo "（没找到相机 udev 规则）"
nmcli -f connection.id,connection.interface-name,ipv4.method,ipv4.addresses,ipv4.never-default con show "Wired connection 1" \
  > "$OUT/system/机械臂网口.txt" 2>/dev/null || true
{
  echo "备份时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "机器：$(hostname)  $(uname -m)"
  head -1 /etc/nv_tegra_release 2>/dev/null
  lsb_release -ds 2>/dev/null
  ldd --version | head -1
  firefox --version 2>/dev/null
  "$ENV/bin/python" --version
} > "$OUT/system/版本.txt" 2>&1
"$ENV/bin/pip" freeze > "$OUT/system/pip_freeze.txt" 2>/dev/null
"$CONDA/bin/conda" list -p "$ENV" --explicit > "$OUT/system/conda_explicit.txt" 2>/dev/null || true

cp "$HERE/restore.sh" "$HERE/restore_root.sh" "$OUT/"
# 2026-11-09 起 Orin 上不放说明文档：恢复说明.txt 只在 Mac 的 teleop/backup/ 里，拷备份到 Mac 后放进去
[ -f "$HERE/恢复说明.txt" ] && cp "$HERE/恢复说明.txt" "$OUT/" || echo "（恢复说明.txt 在 Mac 的 ~/pi0.5_project/teleop/backup/ 里，拷到 Mac 后放进这个备份）"
(cd "$OUT" && sha256sum teleop_env.tar.gz teleop_code.tar.gz > SHA256SUMS)
echo
du -sh "$OUT"/* | sed 's|'"$OUT"'/||'
echo "备份好了：$OUT"
echo "记得拷一份到别的电脑或移动硬盘上（只放在这台 Orin 上不算备份）"
