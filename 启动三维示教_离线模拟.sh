#!/bin/bash
# 启动 ECO65 三维示教的离线模拟（不连机械臂，用 SDK 算法库模拟一台）。用法和“启动三维示教.sh”一样。
exec "$(cd "$(dirname "$0")" && pwd)/启动三维示教.sh" --offline
