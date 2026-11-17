#!/bin/bash
# 启动三维示教：后台服务 + Firefox 打开网页，打开后自动全屏（像一个独立软件，在 Orin 自己的屏幕上直接用）。
# 网页右上角有“窗口 / 全屏”和“最小化”按钮（winctl.py 用系统的 X11 库模拟 F11、最小化窗口）；F11 也能切。
# 用法：
#   ~/teleop/start_twin.sh             连控制器，控制器仿真模式（真机不动；网页左上角能切到真实模式）
#   ~/teleop/start_twin.sh --offline   不连控制器，离线模拟（没开机械臂时调网页用）
# 连不上机械臂（192.168.1.18）时自动改成离线模拟。
# 关网页：关掉 Firefox 窗口（Alt+F4）；关服务：在这个终端窗口按 Ctrl+C（或者关掉终端窗口）。
# 2026-11-03 原来用 --kiosk：全屏退不出来、也不能最小化（用户反馈），改成普通窗口 + F11 全屏 + 网页上的按钮。
cd "$(dirname "$0")"
PY=${TELEOP_PY:-$HOME/miniforge3/envs/teleop/bin/python}
URL=http://localhost:8000
# 优先用 Chrome（2026-11-11 用户：换 Chrome），单独的配置目录，不影响平时用的浏览器；没有 Chrome 才用 Firefox
if command -v google-chrome >/dev/null 2>&1; then
  BROWSER=chrome; PROFILE=$HOME/.eco65_twin_chrome
else
  BROWSER=firefox; PROFILE=$HOME/.eco65_twin_firefox
fi
ARGS="$*"

up() { "$PY" -c "import urllib.request,sys; urllib.request.urlopen('$URL', timeout=1)" >/dev/null 2>&1; }

open_page() {
  # 网页已经开着（比如只是重启了服务）：不再开第二个（同一个配置开两次 Firefox 会报“已经在运行”），网页会自己重连
  if pgrep -f -- "$PROFILE" >/dev/null; then
    echo "网页已经开着，会自己重新连上服务"
    return
  fi
  mkdir -p "$PROFILE"
  # 第一次用这个配置时别弹欢迎页、默认浏览器提示；崩溃后不恢复旧标签页
  cat > "$PROFILE/user.js" <<'EOF'
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("webgl.force-enabled", true);
EOF
  # 第一次打开时窗口最大化（之后 Firefox 记住你自己调的大小）
  [ -f "$PROFILE/xulstore.json" ] || echo '{"chrome://browser/content/browser.xhtml":{"main-window":{"sizemode":"maximized"}}}' \
    > "$PROFILE/xulstore.json"
  if [ "$BROWSER" = chrome ]; then
    google-chrome --user-data-dir="$PROFILE" --new-window --start-maximized --no-first-run --no-default-browser-check "$URL" >/dev/null 2>&1 &
  else
    MOZ_NO_REMOTE=1 firefox --no-remote --profile "$PROFILE" --new-window "$URL" >/dev/null 2>&1 &
  fi
  local ff=$!
  "$PY" winctl.py start >/dev/null 2>&1      # 窗口出来、拿到焦点后切到全屏
  wait $ff
}

if up; then
  echo "服务已经在跑了，只打开网页"
  open_page
  exit 0
fi
if [[ " $ARGS " != *" --offline "* ]] && ! ping -c1 -W1 192.168.1.18 >/dev/null 2>&1; then
  echo "连不上机械臂（192.168.1.18）：检查机械臂电源和网线。这次先用离线模拟启动"
  ARGS="$ARGS --offline"
fi
# 服务起来了再开网页（最多等 30 秒）
( for _ in $(seq 60); do up && break; sleep 0.5; done; open_page ) &
exec "$PY" -u twin_server.py $ARGS
