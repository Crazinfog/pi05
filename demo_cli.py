"""命令行演示：在 Orin 的终端里输入一条指令，机械臂就用模型（π0.5）去做，不用开网页。

    ./demo red                    把红方块放到盘子上（put the red cube on the plate）
    ./demo 黄                     颜色也可以写中文：红 黄 橙 绿 紫 蓝
    ./demo "put the green cube on the plate"     其他英文指令原样发给模型
    ./demo home                   回初始位姿
    ./demo open / close           张开 / 合上夹爪
    ./demo stop                   让正在做的停下（在另一个终端里用）
    ./demo status                 看服务和机械臂现在的状态
    ./demo quit                   关掉后台服务（下次用会自动再开）

做任务时：先张开夹爪、回初始位姿，再让模型控制；放下方块后自动结束、回到初始位姿，可以摆下一次。
跑的过程中按 Ctrl+C 就停下。后台服务（三维示教服务，不开网页）没开着会自动在后台打开，日志在 logs/。
服务刚开时是控制器的仿真模式，第一次做任务会问一句要不要切到真实模式（回车确认），加 -y 不问。
2026-11-10 用户：演示时只想在命令行里输入指令机械臂就动，软件界面先不展示。
"""
import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8000
WS_URL = f"http://127.0.0.1:{PORT}/ws"
ARM_IP = "192.168.1.18"
PID_FILE = os.path.join(HERE, "logs", "demo_server.pid")
DEFAULT_ADDR = "172.16.230.65:8100"
COLORS = {"red": "red", "红": "red", "红色": "red", "yellow": "yellow", "黄": "yellow", "黄色": "yellow",
          "orange": "orange", "橙": "orange", "橙色": "orange", "green": "green", "绿": "green", "绿色": "green",
          "purple": "purple", "紫": "purple", "紫色": "purple", "blue": "blue", "蓝": "blue", "蓝色": "blue"}
COMMANDS = ("home", "open", "close", "stop", "status", "quit")


def say(text):
    print(text, flush=True)


# ---------------- 后台服务 ----------------
def port_open():
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=0.5).close()
        return True
    except OSError:
        return False


def arm_reachable():
    return subprocess.run(["ping", "-c1", "-W1", ARM_IP], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def start_server(offline):
    """后台开三维示教服务（不开网页）；等端口起来。"""
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log_path = os.path.join(HERE, "logs", time.strftime("twin_server_cli_%Y%m%d_%H%M%S.log"))
    args = [sys.executable, "-u", os.path.join(HERE, "twin_server.py")] + (["--offline"] if offline else [])
    with open(log_path, "w") as log:
        p = subprocess.Popen(args, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    with open(PID_FILE, "w") as f:
        f.write(str(p.pid))
    say(f"后台服务没开，正在打开（{'离线模拟，没连机械臂' if offline else '连机械臂'}；日志 {os.path.relpath(log_path, HERE)}）……")
    t0 = time.time()
    while time.time() - t0 < 90:
        if port_open():
            say(f"服务开好了（{time.time() - t0:.0f} 秒）")
            return True
        if p.poll() is not None:
            break
        time.sleep(0.5)
    with open(log_path, errors="replace") as f:
        tail = f.read()[-1500:]
    say("服务没开起来，日志最后几行：\n" + tail)
    return False


def stop_server():
    pid = None
    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
        except ValueError:
            pid = None
    if pid is None:
        say("没有由 ./demo 打开的后台服务（用启动脚本开的服务请在它自己的终端里按 Ctrl+C 关）")
        return
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        say("后台服务已经不在了")
        os.remove(PID_FILE)
        return
    for _ in range(30):
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        os.kill(pid, 9)          # 关的时候偶尔卡住（相机线程），强制结束
    os.remove(PID_FILE)
    say("后台服务关掉了")


# ---------------- 和服务说话 ----------------
class Link:
    """连服务的 websocket：config / state 随时更新，其余消息排进队列。"""

    def __init__(self):
        self.cfg, self.state, self.queue = None, None, asyncio.Queue()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(WS_URL, heartbeat=10, max_msg_size=0)
        self.reader = asyncio.create_task(self._read())
        for _ in range(50):              # 连上后服务先发 config
            if self.cfg is not None:
                break
            await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, *exc):
        self.reader.cancel()
        await self.ws.close()
        await self.session.close()

    async def _read(self):
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue                 # 点云之类的二进制消息不管
            d = json.loads(msg.data)
            if d.get("type") == "config":
                self.cfg = d
            elif d.get("type") == "state":
                self.state = d
            else:
                await self.queue.put(d)

    async def send(self, **m):
        await self.ws.send_str(json.dumps(m))

    async def wait(self, pred, timeout):
        """等一条满足 pred 的消息；超时返回 None。"""
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                return None
            try:
                d = await asyncio.wait_for(self.queue.get(), left)
            except asyncio.TimeoutError:
                return None
            if pred(d):
                return d

    def busy(self):
        st = self.state or {}
        return bool(st.get("executing")) or "回初始位姿中" in str(st.get("status", ""))


async def go_home(link):
    if link.busy():
        say("机械臂正在动，等它停了再回初始位姿")
        return False
    await link.send(type="home")
    first = await link.wait(lambda d: d.get("type") == "msg", 5)
    if first and "正在回初始位姿" not in first.get("text", ""):
        say(first.get("text", ""))
        return False
    say("回初始位姿……")
    done = await link.wait(lambda d: d.get("type") == "msg" and "回初始位姿结束" in d.get("text", ""), 120)
    if done is None:
        say("回初始位姿超时（2 分钟没结束）")
        return False
    return True


async def ensure_real(link, assume_yes):
    cfg = link.cfg or {}
    if cfg.get("offline"):
        say("注意：现在是离线模拟（没连机械臂），机械臂不会动")
        return True
    if cfg.get("real"):
        return True
    if not assume_yes:
        try:
            input("现在是控制器的仿真模式（真机不动）。切到真实模式？机械臂会真的动，手放在急停旁边。[回车确认 / Ctrl+C 取消] ")
        except (EOFError, KeyboardInterrupt):
            say("\n没切，退出")
            return False
    await link.send(type="set_real", real=True)
    r = await link.wait(lambda d: d.get("type") == "msg", 10)
    say(r.get("text", "") if r else "切真实模式没有回音")
    for _ in range(20):
        if (link.cfg or {}).get("real"):
            return True
        await asyncio.sleep(0.1)
    return False


async def run_task(link, task, args):
    if not await ensure_real(link, args.yes):
        return 1
    if link.busy():
        say("机械臂正在动（别的地方在执行），等它停了再来")
        return 1
    await link.send(type="gripper", closed=False)
    if not args.no_home and not await go_home(link):
        return 1
    say(f"指令：{task}")
    await link.send(type="policy_start", addr=args.addr, task=task, replan=args.replan, max_s=args.max,
                    raw=False, auto_end=True)
    r = await link.wait(lambda d: d.get("type") == "policy", 10)
    if r is None or not r.get("ok"):
        say(r.get("text", "模型控制没有回音") if r else "模型控制没有回音（服务卡住了？）")
        return 1
    say("开始（Ctrl+C 停下）")
    # Ctrl+C 只置个标记，由下面的循环发“停止”并等它停稳（不靠 asyncio 收尾时的取消，保证停止指令一定发出去）
    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop_evt.set)
    result, last = None, ""
    try:
        while result is None:
            if stop_evt.is_set():
                print()
                say("停下……")
                await link.send(type="stop")
                result = await link.wait(lambda d: d.get("type") == "policy" and d.get("done"), 5)
                say(result.get("text", "") if result else "已发停止")
                return 130
            d = await link.wait(lambda d: d.get("type") == "policy" and d.get("done"), 0.3)
            if d is not None:
                result = d
                break
            line = str((link.state or {}).get("policy") or (link.state or {}).get("status") or "")
            if line and line != last:
                print("\r\033[K" + line[:150], end="", flush=True)
                last = line
    finally:
        loop.remove_signal_handler(signal.SIGINT)
    print()
    say(result.get("text", ""))
    ok = bool(result.get("ok"))
    if not args.stay:
        await go_home(link)
        say("回到初始位姿了，可以摆下一次" if ok else "已回初始位姿")
    return 0 if ok else 1


async def run_command(link, cmd, args):
    if cmd == "home":
        if not await ensure_real(link, args.yes):
            return 1
        return 0 if await go_home(link) else 1
    if cmd in ("open", "close"):
        await link.send(type="gripper", closed=(cmd == "close"))
        say("夹爪" + ("合上" if cmd == "close" else "张开"))
        await asyncio.sleep(1.0)
        return 0
    if cmd == "stop":
        await link.send(type="stop")
        say("已发停止")
        return 0
    if cmd == "status":
        await asyncio.sleep(0.5)
        cfg, st = link.cfg or {}, link.state or {}
        mode = "离线模拟" if cfg.get("offline") else ("真实模式" if cfg.get("real") else "仿真模式（真机不动）")
        say(f"服务：开着 · {mode}")
        say(f"状态：{st.get('status', '?')}" + (f" · {st['policy']}" if st.get("policy") else ""))
        if st.get("q"):
            say("关节角：" + "  ".join(f"J{i + 1} {v:.1f}°" for i, v in enumerate(st["q"])))
        say("夹爪：" + ("合着" if st.get("gripper_closed") else "张着") + (f" · {st['gripper_msg']}" if st.get("gripper_msg") else ""))
        if st.get("cam_problem"):
            say("相机：" + st["cam_problem"])
        return 0
    return 1


def task_from(words):
    if len(words) == 1 and words[0].lower() in COLORS:
        return f"put the {COLORS[words[0].lower()]} cube on the plate"
    return " ".join(words)


async def amain(args):
    words = args.words
    cmd = words[0].lower() if len(words) == 1 and words[0].lower() in COMMANDS else None
    if cmd == "quit":
        stop_server()
        return 0
    if not port_open():
        if cmd in ("stop", "status"):
            say("后台服务没开")
            return 0 if cmd == "status" else 1
        offline = args.offline or not arm_reachable()
        if offline and not args.offline:
            say(f"连不上机械臂（{ARM_IP}）：开机了吗？网线插在 Orin 上了吗？先用离线模拟打开")
        if not start_server(offline):
            return 1
    async with Link() as link:
        if cmd:
            return await run_command(link, cmd, args)
        return await run_task(link, task_from(words), args)


def main():
    ap = argparse.ArgumentParser(prog="./demo", description="命令行演示：输入指令，机械臂用 π0.5 模型去做",
                                 epilog="例：./demo red    ./demo 绿    ./demo \"put the yellow cube on the plate\"    ./demo home",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="+", help="颜色（red/红…）、英文指令，或 home / open / close / stop / status / quit")
    ap.add_argument("-y", "--yes", action="store_true", help="切真实模式不再问")
    ap.add_argument("--addr", default=DEFAULT_ADDR, help=f"模型服务地址（默认 {DEFAULT_ADDR}）")
    ap.add_argument("--max", type=float, default=40, help="最长跑多少秒（默认 40）")
    ap.add_argument("--replan", type=int, default=8, help="执行几步再问一次模型（默认 8）")
    ap.add_argument("--stay", action="store_true", help="做完不回初始位姿")
    ap.add_argument("--no-home", action="store_true", help="开始前不回初始位姿")
    ap.add_argument("--offline", action="store_true", help="后台服务用离线模拟打开（不连机械臂，只测流程）")
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        sys.exit(130)
    except aiohttp.ClientError as e:
        say(f"连不上后台服务：{e}")
        sys.exit(1)
