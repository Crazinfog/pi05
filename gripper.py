"""知行 CTAG2F90-D 两指夹爪：接在睿尔曼机械臂末端接口上（24V 供电 + RS485），通过机械臂控制器转发 Modbus RTU。

    python gripper.py open                # 张开
    python gripper.py close               # 闭合
    python gripper.py width 0.5           # 开到一半（0 闭合 ~ 1 张开）
    python gripper.py probe               # 只读扫一遍寄存器（不会让夹爪动）

寄存器来自厂家示例 my-robot/Grapper/scripts/GUI2Robot_90.py（从站 1、115200）：使能 → 设速度/力/加减速 → 写目标位置 → 触发运动。
厂家示例里没有读状态的寄存器，2026-10-27 开/合/半开对比实测：0x0419 是当前开口（和目标同一刻度，空夹闭合停在约 973），
0x0403 全闭时为 4096（疑似堵转/夹到标志，未确认）。
三代控制器用 rm_set_modbus_mode / rm_write_registers，四代用 rm_set_tool_rs485_mode / rm_write_modbus_rtu_registers，
自动判断（三代接口返回 -4 就换四代）。
注意：Modbus 是真实 IO，示教器仿真模式下夹爪也会真的动。
"""
import argparse
import json
import os
import threading
import time

SLAVE, BAUD = 1, 115200
REG_ENABLE = 0x0100
REG_POS = 0x0102          # 目标位置，32 位，占 0x0102（高 16 位）和 0x0103（低 16 位）
REG_SPEED = 0x0104        # 0x0104 速度、0x0105 力、0x0106 加速、0x0107 减速，连续四个
REG_TRIGGER = 0x0108      # 写 1 才开始运动
REG_HOME = 0x0402         # 写 1 回零
REG_ACTUAL = 0x0419       # 当前开口（实测，0 全开 ~ 1000 全闭，到位约 0.25 秒）
POS_OPEN, POS_CLOSED = 0, 1000     # 2026-10-27 实测：-D 型号目标位置被夹爪限在 0~1000（写 9000 读回 1000），0 全开、1000 全闭（和厂家示例 0~9000 不同）
# 速度、力、加速、减速：厂家示例都是 100（单位手册里没写，应该是百分比）。2026-11-06 用户：每次夹和松都太猛 → 默认改温和；
# 想再调就在旁边建 gripper_config.json，比如 {"speed": 50, "force": 40, "acc": 50, "dec": 50}，重启服务（或网页“重启夹爪”）生效。
# 注意：调慢以后合上要更久——自动生成里“合夹爪后停”要不短于合到位的时间（原来 100 时约 0.25 秒），不然没夹紧就开始抬
SPEED, FORCE, ACC, DEC = 50, 40, 50, 50
GRIP_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_config.json")
try:
    with open(GRIP_CFG, encoding="utf-8") as _f:
        _c = json.load(_f)
    SPEED, FORCE, ACC, DEC = (int(_c.get(k, v)) for k, v in zip(("speed", "force", "acc", "dec"), (SPEED, FORCE, ACC, DEC)))
except (OSError, ValueError):
    pass
PROBE_RANGES = [(0x0100, 12), (0x0200, 12), (0x0300, 12), (0x0400, 12), (0x0500, 12), (0x0600, 12)]
TOOL_24V = 3              # rm_set_tool_voltage：0 关、1 5V、2 12V、3 24V
# 2026-10-28 两次遇到夹爪“死掉”：读写全部超时（控制器返回 false），重设末端 Modbus 口也没用，末端 24V 断电重启才恢复。
# 触发原因还没查到（单独跑、加快速读状态、强杀进程、切仿真/真实都复现不了），所以后台线程自己盯着：
SILENT_S = 1.5            # 这么久没有一次读写成功，就当夹爪没响应
RECOVER_GAP_S = 5.0       # 两次自动恢复之间至少隔这么久
STUCK_S = 1.5             # 开合指令发出去这么久开口一点没变（离目标还远），就当夹爪掉了使能，重新使能再发一次


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [夹爪] {msg}", flush=True)


class ZhixingGripper:
    """单独开一个到控制器的连接（不和运动控制、读状态抢同一个连接），后台线程执行最新的开合指令，
    set_closed / set_width 立刻返回，不拖慢 50 Hz 的控制循环。"""

    def __init__(self, ip, port=8080, init_sdk=False):
        """init_sdk：单独运行时要先初始化 SDK 的线程模式；在 teleop 里 RealmanRobot 已经初始化过，不要重复。"""
        from Robotic_Arm import rm_robot_interface as rm
        self.rm = rm
        self.arm = rm.RoboticArm(rm.rm_thread_mode_e.RM_TRIPLE_MODE_E) if init_sdk else rm.RoboticArm()
        if self.arm.rm_create_robot_arm(ip, port).id == -1:
            raise RuntimeError(f"夹爪：连不上机械臂控制器 {ip}:{port}")
        self.io = threading.RLock()   # 同一个连接：后台线程的读写和网页点的“重启夹爪”不能同时用
        self.v4 = False
        ret = self._setup()
        if ret != 0:                  # 夹爪可能是“死掉”的状态：断电重启一次再试（启动时本来就会张开夹爪）
            log(f"使能失败（错误码 {ret}），末端断电重启夹爪再试一次")
            ret = self._power_cycle()
        self._check(ret, "使能并设置速度/力/加减速")
        print(f"[夹爪] 已就绪（{'四' if self.v4 else '三'}代控制器接口）")
        self.target, self.running = None, True
        self.last_target = None   # 最近一次开合指令（重启后要重发）
        self.actual = None        # (读到的时刻, 当前开口原始值)，后台约 10 Hz 更新
        self.last_ok = time.time()                 # 最近一次读写成功的时刻
        self.status, self.msg = "ok", ""           # ok / silent（没响应）/ restarting（正在断电重启）
        self.last_recover, self.sent = 0.0, None   # sent：(发出时刻, 目标, 当时开口, 已经重新使能过没有)
        self.wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _setup(self):
        """确认末端 24V、把末端 RS485 设成 Modbus 主站、使能、设速度/力/加减速。返回错误码。"""
        with self.io:
            ret, volt = self.arm.rm_get_tool_voltage()
            if ret == 0 and volt != TOOL_24V:
                log(f"末端电源现在是 {['0V', '5V', '12V', '24V'][volt]}，改成 24V 并等夹爪上电")
                self._check(self.arm.rm_set_tool_voltage(TOOL_24V), "打开末端 24V 电源")
                time.sleep(2.0)
            ret = self.arm.rm_set_modbus_mode(1, BAUD, 2)        # 1 = 末端接口板 RS485 做主站，超时 0.2 秒
            self.v4 = ret == -4
            if self.v4:
                ret = self.arm.rm_set_tool_rs485_mode(0, BAUD)   # 0 = 末端 RS485 做 Modbus RTU 主站
            self._check(ret, "把末端 RS485 设成 Modbus 主站")
            time.sleep(0.2)
            for attempt in range(3):          # 刚切成 Modbus 模式时偶尔第一次写不进去（实测报错码 1），重试几次
                ret = self.write(REG_ENABLE, [1]) or self.write(REG_SPEED, [SPEED, FORCE, ACC, DEC])
                if ret == 0:
                    break
                time.sleep(0.3)
            return ret

    def _power_cycle(self):
        """末端 24V 断电 2 秒再上电（夹爪会松开），等它启动好再重新设置。返回错误码。"""
        with self.io:
            self.arm.rm_set_tool_voltage(0)
            time.sleep(2.0)
            self.arm.rm_set_tool_voltage(TOOL_24V)
            time.sleep(3.0)
            return self._setup()

    def restart(self):
        """断电重启夹爪（网页“重启夹爪”按钮、或自动恢复时用），完了重发最近一次开合指令。返回是否成功。"""
        self.status, self.msg = "restarting", "夹爪没响应，正在断电重启（约 5 秒，夹爪会松开）"
        log("断电重启夹爪")
        self.last_recover = time.time()
        ret = self._power_cycle()
        if ret != 0:
            self.status, self.msg = "silent", f"断电重启后夹爪还是没响应（错误码 {ret}）：检查末端那根线插紧没有、夹爪的灯亮不亮"
            log(self.msg)
            return False
        self.last_ok, self.status, self.msg = time.time(), "ok", ""
        log("重启好了，重发最近一次开合指令")
        if self.last_target is not None:
            self.target = self.last_target
            self.wake.set()
        return True

    def _recover(self):
        """夹爪没响应时：先重设末端 Modbus 口；还不行，夹爪应该是张开的就自动断电重启，
        合着的（可能夹着东西，断电会松开）就不自动断电，等人点“重启夹爪”。"""
        self.last_recover = time.time()
        with self.io:
            if not self.v4:
                self.arm.rm_close_modbus_mode(1)
                time.sleep(0.2)
                self.arm.rm_set_modbus_mode(1, BAUD, 2)
                time.sleep(0.2)
            ret, _ = self.read(REG_ACTUAL, 1)
        if ret == 0:
            log("重设末端 Modbus 口后恢复了")
            self.last_ok = time.time()
            return
        closed = self.last_target is not None and self.last_target > (POS_OPEN + POS_CLOSED) / 2
        if closed:
            self.msg = "夹爪没响应。现在是合着的（可能夹着东西），不自动断电重启：放稳东西后点“重启夹爪”"
            log(self.msg)
            return
        self.restart()

    def health(self):
        """(是否正常, 说明)：网页上显示用。"""
        return self.status == "ok", self.msg

    @staticmethod
    def _check(ret, what):
        if ret != 0:
            raise RuntimeError(f"夹爪：{what}失败，错误码 {ret}")

    # ---------- 寄存器读写（区分三代 / 四代控制器接口）----------
    def write(self, addr, values):
        rm = self.rm
        if self.v4:
            return self.arm.rm_write_modbus_rtu_registers(
                rm.rm_modbus_rtu_write_params_t(addr, SLAVE, 1, len(values), list(values)))
        if len(values) == 1:
            return self.arm.rm_write_single_register(rm.rm_peripheral_read_write_params_t(1, addr, SLAVE), values[0])
        data = [b for v in values for b in ((v >> 8) & 0xFF, v & 0xFF)]   # 三代接口按字节传，高字节在前
        return self.arm.rm_write_registers(rm.rm_peripheral_read_write_params_t(1, addr, SLAVE, len(values)), data)

    def read(self, addr, n):
        """读 n 个保持寄存器，返回 (错误码, 16 位无符号整数列表)。"""
        rm = self.rm
        if self.v4:
            ret, vals = self.arm.rm_read_modbus_rtu_holding_registers(rm.rm_modbus_rtu_read_params_t(addr, SLAVE, 1, n))
            return ret, [v & 0xFFFF for v in vals]
        if 2 < n < 13:      # 三代的多寄存器读按字节返回
            ret, data = self.arm.rm_read_multiple_holding_registers(rm.rm_peripheral_read_write_params_t(1, addr, SLAVE, n))
            return ret, [((data[2 * i] & 0xFF) << 8) | (data[2 * i + 1] & 0xFF) for i in range(n)]
        vals = []
        for i in range(n):
            ret, v = self.arm.rm_read_holding_registers(rm.rm_peripheral_read_write_params_t(1, addr + i, SLAVE))
            if ret != 0:
                return ret, vals
            vals.append(v & 0xFFFF)
        return 0, vals

    # ---------- 动作 ----------
    def move_to(self, pos):
        """同步执行：写目标位置再触发，返回错误码。"""
        pos = int(max(min(pos, max(POS_OPEN, POS_CLOSED)), min(POS_OPEN, POS_CLOSED)))
        ret = self.write(REG_POS, [(pos >> 16) & 0xFFFF, pos & 0xFFFF])
        return ret or self.write(REG_TRIGGER, [1])

    def set_width(self, frac):
        """异步：0 闭合 ~ 1 张开。"""
        self.target = POS_CLOSED + (POS_OPEN - POS_CLOSED) * min(max(frac, 0.0), 1.0)
        self.wake.set()

    def set_closed(self, closed):
        self.set_width(0.0 if closed else 1.0)

    def width(self):
        """当前开口比例：0 闭合 ~ 1 张开；还没读到时是 None。"""
        if self.actual is None:
            return None
        return (self.actual[1] - POS_CLOSED) / (POS_OPEN - POS_CLOSED)

    def _loop(self):
        while self.running:
            self.wake.wait(0.1 if self.status == "ok" else 0.5)   # 没响应时少发点，别把总线占满
            self.wake.clear()
            if self.status == "restarting":
                continue
            target, self.target = self.target, None
            with self.io:
                if target is not None:
                    self.last_target = target
                    ret = self.move_to(target)
                    if ret == 0:
                        self.last_ok = time.time()
                        self.sent = (time.time(), target, self.actual[1] if self.actual else None, False)
                    else:
                        log(f"开合指令失败（错误码 {ret}），稍后重发")
                        if self.target is None:     # 没有更新的指令就重发这一条，不让按钮“按了没反应”
                            self.target = target
                ret, vals = self.read(REG_ACTUAL, 1)    # 顺便读当前开口，录数据要用
            if ret == 0 and vals:
                self.actual = (time.time(), vals[0])
                self.last_ok = time.time()
            now = time.time()
            if now - self.last_ok > SILENT_S:
                if self.status == "ok":
                    self.status, self.msg = "silent", "夹爪没响应，正在自动恢复"
                    log(f"{now - self.last_ok:.1f} 秒没有一次读写成功，开始自动恢复")
                if now - self.last_recover > RECOVER_GAP_S:
                    self._recover()
            elif self.status != "ok":
                self.status, self.msg = "ok", ""
                log("恢复正常")
            # 指令发了、开口却一点没动（离目标还远）：多半是夹爪重新上过电、掉了使能。重新使能再发一次
            s = self.sent
            if s and not s[3] and self.actual and s[2] is not None and now - s[0] > STUCK_S:
                if abs(self.actual[1] - s[1]) > 50 and abs(self.actual[1] - s[2]) < 20:
                    log("开合指令发出去了但夹爪没动，重新使能再发一次")
                    with self.io:
                        self.write(REG_ENABLE, [1])
                    self.target = s[1]
                    self.sent = (s[0], s[1], s[2], True)
                else:
                    self.sent = None

    def probe(self):
        """只读：把几段寄存器读出来打印。动一下夹爪前后各跑一次，变了的就是位置/状态寄存器。"""
        for start, n in PROBE_RANGES:
            ret, vals = self.read(start, n)
            body = " ".join(f"{v:5d}" for v in vals) if ret == 0 else f"读失败（错误码 {ret}）"
            print(f"0x{start:04X}: {body}")

    def close(self):
        self.running = False
        self.wake.set()
        with self.io:
            if not self.v4:
                self.arm.rm_close_modbus_mode(1)     # 末端口恢复成机械臂默认模式
            self.arm.rm_delete_robot_arm()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="知行 CTAG2F90-D 夹爪测试")
    ap.add_argument("cmd", choices=["open", "close", "width", "home", "probe"])
    ap.add_argument("value", nargs="?", type=float, default=0.5, help="width 的开口比例，0 闭合 ~ 1 张开")
    ap.add_argument("--ip", default="192.168.1.18")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    g = ZhixingGripper(args.ip, args.port, init_sdk=True)
    try:
        if args.cmd == "probe":
            g.probe()
        elif args.cmd == "home":
            g._check(g.write(REG_HOME, [1]), "回零")
        else:
            frac = {"open": 1.0, "close": 0.0}.get(args.cmd, args.value)
            g._check(g.move_to(POS_CLOSED + (POS_OPEN - POS_CLOSED) * frac), "运动")
            time.sleep(1.5)
            g.probe()        # 顺便看看动完以后寄存器的样子
    finally:
        g.close()
