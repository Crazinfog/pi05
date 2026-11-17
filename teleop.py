"""手柄遥操作

手柄输入换算成机械臂末端的目标位姿，发给下面两种“机械臂”之一：
  python teleop.py                   假机械臂（MockRobot），只在终端里显示
  python teleop.py --robot realman   睿尔曼机械臂（RealmanRobot），默认切到仿真模式：
                                     真实机械臂不动，只有示教器里的三维模型动
  python teleop.py --robot realman --real   真实模式，机械臂会真的动（要二次确认）
  再加 --input marker：用标记手柄控制（先跑一次 marker_align.py 对齐方向），见下面“标记手柄”

按键：
  START       开始 / 暂停控制（开始前机械臂不会动）
  左摇杆      末端前后 / 左右移动
  右扳机/左扳机 末端上升 / 下降（按得越深越快）
  姿态全部按世界坐标转，人站在固定位置操作，同一个输入永远是同一个转动方向：
  右摇杆      往哪边推，夹爪尖就往哪边倒（绕世界水平轴转，最多倒到 MAX_TILT_DEG，LOCK_TILT=True 时不动）
  十字键左右  绕世界竖直轴转：从上往下看 左 逆时针、右 顺时针（夹爪放平时用来左右摆）
  LB/RB       夹爪绕自身的轴转，只动最后一个关节（调两根手指的方向）。夹爪朝下时从上往下看 LB 逆时针、RB 顺时针
  A           夹爪开 / 合
  B（按住）    慢速模式，精细对准时用
  X           急停（随时都能按）；解除：按住 X 再按 START。回初始位姿的过程中按不了，要用实体急停
  Y           回到初始位姿
  BACK        退出程序

标记手柄（--input marker）：一只手拿贴了 ArUco 标记的手柄，另一只手拿游戏手柄。
  按住 LB      夹爪跟着标记手柄动（绕抓取中心转）；松开 LB 夹爪停住
  两种对应方式（--marker-map）：
    absolute（默认）开始控制后第一次按 LB 时，把手柄当时的位置和姿态对应到夹爪当时的位置和姿态，之后一直按这个关系走：
                    松开 LB 时手挪开或转了，再按下夹爪会限速追到手对应的位姿。所以第一次按 LB 前先把手柄拿正
    relative        每次按下 LB 都重新对应：松开时手随便挪，再按下夹爪不动，可以分几次累积
  右手拿标记手柄，常用功能都放在左手够得着的地方：
  十字键上    重新对应：下次按 LB 时，手的当前位置对应夹爪的当前位置（START、Y、切换精细模式时也会自动重新对应）
  按住 LT      精细模式：手动 1 cm 夹爪只动 MARKER_SCALE_FINE cm（B 也行）
  十字键下    夹爪开 / 合（A 也行）
  按下左摇杆  急停（X 也行）；解除：按住左摇杆或 X，再按 START
  START、Y、BACK 不变；没按 LB 时左摇杆照样能平移；扳机升降、LB/RB 自转在这个模式下关掉
  没连游戏手柄时自动改用终端键盘（2026-11-06，只给标记手柄模式用；在运行程序的终端里按键，不用按回车确认）：
    回车/空格  开始 / 暂停跟手（开始后只要相机看得到标记夹爪就跟着走，相当于 LB 一直按着；暂停再开始＝重新对应）
    g 夹爪开 / 合    f 精细模式开 / 关    h 回初始位姿    x 急停，r 解除急停    q 退出（Ctrl+C 也行）

每次运行都会在 logs/ 下存一份 CSV 日志：每帧的手柄输入、发出的指令、读回的实际状态，用来排查手感问题。

安全保护（只对睿尔曼生效，参数见“安全保护”一节）：
  每发一个目标前先在本地算逆解，够不到、关节突然大转、快到关节限位、会自身相撞，或者按关节角算出
  手臂各段 / 夹爪离桌面太近、法兰离底座太近，就拦住这一步（终端打印 [安全]）。
  实际位置跟不上指令、读不到实际状态、实际手臂已经离桌面太近时自动暂停，按 START 从实际位置接着走。
"""
import argparse
import csv
import json
import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # 没有显示器（比如 SSH 登录）时也能运行

import numpy as np
import pygame
from pygame._sdl2 import controller
from scipy.spatial.transform import Rotation as R

# ======================= 参数（按手感调整） =======================
RATE_HZ = 50                 # 控制循环频率
DEADZONE = 0.08              # 摇杆死区：小于这个幅度当作没推
LIN_SPEED = 0.10             # 正常平移速度，米/秒
LIN_SPEED_SLOW = 0.03        # 慢速模式平移速度
ANG_SPEED = 0.6              # 正常转动速度，弧度/秒
ANG_SPEED_SLOW = 0.2         # 慢速模式转动速度
SMOOTH = 0.3                 # 速度平滑系数（0~1），越小越平滑、越“肉”
LOCK_TILT = False            # True：锁住倾斜，夹爪一直朝下（只能用十字键和 LB/RB 转）
MAX_TILT_DEG = 95            # 夹爪最多偏离竖直朝下多少度（90 是水平）
TOOL_LENGTH = 0.175          # 法兰盘到抓取中心（夹爪张开时两指中间）的距离（米），量得 17~18 cm（到指尖约 20 cm）。姿态绕这个点转；0 表示绕法兰盘转
HEADING_MODE = False         # True：“龙头模式”，前后左右跟着夹爪朝向走
# 干活区域朝哪边：从上往下看，相对底座 x 轴逆时针转多少度。初始位姿、手柄的工作范围框、网页默认视角和桌面都跟着转。
# 2026-11-02 用户：相机（不挪）拍得到的是原来初始位姿（朝 x 轴）左边那 180°，所以初始位姿转到它正中间，+90°
HOME_YAW_DEG = 90
_YAW = R.from_euler("z", HOME_YAW_DEG, degrees=True)
# 工作空间限位（米，朝 x 轴时的范围，按 HOME_YAW_DEG 转过去后取外包框），手柄遥操作用
_WS0 = np.array([[0.20, 0.60],    # x
                 [-0.30, 0.30],   # y
                 [0.02, 0.50]])   # z（抓取中心）。贴近桌面时主要靠桌面保护，这里只是兜底
_ws_corners = _YAW.apply(np.array(np.meshgrid(*_WS0)).reshape(3, -1).T)
WORKSPACE = np.stack([_ws_corners.min(axis=0), _ws_corners.max(axis=0)], axis=1).round(6)
# 初始位姿：位置（米）+ 姿态（xyz 欧拉角，弧度）。朝 x 轴时是法兰在 (0.35, 0, 0.30)、夹爪竖直朝下，再绕竖直轴转 HOME_YAW_DEG
HOME_POS = _YAW.apply([0.35, 0.0, 0.30])
HOME_EULER = (_YAW * R.from_euler("xyz", [np.pi, 0.0, 0.0])).as_euler("xyz")
HOME_Q = [9.7 + HOME_YAW_DEG, 18.6, 78.5, -7.0, -90.0, -80.3]   # 初始位姿对应的关节角（度；朝 x 轴时 J1 是真机读的 9.7°，
                                                               # 整条手臂绕底座转只改 J1）。回初始位姿直接走关节角，
                                                               # 不让控制器从当前姿态现解逆解：机械臂伸直（奇异）时那样会失败
# 网页“把现在的姿态设为初始位姿”存的初始位姿（关节角 + 法兰位姿），有就优先用（2026-11-02 加：初始位姿往左转以后，
# 远一点的地方只有“反手”摆法够得着，初始位姿是“正面”的话每条轨迹都要绕底座转 180° 换摆法，让用户自己摆一个顺手的存下来）
HOME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "home.json")
if os.path.exists(HOME_FILE):
    try:
        with open(HOME_FILE) as _f:
            _h = json.load(_f)
        HOME_Q, HOME_POS, HOME_EULER = [float(v) for v in _h["q"]], np.array(_h["flange_pos"]), np.array(_h["flange_euler"])
    except Exception as _e:
        print(f"[初始位姿] {HOME_FILE} 读不了（{_e}），用默认的")
HOME_SPEED = 20              # 回初始位姿的速度百分比（1~100），先慢一点
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
# ---- 安全保护（只对睿尔曼生效，假机械臂不检查桌面） ----
TABLE_Z = 0.0                # 桌面高度（米，基座坐标系）：桌面和底座安装面一样高
ARM_CLEARANCE = 0.08         # 手臂各段（沿关节轴线取点）离桌面至少多高：连杆半径约 5 cm，再留 3 cm
GRIPPER_CLEARANCE = 0.03     # 夹爪身体（法兰到指尖的轴线）离桌面至少多高：夹爪放平时侧面比轴线低半个夹爪宽
TIP_CLEARANCE = 0.01         # 指尖那一点离桌面至少多高：夹爪竖直朝下时最低的就是它
# 法兰盘到指尖的距离（米）。知行 CTAG2F90 是连杆式平行夹爪：合上时手指沿圆弧往里收的同时往前伸，
# 指尖最多比张开时（量得 20 cm）再伸出约 2.95 cm（2026-10-27 用厂家 URDF+STL 算的；同天真机空夹合上时指尖碰了桌面）。
# 2026-10-28 起离桌面检查按夹爪“那时候”的开口算指尖伸出多少（之前一律按全合，夹着东西时手指停在物体两侧、
# 伸不了那么多，抓取点明明没问题也报太低）：拖目标、执行时按真实夹爪读回的开口；规划时合夹爪那一帧和夹着东西走的那几段
# 按关键帧里填的物体宽度；没填宽度的合夹爪帧照样按全合算
OPEN_TIP_LENGTH = 0.20
TIP_LENGTH = 0.20 + 0.0295
GRIPPER_STROKE = 0.0875      # 两指内侧最大间隙（厂家 URDF：全开 87.5 mm）
# 厂家 URDF + STL 算的：两指间隙（mm）→ 指尖比全开时多伸出多少（mm）
FINGER_GAP_MM = [0.0, 10.9, 21.6, 32.0, 41.9, 51.3, 60.1, 68.2, 75.5, 81.9, 87.5]
FINGER_EXT_MM = [29.5, 28.7, 27.4, 25.5, 23.2, 20.3, 17.0, 13.3, 9.2, 4.8, 0.0]


def limit_msg(j, q, q_min, q_max):
    """第 j 个关节（0 起）快到限位时给人看的说明。q_min / q_max 是已经留了 JOINT_LIMIT_MARGIN_DEG 的范围。"""
    lim = q_min[j] - JOINT_LIMIT_MARGIN_DEG if q[j] < q_min[j] else q_max[j] + JOINT_LIMIT_MARGIN_DEG
    msg = f"J{j + 1} 快到限位了（{q[j]:.0f}°，限位 {lim:+.0f}°，要留 {JOINT_LIMIT_MARGIN_DEG}°）"
    if j == 4:
        # ECO65 和 UR 一样的手腕：J5 接近 ±180° 和接近 0° 一样，J4、J6 两根轴对齐（手腕奇异）。
        # 夹爪水平时就是夹爪指向快和 J4 轴平行了——夹爪侧着指（和“从底座看向物体”的方向差了快 90°）
        msg += "。J5 接近 ±180° 是手腕奇异：夹爪轴快和 J4 轴平行了，夹爪水平时把它转得更朝外（朝着从底座看过去的方向）"
    return msg


def finger_ext(gap):
    """两指间隙（米）→ 指尖比全开时多伸出多少（米）。gap 为 None 时按全合（最长）算。"""
    if gap is None:
        return TIP_LENGTH - OPEN_TIP_LENGTH
    return float(np.interp(gap * 1000, FINGER_GAP_MM, FINGER_EXT_MM)) / 1000


FLANGE_MIN_R = 0.20          # 法兰盘离底座竖直轴至少多远（米）。夹爪放平时法兰会往回缩，抓取中心就得离底座更远
FLANGE_MIN_R = 0.20          # 法兰盘离底座竖直轴至少多远（米）。夹爪放平时法兰会往回缩，抓取中心就得离底座更远
MAX_JOINT_JUMP_DEG = 5       # 相邻两帧指令的逆解，关节最多差多少度；超过说明逆解换了一种姿态，拒绝
JOINT_LIMIT_MARGIN_DEG = 3   # 离关节限位还剩这么多度就不让再往那边走
SINGULAR_MIN_DEG = 10        # J3（手臂伸直）、J5（手腕两根轴对齐）离 0° 少于这么多度就不让再靠近：奇异姿态附近逆解会突然换解
TRACK_ERR_MM = 30            # 抓取中心实际和指令差超过 这么多（毫米）+ TRACK_LAG_S × 指令速度 ……
TRACK_LAG_S = 0.5            # 真机大约比指令慢这么多秒：指令走得快时允许的偏差相应放宽，被挡住时偏差会一直涨，照样能发现
TRACK_ERR_FRAMES = 5         # ……连续这么多帧，就自动暂停
STALE_S = 0.3                # 实际状态超过这么久（秒）没更新，也自动暂停
# ---- 标记手柄 ----
MARKER_SCALE = 1.0           # 手动 1 cm，夹爪动多少 cm（转动同理）
MARKER_SCALE_FINE = 0.4      # 按住 B 时的比例
MARKER_ABSOLUTE = True       # True：绝对对应（默认）；False：相对（每次按 LB 重新对应）。命令行 --marker-map 可以改
MARKER_SPEED_FACTOR = 2.0    # 夹爪追手的最快速度 = 摇杆最快速度的几倍；手动得太快时夹爪会慢一点跟上，不会猛冲


class MockRobot:
    """假机械臂：只记录目标，不发给任何硬件。"""

    def __init__(self):
        self.pos = HOME_POS.copy()
        self.rot = R.from_euler("xyz", HOME_EULER)
        self.gripper_closed = False

    def home(self):
        self.pos, self.rot = HOME_POS.copy(), R.from_euler("xyz", HOME_EULER)

    def send_pose(self, pos, rot):
        self.pos, self.rot = pos.copy(), rot
        return 0

    def read_state(self):
        return time.time(), [float("nan")] * 6, [*self.pos, *self.rot.as_euler("xyz")]

    def check_pose(self, flange, rot):
        return None, float("inf"), ""

    def actual_margin(self):
        return None

    def set_gripper(self, closed):
        if closed != self.gripper_closed:
            print(f"\n[夹爪] {'闭合' if closed else '张开'}")
        self.gripper_closed = closed

    def stop(self):
        pass

    def emergency_stop(self):
        print("\n[急停] （假机械臂）")

    def release_estop(self):
        return True

    def close(self):
        pass


class RealmanRobot:
    """睿尔曼机械臂（SDK：pip install Robotic_Arm）。

    real=False 时切到仿真模式：控制器照常算逆解、查限位，但真实机械臂不动，
    只有示教器网页里的三维模型动。代码和真实模式完全一样，验证完切 real=True 即可。
    """

    def __init__(self, ip, port, real=False, home=True, gripper=True):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_inverse_kinematics_params_t, rm_thread_mode_e
        self.ik_params = rm_inverse_kinematics_params_t
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        if self.arm.rm_create_robot_arm(ip, port).id == -1:
            raise SystemExit(f"连不上机械臂 {ip}:{port}，检查网线，以及笔记本 IP 是否在 192.168.1.x 网段")
        ret, dh = self.arm.rm_get_DH_data()   # 用控制器里的 DH 参数自己算手臂各段的位置
        if ret != 0:
            self.arm.rm_delete_robot_arm()
            raise SystemExit(f"读 DH 参数失败（错误码 {ret}）")
        self.dh = {k: np.array(v, dtype=float) for k, v in dh.items()}
        self.q_min = np.array(self.arm.rm_get_joint_min_pos()[1], dtype=float) + JOINT_LIMIT_MARGIN_DEG
        self.q_max = np.array(self.arm.rm_get_joint_max_pos()[1], dtype=float) - JOINT_LIMIT_MARGIN_DEG
        mode = 1 if real else 0          # SDK 约定：0 仿真，1 真实
        ret = self.arm.rm_set_arm_run_mode(mode)
        ret2, now = self.arm.rm_get_arm_run_mode()
        if ret != 0 or ret2 != 0 or now != mode:
            self.close()
            raise SystemExit(f"切换{'真实' if real else '仿真'}模式失败（错误码 {ret}/{ret2}），先在示教器里手动切换")
        print(f"[机械臂] 已连接 {ip}，当前为{'真实' if now else '仿真'}模式")
        self.gripper_closed = False
        self.gripper, self.want_gripper, self.gripper_try_t = None, gripper, 0.0
        self.ip, self.port = ip, port
        self.connect_gripper()
        self.err_count = 0
        self.q_cmd, self.checked = None, None   # 上一帧指令的逆解，见 check_pose
        # 读一次状态要十几毫秒，放在控制循环里会拖慢它。所以另开一个连接，在后台线程里一直读，
        # 主循环直接拿最新的一份
        self.reader = RoboticArm()
        self.reader.rm_create_robot_arm(ip, port)
        # 急停单独用一个连接：不和运动指令、读状态、夹爪抢，别的线程卡住时也能立刻发出去
        self.estop_arm = RoboticArm()
        self.estop_arm.rm_create_robot_arm(ip, port)
        self.latest = None
        self.reading = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        if home:
            self.home()

    def _read_loop(self):
        while self.reading:
            ret, state = self.reader.rm_get_current_arm_state()
            if ret == 0:
                self.latest = (time.time(), state["joint"][:6], state["pose"])
            time.sleep(0.002)

    def home(self):
        """用规划好的关节运动回到初始位姿，走完才返回。
        透传只能走小步，从远处跳回初始位姿必须靠这个。"""
        self.q_cmd = None
        print("\n[机械臂] 正在回初始位姿……")
        ret = self.arm.rm_movej([float(v) for v in HOME_Q], HOME_SPEED, 0, 0, 1)
        if ret != 0:
            print(f"[警告] 回初始位姿失败，错误码 {ret}（查示教器有没有报错、急停有没有解除）")
        # 打印控制器读回的实际位姿，和 HOME_POS / HOME_EULER 对比，可以核对单位和欧拉角约定
        ret, state = self.arm.rm_get_current_arm_state()
        if ret == 0:
            p = state["pose"]
            print(f"[机械臂] 实际位姿 位置(mm): {p[0]*1000:.1f} {p[1]*1000:.1f} {p[2]*1000:.1f}  "
                  f"姿态(度): {np.degrees(p[3]):.1f} {np.degrees(p[4]):.1f} {np.degrees(p[5]):.1f}")

    def send_pose(self, pos, rot):
        # 位姿透传：不经规划直接发给控制器。follow=False 是低跟随，对发送周期要求宽松
        pose = [float(v) for v in [*pos, *rot.as_euler("xyz")]]
        ret = self.arm.rm_movep_canfd(pose, False)
        if self.checked is not None and np.allclose(self.checked[0], pose):
            self.q_cmd = self.checked[1]
        if ret != 0:
            self.err_count += 1
            if self.err_count % 50 == 1:   # 别每帧都刷屏
                print(f"\n[警告] 位姿透传失败，错误码 {ret}（累计 {self.err_count} 次）")
        return ret

    def send_joints(self, q):
        """关节角透传（低跟随）。网页示教从任意姿态出发的第一段用。"""
        q = [float(v) for v in q]
        ret = self.arm.rm_movej_canfd(q, False)
        self.q_cmd = np.array(q)
        if ret != 0:
            self.err_count += 1
            if self.err_count % 50 == 1:
                print(f"\n[警告] 关节透传失败，错误码 {ret}（累计 {self.err_count} 次）")
        return ret

    def read_state(self):
        """最新的实际状态：(读到的时刻, 6 个关节角(度), 法兰位姿 xyz(米)+欧拉角(弧度))，还没读到时是 None。"""
        return self.latest

    def check_pose(self, flange, rot):
        """发之前先检查法兰目标位姿。返回 (硬性问题, 余量, 说明)：
        硬性问题是逆解失败、关节跳变、自碰撞，不为 None 就不能走；
        余量是手臂和夹爪离桌面安全线最近还差多少（米），负数表示越线。"""
        st = self.latest
        if st is None:
            return "还没读到实际状态", 0.0, ""
        # 参考关节角：上一帧发出去的指令的逆解；刚开始或刚停下时用实际关节角。
        # 不直接跟实际比，是因为真机比指令慢半拍，转姿态时关节能差十几度，是正常的
        q_ref = self.q_cmd if self.q_cmd is not None else np.array(st[1], dtype=float)
        hard, margin, where, q = self.solve(flange, rot, q_ref)
        if hard is None:
            self.checked = ([float(v) for v in [*flange, *rot.as_euler("xyz")]], q)   # 真发出去时，它就成了下一帧的参考
        return hard, margin, where

    def finger_ext(self):
        """夹爪现在指尖比全开时多伸出多少（米）：读得到真实开口就按真实开口算，读不到按指令（合上按全合）。"""
        g = getattr(self, "gripper", None)
        w = g.width() if g is not None else None
        if w is None:
            return finger_ext(None) if self.gripper_closed else 0.0
        return finger_ext(min(max(w, 0.0), 1.0) * GRIPPER_STROKE)

    def solve(self, flange, rot, q_ref, max_jump=MAX_JOINT_JUMP_DEG, ext=None):
        """从关节角 q_ref 出发算法兰目标位姿的逆解并做安全检查（本地计算，不经过控制器，很快）。
        返回 (硬性问题, 余量, 说明, 关节角)，含义同 check_pose；有硬性问题时关节角是 None。
        max_jump：和 q_ref 相比关节最多差多少度；只是看看某个目标能不能到（不是连续轨迹）时可以放宽。
        ext：离桌面检查时指尖多伸出多少（见 finger_ext）；不给就按夹爪现在的开口。"""
        q_ref = np.asarray(q_ref, dtype=float)
        pose = [float(v) for v in [*flange, *rot.as_euler("xyz")]]
        ret, q = self.arm.rm_algo_inverse_kinematics(self.ik_params(list(q_ref), pose, 1))
        if ret != 0:
            return "够不到（逆解失败）", 0.0, "", None
        q = np.array(q)
        if np.max(np.abs(q - q_ref)) > max_jump:
            return "关节要突然大转（接近奇异姿态，或逆解换了一种姿态）", 0.0, "", None
        # 关节限位：已经在限位附近时，只要不再往外走就放行，不然会卡死在那里
        over = ((q < self.q_min) & (q < q_ref)) | ((q > self.q_max) & (q > q_ref))
        if over.any():
            return limit_msg(int(np.argmax(over)), q, self.q_min, self.q_max), 0.0, "", None
        for j, name in [(2, "手臂快伸直了"), (4, "手腕快转到奇异姿态")]:   # 同样只拦“继续靠近”
            if abs(q[j]) < SINGULAR_MIN_DEG and abs(q[j]) < abs(q_ref[j]):
                return f"{name}（J{j + 1} 接近 0°）", 0.0, "", None
        if self.arm.rm_algo_safety_robot_self_collision_detection(list(q)) != 0:
            return "会和自身相撞", 0.0, "", None
        margin, where = model_margin(q, self.finger_ext() if ext is None else ext)   # 整台机械臂最低点离桌面至少 1 cm
        return None, margin, where, q

    def actual_margin(self):
        """按实际关节角算的离桌面余量 (余量, 说明)，用来兜底：预测没拦住时也能发现（主要防手臂、夹爪本体撞桌）。
        手指按张开时的长度算，不加合上时的伸出量：合夹爪那一下指尖往下伸是用户摆抓取点时自己把握的
        （2026-11-02 用户：自己设的落点能保证安全），不然低处一合夹爪执行就会被这里叫停。"""
        st = self.latest
        if st is None:
            return None
        return model_margin(st[1], 0.0)

    def connect_gripper(self):
        """连夹爪（知行 CTAG2F90-D，走末端 Modbus）。连不上不报错，之后按夹爪按钮时会每隔 5 秒再试一次
        （夹爪刚上电、或者重启过时第一次常连不上）。"""
        if not getattr(self, "want_gripper", False) or self.gripper is not None or time.time() - self.gripper_try_t < 5:
            return self.gripper
        self.gripper_try_t = time.time()
        try:
            from gripper import ZhixingGripper
            self.gripper = ZhixingGripper(self.ip, self.port)
            self.gripper.set_closed(self.gripper_closed)
        except Exception as e:
            print(f"[夹爪] 没连上（{e}），夹爪指令只打印，稍后再试")
        return self.gripper

    def set_gripper(self, closed):
        # 知行 CTAG2F90-D，走末端 Modbus（gripper.py）；没连上夹爪时只打印
        g = self.connect_gripper() if hasattr(self, "want_gripper") else getattr(self, "gripper", None)
        if g is not None:
            g.set_closed(closed)        # 每次都发：之前没连上时按过的按钮，状态和真夹爪可能对不上
        elif closed != self.gripper_closed:
            print(f"\n[夹爪] {'闭合' if closed else '张开'}（没连夹爪，只打印）")
        self.gripper_closed = closed

    def stop(self):
        """停住：把透传目标改成当前实际位姿，机械臂就不会继续往之前的目标走。"""
        self.q_cmd = None
        if self.latest is not None:
            self.arm.rm_movep_canfd([float(v) for v in self.latest[2]], False)

    def emergency_stop(self):
        """软件急停：控制器立刻停下机械臂，解除之前不再动。失败就退一步用轨迹急停。"""
        self.q_cmd = None
        arm = getattr(self, "estop_arm", self.arm)
        ret = arm.rm_set_arm_emergency_stop(True)
        if ret == -4:                    # 三代控制器（我们这台）不支持软件急停，用“轨迹急停”：关节以最快速度停下（实测 4 ms 返回）
            ret = arm.rm_set_arm_stop()
        elif ret != 0:
            print(f"\n[警告] 急停指令失败（错误码 {ret}），改用轨迹急停。不放心就按实体急停")
            ret = arm.rm_set_arm_stop()
        return ret

    def release_estop(self):
        """解除软件急停，成功返回 True。"""
        ret = getattr(self, "estop_arm", self.arm).rm_set_arm_emergency_stop(False)
        if ret == -4:                    # 三代控制器用的是轨迹急停，没有要解除的急停状态
            ret = 0
        if ret != 0:
            print(f"\n[警告] 解除急停失败（错误码 {ret}），去示教器里解除")
        return ret == 0

    def close(self):
        self.reading = False
        if getattr(self, "gripper", None) is not None:
            self.gripper.close()
        if hasattr(self, "reader"):
            time.sleep(0.05)             # 等后台线程读完最后一次
            self.reader.rm_delete_robot_arm()
        if hasattr(self, "estop_arm"):
            self.estop_arm.rm_delete_robot_arm()
        self.arm.rm_delete_robot_arm()


def deadzone(v):
    """去掉摇杆中心附近的小抖动，并把剩下的范围重新拉满到 0~1。"""
    if abs(v) < DEADZONE:
        return 0.0
    return float(np.sign(v) * (abs(v) - DEADZONE) / (1 - DEADZONE))


def read_pad(c):
    """读手柄，返回统一格式的字典。摇杆 -1~1，扳机 0~1。"""
    axis = lambda a: c.get_axis(a) / 32767.0
    btn = lambda b: bool(c.get_button(b))
    return {
        "lx": deadzone(axis(pygame.CONTROLLER_AXIS_LEFTX)),
        "ly": deadzone(axis(pygame.CONTROLLER_AXIS_LEFTY)),
        "rx": deadzone(axis(pygame.CONTROLLER_AXIS_RIGHTX)),
        "ry": deadzone(axis(pygame.CONTROLLER_AXIS_RIGHTY)),
        "lt": deadzone(max(axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT), 0.0)),
        "rt": deadzone(max(axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT), 0.0)),
        "A": btn(pygame.CONTROLLER_BUTTON_A),
        "B": btn(pygame.CONTROLLER_BUTTON_B),
        "X": btn(pygame.CONTROLLER_BUTTON_X),
        "Y": btn(pygame.CONTROLLER_BUTTON_Y),
        "LB": btn(pygame.CONTROLLER_BUTTON_LEFTSHOULDER),
        "DL": btn(pygame.CONTROLLER_BUTTON_DPAD_LEFT),
        "DR": btn(pygame.CONTROLLER_BUTTON_DPAD_RIGHT),
        "DU": btn(pygame.CONTROLLER_BUTTON_DPAD_UP),
        "DD": btn(pygame.CONTROLLER_BUTTON_DPAD_DOWN),
        "L3": btn(pygame.CONTROLLER_BUTTON_LEFTSTICK),   # 按下左摇杆
        "RB": btn(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER),
        "START": btn(pygame.CONTROLLER_BUTTON_START),
        "BACK": btn(pygame.CONTROLLER_BUTTON_BACK),
    }


def tilt_deg(rot):
    """夹爪轴（末端 z 轴）偏离竖直朝下多少度。"""
    return float(np.degrees(np.arccos(np.clip(-rot.apply([0.0, 0.0, 1.0])[2], -1.0, 1.0))))


def arm_points(q_deg, dh):
    """按关节角算手臂各段的位置（改进 DH 正解，和 SDK 的正解核对过），每段沿轴线取 5 个点。
    返回 [(部位, 点), ...]。最后一个关节坐标系的原点就是法兰盘。"""
    T, origins = np.eye(4), []
    for i, q in enumerate(np.radians(q_deg)):
        th, al = q + np.radians(dh["offset"][i]), np.radians(dh["alpha"][i])
        a, d = dh["a"][i], dh["d"][i]
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]])
        origins.append(T[:3, 3].copy())
    names = ["", "肩部", "肘部", "小臂", "手腕", "手腕"]   # 第 k 段：从第 k 个坐标系原点到第 k+1 个
    return [(names[k], origins[k - 1] + s * (origins[k] - origins[k - 1]))
            for k in range(1, 6) for s in np.linspace(0.0, 1.0, 5)]


MODEL_CLEARANCE = 0.01       # 整台机械臂（各连杆外壳 + 夹爪，夹爪按全合算）的最低点离桌面至少 1 cm（2026-10-27 用户从 2 cm 改成 1 cm）
FINGER_CLEARANCE = 0.003     # 手指那一段（指尖）只要离桌面 3 mm：指尖本来就要贴近桌面去抓东西（2026-11-02 用户手动抓得很顺，
                             # 按 1 cm 算却报“合上时太低 0.6 cm”——其实合上时指尖还离桌 4 mm）。手臂外壳和夹爪本体仍然 1 cm
GRIPPER_BODY_R = 0.045       # 夹爪按圆柱算的半径（米）：本体 70×50 mm 的对角一半约 4.3 cm，手指张开最宽 ±4.5 cm
GRIPPER_BODY_LEN = 0.134     # 法兰到夹爪本体前端（手指根部）的距离：转接件约 1.9 cm + 本体 11.5 cm
_MODEL = None


def _load_model():
    """读 twin_web/robot/ECO65-B.urdf 和 STL：每个连杆外壳取凸包顶点（连杆自己的坐标系里），关节原点和轴。
    URDF 的关节角就是 SDK 的关节角（正解和 SDK 核对过）。底座和 1 号连杆只会绕竖直轴转、碰不到桌面，不算。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import struct
    import xml.etree.ElementTree as ET
    from scipy.spatial import ConvexHull
    robot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_web", "robot")
    urdf = ET.parse(os.path.join(robot_dir, "ECO65-B.urdf")).getroot()

    def stl(path):
        b = open(path, "rb").read()
        n = struct.unpack("<I", b[80:84])[0]
        a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("x", "<u2")]), count=n)
        v = a["v"].reshape(-1, 3).astype(float)
        return v[ConvexHull(v).vertices]

    joints = []
    for j in urdf.findall("joint"):
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", [float(x) for x in o.get("rpy").split()]).as_matrix()   # URDF rpy：固定轴 x→y→z
        T[:3, 3] = [float(x) for x in o.get("xyz").split()]
        child = j.find("child").get("link")
        mesh = urdf.find(f"link[@name='{child}']/visual/geometry/mesh")
        pts = stl(os.path.join(robot_dir, "meshes", mesh.get("filename").split("/")[-1])) if mesh is not None else None
        axis = np.array([float(x) for x in j.find("axis").get("xyz").split()])
        joints.append((T, axis, child, pts if child not in ("link_1",) else None))
    # 夹爪按真实朝向的长方体算（法兰坐标系，z 沿夹爪伸出方向）。夹爪装上后手指沿法兰 y 方向开合（网页 GRIPPER_YAW_DEG=90）：
    # 本体 50（x）× 70.6（y）mm；手指那一段按张开最宽 ±51 mm（y）、手指宽 25 mm（x）算，长度到全合时的指尖
    def box(hx, hy, z0, z1):
        g = np.array(np.meshgrid([-hx, hx], np.linspace(-hy, hy, 5), np.linspace(z0, z1, 5))).reshape(3, -1).T
        return g
    body = box(0.025, 0.0353, 0.0, GRIPPER_BODY_LEN)
    fingers = box(0.0125, 0.051, GRIPPER_BODY_LEN, OPEN_TIP_LENGTH)   # 全开时的手指；合上多伸出的部分按开口另加
    _MODEL = (joints, body, fingers)
    return _MODEL


LINK_NAMES = {"link_2": "大臂", "link_3": "小臂", "link_4": "手腕", "link_5": "手腕", "link_6": "法兰"}


def model_margin(q_deg, ext=None):
    """按关节角算整台机械臂（连杆外壳 + 夹爪）的最低点离桌面安全线还差多少：返回 (余量 米, 说明)。
    ext：指尖比全开时多伸出多少（米，见 finger_ext）；None 按全合算（最保守）。"""
    joints, body, fingers = _load_model()
    ext = finger_ext(None) if ext is None else ext
    fingers = fingers.copy()
    tip = fingers[:, 2] >= OPEN_TIP_LENGTH - 1e-9
    fingers[tip, 2] += ext
    T, best = np.eye(4), (np.inf, "")
    for (T0, axis, child, pts), q in zip(joints, np.radians(q_deg)):
        Tq = np.eye(4)
        Tq[:3, :3] = R.from_rotvec(axis * q).as_matrix()
        T = T @ T0 @ Tq
        if pts is not None:
            z = float(np.min(pts @ T[2, :3] + T[2, 3]))
            best = min(best, (z - MODEL_CLEARANCE, f"{LINK_NAMES.get(child, child)}离桌面太近"))
    for pts, name, gap in ((body, "夹爪", MODEL_CLEARANCE), (fingers, "夹爪尖", FINGER_CLEARANCE)):
        best = min(best, (float(np.min(pts @ T[2, :3] + T[2, 3])) - gap, f"{name}离桌面太近"))
    return best[0] - TABLE_Z, best[1]


def gripper_points(flange, rot):
    """夹爪轴线上的点：从法兰盘到指尖（按全合时的长度）取 5 个。"""
    axis = rot.apply([0.0, 0.0, 1.0])
    return [("夹爪", flange + s * TIP_LENGTH * axis) for s in np.linspace(0.0, 1.0, 5)]


def gripper_margin(flange, rot):
    """夹爪离桌面的余量：身体留 GRIPPER_CLEARANCE，指尖那一点只留 TIP_CLEARANCE。"""
    pts = gripper_points(flange, rot)
    return min(lowest(pts[:-1], GRIPPER_CLEARANCE), lowest([("夹爪尖", pts[-1][1])], TIP_CLEARANCE))


def lowest(points, clearance):
    """离桌面安全线最近的点：返回 (余量 米, 说明)，余量 = 高度 - 桌面 - 要留的间隙。"""
    return min((p[2] - TABLE_Z - clearance, f"{name}离桌面太近") for name, p in points)


def home_target():
    """初始目标：夹爪竖直朝下，法兰盘在 HOME_POS。返回姿态和抓取中心位置。"""
    rot = R.from_euler("xyz", HOME_EULER)
    return rot, HOME_POS + rot.apply([0.0, 0.0, TOOL_LENGTH])


def center_of(flange_pose):
    """法兰位姿（xyz + 欧拉角）→ 抓取中心位置。"""
    return np.array(flange_pose[:3]) + R.from_euler("xyz", flange_pose[3:6]).apply([0.0, 0.0, TOOL_LENGTH])


CMD_HEADER = ["t", "enabled", "slow", "lx", "ly", "rx", "ry", "lt", "rt", "LB", "RB", "DL", "DR", "A", "Y", "gripper_closed", "clutch",
              "cmd_cx", "cmd_cy", "cmd_cz", "cmd_rx", "cmd_ry", "cmd_rz", "cmd_tilt_deg",
              "cmd_fx", "cmd_fy", "cmd_fz", "send_ret", "cmd_margin_mm", "safety"]
ACT_HEADER = (["act_age_ms", "act_fx", "act_fy", "act_fz", "act_rx", "act_ry", "act_rz",
               "act_cx", "act_cy", "act_cz", "act_tilt_deg", "err_center_mm", "act_margin_mm"]
              + [f"act_j{i}" for i in range(1, 7)])
LOG_HEADER = CMD_HEADER + ACT_HEADER
# 前缀 cmd 是发出的指令，act 是控制器读回的实际值；c 是抓取中心，f 是法兰盘，r 是欧拉角（弧度），j 是关节角（度）
# margin 是离桌面安全线的余量，safety 是这一帧安全保护拦了什么


def safety_check(robot, pos, rot):
    """检查抓取中心目标 pos、姿态 rot 能不能走。返回 (硬性问题, 余量 米, 说明)，见 RealmanRobot.check_pose；
    另外加一条：法兰盘离底座竖直轴不能太近。"""
    flange = pos - rot.apply([0.0, 0.0, TOOL_LENGTH])
    hard, margin, where = robot.check_pose(flange, rot)
    return (hard, *min((margin, where), (float(np.hypot(flange[0], flange[1])) - FLANGE_MIN_R, "法兰离底座太近")))


def step_toward(pos, rot, p_goal, r_goal, max_lin, max_ang):
    """朝目标位姿走一步，但平移不超过 max_lin 米、转动不超过 max_ang 弧度。"""
    d = p_goal - pos
    if np.linalg.norm(d) > max_lin:
        d *= max_lin / np.linalg.norm(d)
    w = (r_goal * rot.inv()).as_rotvec()
    if np.linalg.norm(w) > max_ang:
        w *= max_ang / np.linalg.norm(w)
    return pos + d, R.from_rotvec(w) * rot


def run(c, robot, marker=None):
    """marker：标记手柄模式时是 (MarkerTracker, 相机→机械臂的旋转矩阵, 是否绝对对应)，否则 None。"""
    dt = 1.0 / RATE_HZ
    rot, pos = home_target()     # pos 是抓取中心的位置（TOOL_LENGTH=0 时就是法兰盘）
    vel = np.zeros(7)            # 平滑后的速度：vx vy vz，绕世界 x y z 轴的角速度，绕夹爪自身轴的角速度
    enabled = False
    estop = False                # 按了 X 急停、还没解除
    gripper_closed = False
    prev = {}
    last_print = 0.0
    last_warn = 0.0
    bad_frames = 0               # 偏差连续超标了多少帧
    cmd_speed = 0.0              # 指令最近的移动速度（毫米/秒，平滑过），用来放宽偏差门限
    anchor = None                # 按下 LB 那一刻：(标记位置, 标记姿态, 夹爪目标位置, 夹爪目标姿态, 比例)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, time.strftime("teleop_%Y%m%d_%H%M%S.csv"))
    log = csv.writer(open(log_path, "w", newline="", buffering=1))   # 按行写盘，程序中途崩了也不丢
    log.writerow(LOG_HEADER)
    t_start = time.time()
    print("日志:", log_path)
    print("按 START 开始控制，BACK 退出" if not isinstance(c, KeyPad) else "按回车开始跟手，q 退出")

    while True:
        t0 = time.time()
        send_ret, margin_mm, safety = "", "", ""
        pygame.event.pump()

        # 安全：手柄断开就立刻停住
        if not c.attached():
            if enabled:
                print("\n[警告] 手柄断开，已停止控制")
            enabled, vel[:] = False, 0.0
            robot.stop()
            time.sleep(0.2)
            continue

        pad = read_pad(c)
        # 只在“刚按下”那一刻触发一次（这一帧按下、上一帧没按）
        just = {k: v and not prev.get(k, False) for k, v in pad.items() if isinstance(v, bool)}
        pressed = lambda k: just[k]
        prev = {k: v for k, v in pad.items() if isinstance(v, bool)}

        # ---- 急停：X 随时都能按；解除要按住 X 再按 START，免得误触 ----
        start = pressed("START")
        if (pressed("X") or pressed("L3")) and not estop:
            estop, enabled, vel[:], safety = True, False, 0.0, "急停"
            robot.emergency_stop()
            print("\n[急停] 已急停！确认安全后，按住 X（或左摇杆）再按 START 解除")
        elif estop and start:
            start = False                # 急停中 START 不用来开始控制
            if not (pad["X"] or pad["L3"]):
                print("\n[急停] 急停中。要解除：按住 X（或左摇杆）再按 START")
            elif robot.release_estop():
                estop = False
                print("\n[急停] 已解除。按 START 继续控制（会从实际位置接着走）")

        if pressed("BACK"):
            print("\n退出" + ("（急停还没解除，去示教器里解除）" if estop else ""))
            robot.stop()
            return
        if start:
            enabled, vel[:] = not enabled, 0.0
            if enabled:
                # 从实际位姿接着走：暂停期间积下的偏差不会变成开始时的一下猛冲
                st = robot.read_state()
                if st is None:
                    enabled = False
                    print("\n[控制] 还没读到实际状态，先别开始")
                else:
                    rot, pos = R.from_euler("xyz", st[2][3:6]), center_of(st[2])
                    bad_frames = 0
            else:
                robot.stop()
            print(f"\n[控制] {'开始' if enabled else '暂停'}")
        if pressed("A") or (marker and pressed("DD")):
            gripper_closed = not gripper_closed
            robot.set_gripper(gripper_closed)
        if marker and pressed("DU"):
            anchor = None
            print("\n[标记手柄] 重新对应：下次按住 LB 时，手的当前位置对应夹爪的当前位置")
        if pressed("Y") and not estop:
            anchor = None
            (rot, pos), vel[:] = home_target(), 0.0
            print("\n[复位] 回到初始位姿")
            robot.home()

        # ---- 手柄 → 目标速度 ----
        slow = pad["B"]
        lin_s = LIN_SPEED_SLOW if slow else LIN_SPEED
        ang_s = ANG_SPEED_SLOW if slow else ANG_SPEED
        # 机械臂基座坐标：x 向前，y 向左，z 向上（接真机时要核对）
        vx, vy, vz = -pad["ly"], -pad["lx"], pad["rt"] - pad["lt"]
        if marker:
            vz = 0.0                                     # 标记手柄模式下 LT 是精细模式，不拿来升降
        if HEADING_MODE:
            # 按夹爪当前朝向的水平分量，旋转“前后左右”的方向
            fwd = rot.apply([1.0, 0.0, 0.0])
            h = np.arctan2(fwd[1], fwd[0])
            vx, vy = vx * np.cos(h) - vy * np.sin(h), vx * np.sin(h) + vy * np.cos(h)
        # 右摇杆往前推，夹爪尖往前（+x）倒，要绕 y 轴负向转；往右推，夹爪尖往右（-y）倒，要绕 x 轴负向转
        tilt_fwd = 0.0 if LOCK_TILT else -pad["ry"]
        tilt_right = 0.0 if LOCK_TILT else pad["rx"]
        turn = float(pad["DL"]) - float(pad["DR"])       # 绕世界 z 轴（朝上）：正向是从上往下看逆时针
        spin = float(pad["RB"]) - float(pad["LB"])       # 绕夹爪自身的轴（朝夹爪尖）：夹爪朝下时正向是从上往下看顺时针
        if marker:
            spin = 0.0                                   # 标记手柄模式下 LB 是离合，自转交给手柄
        cmd = np.array([vx * lin_s, vy * lin_s, vz * lin_s,
                        -tilt_right * ang_s, -tilt_fwd * ang_s, turn * ang_s, spin * ang_s])
        if not enabled:
            cmd[:] = 0.0
        vel = SMOOTH * cmd + (1 - SMOOTH) * vel

        # ---- 读实际状态；出问题就自动暂停 ----
        # 实际抓取中心用读回的法兰位姿换算；偏差大说明机械臂没跟上指令（比如被桌子挡住了）
        st = robot.read_state()
        act_margin = robot.actual_margin()
        if st is not None:
            act_c = center_of(st[2])
            err_mm = float(np.linalg.norm(act_c - pos) * 1000)
        if enabled:
            allowed = TRACK_ERR_MM + TRACK_LAG_S * cmd_speed
            bad_frames = bad_frames + 1 if st is not None and err_mm > allowed else 0
            reason = None
            if st is None or t0 - st[0] > STALE_S:
                reason = "读不到实际状态"
            elif bad_frames >= TRACK_ERR_FRAMES:
                reason = f"实际位置跟不上指令（偏差 {err_mm:.0f} mm，允许 {allowed:.0f} mm）"
            elif act_margin is not None and act_margin[0] < -0.02:
                reason = act_margin[1]
            if reason:
                enabled, vel[:], safety = False, 0.0, "自动暂停：" + reason
                robot.stop()
                print(f"\n[安全] {reason}，已自动暂停。确认安全后按 START 继续（会从实际位置接着走）")

        # ---- 标记手柄：按住 LB 时，夹爪目标 = 按下那一刻的目标 + 手相对那一刻的移动和转动 ----
        goal, clutch_state = None, ""
        if marker and enabled and pad["LB"]:
            tracker, R_rc, absolute = marker
            mk = tracker.get()
            scale = MARKER_SCALE_FINE if pad["B"] or pad["lt"] > 0.5 else MARKER_SCALE
            if anchor is not None and anchor[4] != scale:
                anchor = None                            # 中途切换精细模式：重新记一次起点，免得跳
            if mk is None:
                clutch_state = "lost"                   # 看不到标记：夹爪停在原地等
            elif anchor is None:
                anchor = (mk[0], mk[1], pos.copy(), rot, scale)
                clutch_state = "anchor"
            else:
                dp = R_rc @ (mk[0] - anchor[0]) * scale                          # 相机坐标 → 机械臂坐标
                dw = R_rc @ (mk[1] * anchor[1].inv()).as_rotvec() * scale        # 转动也换到机械臂坐标
                goal = (anchor[2] + dp, R.from_rotvec(dw) * anchor[3])
                far = (np.linalg.norm(goal[0] - pos) > 0.01 or (goal[1] * rot.inv()).magnitude() > np.radians(3))
                clutch_state = "approach" if far else "follow"   # approach：正在限速追到手对应的位置
        elif not (marker and marker[2] and enabled):
            anchor = None                                # 相对对应：松开就作废；绝对对应：一直保留，直到暂停/重新对应

        # ---- 积分得到新的目标位姿 ----
        pos_before = pos.copy()
        if enabled:
            if goal is not None:
                vel[:] = 0.0
                new_pos, new_rot = step_toward(pos, rot, *goal, MARKER_SPEED_FACTOR * LIN_SPEED * dt,
                                               MARKER_SPEED_FACTOR * ANG_SPEED * dt)
            else:
                new_pos = pos + vel[:3] * dt
            # 限位范围把当前位置也包进来：已经在范围外时不会被一下子拽回去，只是不能再往外走
            clipped = np.clip(new_pos, np.minimum(WORKSPACE[:, 0], pos), np.maximum(WORKSPACE[:, 1], pos))
            if not np.allclose(clipped, new_pos):
                vel[:3] = np.where(np.isclose(clipped, new_pos), vel[:3], 0.0)
            if goal is None:
                spin_rot = R.from_rotvec([0.0, 0.0, vel[6] * dt])           # 右乘：绕夹爪自身的轴转
                new_rot = R.from_rotvec(vel[3:6] * dt) * rot * spin_rot     # 左乘：绕世界坐标轴转
            if tilt_deg(new_rot) > MAX_TILT_DEG and tilt_deg(new_rot) > tilt_deg(rot):
                if goal is None:
                    new_rot, vel[3:5] = R.from_rotvec([0.0, 0.0, vel[5] * dt]) * rot * spin_rot, 0.0   # 倒到上限就停住，另外两种转法照转
                else:
                    new_rot = rot                        # 标记手柄：倒到上限就不再跟着倒

            # 安全检查：整步不行就只平移，再不行就只转动，都不行就停在原地。
            # 离桌面的余量只要不越线、或者比现在大（正在往安全的方向走），就放行
            _, cur_margin, _ = safety_check(robot, pos, rot)
            for cand_pos, cand_rot, keep in [(clipped, new_rot, slice(0, 7)), (clipped, rot, slice(0, 3)),
                                             (pos, new_rot, slice(3, 7))]:
                hard, margin, where = safety_check(robot, cand_pos, cand_rot)
                if hard is None and (margin >= 0 or margin > cur_margin):
                    moving = np.zeros(7, dtype=bool)
                    moving[keep] = True
                    vel[~moving] = 0.0
                    pos, rot = cand_pos, cand_rot
                    break
                safety = "拦住：" + (hard or where)
            else:
                vel[:] = 0.0
            if safety and time.time() - last_warn > 1.0:
                print(f"\n[安全] {safety[3:]}", end="")
                last_warn = time.time()
            margin_mm = f"{safety_check(robot, pos, rot)[1] * 1000:.0f}"
            send_ret = robot.send_pose(pos - rot.apply([0.0, 0.0, TOOL_LENGTH]), rot)   # 抓取中心位置换回法兰盘位置

        # 指令速度：用时间常数约 0.5 秒平滑，机械臂追上来的过程中门限慢慢收回去
        cmd_speed += min(1.0, dt / TRACK_LAG_S) * (np.linalg.norm(pos - pos_before) * 1000 / dt - cmd_speed)

        # ---- 记日志：这一帧的手柄输入、指令、最新读回的实际状态 ----
        flange = pos - rot.apply([0.0, 0.0, TOOL_LENGTH])
        if st is not None:
            act_cols = [(t0 - st[0]) * 1000, *st[2][:6], *act_c, tilt_deg(R.from_euler("xyz", st[2][3:6])), err_mm,
                        act_margin[0] * 1000 if act_margin else "", *st[1]]
        else:
            act_cols = [""] * len(ACT_HEADER)
        log.writerow([f"{v:.5f}" if isinstance(v, float) else v for v in
                      [t0 - t_start, int(enabled), int(pad["B"]), pad["lx"], pad["ly"], pad["rx"], pad["ry"],
                       pad["lt"], pad["rt"], int(pad["LB"]), int(pad["RB"]), int(pad["DL"]), int(pad["DR"]), int(pad["A"]), int(pad["Y"]),
                       int(gripper_closed), clutch_state, *pos, *rot.as_euler("xyz"), tilt_deg(rot), *flange, send_ret,
                       margin_mm, safety, *act_cols]])

        # ---- 每 0.2 秒刷新一次状态 ----
        if time.time() - last_print > 0.2:
            act_txt = (f"实际 {act_c[0]*1000:4.0f} {act_c[1]*1000:4.0f} {act_c[2]*1000:4.0f}  偏差 {err_mm:4.0f}"
                       if st is not None else "实际 还没读到")
            mode = {"follow": " 跟手", "anchor": " 跟手", "approach": " 追到手的位置", "lost": " 看不到标记"}.get(clutch_state, "")
            print(f"\r{'急停中' if estop else '控制中' if enabled else '暂停中'}{mode} {'慢速' if slow else '正常'} "
                  f"夹爪:{'合' if gripper_closed else '开'}  "
                  f"抓取中心(mm) 指令 {pos[0]*1000:4.0f} {pos[1]*1000:4.0f} {pos[2]*1000:4.0f}  {act_txt}  "
                  f"倾斜:{tilt_deg(rot):3.0f}°   ", end="", flush=True)
            last_print = time.time()

        time.sleep(max(0.0, dt - (time.time() - t0)))


class KeyPad:
    """没带游戏手柄时用终端键盘代替（只给标记手柄模式用）。按键变成手柄按钮的一次“短按”，LB（离合）一直按着，
    所以开始控制后只要相机看得到标记，夹爪就跟着走；看不到就停在原地等。
    回车/空格＝START（开始/暂停）、g＝夹爪开合、f＝精细模式开关、h＝回初始位姿、x＝急停、r＝解除急停、q＝退出。"""
    name = "终端键盘（没连游戏手柄）"
    PULSE_S = 0.12                       # 一次按键当成按住 0.12 秒（控制循环 50 Hz，够它看到“刚按下”）
    KEYS = {"\n": "START", "\r": "START", " ": "START", "g": "DD", "h": "Y", "x": "X", "q": "BACK", "m": "DU"}
    BTN = {pygame.CONTROLLER_BUTTON_START: "START", pygame.CONTROLLER_BUTTON_X: "X", pygame.CONTROLLER_BUTTON_Y: "Y",
           pygame.CONTROLLER_BUTTON_BACK: "BACK", pygame.CONTROLLER_BUTTON_DPAD_DOWN: "DD",
           pygame.CONTROLLER_BUTTON_DPAD_UP: "DU"}

    def __init__(self):
        import sys, termios, tty, atexit
        if not sys.stdin.isatty():
            raise SystemExit("没连游戏手柄，又不是在终端里运行，没法用键盘控制")
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)           # 按一个键就读到，不用回车、不回显；Ctrl+C 照样能退出
        atexit.register(self.close)
        self.until, self.fine, self.estopped = {}, False, False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                ch = os.read(self.fd, 1).decode(errors="ignore").lower()
            except OSError:
                return
            now = time.time()
            if ch == "f":
                self.fine = not self.fine
                print(f"\n[键盘] 精细模式{'开' if self.fine else '关'}")
            elif ch == "r":
                if self.estopped:        # 解除急停＝按住 X 再按 START
                    self.until["X"] = self.until["START"] = now + self.PULSE_S
                    self.estopped = False
            elif ch in self.KEYS:
                b = self.KEYS[ch]
                self.estopped = self.estopped or b == "X"
                self.until[b] = now + self.PULSE_S

    def attached(self):
        return True

    def get_button(self, b):
        if b == pygame.CONTROLLER_BUTTON_LEFTSHOULDER:
            return True                  # LB（离合）一直按着
        name = self.BTN.get(b)
        return name is not None and time.time() < self.until.get(name, 0.0)

    def get_axis(self, a):
        return 32767 if (a == pygame.CONTROLLER_AXIS_TRIGGERLEFT and self.fine) else 0

    def close(self):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def main():
    ap = argparse.ArgumentParser(description="手柄遥操作")
    ap.add_argument("--robot", choices=["mock", "realman"], default="mock", help="mock 假机械臂 / realman 睿尔曼")
    ap.add_argument("--ip", default="192.168.1.18")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--real", action="store_true", help="真实模式（机械臂会真的动），不加则是仿真模式")
    ap.add_argument("--no-gripper", action="store_true", help="不连夹爪（夹爪指令只打印）")
    ap.add_argument("--input", choices=["gamepad", "marker"], default="gamepad", help="marker：用标记手柄控制")
    ap.add_argument("--marker-cam", type=int, default=2, help="跟踪相机编号（/dev/video2 填 2）")
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--marker-size", type=float, default=0.050, help="标记黑色方块边长（米）")
    ap.add_argument("--marker-map", choices=["absolute", "relative"], default="absolute" if MARKER_ABSOLUTE else "relative",
                    help="absolute：手和夹爪一直按第一次按 LB 时的关系对应；relative：每次按 LB 重新对应")
    args = ap.parse_args()

    pygame.init()
    controller.init()
    use_keys = controller.get_count() == 0 or not controller.is_controller(0)
    if use_keys and args.input != "marker":
        raise SystemExit("没检测到标准手柄，先确认蓝牙已连接（标记手柄模式 --input marker 没手柄时可以用键盘）")
    c = None if use_keys else controller.Controller(0)
    print("手柄:", "没连，用终端键盘" if use_keys else c.name)

    if args.robot == "realman":
        if args.real and input("真实模式下机械臂会真的动，手放在急停旁边。输入 yes 继续：") != "yes":
            raise SystemExit("已取消")
        if args.real:
            # 真机保守一点：速度减半、回初始位姿更慢。高度不再单独设下限，由按关节角算的桌面保护来管
            global LIN_SPEED, ANG_SPEED, HOME_SPEED
            LIN_SPEED, ANG_SPEED, HOME_SPEED = LIN_SPEED / 2, ANG_SPEED / 2, 10
            print(f"[真实模式] 平移 {LIN_SPEED*100:.0f} cm/s，夹爪指尖离桌面至少 {TIP_CLEARANCE*100:.0f} cm，"
                  f"手臂离桌面至少 {ARM_CLEARANCE*100:.0f} cm")
        robot = RealmanRobot(args.ip, args.port, real=args.real, gripper=not args.no_gripper)
    else:
        robot = MockRobot()
    marker = None
    if args.input == "marker":
        from marker_align import ALIGN_FILE
        from marker_tracker import MarkerTracker
        old_file = ALIGN_FILE.replace(".npz", ".npy")   # 旧版对齐文件也认
        if os.path.exists(ALIGN_FILE):
            R_rc = np.load(ALIGN_FILE)["R_rc"]
        elif os.path.exists(old_file):
            R_rc = np.load(old_file)
        else:
            robot.close()
            raise SystemExit("还没对齐方向：先运行 python marker_align.py")
        marker = (MarkerTracker(args.marker_cam, args.marker_id, args.marker_size), R_rc,
                  args.marker_map == "absolute")
        if use_keys:
            print(f"[标记手柄] {'绝对' if marker[2] else '相对'}对应，用键盘：回车/空格 开始·暂停跟手；g 夹爪开合；"
                  "f 精细模式；h 回初始位姿；x 急停，r 解除；q 退出")
        else:
            print(f"[标记手柄] {'绝对' if marker[2] else '相对'}对应。按住 LB 跟手；按住 LT 精细；十字键上 重新对应；十字键下 夹爪开合；按下左摇杆 急停")
    try:
        if use_keys:
            c = KeyPad()
        run(c, robot, marker)
    finally:
        try:
            robot.stop()                 # Ctrl+C 退出时也让机械臂停在当前位置
        except Exception:
            pass
        robot.close()
        if marker:
            marker[0].close()


if __name__ == "__main__":
    main()
