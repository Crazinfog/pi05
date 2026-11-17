"""三维示教网页的后台：浏览器里拖机械臂模型、打关键帧、预览轨迹，确认后真机执行。

    python twin_server.py            # 控制器仿真模式：真机不动，只有示教器里的模型动
    python twin_server.py --real     # 真实模式（要二次确认）
    python twin_server.py --offline  # 离线模拟：不连控制器，没有机械臂时调网页用

然后在同一网络里的电脑上用浏览器打开 http://172.16.230.23:8000 （Orin 自己打开 http://localhost:8000）。

分工：网页负责显示和交互；这里负责逆解、安全检查、规划轨迹、发指令给机械臂。
逆解和安全检查和 teleop.py 用的是同一套（本地逆解 + 关节跳变/限位/奇异/自碰撞 + 手臂和夹爪离桌面的高度）。
轨迹从第 1 帧开始，做法借鉴工业机器人和 MoveIt 的成熟方案（Pilz 的 LIN/PTP + 拐角过渡、MTC 的抓放分段、TOTG/TOPP 排时间）：
  - 路径：相邻帧之间走直线，不停的中间帧在拐角处用圆角过渡（不会像样条那样冲过头），带夹爪动作或停顿的帧准确停下；
    某段直线中途到不了时，那一段改成关节空间运动（各关节同步转），也不行才报问题；
  - 时间：沿路径每 3 mm 做一次逆解，按 平移 / 转动 / 每个关节 的速度上限和加速度上限排时间，再抹成 S 形加减速，
    奇异姿态附近关节转得快的地方会自动放慢；
  - 执行：所有点都按规划时解好的关节角发（和检查过的一模一样）；夹爪开合前先等真机真的到位停稳。
可选“人为扰动”：帧与帧之间路径随机鼓一点、姿态晃一点、快慢不均、夹爪动作前犹豫一下（关键帧本身照样准确），
用随机种子复现，预览和执行是同一条。执行时先用关节运动从机械臂实际位置移到第 1 帧（机械臂伸直等奇异姿态也能出发）。
"""
import argparse
import asyncio
import base64
import csv
import json
import os
import re
import threading
import time

import cv2
import numpy as np
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline, Slerp

import teleop as tp

import warnings
warnings.filterwarnings("ignore", message="Gimbal lock")   # 夹爪竖直朝下时欧拉角有一个轴不确定，scipy 会刷警告，不影响结果

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "twin_web")
TEMPLATE_DIR = os.path.join(HERE, "twin_templates")
REC_CFG_PATH = os.path.join(HERE, "record_config.json")   # 录制设置（存储路径、任务描述……），网页上改，存在这里
REC_DEFAULT = {"folder": os.path.join(HERE, "data", "demos"), "task": "", "depth": False, "on": False}
RATE_HZ = 50                     # 规划和执行的采样频率，和 teleop 一样
# 2026-11-03 用户觉得平移 10 cm/s 慢：真实模式上限从 10 cm/s、34°/s、关节 30°/s、0.25 m/s² 提到下面这些（LIBERO 真人示范最快 13~15 cm/s）。
# 开合点前后 5 cm 还是按“接近/离开”慢走，真机滞后约 0.5 s 时偏差报警按 30 mm + 0.5 s × 指令速度放宽，不会误报
SPEED_LIMIT = {False: (0.30, 1.5), True: (0.25, 1.0)}   # 仿真 / 真实模式下允许的最大 平移 m/s、转动 rad/s
JOINT_SPEED = {False: 90.0, True: 60.0}                 # 规划时每个关节最快多少 °/s（仿真 / 真实）
PATH_ACC = {False: 0.8, True: 0.5}                      # 沿路径的加速度上限（m/s²；转动按 ROT_LEN 折算成路程）
ROT_LEN = 0.1            # 转动折算成路程：1 弧度当 10 cm（平移和转动放在一起算路程、加速度）
BLEND_R = 0.04           # 不停的中间帧：最多提前这么远（米）开始拐，用圆角（二次贝塞尔）拐过去，路径不会冲出拐角
DENSE_STEP = (0.003, np.radians(1.5))   # 规划时路径每 3 mm / 1.5° 取一个点做逆解和安全检查
SMOOTH_S = 0.12          # 排好时间后，速度曲线用这么宽（秒）的滑动平均抹两遍：加速度不再突变（S 形加减速）
# 精细段慢、自由移动段快（DemoSpeedup CoRL 2025：整条均匀加速成功率掉一半，只加速自由移动段、精细段保持慢就不掉）：
# 夹爪真正要开合的那一帧前后 PRECISE_DIST 以内（伸进去夹、刚夹起、放下前、刚松开），平移限速 PRECISE_SPEED（网页“接近/离开”可调）
PRECISE_SPEED = 0.07     # m/s（2026-11-04 按 LIBERO 开合点 5 cm 内平均 5.6 cm/s 从 3 改成 5；2026-11-06 实测滑块 5 时平均只有 3.5（起停加减速），改 7，网页默认也是 7）
PRECISE_DIST = 0.05      # m
# 夹爪开合前先等真机到位（真机比指令慢约 0.5 秒）：停稳（0.2 秒里动了不到 0.3 mm）且离指令 ≤1.5 mm；
# 停稳了但还差 1.5~4 mm（到头了，静差）就等它停稳 0.5 秒再动；最多等 3 秒。
# 2026-11-02 以前是“≤4 mm 且这一帧没动 0.3 mm”：真机最后几毫米走得慢，每帧只动 0.1~0.2 mm 也算“停稳”，夹爪在还差 3.9 mm 时就合了
SETTLE_MM, SETTLE_MAX_MM, SETTLE_STILL_S, SETTLE_TIMEOUT_S = 1.5, 4.0, 0.2, 3.0
# 松手以后只再录这么多秒（2026-11-04 用户：和官方一致就裁）：LIBERO 采集脚本在任务成功后 11 步（20 Hz，约 0.55 s）就结束，
# 官方数据里最后一次张开夹爪后 spatial 约 1.1 s、object 约 1.7 s，末端只往上撤 2~3 cm。后面的撤出、回初始照样执行，只是不录
REC_TAIL_S = 1.2
HOVER_TILT_DEG = 15.0        # “先到方块上方再转夹爪”只管夹爪大致竖直朝下（倾斜小于这么多度）的抓取
# 真机移到目标（空格 / M、真机移到这帧）也照这个走（2026-11-06 用户：“到上方然后再旋转，统一一点”）：
GOTO_HOVER_UP = 0.08         # 低处的目标（抓取中心低于 GOTO_LOW_Z）先到它正上方这么高，转完再竖直下去（和自动生成的伸进距离一样）
GOTO_LOW_Z = 0.15
GOTO_NEAR_XY = 0.015         # 水平离目标这么近、方向也只差 GOTO_NEAR_YAW 以内：小修正，直接过去
GOTO_NEAR_YAW = 3.0
# 自动生成里合夹爪 / 松手后停多久：2026-11-06 夹爪速度调成 50（gripper.py）后实测发指令到合到底 0.5~0.6 s（含约 0.2 s 指令延迟，
# 夹 3.3 cm 方块约 0.45 s 夹紧），原来合后停 0.3 s 会没夹紧就抬；松手 0.2 s 时手指可能还没张开就往上撤、把方块带起来。
# 多停的那段静止帧转换时按 /keep 裁掉（夹爪还在动的帧不算静止、会留下），不会让模型学到“停很久”
GRIP_CLOSE_PAUSE = 0.6
GRIP_OPEN_PAUSE = 0.35
STACK_RELEASE_GAP = 0.002         # 上块底面在底块顶面上方 2 mm 松手
STACK_DESCENT_SPEED = 0.025
STACK_ALIGN_HEIGHT = 0.08           # 相对松爪位，腕部观察底块时的高度
STACK_ALIGN_SPEED = np.radians(25)
STACK_ALIGN_PAUSE = 0.30
STACK_RELEASE_ANGLE_DEG = 2.0
HOLD_TOL = 0.002             # 夹着东西时最多比抓取的位置低 2 mm（放下时物体底下就是桌面）
# 执行时小扰动（MimicGen CoRL 2023：执行生成的轨迹时给动作加小噪声，去掉噪声学出来的策略掉 30~40%；
# Data Quality in IL NeurIPS 2023：执行有扰动、示范照常纠正的“系统噪声”提高鲁棒性，动作标签带噪声反而有害）：
# 发给机械臂的是“规划轨迹 + 慢慢晃的偏移”，录进数据的动作仍是规划的干净值——数据里就有“偏了一点再被拉回轨迹”。
# 偏移是每个方向 3 个随机正弦叠起来（频率 EXEC_NOISE_HZ，慢慢晃不是高频抖），总的均方根＝网页上设的毫米数；
# 离夹爪开合点 EXEC_NOISE_FADE 以内渐渐收到 0（抓、放的那一下准确），开头和录制结束前 EXEC_NOISE_RAMP_S 里渐渐加上 / 收掉，
# 录制结束以后（回初始）不加。加了偏移的每个点都重新逆解、做一遍安全检查，不安全的地方附近不加
EXEC_NOISE_HZ = (0.15, 0.6)
EXEC_NOISE_FADE = (0.005, 0.02)     # 离开合点 0.5 cm 以内不加，2 cm 以外加满
EXEC_NOISE_RAMP_S = 0.5
EXEC_NOISE_MAX = 0.008              # 最多 8 mm
FIRST_STEP_MAX_DEG = 3.0   # 执行时第一步离机械臂实际关节角最多差这么多度：出发段从实际位置开始，正常不到 1°；超过说明规划有错，不执行
TRACK_STEP_JUMP = 15.0   # 拖目标时从上一个姿态每次 1 cm / 5° 一小步跟过去，一小步里关节最多转这么多度（再多就是真要换摆法）
FOLLOW_SPEED = {False: (0.10, 0.6), True: (0.05, 0.3)}  # 跟随模式最快 平移 m/s、转动 rad/s（仿真 / 真实）
FOLLOW_JOINT_SPEED = {False: 20.0, True: 10.0}           # 跟随模式调关节时，关节最快多少 °/s
JOG_SPEED = {False: (0.08, 0.8), True: (0.05, 0.55)}     # 键盘点动（WASD/RF/QE）时真机最快 平移 m/s、转动 rad/s（仿真 / 真实）
JOG_IDLE_S = 1.0                                         # 最后一次按键之后这么久、跟到了就收工
DRAG_MAX_JUMP = 30.0     # 拖目标时手臂摆法要连续：离上一个姿态最近的逆解也要有关节转超过这么多度，就当成要换摆法，不自动换
FALLBACK_DETOUR = 1.5   # 曲线走不通改关节运动时，抓取中心的路程最多是直线距离的这么多倍（再加 10 cm）；
                         # 超过说明逆解换了一种手臂摆法、手臂要绕一大圈（2026-10-27 仿真里 J1 转了 160° 绕到背后），不自动做
NEAR_BASE_Z = 0.45     # “法兰离底座太近”只管法兰低于这个高度（米）时：手臂折回底座附近又放低才会压到自己和桌面，
                        # 机械臂竖直伸直时法兰就在底座正上方很高处，不算


def near_base(flange):
    return np.hypot(flange[0], flange[1]) < tp.FLANGE_MIN_R and flange[2] < NEAR_BASE_Z


def to_json(m):
    """转成 JSON；numpy 的数和数组也能转。"""
    return json.dumps(m, ensure_ascii=False,
                      default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist())


def min_jerk(s):
    """最小加加速度曲线：0→1，起点终点速度和加速度都是 0，像人伸手一样慢起慢停。"""
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Twin:
    def __init__(self, robot, real):
        self.robot, self.real = robot, real
        self.lock = threading.Lock()         # 机械臂连接只能一个线程用
        self.executing = False
        self.stop_flag = False
        self.progress = (0, 0)
        self.status = "空闲"
        # 跟随模式：网页上拖目标或调关节时，机械臂实时（限速、逐帧安全检查）跟过去
        self.follow_on, self.follow_goal, self.follow_msg = False, None, ""
        self.jog_on, self.jog_until = False, 0.0   # 键盘点动：借跟随的循环让真机跟着目标（没开跟随时用）
        self.homing = False                        # 正在回初始位姿（rm_movej 阻塞着，不能同时发别的指令）
        self.drag_on = False                 # 拖动示教：用手拖机械臂，网页跟着显示
        self.estopped = False                # 急停后锁住：解除之前不执行、不跟随、不回初始（三代控制器的轨迹急停本身不锁）
        self.scene = None                    # 第三视角相机（scene.Scene），没接就是 None
        self.wrist = None                    # 腕部相机（cameras.DepthCamera），没接就是 None
        # 录示范数据（recorder.Recorder）：rec_cmd 是现在正在发给机械臂的指令（关节角 度, 夹爪合上?），录成 action
        self.recorder, self.rec_cmd, self.rec_result = None, None, None
        # 最近这一次执行 / 手动录的结果，网页左上角一直显示到下一次开始（2026-11-09 用户：停了以后分不清录没录）：
        # None / {state: running | off | saved | not_saved | discarded, name, idx, thumb, why}
        self.rec_run = None
        try:
            self.rec_cfg = {**REC_DEFAULT, **json.load(open(REC_CFG_PATH, encoding="utf-8"))}
        except (OSError, ValueError):
            self.rec_cfg = dict(REC_DEFAULT)
        self.policy_runner = None            # π0.5 模型控制真机（policy_run.PolicyRunner），用到时才建
        threading.Thread(target=self._follow_loop, daemon=True).start()

    def policy(self):
        if self.policy_runner is None:
            import policy_run
            self.policy_runner = policy_run.PolicyRunner(self)
        return self.policy_runner

    # ---- 手臂摆法（同一个目标位姿的几种逆解） ----
    def all_solutions(self, flange, rot, q_ref):
        """法兰位姿的全部逆解（SDK 解析全解），关节角按 ±360° 挪进限位、离 q_ref 最近；去掉重复的。"""
        pose = [float(v) for v in [*flange, *rot.as_euler("xyz")]]
        q_ref = np.array(q_ref, float)
        with self.lock:
            res = self.robot.arm.rm_algo_inverse_kinematics_all(self.robot.ik_params(list(map(float, q_ref)), pose, 1))
        out = []
        for i in range(res.num if res.result == 0 else 0):
            q = np.array([res.q_solve[i][j] for j in range(6)], float)
            for j in range(6):
                cands = [q[j] + k * 360 for k in (-2, -1, 0, 1, 2)
                         if self.robot.q_min[j] <= q[j] + k * 360 <= self.robot.q_max[j]]
                if cands:
                    q[j] = min(cands, key=lambda v: abs(v - q_ref[j]))
            if not any(np.max(np.abs(q - o)) < 1.0 for o in out):
                out.append(q)
        return out

    def ik_branches(self, pos, quat, q_ref, ext=None):
        """同一个抓取中心位姿，6 轴机械臂最多有 8 种手臂摆法（肩、肘、腕各两种）。全列出来：
        关节角、能不能用（限位 / 自碰撞 / 离桌面 / 离底座）、一句话描述、和现在相比关节最多差几度。"""
        rot = R.from_quat(quat)
        flange = np.asarray(pos, float) - rot.apply([0.0, 0.0, tp.TOOL_LENGTH])
        pose = [float(v) for v in [*flange, *rot.as_euler("xyz")]]
        q_ref = np.array(q_ref, float)
        with self.lock:
            res = self.robot.arm.rm_algo_inverse_kinematics_all(self.robot.ik_params(list(map(float, q_ref)), pose, 1))
        out = []
        for i in range(res.num if res.result == 0 else 0):
            q = np.array([res.q_solve[i][j] for j in range(6)], float)
            # 关节角加减 360° 后都是同一个姿态：挑落在限位内、离现在最近的那个
            for j in range(6):
                cands = [q[j] + k * 360 for k in (-2, -1, 0, 1, 2)
                         if self.robot.q_min[j] <= q[j] + k * 360 <= self.robot.q_max[j]]
                if cands:
                    q[j] = min(cands, key=lambda v: abs(v - q_ref[j]))
            if any(np.max(np.abs(q - o["q_arr"])) < 1.0 for o in out):
                continue
            bad = self.check_joints(q, None, ext)     # ext：指尖多伸出多少；不给就按真实夹爪现在的开口（网页列表用）
            if bad is None and near_base(flange):
                bad = "法兰离底座太近"
            out.append({"q_arr": q, "q": list(map(float, q)), "ok": bad is None, "reason": bad or "",
                        "label": self.branch_label(q), "dist": float(np.max(np.abs(q - q_ref)))})
        # 固定顺序（不按“离现在多远”排，不然一点切换，选中的就跳到最上面）：正面在前、手肘朝上在前、手腕不翻在前
        out.sort(key=lambda o: ("反手" in o["label"], "朝下" in o["label"], "翻转" in o["label"], round(o["q"][0])))
        for o in out:
            o.pop("q_arr")
        return out

    def branch_label(self, q):
        """用人话描述手臂摆法：手肘在肩—腕连线的上方还是下方；手臂是正面伸过去还是从背后绕过来；手腕翻没翻。"""
        pts = [p for _, p in tp.arm_points(q, self.robot.dh)]
        s, e, w = pts[5], pts[9], pts[19]            # 肩、肘、腕
        v = w - s
        d = (e - s) - np.dot(e - s, v) / max(np.dot(v, v), 1e-9) * v
        elbow = "手肘朝上" if d[2] >= 0 else "手肘朝下"
        az = np.degrees(np.arctan2(w[1], w[0]))      # 手腕在底座的哪个方向；J1 大致朝着它就是正面伸过去
        front = "正面" if abs((q[0] - az + 180) % 360 - 180) < 90 else "反手（从背后绕过来）"
        wrist = "手腕不翻" if q[4] < 0 else "手腕翻转"
        return f"{front} · {elbow} · {wrist}"

    # ---- 关节模式 ----
    def fk_check(self, q, ext=None):
        """关节角 → (抓取中心, 四元数, 问题或 None)：网页上逐个关节调时用，不经过逆解，手臂摆法就是你调出来的那样。
        ext：仿真夹爪的指尖多伸出多少（网页发来；不给按真实夹爪）。"""
        q = np.clip(np.array(q, float), self.robot.q_min, self.robot.q_max)
        center, rot = self.fk_center(q)
        return center, rot.as_quat(), self.check_joints(q, None, ext), q

    # ---- 自动生成抓放 ----
    AUTO_TAGS = ["预抓取", "抓取", "抬起", "预放置", "放下", "撤出"]

    def hover_turn(self, pick, place):
        """“先到方块上方再转夹爪”用的角度（2026-11-06：示范里 J6 在关节运动途中就转好了，模型没学过看着腕部画面对齐方块，
        真机上夹的总是方块的角）。只管夹爪大致竖直朝下的抓取（倾斜 < HOVER_TILT_DEG），否则返回 None。
        悬停方向＝初始位姿的夹爪方向跟着底座转到抓取点那边（相当于 J6 不动、只有 J1 转）；抓取朝向在 用户给的朝向 + k×90°
        （方块转 90° 一样）里挑离悬停方向最近的（最多转 45°）。放置点原样返回：放的时候夹爪什么方向由 place_orient 按最终的
        抓取方向定（2026-11-09 前这里给放置点也转同样的 k×90°，放置点沿用上一条时会多拧 90°，第 5 条搬运时 J6 转了 134°）。
        返回 (新抓取点, 放置点（没改）, 悬停用的“抓取点”（位置同抓取点、朝向是悬停方向）, 悬停→对齐要转的角度 弧度)。"""
        rp = R.from_quat(pick["quat"])
        if rp.apply([0, 0, 1.0])[2] > -np.cos(np.radians(HOVER_TILT_DEG)):
            return None
        yaw = lambda r: float(np.arctan2(r.apply([1.0, 0, 0])[1], r.apply([1.0, 0, 0])[0]))
        rot_h, c_h = tp.home_target()
        neutral = yaw(rot_h) + float(np.arctan2(pick["pos"][1], pick["pos"][0]) - np.arctan2(c_h[1], c_h[0]))
        k = int(np.round((neutral - yaw(rp)) / (np.pi / 2)))
        rz = R.from_rotvec([0, 0, k * np.pi / 2])
        pick = dict(pick)
        if k:
            pick["quat"] = list(map(float, (rz * R.from_quat(pick["quat"])).as_quat()))
            if pick.get("q"):                       # 绕竖直轴转 +θ 大约就是 J6 −θ（夹爪竖直朝下时），手臂摆法不变
                pick["q"] = list(map(float, pick["q"]))
                pick["q"][5] -= float(np.degrees(k * np.pi / 2))
        rp = R.from_quat(pick["quat"])
        turn = float((neutral - yaw(rp) + np.pi) % (2 * np.pi) - np.pi)     # 悬停方向 − 对齐方向，|turn| ≤ 45°
        hover = dict(pick, quat=list(map(float, (R.from_rotvec([0, 0, turn]) * rp).as_quat())))
        if hover.get("q"):
            hover["q"] = list(map(float, hover["q"]))
            hover["q"][5] -= float(np.degrees(turn))
        return pick, place, hover, -turn

    def place_orient(self, pick, place, align=False):
        """放的时候夹爪什么方向（2026-11-09，用户：第 5 条放进盘子途中手腕转了很多）。
        放置点存的方向是设放置点那一刻按当时的抓取点算的；盘子不动、只给新方块重设抓取点时，这个方向就过时了，
        搬运途中手腕要拧过去（11-09 那 6 条里有 3 条搬运时 J6 转了 30°/75°/134°）。所以生成轨迹时按最终的抓取方向重新定：
          align=False（默认，放盘子）：夹爪方向只跟着底座转＝抓和放时 J6 一样，放置点存的方向不管；
          align=True（叠方块、放进盒子要对齐）：用放置点存的方向，但方块转 90° 一样，在 +k×90° 里挑离“跟着底座转”最近的，
          搬运途中手腕最多转 45°。
        抓取点或放置点夹爪不是大致竖直朝下（倾斜 ≥ HOVER_TILT_DEG）时原样不动。
        返回 (放置点, "follow" / "align" / "keep", 改了多少度)。"""
        rp, rl = R.from_quat(pick["quat"]), R.from_quat(place["quat"])
        c = -np.cos(np.radians(HOVER_TILT_DEG))
        if rp.apply([0, 0, 1.0])[2] > c or rl.apply([0, 0, 1.0])[2] > c:
            return place, "keep", 0.0
        yaw = lambda r: float(np.arctan2(r.apply([1.0, 0, 0])[1], r.apply([1.0, 0, 0])[0]))
        az = lambda p: float(np.arctan2(p[1], p[0]))
        follow = yaw(rp) + az(place["pos"]) - az(pick["pos"])
        d = float((follow - yaw(rl) + np.pi) % (2 * np.pi) - np.pi)
        rot = float(np.round(d / (np.pi / 2)) * np.pi / 2) if align else d
        place = dict(place, quat=list(map(float, (R.from_rotvec([0, 0, rot]) * rl).as_quat())))
        if place.get("q"):
            place["q"] = list(map(float, place["q"]))
            place["q"][5] -= float(np.degrees(rot))
        return place, ("align" if align else "follow"), float(np.degrees(rot))

    def goto_frames(self, kf):
        """真机移到一个目标（空格 / M、真机移到这帧）怎么走，和录的示范一致：夹爪大致竖直朝下时，
        先关节运动到目标正上方（低处的目标抬高 GOTO_HOVER_UP），路上手腕 J6 不拧、夹爪方向只跟着底座转；
        到了原地转到目标的方向、停一下，再竖直下到目标。已经在目标正上方附近就在现在的高度转；
        离目标很近、方向也差不多（小修正）或夹爪不竖直就直接过去。
        （2026-11-06 以前直接用关节运动斜着过去，手腕一路拧，比如从初始位姿到抓取点途中 J6 转了 53°）
        返回 (关键帧列表（最后一帧就是 kf）, 说明)。"""
        st = self.robot.read_state()
        if st is None or kf.get("pos") is None or kf.get("quat") is None:
            return [kf], ""
        pt, rt = np.array(kf["pos"], float), R.from_quat(kf["quat"])
        if rt.apply([0, 0, 1.0])[2] > -np.cos(np.radians(HOVER_TILT_DEG)):
            return [kf], ""
        q_act = np.array(st[1], float)
        c0, r0 = self.fk_center(q_act)
        yaw = lambda r: float(np.arctan2(r.apply([1.0, 0, 0])[1], r.apply([1.0, 0, 0])[0]))
        az = lambda p: float(np.arctan2(p[1], p[0]))
        neutral = yaw(r0) + az(pt) - az(c0)                                 # 路上 J6 不动：方向只跟着 J1 转
        turn = float((yaw(rt) - neutral + np.pi) % (2 * np.pi) - np.pi)     # 到了以后要转多少（目标 − 路上的方向）
        near = float(np.hypot(*(pt[:2] - c0[:2]))) < GOTO_NEAR_XY
        if near and abs(np.degrees(turn)) < GOTO_NEAR_YAW:
            return [kf], ""
        if near:
            if c0[2] < pt[2] - 0.005:
                return [kf], ""
            hz = float(c0[2])                                                # 已经在正上方：就在这个高度转
        else:
            hz = float(pt[2] + (GOTO_HOVER_UP if pt[2] < GOTO_LOW_Z else 0.0))
        if kf.get("q") is not None:
            q_t = np.array(kf["q"], float)
        else:
            q_t, why = self.first_joints(pt, rt, q_act)
            if q_t is None:
                return [kf], ""
            q_t = np.array(q_t, float)
        p_h = [float(pt[0]), float(pt[1]), hz]
        r_h1 = R.from_rotvec([0, 0, -turn]) * rt                             # 目标方向往回转 turn＝路上的方向
        seed = q_t.copy()
        seed[5] += np.degrees(turn)                                          # 绕竖直轴转 +θ 大约就是 J6 −θ
        hard, margin, where, q_h1 = self.solve_center(p_h, list(r_h1.as_quat()), list(seed))
        if hard is not None or margin < 0:
            return [kf], f"目标正上方到不了（{hard or where}），直接过去"
        frames = [{"pos": p_h, "quat": list(map(float, r_h1.as_quat())), "q": list(map(float, q_h1)), "tag": "目标上方"}]
        if hz > pt[2] + 0.005:
            hard, margin, where, q_h2 = self.solve_center(p_h, list(rt.as_quat()), list(q_t))
            if hard is not None or margin < 0:
                return [kf], f"目标正上方到不了（{hard or where}），直接过去"
            frames.append({"pos": p_h, "quat": list(map(float, rt.as_quat())), "q": list(map(float, q_h2)),
                           "tag": "目标上方", "stop": True})
        note = ("先到正上方" if hz > pt[2] + 0.005 else "先过去") + (f"、原地转 {abs(np.degrees(turn)):.0f}°" if abs(np.degrees(turn)) >= 1 else "") \
            + ("、再竖直下去" if hz > pt[2] + 0.005 else "")
        return frames + [kf], note

    def auto_frames(self, pick, place, d, lift, home, width=None, start_home=False, ref_q=None, hover_align=False,
                    place_align=False, stack=False, align_height=STACK_ALIGN_HEIGHT):
        """按两点和参数摆出关键帧（还没验证）。每帧都给出关节角（按抓取点 / 放置点的手臂摆法就近解），
        解不出来的帧返回 (None, 说明)。width：物体宽度（米，夹住的地方）：合上夹爪那一帧和夹着走的几帧，
        离桌面按手指停在物体两侧时指尖伸出多少算；None 按全合算。home：最后回初始位姿；start_home：从初始位姿出发。
        ref_q：抓取点没带关节角时，挑离这个姿态最近的安全摆法（接在上一个动作后面时是上一个动作结束的姿态），不给就按初始位姿。
        hover_align：先到方块正上方再转夹爪（见 hover_turn）：预抓取前加一帧“悬停”（同一位置、夹爪还是“只跟着底座转”的方向），
        预抓取在原地转到对齐方块的角度后停一下再竖直下降；抓取朝向在四个等价角度（方块转 90° 一样）里挑离悬停方向最近的。
        place_align：放的时候夹爪方向（见 place_orient）：False 跟着底座转（抓和放 J6 一样），True 按放置点的方向对齐（取 90° 对称里最近的）。"""
        hold = 0.0 if width is None else tp.finger_ext(width)   # 没勾“按物体宽度检查”：抓取点以用户摆的为准，不加合上时指尖的伸出量
        hov = self.hover_turn(pick, place) if hover_align else None
        hover_base, turn = None, 0.0
        if hov is not None:
            pick, _, hover_base, turn = hov
        place0 = place
        place, pmode, prot = self.place_orient(pick, place0, place_align or stack)

        def build(place):
            return self._build_frames(pick, place, d, lift, hold, width, ref_q, hov is not None, hover_base, turn, stack, align_height)

        frames, why = build(place)
        if frames is None:
            return None, why
        if pmode == "follow":
            # 跟着底座转只是近似（J1 转的角度和方位角差不完全一样，小臂有 5.9 cm 侧偏）：按解出来的 J6 再修一两次，让放和抓的 J6 一样
            jg = next(f["q"][5] for f in frames if f["tag"] == "抓取")
            for _ in range(2):
                dj = next(f["q"][5] for f in frames if f["tag"] == "放下") - jg
                if abs(dj) < 0.3:
                    break
                pl = dict(place, quat=list(map(float, (R.from_rotvec([0, 0, np.radians(dj)]) * R.from_quat(place["quat"])).as_quat())))
                if pl.get("q"):
                    pl["q"] = list(map(float, pl["q"]))
                    pl["q"][5] = float(jg)
                f2, _ = build(pl)
                if f2 is None:
                    break
                frames, place, prot = f2, pl, prot + float(dj)
        if pmode != "keep" and abs((prot + 180) % 360 - 180) > 3:
            nxt = next(f for f in frames if f["tag"] == "放下")
            nxt["place_fix"] = {"mode": pmode, "deg": round(float((prot + 180) % 360 - 180), 1)}
        rot, pos = tp.home_target()
        home_frame = {"pos": list(map(float, pos)), "quat": list(map(float, rot.as_quat())), "q": list(map(float, tp.HOME_Q)),
                      "gripper": "", "pause": 0.0, "joint": True}
        if home:
            frames.append({**home_frame, "tag": "回初始"})
        if start_home:
            # 从初始位姿出发：每条示范起点一样（录数据时模型看到的开头都一样）；初始位姿 → 预抓取走关节运动（标 joint），
            # 路径形状不重要、离得也远，不试直线
            frames[0]["joint"] = True
            frames.insert(0, {**home_frame, "tag": "初始"})
        return frames, None

    def _build_frames(self, pick, place, d, lift, hold, width, ref_q, hovering, hover_base, turn, stack, align_height):
        """auto_frames 用：按抓取点、放置点摆出预抓取…撤出几帧并逐帧解逆解。解不出来返回 (None, 说明)。"""
        up = np.array([0.0, 0.0, 1.0])
        specs = [("预抓取", pick, ("axis", -d), "open", 0.0, 0.0), ("抓取", pick, None, "close", GRIP_CLOSE_PAUSE, hold),
                 ("抬起", pick, ("up", lift), "", 0.0, hold), ("预放置", place, ("up", lift), "", 0.0, hold),
                 ("放下", place, None, "open", GRIP_OPEN_PAUSE, hold), ("撤出", place, ("axis", -d), "", 0.0, 0.0)]
        if stack:
            carry, _, _ = self.place_orient(pick, place)
            specs[3:] = [
                ("底块上方", carry, ("up", align_height), "", STACK_ALIGN_PAUSE, hold),
                ("对齐底块", place, ("up", align_height), "", STACK_ALIGN_PAUSE, hold),
                ("放下", place, None, "open", GRIP_OPEN_PAUSE, hold),
                ("撤出", place, ("up", lift), "", 0.0, 0.0),
            ]
        if hovering:
            # 悬停（张开、夹爪方向只跟着底座转）→ 预抓取（原地转 turn 度对齐方块、停一下）→ 竖直下降抓
            specs = [("悬停", hover_base, ("axis", -d), "open", 0.0, 0.0), ("预抓取", pick, ("axis", -d), "", 0.0, 0.0)] + specs[1:]
        frames, last_q = [], None
        for tag, base, off, grip, pause, ext in specs:
            hold_wrist = stack and tag in ("抬起", "底块上方")
            rot = R.from_quat(base["quat"])
            pos = np.array(base["pos"], float)
            if off:
                pos = pos + (rot.apply([0, 0, 1.0]) if off[0] == "axis" else up) * off[1]
            seed = base.get("q") or (last_q if last_q is not None else None)
            if seed is None:
                # 抓取点没带关节角（没在网页上选过摆法）：在这个点安全的几种摆法里挑离初始位姿（或上一个动作结束时）最近的。
                # 不能从机械臂现在的姿态去连续跟——离得远时会报“要换摆法”，出发段本来就是关节运动，不需要连续
                ok = [b for b in self.ik_branches(base["pos"], base["quat"], tp.HOME_Q if ref_q is None else ref_q, ext)
                      if b["ok"]]
                seed = min(ok, key=lambda b: b["dist"])["q"] if ok else None
            hard, margin, where, q = self.solve_center(list(pos), list(rot.as_quat()), seed, ext)
            if hard is not None or margin < 0:
                if hard is None and grip == "close":
                    where = (f"合上夹爪时指尖会伸出 {ext * 100:.1f} cm（按物体宽度 {width * 100:.0f} cm 算），{where}，"
                             f"抓取点再抬高 {-margin * 100:.1f} cm" if width is not None else
                             f"{where}（没填物体宽度，按空夹全合算）")
                return None, f"{tag}帧：{hard or where}"
            if hold_wrist:
                # 同轴的 J6 只改变夹爪的滚转。固定住抓取时的 J6，再校验实际位姿。
                q, rot, bad = self.lock_wrist(pos, q, last_q[5], ext)
                if bad:
                    return None, f"{tag}帧：{bad}"
            last_q = list(map(float, q))
            frames.append({"pos": list(map(float, pos)), "quat": list(map(float, rot.as_quat())), "q": last_q,
                           "gripper": grip, "pause": pause, "tag": tag})
            if stack and tag in ("抬起", "底块上方", "对齐底块", "放下", "撤出"):
                frames[-1].update(stop=True, stack_linear=True)
                if hold_wrist:
                    frames[-1]["hold_wrist"] = True
                if tag in ("底块上方", "对齐底块"):
                    frames[-1]["stack_align_wait"] = True
                if tag == "对齐底块":
                    frames[-1]["stack_align"] = True
                if tag == "放下":
                    frames[-1]["stack_release"] = True
            if hovering and tag == "预抓取":
                frames[-1]["stop"] = True                       # 原地转完停一下再下降（转的那一下腕部相机看得清方块）
                frames[-1]["turn_deg"] = round(float(np.degrees(turn)), 1)
            if grip == "close" and width is not None:            # 有 grip_width 的合夹爪帧，规划时才按它加指尖伸出量
                frames[-1]["grip_width"] = float(width)
        return frames, None

    def autogen(self, pick, place, d, lift, home, speed, ang_speed, human, seed, width=None, start_home=False, precise=None,
                prev=None, base=0, hover_align=False, place_align=False, stack=False, align_height=STACK_ALIGN_HEIGHT):
        """两点自动生成抓放，并且验证：每帧先单独看能不能到，再规划整条轨迹；不行就自动调整
        （伸进距离缩短、抬起高度改高或改低），找到第一个能走通的方案。放置点没指定手臂摆法时，沿用抓取那一侧的摆法就近解。
        返回 (关键帧, 规划结果, 说明列表)；都不行时关键帧是问题最少的那个方案。
        prev：接在上一个动作后面时，上一个动作的最后一帧（撤出）。这时只验证“上一帧 → 这个动作”这一段（夹爪按张开算、
        不算从实际位置出发的那段），调整也只改这个动作自己的帧；base 是这个动作的第 1 帧在整条里是第几帧（报问题时换成整条的帧号）。"""
        ref_q = prev["q"] if prev is not None else None

        def renum(text):
            # 只验证一段时，plan 里的“第 N 帧”是从上一个动作的最后一帧算起的，换成整条关键帧列表里的帧号
            if prev is None or not text:
                return text
            return re.sub(r"(第 |→)(\d+)", lambda g: g.group(1) + str(int(g.group(2)) + base - 1), text)

        def run(frames):
            if prev is None:
                return self.plan(frames, speed, ang_speed, human, seed, precise)
            frames[0]["stop"] = True                 # 上一个动作撤出后停一下，再像单独一个动作那样从预抓取开始
            p = self.plan([dict(prev, stop=True)] + frames, speed, ang_speed, human, seed, precise, grip0=False, with_approach=False)
            for x in p.get("problems", []):
                x["reason"] = renum(x["reason"])
            return p
        tries, best, notes = [], None, []
        for dd in sorted({d, max(0.03, d * 0.75), max(0.03, d * 0.5)}, reverse=True):
            for ll in [lift, lift + 0.05, max(0.04, lift - 0.04)]:
                tries.append((dd, ll))
        for dd, ll in tries:
            frames, why = self.auto_frames(pick, place, dd, ll, home, width, start_home, ref_q, hover_align, place_align, stack, align_height)
            if frames is None:
                notes.append(f"伸进 {dd * 100:.0f} cm、抬起 {ll * 100:.0f} cm：{why}")
                continue
            p = run(frames)
            bad = sum(x["end"] - x["start"] + 1 for x in p.get("problems", []))
            if best is None or bad < best[0]:
                best = (bad, frames, p, dd, ll)
            if p["ok"]:
                break
            notes.append(f"伸进 {dd * 100:.0f} cm、抬起 {ll * 100:.0f} cm：轨迹中途有问题（{p['problems'][0]['reason']}）")
        if best is None or not best[2]["ok"]:
            # 距离和高度怎么调都不行（包括关键帧本身就摆不出来）：再试抓取点 / 放置点换别的手臂摆法（只试安全的几种）
            def safe_qs(pose):
                # 按“抓的时候夹着物体”算指尖（合上夹爪），不看真实夹爪现在开着还是合着
                # （2026-11-02 真夹爪合着时这里按空夹全合算，列表全红，报“换哪种摆法都不行”）
                return [(b["q"], b["label"]) for b in self.ik_branches(pose["pos"], pose["quat"],
                                                                         pose.get("q") or ref_q or tp.HOME_Q,
                                                                         0.0 if width is None else tp.finger_ext(width)) if b["ok"]][:3]
            done, picks, places = False, safe_qs(pick), safe_qs(place)
            if not picks:
                notes.append("抓取点换哪种手臂摆法都不行（“手臂摆法”列表里全是红的）")
            if not places:
                notes.append("放置点换哪种手臂摆法都不行（“手臂摆法”列表里全是红的）")
            for qp, lp in picks:
                for ql, ll_ in places:
                    for dd, ll in tries[:3]:
                        frames, why = self.auto_frames({**pick, "q": qp}, {**place, "q": ql}, dd, ll, home, width, start_home,
                                                       ref_q, hover_align, place_align, stack, align_height)
                        if frames is None:
                            if dd == tries[0][0] and ll == tries[0][1]:     # 每种摆法组合只记一条，别刷屏
                                notes.append(f"抓取点换成“{lp}”、放置点“{ll_}”：{why}")
                            continue
                        p = run(frames)
                        bad = sum(x["end"] - x["start"] + 1 for x in p.get("problems", []))
                        if best is None or bad < best[0]:
                            best = (bad, frames, p, dd, ll)
                        if p["ok"]:
                            notes.append(f"已自动换了一种手臂摆法（抓取点“{lp}”、放置点“{ll_}”）")
                            done = True
                            break
                        if dd == tries[0][0] and ll == tries[0][1]:
                            notes.append(f"抓取点换成“{lp}”、放置点“{ll_}”：轨迹中途有问题（{p['problems'][0]['reason']}）")
                    if done:
                        break
                if done:
                    break
        if best is None:
            return None, None, notes
        _, frames, p, dd, ll = best
        if p["ok"] and (abs(dd - d) > 1e-6 or abs(ll - lift) > 1e-6):
            notes.append(f"已自动调整：伸进 / 撤出 {dd * 100:.0f} cm，抬起 {ll * 100:.0f} cm")
        fix = next((f.pop("place_fix") for f in frames if "place_fix" in f), None)
        if fix:
            notes.append(f"放置时夹爪方向比放置点存的转了 {fix['deg']:+.0f}°：" +
                         ("跟着手臂转、和抓的时候 J6 一样，搬运途中手腕不拧（要对齐放置点的方向就打开“放的时候对齐放置点的方向”）"
                          if fix["mode"] == "follow" else "按放置点的方向对齐，方块转 90° 一样，挑了搬运途中手腕转得最少的那个"))
        return frames, p, notes

    def autogen_multi(self, actions, d, lift, home, speed, ang_speed, human, seed, start_home=False, precise=None,
                      start_noise=None, exec_noise=None, hover_align=False, place_align=False, stack=False, align_height=STACK_ALIGN_HEIGHT):
        """几个抓放动作按顺序接成一条（比如“红块和蓝块都放进盒子”）。actions：[{pick, place, width}, ...]。
        每个动作单独生成、单独验证、单独调整（伸进距离 / 抬起高度 / 手臂摆法只改它自己的帧，不牵连别的动作）。
        和初始位姿的关系：只有第 1 个动作从初始位姿出发，只有最后一个动作做完回初始位姿；中间不回初始，
        上一个动作撤出后停一下，直接走到下一个动作的预抓取，再像单独一个动作那样伸进去抓（每个动作的抓取过程都一样）。
        后面动作的抓取点没带关节角时，挑离上一个动作结束姿态最近的摆法。最后整条再规划一遍给预览和执行用。"""
        n, frames_all, notes = len(actions), [], []
        if stack and n != 1:
            return None, None, ["叠方块一次生成一组抓放，叠完后刷新点云再设下一块"]
        if stack and not 0.05 <= align_height <= 0.20:
            return None, None, ["放置对齐高度应为 5–20 cm"]
        for i, a in enumerate(actions):
            prev = frames_all[-1] if frames_all else None
            w = a.get("width")
            pick, place = a["pick"], a.get("place")
            if stack:
                bottom, manual = a.get("stack_base"), a.get("stack_place")
                if not (bottom or manual):
                    return None, None, ["请先设为放置点，或在点云上右键选择叠放底块"]
                vertical = -np.cos(np.radians(2.0))
                if R.from_quat(pick["quat"]).apply([0, 0, 1])[2] > vertical:
                    return None, None, ["叠方块需要竖直抓取，请重新设置上块的抓取点"]
                if manual:
                    place = manual
                    if R.from_quat(place["quat"]).apply([0, 0, 1])[2] > vertical:
                        return None, None, ["叠方块需要竖直下放，请让夹爪竖直朝下再设置放置点"]
                else:
                    x, y, top, yaw = (float(bottom[k]) for k in ("x", "y", "top", "yaw"))
                    hold_z = pick["pos"][2] - tp.TABLE_Z
                    if not np.all(np.isfinite([x, y, top, yaw, hold_z])) or top <= tp.TABLE_Z or hold_z <= 0:
                        return None, None, ["底块或抓取点的高度无效，请刷新点云后重选"]
                    rot = R.from_euler("Z", yaw) * R.from_euler("X", np.pi)
                    place = {"pos": [x, y, top + hold_z + STACK_RELEASE_GAP], "quat": rot.as_quat().tolist()}
                if not np.all(np.isfinite(place["pos"])) or place["pos"][2] <= pick["pos"][2]:
                    return None, None, ["叠放时，松爪位置需要高于抓取点"]
                if np.linalg.norm(np.asarray(pick["pos"][:2]) - place["pos"][:2]) < 0.015:
                    return None, None, ["抓取点和放置点太近，请确认选的是两块不同的方块"]
            frames, p, nt = self.autogen(pick, place, d, lift, home and i == n - 1, speed, ang_speed, human, seed,
                                         float(w) if w is not None else None, start_home and i == 0, precise,
                                         prev=prev, base=len(frames_all), hover_align=hover_align, place_align=place_align, stack=stack,
                                         align_height=align_height)
            notes += [(f"动作 {i + 1}：" if n > 1 else "") + t for t in nt]
            if frames is None:
                if n > 1:
                    notes.append(f"动作 {i + 1} 有关键帧摆不出来（上面是每种调整的原因），整条都没生成")
                return None, None, notes
            if n > 1:
                for f in frames:
                    if f.get("tag") not in ("初始", "回初始"):
                        f["tag"] = f"{i + 1}·{f['tag']}"
                if p is not None and not p["ok"]:
                    notes.append(f"动作 {i + 1} 怎么调整都有问题，用的是问题最少的方案")
            if prev is not None:
                prev["stop"] = True
            frames_all += frames
        p = self.plan(frames_all, speed, ang_speed, human, seed, precise, start_noise, exec_noise=exec_noise)
        return frames_all, p, notes

    # ---- 抓取姿态：夹爪朝外摆正 / 按物体自动生成 / 保持方向换摆法 ----
    ALIGN_ENOUGH = 45.0      # 手腕离奇异（J5 离 0° 和 ±180°）超过这么多度就算够了，够了的方向里挑最靠前（转得最少 / 最合适）的
    MAX_GRASP_W = 0.08       # 夹得住的最大宽度（两指最大间隙 87.5 mm，留点余量）
    # 点云自动生成竖直夹时，合上后指尖离物体底下那层（桌面/板子）至少多高。原来和安全检查一样是 FINGER_CLEARANCE=3 mm，
    # 2026-11-09 用户补了 2~3 mm 厚的板子，点云量出来的板面又偏低几毫米（11-06 量过工作区面 −6~+4 mm），
    # 生成的抓取点指尖几乎贴着板子，按用户要求再抬 3 mm（3.3 cm 方块抓取中心约 53 → 56 mm）。只管自动生成，安全检查照旧 3 mm
    AUTO_TIP_CLEARANCE = 0.006
    NEAR_BRANCH_DEG = 100.0  # 先只试和当前摆法差不多的（关节最多差这么多度），都不行再放开去换摆法

    def grasp_frames_ok(self, c, r, seed, d, lift, width, near_only):
        """抓取中心 c、姿态 r 时，预抓取（沿夹爪轴退 d）、抓取、抬起（往上 lift）三帧都能到吗（自动生成抓放也是这三帧打头）。
        width：物体宽度（抓取、抬起两帧离桌面按合上时指尖伸出多少算）。near_only：只用和 seed 差不多的摆法（关节最多差 NEAR_BRANCH_DEG），
        不整条手臂换一种摆法（2026-11-02 以前是“只用正面摆法”：初始位姿往左转后，远一点的地方只有“反手”够得着，
        优先正面会让手臂绕底座转 180° 换摆法）。
        能到返回 ((手腕离奇异最少多少度, 抓取帧关节角, 摆法), None)，不能到返回 (None, 原因)。"""
        hold = 0.0 if width is None else tp.finger_ext(width)   # None：以用户摆的为准，不加合上时的伸出量（0 表示按空夹全合）
        ax = r.apply([0, 0, 1.0])
        up = np.array([0.0, 0.0, 1.0])
        branches = self.ik_branches(list(c), list(r.as_quat()), seed, hold)
        if not branches:
            return None, "够不到（逆解失败）"
        branches = [b for b in branches if not near_only or b["dist"] <= self.NEAR_BRANCH_DEG]
        if not branches:
            return None, "和现在的手臂摆法差太多"
        why = None
        for b in sorted(branches, key=lambda b: (not b["ok"], b["dist"])):
            if not b["ok"]:
                why = why or f"抓取帧：{b['reason']}"
                continue
            q_g = np.array(b["q"], float)
            qs = []
            for name, p, ext in (("抓取", c, hold), ("预抓取", c - ax * d, 0.0), ("抬起", c + up * lift, hold)):
                hard, margin, where, q = self.solve_center(list(p), list(r.as_quat()), q_g, ext)
                if hard is not None or margin < 0:
                    why = why or f"{name}帧：{hard or where}"
                    break
                qs.append(q)
            else:
                wrist = min(min(abs(q[4]) % 360, 360 - abs(q[4]) % 360, abs(180 - abs(q[4]) % 360)) for q in qs)
                elbow = min(abs(q[2]) for q in qs)
                if wrist < tp.SINGULAR_MIN_DEG or elbow < tp.SINGULAR_MIN_DEG:
                    why = why or ("手腕太接近奇异姿态（J5 接近 0° 或 ±180°）" if wrist < tp.SINGULAR_MIN_DEG else "手臂快伸直了（J3 接近 0°）")
                    continue
                return (wrist, q_g, b["label"]), None
        return None, why

    def pick_best(self, cands, seed, d, lift):
        """cands：[(说明, 抓取中心, 姿态, 物体宽度)]，已按偏好排好。先只用和当前摆法差不多的，找不到再放开。
        手腕离奇异 ≥ ALIGN_ENOUGH 的里面挑最靠前的，都不够就挑离奇异最远的。
        返回 (选中的 (说明, c, r, w, wrist, q, 摆法) 或 None, 放开摆法以后最常见的失败原因)。"""
        whys = {}
        for near_only in (True, False):
            found = []
            whys = {}
            for note, c, r, w in cands:
                res, why = self.grasp_frames_ok(c, r, seed, d, lift, w, near_only)
                if res is None:
                    whys[why] = whys.get(why, 0) + 1
                    continue
                found.append((note, c, r, w, *res))
                if res[0] >= self.ALIGN_ENOUGH:
                    return found[-1], None
            if found:
                return max(found, key=lambda t: t[4]), None
        return None, max(whys, key=whys.get) if whys else None

    def align_out(self, pos, quat, seed, d, lift, width):
        """夹爪朝外摆正：抓取中心位置和夹爪倾斜角不变，只让夹爪绕竖直轴转，找一个方向，使预抓取、抓取、抬起三帧都能到、
        手腕离奇异够远（ECO65 和 UR 一样，J5 接近 0° 或 ±180° 都是手腕奇异；夹爪水平时侧着指就会到这里）。
        离奇异超过 ALIGN_ENOUGH 的方向里挑转得最少的，都不够就挑离奇异最远的。返回 (结果 dict 或 None, 说明)。"""
        c, r0 = np.asarray(pos, float), R.from_quat(quat)
        tilt = float(np.degrees(np.arccos(np.clip(-r0.apply([0, 0, 1.0])[2], -1, 1))))
        if tilt < 15:
            return None, "夹爪接近竖直朝下：朝向不影响手腕奇异，不用摆正"
        seed = self._seed(seed)
        cands = [(y, c, R.from_euler("z", y, degrees=True) * r0, width) for y in sorted(range(-180, 180, 5), key=lambda y: (abs(y), y))]
        best, why = self.pick_best(cands, seed, d, lift)
        if best is None:
            return None, f"这个位置夹爪保持这个倾斜角，朝哪个方向都摆不出能抓的姿态（比如：{why}）：换个位置或倾斜角再试"
        yaw, c, r, _, wrist, q_g, label = best
        if yaw == 0:
            text = f"现在的方向就行，不用转（手腕离奇异 {wrist:.0f}°，预抓取、抓取、抬起都能到）"
        else:
            text = (f"夹爪绕竖直轴转了 {abs(yaw)}°（从上往下看{'逆' if yaw > 0 else '顺'}时针），"
                    f"手腕离奇异 {wrist:.0f}°，预抓取、抓取、抬起都能到" + ("" if wrist >= self.ALIGN_ENOUGH else "（这个位置最多只能离这么远）"))
        text += f"；手臂摆法“{label}”"
        return {"pos": list(map(float, c)), "quat": list(map(float, r.as_quat())), "q": list(map(float, q_g)), "yaw": yaw,
                "wrist": wrist}, text

    def _seed(self, seed):
        if seed is not None:
            return list(map(float, seed))
        st = self.robot.read_state()
        return list(map(float, st[1])) if st else list(map(float, tp.HOME_Q))

    def grasp_auto(self, center, mode, box, d, lift, seed, from_box):
        """按抓取中心自动生成最佳抓取姿态。mode："vertical" 竖直向下夹 / "horizontal" 水平夹。
        box：点云框出的物体（size 长宽高、yaw 长边方向、top、bottom）；没有框就按 4×4×5 cm 的小物体算。
        竖直夹：两指沿物体窄的那边合上（两种手指朝向都试）；水平夹：夹爪水平、手指水平开合，从外面伸过去，
        方向每 5° 试一圈，优先接近“从底座看向物体”的方向（手腕离奇异远），物体在两指方向上的宽度要夹得住。
        from_box：中心是点云框自动给的——按合上以后手指的位置定中心（连杆夹爪合上时手指往前伸 ext(w)）：竖直夹让合上后的手指中段
        对着顶面往下 1.25 cm（矮的对着半高）、指尖离桌≥AUTO_TIP_CLEARANCE（6 mm）；水平夹中心往回让 ext(w)、高度按物体半高；够不到或碰桌就每次抬 5 mm 再试。
        用户自己拖过的中心就原样用。返回 (结果 dict 或 None, 说明)。"""
        seed = self._seed(seed)
        c0 = np.asarray(center, float)
        yaw = box["yaw"] if box else 0.0
        lx, ly, h = box["size"] if box else (0.04, 0.04, 0.05)
        top = box["top"] if box else c0[2] + 0.025
        bottom = box["bottom"] if box else top - h
        e1, e2 = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])

        def across(u):                         # 物体在水平方向 u 上有多宽（长方体的投影）
            return float(lx * abs(u @ e1) + ly * abs(u @ e2))

        rots = []                              # (说明, 姿态, 两指方向上的物体宽度, 偏好)
        if mode == "vertical":
            for psi in (yaw, yaw + np.pi / 2):
                for flip in (0.0, np.pi):
                    y = np.array([np.cos(psi + flip), np.sin(psi + flip), 0.0])
                    z = np.array([0.0, 0.0, -1.0])
                    w = across(y[:2])
                    rots.append(("竖直夹", R.from_matrix(np.column_stack([np.cross(y, z), y, z])), w, w))
        else:
            radial = np.arctan2(c0[1], c0[0])
            for a in range(0, 360, 5):
                az = np.radians(a)
                z = np.array([np.cos(az), np.sin(az), 0.0])
                y = np.array([-np.sin(az), np.cos(az), 0.0])
                w = across(y[:2])
                off = abs((az - radial + np.pi) % (2 * np.pi) - np.pi)
                rots.append(("水平夹", R.from_matrix(np.column_stack([np.cross(y, z), y, z])), w, off))
        rots = [t for t in rots if t[2] <= self.MAX_GRASP_W]
        if not rots:
            return None, f"物体太宽：{'竖直' if mode == 'vertical' else '水平'}夹哪个方向都超过夹爪最大开口 {self.MAX_GRASP_W * 100:.0f} cm"
        rots.sort(key=lambda t: t[3])
        # 每个候选姿态的抓取中心。这款夹爪是连杆式的：合上到宽度 w 时手指沿夹爪轴往前伸 ext(w)（夹 4 cm 宽约 2.4 cm），
        # 所以按“合上以后”手指在哪来定中心（2026-11-02：原来按张开时算高度、最高只抬到物体顶面，3 cm 高的小方块合上后指尖总在桌面下）
        half_pad = tp.OPEN_TIP_LENGTH - tp.TOOL_LENGTH          # 抓取中心到张开时指尖：手指夹持段的一半
        cands = []                                               # (说明, 中心, 姿态, 宽度, 最高能抬到的中心高度)
        for n, r, w, _ in rots:
            ext = tp.finger_ext(w)
            if not from_box:
                cands.append((n, c0.copy(), r, w, c0[2]))
            elif mode == "vertical":
                # 合上后手指中段对着“顶面往下 1.25 cm（矮的就对着半高）”，指尖离桌至少 FINGER_CLEARANCE，手指至少夹住物体 5 mm
                z_want = max(bottom + h / 2, top - 0.0125) + ext
                z_min = max(tp.TABLE_Z, bottom) + self.AUTO_TIP_CLEARANCE + half_pad + ext   # 物体垫在东西上时按它底下那一层算
                z_max = top - 0.005 + half_pad + ext
                z = max(z_want, z_min)
                if z <= z_max + 1e-9:
                    cands.append((n, np.array([c0[0], c0[1], z]), r, w, z_max))
            else:
                # 水平夹：合上时手指往前伸 ext，中心先往回让 ext，合上后手指正好对着物体中心；高度按物体半高。
                # 物体沿伸进方向很长时（比如顺着长边伸进去夹窄边），手指根部（中心往后 half_pad）不能插进物体，
                # 就抓靠近夹爪的那一头：中心离物体近端面最多 half_pad - 5 mm
                ax = r.apply([0, 0, 1.0])
                along = min(-ext, -across(ax[:2]) / 2 + half_pad - 0.005)
                cands.append((n, np.array([c0[0], c0[1], bottom + h / 2]) + ax * along, r, w, max(bottom + h / 2, top - 0.01)))
        if not cands:
            return None, f"物体太矮：合上时手指要往下伸，竖直夹夹不住（物体高 {h * 100:.1f} cm），试试水平夹"
        why, best = None, None
        for k in range(0, 13 if from_box else 1):          # 碰桌或够不到就整体每次抬 5 mm 再试（不超过各自的最高高度）
            lifted = [(n, c + [0, 0, 0.005 * k], r, w) for n, c, r, w, zmax in cands if c[2] + 0.005 * k <= zmax + 1e-9]
            if not lifted:
                break
            best, why = self.pick_best(lifted, seed, d, lift)
            if best is not None:
                break
        if best is None:
            hint = "：物体太矮，水平夹时手腕会碰桌，试试竖直夹" if mode == "horizontal" and why and "桌面" in why else ""
            return None, f"{'竖直' if mode == 'vertical' else '水平'}夹摆不出能抓的姿态（比如：{why}）{hint}"
        note, c, r, w, wrist, q_g, label = best
        ax = r.apply([0, 0, 1.0])
        text = (f"{note}：抓取中心 {np.round(c * 1000).astype(int).tolist()} mm，两指方向物体宽 {w * 100:.1f} cm，"
                f"手腕离奇异 {wrist:.0f}°" + (f"，夹爪指向 {np.degrees(np.arctan2(ax[1], ax[0])):.0f}°" if note == "水平夹" else "")
                + (f"，合上后指尖离桌面/板子 {(c[2] - (tp.OPEN_TIP_LENGTH - tp.TOOL_LENGTH) - tp.finger_ext(w) - max(tp.TABLE_Z, bottom)) * 1000:.0f} mm"
                   if note == "竖直夹" else ""))
        text += f"；手臂摆法“{label}”"
        return {"pos": list(map(float, c)), "quat": list(map(float, r.as_quat())), "q": list(map(float, q_g)),
                "width": w, "wrist": wrist}, text

    def grasp_keep(self, pos, quat, seed, d, lift, width):
        """手动选抓取位置：夹爪方向不变，挑一种预抓取、抓取、抬起都能到的手臂摆法。返回 (结果 dict 或 None, 说明)。"""
        c, r = np.asarray(pos, float), R.from_quat(quat)
        best, why = self.pick_best([("", c, r, width)], self._seed(seed), d, lift)
        if best is None:
            return None, f"这个位置保持夹爪方向到不了（{why}）：换个位置，或者点“夹爪朝外摆正”/自动生成姿态"
        _, c, r, _, wrist, q_g, label = best
        return ({"pos": list(map(float, c)), "quat": list(map(float, r.as_quat())), "q": list(map(float, q_g)), "wrist": wrist},
                f"夹爪方向没变，用“{label}”摆法（手腕离奇异 {wrist:.0f}°，预抓取、抓取、抬起都能到）")

    # ---- 初始位姿：把机械臂现在的姿态存成初始位姿 / 恢复默认 ----
    def set_home(self, q=None):
        """把机械臂现在的实际姿态（或给定的关节角 q）设成初始位姿（“回初始位姿”和自动生成的“从初始位姿出发 / 回初始”都用它），
        存到 home.json。只存不动：要过去得点“回初始位姿”。"""
        if q is None:
            st = self.robot.read_state()
            if st is None:
                return "还没读到实际状态"
            q = st[1]
        q = np.array(q, float)
        bad = self.check_joints(q, None, 0.0)
        for j, name in [(2, "手臂快伸直了（J3 接近 0°）"), (4, "手腕在奇异姿态附近（J5 接近 0°）")]:
            if bad is None and abs(q[j]) < tp.SINGULAR_MIN_DEG:
                bad = name
        if bad:
            return f"现在这个姿态不适合当初始位姿：{bad}"
        with self.lock:
            pose = self.robot.arm.rm_algo_forward_kinematics(list(map(float, q)))
        tp.HOME_Q, tp.HOME_POS, tp.HOME_EULER = list(map(float, q)), np.array(pose[:3]), np.array(pose[3:6])
        with open(tp.HOME_FILE, "w") as f:
            json.dump({"q": tp.HOME_Q, "flange_pos": list(map(float, pose[:3])), "flange_euler": list(map(float, pose[3:6])),
                       "time": time.strftime("%Y-%m-%d %H:%M")}, f, indent=1)
        c = tp.home_target()[1]
        return (f"已设为初始位姿（关节 {np.round(q, 1).tolist()}，抓取中心 {np.round(c * 1000).astype(int).tolist()} mm，"
                f"摆法“{self.branch_label(q)}”），存在 home.json")

    # ---- 仿真 / 真实切换 ----
    def set_real(self, real):
        """切控制器的运行模式：0 仿真（真机不动，示教器里的模型动），1 真实。切之前停掉执行、跟随、拖动。"""
        if not hasattr(self.robot.arm, "rm_set_arm_run_mode"):
            return "离线模拟不能切换"
        if self.executing:
            return "正在执行轨迹，先停下再切换"
        if self.follow_on:
            self.set_follow(False)
        self.jog_on = False
        if self.drag_on:
            self.set_drag(False)
        with self.lock:
            ret = self.robot.arm.rm_set_arm_run_mode(1 if real else 0)
            ret2, now = self.robot.arm.rm_get_arm_run_mode()
        if ret != 0 or ret2 != 0 or now != (1 if real else 0):
            return f"切换失败（错误码 {ret}/{ret2}），去示教器里手动切"
        self.real = real
        tp.HOME_SPEED = 10 if real else 20
        self.robot.q_cmd = None
        return "已切到真实模式：机械臂会真的动" if real else "已切到仿真模式：真机不动"

    # ---- 拖动示教 ----
    def set_drag(self, on):
        """打开/关闭控制器的拖动示教（电流环，用手直接拖机械臂）。打开时停掉跟随，网页按实际位置显示。"""
        if not hasattr(self.robot.arm, "rm_start_drag_teach"):
            return "离线模拟不能拖动示教"
        if on and self.executing:
            return "正在执行轨迹，先停下再拖"
        if on and self.follow_on:
            self.set_follow(False)
        self.jog_on = False
        with self.lock:
            ret = self.robot.arm.rm_start_drag_teach(0) if on else self.robot.arm.rm_stop_drag_teach()
        if ret != 0:
            return f"{'打开' if on else '关闭'}拖动示教失败（错误码 {ret}；关节靠近限位时打不开，先挪开一点）"
        self.drag_on = on
        self.robot.q_cmd = None              # 拖完以后从实际位置重新算
        return "拖动示教已打开：用手拖机械臂，网页跟着动；到位后点“记录当前目标”" if on else "拖动示教已关闭"

    # ---- 跟随模式 ----
    def set_follow(self, on):
        """打开跟随：从机械臂实际位置开始，目标先设成实际位置，不会一开就猛冲。"""
        self.jog_on = False
        if not on:
            self.rec_cmd = None                  # 不跟随了：录制时 action 改记实际值
        if on and self.executing:
            return "正在执行轨迹，先停下再开跟随"
        if on and self.drag_on:
            return "拖动示教开着，先关掉再开跟随"
        st = self.robot.read_state()
        if on and st is None:
            return "还没读到实际状态"
        if on:
            self.f_q = np.array(st[1], float)
            self.f_pos, self.f_rot = tp.center_of(st[2]), R.from_euler("xyz", st[2][3:6])
            self.follow_goal = ("joints", self.f_q.copy())
            self.f_bad, self.f_speed = 0, 0.0
        self.follow_on, self.follow_msg = on, ""
        if not on:
            with self.lock:
                self.robot.stop()
        return None

    def jog_pose(self, pos, quat):
        """键盘点动（2026-11-04 用户：WASD 水平、R/F 上下、Q/E 转夹爪，务必真机跟着动——点云的框不准，要看着真夹爪对齐）：
        网页每动一下目标就把目标位姿发过来，这里借跟随的循环让真机限速跟过去（每步逆解、安全检查、盯偏差），
        JOG_SPEED 比跟随快一点。网页在一串按键开始时已经把目标放回真机现在的位置，不会一按就冲去别处。没开跟随时才用。"""
        if self.follow_on:
            return None                          # 开着跟随：网页直接发跟随目标
        if self.executing or self.drag_on or self.estopped or self.homing:
            return "正在执行 / 回初始位姿 / 拖动示教 / 急停中，真机没跟着动"
        st = self.robot.read_state()
        if st is None:
            return "还没读到实际状态，真机没跟着动"
        if not self.jog_on:
            self.f_q = np.array(st[1], float)
            self.f_pos, self.f_rot = tp.center_of(st[2]), R.from_euler("xyz", st[2][3:6])
            self.f_bad, self.f_speed = 0, 0.0
        self.follow_goal = ("pose", np.array(pos, float), R.from_quat(quat))
        self.jog_until = time.time() + JOG_IDLE_S
        self.jog_on = True
        return None

    def _follow_loop(self):
        dt = 1.0 / RATE_HZ
        while True:
            tick = time.time()
            if (self.follow_on or self.jog_on) and not (self.executing or self.estopped or self.drag_on or self.homing):
                try:
                    self._follow_step(dt)
                except Exception as e:          # 出任何意外都先停下，不让机械臂接着走
                    self.follow_on, self.jog_on, self.follow_msg = False, False, f"跟随出错已停：{e}"
            time.sleep(max(0.0, dt - (time.time() - tick)))

    def _follow_step(self, dt):
        v, w = JOG_SPEED[self.real] if self.jog_on and not self.follow_on else FOLLOW_SPEED[self.real]
        goal = self.follow_goal
        st = self.robot.read_state()
        if st is None or time.time() - st[0] > tp.STALE_S:
            self.follow_on, self.jog_on, self.follow_msg = False, False, "读不到实际状态，跟随已停"
            return
        if self.jog_on and time.time() > self.jog_until + 5.0:      # 松手 5 秒还没跟到（被挡住了？）：不再跟
            self.jog_on = False
            return
        prev = self.f_pos.copy()
        if goal[0] == "joints":          # 关节模式：各关节一起限速转过去
            dq = goal[1] - self.f_q
            n = np.max(np.abs(dq))
            step = dq if n <= FOLLOW_JOINT_SPEED[self.real] * dt else dq * (FOLLOW_JOINT_SPEED[self.real] * dt / n)
            if n < 1e-3:
                self.follow_msg = ""
                return
            q = self.f_q + step
            bad = self.check_joints(q, self.flange_r(self.f_q))
            if bad:
                self.follow_msg = f"停在这里：{bad}"
                self.jog_on = False
                return
            with self.lock:
                self.robot.send_joints(q)
            self.f_q = q
            self.rec_cmd = (list(map(float, q)), self.robot.gripper_closed)
            self.f_pos, self.f_rot = self.fk_center(q)
        else:                            # 位姿模式：抓取中心限速走过去，和手柄遥操作一样逐帧逆解
            p_goal, r_goal = goal[1], goal[2]
            if np.linalg.norm(p_goal - self.f_pos) < 1e-4 and (r_goal * self.f_rot.inv()).magnitude() < 1e-3:
                self.follow_msg = ""
                if self.jog_on and time.time() > self.jog_until:    # 跟到了、也没再按键：收工
                    self.jog_on = False
                return
            p, r = tp.step_toward(self.f_pos, self.f_rot, p_goal, r_goal, v * dt, w * dt)
            flange = p - r.apply([0.0, 0.0, tp.TOOL_LENGTH])
            with self.lock:
                hard, margin, where, q = self.robot.solve(flange, r, self.f_q)
            if hard is None and margin < 0:
                hard = where
            if hard is None and near_base(flange):
                hard = "法兰离底座太近"
            if hard is not None:
                self.follow_msg = f"停在这里：{hard}"
                if self.jog_on and time.time() > self.jog_until:    # 过不去、也没再按键：不再试
                    self.jog_on = False
                return
            with self.lock:
                self.robot.send_pose(flange, r)
            self.f_q, self.f_pos, self.f_rot = np.array(q, float), p, r
            self.rec_cmd = (list(map(float, q)), self.robot.gripper_closed)
        self.follow_msg = "跟随中" if self.follow_on else "真机跟着按键动"
        # 和执行轨迹一样盯着实际位置：跟不上（被挡住）就停
        self.f_speed += min(1.0, dt / tp.TRACK_LAG_S) * (np.linalg.norm(self.f_pos - prev) * 1000 / dt - self.f_speed)
        err = float(np.linalg.norm(tp.center_of(st[2]) - self.f_pos) * 1000)
        self.f_bad = self.f_bad + 1 if err > tp.TRACK_ERR_MM + tp.TRACK_LAG_S * self.f_speed else 0
        if self.f_bad >= tp.TRACK_ERR_FRAMES:
            self.set_follow(False)
            self.follow_msg = f"实际位置跟不上（偏差 {err:.0f} mm），跟随已停"

    # ---- 逆解 ----
    def track(self, seed, flange, rot):
        """从 seed 姿态出发，沿直线（位置直线、姿态球面插值）每次 1 cm / 5° 一小步走到目标法兰位姿，每步取离上一步最近的逆解。
        手臂摆法是连续变过去的，所以总的关节变化再大（比如绕底座转半圈、手腕附近 J4/J6 一起转）也不算“跳解”。
        返回 (关节角, None)；中途够不到、或某一小步关节要猛转（真要换摆法、或穿过奇异姿态）返回 (None, 原因)。"""
        with self.lock:
            pose = self.robot.arm.rm_algo_forward_kinematics(list(map(float, seed)))
        p0, r0 = np.array(pose[:3]), R.from_euler("xyz", pose[3:6])
        dp, ang = np.asarray(flange, float) - p0, (rot * r0.inv()).magnitude()
        h0 = 1.0 / int(np.clip(np.ceil(max(np.linalg.norm(dp) / 0.01, ang / np.radians(5))), 1, 300))
        slerp = Slerp([0.0, 1.0], R.concatenate([r0, rot]))
        q, s, h = np.array(seed, float), 0.0, h0
        for _ in range(2000):
            if s >= 1.0 - 1e-9:
                return q, None
            s1 = min(1.0, s + h)
            sols = self.all_solutions(p0 + s1 * dp, slerp([s1])[0], q)
            if not sols:
                return None, "够不到（逆解失败）"
            qn = min(sols, key=lambda v: np.max(np.abs(v - q)))
            jump = float(np.max(np.abs(qn - q)))
            if jump > TRACK_STEP_JUMP:
                # 奇异姿态附近关节转得快，但步子切小了还是连续的；切到很小还要猛转，才是真的换了摆法
                if h > h0 / 32:
                    h /= 2
                    continue
                return None, f"再往这边要换一种手臂摆法（关节要转 {jump:.0f}°）"
            q, s, h = qn, s1, min(h0, h * 2)
        return None, "够不到（逆解失败）"

    def solve_center(self, pos, quat, seed, ext=None):
        """抓取中心位姿 → (问题, 余量, 说明, 关节角)。seed 为空就从实际关节角出发。
        拖目标时手臂摆法要连续：从上一个姿态一小步一小步跟过去（track），这样能到的地方都能到；
        跟不过去时再看全部逆解里离上一个姿态最近的那个，也要大转（换了一种摆法）就不自动换，
        报“要换摆法”，模型停在上一个姿态，要换就去“手臂摆法”里点。
        （2026-10-28 以前只比“最近的那个解差多少度”：拖得快、一步跨几厘米，或者在奇异姿态附近时，
        明明连续能到也会报红，而且一红之后参考姿态不更新，越拖越红。）"""
        rot = R.from_quat(quat)
        flange = np.asarray(pos, float) - rot.apply([0.0, 0.0, tp.TOOL_LENGTH])
        if seed is None:
            st = self.robot.read_state()
            if st is None:
                return "还没读到实际状态", 0.0, "", None
            seed = st[1]
        seed = np.array(seed, float)
        q, why = self.track(seed, flange, rot)
        if q is None:
            sols = self.all_solutions(flange, rot, seed)
            if not sols:
                return "够不到（逆解失败）", 0.0, "", None
            q = min(sols, key=lambda s: np.max(np.abs(s - seed)))
            jump = float(np.max(np.abs(q - seed)))
            if jump > DRAG_MAX_JUMP:
                why = why if "摆法" in why else f"再往这边要换一种手臂摆法（关节要转 {jump:.0f}°）"
                return why + "，要换就去“手臂摆法”里点", 0.0, "", None
        ext = self.robot.finger_ext() if ext is None else ext   # 拖目标时按真实夹爪现在的开口
        bad = self.check_joints(q, self.flange_r(seed), ext)
        margin, where = tp.model_margin(q, ext)
        hard = bad if (bad is not None and margin >= 0) else None   # 离桌面的问题用余量表示
        for j, name in [(2, "手臂快伸直了"), (4, "手腕快转到奇异姿态")]:   # 和 teleop 一样只拦“继续靠近”
            if hard is None and abs(q[j]) < tp.SINGULAR_MIN_DEG and abs(q[j]) < abs(seed[j]) - 0.5:
                hard = f"{name}（J{j + 1} 接近 0°）"
        return hard, margin, where, q

    # ---- 规划 ----
    def first_joints(self, pos, rot, q_now, prefer=None):
        """第 1 帧的关节角：从当前关节角和初始位姿的关节角两处出发各解一次，取能到、安全、和现在差得最少的。
        prefer：这一帧记录时的关节角（用关节按钮调出来的手臂摆法），给了就优先用这种摆法。"""
        flange = pos - rot.apply([0.0, 0.0, tp.TOOL_LENGTH])
        ref = np.array(prefer, float) if prefer is not None else q_now
        best, reason = None, ""
        seeds = ([ref] if prefer is not None else []) + [q_now, np.array(tp.HOME_Q, float)]
        for seed in seeds:
            with self.lock:
                hard, margin, where, q = self.robot.solve(flange, rot, seed, max_jump=360)
            if hard is None and margin >= 0:
                d = np.max(np.abs(q - ref))
                if best is None or d < best[0]:
                    best = (d, q)
            reason = hard or where
        return (best[1], None) if best else (None, reason)

    def fk_center(self, q):
        with self.lock:
            pose = self.robot.arm.rm_algo_forward_kinematics(list(map(float, q)))
        return tp.center_of(pose), R.from_euler("xyz", pose[3:6])

    def flange_r(self, q):
        with self.lock:
            pose = self.robot.arm.rm_algo_forward_kinematics(list(map(float, q)))
        return float(np.hypot(pose[0], pose[1]))

    # ---- 规划的三块积木：几何路径 → 逐点逆解 / 安全检查 → 排时间 ----
    def cart_path(self, P, Rs, human, rng):
        """穿过 P/Rs 这几帧的笛卡尔路径（和 Pilz、UR 的“直线 + 拐角过渡”一样）：相邻帧之间走直线，
        不停的中间帧处用二次贝塞尔圆角拐过去（最多提前 BLEND_R 开始拐，曲线在拐角的三角形里，不会冲过头），
        首尾两帧准确到达；姿态用旋转样条平滑过渡。
        human > 0 时帧和帧之间路径随机鼓一点、姿态晃一点（经过帧时归零）。返回密集的 [(抓取中心, 姿态)]，含起点。"""
        P = [np.asarray(p, float) for p in P]
        m = len(P) - 1
        L = np.array([max(np.linalg.norm(P[j + 1] - P[j]), (Rs[j + 1] * Rs[j].inv()).magnitude() * ROT_LEN, 1e-4)
                      for j in range(m)])
        knots = np.concatenate([[0.0], np.cumsum(L)])
        rot_spline = RotationSpline(knots, R.concatenate(Rs))
        rad = np.zeros(m + 1)
        for i in range(1, m):
            rad[i] = min(BLEND_R, 0.4 * np.linalg.norm(P[i] - P[i - 1]), 0.4 * np.linalg.norm(P[i + 1] - P[i]))
        bump_p = rng.normal(size=(m, 3)) * human * 0.015 * rng.uniform(0.3, 1.0, size=(m, 1)) / np.sqrt(3)
        bump_r = rng.normal(size=(m, 3)) * human * np.radians(3.0) / np.sqrt(3)
        # 一段一段：(起点路程, 终点路程, s∈[0,1] → 位置)；路程和 knots 是同一把尺子，用来查姿态
        pieces = []
        for j in range(m):
            d = P[j + 1] - P[j]
            dl = np.linalg.norm(d)
            e = d / dl if dl > 1e-9 else np.zeros(3)
            ua = knots[j] + (rad[j] / dl * L[j] if dl > 1e-9 else 0.0)
            ub = knots[j + 1] - (rad[j + 1] / dl * L[j] if dl > 1e-9 else 0.0)
            A, B = P[j] + e * rad[j], P[j + 1] - e * rad[j + 1]
            pieces.append((ua, ub, lambda s, A=A, B=B: A + s * (B - A)))
            if j + 1 < m and rad[j + 1] > 0:
                d2 = P[j + 2] - P[j + 1]
                A2 = P[j + 1] + d2 / np.linalg.norm(d2) * rad[j + 1]
                ua2 = knots[j + 1] + rad[j + 1] / np.linalg.norm(d2) * L[j + 1]
                pieces.append((ub, ua2, lambda s, B=B, C=P[j + 1], A2=A2: (1 - s) ** 2 * B + 2 * s * (1 - s) * C + s ** 2 * A2))
        out = [(P[0], Rs[0])]
        for u0, u1, f in pieces:
            dist = np.sum(np.linalg.norm(np.diff([f(s) for s in np.linspace(0, 1, 9)], axis=0), axis=1))
            ang = (rot_spline(u1) * rot_spline(u0).inv()).magnitude()
            n = int(max(1, np.ceil(max(dist / DENSE_STEP[0], ang / DENSE_STEP[1]))))
            for s in np.linspace(0, 1, n + 1)[1:]:
                u = u0 + s * (u1 - u0)
                j = min(int(np.searchsorted(knots, u, side="right")) - 1, m - 1)
                w = np.sin(np.pi * np.clip((u - knots[j]) / L[j], 0, 1)) ** 2   # 帧中间最大，到帧时归零
                out.append((f(s) + w * bump_p[j], rot_spline(u) * R.from_rotvec(w * bump_r[j])))
        return out

    def lock_wrist(self, pos, q, wrist, ext):
        q = np.array(q, float, copy=True)
        q[5] = wrist
        actual, rot = self.fk_center(q)
        if np.linalg.norm(actual - pos) > 0.001:
            return q, rot, "固定手腕后无法保持抓取中心"
        return q, rot, self.check_joints(q, ext=ext)

    def ik_path(self, path, q0, ext=None, hold_wrist=False):
        """密集路径逐点逆解 + 安全检查：从 q0 出发，每点取离上一点最近的解（相邻点关节最多差 5°）。
        执行时发的就是这些关节角。ext：这一段夹爪指尖多伸出多少（离桌面检查用）。返回 (关节角列表, [(点序号, 问题)])。"""
        q, Q, bad = np.array(q0, float), [], []
        for i, (p, r) in enumerate(path):
            flange = p - r.apply([0.0, 0.0, tp.TOOL_LENGTH])
            with self.lock:
                hard, margin, where, qn = self.robot.solve(flange, r, q, ext=ext)
            if hard is None and margin < 0:
                hard = where
            if hard is None and near_base(flange):
                hard = "法兰离底座太近"           # 和 teleop 的规则一样：手臂折回底座附近容易压到自己或桌面
            if hard is None and hold_wrist:
                qn, actual_rot, hard = self.lock_wrist(p, qn, q0[5], ext)
                if hard is None:
                    path[i] = (p, actual_rot)
            if hard is not None:
                bad.append((i, hard))
                qn = q
            q = np.array(qn, float)
            Q.append(q)
        return Q, bad

    def ptp_path(self, q0, q1, ext=None):
        """关节空间直线（PTP：各关节同步转、同时到）：每 0.5° 一个点，逐点检查（ext 同 ik_path）。
        返回 (密集的 [(抓取中心, 姿态)], 关节角列表, 第一个问题或 None)，含起点。"""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = int(max(1, np.ceil(np.max(np.abs(q1 - q0)) / 0.5)))
        path, Q, bad, r_prev = [], [], None, self.flange_r(q0)
        for s in np.linspace(0, 1, n + 1):
            q = q0 + s * (q1 - q0)
            if s > 0:
                bad = bad or self.check_joints(q, r_prev, ext)
                r_prev = self.flange_r(q)
            path.append(self.fk_center(q))
            Q.append(q)
        return path, Q, bad

    def timing(self, path, Q, speed, ang_speed, human=0.0, rng=None, slow_at=(), precise=None):
        """给密集路径排时间（思路同 MoveIt 的 TOTG / TOPP）：沿路径算每处允许的最快速度——平移不超过 speed、
        转动不超过 ang_speed、每个关节不超过 JOINT_SPEED（奇异姿态附近关节转得快，这里就自动放慢）——
        再从前往后、从后往前各扫一遍，让加速度不超过 PATH_ACC，首尾速度为 0；最后速度曲线抹成 S 形。
        slow_at：夹爪要开合的那几帧的抓取中心，离它们 PRECISE_DIST 以内平移不超过 precise（精细段慢）。
        返回按 50 Hz 取的 [(抓取中心, 姿态, 关节角)]，不含起点。"""
        dt = 1.0 / RATE_HZ
        P = np.array([p for p, _ in path])
        Rs = R.concatenate([r for _, r in path])
        Q = np.array(Q, float)
        if len(P) < 2:
            return []
        dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
        dth = (Rs[1:] * Rs[:-1].inv()).magnitude()
        dq = np.max(np.abs(np.diff(Q, axis=0)), axis=1)
        du = np.maximum(np.hypot(dp, ROT_LEN * dth), 1e-6)
        un = np.concatenate([[0.0], np.cumsum(du)])
        if un[-1] < 1e-5:
            return []
        qd = JOINT_SPEED[self.real]
        vlim = np.minimum.reduce([speed * du / np.maximum(dp, 1e-9), ang_speed * du / np.maximum(dth, 1e-9),
                                  qd * du / np.maximum(dq, 1e-9)])
        if human > 0:      # 人手快慢不均：整段快一点或慢一点，沿路程再慢慢起伏
            x = un[1:] / un[-1]
            k, ph = int(rng.integers(1, 3)), rng.uniform(0, 2 * np.pi)
            vlim = vlim * (1 + human * rng.uniform(-0.25, 0.25)) * (1 + human * 0.25 * np.sin(2 * np.pi * k * x + ph))
        if len(slow_at) and precise:   # 精细段：靠近夹爪开合的地方慢下来（从快到慢由下面的加速度限制自然过渡）
            dmin = np.min(np.linalg.norm(P[:, None, :] - np.asarray(slow_at, float)[None, :, :], axis=2), axis=1)
            near = (dmin[:-1] < PRECISE_DIST) | (dmin[1:] < PRECISE_DIST)
            vlim = np.where(near, np.minimum(vlim, precise * du / np.maximum(dp, 1e-9)), vlim)
        a = PATH_ACC[self.real]
        # 拐弯处的向心加速度 v²·曲率 也不能超过 a：转角越急越慢；急转（> 60°，比如抬起后马上放回同一点）就停下再走
        moving = dp > 1e-6
        last = np.maximum.accumulate(np.where(moving, np.arange(len(dp)), -1))   # 原地转姿态的点沿用上一个移动方向
        e = (np.diff(P, axis=0) / np.maximum(dp, 1e-12)[:, None])[np.maximum(last, 0)]
        turn = np.arccos(np.clip(np.sum(e[:-1] * e[1:], axis=1), -1, 1))
        turn = np.where((last[:-1] >= 0) & moving[1:], turn, 0.0)
        sharp = np.flatnonzero(turn > np.radians(60)) + 1
        if sharp.size:
            cuts = [0, *sharp.tolist(), len(P) - 1]
            return [s for i, j in zip(cuts[:-1], cuts[1:]) if j > i
                    for s in self.timing(path[i:j + 1], Q[i:j + 1], speed, ang_speed, human, rng, slow_at, precise)]
        # 只在真的拐弯（前后都有平移）的地方限速；原地转姿态（dp=0）原来会算成 0，每个路径点都停一下再起步，
        # 悬停转夹爪只有约 10°/s、拖“转动”滑块也没用（2026-11-06 查出来）
        corner = np.where(turn > 1e-9,
                          np.sqrt(a * 0.5 * (dp[:-1] + dp[1:]) / np.maximum(turn, 1e-9)) * du[1:] / np.maximum(dp[1:], 1e-9),
                          np.inf)
        box = np.ones(max(1, int(round(SMOOTH_S / dt))))
        box /= box.size
        scale, out = 1.0, []
        for _ in range(4):
            node = np.concatenate([[0.0], np.minimum(np.minimum(vlim[:-1], vlim[1:]) * scale, corner), [0.0]])
            v = node.copy()
            for i in range(len(du)):                       # 从前往后：加速度不超过 a
                v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a * du[i]))
            for i in range(len(du) - 1, -1, -1):           # 从后往前：减速度不超过 a
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a * du[i]))
            vs = v[:-1] + v[1:]
            tn = np.concatenate([[0.0], np.cumsum(np.where(vs > 1e-6, 2 * du / np.maximum(vs, 1e-9), 2 * np.sqrt(du / a)))])
            t = np.arange(0.0, tn[-1] + dt, dt)
            vel = np.diff(np.interp(t, tn, un)) / dt
            vel = np.convolve(np.convolve(vel, box), box)  # S 形加减速：加速度连续变化
            u = np.concatenate([[0.0], np.cumsum(vel) * dt])
            u *= un[-1] / u[-1]
            x = np.interp(u[1:], un, np.arange(len(un), dtype=float))
            i0 = np.minimum(np.floor(x).astype(int), len(un) - 2)
            f = x - i0
            qo = Q[i0] + f[:, None] * (Q[i0 + 1] - Q[i0])
            peak = np.max(np.abs(np.diff(np.vstack([Q[:1], qo]), axis=0))) / dt
            if peak <= qd * 1.1:
                break
            scale *= qd / peak                              # 抹平后关节偶尔略超：整体放慢一点再排
        for i, fi, q in zip(i0, f, qo):
            out.append((P[i] + fi * (P[i + 1] - P[i]), Rs[i] * R.from_rotvec(fi * (Rs[i].inv() * Rs[i + 1]).as_rotvec()), q))
        return out

    def plan(self, keyframes, speed, ang_speed, human=0.0, seed=None, precise=None, start_noise=None, grip0=None,
             with_approach=True, exec_noise=None):
        """从第 1 帧开始依次经过各关键帧。带夹爪动作或停顿的帧（以及第一、最后一帧、标了 stop 的帧）停下，其余帧圆角经过。
        相邻两个停靠帧之间是一段“不停”的笛卡尔路径（直线 + 拐角过渡）；走不通就拆开，一段一段地试直线，
        哪段直线不行就那段改关节运动（PTP），也不行才报问题。另外算“出发段”：执行时先用关节运动从机械臂实际位置
        移到第 1 帧（预览不播这一段）。返回 dict；samples / approach 的每个元素是 (抓取中心, 姿态, 关节角)。
        grip0：开头夹爪是不是合着（不给就按真夹爪现在的状态）；with_approach=False 不算出发段（自动生成时单独验证一个动作用）。
        （2026-11-03 这个参数原来叫 approach，被函数里同名的出发段变量盖掉了，出发段一直没算：
        “真机移到这里”报“已经在目标位置”，执行时第一步直接跳到第 1 帧——改了名，execute 里也加了第一步离实际位置太远就不执行的保护）
        exec_noise：执行时小扰动的大小（米，见 EXEC_NOISE_*）；返回的 sent 是实际发给机械臂的点，samples 仍是干净轨迹（录成动作）。"""
        st = self.robot.read_state()
        if st is None:
            return {"ok": False, "problems": [{"t": 0, "reason": "还没读到实际状态"}]}
        if not keyframes:
            return {"ok": False, "problems": [{"t": 0, "reason": "还没有关键帧"}]}
        v_max, w_max = SPEED_LIMIT[self.real]
        speed, ang_speed = min(float(speed), v_max), min(float(ang_speed), w_max)
        human = float(np.clip(human, 0.0, 1.0))
        seed = int(seed) if seed is not None else int(np.random.default_rng().integers(1 << 30))
        rng = np.random.default_rng(seed)
        dt = 1.0 / RATE_HZ
        precise = PRECISE_SPEED if precise is None else float(np.clip(float(precise), 0.005, speed))
        # 夹爪开合：和那时夹爪的状态一样（比如预抓取帧写了“张开”、夹爪本来就开着）就当没写——不用在那里停下、
        # 不用等到位，也不算精细段。开头按真夹爪现在的状态算
        grip_now, grips = bool(self.robot.gripper_closed) if grip0 is None else bool(grip0), []
        for k in keyframes:
            g = k.get("gripper") or ""
            if g in ("open", "close"):
                if (g == "close") == grip_now:
                    g = ""
                else:
                    grip_now = g == "close"
            grips.append(g)
        pts = [(np.array(k["pos"], float), R.from_quat(k["quat"]), g, float(k.get("pause", 0)))
               for k, g in zip(keyframes, grips)]

        def zones(a, b):
            """a→b 这一段里要慢下来的地方：两头里夹爪真要开合的那一帧"""
            return [pts[k][0] for k in (a, b) if pts[k][2]]
        samples, events, qs, problems, notes, exts = [], [], [], [], [], []

        def problem(i, reason):
            if not problems or problems[-1]["reason"] != reason or i - problems[-1]["end"] > 1:
                problems.append({"start": i, "end": i, "t": round(i * dt, 2), "reason": reason})
            problems[-1]["end"] = i

        def emit(out):
            for p, r, qj in out:
                samples.append((p, r, qj)); qs.append([float(v) for v in qj]); exts.append(hand["ext"])

        # 夹爪状态跟着轨迹走：离桌面检查按“那时候”指尖伸出多少算。开头按真实夹爪现在的开口；
        # 合上之后按这一帧填的物体宽度（没填按空夹全合）；张开之后按全开
        hand = {"ext": self.robot.finger_ext() if grip0 is None else (tp.finger_ext(None) if grip0 else 0.0)}

        stack_release, stack_align_wait = [], []
        kf_at = {}                       # 停靠的关键帧 → 到达时的采样序号（录制在哪一帧结束要用）

        def arrive(k):
            """到达第 k 帧：（人手会犹豫一下）执行夹爪动作，再停顿。"""
            kf_at[k] = len(samples) - 1
            if keyframes[k].get("stack_align_wait"):
                stack_align_wait.append(kf_at[k])
            grip, pause = pts[k][2], pts[k][3]
            extra = human * rng.uniform(0.0, 0.3) if grip else 0.0
            for _ in range(int(round(extra / dt))):
                samples.append(samples[-1]); qs.append(qs[-1]); exts.append(hand["ext"])
            if grip in ("open", "close"):
                i = len(samples) - 1
                events.append((i, grip))
                if grip == "open" and keyframes[k].get("stack_release"):
                    stack_release.append(i)
            if grip == "close":
                # 只有带 grip_width 的帧（勾了“按物体宽度检查”或点云自动生成的）才按物体宽度加合上时指尖的伸出量；
                # 用户自己摆的抓取点以用户为准（2026-11-02 用户：自己设的落点能保证安全）
                width = keyframes[k].get("grip_width")
                hand["ext"] = tp.finger_ext(width) if "grip_width" in keyframes[k] else 0.0
                margin, where = tp.model_margin(qs[-1], hand["ext"])
                if margin < 0:
                    how = f"按物体宽度 {width * 100:.0f} cm 算" if width is not None else "没填物体宽度，按空夹全合算"
                    problem(len(samples) - 1, f"第 {k + 1} 帧合上夹爪时指尖会多伸出 {hand['ext'] * 100:.1f} cm（{how}），"
                                              f"{where}，这一帧要再抬高 {-margin * 100:.1f} cm")
            elif grip == "open":
                hand["ext"] = 0.0
            for _ in range(int(round(pause / dt))):
                samples.append(samples[-1]); qs.append(qs[-1]); exts.append(hand["ext"])

        def held(reason):
            """夹着东西时“夹爪尖离桌面太近”说清楚是合上以后的指尖（比张开时长），不然看着模型明明离桌面还远"""
            if hand["ext"] > 0 and reason and "夹爪尖" in reason:
                return f"{reason}（夹着东西，合上后的指尖比张开时多伸出 {hand['ext'] * 100:.1f} cm，这一段要再抬高一点）"
            return reason

        def lin(a, b, q0):
            bend = 0.0 if keyframes[b].get("stack_linear") else human
            rots = [p[1] for p in pts[a:b + 1]]
            if keyframes[b].get("stack_align"):
                rots[0] = samples[-1][1]
            path = self.cart_path([p[0] for p in pts[a:b + 1]], rots, bend, rng)
            return (path, *self.ik_path(path, q0, hand["ext"], hold_wrist=keyframes[b].get("hold_wrist", False)))

        # 起点扰动：第 1 帧带着关节角（一般是初始位姿）时，每个关节随机偏一点（正态分布，标准差 start_noise 度，最多 2 倍），
        # 位姿按正解跟着变。和 LIBERO 一样（robosuite 初始关节加 0.02 rad≈1° 的高斯噪声），每条示范的起点略有不同，
        # 属于“状态扰动”而不是示范动作噪声。随机数从 seed 派生：预览和执行是同一个起点，“换一种”换一组
        start_q, start_dq = None, 0.0
        sn = 0.0 if start_noise is None else max(0.0, float(start_noise))
        if sn > 0 and keyframes[0].get("q") is not None:
            q0 = np.array(keyframes[0]["q"], float)
            dq = np.clip(np.random.default_rng([seed, 1]).normal(0.0, sn, 6), -2 * sn, 2 * sn)
            qn = np.clip(q0 + dq, self.robot.q_min, self.robot.q_max)
            if self.check_joints(qn, None, hand["ext"]) is None:
                c, r = self.fk_center(qn)
                keyframes = [dict(keyframes[0], q=list(map(float, qn)))] + list(keyframes[1:])
                pts[0] = (np.array(c, float), r, pts[0][2], pts[0][3])
                start_q, start_dq = qn, float(np.max(np.abs(qn - q0)))
            else:
                notes.append("起点扰动后的姿态不安全，这一条用了没扰动的起点")

        # 第 1 帧
        q_act = np.array(st[1], float)
        q_prev, why = self.first_joints(pts[0][0], pts[0][1], q_act, keyframes[0].get("q"))
        if q_prev is None:
            return {"ok": False, "dt": dt, "n": 1, "duration": 0, "q": [list(map(float, q_act))], "events": [],
                    "problems": [{"start": 0, "end": 0, "t": 0, "reason": f"第 1 帧：{why}"}], "samples": [],
                    "approach": [], "seed": seed}
        samples.append((pts[0][0], pts[0][1], np.array(q_prev, float))); qs.append([float(v) for v in q_prev])
        exts.append(hand["ext"])
        arrive(0)

        # 停靠点：第 1 帧、最后一帧、带夹爪动作或停顿的帧；相邻停靠点之间是一段“不停”的路
        # 标了 joint 的帧（比如“回初始”）：从上一帧直接走关节运动过去，路径形状不重要，不用试直线
        joint_kf = {i for i, k in enumerate(keyframes) if k.get("joint")}
        stops = sorted({0, len(pts) - 1} | {i for i, p in enumerate(pts) if p[2] or p[3] > 0}
                       | joint_kf | {i - 1 for i in joint_kf if i > 0} | {i for i, k in enumerate(keyframes) if k.get("stop")})
        for a, b in zip(stops[:-1], stops[1:]):
            if b in joint_kf and b == a + 1:
                qb, why = self.first_joints(pts[b][0], pts[b][1], q_prev, keyframes[b].get("q"))
                if qb is None:
                    problem(len(samples), f"第 {b + 1} 帧：{why}")
                    qb = q_prev
                path, Q, bad = self.ptp_path(q_prev, qb, hand["ext"])
                if bad:
                    problem(len(samples), f"第 {a + 1}→{b + 1} 帧关节运动途中：{bad}")
                emit(self.timing(path, Q, speed, ang_speed, human, rng, zones(a, b), precise))
                q_prev = np.array(qb, float)
                arrive(b)
                continue
            path, Q, bad = lin(a, b, q_prev)
            stack_linear = keyframes[b].get("stack_linear")
            segment_speed = min(speed, STACK_DESCENT_SPEED) if keyframes[b].get("stack_release") else speed
            segment_angular = min(ang_speed, STACK_ALIGN_SPEED) if keyframes[b].get("stack_align") else ang_speed
            if not bad:
                emit(self.timing(path, Q, segment_speed, segment_angular, 0.0 if stack_linear else human,
                                 rng, zones(a, b), precise))
                q_prev = Q[-1]
                arrive(b)
                continue
            if stack_linear:
                n0 = len(samples)
                out = self.timing(path, Q, segment_speed, segment_angular, 0.0, rng, zones(a, b), precise)
                emit(out)
                for i, hard in bad:
                    problem(n0 + round(i / max(1, len(path) - 1) * max(0, len(out) - 1)),
                            f"{keyframes[b]['tag']}直线走不通：{held(hard)}")
                q_prev = Q[-1]
                arrive(b)
                continue
            # 整段走不通：一段一段来，每段先试直线，不行这段改关节运动（这样伸进、抬起、放下这些直线段照样是直的）
            parts, why, qk, ptp_k = [], None, q_prev, []
            for k in range(a, b):
                path_k, Q_k, bad_k = lin(k, k + 1, qk)
                if not bad_k:
                    parts.append((path_k, Q_k)); qk = Q_k[-1]
                    continue
                flange = pts[k + 1][0] - pts[k + 1][1].apply([0.0, 0.0, tp.TOOL_LENGTH])
                with self.lock:   # 记录时的手臂摆法优先
                    s0 = np.array(keyframes[k + 1]["q"], float) if keyframes[k + 1].get("q") else qk
                    hard, margin, where, qn = self.robot.solve(flange, pts[k + 1][1], s0, max_jump=360, ext=hand["ext"])
                if hard is not None or margin < 0:
                    why = f"第 {k + 2} 帧：{held(hard or where)}"
                    break
                path_k, Q_k, bad_j = self.ptp_path(qk, qn, hand["ext"])
                if bad_j:
                    why = f"第 {k + 1}→{k + 2} 帧直线走不通（{bad_k[0][1]}），关节运动途中也不行（{bad_j}）"
                    break
                # 关节运动时抓取中心走的路程比直线长太多，说明要换一种手臂摆法、手臂会绕一大圈，不自动做
                c = np.array([s[0] for s in path_k])
                walk = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
                straight = float(np.linalg.norm(pts[k + 1][0] - pts[k][0]))
                if walk > FALLBACK_DETOUR * straight + 0.10:
                    why = (f"第 {k + 1}→{k + 2} 帧直线走不通（{bad_k[0][1]}），改走关节运动的话手臂要绕一大圈"
                           f"（要走 {walk * 100:.0f} cm，直线只有 {straight * 100:.0f} cm），换个位置或姿态")
                    break
                parts.append((path_k, Q_k)); qk = np.array(qn, float); ptp_k.append(k)
            if why is None:
                notes.append(f"第 {a + 1}→{b + 1} 帧：一口气圆滑走不通（{bad[0][1]}），已改成分段走，"
                             + "、".join(f"第 {k + 1}→{k + 2} 帧" for k in ptp_k) + "改成关节运动")
                for path_k, Q_k in parts:
                    emit(self.timing(path_k, Q_k, speed, ang_speed, human, rng, zones(a, b), precise))
                q_prev = qk
                arrive(b)
                continue
            notes.append(f"第 {a + 1}→{b + 1} 帧：{why}")
            out, n0 = self.timing(path, Q, speed, ang_speed, human, rng, zones(a, b), precise), len(samples)
            emit(out)
            for i, hard in bad:       # 密集点的序号换算成 50 Hz 采样的序号，网页上标红
                problem(n0 + int(round(i / max(1, len(path) - 1) * max(0, len(out) - 1))), held(hard))
            q_prev = Q[-1]
            arrive(b)

        # 夹着东西时不能比抓取的时候还低（2026-11-02 用户：“抓取点就是最低点”——放置点比抓取点低了 1 cm，方块被压在桌面上）：
        # 合夹爪那一刻，物体底下是桌面，记下抓取中心正下方桌面那一点在夹爪坐标系里的位置；之后夹着走的每个采样点，
        # 这一点（跟着夹爪动）都不能低于桌面（留 HOLD_TOL）。夹爪竖直时就是“夹着东西时抓取中心不能低于抓取高度”。
        # 只查桌面：放到更高的台子上没问题；从高台子上抓了放到桌面上会被拦（不知道台子多高），要放低就先放下再重新抓
        ev, hold, deepest = dict(events), None, 0.0
        for i, (p, r, _) in enumerate(samples):
            if ev.get(i) == "close":
                hold = r.inv().apply([0.0, 0.0, tp.TABLE_Z - p[2]])
            if hold is not None:
                bz = float((p + r.apply(hold))[2])
                if bz < tp.TABLE_Z - HOLD_TOL:
                    deepest = max(deepest, tp.TABLE_Z - bz)
                    problem(i, "夹着东西时比抓取的位置还低：物体会压到桌面上（放下的位置不能比抓取的位置低）")
            if ev.get(i) == "open":
                hold = None
        if deepest > 0:
            notes.append(f"夹着东西最多比抓取的位置低了 {deepest * 1000:.0f} mm：把放下的那一帧（和途中的帧）抬高到不低于抓取点")

        # 出发段：执行时先从实际位置用关节运动移到第 1 帧
        q_first = np.array(qs[0], float)
        approach, a_bad = [], None
        if with_approach and np.max(np.abs(q_first - q_act)) >= 1.0:
            path, Q, a_bad = self.ptp_path(q_act, q_first, self.robot.finger_ext())
            approach = self.timing(path, Q, v_max, w_max)   # 出发段不录数据，按模式的速度上限走（关节照样限速）
        if a_bad:
            problems.insert(0, {"start": 0, "end": 0, "t": 0, "reason": f"从实际位置移到第 1 帧的途中：{a_bad}"})
        path = [list(map(float, s[0])) for s in samples[::3]] + [list(map(float, samples[-1][0]))]
        # 录制从第 1 帧录到最后一个不是“回初始”的帧（LIBERO 的示范也是放下撤出就结束，回初始那段不录）。
        # 2026-11-03 试过改成整条都录、轨迹里不放回初始，用户又要回按帧名判断（回初始照样执行，方便复位下一条）
        last_kf = max([k for k, kf in enumerate(keyframes) if kf.get("tag") != "回初始"] or [0])
        rec_end = kf_at.get(last_kf, len(samples) - 1)
        # 最后一下是松手（之前夹过东西）：录到松手后 REC_TAIL_S 秒就停，照 LIBERO 的结尾
        evs = [(i, g) for i, g in events if i <= rec_end]
        rec_tail = bool(evs and evs[-1][1] == "open" and any(g == "close" for _, g in evs[:-1]))
        if rec_tail:
            rec_end = min(rec_end, evs[-1][0] + int(round(REC_TAIL_S / dt)))
        sent, noise = None, 0.0
        if exec_noise and float(exec_noise) > 0 and not problems and not stack_release:
            sent, noise, why = self.exec_noise_path(samples, events, exts, min(float(exec_noise), EXEC_NOISE_MAX), seed,
                                                    rec_end, dt)
            if why:
                notes.append(why)
        return {"ok": not problems, "sent": sent, "exec_noise": round(noise * 1000, 1), "dt": dt, "n": len(samples), "duration": round(len(samples) * dt, 2), "rec_end": rec_end,
                "rec_tail": rec_tail, "stack_release": stack_release, "stack_align_wait": stack_align_wait,
                "q": qs, "events": events, "problems": problems, "notes": notes, "path": path, "seed": seed,
                "approach_s": round(len(approach) * dt, 1), "samples": samples, "approach": approach,
                "speed": speed, "ang_speed": ang_speed, "start_noise": sn, "start_dq": round(start_dq, 2),
                "start_q": list(map(float, start_q)) if start_q is not None else None}

    def exec_noise_path(self, samples, events, exts, amp, seed, rec_end, dt):
        """执行时小扰动：返回 (发给机械臂的点 [(抓取中心, 姿态, 关节角)], 实际用的均方根 米, 说明)。只动位置不动姿态。"""
        n = len(samples)
        rng = np.random.default_rng([int(seed), 2])            # 从规划种子派生：预览和执行是同一组扰动
        t = np.arange(n) * dt
        off = np.zeros((n, 3))
        for ax in range(3):
            for _ in range(3):
                off[:, ax] += rng.normal() * np.sin(2 * np.pi * rng.uniform(*EXEC_NOISE_HZ) * t + rng.uniform(0, 2 * np.pi))
        off *= amp / max(1e-9, float(np.sqrt(np.mean(np.sum(off ** 2, axis=1)))))
        P = np.array([s[0] for s in samples])
        env = np.ones(n)
        if events:
            ev = P[[i for i, _ in events]]
            d = np.min(np.linalg.norm(P[:, None, :] - ev[None, :, :], axis=2), axis=1)
            env *= np.clip((d - EXEC_NOISE_FADE[0]) / (EXEC_NOISE_FADE[1] - EXEC_NOISE_FADE[0]), 0.0, 1.0)
        ramp = max(1, int(round(EXEC_NOISE_RAMP_S / dt)))
        k = np.arange(n)
        env *= np.clip(k / ramp, 0.0, 1.0) * np.clip((rec_end - k) / ramp, 0.0, 1.0)
        out = list(samples)
        for _ in range(3):
            bad = []
            for i in range(n):
                if env[i] <= 0:
                    out[i] = samples[i]
                    continue
                p, r, qc = samples[i]
                p2 = p + env[i] * off[i]
                with self.lock:
                    hard, margin, _, q2 = self.robot.solve(p2 - r.apply([0.0, 0.0, tp.TOOL_LENGTH]), r, qc, ext=exts[i])
                if hard is not None or margin < 0:
                    bad.append(i)
                else:
                    out[i] = (p2, r, np.array(q2, float))
            if not bad:
                break
            # 不安全的点前后 0.5 秒里慢慢收到 0（不能一下跳回干净轨迹）
            for i in bad:
                env[max(0, i - ramp):i + ramp + 1] *= np.clip(np.abs(np.arange(max(0, i - ramp), min(n, i + ramp + 1)) - i) / ramp,
                                                             0.0, 1.0)
        else:
            return None, 0.0, "执行时小扰动：有些地方加了就不安全，这一条没加扰动"
        q_off = np.array([o[2] for o in out]) - np.array([s[2] for s in samples])
        if np.max(np.abs(np.diff(q_off, axis=0))) > 1.0:          # 相邻两步扰动差了 1° 以上：逆解跳了，不用
            return None, 0.0, "执行时小扰动：逆解不连续，这一条没加扰动"
        used = float(np.sqrt(np.mean(np.sum((np.array([o[0] for o in out]) - P) ** 2, axis=1))))
        return out, used, None

    def check_joints(self, q, r_prev=None, ext=None):
        """关节空间那一段的逐帧检查（直接按关节角算，不用逆解）：限位、自碰撞、手臂和夹爪离桌面、法兰离底座。
        r_prev：上一帧法兰离底座竖直轴的距离；已经在底座附近（比如机械臂竖直伸直）时，只要在远离就放行。
        ext：指尖比全开时多伸出多少（tp.finger_ext）；不给就按真实夹爪现在的开口。"""
        out = (q < self.robot.q_min) | (q > self.robot.q_max)
        if out.any():
            return tp.limit_msg(int(np.argmax(out)), q, self.robot.q_min, self.robot.q_max)
        with self.lock:
            if self.robot.arm.rm_algo_safety_robot_self_collision_detection(list(map(float, q))) != 0:
                return "会和自身相撞"
            pose = self.robot.arm.rm_algo_forward_kinematics(list(map(float, q)))
        flange, rot = np.array(pose[:3]), R.from_euler("xyz", pose[3:6])
        margin, where = tp.model_margin(q, self.robot.finger_ext() if ext is None else ext)   # 最低点离桌面至少 1 cm
        if margin < 0:
            return where
        r = np.hypot(flange[0], flange[1])
        return "法兰离底座太近" if near_base(flange) and (r_prev is None or r < r_prev - 1e-4) else None

    # ---- 执行 ----
    def settle(self, p, dt, target_rot=None):
        """等真机停在抓取中心 p 附近（见 SETTLE_*）。返回 (停稳了?, 最后还差多少 mm)。点了停止就马上返回。"""
        t_wait, hist, still_since, err_w = time.time(), [], None, np.nan
        while time.time() - t_wait < SETTLE_TIMEOUT_S and not self.stop_flag:
            st = self.robot.read_state()
            now = time.time()
            if st is not None:
                c_now = tp.center_of(st[2])
                err_w = float(np.linalg.norm(c_now - p) * 1000)
                hist.append((now, c_now))
                past = [c for t_h, c in hist if now - t_h >= SETTLE_STILL_S]
                still = bool(past) and np.linalg.norm(c_now - past[-1]) * 1000 < 0.3
                still_since = (still_since or now) if still else None
                if still:
                    if target_rot is not None:
                        actual_rot = R.from_euler("xyz", st[2][3:6])
                        angle = np.degrees((target_rot * actual_rot.inv()).magnitude())
                        if err_w < SETTLE_MM and angle <= STACK_RELEASE_ANGLE_DEG:
                            return True, err_w
                    elif err_w < SETTLE_MM or (err_w < SETTLE_MAX_MM and now - still_since >= 0.5):
                        return True, err_w
            time.sleep(dt)
        return False, err_w

    def cams_problem(self):
        """录示范要两台相机都有画面（π0.5 训练要第三视角和腕部两路）：缺哪台、为什么；都好返回 None。只看状态，很快。"""
        import recorder
        miss = []
        for label, cam in (("第三视角", self.scene.cam if self.scene else None), ("腕部", self.wrist)):
            if cam is None:
                miss.append(f"{label}相机没接上")
            elif not recorder.cam_fresh(cam):
                miss.append(f"{label}相机没有画面（{getattr(cam, 'state', '')}）")
        return "、".join(miss) or None

    def watch_cameras(self):
        """后台线程：启动时没找到的相机每 5 秒再找一次（插好线不用重启服务）；断了的相机 cameras.DepthCamera 自己会重连。"""
        import cameras
        while True:
            time.sleep(5.0)
            if self.wrist is None:
                try:
                    cam = cameras.find("wrist")
                except Exception:
                    cam = None
                if cam is not None:
                    self.wrist = cam
                    if self.recorder is not None:
                        self.recorder.cams["wrist"] = cam
                    print("[相机] 腕部相机找到了", flush=True)

    def get_recorder(self):
        if self.recorder is None:
            import recorder
            self.recorder = recorder.Recorder(self.robot, {"scene": self.scene.cam if self.scene else None, "wrist": self.wrist},
                                              lambda: self.rec_cmd)
        return self.recorder

    def rec_meta(self, plan=None, keyframes=None):
        """存进示范文件的附加信息（以后查数据、转 LeRobot 时用）。"""
        meta = {"home_q": list(map(float, tp.HOME_Q)), "real": bool(self.real), "tool_length": tp.TOOL_LENGTH,
                "calib_T": json.dumps(self.scene.T.tolist()) if self.scene is not None and self.scene.T is not None else ""}
        if plan is not None:
            meta.update(plan_seed=int(plan.get("seed", 0)), speed=float(plan.get("speed", 0)), ang_speed=float(plan.get("ang_speed", 0)),
                        start_noise_deg=float(plan.get("start_noise", 0.0)), start_q=json.dumps(plan.get("start_q")),
                        exec_noise_mm=float(plan.get("exec_noise", 0.0) if plan.get("sent") else 0.0))
        if keyframes is not None:
            meta["keyframes"] = json.dumps(keyframes, ensure_ascii=False)
        return meta

    def execute(self, plan, on_done, record=None):
        """record：要录制就给 {"folder", "task", "depth", "meta"}：出发段走完、停到第 1 帧开始录，到 plan["rec_end"] 就停、存成一条示范
        （最后一下是松手时＝松手后 REC_TAIL_S 秒，接着撤出、回初始不录；否则＝最后一个不是“回初始”的帧，停稳了再停）；中途停了就不存。"""
        """在后台线程里按 50 Hz 把规划好的位姿发给机械臂（低跟随透传），盯着偏差，出问题就停。"""
        self.executing, self.stop_flag, self.jog_on = True, False, False
        self.status = "执行中"
        os.makedirs(tp.LOG_DIR, exist_ok=True)
        log_path = os.path.join(tp.LOG_DIR, time.strftime("twin_%Y%m%d_%H%M%S.csv"))
        log = csv.writer(open(log_path, "w", newline="", buffering=1))
        log.writerow(["t", "i", "cmd_cx", "cmd_cy", "cmd_cz", "cmd_qx", "cmd_qy", "cmd_qz", "cmd_qw", "gripper_closed",
                      "act_cx", "act_cy", "act_cz", "err_mm"] + [f"act_j{j}" for j in range(1, 7)])
        # 先走出发段（实际位置 → 第 1 帧，关节运动），再走规划好的轨迹
        # 执行时小扰动：发给机械臂的是 sent（规划轨迹 + 慢慢晃的偏移），录进数据的动作是干净的 samples
        clean = plan["approach"] + plan["samples"]
        steps = plan["approach"] + (plan.get("sent") or plan["samples"])
        events = {i + len(plan["approach"]): g for i, g in plan["events"]}
        stack_release = {i + len(plan["approach"]) for i in plan.get("stack_release", [])}
        stack_align_wait = {i + len(plan["approach"]) for i in plan.get("stack_align_wait", [])}
        rec = self.get_recorder() if record else None
        rec_start, rec_end = len(plan["approach"]), len(plan["approach"]) + plan.get("rec_end", len(plan["samples"]) - 1)
        self.rec_result = None
        saver = []                       # 后台存文件的线程（松手后停录时机械臂还在撤出，存文件不能卡住发指令）
        dt, n = plan["dt"], len(steps)
        st0 = self.robot.read_state()
        if st0 is not None and steps and steps[0][2] is not None:
            jump0 = float(np.max(np.abs(np.array(steps[0][2], float) - np.array(st0[1], float))))
            if jump0 > FIRST_STEP_MAX_DEG:
                self.executing, self.rec_cmd = False, None
                self.status = f"已停止：第一步离机械臂现在的姿态差 {jump0:.1f}°（应该从实际位置出发），没有执行"
                on_done(self.status, log_path)
                return
        cmd_speed, bad, reason, t0, end_note = 0.0, 0, None, time.time(), ""
        prev_p = steps[0][0]
        try:
            for i, (p, r, qj) in enumerate(steps):
                tick = time.time()
                if self.stop_flag:
                    reason = "手动停止"
                    break
                if rec is not None and i == rec_start:
                    # 出发段走完了：等真机停到第 1 帧再开始录（示范从静止的初始姿态开始）
                    self.status = "等机械臂停到第 1 帧再开始录"
                    self.settle(p, dt)
                    if self.stop_flag:
                        reason = "手动停止"
                        break
                    err = rec.start(record["folder"], record["task"], record.get("depth", False), "execute", record.get("meta"))
                    if err:
                        reason = f"录制没开起来：{err}"
                        break
                    self.status = "执行中（录制中）"
                flange = p - r.apply([0.0, 0.0, tp.TOOL_LENGTH])
                qc = clean[i][2]
                self.rec_cmd = ((list(map(float, qc)), self.robot.gripper_closed, list(map(float, qj)))
                                if qc is not None and qj is not None else None)
                with self.lock:
                    if qj is not None:
                        self.robot.send_joints(qj)    # 规划时解好、检查过的关节角，原样透传
                    else:
                        self.robot.send_pose(flange, r)
                if i in stack_align_wait:
                    settled, err_w = self.settle(p, dt, r)
                    if not settled:
                        reason = "手动停止" if self.stop_flag else f"放置对齐点未到位（位置差 {err_w:.1f} mm，需同时满足角度要求）"
                        break
                g = getattr(self.robot, "gripper", None)
                if i in events and g is not None and not g.health()[0]:
                    reason = f"夹爪没响应，没法{'合上' if events[i] == 'close' else '张开'}（{g.health()[1]}）"
                    break
                if i in events:
                    # 真机比指令慢约 0.5 秒：先等它真的到位停稳再开合夹爪（2026-10-27 的日志里夹爪合上时还差 18 mm）
                    strict = i in stack_release
                    settled, err_w = self.settle(p, dt, r if strict else None)
                    if not settled:
                        if not self.stop_flag:
                            alignment = "，叠放时也检查角度" if strict else "，可能被东西顶住了"
                            reason = (f"夹爪{'合上' if events[i] == 'close' else '张开'}前等了 {SETTLE_TIMEOUT_S:.0f} 秒还没到位"
                                      f"（位置差 {err_w:.1f} mm{alignment}）")
                            break
                    if self.stop_flag:
                        continue                      # 等的时候点了停止：不开合，下一轮就停
                    with self.lock:
                        self.robot.set_gripper(events[i] == "close")
                # 和 teleop 一样：偏差超过 30 mm + 0.5 s × 指令速度，连续 5 帧就停
                cmd_speed += min(1.0, dt / tp.TRACK_LAG_S) * (np.linalg.norm(p - prev_p) * 1000 / dt - cmd_speed)
                prev_p = p
                st = self.robot.read_state()
                err = np.nan
                if st is None or time.time() - st[0] > tp.STALE_S:
                    reason = "读不到实际状态"
                    break
                act_c = tp.center_of(st[2])
                err = float(np.linalg.norm(act_c - p) * 1000)
                bad = bad + 1 if err > tp.TRACK_ERR_MM + tp.TRACK_LAG_S * cmd_speed else 0
                if bad >= tp.TRACK_ERR_FRAMES:
                    reason = f"实际位置跟不上指令（偏差 {err:.0f} mm）"
                    break
                am = self.robot.actual_margin()
                if am is not None and am[0] < -0.02:
                    reason = am[1]
                    break
                log.writerow([f"{time.time() - t0:.3f}", i, *[f"{v:.5f}" for v in p], *[f"{v:.5f}" for v in r.as_quat()],
                              int(self.robot.gripper_closed), *[f"{v:.5f}" for v in act_c], f"{err:.1f}",
                              *[f"{v:.3f}" for v in st[1]]])
                self.progress = (i + 1, n)
                if rec is not None and i == rec_end and rec.running:
                    self.rec_cmd = ((list(map(float, qc)), self.robot.gripper_closed, list(map(float, qj)))
                                    if qc is not None and qj is not None else None)
                    if not plan.get("rec_tail"):
                        self.settle(p, dt)       # 停在最后一个不是“回初始”的帧：等真机停稳再停（指令比真机快约 0.5 秒）
                    if not self.stop_flag:
                        def saved(res):
                            self.rec_result = res
                            # 存好了马上告诉网页（左上角出现“成功了吗”），不用等回到初始位姿。原来要等整个执行完才发，
                            # 回初始途中点成功时“存好了”的提示被盖掉、看不到（2026-11-04 用户）
                            if record.get("on_saved"):
                                record["on_saved"](res)
                        saver.append(rec.stop(save=True, extra={"end": "rec_tail" if plan.get("rec_tail") else "rec_end"},
                                              background=saved))
                        self.status = "执行中（录好了，撤出、回初始位姿）" if i < n - 1 else "执行中"
                time.sleep(max(0.0, dt - (time.time() - tick)))
            else:
                # 指令发完了，真机还差约 0.5 秒的路：等它停稳再算“执行完”。网页收到“完成”会把目标放回实际位置，
                # 原来这时真机还在路上（最后几毫米走得慢），目标停在离终点 6 mm、高 3 mm 的地方，接着“设为抓取点”就存偏了
                # （2026-11-02 用户：按 M 确认过的抓取点，执行时夹偏四分之一个方块）
                self.status = "等机械臂停稳"
                settled, err_w = self.settle(steps[-1][0], dt)
                if self.stop_flag:
                    reason = "手动停止"
                elif not settled:
                    end_note = f"（最后还差 {err_w:.1f} mm 没停稳）"
        finally:
            if reason:
                with self.lock:
                    self.robot.stop()
            if rec is not None and rec.running:        # 中途停了：这一条不完整，不存
                path, text = rec.stop(save=False)
                self.rec_result = (None, f"{text}：{reason or '执行结束前没录完'}")
            for th in saver:                           # 等后台存完再报“执行完”（结果要一起发给网页）
                th.join(timeout=60)
            self.rec_cmd = None
            self.executing = False
            self.status = f"已停止：{reason}" if reason else "执行完成" + end_note
            on_done(self.status, log_path)


class OfflineRobot(tp.RealmanRobot):
    """离线模拟：不连控制器，用 SDK 的算法库（同一套逆解、自碰撞）加 ECO65 的 DH 参数模拟一台机械臂。
    关节按一阶滞后去追目标（时间常数 0.15 秒），用来在没有机械臂时调网页。"""

    DH = dict(d=[0.1625, 0.0, 0.0, -0.0589, 0.11, 0.0795], a=[0.0, -0.086, 0.26, 0.24, 0.0, 0.0],
              alpha=[0.0, -90.0, 0.0, 0.0, 90.0, -90.0], offset=[0.0, -90.0, 0.0, 90.0, 0.0, 0.0])
    Q_MIN, Q_MAX = [-178, -178, -160, -178, -178, -360], [178, 135, 145, 178, 178, 360]   # 控制器里读到的关节限位
    HOME_Q = tp.HOME_Q                                                                   # 初始位姿的关节角

    def __init__(self):
        from Robotic_Arm.rm_robot_interface import (Algo, rm_dh_t, rm_force_type_e, rm_inverse_kinematics_params_t,
                                                    rm_robot_arm_model_e)
        self.ik_params = rm_inverse_kinematics_params_t
        pad = lambda v: list(v) + [0.0, 0.0]
        self.arm = Algo(rm_robot_arm_model_e.RM_MODEL_UNIVERSAL_E, rm_force_type_e.RM_MODEL_RM_B_E, 6,
                        rm_dh_t(*(pad(self.DH[k]) for k in ("d", "a", "alpha", "offset"))))
        self.arm.rm_algo_set_joint_min_limit(list(map(float, self.Q_MIN)))
        self.arm.rm_algo_set_joint_max_limit(list(map(float, self.Q_MAX)))
        self.dh = {k: np.array(v, float) for k, v in self.DH.items()}
        self.q_min = np.array(self.Q_MIN, float) + tp.JOINT_LIMIT_MARGIN_DEG
        self.q_max = np.array(self.Q_MAX, float) - tp.JOINT_LIMIT_MARGIN_DEG
        self.gripper_closed, self.q_cmd, self.checked, self.frozen = False, None, None, False
        self.q = np.array(self.HOME_Q, float)
        self.q_goal = self.q.copy()
        self.latest = None
        self.reading = True
        threading.Thread(target=self._sim_loop, daemon=True).start()
        print("[离线模拟] 没有连控制器，用 SDK 算法库模拟一台 ECO65")

    def _sim_loop(self):
        dt = 0.01
        while self.reading:
            if not self.frozen:
                self.q += min(1.0, dt / 0.15) * (self.q_goal - self.q)
            pose = self.arm.rm_algo_forward_kinematics(list(self.q))
            self.latest = (time.time(), list(self.q), list(pose))
            time.sleep(dt)

    def send_pose(self, pos, rot):
        hard, _, _, q = self.solve(pos, rot, self.q_goal)
        if hard is None:
            self.q_goal = q
        return 0 if hard is None else 1

    def send_joints(self, q):
        self.q_goal = np.array(q, float)
        return 0

    def home(self):
        self.q_goal = np.array(tp.HOME_Q, float)      # 网页可能改过初始位姿，用 tp 里最新的
        time.sleep(1.5)

    def stop(self):
        self.q_goal = self.q.copy()

    def emergency_stop(self):
        self.frozen = True
        self.q_goal = self.q.copy()

    def release_estop(self):
        self.frozen = False
        return True

    def close(self):
        self.reading = False


# ======================= 网页 =======================
def make_app(twin):
    clients = set()
    loop_holder = {}

    async def broadcast(msg):
        data = to_json(msg)
        for ws in list(clients):
            try:
                await ws.send_str(data)
            except Exception:
                clients.discard(ws)

    def gripper_width():
        """真实夹爪当前开口（0 闭合 ~ 1 张开）；没连夹爪时按指令估计。"""
        g = getattr(twin.robot, "gripper", None)
        w = g.width() if g is not None else None
        return w if w is not None else (0.0 if twin.robot.gripper_closed else 1.0)

    def gripper_msg():
        """夹爪不正常时的一句话（没响应、正在重启……），正常时空。"""
        g = getattr(twin.robot, "gripper", None)
        if g is None:
            return "夹爪没连上（按夹爪按钮会重试）" if getattr(twin.robot, "want_gripper", False) else ""
        ok, msg = g.health()
        return "" if ok else (msg or "夹爪没响应")

    async def state_loop(app):
        """每秒 10 次把实际关节角和执行进度推给网页。"""
        while True:
            st = twin.robot.read_state()
            if clients:
                await broadcast({"type": "state", "q": list(map(float, st[1])) if st else None,
                                 "pose": list(map(float, st[2][:6])) if st else None,
                                 "executing": twin.executing, "progress": twin.progress, "status": twin.status,
                                 "gripper_closed": twin.robot.gripper_closed,
                                 "gripper_width": gripper_width(),
                                 "gripper_ok": (getattr(twin.robot, "gripper", None) is not None)
                                               if getattr(twin.robot, "want_gripper", False) else None,
                                 "gripper_msg": gripper_msg(), "finger_ext": twin.robot.finger_ext(),
                                 "follow_on": twin.follow_on, "follow_msg": twin.follow_msg, "drag_on": twin.drag_on,
                                 "jog_on": twin.jog_on,
                                 "recording": (len(twin.recorder.frames) / 15.0) if twin.recorder is not None and twin.recorder.running else None,
                                 "cam_problem": twin.cams_problem(),
                                 "policy": twin.policy_runner.info if twin.policy_runner is not None and twin.policy_runner.running else None})
            await asyncio.sleep(0.1)

    async def on_startup(app):
        loop_holder["loop"] = asyncio.get_running_loop()
        app["state_task"] = asyncio.create_task(state_loop(app))

    async def index(request):
        return web.FileResponse(os.path.join(WEB_DIR, "index.html"))

    def recorder_mod():
        import recorder
        return recorder

    def run_result(path, text):
        """最近这一次录制的结果（网页左上角显示）：存下了是第几条＋开头的小图，没存是为什么。"""
        if path:
            rec = twin.recorder
            th = rec.last_thumb if rec is not None and rec.last_saved == path else None
            name = os.path.basename(path)
            return {"state": "saved", "name": name, "idx": int(name[8:14]),
                    "thumb": ("data:image/jpeg;base64," + base64.b64encode(th).decode()) if th else None}
        why = (text or "").split("：", 1)[-1] if "中途停了" in (text or "") else (text or "")
        return {"state": "not_saved", "why": why}

    def rec_msg(text=None, saved=False):
        """录制卡片要显示的：设置、存储路径的情况（几条、剩多少空间）、刚才的结果。"""
        rec = twin.recorder
        return {"type": "rec", "cfg": twin.rec_cfg, "info": recorder_mod().folder_info(twin.rec_cfg["folder"]), "text": text,
                "saved": saved, "run": twin.rec_run,
                "cams": {"scene": recorder_mod().cam_fresh(twin.scene.cam if twin.scene else None),
                         "wrist": recorder_mod().cam_fresh(twin.wrist)}}

    def config_msg():
        return {
            "type": "config", "real": twin.real, "offline": isinstance(twin.robot, OfflineRobot), "tool_length": tp.TOOL_LENGTH, "tip_length": tp.TIP_LENGTH,
            "model_clearance": tp.MODEL_CLEARANCE, "finger_clearance": tp.FINGER_CLEARANCE, "gripper_r": tp.GRIPPER_BODY_R,
            "table_z": tp.TABLE_Z, "speed_limit": SPEED_LIMIT[twin.real],
            "q_min": list(map(float, twin.robot.q_min)), "q_max": list(map(float, twin.robot.q_max)),
            "home": {"pos": list(map(float, tp.home_target()[1])), "quat": list(map(float, tp.home_target()[0].as_quat()))},
            "home_q": list(map(float, tp.HOME_Q)), "home_yaw_deg": tp.HOME_YAW_DEG,
            "camera": twin.scene is not None,
            "calib": twin.scene.calib_text() if twin.scene is not None else "没有第三视角相机",
            "calibrated": twin.scene is not None and twin.scene.T is not None,
            "wrist_camera": twin.wrist is not None,
        }

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
        await ws.prepare(request)
        clients.add(ws)
        await ws.send_str(to_json(config_msg()))
        await ws.send_str(to_json(rec_msg()))
        loop = asyncio.get_running_loop()
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                kind, reply = m.get("type"), None
                if kind == "solve":
                    if twin.executing:
                        continue
                    # ext：网页上仿真夹爪（目标那台）的指尖多伸出多少——目标的夹爪和真夹爪分开，不按真夹爪的开口算
                    hard, margin, where, q = await loop.run_in_executor(
                        None, twin.solve_center, m["pos"], m["quat"], m.get("seed"), m.get("ext"))
                    reply = {"type": "solve", "id": m.get("id"), "ok": hard is None and margin >= 0,
                             "reason": hard or (where if margin < 0 else ""), "margin": round(float(margin) * 1000),
                             "q": list(map(float, q)) if q is not None else None}
                elif kind == "plan":
                    if twin.executing:
                        continue
                    p = await loop.run_in_executor(None, twin.plan, m["keyframes"], m["speed"], m["ang_speed"],
                                                   m.get("human", 0.0), m.get("seed"), m.get("precise"), m.get("start_noise"),
                                                   None, True, m.get("exec_noise"))
                    p.pop("samples", None); p.pop("approach", None); p.pop("sent", None)
                    reply = {"type": "plan", **p}
                elif kind == "branches":
                    seed = m.get("seed")
                    if seed is None:
                        st = twin.robot.read_state()
                        seed = st[1] if st else tp.HOME_Q
                    b = await loop.run_in_executor(None, twin.ik_branches, m["pos"], m["quat"], seed, m.get("ext"))
                    reply = {"type": "branches", "id": m.get("id"), "items": b}
                elif kind == "fk":
                    c, quat, bad, q = await loop.run_in_executor(None, twin.fk_check, m["q"], m.get("ext"))
                    reply = {"type": "fk", "id": m.get("id"), "pos": list(map(float, c)), "quat": list(map(float, quat)),
                             "ok": bad is None, "reason": bad or "", "q": list(map(float, q))}
                elif kind == "autogen":
                    if twin.executing:
                        continue
                    # actions：几个抓放动作按顺序做；老消息只有 pick / place 两个点就当一个动作
                    actions = m.get("actions") or [{"pick": m["pick"], "place": m["place"], "width": m.get("width")}]
                    frames, p, notes = await loop.run_in_executor(
                        None, twin.autogen_multi, actions, float(m.get("dist", 0.08)), float(m.get("lift", 0.10)),
                        bool(m.get("home", True)), m["speed"], m["ang_speed"], m.get("human", 0.0), m.get("seed"),
                        bool(m.get("start_home", False)), m.get("precise"), m.get("start_noise"), m.get("exec_noise"),
                        bool(m.get("hover_align", False)), bool(m.get("place_align", False)), m.get("mode") == "stack",
                        float(m.get("align_height", STACK_ALIGN_HEIGHT)))
                    if p is not None:
                        p.pop("samples", None); p.pop("approach", None); p.pop("sent", None)
                    reply = {"type": "autogen", "keyframes": frames, "plan": p, "notes": notes}
                elif kind == "set_home":
                    if twin.executing:
                        reply = {"type": "msg", "text": "正在执行轨迹，先停下再改初始位姿"}
                    else:
                        text = await loop.run_in_executor(None, twin.set_home, m.get("q"))
                        await broadcast(config_msg())       # 网页里的初始位姿（放到初始位姿、自动生成）跟着更新
                        reply = {"type": "msg", "text": text}
                elif kind == "cam_jpeg":
                    # 相机的实时画面（网页轮询）：第三视角（调相机位置、标定时看，叠加棋盘格角点）或腕部
                    if m.get("which") == "wrist":
                        lat = twin.wrist.latest if twin.wrist is not None else None
                        data = None
                        if lat is not None:
                            ok, buf = cv2.imencode(".jpg", lat[1], [cv2.IMWRITE_JPEG_QUALITY, 70])
                            data = buf.tobytes() if ok else None
                        reply = {"type": "cam_jpeg", "which": "wrist", "data": base64.b64encode(data).decode() if data else None,
                                 "text": "" if twin.wrist is not None else "没有腕部相机（没接或启动时没打开）"}
                    else:
                        zones = bool(m.get("zones"))
                        data = await loop.run_in_executor(None, lambda: twin.scene.jpeg(zones=zones)) if twin.scene else None
                        reply = {"type": "cam_jpeg", "which": "scene", "data": base64.b64encode(data).decode() if data else None,
                                 "text": "" if twin.scene else "没有第三视角相机（没接或启动时没打开）"}
                elif kind == "rec_config":
                    # 录制设置：存储路径、任务描述、存不存深度、执行时录不录。存到 record_config.json，下次打开还在
                    for k in ("folder", "task", "depth", "on"):
                        if k in m:
                            twin.rec_cfg[k] = recorder_mod().expand(m[k]) if k == "folder" else (bool(m[k]) if k in ("depth", "on") else str(m[k]))
                    with open(REC_CFG_PATH, "w", encoding="utf-8") as f:
                        json.dump(twin.rec_cfg, f, ensure_ascii=False, indent=1)
                    await broadcast(rec_msg())
                elif kind == "ls_dir":
                    reply = {"type": "ls_dir", **recorder_mod().list_dir(m.get("path") or twin.rec_cfg["folder"])}
                elif kind == "mkdir":
                    base = recorder_mod().expand(m.get("path"))
                    name = "".join(ch for ch in str(m.get("name", "")) if ch not in '/\\:*?"<>|').strip()
                    if not name:
                        reply = {"type": "ls_dir", **recorder_mod().list_dir(base), "error": "文件夹名不能是空的"}
                    else:
                        try:
                            os.makedirs(os.path.join(base, name), exist_ok=True)
                            reply = {"type": "ls_dir", **recorder_mod().list_dir(os.path.join(base, name))}
                        except OSError as e:
                            reply = {"type": "ls_dir", **recorder_mod().list_dir(base), "error": f"建不了：{e.strerror}"}
                elif kind == "rec_manual":
                    # 手动录：开着跟随、拖动示教时也能录（没有指令时 action 记实际值）
                    rec = twin.get_recorder()
                    if m.get("on"):
                        if twin.executing:
                            reply = rec_msg("正在执行轨迹：执行时录制用“执行时录制”开关")
                        elif twin.cams_problem():
                            reply = rec_msg(f"{twin.cams_problem()}：录不全两路画面，没开始录")
                        else:
                            err = rec.start(twin.rec_cfg["folder"], twin.rec_cfg["task"], twin.rec_cfg["depth"], "manual", twin.rec_meta())
                            if not err:
                                twin.rec_run = {"state": "running"}
                            reply = rec_msg(err or "手动录制中……再点一次停止并保存")
                    else:
                        path, text = await loop.run_in_executor(None, lambda: rec.stop(save=True, extra={"end": "manual"}))
                        twin.rec_run = run_result(path, text)
                        await broadcast(rec_msg(text, saved=path is not None))
                elif kind == "rec_discard":
                    rec = twin.recorder
                    name = os.path.basename(rec.last_saved) if rec is not None and rec.last_saved else None
                    text = rec.discard_last() if rec is not None else "这次还没录过"
                    if name and not (rec.last_saved and os.path.exists(rec.last_saved)):
                        twin.rec_run = {"state": "discarded", "name": name, "idx": int(name[8:14])}
                    await broadcast(rec_msg(text))
                elif kind == "win":
                    # 网页开在 Orin 自己屏幕上时：切换全屏 / 最小化 Firefox 窗口（网页自己做不到，见 winctl.py）
                    import winctl
                    text = await loop.run_in_executor(None, winctl.do, m.get("action"))
                    if text:
                        reply = {"type": "msg", "text": text}
                elif kind in ("board_find", "board_touch", "board_calib", "calib_refine", "cloud", "segment") and twin.scene is None:
                    text = "没有第三视角相机（没接或启动时没打开），点云和标定用不了"
                    reply = ({"type": "segment", "box": None, "text": text, "stack_request": m["stack_request"]}
                             if kind == "segment" and "stack_request" in m else {"type": "msg", "text": text})
                elif kind == "board_find":
                    # 外参标定第 1 步：画面里找棋盘格（几秒），找到后实时画面上标出要碰的四个角 1~4
                    text = await loop.run_in_executor(None, twin.scene.find_board)
                    reply = {"type": "calib", "text": text, "board": twin.scene.board is not None}
                elif kind == "board_touch":
                    # 第 2 步：夹爪合上、竖直朝下，指尖碰到第 m["label"] 个角时点记录：记下合上的指尖在底座坐标里的位置
                    st = twin.robot.read_state()
                    if st is None:
                        reply = {"type": "calib", "text": "还没读到机械臂实际位置"}
                    elif twin.robot.finger_ext() < 0.8 * tp.finger_ext(None):
                        reply = {"type": "calib", "text": "先把夹爪合上（点“夹爪合上”），用合拢的指尖去碰角点"}
                    else:
                        flange, rot = np.array(st[2][:3]), R.from_euler("xyz", st[2][3:6])
                        tip = flange + rot.apply([0.0, 0.0, tp.OPEN_TIP_LENGTH + twin.robot.finger_ext()])   # 按真实开口算指尖伸出多少
                        reply = {"type": "calib", "text": twin.scene.touch(m["label"], tip),
                                 "touched": list(twin.scene.touched)}
                elif kind in ("board_calib", "calib_refine"):
                    # 第 3 步：算外参（再用点云里的机械臂和模型做 ICP 精修）；calib_refine：只精修现在的外参。
                    # 精修要机械臂静止（拿的是这一刻的关节角对这一刻的点云）
                    st = twin.robot.read_state()
                    if twin.executing or twin.follow_on:
                        reply = {"type": "calib", "text": "机械臂在动（执行/跟随中），停下来再标定"}
                    elif kind == "calib_refine" and st is None:
                        reply = {"type": "calib", "text": "还没读到机械臂关节角"}
                    else:
                        q = list(map(float, st[1])) if st else None
                        fn = twin.scene.calibrate if kind == "board_calib" else twin.scene.refine
                        ok, text = await loop.run_in_executor(None, fn, q)
                        reply = {"type": "calib", "text": text, "done": ok}
                        if ok:
                            await broadcast(config_msg())
                elif kind == "scene_info":
                    # 第三视角相机的内参、畸变、外参和画面大小（离线算工作区、裁剪框用；内参是相机里读的，标定文件里没存）
                    sc = twin.scene
                    f = sc.frame() if sc is not None else None
                    reply = {"type": "scene_info", "K": sc.K.tolist() if sc is not None else None,
                             "dist": np.asarray(sc.dist).ravel().tolist() if sc is not None else None,
                             "T": sc.T.tolist() if sc is not None and sc.T is not None else None,
                             "shape": list(f[0].shape[:2]) if f is not None else None}
                elif kind == "cloud":
                    # 点云（底座坐标）用二进制发：'PCLD' + 点数 + xyz float32 + rgb uint8，几万个点约 1 MB
                    st = twin.robot.read_state()           # 拍点云这一刻的关节角：分割时去掉机械臂自己的点要用
                    res = await loop.run_in_executor(None, twin.scene.make_cloud, list(map(float, st[1])) if st else None)
                    if res is None:
                        reply = {"type": "msg", "text": "还没标定外参：先在“相机外参标定”里标一下，点云才能和机械臂对上"
                                 if twin.scene.T is None else "相机还没有画面"}
                    else:
                        xyz, rgb = res
                        await ws.send_bytes(b"PCLD" + np.uint32(len(xyz)).tobytes() + xyz.tobytes() + rgb.tobytes())
                elif kind == "segment":
                    # 点一下物体：去掉桌面和机械臂自己的点，从点到的地方按颜色长出物体，框出带朝向的长方体；tol：网页上的颜色容差
                    kw = {"tol": float(m["tol"])} if m.get("tol") is not None else {}
                    if m.get("square"):
                        kw["square"] = True
                    box, text = await loop.run_in_executor(None, lambda: twin.scene.segment(m["point"], **kw))
                    reply = {"type": "segment", "box": box, "text": text, "stack_request": m.get("stack_request")}
                elif kind == "grasp_auto":
                    # 按抓取中心（和框出的物体）自动生成抓取姿态；from_box：中心是框给的（程序定高度），否则用户拖过的中心原样用
                    res, text = await loop.run_in_executor(
                        None, twin.grasp_auto, m["pos"], m.get("mode", "vertical"), m.get("box"), float(m.get("dist", 0.08)),
                        float(m.get("lift", 0.10)), m.get("seed"), bool(m.get("from_box", False)))
                    reply = {"type": "grasp", "ok": res is not None, "text": text, **(res or {})}
                elif kind == "grasp_keep":
                    # 手动选抓取位置：夹爪方向不变，挑一种预抓取、抓取、抬起都能到的手臂摆法
                    res, text = await loop.run_in_executor(
                        None, twin.grasp_keep, m["pos"], m["quat"], m.get("seed"), float(m.get("dist", 0.08)),
                        float(m.get("lift", 0.10)), float(m["width"]) if m.get("width") is not None else None)
                    reply = {"type": "grasp", "ok": res is not None, "text": text, **(res or {})}
                elif kind == "align_out":
                    res, text = await loop.run_in_executor(
                        None, twin.align_out, m["pos"], m["quat"], m.get("seed"), float(m.get("dist", 0.08)),
                        float(m.get("lift", 0.10)), float(m["width"]) if m.get("width") is not None else None)
                    reply = {"type": "align", "ok": res is not None, "text": text, **(res or {})}
                elif kind == "set_real":
                    text = await loop.run_in_executor(None, twin.set_real, bool(m["real"]))
                    await broadcast(config_msg())
                    reply = {"type": "msg", "text": text}
                elif kind == "drag":
                    reply = {"type": "msg", "text": await loop.run_in_executor(None, twin.set_drag, bool(m["on"]))}
                elif kind == "follow":
                    err = "急停中，先点“解除急停”" if (m["on"] and twin.estopped) else twin.set_follow(bool(m["on"]))
                    reply = {"type": "msg", "text": err or ("跟随模式已打开：拖目标或调关节，机械臂会跟过去" if m["on"] else "跟随模式已关闭")}
                elif kind == "goal":
                    if twin.follow_on:
                        if m.get("q") is not None:
                            twin.follow_goal = ("joints", np.clip(np.array(m["q"], float), twin.robot.q_min, twin.robot.q_max))
                        else:
                            twin.follow_goal = ("pose", np.array(m["pos"], float), R.from_quat(m["quat"]))
                elif kind == "execute":
                    if twin.executing:
                        continue
                    if twin.estopped:
                        await ws.send_str(to_json({"type": "msg", "text": "急停中，先点“解除急停”"}))
                        continue
                    if twin.drag_on:
                        await ws.send_str(to_json({"type": "msg", "text": "拖动示教开着，先关掉再执行"}))
                        continue
                    if twin.rec_cfg.get("on") and twin.cams_problem():
                        await ws.send_str(to_json(rec_msg(f"{twin.cams_problem()}：录不全两路画面，没有执行（检查相机线，恢复画面后再点执行）")))
                        continue
                    if twin.follow_on:
                        twin.set_follow(False)
                    # 用预览时的同一个随机种子，执行的就是预览看到的那条轨迹（起点扰动也一样）
                    p = await loop.run_in_executor(None, twin.plan, m["keyframes"], m["speed"], m["ang_speed"],
                                                   m.get("human", 0.0), m.get("seed"), m.get("precise"), m.get("start_noise"),
                                                   None, True, m.get("exec_noise"))
                    if not p["ok"]:
                        p.pop("samples", None); p.pop("approach", None); p.pop("sent", None)
                        reply = {"type": "plan", **p, "refused": "轨迹有问题，没有执行"}
                    else:
                        cfg = twin.rec_cfg
                        record = ({"folder": cfg["folder"], "task": cfg["task"], "depth": cfg["depth"],
                                   "meta": twin.rec_meta(p, m["keyframes"])} if cfg.get("on") else None)

                        announced = []

                        def on_saved(result):
                            path, text = result
                            announced.append(path)
                            twin.rec_run = run_result(path, text)
                            asyncio.run_coroutine_threadsafe(broadcast(rec_msg(text, saved=path is not None)), loop)

                        def done(status, log_path):
                            msgs = [{"type": "done", "status": status, "log": log_path}]
                            if record is not None:
                                path, text = twin.rec_result or (None, "没录上")
                                if not announced:                       # 没走到存文件那一步（松手前停了、出错了）
                                    twin.rec_run = run_result(None, text)
                                # 存好时已经发过“存好了”：这里只刷新一下状态，不再覆盖（这期间可能已经标了成功）
                                msgs.append(rec_msg() if announced and path == announced[-1] else rec_msg(text, saved=path is not None))
                            for mm in msgs:
                                asyncio.run_coroutine_threadsafe(broadcast(mm), loop)
                        if record is not None:
                            record["on_saved"] = on_saved
                        twin.rec_run = {"state": "running"} if record is not None else {"state": "off"}
                        await broadcast(rec_msg())
                        threading.Thread(target=twin.execute, args=(p, done, record), daemon=True).start()
                        reply = {"type": "started", "n": p["n"], "duration": round(p["duration"] + p["approach_s"], 1),
                                 "text": (f"执行中，约 {p['duration'] + p['approach_s']:.0f} 秒，录制开着：停到第 1 帧开始录，"
                                          + (f"录到松手后约 {REC_TAIL_S:g} 秒（撤出、回初始那段不录）" if p.get("rec_tail") else
                                             f"录到第 {max([k for k, kf in enumerate(m['keyframes']) if kf.get('tag') != '回初始'] or [0]) + 1} 帧"
                                             "（回初始那段不录）"))
                                 if record else None}
                elif kind == "goto":
                    # 真机移到某一个关键帧：用关节运动从实际位置过去（和“执行”的出发段一样：途中逐点检查、限速、盯偏差、能停能急停），
                    # m["gripper"] 给了（"open"/"close"）而且和真夹爪现在不一样，到了以后再开合（等到位停稳再动）
                    if twin.executing:
                        continue
                    if twin.estopped:
                        reply = {"type": "msg", "goto": True, "text": "没有移动：急停中，先点“解除急停”"}
                    elif twin.drag_on:
                        reply = {"type": "msg", "goto": True, "text": "没有移动：拖动示教开着，先关掉"}
                    else:
                        if twin.follow_on:
                            twin.set_follow(False)
                        kf = {k: m["keyframe"][k] for k in ("pos", "quat", "q", "grip_width") if m["keyframe"].get(k) is not None}
                        grip = m.get("gripper")
                        kf["gripper"] = grip if grip in ("open", "close") and (grip == "close") != twin.robot.gripper_closed else ""
                        v_max, w_max = SPEED_LIMIT[twin.real]
                        # 和录的示范一样：先到正上方（手腕不拧）、原地转、再竖直下去（goto_frames）；这样走不通就直接过去
                        frames, how = await loop.run_in_executor(None, twin.goto_frames, kf)
                        p = await loop.run_in_executor(None, twin.plan, frames, v_max, w_max)
                        if not p["ok"] and len(frames) > 1:
                            p = await loop.run_in_executor(None, twin.plan, [kf], v_max, w_max)
                            how = "先到正上方再转的路走不通，直接过去" if p["ok"] else how
                        name = m.get("name") or f"第 {m.get('index', 0) + 1} 帧"    # 快捷键 M 移到目标时 name="目标位置"
                        if not p["ok"]:
                            why = "；".join(x["reason"] for x in p["problems"][:2])
                            reply = {"type": "msg", "goto": True, "text": f"没有移动：到{name}的路上有问题（{why}）"}
                        elif not p["approach"] and not p["events"]:
                            reply = {"type": "msg", "goto": True, "text": f"机械臂已经在{name}了"}
                        else:
                            def done(status, log_path, name=name):
                                asyncio.run_coroutine_threadsafe(
                                    broadcast({"type": "done", "goto": True, "status": status.replace("执行完成", f"已移到{name}"), "log": log_path}), loop)
                            threading.Thread(target=twin.execute, args=(p, done), daemon=True).start()
                            reply = {"type": "started", "n": p["n"], "duration": round(p["duration"] + p["approach_s"], 1),
                                     "text": f"正在移到{name}" + (f"（{how}）" if how else "") + f"，约 {p['duration'] + p['approach_s']:.0f} 秒"
                                             + ("，到了以后夹爪" + ("合上" if kf["gripper"] == "close" else "张开") if kf["gripper"] else "")}
                elif kind == "policy_test":
                    # 模型控制：只连一下模型服务、拿现在的画面问一次，机械臂不动
                    if twin.executing:
                        reply = {"type": "policy", "ok": False, "text": "正在执行 / 模型控制中，等它停了再测"}
                    else:
                        ok, text = await loop.run_in_executor(None, twin.policy().test, m.get("addr"), m.get("task"))
                        reply = {"type": "policy", "ok": ok, "text": text}
                elif kind == "policy_start":
                    # 模型控制真机（policy_run.py）：每隔几步拿两路相机画面＋关节角问一次台式机上的 π0.5，按回来的动作 50 Hz 发给机械臂
                    import policy_run
                    why = ("正在执行 / 模型控制中" if twin.executing else "急停中，先点“解除急停”" if twin.estopped
                           else "拖动示教开着，先关掉" if twin.drag_on else "正在回初始位姿" if twin.homing
                           else policy_run.cams_problem(twin))
                    if why:
                        reply = {"type": "policy", "ok": False, "text": f"没有开始：{why}"}
                    else:
                        if twin.follow_on:
                            twin.set_follow(False)
                        twin.jog_on = False
                        cfg = {k: m.get(k) for k in ("addr", "task", "replan", "max_s", "raw", "auto_end")}

                        def pol_done(status, log_dir):
                            for mm in ({"type": "policy", "ok": status.startswith("模型控制结束"), "text": status, "done": True},
                                       {"type": "done", "status": status, "log": log_dir}):
                                asyncio.run_coroutine_threadsafe(broadcast(mm), loop)
                        threading.Thread(target=twin.policy().run, args=(cfg, pol_done), daemon=True).start()
                        reply = {"type": "policy", "ok": True,
                                 "text": "开始：连模型服务、先问一次（服务刚起来时要十几秒），之后机械臂开始动。停：空格 / “停止” / 急停"}
                elif kind == "stop":
                    twin.stop_flag = True
                    twin.jog_on = False
                elif kind == "jog":
                    why = twin.jog_pose(m["pos"], m["quat"])
                    if why:
                        reply = {"type": "msg", "text": why}
                elif kind == "estop":
                    # 急停：不等锁、不排队（用单独的连接发），同时让执行和跟随的循环停下
                    twin.stop_flag = True
                    twin.follow_on = twin.jog_on = False
                    ret = await loop.run_in_executor(None, twin.robot.emergency_stop)
                    twin.estopped = True
                    twin.status = "急停中"
                    reply = {"type": "msg", "text": "已急停。确认安全后点“解除急停”" if not ret
                             else f"急停指令失败（错误码 {ret}），马上按实体急停！"}
                elif kind == "release":
                    with twin.lock:
                        ok = twin.robot.release_estop()
                    if ok:
                        twin.estopped = False
                    twin.status = "空闲" if ok else "解除急停失败"
                    reply = {"type": "msg", "text": "已解除急停" if ok else "解除急停失败，去示教器里解除"}
                elif kind == "home":
                    if twin.follow_on:
                        twin.set_follow(False)
                    twin.jog_on = False
                    if twin.estopped:
                        reply = {"type": "msg", "text": "急停中，先点“解除急停”"}
                    elif twin.drag_on:
                        reply = {"type": "msg", "text": "拖动示教开着，先关掉再回初始位姿"}
                    elif not twin.executing:
                        def go_home():
                            twin.status, twin.homing = "回初始位姿中", True
                            try:
                                with twin.lock:
                                    twin.robot.home()
                            finally:
                                twin.homing = False
                            twin.status = "空闲"
                            asyncio.run_coroutine_threadsafe(broadcast({"type": "msg", "text": "回初始位姿结束"}), loop)
                        # 放到后台线程里走，不等它：走的过程中这个网页发来的急停、停止也能马上处理
                        threading.Thread(target=go_home, daemon=True).start()
                        reply = {"type": "msg", "text": "正在回初始位姿……（随时可以急停）"}
                elif kind == "gripper":
                    with twin.lock:
                        twin.robot.set_gripper(bool(m["closed"]))
                elif kind == "gripper_restart":
                    g = getattr(twin.robot, "gripper", None)
                    if twin.executing:
                        reply = {"type": "msg", "text": "正在执行轨迹，先停下再重启夹爪"}
                    elif g is None:
                        twin.robot.gripper_try_t = 0.0          # 还没连上：马上重新连一次（连的时候失败会自动断电重启）
                        ok = await loop.run_in_executor(None, twin.robot.connect_gripper) is not None
                        reply = {"type": "msg", "text": "夹爪连上了" if ok else "夹爪还是连不上：检查末端那根线和夹爪的灯"}
                    else:
                        ok = await loop.run_in_executor(None, g.restart)
                        reply = {"type": "msg", "text": "夹爪重启好了" if ok else g.health()[1]}
                elif kind == "save":
                    os.makedirs(TEMPLATE_DIR, exist_ok=True)
                    name = "".join(ch for ch in m["name"] if ch.isalnum() or ch in "-_") or "template"
                    with open(os.path.join(TEMPLATE_DIR, name + ".json"), "w") as f:
                        json.dump(m["data"], f, ensure_ascii=False, indent=1)
                    reply = {"type": "msg", "text": f"已保存模板 {name}"}
                elif kind == "list":
                    os.makedirs(TEMPLATE_DIR, exist_ok=True)
                    reply = {"type": "templates", "names": sorted(f[:-5] for f in os.listdir(TEMPLATE_DIR) if f.endswith(".json"))}
                elif kind == "load":
                    path = os.path.join(TEMPLATE_DIR, os.path.basename(m["name"]) + ".json")
                    with open(path) as f:
                        reply = {"type": "template", "name": m["name"], "data": json.load(f)}
                if reply is not None:
                    await ws.send_str(to_json(reply))
        finally:
            clients.discard(ws)
        return ws

    async def no_cache(request, response):
        response.headers["Cache-Control"] = "no-cache"   # 改了网页文件后刷新就能拿到新版，不用强制刷新

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_response_prepare.append(no_cache)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", WEB_DIR)
    return app


def main():
    ap = argparse.ArgumentParser(description="三维示教网页")
    ap.add_argument("--ip", default="192.168.1.18")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--real", action="store_true", help="真实模式（机械臂会真的动），不加则是控制器仿真模式")
    ap.add_argument("--web-port", type=int, default=8000)
    ap.add_argument("--offline", action="store_true", help="离线模拟：不连控制器，用来没有机械臂时调网页")
    ap.add_argument("--no-camera", action="store_true", help="不开第三视角相机")
    args = ap.parse_args()
    if args.real:
        if input("真实模式下机械臂会真的动，手放在急停旁边。输入 yes 继续：") != "yes":
            raise SystemExit("已取消")
        tp.HOME_SPEED = 10          # 和 teleop 一样：真实模式下回初始位姿慢一半
    robot = OfflineRobot() if args.offline else tp.RealmanRobot(args.ip, args.port, real=args.real, home=False)
    twin = Twin(robot, args.real)
    if not args.no_camera:
        try:
            import cameras
            import scene
            cams = cameras.open_cameras(["scene", "wrist"])   # 两台一起开（open_cameras 每次新建 Context，分两次开前一台会断）
            if cams.get("scene") is not None:
                twin.scene = scene.Scene(cams["scene"])
                print("[相机] 第三视角相机已打开" + ("，" + twin.scene.calib_text() if twin.scene.calib_info else "，还没标定外参"))
            else:
                print("[相机] 没找到第三视角相机（Gemini 335）")
            twin.wrist = cams.get("wrist")
            print("[相机] 腕部相机已打开" if twin.wrist is not None else "[相机] 没找到腕部相机（Gemini 305），每 5 秒再找一次；找到之前不能录示范")
            threading.Thread(target=twin.watch_cameras, daemon=True).start()
        except Exception as e:             # 相机没接或打不开：网页其他功能照常用
            print(f"[相机] 没打开相机（{e}）")
    print(f"打开浏览器：http://<这台电脑的 IP>:{args.web_port}（Orin 是 172.16.230.23；本机用 http://localhost:{args.web_port}），Ctrl+C 退出")
    try:
        web.run_app(make_app(twin), host="0.0.0.0", port=args.web_port, print=None)
    finally:
        if twin.recorder is not None and twin.recorder.running:
            twin.recorder.stop(save=False)
        if twin.scene is not None:
            twin.scene.cam.close()
        if twin.wrist is not None:
            twin.wrist.close()
        robot.close()


if __name__ == "__main__":
    main()
