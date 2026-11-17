"""用 π0.5 模型控制真机（2026-11-06，用户：模型部署好了，上真机试试）。

模型跑在 4090 台式机上（teleop/train/serve.sh 起的服务，websocket 172.16.230.65:8000），这里是机器人这头的客户端，
接在三维示教服务里（相机和机械臂只能一个程序占着）：

    每隔几步（默认 8 步）：两台相机最新画面 + 实际关节角 + 夹爪开口 + 任务描述
        → 和训练时完全一样处理（先压成质量 90 的 JPEG 再解开，和录制时一样；裁剪/缩放用 train/eco65_io.py）
        → 发给模型服务 → 收回 15 步动作（15 帧/秒 = 1 秒）：6 个绝对关节角（弧度）+ 夹爪（1 开 / 0 合）
    50 Hz 控制循环：按时间在这一段动作里插值出现在该发的关节角，限速后过一遍和执行轨迹一样的安全检查
        （限位、自碰撞、整台机械臂离桌面、法兰离底座、抓取中心不出工作范围），再用 rm_movej_canfd 发出去；
        夹爪那一维带回差（< 0.3 合、> 0.7 开），变了才发；盯着实际位置跟不跟得上指令。
    下一段在这一段用完之前就去问（在后台线程里），回来以后按“拍画面那一刻”对齐接着用，机械臂不停顿。

训练数据里动作就是“那一刻发给机械臂的指令”，实际关节角比指令慢约 0.47 秒——这里发的方式和录制时一样，所以对得上。
停止：网页“停止”/空格/急停；到最长时间、安全检查拦住、模型服务没回应、相机没画面也会停（机械臂停在原地）。
每次运行在 logs/policy_时间/ 下存：control.csv（每一拍的目标、发出的指令、实际关节角、偏差）、
chunks.npz（每一段模型输出的动作和拍画面的时刻）、每次送给模型的两张图（左第三视角、右腕部，224×224）。
"""
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np

import teleop as tp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "train"))          # eco65_io：训练和部署共用的图像处理
import eco65_io  # noqa: E402
from openpi_client import msgpack_numpy  # noqa: E402   # openpi 自带的消息格式（从台式机 ~/openpi 拷过来的，没改）

DEFAULT_ADDR = "172.16.230.65:8000"
DEFAULT_TASK = "put the red cube on the plate"
FPS = 15                    # 训练数据的帧率：一段 15 步 = 1 秒
H = 15                      # 模型一次给多少步（pi05_eco65_red_plate_lora 的 action_horizon）
RATE_HZ = 50                # 控制循环频率（和执行轨迹一样）
JOINT_SPEED = {False: 90.0, True: 45.0}   # 发出去的指令每个关节最快多少 °/s（仿真 / 真实）；示范里大约 30 °/s
MAX_JUMP_DEG = 25.0         # 模型给的目标离现在的指令超过这么多度就不跟、停下（正常最多十几度：指令本来就比实际快约 0.5 秒）
# 抓取中心允许的范围（米，底座坐标）：示范在 x ±0.25、y 0.3~0.6，出了这个框就停。
CENTER_BOX = ((-0.40, 0.40), (0.15, 0.75), (0.035, 0.45))
# 高度：示范里合夹爪时抓取中心离桌 53~54 mm（夹 3.3 cm 方块合上后指尖离桌约 3 mm）。下面的离桌面检查手指按张开长度算，
# 合上以后手指往下多伸的 2~3 cm 靠这条管：指令要低于 48 mm 时不往下走、原地等（模型一般下一段就改过来），等了 2 秒还要往下才停
Z_FLOOR, Z_HOLD_S = 0.048, 2.0
# 夹着东西时（2026-11-06 真机：放下时指令降到 48 mm，夹着 3.3 cm 方块时手指最低点到了桌面下 3.5 mm、碰了桌面）：
# 合上以后手指比张开时往下多伸约 2.5 cm，所以夹着时不许比“合夹爪那一刻的抓取中心高度”更低——和生成示范时规划里的
# HOLD_TOL 一个意思（方块是从桌面上夹起来的，夹着它不能比夹的时候还低）。要往下就原地等，等 HOLD_WAIT_S 还要往下才停
HOLD_TOL = 0.002            # 最多比合夹爪时低 2 mm（读数误差）
HOLD_WAIT_S = 4.0           # 放下时模型可能先往下探一点再松手，多等一会儿
# 空夹：夹着 3.3 cm 方块时开口读数约 0.22，空夹合到底约 0.03。合上 GRIP_SETTLE_S 秒后还 < EMPTY_W 就是没夹到，停下
# （真机两次没夹住以后模型每 0.6 秒开合一次、乱晃：训练数据里没有“没夹到再抓”，空夹的状态模型没见过）
AUTO_END_S = 1.5            # 自动结束（命令行演示用 auto_end）：夹着东西松手后再跑这么久就停（示范也是松手后约 1.2 秒结束）
EMPTY_W, GRIP_SETTLE_S = 0.10, 1.2   # 夹爪调慢了（gripper.py SPEED 50），合到位要更久，多等一会儿再判断
GRIP_CLOSE_BELOW, GRIP_OPEN_ABOVE = 0.3, 0.7
GRIP_MIN_S = 0.6           # 夹爪刚动过这么久之内不再反过来动（离线测到过 0.2 秒里开合三次：两段动作交接时模型的值来回跨线）
# “解除限制”（网页模型控制卡片的按钮，2026-11-06 用户要看原版 pi05_base 不受约束时怎么动）：去掉工作范围框、高度下限
# （含夹着时的）、“目标一下跳太远就停”、空夹就停，关节限速放宽到 RAW_JOINT_SPEED、夹爪防抖缩到 RAW_GRIP_MIN_S；
# 保留保护机械臂本身的检查：离桌面（手臂 1 cm、指尖 3 mm）、自碰撞、关节限位、法兰离底座、跟不上指令（被挡住）、读不到状态
RAW_JOINT_SPEED = 90.0      # °/s；不限速的话两段动作交接时目标会跳十几度，低跟随透传一下子冲过去
RAW_GRIP_MIN_S = 0.2
JPEG_QUALITY = 90           # 录制时存 JPEG 的质量（recorder.JPEG_QUALITY），送给模型前同样压一遍
STALE_IMG_S = 0.5           # 相机画面超过这么久没更新就算没画面
FIRST_TIMEOUT_S = 60.0      # 第一次推理（服务刚起来要编译，十几秒）
RESP_TIMEOUT_S = 3.0        # 之后每次推理最多等这么久（正常约 0.1 秒）
NO_CHUNK_S = 3.0            # 一段动作用完以后停在最后一步等下一段，等这么久还没来才停（Orin 走 Wi-Fi，到台式机 ping 偶尔 300 ms 以上）
HOME_WARN_DEG = 5.0
# 实际状态偶尔会卡一下：2026-11-06 夹爪 Modbus 读失败（data parsing error，自动断电重启）的那一两秒里，读状态的连接也停了 0.3 秒以上。
# 超过 tp.STALE_S（0.3 秒）没更新：指令原地不动等它（不盲走），超过 STATE_STOP_S 才停
STATE_STOP_S = 1.0


class PolicyClient:
    """连 openpi 的模型服务（scripts/serve_policy.py）。和 openpi_client.WebsocketClientPolicy 一样的协议，
    只是加了超时：服务没开、网断了不会一直卡住。"""

    def __init__(self, addr):
        import websockets.sync.client as wsc
        uri = addr if addr.startswith("ws") else "ws://" + addr
        self.ws = wsc.connect(uri, compression=None, max_size=None, open_timeout=5)
        self.meta = msgpack_numpy.unpackb(self.ws.recv(timeout=10))
        self.packer = msgpack_numpy.Packer()

    def infer(self, obs, timeout=RESP_TIMEOUT_S):
        self.ws.send(self.packer.pack(obs))
        r = self.ws.recv(timeout=timeout)
        if isinstance(r, str):
            raise RuntimeError("模型服务报错：" + r[-300:])
        return msgpack_numpy.unpackb(r)

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def observe(twin, task):
    """拍一次观测，处理成和训练数据一模一样的格式。返回 (obs, 拍的时刻, 实际关节角 度)。"""
    st = twin.robot.read_state()
    if st is None or time.time() - st[0] > STATE_STOP_S:
        raise RuntimeError("读不到机械臂状态")
    imgs = {}
    for name, label, cam in (("scene", "第三视角", twin.scene.cam if twin.scene is not None else None),
                             ("wrist", "腕部", twin.wrist)):
        lat = getattr(cam, "latest", None) if cam is not None else None
        if lat is None or time.time() - lat[0] > STALE_IMG_S:
            raise RuntimeError(f"{label}相机没有画面")
        ok, buf = cv2.imencode(".jpg", lat[1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        imgs[name] = eco65_io.decode_bgr(buf.tobytes())
    g = getattr(twin.robot, "gripper", None)
    w = g.width() if g is not None else None
    width = float(np.clip(w, 0.0, 1.0)) if w is not None else (0.0 if twin.robot.gripper_closed else 1.0)
    q = np.asarray(st[1], float)
    obs = {"observation/image": eco65_io.scene_to_model(imgs["scene"]),
           "observation/wrist_image": eco65_io.wrist_to_model(imgs["wrist"]),
           "observation/state": np.r_[np.radians(q), width].astype(np.float32),
           "prompt": task}
    return obs, time.time(), q


def cams_problem(twin):
    for label, cam in (("第三视角", twin.scene.cam if twin.scene is not None else None), ("腕部", twin.wrist)):
        lat = getattr(cam, "latest", None) if cam is not None else None
        if lat is None or time.time() - lat[0] > STALE_IMG_S:
            return f"{label}相机没有画面（模型训练时两路相机都有，缺一路不能跑）"
    return None


class PolicyRunner:
    def __init__(self, twin):
        self.twin = twin
        self.running = False
        self.info = ""                      # 正在跑时给网页显示的一行
        self._result = self._error = None
        self._busy = False

    # ---- 不动机械臂，只问一次 ----
    def test(self, addr, task):
        why = cams_problem(self.twin)
        if why:
            return False, why
        try:
            t = time.time()
            client = PolicyClient(addr or DEFAULT_ADDR)
            t_conn = time.time() - t
        except Exception as e:
            return False, f"连不上模型服务 {addr}：{e}（台式机上先跑 bash ~/libw/eco65/serve.sh）"
        try:
            obs, _, q = observe(self.twin, task or DEFAULT_TASK)
            t = time.time()
            a = np.asarray(client.infer(obs, timeout=FIRST_TIMEOUT_S)["actions"], float)
            t1 = time.time() - t
            t = time.time()
            client.infer(obs)
            t2 = time.time() - t
        except Exception as e:
            return False, f"推理出错：{e}"
        finally:
            client.close()
        dq = np.degrees(a[:, :6]) - q[None]
        far = float(np.abs(dq).max())
        return True, (f"连上了（{t_conn * 1000:.0f} ms），推理 {t1 * 1000:.0f} ms（再问一次 {t2 * 1000:.0f} ms）。"
                      f"动作 {a.shape[0]} 步×{a.shape[1]}；1 秒后各关节离现在 {np.round(dq[-1], 1).tolist()}°"
                      f"（最多 {far:.1f}°），夹爪 {a[0, 6]:.2f}→{a[-1, 6]:.2f}（1 开 0 合）。机械臂没动")

    # ---- 后台问下一段 ----
    def _ask(self, client, task, img_dir, idx):
        try:
            obs, t_obs, _ = observe(self.twin, task)
            t = time.time()
            a = np.asarray(client.infer(obs)["actions"], float)
            ms = (time.time() - t) * 1000
            pair = np.hstack([obs["observation/image"], obs["observation/wrist_image"]])
            cv2.imwrite(os.path.join(img_dir, f"{idx:04d}.jpg"), pair[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._result = (t_obs, a, ms)
        except Exception as e:
            self._error = str(e) or type(e).__name__
        finally:
            self._busy = False

    def run(self, cfg, on_done):
        """cfg: addr, task, replan（一段里执行几步再问）, max_s（最长秒数）, auto_end（夹着东西松手后自动结束，命令行演示用）。
        在后台线程里跑，结束调 on_done(状态, 日志文件夹)。"""
        twin, robot = self.twin, self.twin.robot
        addr, task = cfg.get("addr") or DEFAULT_ADDR, cfg.get("task") or DEFAULT_TASK
        replan = int(np.clip(cfg.get("replan", 8), 1, H))
        max_s = float(np.clip(cfg.get("max_s", 40), 5, 300))
        auto_end = bool(cfg.get("auto_end"))
        raw = bool(cfg.get("raw"))       # 解除限制（只保留防撞桌、防自撞、关节限位这类保护机械臂的检查）
        joint_speed = RAW_JOINT_SPEED if raw else JOINT_SPEED[bool(twin.real)]
        grip_min_s = RAW_GRIP_MIN_S if raw else GRIP_MIN_S
        twin.executing, twin.stop_flag, twin.jog_on = True, False, False
        self.running, self._result, self._error, self._busy = True, None, None, False
        twin.status = self.info = "模型控制：连模型服务"
        log_dir = os.path.join(tp.LOG_DIR, time.strftime("policy_%Y%m%d_%H%M%S"))
        img_dir = os.path.join(log_dir, "obs")
        os.makedirs(img_dir, exist_ok=True)
        f_log = open(os.path.join(log_dir, "control.csv"), "w", newline="", buffering=1)
        log = csv.writer(f_log)
        log.writerow(["t", "chunk", *[f"tgt_j{j}" for j in range(1, 7)], *[f"cmd_j{j}" for j in range(1, 7)], "grip_val",
                      "gripper_closed", *[f"act_j{j}" for j in range(1, 7)], "act_cx", "act_cy", "act_cz", "err_mm"])
        chunks, reason, client = [], None, None
        note = "（解除了限制：只保留防撞桌、防自撞、关节限位）" if raw else ""
        n_grip, n_stale, n_floor, dt = 0, 0, 0, 1.0 / RATE_HZ
        t_start = time.time()
        try:
            client = PolicyClient(addr)
            # 第一段：机械臂还不动，先问一次（服务刚起来时要编译十几秒）
            twin.status = self.info = "模型控制：第一次推理（服务刚起来要编译，可能十几秒）"
            obs, t_obs, q0 = observe(twin, task)
            off_home = float(np.abs(q0 - np.asarray(tp.HOME_Q, float)).max())
            if off_home > HOME_WARN_DEG:
                note += f"（起点离初始位姿差 {off_home:.0f}°，训练时都从初始位姿出发）"
            t = time.time()
            A = np.asarray(client.infer(obs, timeout=FIRST_TIMEOUT_S)["actions"], float)
            lat_ms = (time.time() - t) * 1000
            cv2.imwrite(os.path.join(img_dir, "0000.jpg"),
                        np.hstack([obs["observation/image"], obs["observation/wrist_image"]])[..., ::-1])
            t0c = time.time()            # 第一段从“现在”开始用（问的时候机械臂没动，拍画面的时刻不重要）
            chunks.append((0.0, lat_ms, A))
            lat_est = min(lat_ms, 300.0) / 1000
            q_cmd = q0.copy()
            r_prev = twin.flange_r(q_cmd)
            c_prev, cmd_speed, bad, chunk_id, held, t_grip, n_stale = None, 0.0, 0, 0, 0.0, 0.0, 0
            z_grasp, n_floor = None, 0       # 合夹爪那一刻的抓取中心高度；高度下限拦住了多少拍
            t_release = None                 # 夹着东西松手的时刻（auto_end 用）
            c_now = twin.fk_center(q_cmd)[0]
            t_start = time.time()
            twin.status = "模型控制中" + note
            while True:
                tick = time.time()
                if twin.stop_flag:
                    reason = "手动停止"
                    break
                if tick - t_start > max_s:
                    note += f"（到最长时间 {max_s:.0f} 秒了）"
                    break
                if auto_end and t_release is not None and tick - t_release > AUTO_END_S:
                    note += "（放下后自动结束）"
                    break
                if self._error:
                    reason = f"问模型时出错：{self._error}"
                    break
                st = robot.read_state()
                age = time.time() - st[0] if st is not None else 1e9
                if age > STATE_STOP_S:
                    reason = f"读不到实际状态（{min(age, 99):.1f} 秒没更新）"
                    break
                if age > tp.STALE_S:              # 状态卡了一下：这一拍不发新指令，原地等它恢复
                    n_stale += 1
                    time.sleep(max(0.0, dt - (time.time() - tick)))
                    continue
                if self._result is not None:      # 新的一段回来了：按拍画面的时刻对齐接着用
                    t_obs, A, ms = self._result
                    self._result = None
                    t0c = t_obs
                    chunk_id += 1
                    chunks.append((t_obs - t_start, ms, A))
                    lat_est = 0.7 * lat_est + 0.3 * ms / 1000
                el = tick - t0c
                if not self._busy and el >= replan / FPS - lat_est - 0.03:   # 这一段快用到第 replan 步了：去问下一段
                    self._busy = True
                    threading.Thread(target=self._ask, args=(client, task, img_dir, len(chunks)), daemon=True).start()
                k = el * FPS
                if k > H - 1 + NO_CHUNK_S * FPS:
                    reason = "模型服务太久没回下一段动作"
                    break
                kk = min(max(k, 0.0), H - 1.0)
                i0 = int(kk)
                i1, f = min(i0 + 1, H - 1), kk - int(kk)
                a = A[i0] * (1 - f) + A[i1] * f
                q_tgt, g_val = np.degrees(a[:6]), float(a[6])
                gap = np.abs(q_tgt - q_cmd)
                if gap.max() > MAX_JUMP_DEG and not raw:
                    j = int(np.argmax(gap))
                    reason = f"模型给的目标离现在的指令差 {gap[j]:.0f}°（J{j + 1}），没跟"
                    break
                step = joint_speed * dt
                q_new = q_cmd + np.clip(q_tgt - q_cmd, -step, step)
                # 手指按张开时的长度算（ext=0），和执行轨迹时的兜底检查一样：合上以后指尖往下多伸 2~3 cm，按合上算的话
                # 在示范的抓取高度（中心离桌 53 mm）一合夹爪就被误叫停（2026-11-06 离线测出来的）；抓取高度靠 CENTER_BOX 的下限管
                why = twin.check_joints(q_new, r_prev, ext=0.0)
                if why:
                    reason = f"安全检查拦住：{why}"
                    break
                c_new, _ = twin.fk_center(q_new)
                # 高度下限：张开时 48 mm；夹着东西时不低于合夹爪那一刻的高度（再留 2 mm）
                floor, wait_s = Z_FLOOR, Z_HOLD_S
                if robot.gripper_closed and z_grasp is not None:
                    floor, wait_s = max(Z_FLOOR, z_grasp - HOLD_TOL), HOLD_WAIT_S
                if not raw and c_new[2] < floor and c_new[2] < c_now[2]:   # 要往下限以下走：这一拍不动，原地等
                    q_new, c_new = q_cmd, c_now
                    held += dt
                    n_floor += 1
                    if held > wait_s:
                        reason = (f"模型一直要把抓取中心降到 {floor * 1000:.0f} mm 以下（"
                                  + ("夹着东西时不能比夹起来的高度还低" if floor > Z_FLOOR else "比示范的抓取高度还低")
                                  + f"），等了 {wait_s:.0f} 秒停下")
                        break
                else:
                    held = 0.0
                out = [] if raw else [n for n, (lo, hi), v in zip("xyz", CENTER_BOX, c_new) if not lo <= v <= hi]
                if out:
                    reason = (f"抓取中心要出工作范围了（{np.round(c_new * 1000).astype(int).tolist()} mm，"
                              f"{'、'.join(out)} 超了）")
                    break
                with twin.lock:
                    robot.send_joints(q_new)
                twin.rec_cmd = (list(map(float, q_new)), robot.gripper_closed, list(map(float, q_new)))
                q_cmd, r_prev, c_now = q_new, twin.flange_r(q_new), c_new
                want = robot.gripper_closed
                if g_val < GRIP_CLOSE_BELOW:
                    want = True
                elif g_val > GRIP_OPEN_ABOVE:
                    want = False
                if want != robot.gripper_closed and tick - t_grip >= grip_min_s:
                    if not want and robot.gripper_closed and tick - t_grip > GRIP_SETTLE_S:
                        t_release = tick             # 合着超过 GRIP_SETTLE_S（没被判成空夹）后张开＝放下了
                    with twin.lock:
                        robot.set_gripper(want)
                    n_grip, t_grip = n_grip + 1, tick
                    # 合夹爪：记下这一刻实际的抓取中心高度（夹着时的高度下限）；张开：清掉
                    z_grasp = float(tp.center_of(st[2])[2]) if want else None
                g = getattr(robot, "gripper", None)
                if not raw and robot.gripper_closed and g is not None and tick - t_grip > GRIP_SETTLE_S:
                    w = g.width()
                    if w is not None and w < EMPTY_W:
                        reason = f"夹空了：合上 {tick - t_grip:.1f} 秒后开口 {w:.2f}（夹着方块约 0.22），没夹到方块"
                        break
                # 和执行轨迹一样盯着：读不到状态、跟不上指令、实际手臂越线就停
                if c_prev is not None:
                    cmd_speed += min(1.0, dt / tp.TRACK_LAG_S) * (np.linalg.norm(c_new - c_prev) * 1000 / dt - cmd_speed)
                c_prev = c_new
                act_c = tp.center_of(st[2])
                err = float(np.linalg.norm(act_c - c_new) * 1000)
                bad = bad + 1 if err > tp.TRACK_ERR_MM + tp.TRACK_LAG_S * cmd_speed else 0
                if bad >= tp.TRACK_ERR_FRAMES:
                    reason = f"实际位置跟不上指令（偏差 {err:.0f} mm）"
                    break
                am = robot.actual_margin()
                if am is not None and am[0] < -0.02:
                    reason = am[1]
                    break
                log.writerow([f"{tick - t_start:.3f}", chunk_id, *[f"{v:.3f}" for v in q_tgt], *[f"{v:.3f}" for v in q_new],
                              f"{g_val:.3f}", int(robot.gripper_closed), *[f"{v:.3f}" for v in st[1]],
                              *[f"{v:.5f}" for v in act_c], f"{err:.1f}"])
                twin.progress = (int(tick - t_start), int(max_s))
                self.info = (f"模型控制中 {tick - t_start:4.1f}/{max_s:.0f} 秒 · 第 {chunk_id + 1} 段 · 推理 {lat_est * 1000:.0f} ms · "
                             f"夹爪{'合' if robot.gripper_closed else '开'}（模型 {g_val:.2f}）· 抓取中心 "
                             f"{np.round(c_new * 1000).astype(int).tolist()} mm" + note)
                time.sleep(max(0.0, dt - (time.time() - tick)))
        except Exception as e:
            reason = f"出错：{e}"
        finally:
            with twin.lock:
                try:
                    robot.stop()                  # 停在原地（透传目标换成实际位置）
                except Exception:
                    pass
            if client is not None:
                for _ in range(30):               # 等后台那次推理结束再关连接
                    if not self._busy:
                        break
                    time.sleep(0.1)
                client.close()
            if chunks:
                np.savez(os.path.join(log_dir, "chunks.npz"), t_obs=np.array([c[0] for c in chunks]),
                         infer_ms=np.array([c[1] for c in chunks]), actions=np.stack([c[2] for c in chunks]),
                         task=task, replan=replan, addr=addr, raw=raw)
            f_log.close()
            twin.rec_cmd = None
            twin.executing = self.running = False
            used = time.time() - t_start
            twin.status = (f"模型控制已停止：{reason}" if reason else "模型控制结束") + \
                f"（{used:.1f} 秒、{len(chunks)} 段、夹爪动了 {n_grip} 次" + \
                (f"、状态卡住原地等了 {n_stale * dt:.1f} 秒" if n_stale else "") + \
                (f"、高度下限拦住 {n_floor * dt:.1f} 秒" if n_floor else "") + "）" + note
            self.info = ""
            on_done(twin.status, log_dir)
