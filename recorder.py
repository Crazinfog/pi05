"""录示范数据：真机执行一条轨迹时（或者手动点开始 / 停止），按 15 Hz 记下两台相机的彩色图、机械臂的实际状态、
夹爪开口和当时发给机械臂的指令，每条示范存一个 HDF5 文件（episode_000000.hdf5、episode_000001.hdf5 …）。
以后在 4090 上仿照 openpi 的 examples/aloha_real/convert_aloha_data_to_lerobot.py 转成 LeRobot 格式再微调。

每个文件里（关节角是弧度、位置是米；夹爪 0 闭合 ~ 1 张开）：
  /observations/qpos              (T, 6)  float32  实际关节角
  /observations/gripper           (T,)    float32  实际夹爪开口（读不到时按指令）
  /observations/ee_pose           (T, 6)  float32  实际法兰位姿（xyz + 外旋 xyz 欧拉角，和 SDK 一样）
  /observations/images/scene      (T,)    JPEG 字节  第三视角（Gemini 335，640×480 BGR 编码）
  /observations/images/wrist      (T,)    JPEG 字节  腕部（Gemini 305）
  /observations/images_t/<相机>   (T,)    float64  这一帧图像是相机什么时候拍的（相对开始录的秒数）
  /observations/depth/<相机>      (T,)    PNG 字节   16 位深度，单位毫米（勾了“同时存深度图”才有）
  /action                         (T, 7)  float32  指令关节角(6) + 夹爪指令(1 张开 / 0 合上)；
                                                   没有指令时（手动录、拖动示教）就填实际值
  /action_sent                    (T, 6)  float32  实际发给机械臂的关节角（执行时小扰动开着时＝干净轨迹＋扰动，否则和 /action 前 6 列一样）
  /timestamp                      (T,)    float64  从开始录起的秒数
  /idle                           (T,)    uint8    1＝这一帧几乎不动（手臂 <1 cm/s、关节 <2°/s、夹爪开口和指令都没变）
  /keep                           (T,)    uint8    转换时建议留下的帧：不是静止帧，或者在夹爪指令变化前后 KEEP_AROUND_S 以内
                                                   （停住不动的段会让策略卡住：OpenVLA 删了无动作帧、OPTIMUS 缩短了停住的段）
  attrs：task（给模型的任务描述）、fps、created、source（execute / manual）、cameras、calib_T、home_q、ok、
         success（存下时就是 1；用户点了“失败”的写 0 并挪进 _废弃/；2026-11-09 前要用户每条标，-1＝还没标）、exec_noise_mm 等

LIBERO 的示范是从固定初始姿态开始、任务一完成就结束、不回初始位姿，所以执行时从第 1 帧开始录，最后一下是松手的
录到松手后 1.2 秒（官方数据松手后约 1.1 秒就结束），撤出、回初始照样执行（复位下一条）但不录。
查看某一条录了什么：python view_episode.py [文件]（打印内容，导出视频和缩略图到 logs/preview/）。
"""
import json
import os
import re
import shutil
import threading
import time

import cv2
import numpy as np

FPS = 15                                   # 和相机帧率一样（两台都是 640×480@15）
JPEG_QUALITY = 90
IDLE_LIN, IDLE_JOINT, IDLE_GRIP = 0.01, np.radians(2.0), 0.01   # 静止帧：手臂 <1 cm/s、关节 <2°/s、夹爪开口变化 <1%
KEEP_AROUND_S = 0.3                        # 夹爪指令变化前后这么久以内的静止帧照样留（夹爪开合本身）
EP_RE = re.compile(r"^episode_(\d{6})\.hdf5$")
TRASH = "_废弃"                            # “删除上一条”其实是挪到这个子文件夹（想要回来还能拿回来）
CAM_LABEL = {"scene": "第三视角", "wrist": "腕部"}


def cam_fresh(cam, max_age=1.0):
    """这台相机最近 max_age 秒里有没有新画面（cameras.DepthCamera 有 fresh；别的对象按 latest 的时刻看）。"""
    if cam is None:
        return False
    if hasattr(cam, "fresh"):
        return cam.fresh(max_age)
    lat = getattr(cam, "latest", None)
    return lat is not None and time.time() - lat[0] < max_age


def expand(path):
    return os.path.abspath(os.path.expanduser(str(path or "").strip() or "~"))


def episodes_in(folder):
    """文件夹里已经有的示范文件名（排好序）。"""
    try:
        return sorted(f for f in os.listdir(folder) if EP_RE.match(f))
    except OSError:
        return []


def next_index(folder):
    eps = episodes_in(folder)
    return int(EP_RE.match(eps[-1]).group(1)) + 1 if eps else 0


def thumb(frames, width=192):
    """这一条开头的小图（第三视角，没有就用腕部），JPEG 字节；每条方块摆的地方不一样，看一眼就知道是哪一条。"""
    for name in ("scene", "wrist"):
        buf = frames[0]["img"].get(name) if frames else None
        if buf:
            im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
                ok, out = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    return out.tobytes()
    return None


def count_success(folder):
    """文件夹里标成成功（success=1）的有几条。"""
    import h5py
    k = 0
    for f in episodes_in(folder):
        try:
            with h5py.File(os.path.join(folder, f), "r") as h:
                k += int(h.attrs.get("success", -1)) == 1
        except OSError:
            pass
    return k


def list_dir(path):
    """网页上选存储路径用：这个文件夹下面的子文件夹、里面已经有几条示范、所在磁盘还剩多少空间。"""
    path = expand(path)
    while not os.path.isdir(path) and os.path.dirname(path) != path:   # 还没建的路径：往上找到存在的那一层
        path = os.path.dirname(path)
    try:
        dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith(".") and d != TRASH)
    except OSError as e:
        return {"path": path, "parent": os.path.dirname(path), "dirs": [], "error": f"打不开：{e.strerror}"}
    free = shutil.disk_usage(path).free / 1e9
    return {"path": path, "parent": os.path.dirname(path), "dirs": dirs, "episodes": len(episodes_in(path)),
            "free_gb": round(free, 1), "writable": os.access(path, os.W_OK)}


def folder_info(folder):
    """存储路径现在的情况（网页上显示）：存在不存在、有几条、剩多少空间。"""
    folder = expand(folder)
    probe = folder
    while not os.path.isdir(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free / 1e9
    exists = os.path.isdir(folder)
    return {"folder": folder, "exists": exists, "episodes": len(episodes_in(folder)) if exists else 0,
            "next": next_index(folder) if exists else 0, "free_gb": round(free, 1),
            "writable": os.access(folder if exists else probe, os.W_OK)}


def idle_mask(frames):
    """每帧是不是“几乎不动”，以及转换时建议留哪些帧。返回 (idle, keep) 两个布尔数组。"""
    pos = np.array([f["pose"][:3] for f in frames])
    q = np.array([f["q"] for f in frames])
    grip = np.array([f["grip"] for f in frames])
    cmd = np.array([f["act"][6] for f in frames])
    n = len(frames)
    lin = np.r_[np.inf, np.linalg.norm(np.diff(pos, axis=0), axis=1) * FPS]
    jnt = np.r_[np.inf, np.max(np.abs(np.diff(q, axis=0)), axis=1) * FPS]
    dg = np.r_[np.inf, np.abs(np.diff(grip))]
    dc = np.r_[True, np.diff(cmd) != 0]
    idle = (lin < IDLE_LIN) & (jnt < IDLE_JOINT) & (dg < IDLE_GRIP) & ~dc
    keep = ~idle
    w = int(round(KEEP_AROUND_S * FPS))
    for i in np.flatnonzero(dc[1:]) + 1:                 # 夹爪指令变化那一帧前后
        keep[max(0, i - w):min(n, i + w + 1)] = True
    keep[0] = True                                        # 第一帧留着（起点）
    return idle, keep


class Recorder:
    """后台线程按 FPS 取一帧：两台相机最新的画面 + 机械臂实际状态 + 指令。stop(save=True) 时一次写成 HDF5。

    robot：teleop.RealmanRobot（read_state、gripper_closed、gripper）；cams：{"scene": DepthCamera, "wrist": DepthCamera}；
    command：一个函数，返回 (指令关节角 度, 夹爪指令合上?) 或 None（没有在发指令）。"""

    def __init__(self, robot, cams, command):
        self.robot, self.cams, self.command = robot, cams, command
        self.running = False
        self.frames, self.meta, self.thread, self.t0 = [], {}, None, 0.0
        self.lock = threading.Lock()
        self.last_saved = None                 # 最近这一次录的、存下了的那一条（点“失败”时挪进 _废弃 的就是它）
        self.last_thumb = None                 # 它开头的第三视角小图（JPEG），网页左上角显示，一眼认出是哪一条

    def start(self, folder, task, depth=False, source="execute", meta=None):
        """开始录。返回 None（开始了）或者为什么开不了。"""
        if self.running:
            return "已经在录了"
        folder = expand(folder)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            return f"存储路径建不了（{e.strerror}）：{folder}"
        if not os.access(folder, os.W_OK):
            return f"存储路径没有写权限：{folder}"
        if shutil.disk_usage(folder).free < 2e9:
            return f"存储路径所在的磁盘只剩 {shutil.disk_usage(folder).free / 1e9:.1f} GB，先清一清"
        if self.robot.read_state() is None:
            return "还没读到机械臂状态"
        bad = [n for n, cam in self.cams.items() if not cam_fresh(cam)]
        if bad:      # 两路画面都要有（π0.5 训练用第三视角＋腕部），缺一路录下来也用不了
            return f"{'、'.join(CAM_LABEL.get(n, n) for n in bad)}相机没有画面，没开始录"
        self.folder, self.depth = folder, bool(depth)
        # 开始新的一次录制：“失败”只管最近这一次。这次没存下来（中途停了）时，不能让“失败”去丢上一条好的（2026-11-09）
        self.last_saved, self.last_thumb = None, None
        self.meta = {"task": task or "", "source": source, **(meta or {})}
        self.frames, self.t0, self.running = [], time.time(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return None

    def _grab(self):
        st = self.robot.read_state()
        if st is None:
            return None
        now = time.time() - self.t0
        g = getattr(self.robot, "gripper", None)
        w = g.width() if g is not None else None
        width = float(np.clip(w, 0.0, 1.0)) if w is not None else (0.0 if self.robot.gripper_closed else 1.0)
        cmd = self.command()        # (录成动作的关节角, 夹爪合上?, [实际发出去的关节角]) 或 None
        if cmd is not None:
            q_cmd, closed_cmd = cmd[0], cmd[1]
            q_sent = cmd[2] if len(cmd) > 2 else q_cmd
        else:
            q_cmd, closed_cmd = st[1], self.robot.gripper_closed
            q_sent = q_cmd
        fr = {"t": now, "q": np.radians(np.asarray(st[1], float)), "pose": np.asarray(st[2][:6], float), "grip": width,
              "act": np.r_[np.radians(np.asarray(q_cmd, float)), 0.0 if closed_cmd else 1.0],
              "sent": np.radians(np.asarray(q_sent, float)), "img": {}, "img_t": {}, "depth": {}}
        for name, cam in self.cams.items():
            lat = cam.latest if cam is not None else None
            if lat is None or time.time() - lat[0] > 0.5:     # 半秒没更新就算这一帧没画面（断线时 latest 停在旧画面上）
                continue
            ok, buf = cv2.imencode(".jpg", lat[1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                fr["img"][name] = buf.tobytes()
                fr["img_t"][name] = lat[0] - self.t0
            if self.depth:
                ok, dbuf = cv2.imencode(".png", np.clip(lat[2] * 1000.0, 0, 65535).astype(np.uint16))
                if ok:
                    fr["depth"][name] = dbuf.tobytes()
        return fr

    def _loop(self):
        period, nxt = 1.0 / FPS, time.time()
        while self.running:
            fr = self._grab()
            if fr is not None:
                with self.lock:
                    self.frames.append(fr)
            nxt += period
            time.sleep(max(0.0, nxt - time.time()))

    def stop(self, save=True, ok=True, extra=None, background=None):
        """停止录。save：存不存（中途停了就不存）。返回 (存下的文件路径或 None, 说明)。
        background 给了回调：马上停止取帧，在后台线程里存文件，存完调 background((路径, 说明))，返回那个线程
        （执行轨迹时松手后就停录，机械臂还在撤出，写文件要 1~4 秒，不能卡住发指令）。"""
        if not self.running:
            return None, "没在录"
        self.running = False
        if background is not None:
            th = threading.Thread(target=lambda: background(self._finish(save, ok, extra)), daemon=True)
            th.start()
            return th
        return self._finish(save, ok, extra)

    def _finish(self, save, ok, extra):
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self.lock:
            frames, self.frames = self.frames, []
        if not save:
            return None, f"没存（录了 {len(frames) / FPS:.1f} 秒，中途停了）"
        if len(frames) < 5:
            return None, "录到的帧太少，没存"
        lost = {n: sum(n not in f["img"] for f in frames) for n in self.cams}
        lost = {n: k for n, k in lost.items() if k}
        if lost:     # 录到一半相机断了：存下来也少一路画面，不存，重录
            return None, ("录的时候" + "、".join(f"{CAM_LABEL.get(n, n)}相机有 {k} 帧没画面" for n, k in lost.items())
                          + "（线松了？），这条没存，重录一条")
        import h5py
        idx = next_index(self.folder)
        path = os.path.join(self.folder, f"episode_{idx:06d}.hdf5")
        tmp = path + ".tmp"
        names = [n for n in self.cams if all(n in f["img"] for f in frames)]
        missing = [n for n in self.cams if n not in names]
        vlen = h5py.vlen_dtype(np.uint8)
        with h5py.File(tmp, "w") as h:
            obs = h.create_group("observations")
            obs["qpos"] = np.stack([f["q"] for f in frames]).astype(np.float32)
            obs["gripper"] = np.array([f["grip"] for f in frames], np.float32)
            obs["ee_pose"] = np.stack([f["pose"] for f in frames]).astype(np.float32)
            h["action"] = np.stack([f["act"] for f in frames]).astype(np.float32)
            h["action_sent"] = np.stack([f["sent"] for f in frames]).astype(np.float32)
            h["timestamp"] = np.array([f["t"] for f in frames], np.float64)
            idle, keep = idle_mask(frames)
            h["idle"], h["keep"] = idle.astype(np.uint8), keep.astype(np.uint8)
            for group, key in (("images", "img"), ("depth", "depth")):
                if key == "depth" and not self.depth:
                    continue
                g = obs.create_group(group)
                for n in names:
                    ds = g.create_dataset(n, (len(frames),), dtype=vlen)
                    for i, f in enumerate(frames):
                        ds[i] = np.frombuffer(f[key][n], np.uint8)
            ts = obs.create_group("images_t")
            for n in names:
                ts[n] = np.array([f["img_t"][n] for f in frames], np.float64)
            attrs = {**self.meta, **(extra or {}), "fps": FPS, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "cameras": json.dumps(names), "ok": bool(ok), "frames": len(frames),
                     "joint_unit": "rad", "gripper": "0 闭合 ~ 1 张开", "action": "指令关节角(rad)×6 + 夹爪指令(1 张开/0 合上)",
                     "action_sent": "实际发给机械臂的关节角(rad)×6：执行时小扰动开着时＝干净轨迹+扰动，否则和 action 一样",
                     "success": 1}      # 默认算成功（2026-11-09 用户：别每条都要点成功，失败了点“失败”就行）
            for k, v in attrs.items():
                h.attrs[k] = v if isinstance(v, (str, bool, int, float, np.ndarray)) else json.dumps(v)
        os.replace(tmp, path)
        self.last_saved = path
        self.last_thumb = thumb(frames)
        mb = os.path.getsize(path) / 1e6
        note = f"，没录到{'、'.join(missing)}相机" if missing else ""
        idle, keep = idle_mask(frames)
        return path, (f"存好了第 {idx} 条：{len(frames) / FPS:.1f} 秒、{len(frames)} 帧、{mb:.0f} MB{note}；"
                      f"静止帧 {idle.mean() * 100:.0f}%（转换时会裁掉 {(~keep).sum()} 帧）")

    def _set_success(self, path, value):
        import h5py
        with h5py.File(path, "r+") as h:
            h.attrs["success"] = int(value)

    def _trash(self, path):
        """挪进存储路径下的 _废弃 文件夹（不真删，想要回来还能拿回来）。"""
        folder = os.path.dirname(path)
        os.makedirs(os.path.join(folder, TRASH), exist_ok=True)
        os.replace(path, os.path.join(folder, TRASH, time.strftime("%Y%m%d_%H%M%S_") + os.path.basename(path)))
        if self.last_saved == path:
            self.last_saved = None
        return (f"{os.path.basename(path)} 标成失败，挪到了 {TRASH}/ 里（不算进数据集了，"
                f"下一条还用这个编号）")

    def discard_last(self):
        """刚录的那一条没做成（只留成功的示范，MimicGen 等生成数据的做法都是这样）：写 success=0、挪进 _废弃。
        存下时默认就算成功，不用再标（2026-11-09 前每条都要点“成功 / 失败”，没标不让录下一条）。"""
        if not self.last_saved or not os.path.exists(self.last_saved):
            return "没有刚录的那一条（只能撤掉这次开网页以来录的最后一条）"
        self._set_success(self.last_saved, 0)
        return self._trash(self.last_saved)
