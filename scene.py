"""第三视角相机（Gemini 335）：实时画面、棋盘格外参标定、点云、点一下分割物体。

外参标定（和 BEVFusion 里用标定板标相机外参一个思路）：
  1. 棋盘格平放在桌上、机械臂够得到的地方，整块在画面里，机械臂挪开 → find_board() 识别角点，标出四个外角 ①②③④；
  2. 夹爪合上、竖直朝下，用拖动示教把指尖依次碰到 ①②③④ → touch() 记下每个角在底座坐标里的位置；
  3. calibrate()（solve_extrinsic）：桌面的倾斜和高度用深度图里棋盘格周围那片桌面拟合的平面，角点的三维位置用像素射线和
     这个平面求交，水平位置和朝向用四个角点和碰点在水平面里配准；最后 refine_with_robot() 拿点云里的机械臂和 URDF 模型
     做 ICP 精修（机械臂又大又在画面中间，比画面角上六七十个像素的小棋盘格准得多），存到 scene_calib.npz。
     2026-11-02 以前用 PnP + 带尺度配准：格子边长按碰点量，碰点每个角往外偏 5 mm 边长就大 7%，PnP 的距离跟着远 7%
     （1.1 m 处 8 cm），点云整体浮在桌面上方、和模型对不上（棋盘格那块深度点离板面 9.8 cm）。
     同一帧上比较：旧外参机械臂点离模型中位 25 mm；新的桌面平面 + 碰点 12 mm；再用机械臂 ICP 精修 3 mm（87% 在 1 cm 内）。
点云：深度图（已对齐到彩色）按内参和畸变反投影，用外参变到底座坐标，裁到工作区，隔一个像素取一个点。
分割：点一下物体 → 种子点附近拟合桌面、去掉桌面，从种子点长出一团连在一起的点，xy 平面上 PCA 拟合一个带朝向的长方体。
"""
import json
import os
import threading
import time
import warnings

import cv2
import numpy as np
from scipy.spatial import cKDTree

import cameras

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(HERE, "scene_calib.npz")
# 工作区（夹爪竖直抓 3.3 cm 方块、预抓取 8 cm、抬起 10 cm 都够得着的桌面范围，底座坐标），离线算好存这里；
# 相机画面上可以叠加显示（网页“相机画面”卡片的“显示工作区”开关）。2026-11-03 按用户要求一直到鼠标垫那边
WORK_AREA_PATH = os.path.join(HERE, "work_area.json")


def load_work_area():
    try:
        with open(WORK_AREA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def draw_work_area(img, wa, T, K, dist):
    """在相机画面上画工作区：黄线是边界、细线和字母是分区、红色是初始位姿时手臂挡住的地方、白线是两张桌子的接缝。"""
    Ti = np.linalg.inv(T)
    rvec, _ = cv2.Rodrigues(Ti[:3, :3])

    def pr(pts):
        P = np.c_[np.asarray(pts, float), np.zeros(len(pts))]
        uv, _ = cv2.projectPoints(P, rvec, Ti[:3, 3], K, dist)
        return uv.reshape(-1, 2).astype(np.int32)
    inside = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(inside, [pr(wa["polygon"])], 1)
    over = img.copy()
    for hp in wa.get("hidden", []):
        cv2.fillPoly(over, [pr(hp)], (60, 60, 230))
    img[:] = cv2.addWeighted(over, 0.4, img, 0.6, 0)
    lines = np.zeros(img.shape[:2], np.uint8)
    for z in wa.get("zones", []):
        cv2.polylines(lines, [pr(z["poly"])], True, 1, 1)
    img[(lines > 0) & (inside > 0)] = (0, 220, 255)
    if wa.get("seam_x") is not None:
        a, b = pr([[wa["seam_x"], wa["seam_y"][0]], [wa["seam_x"], wa["seam_y"][1]]])
        seam = np.zeros(img.shape[:2], np.uint8)
        cv2.line(seam, tuple(map(int, a)), tuple(map(int, b)), 1, 2)
        img[(seam > 0) & (inside > 0)] = (255, 255, 255)
    cv2.polylines(img, [pr(wa["polygon"])], True, (0, 220, 255), 2, cv2.LINE_AA)
    for z in wa.get("zones", []):
        c = pr(z["poly"]).mean(0).astype(int)
        cv2.putText(img, z["label"], (int(c[0]) - 8, int(c[1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, z["label"], (int(c[0]) - 8, int(c[1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
ROBOT_DIR = os.path.join(HERE, "twin_web", "robot")   # URDF 和 STL（机械臂模型精修外参用）
_ROBOT = None
CLOUD_STRIDE = 2                       # 点云隔几个像素取一个点（0.8 m 处一个像素约 1.7 mm）
WORKSPACE = np.array([[-0.9, 0.9], [-0.9, 0.9], [-0.03, 0.7]])    # 点云只留底座坐标里这个范围（米）：底座周围 0.9 m，
                                                                  # 干活区域朝哪边见 teleop.HOME_YAW_DEG
TABLE_GAP = 0.008                      # 离桌面不到这么高的点当成桌面
GROW_R = 0.012                         # 分割时相邻点距离不超过这么多就算连在一起
COLOR_TOL = 40.0                       # 分割时颜色容差（Lab 距离，亮度按一半算）：和点到的颜色差这么多以内、而且比桌面色更近，才算同一个物体
                                       # （点云的颜色噪声不小：MJPG 压缩 + 深度对齐到彩色有错位，同一个红方块的点 Lab 能差 30 左右）
TABLE_COLOR_MIN = 15.0                 # 点到的颜色和旁边桌面颜色差不到这么多：颜色分不开，只按形状长
MAX_OBJ_R = 0.20                       # 物体最多从点到的地方往外长 20 cm（再大也夹不住）
MIN_COLOR_PTS = 8                      # 按颜色长出至少这么多点才用（否则退回按形状长）
EXPAND_R, EXPAND_TOL, EXPAND_MIN_H = 0.015, 1.5, 0.003   # 按颜色长完再往外扩：1.5 cm 以内、颜色放宽到 1.5 倍容差、离桌面 3 mm 以上
ROBOT_PAD = 0.015                      # 离机械臂模型表面 1.5 cm 以内的点当成机械臂自己
TOUCH_LABELS = ["①", "②", "③", "④"]


class Scene:
    def __init__(self, cam=None):
        self.cam = cam if cam is not None else cameras.open_cameras(["scene"]).get("scene")   # 外面开好的可以直接给
        if self.cam is None:
            raise RuntimeError("没找到第三视角相机（Gemini 335）")
        self.K = self.cam.K
        self.dist = self.cam.dist
        self.T = None                  # 外参：相机坐标 → 底座坐标（4x4）
        self.calib_info = None
        self._rays = {}                # 每个像素去畸变后的射线 (x/z, y/z)，按 (高, 宽, 间隔) 缓存
        if os.path.exists(CALIB_PATH):
            d = np.load(CALIB_PATH)
            self.T = d["T"]
            self.calib_info = {k: d[k].item() for k in CALIB_KEYS if k in d.files}
        self.board = None              # 识别到的棋盘格：dict(size, corners, outer, image)
        self.touched = {}              # 碰过的外角：标签 → 底座坐标
        self.cloud = None              # 最近一次发给网页的点云：(xyz, rgb)
        self.lock = threading.Lock()

    def frame(self):
        st = self.cam.latest
        return None if st is None else (st[1], st[2])

    # ---------------- 实时画面 ----------------
    def jpeg(self, quality=70, zones=False):
        """当前彩色图（JPEG 字节），标定时叠加识别到的棋盘格角点和 ①②③④；zones：叠加工作区（要先标定过外参）。"""
        f = self.frame()
        if f is None:
            return None
        img = f[0].copy()
        if zones and self.T is not None:
            wa = load_work_area()
            if wa:
                draw_work_area(img, wa, self.T, self.K, self.dist)
        if self.board is not None:
            cv2.drawChessboardCorners(img, self.board["size"], self.board["corners"].reshape(-1, 1, 2).astype(np.float32), True)
            for lab, i in zip(TOUCH_LABELS, self.board["outer"]):
                p = tuple(int(v) for v in self.board["corners"][i])
                color = (0, 200, 0) if lab in self.touched else (0, 0, 255)
                cv2.circle(img, p, 9, color, 2)
                cv2.putText(img, str(TOUCH_LABELS.index(lab) + 1), (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ---------------- 棋盘格外参标定 ----------------
    def find_board(self):
        """在当前画面里找棋盘格（内角点数自动试，大的先试）。返回说明文字。"""
        f = self.frame()
        if f is None:
            return "相机还没有画面"
        gray = cv2.cvtColor(f[0], cv2.COLOR_BGR2GRAY)
        # 角点多的先试：小规格会先在大板的一部分上对上（2026-10-28 先试列数大的，结果认成了小板的 12×4 一截）
        sizes = [self.board["size"]] if self.board else []
        sizes += sorted(((w, h) for w in range(14, 3, -1) for h in range(w, 3, -1)), key=lambda s: -s[0] * s[1])
        for size in sizes:
            ok, c = cv2.findChessboardCornersSB(gray, size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ok:
                c = c.reshape(-1, 2).astype(np.float64)
                w, h = size
                self.board = {"size": size, "corners": c, "outer": [0, w - 1, w * h - 1, w * (h - 1)]}
                self.touched = {}
                return (f"找到棋盘格：内角点 {w}×{h}。画面上红圈 1~4 是要碰的四个角：夹爪合上、竖直朝下，"
                        "用拖动示教把指尖碰到角上（指尖正对角点），每碰一个点一次“记录”")
        self.board = None
        return "没找到棋盘格：要整块在画面里、别反光、别被挡住（机械臂先挪开）"

    def touch(self, label, tip):
        """记下第 label 个外角（"①"~"④"）被指尖碰到时，指尖在底座坐标里的位置。"""
        if self.board is None:
            return "先点“识别棋盘格”"
        self.touched[label] = np.asarray(tip, float)
        left = [l for l in TOUCH_LABELS if l not in self.touched]
        return f"记下了角 {label}：{np.round(tip * 1000).astype(int).tolist()} mm" + (f"，还差 {''.join(left)}" if left else "，四个角都碰过了，可以计算外参")

    def calibrate(self, q=None):
        """用四个碰点 + 画面里的棋盘格 + 深度图算外参，q（机械臂现在的关节角，度）给了就再用机械臂模型精修，存盘。
        返回 (成功?, 说明)。"""
        if self.board is None or len(self.touched) < 3:
            return False, "至少要碰 3 个角（最好 4 个）"
        depth = self.median_depth()
        if depth is None:
            return False, "相机还没有画面"
        T, info = solve_extrinsic(self.board["corners"], self.board["size"], self.board["outer"], self.touched,
                                  depth, self.K, self.dist)
        if T is None:
            return False, info
        if q is not None:
            T2, fit = refine_with_robot(T, depth, self.K, self.dist, q)
            info.update(fit)
            if T2 is not None:
                T = T2
        info["time"] = time.strftime("%Y-%m-%d %H:%M")
        self.T, self.calib_info = T, info
        np.savez(CALIB_PATH, T=T, corners=self.board["corners"], size=np.array(self.board["size"]),
                 touched=np.array([self.touched.get(l, [np.nan] * 3) for l in TOUCH_LABELS]),
                 **{k: np.array(v) for k, v in info.items()})
        return True, self.calib_text()

    def refine(self, q):
        """只用机械臂模型精修现在的外参（相机被碰了一下、或者想换个姿态再核对时用）。q：机械臂现在的关节角（度）。"""
        if self.T is None:
            return False, "先标定一次外参（精修要从一个差不多的外参开始）"
        depth = self.median_depth()
        if depth is None:
            return False, "相机还没有画面"
        T2, fit = refine_with_robot(self.T, depth, self.K, self.dist, q)
        if T2 is None:
            return False, fit["refine_msg"]
        info = dict(self.calib_info or {})
        info.update(fit)
        info["time"] = time.strftime("%Y-%m-%d %H:%M")
        self.T, self.calib_info = T2, info
        old = dict(np.load(CALIB_PATH)) if os.path.exists(CALIB_PATH) else {}
        old.update(T=T2, **{k: np.array(v) for k, v in info.items()})
        np.savez(CALIB_PATH, **old)
        return True, self.calib_text()

    def median_depth(self, n=10, timeout=3.0):
        """连续 n 帧深度取中值（单帧噪声 1 m 处约 2~3 mm），一半以上帧没测到的像素当 0。"""
        frames, last, t0 = [], None, time.time()
        while len(frames) < n and time.time() - t0 < timeout:
            st = self.cam.latest
            if st is not None and st[3] != last:
                frames.append(st[2])
                last = st[3]
            time.sleep(0.02)
        if not frames:
            return None
        d = np.stack(frames)
        valid = (d > 0).sum(0) * 2 > len(frames)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # 全是 NaN 的像素（一直没测到）
            med = np.nanmedian(np.where(d > 0, d, np.nan), axis=0)
        return np.where(valid, med, 0).astype(np.float32)

    def calib_text(self):
        if self.calib_info is None:
            return "还没标定外参"
        c = self.calib_info
        if "table_z" not in c:        # 2026-11-02 以前的旧标定（格子边长按碰点算，点云整体偏）
            return f"外参（{c['time']}）是旧方法标的，点云会对不上：重新标定一次"
        text = (f"外参（{c['time']}）：格子边长（深度量的）{c['square'] * 1000:.1f} mm，桌面在底座 z={c['table_z'] * 1000:+.0f} mm，"
                f"碰点残差 {c['touch_rms'] * 1000:.1f} mm，重投影误差 {c['reproj']:.2f} 像素，桌面平面残差 {c['table_err'] * 1000:.1f} mm")
        warn = []
        if "square_touch" in c and abs(c["square_touch"] / c["square"] - 1) > 0.03:
            warn.append(f"四个碰点围出来的格子是 {c['square_touch'] * 1000:.1f} mm，每个角平均往"
                        f"{'外' if c['square_touch'] > c['square'] else '里'}偏了约 {c['touch_bias'] * 1000:.0f} mm（指尖没正对角点中心？）")
        if c.get("touch_tilt", 0) > 1.5:
            warn.append(f"碰点连成的面比桌面歪了 {c['touch_tilt']:.1f}°（有的角碰得深有的浅？）")
        if "refine_msg" in c:
            text += "。" + c["refine_msg"]
        if warn and not c.get("refined"):
            text += "。注意：" + "；".join(warn) + "。外参的倾斜和高度按深度图里的桌面算，水平位置按碰点算，碰点偏了重标时碰准一点更好"
        return text

    # ---------------- 点云 ----------------
    def rays(self, shape, stride):
        """每隔 stride 个像素一条去畸变的射线 (x/z, y/z)，形状 (高/stride, 宽/stride, 2)。彩色相机有一点畸变
        （画面边上约 0.7%，1.1 m 处约 5 mm），反投影时按内参 + 畸变算。"""
        key = (shape[0], shape[1], stride)
        if key not in self._rays:
            v, u = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
            uv = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float64)
            self._rays[key] = cv2.undistortPoints(uv, self.K, self.dist).reshape(v.shape[0], v.shape[1], 2)
        return self._rays[key]

    def backproject(self, depth, T, mask=None, stride=1):
        """深度图 → 底座坐标里的点（只要 mask 里、深度有效的）。返回 (点 N×3, 选中的像素 布尔数组)。"""
        d = depth[::stride, ::stride]
        ok = d > 0
        if mask is not None:
            ok &= mask[::stride, ::stride]
        r = self.rays(depth.shape, stride)[ok]
        p = np.c_[r, np.ones(len(r))] * d[ok][:, None]
        return p @ T[:3, :3].T + T[:3, 3], ok

    def make_cloud(self, q=None):
        """当前画面的点云（底座坐标，裁到工作区）：(xyz float32 N×3, rgb uint8 N×3)。没标外参返回 None。
        q：这一刻机械臂的关节角（度），存下来，分割时用来去掉机械臂自己的点。"""
        f = self.frame()
        if f is None or self.T is None:
            return None
        color, depth = f
        s = CLOUD_STRIDE
        p, ok = self.backproject(depth, self.T, stride=s)
        rgb = color[::s, ::s][ok][:, ::-1]                                    # BGR → RGB
        keep = np.all((p >= WORKSPACE[:, 0]) & (p <= WORKSPACE[:, 1]), axis=1)
        with self.lock:
            self.cloud = (p[keep].astype(np.float32), np.ascontiguousarray(rgb[keep]).astype(np.uint8))
            self.cloud_q = q
            return self.cloud

    # ---------------- 点一下分割物体 ----------------
    def segment(self, seed, q=None, tol=COLOR_TOL, square=False):
        """在最近一次的点云里，从离 seed 最近的点长出物体：返回 (结果 dict 或 None, 说明)。
        结果：idx（点云里属于物体的点的序号）、center、size（沿朝向的长、宽、高）、yaw（长边方向，弧度）、top、bottom。
        q：机械臂的关节角（度），给了就先把机械臂自己的点去掉（夹爪挨着物体时不会连成一大块）；不给就用拍这份点云时的关节角。
        tol：颜色容差（Lab 距离，亮度按一半算）：越大越不看颜色，0 表示不按颜色、只按形状长。

        2026-11-02 改成按颜色长（用户：点云有颜色，直接点很容易把背景带进去成一个大框）：
        原来只按形状——高出桌面 8 mm、相邻 12 mm 以内就连成一块，物体挨着线、盒子、夹爪、墙就整片长过去；
        现在从点到的那一点的颜色出发，只长到“和点到的颜色差 tol 以内、而且比旁边桌面的颜色更近”的点，
        颜色一变（挨着的别的物体、线、盒子）就停；离点到的地方最多 MAX_OBJ_R。
        物体和桌面颜色差不多、或者按颜色只长出几个点时，退回按形状长 + 只用上半部分的点定位置。"""
        with self.lock:
            if self.cloud is None:
                return None, "还没有点云"
            p = self.cloud[0].astype(np.float64)
            rgb = self.cloud[1]
            if q is None:
                q = getattr(self, "cloud_q", None)
        seed = np.asarray(seed, float)
        tree = cKDTree(p)
        # 种子附近 25 cm 里拟合桌面（低的点，迭代去掉离群点）
        near = np.array(tree.query_ball_point(seed, 0.25))
        if len(near) < 30:
            return None, "这里点太少"
        low = near[p[near, 2] < np.percentile(p[near, 2], 40) + 0.01]
        A = np.c_[p[low, :2], np.ones(len(low))]
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A, p[low, 2], rcond=None)
            res = np.abs(A @ coef - p[low, 2])
            keep = res < max(0.003, np.percentile(res, 60))
            low, A = low[keep], A[keep]
        height = p[:, 2] - (np.c_[p[:, :2], np.ones(len(p))] @ coef)
        in_r = np.linalg.norm(p[:, :2] - seed[:2], axis=1) < MAX_OBJ_R
        ok = (height >= TABLE_GAP) & in_r
        robot_note = ""
        if q is not None:                         # 机械臂自己的点不算（离模型表面 1.5 cm 以内）
            d_robot = cKDTree(robot_surface(q)).query(p, distance_upper_bound=ROBOT_PAD)[0]
            on_robot = d_robot < ROBOT_PAD
            ok &= ~on_robot
            in_r &= ~on_robot
        _, start = tree.query(seed)
        if height[start] < TABLE_GAP:
            return None, "点到的是桌面：点在物体上"
        if q is not None and on_robot[start]:
            return None, "点到的是机械臂自己：点在物体上"
        lab = cv2.cvtColor(np.ascontiguousarray(rgb).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
        lab *= [0.5, 1.0, 1.0]                    # 亮度按一半算：同一个物体朝光和背光的面亮度差得多，颜色（a、b）差得少
        # 点到的颜色：种子 1 cm 以内、高出桌面、颜色和点到的那一点接近的点的中值（点在两个物体交界处时不把两种颜色混在一起）；
        # 旁边桌面的颜色
        around = np.array(tree.query_ball_point(p[start], 0.010), int)
        around = around[ok[around] & (np.linalg.norm(lab[around] - lab[start], axis=1) < max(tol, 1.0) * 0.5)]
        c_seed = np.median(lab[np.r_[around, start]], axis=0)
        table_near = near[height[near] < 0.004]
        c_tab = np.median(lab[table_near], axis=0) if len(table_near) >= 30 else None
        use_color = tol > 0 and (c_tab is None or np.linalg.norm(c_seed - c_tab) >= TABLE_COLOR_MIN)

        def grow(color):
            seen = np.zeros(len(p), bool)
            seen[start] = True
            frontier = np.array([start])
            while len(frontier) and seen.sum() < 40000:
                nb = tree.query_ball_point(p[frontier], GROW_R)
                src = np.repeat(frontier, [len(x) for x in nb])
                dst = np.fromiter((j for x in nb for j in x), int, count=len(src))
                m = ok[dst] & ~seen[dst]
                src, dst = src[m], dst[m]
                if color:
                    d_obj = np.linalg.norm(lab[dst] - c_seed, axis=1)
                    m = d_obj < tol
                    if c_tab is not None:
                        m &= d_obj < np.linalg.norm(lab[dst] - c_tab, axis=1)
                    dst = dst[m]
                frontier = np.unique(dst)
                seen[frontier] = True
            return np.flatnonzero(seen)

        how = ""
        idx = grow(True) if use_color else np.array([], int)
        if use_color and len(idx) >= MIN_COLOR_PTS:       # 被挡住一大半的小物体颜色点也就十来个：照样按颜色框，比连着旁边的物体按形状框准
            how = "按颜色长"
            # 再往外扩一两圈：紧挨着的、颜色放宽到 1.5 倍容差（还要比桌面色更近）、离桌面 3 mm 以上的点也算——
            # 朝相机那一面下半截颜色暗、离桌面不到 8 mm，原来收不进来，框往亮的那一侧偏了几毫米（2026-11-02 用户：没把有颜色的点都框进去）
            c_obj = np.median(lab[idx], axis=0)
            low_ok = (height >= EXPAND_MIN_H) & in_r
            member = np.zeros(len(p), bool)
            member[idx] = True
            for _ in range(3):
                nb = tree.query_ball_point(p[np.flatnonzero(member)], EXPAND_R)
                cand = np.unique(np.fromiter((j for x in nb for j in x), int))
                cand = cand[low_ok[cand] & ~member[cand]]
                if not len(cand):
                    break
                d_o = np.linalg.norm(lab[cand] - c_obj, axis=1)
                m = d_o < tol * EXPAND_TOL
                if c_tab is not None:
                    m &= d_o < np.linalg.norm(lab[cand] - c_tab, axis=1)
                if not m.any():
                    break
                member[cand[m]] = True
            grown = int(member.sum()) - len(idx)
            idx = np.flatnonzero(member)
            if grown:
                how += f"，边上颜色暗的点补进来 {grown} 个"
        else:
            why = ("按颜色只长出几个点" if use_color else "颜色和桌面差不多" if tol > 0 else "")
            # 颜色和桌面分不开（或者按颜色只长出几个点）：只按形状长，位置和大小用上半部分的点（边上被抬高的桌面点在下半部分）
            idx = grow(False)
            if len(idx) < 20:
                return None, "长出来的点太少，换个地方点"
            h_all = height[idx]
            half = idx[h_all >= 0.5 * float(np.percentile(h_all, 98))]
            if len(half) >= 20:
                idx = half
            how = f"{why}，只按形状长" if why else "只按形状长"
        # 去掉孤零零的噪点（1 cm 以内同一物体的点不到 2 个），剩下的点全部框进去（原来两头各去掉 2%，点少的小方块每边少 1~2 mm）
        sub = cKDTree(p[idx])
        cnt = np.array([len(x) for x in sub.query_ball_point(p[idx], 0.010)])
        if (cnt >= 3).sum() >= max(MIN_COLOR_PTS, len(idx) // 2):
            idx = idx[cnt >= 3]
        q_pts = p[idx]
        xy = q_pts[:, :2] - q_pts[:, :2].mean(axis=0)
        evals, evecs = np.linalg.eigh(np.cov(xy.T))
        e1 = evecs[:, 1]                                                      # 长边方向
        yaw = float(np.arctan2(e1[1], e1[0]))
        Rz = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        loc = xy @ Rz.T
        lo, hi = loc.min(axis=0), loc.max(axis=0)
        mid = (lo + hi) / 2
        center_xy = q_pts[:, :2].mean(axis=0) + mid @ Rz
        top = float(np.percentile(q_pts[:, 2], 99))
        if square:
            # 正方形没有稳定的主轴。用顶面轮廓的最小外接矩形找边。
            face = q_pts[np.abs(q_pts[:, 2] - top) <= 0.005, :2]
            if len(face) < 16:
                return None, "方块顶面点太少，请刷新点云并点选顶面"
            center_xy, sides, angle = cv2.minAreaRect(face.astype(np.float32))
            w, h = sides
            if not (0.015 <= min(w, h) <= max(w, h) <= 0.085 and min(w, h) / max(w, h) >= 0.75):
                return None, "顶面轮廓不像正方形，请检查框选结果或调整颜色容差"
            center_xy = np.asarray(center_xy, float)
            yaw = float(np.radians(angle))
            lo, hi = np.zeros(2), np.array(sides)
        bottom = float(np.c_[center_xy[None], [[1.0]]] @ coef)
        size = [float(hi[0] - lo[0]), float(hi[1] - lo[1]), top - bottom]
        box = {"idx": idx.tolist(), "center": [float(center_xy[0]), float(center_xy[1]), (top + bottom) / 2],
               "size": size, "yaw": yaw, "top": top, "bottom": bottom}
        if q is not None:
            how += "，去掉了机械臂自己的点"
        return box, f"框出物体：{size[0] * 100:.1f} × {size[1] * 100:.1f} × {size[2] * 100:.1f} cm（{len(idx)} 个点，{how}）"


CALIB_KEYS = ("square", "square_touch", "touch_bias", "touch_rms", "touch_tilt", "reproj", "table_err", "table_z",
              "refined", "refine_msg", "arm_fit", "time")


def _robot_model():
    """机械臂表面采样点（URDF 每个连杆的 STL 按面积均匀撒点，约 13 万个，连杆自己的坐标系）+ 夹爪本体长方体表面。缓存。"""
    global _ROBOT
    if _ROBOT is not None:
        return _ROBOT
    import struct
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(0)

    def samples(name):
        b = open(os.path.join(ROBOT_DIR, "meshes", name), "rb").read()
        n = struct.unpack("<I", b[80:84])[0]
        v = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("x", "<u2")]),
                          count=n)["v"].astype(np.float64)
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        m = max(2000, int(area.sum() * 4e5))                                  # 每平方厘米 40 个点
        i = rng.choice(len(v), m, p=area / area.sum())
        r1, r2 = np.sqrt(rng.random(m)), rng.random(m)
        return v[i, 0] * (1 - r1)[:, None] + v[i, 1] * (r1 * (1 - r2))[:, None] + v[i, 2] * (r1 * r2)[:, None]

    urdf = ET.parse(os.path.join(ROBOT_DIR, "ECO65-B.urdf")).getroot()
    mesh = urdf.find("link[@name='base_link']/visual/geometry/mesh")
    base = samples(mesh.get("filename").split("/")[-1]) if mesh is not None else np.zeros((0, 3))
    joints = []
    for j in urdf.findall("joint"):
        o = j.find("origin")
        T0 = np.eye(4)
        T0[:3, :3] = Rot.from_euler("xyz", [float(x) for x in o.get("rpy").split()]).as_matrix()
        T0[:3, 3] = [float(x) for x in o.get("xyz").split()]
        mesh = urdf.find(f"link[@name='{j.find('child').get('link')}']/visual/geometry/mesh")
        pts = samples(mesh.get("filename").split("/")[-1]) if mesh is not None else None
        joints.append((T0, np.array([float(x) for x in j.find("axis").get("xyz").split()]), pts))
    # 夹爪本体：法兰坐标系里 50（x）× 70.6（y）mm、长 13.4 cm 的长方体表面（手指会动，不算）
    g = np.array(np.meshgrid(np.linspace(-0.025, 0.025, 11), np.linspace(-0.0353, 0.0353, 15), np.linspace(0, 0.134, 28))).reshape(3, -1).T
    g = g[(np.abs(np.abs(g[:, 0]) - 0.025) < 1e-9) | (np.abs(np.abs(g[:, 1]) - 0.0353) < 1e-9)]
    _ROBOT = (base, joints, g)
    return _ROBOT


def robot_surface(q_deg):
    """关节角 q（度）时机械臂表面的点（底座坐标，N×3）。"""
    from scipy.spatial.transform import Rotation as Rot
    base, joints, grip = _robot_model()
    out, T = [base], np.eye(4)
    for (T0, axis, pts), q in zip(joints, np.radians(q_deg)):
        Tq = np.eye(4)
        Tq[:3, :3] = Rot.from_rotvec(axis * q).as_matrix()
        T = T @ T0 @ Tq
        if pts is not None:
            out.append(pts @ T[:3, :3].T + T[:3, 3])
    out.append(grip @ T[:3, :3].T + T[:3, 3])
    return np.vstack(out)


def refine_with_robot(T, depth, K, dist, q_deg):
    """拿点云里的机械臂和模型做 ICP，精修外参 T（相机 → 底座）。返回 (新 T 或 None, 说明 dict)。
    只用离桌面 3 cm 以上、离模型不远的点；对上的距离阈值 5 cm → 1.2 cm 逐步收紧。对不上（残差大、挪得太多）就不用。"""
    from scipy.spatial.transform import Rotation as Rot
    model = robot_surface(q_deg)
    tree = cKDTree(model)
    v, u = np.nonzero(depth[::2, ::2] > 0)
    v, u = v * 2, u * 2
    r = cv2.undistortPoints(np.c_[u, v].astype(np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    Pc = np.c_[r, np.ones(len(r))] * depth[v, u][:, None].astype(np.float64)

    def fit(TT):
        P = Pc @ TT[:3, :3].T + TT[:3, 3]
        d = tree.query(P, distance_upper_bound=0.05)[0]
        arm = (d < 0.05) & (P[:, 2] > 0.03)
        return (float(np.median(d[arm])) if arm.sum() > 100 else np.inf), (float(np.mean(d[arm] < 0.01)) if arm.sum() else 0.0), int(arm.sum())

    before = fit(T)
    P0 = Pc @ T[:3, :3].T + T[:3, 3]
    cand = Pc[(tree.query(P0, distance_upper_bound=0.08)[0] < 0.08) & (P0[:, 2] > 0.03)]
    if len(cand) < 500:
        return None, {"refine_msg": f"画面里机械臂的点太少（{len(cand)} 个），没法用机械臂精修：机械臂要整个在画面里"}
    Ti = T.copy()
    for thr in [0.05] * 5 + [0.03] * 5 + [0.02] * 10 + [0.012] * 10:
        P = cand @ Ti[:3, :3].T + Ti[:3, 3]
        d, j = tree.query(P)
        sel = (d < thr) & (P[:, 2] > 0.03)
        if sel.sum() < 300:
            break
        a, b = P[sel], model[j[sel]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((b - mb).T @ (a - ma))
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        dT = np.eye(4)
        dT[:3, :3] = U @ D @ Vt
        dT[:3, 3] = mb - dT[:3, :3] @ ma
        Ti = dT @ Ti
    after = fit(Ti)
    delta = Ti @ np.linalg.inv(T)
    move = float(np.linalg.norm(delta[:3, 3]))
    rot = float(np.degrees(Rot.from_matrix(delta[:3, :3]).magnitude()))
    if after[0] > 0.008 or after[1] < 0.6 or move > 0.12 or rot > 10:
        return None, {"refined": False, "arm_fit": before[0],
                      "refine_msg": (f"机械臂模型精修没对上（精修后机械臂点离模型中位 {after[0] * 1000:.0f} mm、1 cm 内 {after[1] * 100:.0f}%，"
                                     f"要挪 {move * 1000:.0f} mm、转 {rot:.1f}°），没用：机械臂要整个在画面里、别被挡住")}
    P = Pc @ Ti[:3, :3].T + Ti[:3, 3]                                        # 精修后桌面在多高：离模型远、低处的点
    low = (np.abs(P[:, 2]) < 0.03) & (tree.query(P, distance_upper_bound=0.08)[0] > 0.08)
    table_z = float(np.median(P[low, 2])) if low.sum() > 500 else np.nan
    return Ti, {"refined": True, "arm_fit": after[0], **({"table_z": table_z} if np.isfinite(table_z) else {}),
                "refine_msg": (f"已用机械臂模型精修（挪了 {move * 1000:.0f} mm、转了 {rot:.1f}°）："
                               f"机械臂点云离模型中位 {before[0] * 1000:.0f} → {after[0] * 1000:.0f} mm，1 cm 以内 {after[1] * 100:.0f}%")}


def solve_extrinsic(corners, size, outer, touched, depth, K, dist):
    """算外参（相机 → 底座）。corners：画面里的内角点（像素，N×2）；size：(列, 行)；outer：四个外角在 corners 里的序号；
    touched：{"①": 底座坐标, ...}（至少 3 个）；depth：对齐到彩色的深度图（米）。返回 (T 4×4, 说明 dict) 或 (None, 原因)。

    思路（2026-11-02 第二版）：棋盘格在画面里很小（1 m 外 15 cm 的板只有六七十个像素），光靠 PnP 算出来的朝向和距离都不太准；
    碰点又容易整体往外偏几毫米。所以哪个量准就用哪个：
      · 桌面的朝向和高度：深度图里棋盘格那一大片桌面拟合平面（几千个点），机械臂底座装在桌面上，桌面就是底座的水平面；
      · 角点的三维位置和格子边长：角点像素的射线和这个平面求交（不用 PnP）；
      · 剩下的水平位置和朝向（x、y、绕竖直轴转多少）：四个角点和碰点在水平面里做二维刚体配准；高度取碰点的平均高度。"""
    w, h = size
    corners = np.asarray(corners, np.float64)
    grid = np.array([[i % w, i // w, 0.0] for i in range(w * h)])
    labs = [l for l in TOUCH_LABELS if l in touched]
    if len(labs) < 3:
        return None, "至少要碰 3 个角（最好 4 个）"
    # 1. 深度图里的点（相机坐标，按内参 + 畸变）
    v, u = np.nonzero(depth > 0)
    r = cv2.undistortPoints(np.c_[u, v].astype(np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    P = np.c_[r, np.ones(len(r))] * depth[v, u][:, None].astype(np.float64)
    # 2. 板面所在的桌面：先拿棋盘格那块拟合，再往外扩到周围 30 cm 里离平面 6 mm 以内的点，迭代几次
    hull = cv2.convexHull(corners.astype(np.float32))
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
    on_board = mask[v, u].astype(bool)
    if on_board.sum() < 200:
        return None, "棋盘格那块深度图没测到（反光？）：换个角度或换块不反光的板"

    def fit_plane(Q):
        c = Q.mean(0)
        n = np.linalg.eigh(np.cov((Q - c).T))[1][:, 0]     # 最小特征值的方向就是法向
        return (n if n @ c < 0 else -n), c              # 法向朝相机（相机在原点，看 +z）
    n, c = fit_plane(P[on_board])
    center = P[on_board].mean(0)
    for rad in (0.15, 0.25, 0.30, 0.30):
        sel = (np.abs((P - c) @ n) < 0.006) & (np.linalg.norm(P - center, axis=1) < rad)
        n, c = fit_plane(P[sel])
    res = (P[sel] - c) @ n
    table_err = float(np.median(np.abs(res)))
    # 3. 角点：像素射线和平面求交（板子薄，就当在桌面上；板面那块比桌面高多少按板面点的平均算）
    lift = float(np.median((P[on_board] - c) @ n))
    rc = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rc = np.c_[rc, np.ones(len(rc))]
    C = rc * (((c + n * lift) @ n) / (rc @ n))[:, None]
    adj = [np.linalg.norm(C[j * w + i + 1] - C[j * w + i]) for j in range(h) for i in range(w - 1)]
    adj += [np.linalg.norm(C[(j + 1) * w + i] - C[j * w + i]) for j in range(h - 1) for i in range(w)]
    square = float(np.median(adj))
    # PnP 只用来报重投影误差（检查角点有没有认错）
    ok, rvec, tvec = cv2.solvePnP(grid * square, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(grid * square, rvec, tvec, K, dist)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
    # 4. 相机坐标先转到“桌面朝上”：法向 n → 底座 z
    z_ax = n
    x_ax = np.cross([0.0, 1.0, 0.0], z_ax)
    x_ax /= np.linalg.norm(x_ax)
    R0 = np.stack([x_ax, np.cross(z_ax, x_ax), z_ax])     # 行向量是新坐标轴
    src = (C[[outer[TOUCH_LABELS.index(l)] for l in labs]]) @ R0.T
    dst = np.array([touched[l] for l in labs], float)
    # 5. 水平面里二维刚体配准（x、y、绕竖直轴），高度取碰点平均
    ms, md = src[:, :2].mean(0), dst[:, :2].mean(0)
    H = (src[:, :2] - ms).T @ (dst[:, :2] - md)
    yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    t = np.r_[md - Rz[:2, :2] @ ms, dst[:, 2].mean() - src[:, 2].mean()]
    Rm = Rz @ R0
    fitted = src @ Rz.T + t
    touch_rms = float(np.sqrt(np.mean(np.sum((fitted - dst) ** 2, axis=1))))
    # 碰点围出来的比板子大多少、碰点自己连成的面歪了多少（只用来提示）
    k_touch = umeyama(src[:, :2], dst[:, :2])[0]
    touch_bias = float(abs(k_touch - 1) * np.mean(np.linalg.norm(src[:, :2] - ms, axis=1)))
    dz = dst[:, 2] - dst[:, 2].mean()
    A = np.c_[dst[:, :2] - md, np.ones(len(dst))]
    tilt = float(np.degrees(np.arctan(np.linalg.norm(np.linalg.lstsq(A, dz, rcond=None)[0][:2])))) if len(dst) >= 3 else 0.0
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return T, {"square": square, "square_touch": float(square * k_touch), "touch_bias": touch_bias, "touch_rms": touch_rms,
               "touch_tilt": tilt, "reproj": reproj, "table_err": table_err, "table_z": float(dst[:, 2].mean() - lift)}


def umeyama(src, dst, scale=True):
    """配准：dst ≈ s·R·src + t（最小二乘）；scale=False 时 s 固定为 1（刚体）。返回 (s, R, t)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(axis=1).mean() if scale else 1.0
    return s, R, mu_d - s * R @ mu_s
