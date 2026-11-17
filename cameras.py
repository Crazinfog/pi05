"""两台奥比中光深度相机：Gemini 335（固定在旁边，第三视角）和 Gemini 305（腕部）。

    python cameras.py               # 测试：两台一起开 5 秒，报帧率，存彩色图、深度图和内参到 logs/camera_test/

每台相机同时开彩色和深度，深度用 SDK 的对齐滤波器对齐到彩色图（每个彩色像素都有对应的深度）。
现在两台都接在 USB 2.0 上，带宽有限，所以用 640x480、15 fps、彩色走 MJPG 压缩；以后换 USB 3.0 可以调高。
"""
import os
import threading
import time

import cv2
import numpy as np
from pyorbbecsdk import AlignFilter, Config, Context, OBFormat, OBSensorType, OBStreamType, Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
SERIALS = {"scene": "CP0T26300080", "wrist": "CV2L36000027"}   # Gemini 335 第三视角；Gemini 305 腕部
WIDTH, HEIGHT, FPS = 640, 480, 15
STALL_S = 3.0     # 这么久没收到一帧就重新连这台相机（2026-11-03 腕部相机线跟着手臂动，掉线重新枚举后旧句柄一直收不到帧，录了 3 条都没有腕部画面）
_ctx = None


def pick(profiles, fmt):
    """按 640x480 / 15fps / 指定格式挑一种，挑不到就用默认的。"""
    try:
        return profiles.get_video_stream_profile(WIDTH, HEIGHT, fmt, FPS)
    except Exception:
        return profiles.get_default_video_stream_profile()


class DepthCamera:
    """后台线程不停取帧，latest 里放最新的一份：(时刻, 彩色图 BGR, 深度图 米, 序号)。"""

    def __init__(self, device, name, serial=None):
        self.name, self.serial = name, serial
        self.latest, self.count, self.running = None, 0, True
        self.reconnects, self.last_t, self.state = 0, time.time(), "正常"
        self._start(device)
        threading.Thread(target=self._loop, daemon=True).start()

    def _start(self, device):
        self.pipe = Pipeline(device)
        cfg = Config()
        cfg.enable_stream(pick(self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR), OBFormat.MJPG))
        cfg.enable_stream(pick(self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR), OBFormat.Y16))
        self.pipe.start(cfg)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        p = self.pipe.get_camera_param()
        ci = p.rgb_intrinsic
        self.K = np.array([[ci.fx, 0, ci.cx], [0, ci.fy, ci.cy], [0, 0, 1]], dtype=np.float64)   # 彩色相机内参
        cd = p.rgb_distortion                                                                       # 畸变，换成 OpenCV 的顺序
        self.dist = np.array([getattr(cd, k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], dtype=np.float64)
        self.size = (ci.width, ci.height)

    def _reopen(self):
        """按序列号在同一个 Context 里重新找到这台相机、重开（不新建 Context：新建会把另一台断掉）。"""
        try:
            self.pipe.stop()
        except Exception:
            pass
        dl = _ctx.query_devices()
        for i in range(dl.get_count()):
            dev = dl.get_device_by_index(i)
            if dev.get_device_info().get_serial_number() == self.serial:
                self._start(dev)
                return True
        return False

    def fresh(self, max_age=1.0):
        """最近 max_age 秒里有没有新画面。"""
        lat = self.latest
        return lat is not None and time.time() - lat[0] < max_age

    def _loop(self):
        while self.running:
            try:
                frames = self.pipe.wait_for_frames(200)
            except Exception:
                frames = None
            if frames is None:
                if self.serial and time.time() - self.last_t > STALL_S:
                    self.state = "正在重连"
                    print(f"[相机] {self.name} 已经 {time.time() - self.last_t:.0f} 秒没有画面，重新连接……", flush=True)
                    try:
                        ok = self._reopen()
                    except Exception as e:
                        ok = False
                        print(f"[相机] {self.name} 重连出错：{e}", flush=True)
                    self.reconnects += ok
                    self.state = "已重连，等画面" if ok else "没找到这台相机（线没接好？）"
                    print(f"[相机] {self.name} {self.state}", flush=True)
                    self.last_t = time.time()          # 再等 STALL_S 还没画面就再试
                continue
            frames = self.align.process(frames)
            if frames is None:
                continue
            frames = frames.as_frame_set()
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if c is None or d is None:
                continue
            color = cv2.imdecode(np.frombuffer(c.get_data(), np.uint8), cv2.IMREAD_COLOR)   # MJPG → BGR
            depth = (np.frombuffer(d.get_data(), np.uint16).reshape(d.get_height(), d.get_width())
                     .astype(np.float32) * d.get_depth_scale() / 1000.0)                    # → 米，0 表示没测到
            if color is None:
                continue
            self.count += 1
            self.last_t = time.time()
            self.state = "正常"
            self.latest = (self.last_t, color, depth, self.count)

    def close(self):
        self.running = False
        time.sleep(0.3)
        try:
            self.pipe.stop()
        except Exception:
            pass


def open_cameras(names=None):
    """按序列号打开相机，返回 {"scene": DepthCamera, "wrist": DepthCamera}（没接的就没有）。names 只开其中几台。"""
    global _ctx
    _ctx = Context()                 # Context 要一直留着，被回收后设备就打不开了
    dl = _ctx.query_devices()
    cams = {}
    for i in range(dl.get_count()):
        dev = dl.get_device_by_index(i)
        sn = dev.get_device_info().get_serial_number()
        for name, s in SERIALS.items():
            if sn == s and (names is None or name in names):
                cams[name] = DepthCamera(dev, name, sn)
    return cams


def find(name):
    """启动时没找到的那台：在已有的 Context 里再找一次（插好线以后不用重启服务）。找不到返回 None。"""
    global _ctx
    if _ctx is None:
        _ctx = Context()
    dl = _ctx.query_devices()
    for i in range(dl.get_count()):
        dev = dl.get_device_by_index(i)
        if dev.get_device_info().get_serial_number() == SERIALS[name]:
            return DepthCamera(dev, name, SERIALS[name])
    return None


if __name__ == "__main__":
    out = os.path.join(HERE, "logs", "camera_test")
    os.makedirs(out, exist_ok=True)
    cams = open_cameras()
    print("打开了:", list(cams))
    time.sleep(1.0)
    start = {n: c.count for n, c in cams.items()}
    time.sleep(5.0)
    for n, c in cams.items():
        fps = (c.count - start[n]) / 5.0
        if c.latest is None:
            print(f"{n}: 没收到画面")
            continue
        _, color, depth, _ = c.latest
        valid = depth > 0
        print(f"{n}: {fps:.1f} fps，彩色 {color.shape[1]}x{color.shape[0]}，深度有效像素 {valid.mean() * 100:.0f}%，"
              f"深度范围 {depth[valid].min():.2f}~{depth[valid].max():.2f} m，中间那点 {depth[depth.shape[0] // 2, depth.shape[1] // 2]:.3f} m")
        print(f"   内参 fx {c.K[0, 0]:.1f} fy {c.K[1, 1]:.1f} cx {c.K[0, 2]:.1f} cy {c.K[1, 2]:.1f}，尺寸 {c.size}")
        cv2.imwrite(os.path.join(out, f"{n}_color.jpg"), color)
        vis = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 1.5), alpha=255 / 1.5), cv2.COLORMAP_JET)
        vis[~valid] = 0
        cv2.imwrite(os.path.join(out, f"{n}_depth.png"), vis)
        np.save(os.path.join(out, f"{n}_K.npy"), c.K)
    for c in cams.values():
        c.close()
    print("图片存在", out)
