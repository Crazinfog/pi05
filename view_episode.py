"""看一条录好的示范：打印文件里有什么，再导出一段视频和一张缩略图（Orin 上双击就能打开）。

    python view_episode.py                                         # 存储路径里最新的一条
    python view_episode.py data/demos/episode_000001.hdf5          # 指定某一条

视频（.avi）：左边第三视角、右边腕部，下面一行字是时间、夹爪指令和实际开口、指令和实际关节角差多少度，
转换时会被裁掉的静止帧标红框。缩略图（.jpg）：开头、合夹爪、合上后 1 秒、松手、结尾几帧。
都存在 logs/preview/ 里（不放进数据文件夹，免得混进数据集）。
"""
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "logs", "preview")
BAR = 44                         # 视频下面那一行字的高度（像素）


def latest():
    """存储路径（record_config.json 里的 folder）里编号最大的一条。"""
    folder = os.path.join(HERE, "data", "demos")
    try:
        with open(os.path.join(HERE, "record_config.json"), encoding="utf-8") as f:
            folder = os.path.expanduser(json.load(f).get("folder", folder))
    except (OSError, ValueError):
        pass
    eps = sorted(glob.glob(os.path.join(folder, "episode_*.hdf5")))
    if not eps:
        sys.exit(f"{folder} 里还没有录好的示范")
    return eps[-1]


def describe(h):
    """打印文件里每一项：路径、形状、类型，以及记录的信息（任务描述、成功与否……）。"""
    print("文件里有：")

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            kind = "JPEG/PNG 字节（每帧一张图）" if obj.dtype.kind == "O" else str(obj.dtype)
            print(f"  /{name:34s} {str(obj.shape):12s} {kind}")
    h.visititems(show)
    print("记录的信息：")
    for k in ("task", "success", "fps", "frames", "created", "source", "end", "cameras", "exec_noise_mm", "start_noise_deg"):
        if k in h.attrs:
            v = h.attrs[k]
            if k == "success":
                v = {1: "1（标了成功）", 0: "0（标了失败）", -1: "-1（还没标）"}.get(int(v), v)
            print(f"  {k:16s} {v}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest()
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"== {path}")
    with h5py.File(path, "r") as h:
        describe(h)
        t = h["timestamp"][:]
        act = h["action"][:]
        q = h["observations/qpos"][:]
        grip = h["observations/gripper"][:]
        keep = h["keep"][:].astype(bool) if "keep" in h else np.ones(len(t), bool)
        cams = [c for c in ("scene", "wrist") if f"observations/images/{c}" in h]
        imgs = {c: h[f"observations/images/{c}"] for c in cams}
        n = len(t)
        fps = float(h.attrs.get("fps", 15))
        cmd_open = act[:, 6] > 0.5
        events = [(i, "open" if cmd_open[i] else "close") for i in range(1, n) if cmd_open[i] != cmd_open[i - 1]]
        err = np.degrees(np.abs(act[:, :6] - q)).max(axis=1)      # 指令比实际领先（真机慢约 0.5 秒），动起来时差几度很正常

        print(f"共 {n} 帧、{t[-1] - t[0]:.1f} 秒；夹爪指令变化："
              + ("、".join(f"{t[i] - t[0]:.1f} 秒{'张开' if g == 'open' else '合上'}" for i, g in events) or "没有"))
        print(f"转换时留 {keep.sum()} 帧（{keep.mean() * 100:.0f}%），裁掉 {n - keep.sum()} 帧静止帧")

        os.makedirs(OUT_DIR, exist_ok=True)
        decode = lambda c, i: cv2.imdecode(np.frombuffer(imgs[c][i], np.uint8), cv2.IMREAD_COLOR)
        h0, w0 = decode(cams[0], 0).shape[:2]
        avi = os.path.join(OUT_DIR, f"{name}.avi")
        vw = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w0 * len(cams), h0 + BAR))
        for i in range(n):
            tiles = []
            for c in cams:
                im = decode(c, i)
                cv2.putText(im, c, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if not keep[i]:
                    cv2.rectangle(im, (2, 2), (im.shape[1] - 3, im.shape[0] - 3), (0, 0, 255), 4)
                tiles.append(im)
            bar = np.zeros((BAR, w0 * len(cams), 3), np.uint8)
            text = (f"t={t[i] - t[0]:5.1f}s   gripper cmd: {'OPEN ' if cmd_open[i] else 'CLOSE'}  actual opening {grip[i]:.2f}"
                    f"   |cmd - actual| {err[i]:4.1f} deg   {'kept' if keep[i] else 'TRIMMED (idle)'}")
            cv2.putText(bar, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255) if keep[i] else (90, 90, 255), 2)
            vw.write(np.vstack([np.hstack(tiles), bar]))
        vw.release()

        # 缩略图：开头、每次夹爪动作和 1 秒后、结尾
        idx = [0] + [j for i, _ in events for j in (i, min(n - 1, i + int(fps)))] + [n - 1]
        cols = []
        for i in idx:
            col = [cv2.resize(decode(c, i), (w0 // 2, h0 // 2)) for c in cams]
            cv2.putText(col[0], f"{t[i] - t[0]:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cols.append(np.vstack(col))
        jpg = os.path.join(OUT_DIR, f"{name}.jpg")
        cv2.imwrite(jpg, np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"视频：{avi}\n缩略图：{jpg}")


if __name__ == "__main__":
    main()
