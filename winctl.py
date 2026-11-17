"""在 Orin 本机上控制三维示教的 Firefox 窗口：切换全屏（模拟按 F11）、最小化。

网页自己做不到这两件事（网页里的脚本不能最小化浏览器窗口，也退不出 F11 那种全屏），所以网页上的按钮让后台来做。
用的是系统本来就有的 X11 库（libX11、libXtst，ctypes 调用），不用另外装东西。只在网页开在 Orin 自己屏幕上时有用。

命令行：python winctl.py start        等窗口出来、拿到焦点后切到全屏（start_twin.sh 开网页时用）
        python winctl.py fullscreen   切换全屏 / 窗口
        python winctl.py minimize     最小化
"""
import ctypes
import os
import sys
import time

TITLE = "ECO65 三维示教"          # 网页标题（index.html 的 <title>），用它找到那个 Firefox 窗口
XK_F11 = 0xFFC8

c_void_p, c_ulong, c_long, c_int, c_uint = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long, ctypes.c_int, ctypes.c_uint


def _env():
    # 从桌面图标启动时本来就有 DISPLAY / XAUTHORITY；从 ssh 启动时没有，补上本机屏幕的
    os.environ.setdefault("DISPLAY", ":0")
    if "XAUTHORITY" not in os.environ:
        for p in (f"/run/user/{os.getuid()}/gdm/Xauthority", os.path.expanduser("~/.Xauthority")):
            if os.path.exists(p):
                os.environ["XAUTHORITY"] = p
                break


class _X:
    def __init__(self):
        _env()
        x = self.x = ctypes.cdll.LoadLibrary("libX11.so.6")
        self.xt = ctypes.cdll.LoadLibrary("libXtst.so.6")
        x.XOpenDisplay.restype, x.XOpenDisplay.argtypes = c_void_p, [ctypes.c_char_p]
        x.XDefaultRootWindow.restype, x.XDefaultRootWindow.argtypes = c_ulong, [c_void_p]
        x.XDefaultScreen.restype, x.XDefaultScreen.argtypes = c_int, [c_void_p]
        x.XInternAtom.restype, x.XInternAtom.argtypes = c_ulong, [c_void_p, ctypes.c_char_p, c_int]
        x.XGetWindowProperty.restype = c_int
        x.XGetWindowProperty.argtypes = [c_void_p, c_ulong, c_ulong, c_long, c_long, c_int, c_ulong,
                                         ctypes.POINTER(c_ulong), ctypes.POINTER(c_int), ctypes.POINTER(c_ulong),
                                         ctypes.POINTER(c_ulong), ctypes.POINTER(c_void_p)]
        x.XFree.argtypes = [c_void_p]
        x.XIconifyWindow.argtypes = [c_void_p, c_ulong, c_int]
        x.XKeysymToKeycode.restype, x.XKeysymToKeycode.argtypes = ctypes.c_ubyte, [c_void_p, c_ulong]
        x.XFlush.argtypes = [c_void_p]
        x.XCloseDisplay.argtypes = [c_void_p]
        self.xt.XTestFakeKeyEvent.argtypes = [c_void_p, c_uint, c_int, c_ulong]
        self.d = x.XOpenDisplay(None)
        if not self.d:
            raise RuntimeError(f"连不上屏幕（DISPLAY={os.environ.get('DISPLAY')}）")
        self.root = x.XDefaultRootWindow(self.d)

    def close(self):
        self.x.XCloseDisplay(self.d)

    def prop(self, win, name):
        """窗口属性：32 位的返回整数列表，8 位的返回 bytes；没有返回 None。"""
        atom = self.x.XInternAtom(self.d, name.encode(), 1)
        if not atom:
            return None
        typ, fmt, n, after, data = c_ulong(), c_int(), c_ulong(), c_ulong(), c_void_p()
        if self.x.XGetWindowProperty(self.d, win, atom, 0, 1 << 16, 0, 0, ctypes.byref(typ), ctypes.byref(fmt),
                                     ctypes.byref(n), ctypes.byref(after), ctypes.byref(data)) != 0 or not data.value:
            return None
        try:
            if fmt.value == 32:                  # Xlib 里 32 位格式的数据按 C long 存（64 位机上 8 字节）
                return list((c_ulong * n.value).from_address(data.value))
            return ctypes.string_at(data.value, n.value)
        finally:
            self.x.XFree(data)

    def find(self):
        """标题里有 TITLE 的 Firefox 窗口（有几个就取最后打开的）。"""
        found = None
        for w in self.prop(self.root, "_NET_CLIENT_LIST") or []:
            name = (self.prop(w, "_NET_WM_NAME") or b"").decode("utf-8", "ignore")
            if TITLE in name and "Firefox" in name:
                found = w
        return found

    def active(self):
        a = self.prop(self.root, "_NET_ACTIVE_WINDOW")
        return a[0] if a else None

    def states(self, win):
        names = []
        for a in self.prop(win, "_NET_WM_STATE") or []:
            self.x.XGetAtomName.restype, self.x.XGetAtomName.argtypes = c_void_p, [c_void_p, c_ulong]
            p = self.x.XGetAtomName(self.d, a)
            if p:
                names.append(ctypes.string_at(p).decode())
                self.x.XFree(p)
        return names

    def key(self, keysym):
        kc = self.x.XKeysymToKeycode(self.d, keysym)
        self.xt.XTestFakeKeyEvent(self.d, kc, 1, 0)
        self.xt.XTestFakeKeyEvent(self.d, kc, 0, 0)
        self.x.XFlush(self.d)


def do(action, wait=0.0):
    """action：fullscreen（切换全屏 / 窗口）、minimize（最小化）、start（等窗口出来并拿到焦点，再切到全屏）。
    成功返回 None，不行返回原因（给网页显示）。"""
    try:
        x = _X()
    except Exception as e:
        return f"控制不了窗口：{e}"
    try:
        t_end = time.time() + wait
        while True:
            w = x.find()
            if w and (action != "start" or x.active() == w):
                break
            if time.time() > t_end:
                return "没找到三维示教的 Firefox 窗口（要在 Orin 自己的屏幕上打开）" if not w else "Firefox 窗口没拿到焦点，没切全屏"
            time.sleep(0.3)
        if action == "minimize":
            x.x.XIconifyWindow(x.d, w, x.x.XDefaultScreen(x.d))
            x.x.XFlush(x.d)
        elif action in ("fullscreen", "start"):
            if action == "start" and "_NET_WM_STATE_FULLSCREEN" in x.states(w):
                return None
            if x.active() != w:
                return "Firefox 窗口没拿到焦点（先点一下网页）"
            x.key(XK_F11)                        # Firefox 自己的全屏：F11 再按一次回到窗口
        else:
            return f"不认识的操作：{action}"
        return None
    finally:
        x.close()


def state():
    """调试用：窗口找到没有、现在的状态。"""
    x = _X()
    try:
        w = x.find()
        return {"window": w, "active": x.active() == w if w else None, "states": x.states(w) if w else None}
    finally:
        x.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "state"
    if cmd == "state":
        print(state())
    else:
        r = do(cmd, wait=20.0 if cmd == "start" else 0.0)
        print(r or "好了")
