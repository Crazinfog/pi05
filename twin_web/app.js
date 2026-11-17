// 三维示教网页：显示 ECO65 模型，拖动夹爪控件，打关键帧，预览轨迹，让真机执行。
// 坐标和机械臂基座一致：x 前、y 左、z 上，单位米。逆解、安全检查、规划都在后台 twin_server.py 里做。
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { TransformControls } from './vendor/TransformControls.js';
import { STLLoader } from './vendor/STLLoader.js';

const $ = (id) => document.getElementById(id);
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// ---------------- 场景 ----------------
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf5f5f7);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
camera.position.set(1.25, -1.15, 0.95);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0.15, 0, 0.3);
orbit.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(1, -1, 2);
scene.add(sun);
scene.add(new THREE.AxesHelper(0.1));

let cfg = { tool_length: 0.175, tip_length: 0.2295, tip_clearance: 0.01, gripper_clearance: 0.03, table_z: 0, real: false, speed_limit: [0.2, 1.2] };
// 夹爪绕法兰轴线的安装角度（度）：2026-10-27 用户看真机发现手指开合方向和网页模型差约 90°。
// 不对的话改这个数（只影响网页上手指画在哪个方向，不影响位置计算）
const GRIPPER_YAW_DEG = 90;
const OPEN_TIP = 0.20;   // 法兰到指尖（夹爪张开时），画模型用；安全检查按全合时更长的 cfg.tip_length
const table = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.2),
  new THREE.MeshStandardMaterial({ color: 0xdfe3ea, transparent: true, opacity: 0.7 }));
table.position.set(0.35, 0, 0);
scene.add(table);
const grid = new THREE.GridHelper(1.4, 14, 0xb8bec9, 0xd3d7de);
grid.rotation.x = Math.PI / 2;
grid.position.set(0.35, 0, 0.0005);
scene.add(grid);
// 干活区域朝哪边（后台 config 的 home_yaw_deg，见 teleop.HOME_YAW_DEG）：桌面、网格、默认视角绕竖直轴转过去。只在第一次收到时转，
// 之后你自己转的视角不动
let yawApplied = false;
function applyHomeYaw(deg) {
  if (yawApplied || !deg) return;
  yawApplied = true;
  const a = THREE.MathUtils.degToRad(deg), rot = (v) => v.applyAxisAngle(new THREE.Vector3(0, 0, 1), a);
  rot(table.position); table.rotation.z += a;
  rot(grid.position);
  rot(camera.position); rot(orbit.target);
  orbit.update();
}

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ---------------- 机械臂模型（读 URDF） ----------------
const stlCache = {};
const stlLoader = new STLLoader();
function loadSTL(url) {
  if (!stlCache[url]) stlCache[url] = new Promise((ok, fail) => stlLoader.load(url, ok, undefined, fail));
  return stlCache[url];
}

async function makeRobot(material, withGripper) {
  const text = await (await fetch('./robot/ECO65-B.urdf')).text();
  const xml = new DOMParser().parseFromString(text, 'application/xml');
  const nums = (s) => (s || '0 0 0').trim().split(/\s+/).map(Number);
  const root = new THREE.Group();
  const linkObj = { base_link: root };
  const joints = [];
  for (const j of xml.querySelectorAll('joint')) {
    const o = j.querySelector('origin');
    const xyz = nums(o.getAttribute('xyz')), rpy = nums(o.getAttribute('rpy'));
    const originObj = new THREE.Group();
    originObj.position.set(...xyz);
    originObj.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');   // URDF 的 rpy：R = Rz(y)·Ry(p)·Rx(r)
    const jointObj = new THREE.Group();
    originObj.add(jointObj);
    linkObj[j.querySelector('parent').getAttribute('link')].add(originObj);
    linkObj[j.querySelector('child').getAttribute('link')] = jointObj;
    joints.push({ obj: jointObj, axis: new THREE.Vector3(...nums(j.querySelector('axis').getAttribute('xyz'))) });
  }
  for (const l of xml.querySelectorAll('link')) {
    const mesh = l.querySelector('visual mesh');
    if (!mesh) continue;
    const url = './robot/meshes/' + mesh.getAttribute('filename').split('/').pop();
    loadSTL(url).then((geo) => linkObj[l.getAttribute('name')].add(new THREE.Mesh(geo, material)));
  }
  const flange = linkObj['link_6'];
  let fingers = null;
  if (withGripper) {
    // 简化的夹爪：一根圆柱（法兰到指尖）+ 两根手指 + 抓取中心的小球
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.03, OPEN_TIP - 0.06, 20), material);
    body.rotation.x = Math.PI / 2;
    body.position.z = (OPEN_TIP - 0.06) / 2;
    g.add(body);
    fingers = [-1, 1].map((s) => {
      const f = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.025, 0.06), material);
      f.position.set(s * 0.045, 0, OPEN_TIP - 0.03);
      g.add(f);
      return f;
    });
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.008, 16, 12), new THREE.MeshBasicMaterial({ color: 0xff8800 }));
    dot.position.z = cfg.tool_length;
    g.add(dot);
    g.rotation.z = THREE.MathUtils.degToRad(GRIPPER_YAW_DEG);   // 夹爪装在法兰上转了一个角度，手指开合方向跟着转
    flange.add(g);
  }
  // 夹爪开口：width 0 闭合 ~ 1 张开（知行 CTAG2F90 最大开口 90 mm）。setGripper 设目标，update 每帧慢慢动过去
  const grip = { w: 1, goal: 1 };
  function placeFingers() {
    if (fingers) fingers.forEach((f, i) => {
      f.position.x = (i ? 1 : -1) * (0.006 + 0.039 * grip.w);
      f.position.z = OPEN_TIP - 0.03 + extend(grip.w);
    });
  }
  return {
    root,
    setJoints(qDeg) {
      qDeg.forEach((q, i) => joints[i] && joints[i].obj.quaternion.setFromAxisAngle(joints[i].axis, THREE.MathUtils.degToRad(q)));
    },
    setGripper(width, instant) {        // width：0~1；也可以传 true（合上）/ false（张开）
      grip.goal = typeof width === 'boolean' ? (width ? 0 : 1) : width;
      if (instant) { grip.w = grip.goal; placeFingers(); }
    },
    update(dt) {                        // 实测全程开合约 0.25 秒
      const step = dt / 0.25;
      grip.w += Math.max(-step, Math.min(step, grip.goal - grip.w));
      placeFingers();
    },
  };
}

// 连杆夹爪合上时手指沿圆弧往前伸，指尖最多比张开时再伸出约 3 cm（和 teleop.py 的 FINGER_GAP_MM/FINGER_EXT_MM 同一张表）。
// w：开口 0 闭合 ~ 1 全开（1 = GRIPPER_STROKE 87.5 mm）
const GRIPPER_STROKE = 0.0875;
const EXT = [[0, 0.0295], [0.125, 0.0287], [0.247, 0.0274], [0.366, 0.0255], [0.479, 0.0232], [0.586, 0.0203],
             [0.687, 0.017], [0.779, 0.0133], [0.863, 0.0092], [0.936, 0.0048], [1, 0]];
function extend(w) {
  for (let i = 1; i < EXT.length; i++) if (w <= EXT[i][0]) {
    const [g0, e0] = EXT[i - 1], [g1, e1] = EXT[i];
    return e0 + (e1 - e0) * (w - g0) / (g1 - g0);
  }
  return 0;
}

// 仿真夹爪（目标那台灰色模型的夹爪）和真实夹爪分开（2026-11-02 用户要求）：
// 仿真夹爪只在网页里开合，记录关键帧时记的是它；真实夹爪用“机械臂”那一节的按钮单独开合。
let simGrip = 'open', realGripClosed = false, realGripWidth = 1;
function simWidth() {                    // 仿真夹爪合上时画成夹着“物体宽度”那么宽的东西
  return simGrip === 'open' ? 1 : Math.min(1, Math.max(0, (+$('objW').value || 0) / 100 / GRIPPER_STROKE));
}
function simExt() {
  // 仿真夹爪的指尖比全开时多伸出多少（离桌面检查用）。和规划一样以用户为准：只有勾了“按物体宽度检查合上时的指尖”
  // 才按物体宽度算合上的伸出量，否则按张开时的长度（只查指尖离桌 3 mm）
  return simGrip === 'close' && $('checkClose').checked ? extend(simWidth()) : 0;
}
function showSimGrip() {
  if (preview && !playing) preview.setGripper(simWidth());
  $('simOpen').className = simGrip === 'open' ? 'pick' : '';
  $('simClose').className = simGrip === 'close' ? 'pick' : '';
}
function setSimGrip(g) {
  if (g === simGrip) return;
  simGrip = g;
  showSimGrip();
  keepAboveTable();                      // 合上（勾了检查）时指尖更长，目标可能要往上推
  showTarget();
  requestSolve(true);
}

const previewMat = new THREE.MeshStandardMaterial({ color: 0xe8e8e8, metalness: 0.1, roughness: 0.6 });
const actualMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.28, depthWrite: false });
let preview = null, actual = null;

// ---------------- 拖动控件（目标 = 抓取中心的位姿） ----------------
const target = new THREE.Object3D();
target.add(new THREE.AxesHelper(0.07));
scene.add(target);
const tc = new TransformControls(camera, renderer.domElement);
tc.setSize(0.8);
tc.setSpace('world');
tc.attach(target);
scene.add(tc);
tc.addEventListener('dragging-changed', (e) => (orbit.enabled = !e.value));
// 拖目标时直接挡住：夹爪低于桌面安全线就把目标往上推回去，拖不下去。
// 指尖按仿真夹爪算（simExt：张开、或合上但没勾检查时按张开长度），不看真夹爪（2026-11-02 起两个夹爪分开）
function keepAboveTable() {      // 夹爪按真实朝向的长方体算（和后台 teleop.model_margin 一样），最低点要离桌面 cfg.model_clearance
  const flange = target.position.clone().addScaledVector(new THREE.Vector3(0, 0, 1).applyQuaternion(target.quaternion), -cfg.tool_length);
  const tip = OPEN_TIP + simExt();
  let need = -Infinity;            // 夹爪本体留 model_clearance（1 cm），手指那一段只留 finger_clearance（3 mm，指尖本来就要贴近桌面）
  for (const [hx, hy, z0, z1, gap] of [[0.025, 0.0353, 0, 0.134, cfg.model_clearance ?? 0.01],
                                       [0.0125, 0.051, 0.134, tip, cfg.finger_clearance ?? 0.003]])
    for (const x of [-hx, hx]) for (const y of [-hy, hy]) for (const z of [z0, z1]) {
      const p = new THREE.Vector3(x, y, z).applyQuaternion(target.quaternion).add(flange);
      need = Math.max(need, cfg.table_z + gap - p.z);
    }
  if (need > 0) target.position.z += need;
}
tc.addEventListener('objectChange', () => {
  if (vertLock && tc.getMode() === 'rotate') target.quaternion.copy(vertQuat(yawOf(target.quaternion)));   // 只留转向
  keepAboveTable(); showTarget(); requestSolve();
  if (following) send({ type: 'goal', ...targetPose() });
});

// ---------------- 竖直抓：只调 xyz 和夹爪绕竖直轴转（2026-11-04 用户：只要夹爪正着抓，不要别的自由度） ----------------
// 竖直朝下、转向 yaw 的姿态 = 绕世界竖直轴转 yaw · 绕 x 轴翻 180°（夹爪轴＝末端 z 轴朝下，和初始位姿一样）
let vertLock = true;
try { vertLock = localStorage.getItem('vertLock') !== '0'; } catch (e) { /* 读不了就默认开 */ }
const AXIS_X = new THREE.Vector3(1, 0, 0), AXIS_Z = new THREE.Vector3(0, 0, 1);
function vertQuat(yaw) {
  return new THREE.Quaternion().setFromAxisAngle(AXIS_Z, yaw).multiply(new THREE.Quaternion().setFromAxisAngle(AXIS_X, Math.PI));
}
function yawOf(q) {        // 夹爪现在的转向：末端 x 轴在水平面里的方向（x 轴接近竖直时改用 y 轴）
  const x = AXIS_X.clone().applyQuaternion(q), y = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
  return Math.hypot(x.x, x.y) > 0.5 ? Math.atan2(x.y, x.x) : Math.atan2(y.x, -y.y);
}
function updateGizmo() {   // 竖直抓开着时，旋转控件只剩绕竖直轴那一圈（世界坐标）
  const only = vertLock && tc.getMode() === 'rotate';
  tc.showX = !only; tc.showY = !only; tc.showZ = true;
  if (vertLock) tc.setSpace('world');
}
function targetMovedByUser(force = true) {
  keepAboveTable(); showTarget(); requestSolve(force);
  if (following) send({ type: 'goal', ...targetPose() });
}
function setVertLock(on) {
  vertLock = on;
  try { localStorage.setItem('vertLock', on ? '1' : '0'); } catch (e) { /* 存不了就算了 */ }
  $('vertLock').checked = on;
  updateGizmo();
  if (on && tiltDeg(target.quaternion) > 0.5) {          // 打开时先把夹爪摆成竖直朝下（转向不变）
    target.quaternion.copy(vertQuat(yawOf(target.quaternion)));
    targetMovedByUser();
  }
}
// ---------------- 键盘点动：WASD 水平移动、R/F 上下、Q/E 转夹爪，真机跟着动 ----------------
// 2026-11-04 用户：WASD 控制末端在水平面移动、R/F 上下、Q 顺时针 / E 逆时针转夹爪，务必真机跟着动（点云的框不准，要看着真夹爪对齐）。
// 方向按世界坐标、跟着初始位姿的朝向（2026-11-04 用户：按屏幕不直观）：W 沿初始位姿朝向往前（home_yaw_deg，现在是 +y）、
// S 往后、A 左、D 右，不随视角变。按一下走一小步（5 mm / 5°，Shift 1 mm / 1°），
// 按住超过 0.3 秒就匀速一直走（3 cm/s、30°/s，Shift 0.5 cm/s、5°/s），几个键可以一起按（斜着走）。不用键盘自己的连发。
// 真机：一串按键开始时目标如果不在真机那儿，先把目标放回真机现在的位置（不会一按就冲去别处），之后每动一下把目标位姿发给后台，
// 后台限速跟过去（逐步逆解、安全检查，跟不上 / 碰到限制就停）。开着跟随时由跟随去跟；正在执行、拖动示教时只动模型
const JOG_KEYS = { KeyW: [0, 1, 0], KeyS: [0, -1, 0], KeyA: [-1, 0, 0], KeyD: [1, 0, 0], KeyR: [0, 0, 1], KeyF: [0, 0, -1],
                   KeyQ: -1, KeyE: 1 };      // 数字：绕竖直轴转的方向（从上往下看逆时针为正）
const MOVE_STEP = 0.005, MOVE_STEP_FINE = 0.001, MOVE_RATE = 0.03, MOVE_RATE_FINE = 0.005;
const YAW_STEP = 5, YAW_STEP_FINE = 1, YAW_RATE = 30, YAW_RATE_FINE = 5, HOLD_S = 0.3;
const held = new Map();                 // 按住的键 → { fine, t0 }
let jogUntil = 0, jogLive = false, holdLast = 0;
function viewAxes() {                   // “前”＝初始位姿朝向（世界坐标水平面里），“右”＝面朝前时的右手边
  const a = THREE.MathUtils.degToRad(cfg.home_yaw_deg ?? 90);
  const f = new THREE.Vector3(Math.cos(a), Math.sin(a), 0);
  return { f, r: new THREE.Vector3().crossVectors(f, AXIS_Z).normalize() };
}
function nearActual() {                 // 目标就在真机现在的位置（3 mm、2° 以内）
  if (!lastActual) return false;
  const c = centerFromFlange(lastActual);
  return target.position.distanceTo(new THREE.Vector3(...c.pos)) < 0.003 &&
    THREE.MathUtils.radToDeg(new THREE.Quaternion().fromArray(c.quat).angleTo(target.quaternion)) < 2;
}
function jogBegin() {                   // 一串按键开始：决定真机跟不跟
  if (jogLive && performance.now() > jogUntil) jogLive = false;
  jogUntil = performance.now() + 1500;
  if (jogLive || following) return;
  const info = $('solveinfo');
  if (robotExec || dragging || playing) {
    info.className = 'warn';
    info.textContent = '正在执行 / 拖动示教：按键只动了模型，真机没跟着动';
    return;
  }
  if (!lastActual || !prevActualQ) return;
  if (!nearActual()) {                  // 从真机现在的位置开始动，不会一按就冲去别处
    const c = centerFromFlange(lastActual);
    setTargetFromJoints(c.pos, c.quat, prevActualQ);
    info.className = 'warn';
    info.textContent = '目标先放回了真机现在的位置，按键时真机跟着动';
  }
  jogLive = true;
}
function applyNudge(code, amount) {     // amount：米 / 度
  const v = JOG_KEYS[code];
  if (typeof v === 'number') {
    const d = THREE.MathUtils.degToRad(v * amount);
    if (vertLock) target.quaternion.copy(vertQuat(yawOf(target.quaternion) + d));
    else target.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(AXIS_Z, d));
  } else {
    const { f, r } = viewAxes();
    target.position.addScaledVector(r, v[0] * amount).addScaledVector(f, v[1] * amount);
    target.position.z += v[2] * amount;
  }
}
function afterNudge(force) {
  targetMovedByUser(force);             // 挡桌面、显示、逆解，开着跟随时发跟随目标
  jogUntil = performance.now() + 1500;
  if (jogLive && !following && !robotExec) send({ type: 'jog', ...targetPose() });
}
function holdTick() {
  if (!held.size) return;
  const now = performance.now(), dt = Math.min(0.1, (now - holdLast) / 1000);
  holdLast = now;
  let any = false;
  for (const [code, h] of held) {
    if (now - h.t0 < HOLD_S * 1000) continue;
    const rot = typeof JOG_KEYS[code] === 'number';
    applyNudge(code, (rot ? (h.fine ? YAW_RATE_FINE : YAW_RATE) : (h.fine ? MOVE_RATE_FINE : MOVE_RATE)) * dt);
    any = true;
  }
  if (any) afterNudge(false); else jogUntil = now + 1500;
  requestAnimationFrame(holdTick);
}
function jogKeyDown(code, fine) {
  if (held.has(code)) return;
  jogBegin();
  const rot = typeof JOG_KEYS[code] === 'number';
  applyNudge(code, rot ? (fine ? YAW_STEP_FINE : YAW_STEP) : (fine ? MOVE_STEP_FINE : MOVE_STEP));
  afterNudge(true);
  if (!held.size) { holdLast = performance.now(); requestAnimationFrame(holdTick); }
  held.set(code, { fine, t0: performance.now() });
}
function jogKeyUp(code) {
  if (held.delete(code) && !held.size) requestSolve(true);   // 停下的那一刻再解一次，模型和关节框落在最后的位置上
}
$('vertLock').onchange = () => setVertLock($('vertLock').checked);
for (const [id, code] of [['yawL', 'KeyE'], ['yawR', 'KeyQ']]) {        // 左下角两个按钮：和 E / Q 一样，按住一直转
  $(id).addEventListener('pointerdown', (e) => { e.preventDefault(); jogKeyDown(code, e.shiftKey); });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) $(id).addEventListener(ev, () => jogKeyUp(code));
}
window.addEventListener('keyup', (e) => { if (JOG_KEYS[e.code] !== undefined) jogKeyUp(e.code); });
window.addEventListener('blur', () => { for (const c of [...held.keys()]) jogKeyUp(c); });
$('vertLock').checked = vertLock;
updateGizmo();
let targetReady = false;

function targetPose() {
  return { pos: target.position.toArray(), quat: target.quaternion.toArray() };   // 四元数顺序 x y z w
}
function setTarget(pos, quat) {
  target.position.fromArray(pos);
  target.quaternion.fromArray(quat);
  targetReady = true;
  showTarget();
  requestSolve(true);
}
function tiltDeg(q) {   // 夹爪轴（末端 z 轴）偏离竖直朝下多少度
  const z = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
  return THREE.MathUtils.radToDeg(Math.acos(Math.max(-1, Math.min(1, -z.z))));
}
function showTarget() {
  const p = target.position;
  $('target').innerHTML = `x ${(p.x * 1000).toFixed(0)} · y ${(p.y * 1000).toFixed(0)} · z ${(p.z * 1000).toFixed(0)} mm，倾斜 ${tiltDeg(target.quaternion).toFixed(0)}°`;
}
// 实际法兰位姿（xyz + 外旋 xyz 欧拉角）→ 抓取中心位姿
function centerFromFlange(pose) {
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(pose[3], pose[4], pose[5], 'ZYX'));
  const p = new THREE.Vector3(0, 0, cfg.tool_length).applyQuaternion(q).add(new THREE.Vector3(pose[0], pose[1], pose[2]));
  return { pos: p.toArray(), quat: q.toArray() };
}

// ---------------- 和后台通信 ----------------
let ws, lastActual = null, lastQ = null, solveBusy = false, solvePending = false, solveId = 0, playing = false;
let solveSent = 0, solveOk = null;     // 最近一次逆解请求的时刻、结果能不能到（设抓取点/放置点时要用）
let prevActualQ = null;                // 上一次收到的实际关节角：看机械臂是不是被网页以外的方式挪了
let extUntil = 0;                      // 机械臂被外面挪动以后，这个时刻之前小的挪动也让目标跟着
let robotExec = false, execEndedAt = 0; // 真机在不在执行轨迹 / 移到目标；上一次什么时候停的（空格键用）
function send(m) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); }
// 目标位置和关节角要是同一个时刻的：调关节、点手臂摆法以后位置要等后台算完正解才更新，拖目标以后关节角要等逆解回来才更新。
// 这时候点“设为抓取点 / 放置点”“记录当前目标”会存成“旧位置 + 新关节角”（2026-10-28 用户发现点了坐标没变）。
// 所以这几个按钮都先等算完（最多等 1.5 秒）再存
let pendingFn = null;
function targetBusy() {
  const now = performance.now();
  return (fkBusy && now - fkSent < 1500) || ((solveBusy || solvePending) && now - solveSent < 1500);
}
function whenTargetReady(fn) {
  if (!targetBusy()) { pendingFn = null; fn(); return; }
  pendingFn = fn;
  setTimeout(flushPending, 1600);      // 万一回复丢了，1.5 秒后照样存
}
function flushPending() {
  if (pendingFn && !targetBusy()) { const fn = pendingFn; pendingFn = null; fn(); }
}
function requestSolve(force) {
  if (playing) return;
  if (solveBusy && !force) { solvePending = true; return; }
  solveBusy = true;
  solveSent = performance.now();
  send({ type: 'solve', id: ++solveId, ...targetPose(), seed: lastQ, ext: simExt() });
}
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onclose = () => { $('mode').textContent = '连接断开，重连中…'; setTimeout(connect, 1500); };
  ws.binaryType = 'arraybuffer';         // 点云用二进制发
  ws.onmessage = (ev) => (ev.data instanceof ArrayBuffer ? onCloud(ev.data) : onMessage(JSON.parse(ev.data)));
}

function onMessage(m) {
  if (m.type === 'config') {
    cfg = { ...cfg, ...m };
    applyHomeYaw(m.home_yaw_deg);
    if (m.calib !== undefined) {
      $('calibInfo').textContent = m.calib;
      $('calibInfo').className = m.calibrated ? 'ok' : 'warn';
      if (!m.calibrated) $('cloudInfo').textContent = '先在“相机外参标定”里标一下外参，点云才能和机械臂对上';
    }
    $('mode').textContent = m.offline ? '离线模拟：没连机械臂' : m.real ? '● 真实模式：机械臂会真的动（点击切到仿真）' : '○ 仿真模式：真机不动（点击切到真实）';
    $('mode').className = 'mode ' + (m.real ? 'real' : 'sim');
    $('speed').max = Math.round(m.speed_limit[0] * 100);
    $('ang').max = Math.round(THREE.MathUtils.radToDeg(m.speed_limit[1]));
    if (+$('speed').value > +$('speed').max) $('speed').value = $('speed').max;
    if (+$('ang').value > +$('ang').max) $('ang').value = $('ang').max;
    showSpeeds();
    buildJoints();
    send({ type: 'list' });
  } else if (m.type === 'state') {
    if (robotExec && !m.executing) execEndedAt = performance.now();
    robotExec = !!m.executing;
    if (m.q && actual) actual.setJoints(m.q);
    lastActual = m.pose;
    if (!targetReady && m.pose) { const c = centerFromFlange(m.pose); setTarget(c.pos, c.quat); }
    $('status').textContent = m.status + (m.executing ? `（${m.progress[0]}/${m.progress[1]}）` : '')
      + (m.gripper_msg ? ' · ' + m.gripper_msg : '');
    if (m.gripper_closed !== undefined) {
      realGripClosed = !!m.gripper_closed;
      realGripWidth = m.gripper_width ?? (realGripClosed ? 0 : 1);
      $('realGripState').textContent = (realGripClosed ? '合上' : '张开') + ` · 开口 ${(realGripWidth * GRIPPER_STROKE * 1000).toFixed(0)} mm`;
    }
    if (m.executing && m.progress[1]) $('progress').firstChild.style.width = (100 * m.progress[0] / m.progress[1]) + '%';
    if (actual) actual.setGripper(m.gripper_width ?? (m.gripper_closed ? 0 : 1));
    following = !!m.follow_on;
    recording = m.recording ?? null;           // 正在录：已经录了几秒；没在录是 null
    $('camWarn').hidden = !m.cam_problem;      // 录示范要两台相机都有画面
    if (m.policy) { $('polInfo').className = ''; $('polInfo').textContent = m.policy; }   // 模型控制中：每 0.1 秒一行状态
    if (m.cam_problem) $('camWarn').textContent = '⚠ ' + m.cam_problem;
    $('recBadge').hidden = recording === null;
    showRecRun();                              // 正在录时收起来（左上角显示“● 录制中”）
    if (recording !== null) $('recBadge').textContent = `● 录制中 ${recording.toFixed(1)} 秒`;
    $('recManual').textContent = recording !== null && !m.executing ? `停止并保存（${recording.toFixed(0)} 秒）` : '手动开始录';
    $('recManual').className = recording !== null && !m.executing ? 'on' : '';
    dragging = !!m.drag_on;
    $('drag').textContent = '拖动示教：' + (dragging ? '开（点击关闭）' : '关');
    $('drag').className = dragging ? 'on' : '';
    // 机械臂不是网页在驱动（没在执行、没开跟随）却动了——网页的拖动示教、末端拖动按钮、示教器挪的——
    // 目标、模型、关节框都跟到实际位置。之前只有网页的拖动示教会跟，用手挪了真机以后“设为放置点”
    // 存的还是原来的目标（2026-10-28 用户发现）
    const moved = m.q && prevActualQ ? Math.max(...m.q.map((v, j) => Math.abs(v - prevActualQ[j]))) : 0;
    if (m.q) prevActualQ = [...m.q];
    // 0.1 秒里转了 0.2° 以上才算在动（读数抖动约 0.01°）；动起来以后 1 秒内小的挪动也跟（最后几毫米走得慢，
    // 原来每 0.1 秒不到 0.2° 就不跟了，目标停在离真机停下的地方还差几毫米）
    // 网页在驱动（执行轨迹、跟随、Q/E 带着真机转夹爪）时的运动不算“被挪动”（转夹爪时原来被当成外面挪的，目标被拉回真机、越转越慢）
    const byPage = m.executing || following || m.jog_on || performance.now() < jogUntil;
    if (moved > 0.2 && !byPage) extUntil = performance.now() + 1000;
    const external = !byPage && (moved > 0.2 || (performance.now() < extUntil && moved > 0.02));
    if ((dragging || external) && m.pose && m.q && !playing) {
      const c = centerFromFlange(m.pose);
      setTargetFromJoints(c.pos, c.quat, m.q);
      if (external && !dragging) { $('solveinfo').className = 'warn'; $('solveinfo').textContent = '机械臂被挪动了，目标跟到了实际位置'; }
    }
    $('follow').textContent = '跟随模式：' + (following ? '开（点击关闭）' : '关');
    $('follow').className = following ? 'on' : '';
    if (m.follow_msg) { $('followinfo').textContent = m.follow_msg; $('followinfo').className = m.follow_msg.startsWith('停') || m.follow_msg.includes('已停') ? 'bad' : 'warn'; }
  } else if (m.type === 'solve') {
    solveBusy = false;
    if (m.q && preview && !playing) { preview.setJoints(m.q); lastQ = m.q; showJoints(m.q); }
    requestBranches();
    const info = $('solveinfo');
    if (m.ok) { info.className = 'ok'; info.textContent = `能到达 · 离桌面安全线还有 ${m.margin} mm`; }
    else { info.className = 'bad'; info.textContent = '到不了：' + m.reason + (m.q ? '' : '（模型停在上一个能到的姿态）'); }
    previewMat.color.set(m.ok ? 0xe8e8e8 : 0xffb3b3);
    solveOk = m.ok;
    if (solvePending) { solvePending = false; requestSolve(); }
    flushPending();
  } else if (m.type === 'cam_jpeg') {
    camWaiting = false;
    if (m.which && m.which !== camWhich()) return;     // 刚切换了相机，上一台的画面不要了
    if (m.data) { $('camImg').src = 'data:image/jpeg;base64,' + m.data; camFrames++; }
    else $('camInfo').textContent = m.text || '没有画面';
  } else if (m.type === 'rec') {
    onRec(m);
  } else if (m.type === 'ls_dir') {
    onLsDir(m);
  } else if (m.type === 'calib') {
    onCalib(m);
  } else if (m.type === 'segment') {
    onSegment(m);
  } else if (m.type === 'grasp') {
    onGrasp(m);
  } else if (m.type === 'align') {
    onAlign(m);
  } else if (m.type === 'autogen') {
    onAutogen(m);
  } else if (m.type === 'branches') {
    if (m.id === branchId) showBranches(m.items);
  } else if (m.type === 'fk') {
    if (m.id !== fkId) return;            // 只认最新一次
    fkBusy = false;
    setTargetFromJoints(m.pos, m.quat, m.q);
    requestBranches();
    const info = $('solveinfo');
    info.className = m.ok ? 'ok' : 'bad';
    info.textContent = m.ok ? '关节模式：这个摆法安全' : '关节模式：' + m.reason;
    previewMat.color.set(m.ok ? 0xe8e8e8 : 0xffb3b3);
    solveOk = m.ok;
    flushPending();
  } else if (m.type === 'plan') {
    showPlan(m);
    if (m.refused) alert(m.refused);
    else playPlan(m);
  } else if (m.type === 'started') {
    $('planinfo').className = 'warn';
    $('planinfo').textContent = (m.text || `执行中，约 ${m.duration} 秒`) + '。随时可以点“停止”或“急停”';
    if (m.text) { $('solveinfo').className = 'warn'; $('solveinfo').textContent = m.text + '（空格停止）'; }
  } else if (m.type === 'done') {
    const arrived = m.status.startsWith('已移到');
    if (arrived) { $('solveinfo').className = 'ok'; $('solveinfo').textContent = m.status; }
    $('planinfo').className = m.status.startsWith('执行完成') || arrived ? 'ok' : 'bad';
    $('planinfo').textContent = m.status + '（日志 ' + m.log.split('/').pop() + '）';
    // 执行完一条轨迹：控件放回实际位置（后台等真机停稳了才报完成）。
    // 移到目标 / 某一帧（M、“真机移到这帧”）成功了：目标本来就是要去的那个点，原样留着，不拿实际位置去覆盖——
    // 实际位置和指令总差一点（停稳时 ≤1.5 mm），而且原来报完成时真机还在路上，目标被换成了路上的点，
    // 接着“设为抓取点”就存偏了 6 mm（2026-11-02）
    if (!(m.goto && arrived)) { targetReady = false; lastQ = null; }
    solveBusy = solvePending = false;   // 执行时后台不回逆解请求，别让“正在算”一直挂着（不然之后拖目标不再更新）
    if (m.goto && arrived) requestSolve(true);
    // 录完一条：随机种子换掉，同一组关键帧再录一条时起点扰动、执行扰动不一样（不然每条都是同一组扰动）
    if (!m.goto && $('recOn').checked && planSeed !== null) {
      planSeed = null;
      $('planinfo').textContent += '；下一条会换一组起点扰动和执行扰动（想先看就点“预览”）';
    }
  } else if (m.type === 'policy') {
    $('polInfo').className = m.ok ? 'ok' : 'bad';
    $('polInfo').textContent = m.text;
    if (m.done || !m.ok) { $('solveinfo').className = m.ok ? 'ok' : 'bad'; $('solveinfo').textContent = m.text; }
  } else if (m.type === 'msg') {
    $('planinfo').className = 'muted';
    $('planinfo').textContent = m.text;
    if (m.goto) { $('solveinfo').className = m.text.startsWith('没有移动') ? 'bad' : 'muted'; $('solveinfo').textContent = m.text; }
  } else if (m.type === 'templates') {
    $('tplList').innerHTML = m.names.map((n) => `<option>${n}</option>`).join('');
  } else if (m.type === 'template') {
    keyframes = m.data.keyframes || [];   // 只载入关键帧；速度和扰动滑块保持推荐值，不用模板里存的
    planSeed = null;
    selected = keyframes.length ? 0 : -1;
    renderKeyframes();
    $('tplName').value = m.name;
  }
}

// ---------------- 关键帧 ----------------
let keyframes = [], selected = -1;
const kfGroup = new THREE.Group();
scene.add(kfGroup);
function renderKeyframes() {
  $('kflist').innerHTML = keyframes.map((k, i) => `
    <li data-i="${i}" class="${i === selected ? 'sel' : ''}">
      <div class="top"><b>帧 ${i + 1}${k.tag ? ' · ' + k.tag : ''}</b><span class="muted">${k.pos.map((v) => (v * 1000).toFixed(0)).join(', ')} mm · 倾斜 ${tiltDeg(new THREE.Quaternion(...k.quat)).toFixed(0)}°</span></div>
      <div class="sub">
        <label>夹爪 <select data-f="gripper" data-i="${i}">
          <option value="" ${!k.gripper ? 'selected' : ''}>不变</option>
          <option value="open" ${k.gripper === 'open' ? 'selected' : ''}>张开</option>
          <option value="close" ${k.gripper === 'close' ? 'selected' : ''}>合上</option></select></label>
        <label>停顿 <input data-f="pause" data-i="${i}" type="number" min="0" max="10" step="0.5" value="${k.pause || 0}"> 秒</label>
        <span class="muted">到这帧时夹爪${gripAt(i) === 'close' ? '合着' : '张着'}</span>
        <button data-goto="${i}" title="真机移到这一帧：夹爪朝下时先到正上方（路上手腕不拧）、原地转、再竖直下去（途中逐点检查、限速，能停能急停）">真机移到这帧</button>
      </div></li>`).join('');
  // 场景里画出关键帧位置和连线
  kfGroup.clear();
  const pts = keyframes.map((k) => new THREE.Vector3(...k.pos));
  pts.forEach((p, i) => {
    const s = new THREE.Mesh(new THREE.SphereGeometry(i === selected ? 0.012 : 0.008, 12, 8),
      new THREE.MeshBasicMaterial({ color: i === selected ? 0x2f6feb : 0x8250df }));
    s.position.copy(p);
    s.userData.kf = i;                    // 右键点到小球时知道是哪一帧
    kfGroup.add(s);
  });
  if (pts.length > 1) kfGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x8250df })));
}
// 第 i 帧时夹爪是开是合：往前找最近一个写了开合的帧（都没写就当张开，预览也是从张开开始）
function gripAt(i) {
  for (let j = Math.min(i, keyframes.length - 1); j >= 0; j--) if (keyframes[j].gripper) return keyframes[j].gripper;
  return 'open';
}
// 记录 / 更新第 i 帧时，夹爪那一栏按仿真夹爪填：和上一帧时的状态一样就填“不变”，不一样就填张开或合上。
// 第 1 帧总是写明（执行时先让真夹爪对上）
function gripFieldFor(i) {
  return i === 0 ? simGrip : (simGrip === gripAt(i - 1) ? '' : simGrip);
}
$('kflist').addEventListener('click', (e) => {
  const go = e.target.closest('button[data-goto]');
  if (go) { gotoKeyframe(+go.dataset.goto); return; }
  const li = e.target.closest('li');
  if (!li || e.target.closest('select,input,button')) return;
  selected = +li.dataset.i;
  const k = keyframes[selected];
  if (k.q) setTargetFromJoints(k.pos, k.quat, k.q); else setTarget(k.pos, k.quat);
  setSimGrip(gripAt(selected));          // 仿真夹爪也摆成这一帧时的样子
  renderKeyframes();
});
$('kflist').addEventListener('change', (e) => {
  const f = e.target.dataset.f, i = +e.target.dataset.i;
  if (f === 'gripper') {
    keyframes[i].gripper = e.target.value;
    if (selected >= 0) setSimGrip(gripAt(selected));
    renderKeyframes();
  }
  if (f === 'pause') keyframes[i].pause = Math.max(0, +e.target.value);
});
$('kfAdd').onclick = () => whenTargetReady(() => {   // 等目标位置和关节角对上了再记（见 whenTargetReady）
  if (!targetReady) return;
  const i = selected + 1;                 // 插在所选的后面
  const k = { ...targetPose(), q: lastQ ? [...lastQ] : null, gripper: gripFieldFor(i), pause: 0 };
  keyframes.splice(i, 0, k);
  selected = i;
  renderKeyframes();
});
$('kfUpdate').onclick = () => whenTargetReady(() => {
  if (selected < 0) return;
  Object.assign(keyframes[selected], targetPose(), { q: lastQ ? [...lastQ] : null, gripper: gripFieldFor(selected) });
  renderKeyframes();
});
// 真机移到目标（灰色仿真模型现在的位置，快捷键 空格 / M）：按模型现在的手臂摆法（lastQ）用关节运动过去，和“真机移到这帧”走同一个后台 goto。
// 只动手臂，夹爪不动（两个夹爪分开）。目标是红的（到不了）就不动；真实模式先确认（按回车就是确定）
function gotoTarget() {
  whenTargetReady(() => {
    const info = $('solveinfo');
    if (!targetReady) return;
    if (solveOk === false) { info.className = 'bad'; info.textContent = '目标现在到不了（模型是红的），真机没动'; return; }
    playing = false;
    info.className = 'muted';
    info.textContent = '正在规划到目标的路…';
    send({ type: 'goto', name: '目标位置', keyframe: { ...targetPose(), q: lastQ ? [...lastQ] : null }, gripper: null });
  });
}
$('toTarget').onclick = gotoTarget;

// 真机移到第 i 帧：关节运动过去（后台 goto：和执行的出发段一样逐点检查、限速、盯偏差），勾着的话到了再把真夹爪开合成这一帧的样子
function gotoKeyframe(i) {
  const k = keyframes[i];
  if (!k) return;
  const g = $('gotoGrip').checked ? gripAt(i) : null;
  playing = false;
  selected = i;
  if (k.q) setTargetFromJoints(k.pos, k.quat, k.q); else setTarget(k.pos, k.quat);   // 目标也放到这一帧（移完原样留着）
  setSimGrip(gripAt(i));                 // 和点选这一帧一样：仿真夹爪摆成这一帧的样子
  renderKeyframes();
  $('planinfo').className = 'muted';
  $('planinfo').textContent = `正在规划到第 ${i + 1} 帧的路…`;
  const gw = k.grip_width !== undefined ? k.grip_width : (g === 'close' ? closeWidth() : null);   // 到了合夹爪时按物体宽度查指尖
  send({ type: 'goto', index: i, keyframe: { pos: k.pos, quat: k.quat, q: k.q, grip_width: gw }, gripper: g });
}
$('kfDel').onclick = () => { if (selected >= 0) { keyframes.splice(selected, 1); selected = Math.min(selected, keyframes.length - 1); renderKeyframes(); } };
$('kfClear').onclick = () => { if (confirm('清空所有关键帧？')) { keyframes = []; selected = -1; renderKeyframes(); } };
function move(d) {
  const j = selected + d;
  if (selected < 0 || j < 0 || j >= keyframes.length) return;
  [keyframes[selected], keyframes[j]] = [keyframes[j], keyframes[selected]];
  selected = j;
  renderKeyframes();
}
$('kfUp').onclick = () => move(-1);
$('kfDown').onclick = () => move(1);



// ---------------- 手臂摆法（同一目标的几种逆解） ----------------
let branchId = 0, branchTimer = null, branchItems = [];
function requestBranches() {               // 拖动时连续触发，等停 0.15 秒再算
  clearTimeout(branchTimer);
  branchTimer = setTimeout(() => { if (targetReady) send({ type: 'branches', id: ++branchId, ...targetPose(), seed: lastQ, ext: simExt() }); }, 150);
}
function showBranches(items) {
  branchItems = items;
  // 抓取点 / 放置点刚设时手臂摆法列表可能还没算到：算到了就把摆法名字补上
  let filled = false;
  for (const p of actions.flatMap((a) => [a.pick, a.place])) {
    const o = p && !p.label && p.q && items.find((b) => b.q.every((v, j) => Math.abs(v - p.q[j]) < 1));
    if (o) { p.label = o.label; filled = true; }
  }
  if (filled) showAuto();
  if (!items.length) { $('branches').innerHTML = '<span class="bad">这个位置没有逆解（够不到）</span>'; return; }
  const cur = lastQ ? items.findIndex((o) => o.q.every((v, j) => Math.abs(v - lastQ[j]) < 1)) : -1;
  $('branches').innerHTML = items.map((o, i) => `
    <button data-b="${i}" class="${i === cur ? 'sel' : ''} ${o.ok ? '' : 'badb'}">
      ${i === cur ? '● ' : ''}${o.label}${o.ok ? '' : '（' + o.reason + '）'}
      <span class="muted">· 关节最多转 ${o.dist.toFixed(0)}°</span></button>`).join('');
}
$('branches').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  const o = branchItems[+b.dataset.b];
  setJoints(o.q);
});

// ---------------- 关节（逐个调） ----------------
let fkId = 0, fkBusy = false, fkSent = 0, following = false, dragging = false;
function buildJoints() {
  if (!cfg.q_min) return;
  $('joints').innerHTML = cfg.q_min.map((lo, i) => `
    <div class="jrow"><b>J${i + 1}</b>
      <button data-j="${i}" data-d="-1">−</button>
      <input data-j="${i}" type="number" step="0.5" min="${lo.toFixed(0)}" max="${cfg.q_max[i].toFixed(0)}">
      <button data-j="${i}" data-d="1">+</button>
      <span class="muted">${lo.toFixed(0)}~${cfg.q_max[i].toFixed(0)}°</span></div>`).join('');
  if (lastQ) showJoints(lastQ);
}
function showJoints(q) {
  document.querySelectorAll('#joints input').forEach((el) => {
    if (document.activeElement !== el) el.value = (+q[+el.dataset.j]).toFixed(1);
  });
}
function setJoints(q) {          // 按关节角摆：后台算正解，目标跟着这个摆法走，不重新逆解（不会换摆法）
  if (!cfg.q_min) return;
  q = q.map((v, i) => Math.min(cfg.q_max[i], Math.max(cfg.q_min[i], v)));
  lastQ = q;
  if (preview) preview.setJoints(q);
  showJoints(q);
  fkBusy = true;                   // 目标位置要等正解回来才更新
  fkSent = performance.now();
  send({ type: 'fk', id: ++fkId, q, ext: simExt() });
  if (following) send({ type: 'goal', q });
}
function setTargetFromJoints(pos, quat, q) {
  target.position.fromArray(pos);
  target.quaternion.fromArray(quat);
  targetReady = true;
  lastQ = [...q];
  if (preview && !playing) preview.setJoints(q);
  showJoints(q);
  showTarget();
}
$('joints').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b || !lastQ) return;
  const q = [...lastQ];
  q[+b.dataset.j] += (+b.dataset.d) * (e.shiftKey ? 5 : 1);
  setJoints(q);
});
$('joints').addEventListener('change', (e) => {
  if (e.target.tagName !== 'INPUT' || !lastQ) return;
  const q = [...lastQ];
  q[+e.target.dataset.j] = +e.target.value;
  setJoints(q);
});
$('drag').onclick = () => {
  if (!dragging && !confirm('打开拖动示教后机械臂会变“软”，可以用手直接拖。\n控制器里夹爪负载没设好的话，松手时手臂可能会慢慢往下沉，先扶住机械臂。打开？')) return;
  send({ type: 'drag', on: !dragging });
};
$('follow').onclick = () => {
  if (!following) {
    $('toActual').click();             // 目标放回实际位置，后台也从实际位置开始，不会一开就冲出去
  }
  send({ type: 'follow', on: !following });
};


// ---------------- 界面：面板宽度可拖、每一节可折叠、仿真/真实切换 ----------------
(function layout() {
  const store = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } };
  const panel = $('panel'), split = $('splitter');
  const w0 = +store('twinPanelW');
  if (w0) { panel.style.width = w0 + 'px'; resize(); }
  split.addEventListener('mousedown', (e) => {
    e.preventDefault();
    split.classList.add('drag');
    const move = (ev) => { panel.style.width = Math.max(260, $('dock').getBoundingClientRect().left - ev.clientX - 3) + 'px'; resize(); };
    const up = () => { split.classList.remove('drag'); store('twinPanelW', panel.offsetWidth);
      window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  });
  // 最右边一列按钮，每节一个：点一下在左边打开那一节（按点的先后往下排），再点一下或点卡片标题上的 × 关掉。
  // 打开了哪几节存在浏览器里；第一次默认开相机画面、自动生成抓放、录制数据（2026-11-04 用户：避免杂乱）
  const SHORT = { '相机外参标定': '外参标定', '目标（夹爪抓取中心）': '目标', '关节（逐个调）': '关节', '自动生成抓放': '自动生成',
                  '预览与执行': '预览执行' };
  const DEFAULT_OPEN = ['相机画面', '自动生成抓放', '录制数据'];
  const dock = $('dock'), cards = {};
  panel.querySelectorAll(':scope > h2').forEach((h) => {      // 把每个标题和后面的内容包成一张卡片
    const sec = document.createElement('div');
    sec.className = 'sec';
    while (h.nextSibling && !(h.nextSibling.tagName === 'H2')) sec.appendChild(h.nextSibling);
    h.after(sec);
    const card = document.createElement('div');
    card.className = 'card';
    h.before(card);
    card.append(h, sec);
    const key = h.textContent.trim();
    cards[key] = card;
    h.title = '关掉这一节（右边的按钮可以再打开）';
    h.addEventListener('click', () => closeCard(key));
    const b = document.createElement('button');
    b.textContent = SHORT[key] || key;
    b.title = key;
    b.dataset.k = key;
    b.addEventListener('click', () => (card.hidden ? openCard(key) : closeCard(key)));
    dock.append(b);
  });
  const hint = document.createElement('div');
  hint.id = 'emptyHint';
  hint.textContent = '点右边的按钮打开要用的功能';
  panel.prepend(hint);
  let open = null;
  try { open = JSON.parse(store('twinOpenCards')); } catch (e) { open = null; }
  if (!Array.isArray(open)) open = DEFAULT_OPEN;
  open = open.filter((k) => cards[k]);
  function refreshCards() {
    open.forEach((k) => panel.append(cards[k]));           // 按打开的先后排
    for (const [k, c] of Object.entries(cards)) c.hidden = !open.includes(k);
    dock.querySelectorAll('button').forEach((b) => b.classList.toggle('open', open.includes(b.dataset.k)));
    hint.hidden = open.length > 0;
    store('twinOpenCards', JSON.stringify(open));
  }
  function openCard(k) {
    if (!open.includes(k)) open.push(k);
    refreshCards();
    cards[k].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  function closeCard(k) { open = open.filter((x) => x !== k); refreshCards(); }
  refreshCards();
  // 长的灰字说明默认收起来，点“说明”展开（苹果官网那种“了解更多 ›”，界面清爽些）；带 id 的是程序会改的提示，不收
  panel.querySelectorAll('.sec > div.muted[style*="font-size:12px"]:not([id])').forEach((d) => {
    if (d.textContent.trim().length < 40) return;
    const det = document.createElement('details');
    det.className = 'tip';
    const sum = document.createElement('summary');
    sum.textContent = '说明';
    d.before(det);
    det.append(sum, d);
    d.removeAttribute('style');
    d.className = '';
  });
})();
$('hudStop').onclick = () => send({ type: 'estop' });   // 三维视图左上角一直看得到的急停
$('mode').onclick = () => {
  if (cfg.offline) return;
  const toReal = !cfg.real;
  // 2026-11-03 用户：真机动作（执行、移到这帧、M、回初始位姿、跟随）都不再单独确认，只在切到真实模式时确认这一次
  if (toReal && !confirm('切到真实模式？\n切过去以后，真机执行、移到这帧 / 目标（空格）、回初始位姿、跟随都不再单独确认，点了真机就动。手放在急停旁边。')) return;
  send({ type: 'set_real', real: toReal });
};


// ---------------- 自动生成抓放：两点 → 一整套关键帧 ----------------
// 可以有几个动作（几次抓放连着做）：每个动作一个抓取点、一个放置点；“设为抓取点 / 放置点”设的是列表里选中的那个动作
let actions = [{ pick: null, place: null }], actSel = 0;
const stackMode = () => $('autoMode').value === 'stack';
$('autoMode').onchange = () => {
  $('stackHelp').hidden = !stackMode();
  $('stackAlignRow').hidden = !stackMode();
  $('placeAlign').disabled = stackMode();
  showAuto();
};
const autoGroup = new THREE.Group();
scene.add(autoGroup);
function showAuto(changed) {
  // 位置 + 倾斜 + 手臂摆法：只转了夹爪方向或换了摆法时位置数字不变，但这里能看出来确实更新了
  const fmt = (p) => p ? `${p.pos.map((v) => (v * 1000).toFixed(0)).join(', ')} mm · 倾斜 ${tiltDeg(new THREE.Quaternion(...p.quat)).toFixed(0)}°`
    + (p.label ? ` · ${p.label}` : '') : '未设';
  const many = actions.length > 1;
  const dropText = (a) => {
    if (a.pending === 'place') return '等目标算完再记…';
    if (stackMode() && a.stackBase) {
      const { x, y, top } = a.stackBase;
      return [x, y, top].map((v) => (v * 1000).toFixed(0)).join(', ') + ' mm';
    }
    return fmt(stackMode() ? a.stackPlace : a.place);
  };
  $('actlist').innerHTML = actions.map((a, i) => `
    <li data-a="${i}" class="${i === actSel ? 'sel' : ''}">
      <div class="top"><b>${many ? '动作 ' + (i + 1) : '抓放'}</b>${many || a.pick || a.place || a.stackBase || a.stackPlace ? `<button class="link" data-del="${i}">${many ? '删除' : '清空'}</button>` : ''}</div>
      <div class="row" data-k="pick">抓取点：${a.pending === 'pick' ? '等目标算完再记…' : fmt(a.pick)}</div>
      <div class="row" data-k="place">${stackMode() && a.stackBase ? '底块顶面' : '放置点'}：${dropText(a)}</div></li>`).join('');
  if (changed) {                       // 刚更新的那一行闪一下
    const row = $('actlist').querySelector(`li[data-a="${actSel}"] .row[data-k="${changed}"]`);
    if (row) {
      row.style.transition = 'none'; row.style.background = 'rgba(52,199,89,0.25)';
      requestAnimationFrame(() => { row.style.transition = 'background 0.8s'; row.style.background = 'transparent'; });
    }
  }
  autoGroup.clear();
  actions.forEach((a, i) => {          // 抓取点绿、放置点橙；选中的动作大一点、不透明，抓取点到放置点连一条线
    const on = i === actSel;
    const base = a.stackBase;
    const drop = stackMode() ? (base ? { pos: [base.x, base.y, base.top] } : a.stackPlace) : a.place;
    [[a.pick, 0x34c759], [drop, 0xff9f0a]].forEach(([p, color]) => {
      if (!p) return;
      const s = new THREE.Mesh(new THREE.SphereGeometry(on ? 0.014 : 0.009, 20, 14),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: on ? 0.9 : 0.45 }));
      s.position.fromArray(p.pos);
      autoGroup.add(s);
    });
    if (a.pick && drop) {
      autoGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a.pick.pos), new THREE.Vector3(...drop.pos)]),
        new THREE.LineBasicMaterial({ color: on ? 0x0071e3 : 0x86868b, transparent: true, opacity: on ? 0.8 : 0.4 })));
    }
  });
}
$('actlist').addEventListener('click', (e) => {
  const del = e.target.closest('[data-del]');
  if (del) {
    const i = +del.dataset.del;
    if (actions.length === 1) actions = [{ pick: null, place: null }];
    else {
      if (!confirm(`删除动作 ${i + 1}？`)) return;
      actions.splice(i, 1);
    }
    actSel = Math.min(actSel > i ? actSel - 1 : actSel, actions.length - 1);
    showAuto();
    return;
  }
  const li = e.target.closest('li[data-a]');
  if (li) { actSel = +li.dataset.a; showAuto(); }
});
$('actAdd').onclick = () => {
  actions.push({ pick: null, place: null });
  actSel = actions.length - 1;
  showAuto();
};
// 设抓取点 / 放置点：目标位置和关节角要是同一个时刻的。调关节、点手臂摆法以后，位置要等后台算完正解才更新，
// 拖目标以后关节角要等逆解回来才更新——这时候点按钮会存成“旧位置 + 新关节角”（2026-10-28 用户发现点了坐标没变）。
// 所以还在算的时候先记下要设哪个，等算完再存
function branchLabel() {             // 现在选中的手臂摆法（“手臂摆法”列表里和当前关节角一样的那个）
  const o = lastQ && branchItems.find((b) => b.q.every((v, j) => Math.abs(v - lastQ[j]) < 1));
  return o ? o.label : '';
}
function grab(kind, a, manualStackPlace) {
  if (!actions.includes(a)) return;
  delete a.pending;
  if (!targetReady) { alert('先把目标放到位置'); showAuto(); return; }
  const name = kind === 'pick' ? '抓取点' : '放置点';
  if (solveOk === false && !confirm(`目标现在显示“到不了”，还要设成${name}吗？（生成时这一帧多半也到不了）`)) { showAuto(); return; }
  const pose = { ...targetPose(), q: lastQ ? [...lastQ] : null, label: branchLabel() };
  if (manualStackPlace) {
    a.stackPlace = pose;
    delete a.stackBase;
    if (stackRequest?.action === a) stackRequest = null;
  } else {
    a[kind] = pose;
  }
  if (kind === 'pick') a.w = +$('objW').value;   // 这个动作夹的东西多宽（点云框出来会自动填），勾了“按物体宽度检查”时用
  showAuto(kind);
}
function grabLater(kind) {
  const action = actions[actSel];
  const manualStackPlace = kind === 'place' && stackMode();
  action.pending = kind;
  showAuto();
  whenTargetReady(() => grab(kind, action, manualStackPlace));
}
$('setPick').onclick = () => grabLater('pick');
$('setPlace').onclick = () => grabLater('place');
showAuto();
// 生成关键帧：exec=false 生成后播预览；exec=true（“生成关键帧并执行”、2 键，2026-11-09）验证通过就直接真机执行，有问题不执行
let autoExec = false;          // 等着的这次自动生成，回来以后要不要直接执行
let autoKfSnap = null;         // 上次自动生成的关键帧（没手动改过就不用问“替换吗”）
function genKeyframes(exec) {
  if (exec && (robotExec || playing || performance.now() - execEndedAt < 1000)) {
    $('planinfo').className = 'warn';
    $('planinfo').textContent = robotExec ? '真机正在动，停下以后再生成并执行' : '刚停下 / 正在播预览，等一下再按';
    return;
  }
  const stack = stackMode();
  if (stack && (actions.length !== 1 || !actions[0].pick || !(actions[0].stackBase || actions[0].stackPlace))) {
    alert('叠方块一次生成一组。请保留一个动作，设好抓取点和放置点，或右键点云选择底块。');
    return;
  }
  const miss = stack ? -1 : actions.findIndex((a) => !a.pick || !a.place);
  if (miss >= 0) {
    alert(actions.length > 1 ? `动作 ${miss + 1} 还没设好抓取点和放置点（不要的动作可以删掉）` : '先设好抓取点和放置点');
    return;
  }
  // 现在的关键帧就是上次自动生成的、没手动改过：直接换，不问（反复录的时候每条都要问太烦）
  if (keyframes.length && JSON.stringify(keyframes) !== autoKfSnap
      && !confirm(`用自动生成的关键帧替换现在的 ${keyframes.length} 个？（这些关键帧是手动记录或改过的）`)) return;
  if (exec && $('recOn').checked && !$('recTask').value.trim() && !confirm('录制开着，但还没填任务描述。照样录？')) return;
  autoExec = exec;
  playing = false;
  $('planinfo').className = 'muted';
  $('planinfo').textContent = '正在生成并验证…（不行会自动调整距离和高度再试' + (actions.length > 1 ? '，每个动作单独调整）' : '）')
    + (exec ? '，通过了就直接执行' : '');
  $('problems').innerHTML = '';
  const check = $('checkClose').checked;
  send({ type: 'autogen', mode: stack ? 'stack' : 'plate',
         align_height: (+$('stackAlignHeight').value || 8) / 100,
         actions: actions.map((a) => ({ pick: a.pick, place: a.place, stack_base: a.stackBase, stack_place: a.stackPlace,
                                        width: check ? Math.max(0, (a.w ?? +$('objW').value) || 0) / 100 : null })),
         dist: (+$('apDist').value || 8) / 100, lift: (+$('liftH').value || 10) / 100, home: $('backHome').checked,   // 回初始那段执行但不录（按帧名“回初始”判断）
         start_home: $('startHome').checked,   // 初始位姿 → 预抓取（关节运动）→ …，每条示范起点一样；几个动作时只有第 1 个
         hover_align: $('hoverAlign').checked,   // 先到方块上方（途中夹爪不转）→ 原地转夹爪对齐 → 竖直下降（2026-11-06）
         place_align: $('placeAlign').checked,   // 不勾：放的时候夹爪跟着手臂转（J6 和抓时一样）；勾上：按放置点的方向对齐（2026-11-09）
         speed: $('speed').value / 100, ang_speed: THREE.MathUtils.degToRad(+$('ang').value), precise: $('precise').value / 100,
         start_noise: +$('startNoise').value, exec_noise: +$('execNoise').value / 1000, human: $('human').value / 100, seed: null });
};
$('genKf').onclick = () => genKeyframes(false);
$('genExec').onclick = () => genKeyframes(true);
function onAutogen(m) {
  const exec = autoExec;
  autoExec = false;
  if (!m.keyframes) {                     // 每一种调整都试过了，连单个关键帧都到不了
    $('planinfo').className = 'bad';
    $('planinfo').textContent = '自动生成失败：抓取点或放置点附近有到不了的关键帧，换个位置、夹爪方向或手臂摆法再试';
    $('problems').innerHTML = (m.notes || []).map((t) => `<li class="bad">${t}</li>`).join('');
    return;
  }
  keyframes = m.keyframes;
  autoKfSnap = JSON.stringify(keyframes);
  selected = 0;
  renderKeyframes();
  showPlan(m.plan);
  const tips = (m.notes || []).filter((t) => t.startsWith('已自动'));
  if (m.plan.ok) {
    $('planinfo').textContent = '自动生成并验证通过，' + $('planinfo').textContent + (tips.length ? '。' + tips[0] : '');
  } else {
    $('planinfo').textContent = '自动生成了，但怎么调整都有问题（下面是问题最少的方案）：';
    $('problems').innerHTML += (m.notes || []).map((t) => `<li class="muted">试过 ${t}</li>`).join('');
  }
  if (exec) {
    // 用生成时的同一个随机种子（showPlan 里记下了）再规划一遍并执行，执行的就是验证过的这一条
    if (m.plan.ok && !robotExec) { $('planinfo').textContent += '；正在执行'; planRequest('execute'); }
    else if (m.plan.ok) $('planinfo').textContent += '；真机正在动，没有执行';
    else $('planinfo').textContent = '生成的轨迹有问题，没有执行：' + $('planinfo').textContent;
    return;
  }
  playPlan(m.plan);
}

// ---------------- 速度、预览、执行 ----------------
function showSpeeds() {
  $('speedVal').textContent = $('speed').value; $('angVal').textContent = $('ang').value; $('preciseVal').textContent = $('precise').value;
  $('startNoiseVal').textContent = $('startNoise').value;
  $('execNoiseVal').textContent = $('execNoise').value;
}
// 每次打开/刷新网页，滑块都回到 index.html 里写的推荐值（Firefox 刷新时会把上次拖过的值填回来）
function resetSliders() {
  document.querySelectorAll('input[type=range]').forEach((r) => { r.value = r.defaultValue; });
  $('humanVal').textContent = $('human').value;
  $('segTolVal').textContent = +$('segTol').value === 0 ? '0（只按形状）' : $('segTol').value;
  showSpeeds();
}
resetSliders();
window.addEventListener('pageshow', (e) => { if (e.persisted) resetSliders(); });
$('speed').oninput = showSpeeds;
$('ang').oninput = showSpeeds;
$('precise').oninput = showSpeeds;
$('startNoise').oninput = () => { showSpeeds(); planSeed = null; };
$('execNoise').oninput = showSpeeds;
let planSeed = null;           // 人为扰动的随机种子：预览和执行用同一个，执行的就是预览看到的那条
function planRequest(type) {
  if (!keyframes.length) { alert('先记录至少一个关键帧'); return; }
  // 勾着“按物体宽度检查”时，手动记录的合夹爪帧也按物体宽度算合上后指尖伸出多少（规划时这一帧和之后夹着走、放下都按它查离桌面）；
  // 自动生成的帧自己带 grip_width
  const w = closeWidth();
  const kfs = keyframes.map((k) => (k.gripper === 'close' && k.grip_width === undefined && w !== null ? { ...k, grip_width: w } : k));
  send({ type, keyframes: kfs, speed: $('speed').value / 100, ang_speed: THREE.MathUtils.degToRad(+$('ang').value),
         precise: $('precise').value / 100, start_noise: +$('startNoise').value, exec_noise: +$('execNoise').value / 1000,
         human: $('human').value / 100, seed: planSeed });
}
// 场景里画出规划出来的路径（抓取中心走过的曲线）
const pathGroup = new THREE.Group();
scene.add(pathGroup);
function showPath(m) {
  pathGroup.clear();
  if (!m.path || m.path.length < 2) return;
  const pts = m.path.map((p) => new THREE.Vector3(...p));
  pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: m.ok ? 0xf08c00 : 0xcf222e })));
}
function showPlan(m) {
  const ok = m.ok;
  if (m.seed !== undefined) planSeed = m.seed;
  const go = (m.start_dq ? `；起点扰动：第 1 帧关节最多偏 ${m.start_dq}°` : '')
    + (m.exec_noise ? `；执行时小扰动 ${m.exec_noise} mm（录的动作是干净轨迹）` : '')
    + (m.approach_s ? `；执行时先用关节运动从实际位置移到第 1 帧，约 ${m.approach_s} 秒` : '');
  $('planinfo').className = ok ? 'ok' : 'bad';
  $('planinfo').textContent = ok ? `轨迹没问题，共 ${m.duration} 秒（从第 1 帧开始${go}）` : '轨迹有问题，红色部分到不了：';
  $('problems').innerHTML = (m.notes || []).map((t) => `<li class="warn">${t}</li>`).join('')
    + (m.problems || []).map((p) => `<li class="bad">第 ${p.t} 秒起：${p.reason}</li>`).join('');
  showPath(m);
}
function playPlan(m) {
  if (!m.q || !m.q.length || !preview) return;
  playing = true;
  const t0 = performance.now(), n = m.q.length, bad = new Set();
  (m.problems || []).forEach((p) => { for (let i = p.start; i <= p.end; i++) bad.add(i); });
  const grip = new Map(m.events);
  let closed = null;
  preview.setGripper(1, true);           // 预览从夹爪张开开始
  function step() {
    const i = Math.min(n - 1, Math.floor((performance.now() - t0) / 1000 / m.dt));
    preview.setJoints(m.q[i]);
    for (const [k, g] of grip) if (k <= i) closed = g === 'close';
    if (closed !== null) preview.setGripper(closed);   // 手指在 update 里慢慢动过去
    previewMat.color.set(bad.has(i) ? 0xffb3b3 : 0xe8e8e8);
    $('progress').firstChild.style.width = (100 * (i + 1) / n) + '%';
    if (i < n - 1 && playing) requestAnimationFrame(step);
    else { playing = false; showSimGrip(); requestSolve(true); }
  }
  step();
}
$('preview').onclick = () => planRequest('plan');
$('reroll').onclick = () => { planSeed = null; planRequest('plan'); };
$('human').oninput = () => { $('humanVal').textContent = $('human').value; planSeed = null; };
$('execute').onclick = () => {
  if ($('recOn').checked && !$('recTask').value.trim() && !confirm('录制开着，但还没填任务描述。照样录？')) return;
  playing = false;
  planRequest('execute');
};
$('stop').onclick = () => { playing = false; send({ type: 'stop' }); };
$('estop').onclick = () => send({ type: 'estop' });
$('release').onclick = () => send({ type: 'release' });
$('home').onclick = () => { send({ type: 'home' }); targetReady = false; lastQ = null; };
// 真实夹爪：只开合真夹爪，不动仿真夹爪
$('gripOpen').onclick = () => send({ type: 'gripper', closed: false });
$('gripClose').onclick = () => send({ type: 'gripper', closed: true });
// 仿真夹爪：只在网页里开合（记录关键帧时记它）
$('simOpen').onclick = () => setSimGrip('open');
$('simClose').onclick = () => setSimGrip('close');
$('simFromReal').onclick = () => setSimGrip(realGripClosed ? 'close' : 'open');
$('objW').addEventListener('input', () => { if (simGrip === 'close') { showSimGrip(); keepAboveTable(); requestSolve(true); } });
$('checkClose').addEventListener('change', () => { if (simGrip === 'close') { keepAboveTable(); showTarget(); requestSolve(true); } });
$('gripRestart').onclick = () => {        // 夹爪“死掉”（读写全超时）时：末端 24V 断电重启
  if (confirm('末端断电 2 秒再上电，重启夹爪（约 5 秒）。夹着东西会松掉，确定？')) send({ type: 'gripper_restart' });
};
$('toActual').onclick = () => { if (lastActual) { lastQ = null; const c = centerFromFlange(lastActual); setTarget(c.pos, c.quat); } };
// 夹爪朝外摆正：后台绕竖直轴扫一圈方向，挑预抓取/抓取/抬起都能到、手腕离奇异够远、转得最少的（伸进距离、抬起高度、物体宽度用自动生成卡片里的）
$('alignOut').onclick = () => whenTargetReady(() => {
  if (!targetReady) return;
  $('solveinfo').className = 'muted';
  $('solveinfo').textContent = '正在找夹爪朝外的方向…';
  send({ type: 'align_out', ...targetPose(), seed: lastQ,
         dist: (+$('apDist').value || 8) / 100, lift: (+$('liftH').value || 10) / 100,
         width: closeWidth() });
});
function onAlign(m) {
  if (!m.ok) { $('solveinfo').className = 'bad'; $('solveinfo').textContent = m.text; return; }
  const jump = lastQ ? Math.max(...m.q.map((v, j) => Math.abs(v - lastQ[j]))) : 0;
  setJoints(m.q);                      // 按解好的关节角摆（摆法就是检查过的那种），目标等正解回来更新
  setTimeout(() => { $('solveinfo').className = 'ok'; $('solveinfo').textContent = m.text; }, 300);
}
// 初始位姿的关节角是固定的：直接按关节角摆（不从现在的手臂摆法去逆解，不会报“要换摆法”）
$('toHome').onclick = () => { if (cfg.home_q) setJoints(cfg.home_q); else if (cfg.home) setTarget(cfg.home.pos, cfg.home.quat); };
$('modeBtn').onclick = () => {
  tc.setMode(tc.getMode() === 'translate' ? 'rotate' : 'translate');
  $('modeBtn').textContent = '切换：' + (tc.getMode() === 'translate' ? '旋转' : '平移');
  updateGizmo();
};
$('tplSave').onclick = () => {
  const name = $('tplName').value.trim();
  if (!name) { alert('先填模板名'); return; }
  send({ type: 'save', name, data: { keyframes, speed: +$('speed').value, ang: +$('ang').value, precise: +$('precise').value,
                                     start_noise: +$('startNoise').value, exec_noise: +$('execNoise').value, human: +$('human').value } });
  setTimeout(() => send({ type: 'list' }), 300);
};
$('tplLoad').onclick = () => { if ($('tplList').value) send({ type: 'load', name: $('tplList').value }); };
$('tplRefresh').onclick = () => send({ type: 'list' });
window.addEventListener('keydown', (e) => {
  if (e.target.closest?.('input,select')) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;          // Ctrl+Q 是关 Firefox，别抢
  // 2026-11-04 用户：WASD 水平移动、R/F 上下、Q 顺时针 / E 逆时针转夹爪，真机跟着动；原来 W/E 切控件、Q 切坐标暂时不要
  if (JOG_KEYS[e.code] !== undefined) { e.preventDefault(); if (!e.repeat) jogKeyDown(e.code, e.shiftKey); }
  if (e.key === ' ') {
    // 空格＝原来 M 的功能（真机移到目标）。真机正在动时按空格还是停止；刚停下 1 秒内不接着动（防止连按）
    e.preventDefault();
    if (robotExec || playing) { playing = false; send({ type: 'stop' }); }
    else if (performance.now() - execEndedAt > 1000 && !e.repeat) gotoTarget();
  }
  if (e.key === 'm' || e.key === 'M') { e.preventDefault(); gotoTarget(); }
  // 2026-11-04 用户：1 键切换真夹爪开 / 合（按当时真夹爪的状态反过来）
  if ((e.code === 'Digit1' || e.code === 'Numpad1') && !e.repeat) { e.preventDefault(); send({ type: 'gripper', closed: !realGripClosed }); }
  // 2026-11-09 用户：2 键＝生成关键帧并执行（和自动生成卡片里的按钮一样）
  if ((e.code === 'Digit2' || e.code === 'Numpad2') && !e.repeat) { e.preventDefault(); genKeyframes(true); }
});

// ---------------- 启动 ----------------
(async () => {
  preview = await makeRobot(previewMat, true);
  actual = await makeRobot(actualMat, true);
  scene.add(preview.root, actual.root);
  showSimGrip();
  showSpeeds();
  renderKeyframes();
  connect();
})();
let lastFrame = performance.now();
(function loop() {
  const now = performance.now(), dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  if (preview) preview.update(dt);
  if (actual) actual.update(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
})();

// ---------------- 第三视角相机画面（调相机位置、标定时看） ----------------
// 这一节展开、勾着“实时刷新”时，每 0.25 秒要一张（上一张回来了才要下一张，网慢时自动变慢）
let camWaiting = false, camFrames = 0, camAsked = 0;
setInterval(() => {
  const open = !$('camBox').closest('.card')?.hidden;
  if (!open || !$('camLive').checked || !ws || ws.readyState !== 1) return;
  if (camWaiting && performance.now() - camAsked < 3000) return;   // 回复丢了的话 3 秒后重新要
  camWaiting = true;
  camAsked = performance.now();
  send({ type: 'cam_jpeg', which: camWhich(), zones: $('camZones').checked });
}, 250);
function camWhich() { return document.querySelector('input[name="camWhich"]:checked').value; }
document.querySelectorAll('input[name="camWhich"]').forEach((el) => el.addEventListener('change', () => { camWaiting = false; }));

// ---------------- 录制数据：执行时录成一条示范（HDF5），存储路径可以选 ----------------
let recording = null, recInfo = {};
function onRec(m) {
  const c = m.cfg || {};
  if (document.activeElement !== $('recFolder')) $('recFolder').value = c.folder || '';
  if (document.activeElement !== $('recTask')) $('recTask').value = c.task || '';
  $('recOn').checked = !!c.on;
  $('recDepth').checked = !!c.depth;
  recInfo = m.info || {};
  const i = recInfo;
  const where = !i.exists ? '这个文件夹还没有，录第一条时自动建' : `已有 ${i.episodes} 条，下一条是第 ${i.next} 条`;
  const cams = m.cams ? [m.cams.scene ? '' : '没有第三视角相机', m.cams.wrist ? '' : '没有腕部相机'].filter(Boolean).join('、') : '';
  $('recInfo').className = !i.writable ? 'bad' : cams ? 'warn' : 'muted';
  $('recInfo').textContent = `${where} · 磁盘还剩 ${i.free_gb} GB` + (i.writable ? '' : ' · 没有写权限！') + (cams ? ` · ${cams}，现在录不了` : '');
  if (m.text) { $('recMsg').textContent = m.text; $('recMsg').className = m.saved ? 'ok' : (m.text.startsWith('没') || m.text.includes('不了') ? 'bad' : 'muted'); }
  // 别的提示（比如相机缺画面没开始录）在三维视图左上角亮几秒；存好了 / 没存 / 标失败由下面一直显示的 recRun 说，不再闪
  if (m.text && !m.saved && /没录上|不全|没开始/.test(m.text) && !/中途停了|标成失败/.test(m.text)) {
    const good = m.saved || m.text.startsWith('✓');
    $('recToast').textContent = m.saved ? m.text.split('；')[0] : m.text;
    $('recToast').className = 'rectoast' + (good ? '' : ' bad');
    $('recToast').hidden = false;
    clearTimeout(recToastTimer);
    recToastTimer = setTimeout(() => { $('recToast').hidden = true; }, 6000);
  }
  recRun = m.run || null;
  showRecRun();
  $('recDiscard').disabled = !(recRun && recRun.state === 'saved');
  // 已录几条：录上一条数字加 1 并闪一下；停了没存数字不变（2026-11-09 用户：一条半分钟就录完，看时间分不清录没录）
  const n = i.episodes ?? 0;
  $('recCount').textContent = `已录 ${n} 条`;
  $('recCount').hidden = !c.on;
  if (recCountLast !== null && n > recCountLast) {
    $('recCount').classList.remove('flash'); void $('recCount').offsetWidth; $('recCount').classList.add('flash');
  }
  recCountLast = n;
}
// 左上角一直显示“刚才这一次”录没录（到下一次开始为止）：录上了＝绿色“✓ 刚才这次录成了第 n 条”＋开头的小图＋失败按钮；
// 没录＝灰色“✗ 刚才这次没录（为什么）”，不显示上一条，免得点“失败”丢掉上一条好的；标了失败＝红色。
// 默认算成功，失败了才点“失败”（2026-11-09 用户：别每条都要点成功）
let recRun = null, recCountLast = null;
let recToastTimer = null;
function showRecRun() {
  const r = recRun;
  const show = r && r.state !== 'running' && recording === null;
  $('recRun').hidden = !show;
  if (!show) return;
  const saved = r.state === 'saved';
  $('recRun').className = 'recrun' + (saved ? ' ok' : r.state === 'discarded' ? ' bad' : '');
  $('recRunText').textContent = saved ? `✓ 刚才这次录成了第 ${r.idx} 条`
    : r.state === 'discarded' ? `✗ 第 ${r.idx} 条标成失败，挪进 _废弃 了`
    : r.state === 'off' ? '刚才这次没录（录制关着）'
    : `✗ 刚才这次没录（${r.why || '没存'}）`;
  $('recThumb').hidden = !(saved && r.thumb);
  if (saved && r.thumb) $('recThumb').src = r.thumb;
  $('hudBad').hidden = !saved;
}
const discardRec = () => send({ type: 'rec_discard' });
// 窗口按钮：只有网页开在 Orin 自己屏幕上（localhost）时有用——后台模拟 F11、最小化 Firefox 窗口（网页自己做不到）
if (['localhost', '127.0.0.1', '::1', '[::1]'].includes(location.hostname)) {
  $('winctl').hidden = false;
  const showWin = () => { $('winFull').textContent = window.fullScreen ? '窗口' : '全屏'; };
  window.addEventListener('resize', () => setTimeout(showWin, 100));
  showWin();
  $('winFull').onclick = () => send({ type: 'win', action: 'fullscreen' });
  $('winMin').onclick = () => send({ type: 'win', action: 'minimize' });
}
$('hudBad').onclick = discardRec;
const recSend = (patch) => send({ type: 'rec_config', ...patch });
$('recOn').onchange = () => recSend({ on: $('recOn').checked });
$('recDepth').onchange = () => recSend({ depth: $('recDepth').checked });
$('recFolder').onchange = () => recSend({ folder: $('recFolder').value });
$('recTask').onchange = () => recSend({ task: $('recTask').value.trim() });
$('recFolder').onkeydown = $('recTask').onkeydown = (e) => { if (e.key === 'Enter') e.target.blur(); };
$('recManual').onclick = () => send({ type: 'rec_manual', on: recording === null });
$('recDiscard').onclick = discardRec;
// 选存储路径：在 Orin 上一层层点进去，可以新建文件夹
let dirCur = '';
$('recBrowse').onclick = () => { $('dirPicker').hidden = false; send({ type: 'ls_dir', path: $('recFolder').value }); };
function onLsDir(m) {
  dirCur = m.path;
  $('dirPath').textContent = m.path;
  $('dirUp').disabled = !m.parent || m.parent === m.path;
  $('dirUp').onclick = () => send({ type: 'ls_dir', path: m.parent });
  $('dirList').innerHTML = (m.dirs || []).map((d) => `<button data-d="${d.replace(/"/g, '&quot;')}">📁 ${d}</button>`).join('')
    || '<span class="muted" style="font-size:12px">（没有子文件夹）</span>';
  $('dirMsg').className = m.error ? 'bad' : 'muted';
  $('dirMsg').textContent = m.error || `这里有 ${m.episodes ?? 0} 条示范 · 磁盘还剩 ${m.free_gb} GB` + (m.writable === false ? ' · 没有写权限' : '');
}
$('dirList').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-d]');
  if (b) send({ type: 'ls_dir', path: dirCur.replace(/\/$/, '') + '/' + b.dataset.d });
});
$('dirNew').onclick = () => { const n = prompt('新文件夹的名字（比如 pick_red_cube）'); if (n) send({ type: 'mkdir', path: dirCur, name: n }); };
$('dirPick').onclick = () => { $('recFolder').value = dirCur; recSend({ folder: dirCur }); $('dirPicker').hidden = true; };
$('dirCancel').onclick = () => { $('dirPicker').hidden = true; };
setInterval(() => {                    // 每秒显示一次实际帧率
  if ($('camLive').checked && camFrames) $('camInfo').textContent = `${camFrames} 帧/秒 · 点画面放大`;
  camFrames = 0;
}, 1000);
$('camBox').onclick = () => $('camBox').classList.toggle('big');

// 初始位姿：把机械臂现在的实际姿态存成初始位姿（后台存 home.json，重启后也用）
$('setHome').onclick = () => {
  if (confirm('把机械臂现在的实际姿态设为初始位姿？之后“回初始位姿”和自动生成的“从初始位姿出发 / 回初始”都用这个。')) send({ type: 'set_home' });
};

// 合上时的指尖检查：不勾就发 null（以用户摆的抓取点为准，不加伸出量）；勾上发物体宽度（米，0 = 按空夹全合）
function closeWidth() { return $('checkClose').checked ? Math.max(0, +$('objW').value || 0) / 100 : null; }
$('checkClose').onchange = () => { $('objW').disabled = !$('checkClose').checked; };

// ---------------- 相机外参标定（棋盘格 + 用合上的指尖碰四个角） ----------------
$('boardFind').onclick = () => { $('calibMsg').className = 'muted'; $('calibMsg').textContent = '正在找棋盘格（几秒）…'; send({ type: 'board_find' }); };
$('boardCalib').onclick = () => { $('calibMsg').className = 'muted'; $('calibMsg').textContent = '正在计算（再用机械臂模型精修，十几秒）…'; send({ type: 'board_calib' }); };
$('calibRefine').onclick = () => { $('calibMsg').className = 'muted'; $('calibMsg').textContent = '正在用机械臂模型精修（十几秒）…'; send({ type: 'calib_refine' }); };
document.querySelectorAll('button.touch').forEach((b) => { b.onclick = () => send({ type: 'board_touch', label: b.dataset.l }); });
function onCalib(m) {
  $('calibMsg').textContent = m.text;
  $('calibMsg').className = m.done ? 'ok' : (m.done === false || m.text.startsWith('没') || m.text.startsWith('先') || m.text.includes('至少') ? 'warn' : 'muted');
  if (m.board) document.querySelectorAll('button.touch').forEach((b) => b.classList.remove('on'));
  if (m.touched) document.querySelectorAll('button.touch').forEach((b) => b.classList.toggle('on', m.touched.includes(b.dataset.l)));
}

// ---------------- 点云抓取：显示点云 → 点一下物体框出来 → 自动生成抓取姿态 / 手动选位置 ----------------
let cloud = null, cloudColors = null, cloudBusy = false, cloudAsked = 0, segBox = null, segBoxObj = null, segText = '', graspFromBox = false;
const cloudMat = new THREE.PointsMaterial({ size: 0.004, vertexColors: true });
function requestCloud() {
  if (cloudBusy && performance.now() - cloudAsked < 5000) return;   // 上一份还没回来（网慢）；5 秒没回就重新要
  cloudBusy = true;
  cloudAsked = performance.now();
  send({ type: 'cloud' });
}
function onCloud(buf) {
  cloudBusy = false;
  const dv = new DataView(buf);
  if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'PCLD') return;
  const n = dv.getUint32(4, true);
  const xyz = new Float32Array(buf.slice(8, 8 + n * 12));
  const rgb = new Uint8Array(buf, 8 + n * 12, n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) col[i] = rgb[i] / 255;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  if (cloud) { scene.remove(cloud); cloud.geometry.dispose(); }
  cloud = new THREE.Points(g, cloudMat);
  cloud.visible = $('cloudShow').checked;
  scene.add(cloud);
  cloudColors = col.slice();
  $('cloudInfo').className = 'muted';
  $('cloudInfo').textContent = `点云 ${n} 个点（${new Date().toLocaleTimeString()}）· 在点云上点一下物体`;
}
$('cloudGet').onclick = requestCloud;
setInterval(() => { if ($('cloudLive').checked && ws && ws.readyState === 1) requestCloud(); }, 2000);
$('cloudShow').onchange = () => { if (cloud) cloud.visible = $('cloudShow').checked; if (segBoxObj) segBoxObj.visible = $('cloudShow').checked; };
const radioVal = (name) => document.querySelector(`input[name="${name}"]:checked`).value;

// 在点云上点一下（按下和松开几乎没动才算点，拖动是转视角；点在拖动控件上不算）
let downXY = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downXY = tc.axis || e.button !== 0 ? null : [e.clientX, e.clientY]; });   // 只认左键（右键是菜单）
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downXY || !cloud || !cloud.visible || playing) return;
  if (Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]) > 4) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = 0.006;
  ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
  const hit = ray.intersectObject(cloud)[0];
  if (!hit) return;
  const pos = cloud.geometry.attributes.position;
  const p = [pos.getX(hit.index), pos.getY(hit.index), pos.getZ(hit.index)];
  $('cloudLive').checked = false;          // 点选以后先别自动刷新，不然框和点对不上
  if (radioVal('pickMode') === 'auto') {
    $('cloudInfo').className = 'muted';
    $('cloudInfo').textContent = '正在框选物体…';
    lastSeg = p;
    send({ type: 'segment', point: p, tol: +$('segTol').value, square: stackMode() });
  } else {
    // 手动：抓取中心放到点到的地方，夹爪方向不变，挑一种能到的摆法。点在框外就把之前框的物体清掉
    // （“按抓取中心重新生成姿态”会按框的大小定两指方向，框的是别的物体就不对了）
    if (segBox) {
      const dx = p[0] - segBox.center[0], dy = p[1] - segBox.center[1], c = Math.cos(segBox.yaw), s = Math.sin(segBox.yaw);
      const inside = Math.abs(dx * c + dy * s) <= segBox.size[0] / 2 + 0.01 && Math.abs(-dx * s + dy * c) <= segBox.size[1] / 2 + 0.01
                     && p[2] <= segBox.top + 0.01;
      if (!inside) clearSeg();
    }
    target.position.fromArray(p);
    targetReady = true;
    showTarget();
    $('cloudInfo').className = 'muted';
    $('cloudInfo').textContent = '正在找能到的手臂摆法（夹爪方向不变）…';
    graspFromBox = false;
    send({ type: 'grasp_keep', ...targetPose(), seed: lastQ, dist: (+$('apDist').value || 8) / 100,
           lift: (+$('liftH').value || 10) / 100, width: closeWidth() });
  }
});
// 颜色容差：拖滑块时用上一次点的地方重新框（不用再点一次）
let lastSeg = null, segTimer = null;
$('segTol').oninput = () => {
  $('segTolVal').textContent = +$('segTol').value === 0 ? '0（只按形状）' : $('segTol').value;
  clearTimeout(segTimer);
  if (lastSeg && radioVal('pickMode') === 'auto')
    segTimer = setTimeout(() => { $('cloudInfo').className = 'muted'; $('cloudInfo').textContent = '正在重新框选…';
                                  send({ type: 'segment', point: lastSeg, tol: +$('segTol').value, square: stackMode() }); }, 250);
};
function clearSeg() {                     // 去掉橙色框和橙色点
  if (segBoxObj) { scene.remove(segBoxObj); segBoxObj.geometry.dispose(); segBoxObj = null; }
  if (cloud && cloudColors) { cloud.geometry.attributes.color.array.set(cloudColors); cloud.geometry.attributes.color.needsUpdate = true; }
  segBox = null;
  segText = '';
}
let stackRequest = null, stackRequestId = 0;
function selectStackBase(point) {
  $('cloudLive').checked = false;
  clearTimeout(segTimer);
  const action = actions[actSel];
  stackRequest = { id: ++stackRequestId, action };
  $('cloudInfo').className = 'muted';
  $('cloudInfo').textContent = '正在识别底块顶面…';
  send({ type: 'segment', point, tol: +$('segTol').value, square: true, stack_request: stackRequest.id });
}
function onSegment(m) {
  const baseRequest = m.stack_request != null;
  if (baseRequest && stackRequest?.id !== m.stack_request) return;
  const action = baseRequest ? stackRequest.action : null;
  if (baseRequest) {
    stackRequest = null;
    if (!actions.includes(action) || !stackMode()) return;
  }
  clearSeg();
  if (!m.box) { $('cloudInfo').className = 'bad'; $('cloudInfo').textContent = m.text; return; }
  const b = m.box;
  segBox = { center: b.center, size: b.size, yaw: b.yaw, top: b.top, bottom: b.bottom };
  segText = m.text;
  if (cloud) {                             // 物体的点涂成橙色
    const c = cloud.geometry.attributes.color;
    for (const i of b.idx) c.setXYZ(i, 1.0, 0.55, 0.1);
    c.needsUpdate = true;
  }
  segBoxObj = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...b.size)),
                                     new THREE.LineBasicMaterial({ color: 0xff7a00 }));
  segBoxObj.position.set(b.center[0], b.center[1], (b.top + b.bottom) / 2);
  segBoxObj.rotation.z = b.yaw;
  scene.add(segBoxObj);
  if (baseRequest) {
    action.stackBase = { x: b.center[0], y: b.center[1], top: b.top, yaw: b.yaw };
    delete action.stackPlace;
    $('cloudInfo').className = 'ok';
    $('cloudInfo').textContent = '已设为叠放底块。请确认橙框贴合顶面边缘，再生成预览。';
    showAuto();
    return;
  }
  $('cloudInfo').className = 'muted';
  $('cloudInfo').textContent = m.text + '，正在生成抓取姿态…';
  requestGrasp(true);
}
function requestGrasp(fromBox) {
  graspFromBox = fromBox && !!segBox;
  send({ type: 'grasp_auto', pos: fromBox && segBox ? segBox.center : targetPose().pos, mode: radioVal('graspMode'), box: segBox,
         dist: (+$('apDist').value || 8) / 100, lift: (+$('liftH').value || 10) / 100, seed: lastQ, from_box: fromBox && !!segBox });
}
function onGrasp(m) {
  $('cloudInfo').className = m.ok ? 'ok' : 'bad';
  $('cloudInfo').textContent = (graspFromBox && segText ? segText + '。' : '') + m.text;   // 框出来的大小也留着
  if (!m.ok) return;
  const jump = lastQ ? Math.max(...m.q.map((v, j) => Math.abs(v - lastQ[j]))) : 0;
  setJoints(m.q);                          // 按解好的关节角摆，目标（抓取中心）等正解回来更新
  if (m.width) $('objW').value = (m.width * 100).toFixed(1);   // 框出来的物体宽度填进“物体宽度”（勾上检查时用）
}
$('graspRegen').onclick = () => whenTargetReady(() => {
  $('cloudInfo').className = 'muted';
  $('cloudInfo').textContent = '按现在的抓取中心生成姿态…';
  requestGrasp(false);
});
$('graspKeep').onclick = () => whenTargetReady(() => {
  $('cloudInfo').className = 'muted';
  $('cloudInfo').textContent = '正在找能到的手臂摆法（夹爪方向不变）…';
  graspFromBox = false;
  send({ type: 'grasp_keep', ...targetPose(), seed: lastQ, dist: (+$('apDist').value || 8) / 100,
         lift: (+$('liftH').value || 10) / 100, width: closeWidth() });
});


// ---------------- 右键菜单 ----------------
// 三维视图：右键点一下（不拖）弹菜单，右键拖动照样是平移视角。点到什么弹什么：关键帧小球 / 灰色模型或目标 / 点云 / 空白处。
// 关键帧列表、动作列表的每一行也能右键。菜单里的每一项都是调现成的按钮功能，不另写一套逻辑
const ctxMenu = document.createElement('div');
ctxMenu.id = 'ctxmenu';
ctxMenu.hidden = true;
document.body.appendChild(ctxMenu);
let ctxItems = [];
function showMenu(x, y, title, items) {     // items：{ label, fn, danger, off } 或 '-'（分隔线）
  ctxItems = items;
  ctxMenu.innerHTML = (title ? `<div class="ctxtitle">${title}</div>` : '') + items.map((it, i) => it === '-' ? '<hr>'
    : `<button data-i="${i}" class="${it.danger ? 'danger' : ''}" ${it.off ? 'disabled' : ''}>${it.label}</button>`).join('');
  ctxMenu.hidden = false;
  const w = ctxMenu.offsetWidth, h = ctxMenu.offsetHeight;
  ctxMenu.style.left = Math.min(x, innerWidth - w - 8) + 'px';
  ctxMenu.style.top = Math.min(y, innerHeight - h - 8) + 'px';
}
const hideMenu = () => { ctxMenu.hidden = true; };
ctxMenu.addEventListener('click', (e) => {
  const b = e.target.closest('button[data-i]');
  if (!b) return;
  hideMenu();
  ctxItems[+b.dataset.i].fn();
});
document.addEventListener('pointerdown', (e) => { if (!ctxMenu.contains(e.target)) hideMenu(); }, true);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideMenu(); });
window.addEventListener('blur', hideMenu);

const ROBOT_BUSY = () => playing;
function selectKf(i) {
  selected = i;
  const k = keyframes[i];
  if (k.q) setTargetFromJoints(k.pos, k.quat, k.q); else setTarget(k.pos, k.quat);
  setSimGrip(gripAt(i));
  renderKeyframes();
}
function kfMenu(x, y, i) {
  const k = keyframes[i];
  if (!k) return;
  showMenu(x, y, `帧 ${i + 1}${k.tag ? ' · ' + k.tag : ''}`, [
    { label: '真机移到这帧', fn: () => gotoKeyframe(i) },
    { label: '选中这帧（目标放过去）', fn: () => selectKf(i) },
    '-',
    { label: '用当前目标更新这帧', fn: () => { selected = i; renderKeyframes(); $('kfUpdate').click(); } },
    { label: '在这帧前面插入当前目标', fn: () => { selected = i - 1; $('kfAdd').click(); } },
    { label: '在这帧后面插入当前目标', fn: () => { selected = i; $('kfAdd').click(); } },
    { label: '上移', fn: () => { selected = i; $('kfUp').click(); }, off: i === 0 },
    { label: '下移', fn: () => { selected = i; $('kfDown').click(); }, off: i === keyframes.length - 1 },
    '-',
    { label: '删除这帧', danger: true, fn: () => { selected = i; $('kfDel').click(); } },
    ...commonTail(),
  ]);
}
// 每个三维菜单最后都有的几项：真机执行、刷新点云
function commonTail() {      // 2026-11-04 用户：真夹爪开合也放进右键
  return ['-', { label: '真夹爪张开', fn: () => send({ type: 'gripper', closed: false }) },
          { label: '真夹爪合上', fn: () => send({ type: 'gripper', closed: true }) },
          '-', { label: '真机执行轨迹', fn: () => $('execute').click(), off: !keyframes.length || ROBOT_BUSY() },
          { label: '刷新点云', fn: () => $('cloudGet').click() }];
}
// 放置点放到点云上点到的地方的正上方：按选中动作的抓取点，物体底面离这里 DROP_GAP（1.2 cm）松手（抓取点离桌多高，夹着时物体底就在抓取中心下面多少）
const DROP_GAP = 0.012;   // 2026-11-04 用户：放盘子时还可以再低 3 mm（原来 1.5 cm）
function placeAbove(p) {
  const a = actions[actSel];
  const hold = a && a.pick ? a.pick.pos[2] - cfg.table_z : 0.056;    // 没设抓取点时按 3.3 cm 方块竖直夹（抓取中心离桌约 56 mm；2026-11-09 自动生成的指尖离桌 3→6 mm）
  const pos = [p[0], p[1], p[2] + hold + DROP_GAP];
  if (!a || !a.pick) { setTarget(pos, target.quaternion.toArray()); grabLater('place'); return; }
  // 夹爪跟着手臂转：搬运途中 J6 不动（2026-11-04 用户：放盘子朝向无所谓；原来照抄抓取点朝向，J6 要反着转和 J1 一样多，
  // 25~45° 的白转）。先按“抓取点→放置点绕底座转过的角度”转夹爪，逆解回来后再按 J6 还差多少补一次。叠放要对齐时用 Q/E 手动调
  const dAz = Math.atan2(pos[1], pos[0]) - Math.atan2(a.pick.pos[1], a.pick.pos[0]);
  setTarget(pos, new THREE.Quaternion().setFromAxisAngle(AXIS_Z, dAz).multiply(new THREE.Quaternion().fromArray(a.pick.quat)).toArray());
  const j6 = a.pick.q ? a.pick.q[5] : null;
  whenTargetReady(() => {
    const d = j6 !== null && lastQ ? ((lastQ[5] - j6) % 360 + 540) % 360 - 180 : 0;   // 度，挪到 -180~180
    if (Math.abs(d) > 0.3 && Math.abs(d) < 30) {     // 绕竖直轴转 +1° 时 J6 −1°（夹爪朝下）
      target.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(AXIS_Z, THREE.MathUtils.degToRad(d)));
      showTarget(); requestSolve(true);
    }
    grabLater('place');
  });
}
function cloudHit(e) {
  if (!cloud || !cloud.visible) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = 0.006;
  ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
  const hit = ray.intersectObject(cloud)[0];
  if (!hit) return null;
  const pos = cloud.geometry.attributes.position;
  return [pos.getX(hit.index), pos.getY(hit.index), pos.getZ(hit.index)];
}
function viewMenu(e) {
  camera.updateMatrixWorld(); scene.updateMatrixWorld();   // 窗口在后台时可能还没重画过，矩阵是旧的
  const rect = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.params.Line.threshold = 0.003;       // 默认 1（米）：坐标轴、路径线会把整片空白都算成“点到了”
  ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);
  const x = e.clientX, y = e.clientY;
  const kfHit = ray.intersectObjects(kfGroup.children.filter((o) => o.userData.kf !== undefined))[0];
  if (kfHit) { kfMenu(x, y, kfHit.object.userData.kf); return; }
  const modelHit = preview && ray.intersectObject(preview.root, true).find((h) => h.object.isMesh);   // 只认灰色模型的实体
  const p = cloudHit(e);
  if (modelHit) {
    const many = actions.length > 1 ? `动作 ${actSel + 1} 的` : '';
    showMenu(x, y, '目标（灰色模型）', [
      { label: `设为${many}抓取点`, fn: () => grabLater('pick') },
      { label: `设为${many}放置点`, fn: () => grabLater('place') },
      { label: '记录为关键帧', fn: () => $('kfAdd').click() },
      '-',
      { label: '真机移到这里（空格）', fn: gotoTarget, off: ROBOT_BUSY() },
      { label: '放到初始位姿', fn: () => $('toHome').click() },
      { label: '回到实际位置', fn: () => $('toActual').click() },
      ...commonTail(),
    ]);
    return;
  }
  if (p) {
    showMenu(x, y, `点云 · ${p.map((v) => (v * 1000).toFixed(0)).join(', ')} mm`, [
      { label: '框选这个物体，生成抓取姿态', fn: () => {
        $('cloudLive').checked = false;
        $('cloudInfo').className = 'muted'; $('cloudInfo').textContent = '正在框选物体…';
        lastSeg = p;
        send({ type: 'segment', point: p, tol: +$('segTol').value, square: stackMode() });
      } },
      { label: '抓取中心放这里（夹爪方向不变）', fn: () => {
        $('cloudLive').checked = false;
        target.position.fromArray(p); targetReady = true; showTarget();
        graspFromBox = false;
        $('cloudInfo').className = 'muted'; $('cloudInfo').textContent = '正在找能到的手臂摆法（夹爪方向不变）…';
        send({ type: 'grasp_keep', ...targetPose(), seed: lastQ, dist: (+$('apDist').value || 8) / 100,
               lift: (+$('liftH').value || 10) / 100, width: closeWidth() });
      } },
      stackMode() ? { label: '设为叠放底块', fn: () => selectStackBase(p) }
        : { label: `${actions.length > 1 ? `动作 ${actSel + 1} 的` : ''}放置点设在这里上方（物体底离这里 ${(DROP_GAP * 100).toFixed(1)} cm 松手）`, fn: () => placeAbove(p) },
      ...commonTail(),
    ]);
    return;
  }
  showMenu(x, y, '', [
    { label: '真机执行轨迹', fn: () => $('execute').click(), off: !keyframes.length || ROBOT_BUSY() },
    { label: '预览轨迹', fn: () => $('preview').click(), off: !keyframes.length },
    { label: '真机移到目标（空格）', fn: gotoTarget, off: ROBOT_BUSY() },
    { label: '刷新点云', fn: () => $('cloudGet').click() },
    '-',
    { label: '真夹爪张开', fn: () => send({ type: 'gripper', closed: false }) },
    { label: '真夹爪合上', fn: () => send({ type: 'gripper', closed: true }) },
    '-',
    { label: '真机回初始位姿', fn: () => $('home').click(), off: ROBOT_BUSY() },
    { label: '目标放到初始位姿', fn: () => $('toHome').click() },
    { label: '目标回到实际位置', fn: () => $('toActual').click() },
    { label: '重置视角', fn: resetView },
    '-',
    { label: '急停', danger: true, fn: () => send({ type: 'estop' }) },
  ]);
}
function resetView() {
  const a = THREE.MathUtils.degToRad(cfg.home_yaw_deg || 0), rot = (v) => v.applyAxisAngle(new THREE.Vector3(0, 0, 1), a);
  camera.position.copy(rot(new THREE.Vector3(1.25, -1.15, 0.95)));
  orbit.target.copy(rot(new THREE.Vector3(0.15, 0, 0.3)));
  orbit.update();
}
let rightDown = null;
renderer.domElement.addEventListener('pointerdown', (e) => { if (e.button === 2) rightDown = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (e.button !== 2 || !rightDown) return;
  const moved = Math.hypot(e.clientX - rightDown[0], e.clientY - rightDown[1]);
  rightDown = null;
  if (moved <= 4) viewMenu(e);           // 右键拖动是平移视角，不弹菜单
});
renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
// 关键帧列表、动作列表：右键哪一行弹哪一行的菜单
$('kflist').addEventListener('contextmenu', (e) => {
  const li = e.target.closest('li[data-i]');
  if (!li) return;
  e.preventDefault();
  kfMenu(e.clientX, e.clientY, +li.dataset.i);
});
$('actlist').addEventListener('contextmenu', (e) => {
  const li = e.target.closest('li[data-a]');
  if (!li) return;
  e.preventDefault();
  const i = +li.dataset.a;
  actSel = i;
  showAuto();
  const name = actions.length > 1 ? `动作 ${i + 1}` : '抓放';
  showMenu(e.clientX, e.clientY, name, [
    { label: '用当前目标重设抓取点', fn: () => grabLater('pick') },
    { label: '用当前目标重设放置点', fn: () => grabLater('place') },
    { label: '＋ 在后面添加一个动作', fn: () => { actions.splice(i + 1, 0, { pick: null, place: null }); actSel = i + 1; showAuto(); } },
    '-',
    { label: actions.length > 1 ? '删除这个动作' : '清空', danger: true, fn: () => {
      if (actions.length === 1) actions = [{ pick: null, place: null }]; else actions.splice(i, 1);
      actSel = Math.min(actSel, actions.length - 1);
      showAuto();
    } },
  ]);
});

// “显示工作区”开关记在浏览器里（下次打开还是这样）
(() => {
  try { $('camZones').checked = localStorage.getItem('camZones') === '1'; } catch (e) { /* 读不了就默认关 */ }
  $('camZones').addEventListener('change', () => { try { localStorage.setItem('camZones', $('camZones').checked ? '1' : '0'); } catch (e) { /* 存不了就算了 */ } });
})();

// ---- 模型控制（π0.5，2026-11-06）：台式机上的模型服务给动作，后台 policy_run.py 发给机械臂 ----
const polStore = (k, v) => { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } };
function polShow() { $('polReplanVal').textContent = $('polReplan').value; $('polMaxVal').textContent = $('polMax').value; }
$('polReplan').addEventListener('input', polShow);
$('polMax').addEventListener('input', polShow);
window.addEventListener('pageshow', polShow);     // 从缓存回来时 resetSliders 把滑块放回默认值，数字跟着刷新
polShow();
if (polStore('polAddr')) $('polAddr').value = polStore('polAddr');
if (polStore('polTask')) $('polTask').value = polStore('polTask');
function polCfg() {
  polStore('polAddr', $('polAddr').value.trim());
  polStore('polTask', $('polTask').value.trim());
  return { addr: $('polAddr').value.trim(), task: $('polTask').value.trim(), replan: +$('polReplan').value, max_s: +$('polMax').value,
           raw: polRaw };
}
$('polTest').addEventListener('click', () => {
  $('polInfo').className = '';
  $('polInfo').textContent = '正在连模型服务、用现在的画面问一次（服务刚起来时第一次要十几秒）…';
  send({ type: 'policy_test', ...polCfg() });
});
// “解除限制”按钮：按一下开、再按一下关；开着时按钮变红。不存到浏览器里，每次打开网页都是关的（安全起见）
let polRaw = false;
function polRawShow() {
  $('polRaw').textContent = polRaw ? '解除限制：开（只保留防撞桌、防自撞、关节限位）' : '解除限制：关';
  $('polRaw').classList.toggle('danger', polRaw);
}
$('polRaw').addEventListener('click', () => { polRaw = !polRaw; polRawShow(); });
polRawShow();
$('polStart').addEventListener('click', () => send({ type: 'policy_start', ...polCfg() }));
$('polStop').addEventListener('click', () => send({ type: 'stop' }));
