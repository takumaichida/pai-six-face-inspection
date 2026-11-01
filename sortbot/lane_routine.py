"""Robot program for the 3-camera conveyor cell.

Camera 1 (lane, top) sees the top face as the part passes. The robot grips the part by its +-x faces,
shows the bottom face (camera 2) and one free side face to the single side camera (camera 3), turns
joint 7 by 180 deg to show the opposite face, regrasps on the pad rotated by 90 deg, shows the two faces
that were under the fingers, and finally puts good parts back into their own slot on the moving belt.
"""
import numpy as np

from .env import CLOSED, CTRL_DT, OPEN, grasp_rot
from .lane import HALF, LX, NG_BOX, NG_TOP, N_PARTS, PART_Z, PICK_Y, REGRASP, REGRASP_TOP, STATION

V_MAX = 0.32
YAW_RATE = 1.6
SPIN_RATE = 2.4
SPIN_BACK = False  # with the side camera on -x the wrist stays mid-range after a half turn
Z_ABOVE = PART_Z + 0.12
Z_GRASP = PART_Z + 0.004
HOLD = 110  # once the part is held, squeeze lightly (about 3 N) so the fingers do not sink into it
TOP_VIEW = ("gateTop", np.array([0, 0, 1.0]), "上面")
BOTTOM_VIEW = ("stBottom", np.array([0, 0, -1.0]), "底面")
SIDE_DIR = np.array([-1.0, 0, 0])
MODES = ("six_face", "no_regrasp", "top_only")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LaneRobot:
    def __init__(self, cell):
        self.c = cell
        pos, rot = cell.tcp_pose()
        self.psi_home = float(np.arctan2(rot[1, 0], rot[0, 0]))
        self.t_pos, self.t_yaw = pos.copy(), self.psi_home
        self.q_cmd = cell.data.qpos[cell.arm_qadr].copy()
        self.grip = OPEN

    def tick(self):
        self.q_cmd = self.c.ik(self.t_pos, grasp_rot(self.t_yaw), self.q_cmd, iters=40)
        self.c.step(np.concatenate([self.q_cmd, [self.grip]]))

    def move(self, goal, yaw=None, settle=2, tol=0.002, max_steps=300):
        """goal: xyz, or a callable returning xyz (tracks a moving target)."""
        yaw = self.t_yaw if yaw is None else yaw
        for _ in range(max_steps):
            g = np.asarray(goal() if callable(goal) else goal, float)
            d = g - self.t_pos
            dist = np.linalg.norm(d)
            step = V_MAX * CTRL_DT
            self.t_pos = g.copy() if dist <= step else self.t_pos + d / dist * step
            dy = wrap(yaw - self.t_yaw)
            self.t_yaw += np.clip(dy, -YAW_RATE * CTRL_DT, YAW_RATE * CTRL_DT)
            self.tick()
            if dist <= max(step, tol) and abs(dy) <= YAW_RATE * CTRL_DT:
                break
        for _ in range(settle):
            if callable(goal):
                self.t_pos = np.asarray(goal(), float)
            self.tick()

    def spin(self, delta=None, settle=3):
        """Turn the hand 180 deg about its axis using joint 7 only (no arm reconfiguration).
        Returns the angle turned so the caller can turn back before the next move."""
        c = self.c
        q7 = self.q_cmd[6]
        if delta is None:
            delta = -np.pi if q7 + np.pi > c.jhi[6] - 0.05 else np.pi
        n = int(np.ceil(abs(delta) / (SPIN_RATE * CTRL_DT)))
        for k in range(n):
            self.q_cmd[6] = q7 + delta * (k + 1) / n
            c.step(np.concatenate([self.q_cmd, [self.grip]]))
        self.t_yaw = self.t_yaw + delta
        for _ in range(settle):
            c.step(np.concatenate([self.q_cmd, [self.grip]]))
        return delta

    def wait(self, steps):
        for _ in range(steps):
            self.tick()

    def gripper(self, value, steps=5):
        self.grip = value
        self.wait(steps)

    def best_yaw(self, pos, base, spin=False):
        """Of the two hand yaws that grip the same pair of faces (base, base + pi), pick the one that keeps
        the wrist (joint 7) clear of its limits and, when asked, leaves room to turn it 180 deg."""
        c, best, best_s = self.c, None, -np.inf
        for k in (0, 1):
            y = self.yaw_near(base + k * np.pi, period=2 * np.pi)
            q7 = c.ik(np.asarray(pos, float), grasp_rot(y), self.q_cmd, iters=200)[6]
            s = min(q7 - c.jlo[6], c.jhi[6] - q7, 0.5)
            if spin:
                s += min(max(c.jhi[6] - q7 - np.pi, q7 - np.pi - c.jlo[6]), 0.3)
            s -= 0.05 * abs(wrap(y - self.t_yaw))
            if s > best_s:
                best, best_s = y, s
        return best

    def yaw_near(self, target, period=np.pi):
        k = np.round((self.t_yaw - target) / period)
        return target + k * period


class LaneProgram:
    def __init__(self, cell, judge, mode="six_face", threshold=0.5, early_exit=True):
        assert mode in MODES
        self.c, self.judge, self.mode = cell, judge, mode
        self.th, self.early_exit = threshold, early_exit
        self.r = LaneRobot(cell)
        self.probs = {i: [] for i in range(N_PARTS)}
        self.views = {i: [] for i in range(N_PARTS)}
        self.log, self.results = [], []
        self.status = ""
        self.capture_listeners = []
        cell.gate_hook = self.on_gate

    # ---------------------------------------------------------- imaging
    def capture(self, i, cam, face_dir, label):
        img = self.c.render(cam)
        p = float(self.judge(cam, img, i))
        face = self.c.facing_face(i, face_dir)
        self.probs[i].append(p); self.views[i].append(label)
        self.log.append(dict(part=i, cam=cam, view=label, face=face, p=p, t=self.c.time))
        for fn in self.capture_listeners:
            fn(i, cam, img, label, p)
        return p

    def on_gate(self, i):
        self.capture(i, *TOP_VIEW)

    def _ng_found(self, i):
        return max(self.probs[i] or [0.0]) > self.th

    # ------------------------------------------------------------- moves
    def _wait_for_part(self, i):
        c, r = self.c, self.r
        self.pick_yaw = r.best_yaw([LX, PICK_Y, Z_GRASP], c.part_yaw(i) + np.pi / 2)
        r.move([LX, PICK_Y, Z_ABOVE + 0.05], self.pick_yaw, settle=1)
        for _ in range(3000):
            if not c.lane_running and c.data.qpos[c.part_qadr[i] + 1] >= PICK_Y - 1e-6:
                return True
            r.tick()
        return False

    def grasp_ok(self, i):
        """The fingers stopped on the part (opening near its width) and the part sits between the pads."""
        c = self.c
        tcp, rot = c.tcp_pose()
        rel = rot.T @ (c.part_pos(i) - tcp)
        opening = float(c.data.qpos[c.finger_qadr])
        return abs(rel[0]) < 0.012 and abs(rel[1]) < 0.012 and 0.014 < opening < 0.026

    def _grasp_at(self, i, pos, yaw):
        """Grip the part at pos (xyz, or a callable for a moving part). Returns False if the grip missed."""
        c, r = self.c, self.r
        at = (lambda dz: lambda: np.asarray(pos(), float) + [0, 0, dz]) if callable(pos) else \
             (lambda dz: np.asarray(pos, float) + [0, 0, dz])
        r.move(at(0.12), yaw)
        r.move(at(0.004), yaw, settle=5)
        r.grip = CLOSED
        for _ in range(6):                       # keep following the part while the fingers close
            if callable(pos):
                r.t_pos = at(0.004)()
            r.tick()
        if not self.grasp_ok(i):
            if getattr(self, "debug", False):
                tcp, rot = c.tcp_pose()
                g = np.asarray(pos() if callable(pos) else pos, float)
                print("grip miss", i, "part-tcp world", np.round((c.part_pos(i) - tcp)[:2] * 1000, 1),
                      "target-part", np.round((g - c.part_pos(i))[:2] * 1000, 1), "t_pos-tcp", np.round((r.t_pos - tcp)[:2] * 1000, 1))
            r.gripper(OPEN, steps=4)
            r.move(at(0.12), yaw, settle=0)
            return False
        c.attach(i); r.grip = HOLD
        lift = r.t_pos + [0, 0, 0.116]
        r.move(lift, yaw, settle=0)
        return True

    def _release_at(self, xyz):
        r = self.r
        r.move([xyz[0], xyz[1], xyz[2] + 0.12])
        r.move([xyz[0], xyz[1], xyz[2] + 0.004], settle=2)
        self.c.detach_all(); r.gripper(OPEN, steps=7)
        r.move([xyz[0], xyz[1], xyz[2] + 0.12], settle=0)

    def _to_ng(self, i):
        n = sum(1 for x in self.results if x["decision"] == "NG")
        off = np.array([((n % 3) - 1) * 0.045, ((n // 3) % 3 - 1) * 0.045])
        self._release_at([NG_BOX[0] + off[0], NG_BOX[1] + off[1], NG_TOP + HALF + 0.03])

    def _show_two_sides(self, i, tag):
        """At the station: show the free face along hand x to the side camera, spin, show the opposite."""
        r = self.r
        r.move(STATION, r.best_yaw(STATION, 0.0, spin=True), settle=4)
        if tag == "1":
            self.capture(i, *BOTTOM_VIEW)
            if self.early_exit and self._ng_found(i):
                return
        self.capture(i, "stSide", SIDE_DIR, f"側面{tag}A")
        if self.early_exit and self._ng_found(i):
            return
        self.status = f"部品{i + 1}: 手首を180°回して反対側の面"
        delta = r.spin()
        self.capture(i, "stSide", SIDE_DIR, f"側面{tag}B")
        if SPIN_BACK:
            r.spin(-delta, settle=1)  # back to the earlier wrist angle before carrying the part on

    def _place_back(self, i, x_slot, odo_pick, y_pick=PICK_Y):
        c, r = self.c, self.r
        self.status = f"部品{i + 1}: 良品 → 流れているレーンの元の位置へ戻す"
        slot = lambda z: lambda: np.array([x_slot, y_pick + (c.odo - odo_pick), z])  # noqa: E731
        r.move(slot(Z_ABOVE), settle=0)
        r.move(slot(Z_GRASP + 0.002), settle=3)
        c.detach_all(); r.gripper(OPEN, steps=7)
        r.move(slot(Z_ABOVE), settle=0)

    # -------------------------------------------------------------- flow
    def run_part(self, i):
        c, r = self.c, self.r
        self.status = f"部品{i + 1}: カメラ1で上面を撮影 → 停止位置へ"
        if not self._wait_for_part(i):
            raise RuntimeError(f"part {i} never reached the pick point")
        t0 = c.time
        p = c.part_pos(i)
        if self._ng_found(i):
            self.status = f"部品{i + 1}: 上面で不良を検出 → NG箱へ"
            self._grasp_at(i, p, self.pick_yaw)
            c.picked.add(i); c.lane_running = True
            self._to_ng(i); decision = "NG"
        elif self.mode == "top_only":
            self.status = f"部品{i + 1}: 上面OK → そのまま流す"
            c.picked.add(i); c.lane_running = True
            decision = "OK"
        else:
            self.status = f"部品{i + 1}: 左右の面を掴んで持ち上げ（レーンは再始動）"
            self._grasp_at(i, p, self.pick_yaw)
            odo_pick = c.odo
            c.picked.add(i); c.lane_running = True
            decision = self.inspect_held(i, p[0], odo_pick)
        rec = dict(part=i, decision=decision, n_views=len(self.probs[i]),
                   p_max=float(max(self.probs[i] or [0])), t_pick=t0, t_done=c.time)
        self.results.append(rec)
        return rec

    def inspect_held(self, i, x_slot, odo_pick, y_pick=PICK_Y):
        """With the part in the hand: bottom + side faces, regrasp, the other two sides, then sort it."""
        c, r = self.c, self.r
        self.status = f"部品{i + 1}: カメラ2で底面、カメラ3で側面"
        self._show_two_sides(i, "1")
        if not self._ng_found(i) and self.mode == "six_face":
            self.status = f"部品{i + 1}: 持ち替え（90°回して掴み直す）"
            self._release_at([REGRASP[0], REGRASP[1], REGRASP_TOP + HALF])
            q = c.part_pos(i)
            self._grasp_at(i, q, r.best_yaw(q + [0, 0, 0.004], r.t_yaw + np.pi / 2))
            self.status = f"部品{i + 1}: 指で隠れていた2面をカメラ3で"
            self._show_two_sides(i, "2")
        if self._ng_found(i):
            self.status = f"部品{i + 1}: 不良を検出 → NG箱へ"
            self._to_ng(i)
            return "NG"
        self._place_back(i, x_slot, odo_pick, y_pick)
        return "OK"

    def run(self):
        for i in range(N_PARTS):
            self.run_part(i)
        self.status = "全数検査完了"
        self.r.move([LX - 0.15, 0.10, 0.35], settle=5)
        return self.results
