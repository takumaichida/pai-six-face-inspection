"""Camera-synchronised picking from a lane that never stops.

Camera 1 streams at the control rate (10 Hz). While a part is in view, the pose network estimates its offset and
angle in every frame. Each frame is stamped with the conveyor encoder reading taken at the same moment, so a track
gives the part's place on the belt, its side offset and its angle. Afterwards the encoder alone says where the
part is (as in industrial conveyor tracking). The estimate reaches the robot `latency` seconds after each frame
(inference + network). With compensation the robot pairs each estimate with the encoder reading of its capture
time; without it, it pairs it with the reading at arrival, so it aims behind the part by belt speed x latency.
The robot then follows the moving part, grips it and continues with the six-face inspection.
"""
import time

import numpy as np

from .env import OPEN
from .imitation import policy_grasp
from .lane import GATE_Y, LX, PICK_Y, V_LANE
from .lane_routine import TOP_VIEW, Z_ABOVE, Z_GRASP, LaneProgram
from .pose import bright_fraction

ON, OFF = 0.08, 0.03          # bright-pixel share that starts / ends a track (presence sensor)
FULL = 0.20                   # frames brighter than this show the whole part; only they are used for the fit
PICK_ZONE = PICK_Y            # the robot grips once the part has reached this y
TRACK_LEAD = 0.1              # one control step of look-ahead [s]


class SyncProgram(LaneProgram):
    def __init__(self, cell, judge, pose, latency=0.3, compensate=True, grasp_policy=None, grasp_only=False, **kw):
        super().__init__(cell, judge, **kw)
        cell.gate_hook = None
        cell.sensor_hook = self.on_frame
        self.pose, self.latency, self.compensate = pose, latency, compensate
        self.grasp_policy = grasp_policy      # imitation-learned grasp (sortbot.imitation); None = the program
        self.grasp_only = grasp_only          # evaluation: set the part aside right after the grip
        self.policy_active, self.wrist_img = False, None
        self.frames, self.in_view = [], False
        self.tracks = []              # one per part, in arrival order
        self.infer_ms = []
        self.picks = []               # per grip: estimate vs truth, success

    # ------------------------------------------------------- camera stream
    def on_frame(self, cell):
        img = cell.render("gateTop")
        frac = bright_fraction(img)
        if not self.in_view and frac > ON:
            self.in_view, self.frames = True, []
        if self.in_view:
            t0 = time.perf_counter()
            dx, dy, yaw = self.pose(img)
            self.infer_ms.append((time.perf_counter() - t0) * 1000)
            self.frames.append((cell.time, dx, dy, yaw, img, frac, cell.odo))
            if frac < OFF:
                self.in_view = False
                self.finish_track(cell)

    def finish_track(self, cell):
        fr = [f for f in self.frames if f[5] > FULL]
        if len(fr) < 3:
            return
        t = np.array([f[0] for f in fr]); y = GATE_Y + np.array([f[2] for f in fr])
        odo = np.array([f[6] for f in fr])
        v = float(np.polyfit(t - t[0], y, 1)[0])                 # belt speed seen by the camera (check only)
        y_belt = float(np.median(y - odo))                      # place on the belt = y - encoder
        x = LX + float(np.median([f[1] for f in fr]))
        a4 = np.array([f[3] for f in fr]) * 4
        yaw = float(np.arctan2(np.sin(a4).mean(), np.cos(a4).mean()) / 4)
        best = min(fr, key=lambda f: abs(f[2]))
        i = len(self.tracks)
        tr = dict(part=i, y_belt=y_belt, v=v, x=x, yaw=yaw, img=best[4], t_avail=float(fr[-1][0]) + self.latency)
        if i < len(cell.part_body):                             # truth, for evaluation only
            p = cell.part_pos(i)
            tr["truth"] = dict(x=float(p[0]), y_at=float(p[1]), t=cell.time, yaw=cell.part_yaw(i))
        self.tracks.append(tr)

    def predict_y(self, tr, t):
        """Part y at time t (t >= now): encoder now + belt speed x (t - now), minus the lag if not compensated."""
        c = self.c
        enc_speed = V_LANE if c.lane_running else 0.0             # encoder rate
        y = tr["y_belt"] + c.odo + enc_speed * (t - c.time)
        return y if self.compensate else y - enc_speed * self.latency

    # ---------------------------------------------------------------- flow
    def wait_track(self, i, timeout=120):
        r, c = self.r, self.c
        t_end = c.time + timeout
        while c.time < t_end:
            if len(self.tracks) > i and c.time >= self.tracks[i]["t_avail"]:
                return self.tracks[i]
            r.tick()
        return None

    def run_part(self, i):
        c, r = self.c, self.r
        self.status = f"部品{i + 1}: カメラ1が撮り続け、位置と向きをAIで推定"
        r.move([LX - 0.05, PICK_ZONE + 0.10, Z_ABOVE + 0.05], settle=0)
        tr = self.wait_track(i)
        if tr is None:
            rec = dict(part=i, decision="MISSED", n_views=0, p_max=0.0, t_pick=c.time, t_done=c.time, grip=False)
            self.results.append(rec)
            return rec
        self.capture_image(i, "gateTop", tr["img"], *TOP_VIEW[1:])
        if self.mode == "top_only" and not self._ng_found(i):
            self.status = f"部品{i + 1}: 上面OK → そのまま流す"
            rec = dict(part=i, decision="OK", n_views=1, p_max=float(max(self.probs[i])), t_pick=c.time,
                       t_done=c.time, grip=None)
            self.results.append(rec)
            return rec
        goal = lambda: np.array([tr["x"], self.predict_y(tr, c.time + TRACK_LEAD), Z_GRASP - 0.004])  # noqa: E731
        yaw = r.best_yaw([tr["x"], PICK_ZONE, Z_GRASP], tr["yaw"] + np.pi / 2)
        self.status = f"部品{i + 1}: 流れる部品を追いかけてつかむ"
        while self.predict_y(tr, c.time + 1.5) < PICK_ZONE:   # let it reach the pick zone
            r.move(lambda: goal() + [0, 0, 0.17], yaw, settle=0, max_steps=1)
        t0 = c.time
        if self.grasp_policy is not None:
            self.status = f"部品{i + 1}: 模倣学習した方策が、手先カメラを見て追いかけてつかむ"
            self.policy_active = True
            ok = policy_grasp(self, i, yaw, self.grasp_policy, on_step=lambda img: setattr(self, "wrist_img", img))
            self.policy_active = False
        else:
            ok = self._grasp_at(i, goal, yaw)
        truth = c.part_pos(i)
        self.picks.append(dict(part=i, ok=bool(ok), err_x=float(tr["x"] - truth[0]),
                               err_y=float(goal()[1] - truth[1]) if not ok else 0.0,
                               v_est=tr["v"], latency=self.latency, compensate=self.compensate))
        if not ok:
            self.status = f"部品{i + 1}: つかみ損ね（未検査でレーンを流れる）"
            rec = dict(part=i, decision="MISSED", n_views=len(self.probs[i]), p_max=float(max(self.probs[i] or [0])),
                       t_pick=t0, t_done=c.time, grip=False)
            self.results.append(rec)
            return rec
        y_pick, odo_pick = float(goal()[1]), c.odo
        c.picked.add(i)
        if self.grasp_only:                   # evaluation of the grip only: take the part off the line
            c.detach_all()
            a = c.part_qadr[i]
            c.data.qpos[a:a + 3] = [2.0 + 0.1 * i, 0.0, 0.03]
            c.data.qvel[c.part_dadr[i]:c.part_dadr[i] + 6] = 0
            r.gripper(OPEN, steps=3)
            rec = dict(part=i, decision="GRIPPED", n_views=len(self.probs[i]), p_max=0.0, t_pick=t0, t_done=c.time,
                       grip=True)
            self.results.append(rec)
            return rec
        if self._ng_found(i):
            self.status = f"部品{i + 1}: 上面で不良を検出 → NG箱へ"
            self._to_ng(i); decision = "NG"
        else:
            decision = self.inspect_held(i, tr["x"], odo_pick, y_pick)
        rec = dict(part=i, decision=decision, n_views=len(self.probs[i]), p_max=float(max(self.probs[i] or [0])),
                   t_pick=t0, t_done=c.time, grip=True)
        self.results.append(rec)
        return rec

    def capture_image(self, i, cam, img, face_dir, label):
        p = float(self.judge(cam, img, i))
        face = self.c.facing_face(i, face_dir)
        self.probs[i].append(p); self.views[i].append(label)
        self.log.append(dict(part=i, cam=cam, view=label, face=face, p=p, t=self.c.time))
        for fn in self.capture_listeners:
            fn(i, cam, img, label, p)
        return p
