"""Conveyor-line version of the inspection cell.

Parts ride an infeed lane (+y). A camera gate on the lane images the top face and both side faces
(normals +-x). At the pick point the lane stops, the robot grips the part by exactly those two already
inspected faces and lifts it; the lane restarts at once. At the station the bottom face and the two
faces along the lane direction (normals +-y) are imaged. Good parts are put back into their own slot on
the moving lane (conveyor tracking); defective ones go to the NG box.
"""
import mujoco
import numpy as np

from .cell import DEFECT_TYPES, FACE_NORMALS, HALF, PART_RGBA, frame_on_face, quat_from_mat
from .env import CLOSED, CTRL_DT, OPEN, PANDA_XML, lookat_quat

LX = 0.52                    # lane centre line (x)
BELT_Z = 0.06                # belt surface height
PART_Z = BELT_Z + HALF
GATE_Y = -0.32               # camera gate on the lane
PICK_Y = -0.16                # pick point (lane stops here)
LANE_Y0, LANE_Y1 = -5.40, 1.80
V_LANE = 0.03                # belt speed [m/s]
PITCH = 0.28                 # spacing between parts on the belt
N_PARTS = 8
STATION = np.array([0.25, 0.33, 0.30])
NG_BOX = np.array([0.24, -0.34])
REGRASP = np.array([0.30, 0.16])
REGRASP_TOP = 0.06
NG_TOP = 0.05
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, 0.785])
WRIST_EYE = np.array([0.052, 0.0, 0.030])   # wrist camera position in the hand frame
FINGER_FRICTION = [2.5, 0.02, 0.0005]        # rubber pads on the fingertips (sliding, torsional [m], rolling [m])
GRASP_ASSIST = False                          # True: fix the gripped part to the hand (weld); False: friction only


def _dark(spec_body, **kw):
    kw.setdefault("contype", 0); kw.setdefault("conaffinity", 0)
    return spec_body.add_geom(**kw)


def build_lane():
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):
        spec.delete(key)
    # OpenGL renders at most 8 lights (the headlight included): drop the model's own light, use ours.
    for light in list(spec.lights):
        spec.delete(light)
    spec.option.timestep = 0.0005            # 2 ms let a pinched part creep out of the fingers; 0.5 ms holds it
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    grip = spec.actuator("actuator8")
    gp = list(grip.gainprm); gp[0] = 0.1568627451; grip.gainprm = gp
    bp = list(grip.biasprm); bp[1], bp[2] = -1000.0, -40.0; grip.biasprm = bp
    # Headlight = a light on every camera (like the ring lights fitted to inspection cameras).
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.50, 0.50, 0.50]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 1, 0])
    for name in ("left_finger", "right_finger"):
        for g in spec.body(name).geoms:
            g.friction = FINGER_FRICTION
            g.solref = [0.004, 1.0]                  # stiff pad contact: less creep of the held part
            g.solimp = [0.99, 0.999, 0.001, 0.5, 2]
    # Wrist camera beside the fingers (offset along the hand x axis, so the fingers never block it), looking
    # past the fingertips: the input of the imitation-learned grasping policy.
    _dark(hand, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[WRIST_EYE[0] + 0.008, 0, WRIST_EYE[2] - 0.012],
          size=[0.012, 0.016, 0.014], rgba=[0.12, 0.12, 0.14, 1])
    hand.add_camera(name="wrist", pos=WRIST_EYE.tolist(), fovy=70,
                    quat=lookat_quat(WRIST_EYE, np.array([0.0, 0.0, 0.14]), up=np.array([1.0, 0.0, 0.0])))
    B = mujoco.mjtGeom.mjGEOM_BOX
    C = mujoco.mjtGeom.mjGEOM_CYLINDER
    wb = spec.worldbody
    wb.add_light(pos=[0.45, 0.0, 1.8], dir=[0, 0, -1], diffuse=[0.70, 0.70, 0.70], castshadow=True)
    wb.add_light(pos=[LX, GATE_Y, BELT_Z + 0.26], dir=[0, 0, -1], diffuse=[0.35, 0.35, 0.35], castshadow=True,
                 cutoff=35)
    # Station lighting: ring light under the part, one lamp beside the side camera.
    wb.add_light(pos=[STATION[0], STATION[1], 0.17], dir=[0, 0, 1], diffuse=[0.55, 0.55, 0.55], castshadow=True,
                 cutoff=60)
    wb.add_light(pos=[STATION[0] - 0.17, STATION[1] - 0.05, STATION[2] + 0.04], dir=[1, 0.25, -0.2],
                 diffuse=[0.55, 0.55, 0.55], castshadow=True, cutoff=50)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], rgba=[0.50, 0.46, 0.40, 1])

    # Conveyor: belt (collision surface), side rails, legs, end rollers.
    ly = (LANE_Y1 + LANE_Y0) / 2; lh = (LANE_Y1 - LANE_Y0) / 2
    wb.add_geom(name="belt", type=B, pos=[LX, ly, BELT_Z - 0.01], size=[0.06, lh, 0.01],
                rgba=[0.13, 0.14, 0.15, 1], friction=[1.0, 0.02, 0.002])
    for sx in (-1, 1):
        _dark(wb, type=B, pos=[LX + sx * 0.068, ly, BELT_Z - 0.009], size=[0.008, lh, 0.013],
              rgba=[0.62, 0.64, 0.67, 1])
    for yy in np.arange(LANE_Y0 + 0.1, LANE_Y1, 0.5):
        for sx in (-1, 1):
            _dark(wb, type=B, pos=[LX + sx * 0.06, yy, (BELT_Z - 0.03) / 2], size=[0.01, 0.01, (BELT_Z - 0.03) / 2],
                  rgba=[0.45, 0.46, 0.48, 1])
    # Stripes painted on the belt so its motion is visible.
    belt_marks = []
    for k, yy in enumerate(np.arange(LANE_Y0, LANE_Y1, 0.07)):
        belt_marks.append(_dark(wb, name=f"mark{k}", type=B, pos=[LX, yy, BELT_Z + 0.0003],
                                size=[0.055, 0.004, 0.0003], rgba=[0.22, 0.23, 0.25, 1]))
    # Camera 1: top camera on a bracket over the lane.
    _dark(wb, type=B, pos=[LX + 0.17, GATE_Y, 0.21], size=[0.012, 0.012, 0.21], rgba=[0.75, 0.77, 0.80, 1])
    _dark(wb, type=B, pos=[LX + 0.085, GATE_Y, 0.42], size=[0.095, 0.012, 0.012], rgba=[0.75, 0.77, 0.80, 1])
    _dark(wb, type=B, pos=[LX, GATE_Y, BELT_Z + 0.29], size=[0.02, 0.02, 0.03], rgba=[0.12, 0.12, 0.14, 1])
    wb.add_camera(name="gateTop", pos=[LX, GATE_Y, BELT_Z + 0.26], quat=[0.7071068, 0, 0, -0.7071068], fovy=20)

    # Inspection station: up-looking camera with ring light under the part.
    tgt = STATION
    _dark(wb, type=B, pos=[tgt[0], tgt[1], 0.04], size=[0.06, 0.06, 0.04], rgba=[0.15, 0.15, 0.17, 1])
    _dark(wb, type=C, pos=[tgt[0], tgt[1], 0.10], size=[0.018, 0.025, 0], rgba=[0.12, 0.12, 0.14, 1])
    _dark(wb, type=C, pos=[tgt[0], tgt[1], 0.127], size=[0.03, 0.002, 0], rgba=[0.95, 0.95, 0.9, 1])
    wb.add_camera(name="stBottom", pos=[tgt[0], tgt[1], 0.16], quat=[0, 1, 0, 0], fovy=36)
    # Camera 3: one side camera (-x side, looking along +x); the wrist turns to show each free face.
    # Facing the hand along x keeps joint 7 near the middle of its range, so it can turn 180 deg either way.
    eye = np.array([tgt[0] - 0.20, tgt[1], tgt[2]])
    cam = wb.add_camera(name="stSide", pos=eye.tolist(), fovy=21)
    cam.quat = lookat_quat(eye, tgt)
    _dark(wb, type=B, pos=[eye[0] - 0.05, eye[1], eye[2]], size=[0.025, 0.02, 0.02], rgba=[0.12, 0.12, 0.14, 1])
    _dark(wb, type=C, pos=[eye[0] - 0.05, eye[1], eye[2] / 2], size=[0.008, eye[2] / 2 - 0.02, 0],
          rgba=[0.3, 0.3, 0.32, 1])
    # Regrasp pad.
    wb.add_geom(name="regrasp_pad", type=B, pos=[REGRASP[0], REGRASP[1], REGRASP_TOP / 2],
                size=[0.05, 0.05, REGRASP_TOP / 2], rgba=[0.62, 0.64, 0.68, 1], friction=[1.0, 0.02, 0.002])

    # NG box (red tray with walls).
    ng = wb.add_body(name="ng_box", pos=[NG_BOX[0], NG_BOX[1], 0])
    ng.add_geom(type=B, pos=[0, 0, NG_TOP / 2], size=[0.08, 0.08, NG_TOP / 2], rgba=[0.75, 0.12, 0.10, 1])
    for sx, sy, lx, lyy in [(1, 0, 0.005, 0.085), (-1, 0, 0.005, 0.085), (0, 1, 0.085, 0.005), (0, -1, 0.085, 0.005)]:
        ng.add_geom(type=B, pos=[sx * 0.08, sy * 0.08, NG_TOP + 0.015], size=[lx, lyy, 0.015],
                    rgba=[0.80, 0.15, 0.12, 1])

    for i in range(N_PARTS):
        b = wb.add_body(name=f"part{i}", pos=[LX, PICK_Y - 0.45 - i * PITCH, PART_Z])
        b.add_freejoint(name=f"part{i}_joint")
        b.add_geom(name=f"part{i}_geom", type=B, size=[HALF] * 3, rgba=PART_RGBA, mass=0.05,
                   friction=[2.0, 0.05, 0.01], condim=6)
        common = dict(contype=0, conaffinity=0, density=0, group=1)
        b.add_geom(name=f"part{i}_hole", type=C, pos=[0, 0, HALF + 0.0002], size=[0.0055, 0.0002, 0],
                   rgba=[0.18, 0.18, 0.20, 1], **common)
        b.add_geom(name=f"part{i}_scratch", type=B, size=[0.01, 0.0009, 0.0002], rgba=[0.08, 0.08, 0.09, 0], **common)
        b.add_geom(name=f"part{i}_stain", type=C, size=[0.005, 0.0002, 0], rgba=[0.38, 0.22, 0.08, 0], **common)
        b.add_geom(name=f"part{i}_chip", type=B, size=[0.004, 0.004, 0.0002], rgba=[0.25, 0.25, 0.27, 0], **common)
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"hold{i}", name1="hand", name2=f"part{i}",
                          objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)

    eye = np.array([1.35, 0.55, 0.95])
    cam = wb.add_camera(name="video", pos=eye.tolist(), fovy=50)
    cam.quat = lookat_quat(eye, np.array([0.38, -0.02, 0.12]))
    eye2 = np.array([0.95, -0.95, 0.55])
    cam = wb.add_camera(name="video_gate", pos=eye2.tolist(), fovy=40)
    cam.quat = lookat_quat(eye2, np.array([LX, GATE_Y + 0.05, 0.10]))
    eye3 = np.array([0.50, 0.60, 0.40])
    cam = wb.add_camera(name="video_station", pos=eye3.tolist(), fovy=40)
    cam.quat = lookat_quat(eye3, np.array([0.23, 0.30, 0.23]))
    return spec.compile()


class LaneCell:
    """Simulation state + conveyor logic. Gate captures are reported through `gate_hook(i)`."""

    def __init__(self, img_size=128, video_size=(1280, 720)):
        self.model = build_lane()
        m = self.model
        self.data = mujoco.MjData(m)
        self.ik_data = mujoco.MjData(m)
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.jlo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.jhi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self.tcp = m.site("tcp").id
        self.hand_body = m.body("hand").id
        self.part_qadr = [m.jnt_qposadr[m.joint(f"part{i}_joint").id] for i in range(N_PARTS)]
        self.part_dadr = [m.jnt_dofadr[m.joint(f"part{i}_joint").id] for i in range(N_PARTS)]
        self.part_body = [m.body(f"part{i}").id for i in range(N_PARTS)]
        self.defect_gid = [{t: m.geom(f"part{i}_{t}").id for t in DEFECT_TYPES} for i in range(N_PARTS)]
        for g in self.defect_gid:
            for gid in g.values():
                m.geom_sameframe[gid] = 0
        self.mark_gid = [m.geom(f"mark{k}").id for k in range(len([1 for g in range(m.ngeom) if m.geom(g).name.startswith("mark")]))]
        for gid in self.mark_gid:
            m.geom_sameframe[gid] = 0
        self.mark_y0 = m.geom_pos[self.mark_gid, 1].copy()
        self.hold_eq = [m.equality(f"hold{i}").id for i in range(N_PARTS)]
        self.finger_body = {m.body("left_finger").id, m.body("right_finger").id}
        # Stiff grasp assist: the part must turn with the hand even when the wrist turns quickly.
        m.eq_solref[self.hold_eq] = [4 * m.opt.timestep, 1.0]
        m.eq_solimp[self.hold_eq] = [0.99, 0.999, 0.001, 0.5, 2]
        self.n_sub = int(round(CTRL_DT / m.opt.timestep))
        self.renderer = mujoco.Renderer(m, img_size, img_size)
        self.video_renderer = mujoco.Renderer(m, video_size[1], video_size[0])
        self.img_size = img_size
        self.rng = np.random.default_rng(0)
        self.defect_list = [[] for _ in range(N_PARTS)]
        self.time = 0.0
        self.frame_hook = None
        self.gate_hook = None
        self.sensor_hook = None   # called every control step (camera stream)
        self.continuous = False   # True: the lane never stops; parts arrive at uneven spacing and angles
        self.lane_running = True
        self.odo = 0.0
        self.held = None
        self.picked = set()      # parts already taken at the pick point (never stop the lane again)
        self.gated = set()

    # ------------------------------------------------------------ defects
    def set_defect(self, i, specs):
        m = self.model
        for gid in self.defect_gid[i].values():
            m.geom_rgba[gid, 3] = 0.0
        specs = [] if specs is None else (specs if isinstance(specs, list) else [specs])
        self.defect_list[i] = specs
        for sp in specs:
            gid = self.defect_gid[i][sp["type"]]
            n = FACE_NORMALS[sp["face"]]
            r = frame_on_face(n, sp["ang"])
            m.geom_pos[gid] = n * (HALF + 0.0003) + r[:, 0] * sp["u"] + r[:, 1] * sp["v"]
            m.geom_quat[gid] = quat_from_mat(r)
            sc = sp["scale"]
            if sp["type"] == "scratch":
                m.geom_size[gid] = [0.006 + 0.008 * sc, 0.0009 + 0.0006 * sc, 0.0002]
            elif sp["type"] == "stain":
                m.geom_size[gid] = [0.0030 + 0.0035 * sc, 0.0002, 0]
            else:
                m.geom_size[gid] = [0.003 + 0.003 * sc, 0.003 + 0.003 * sc, 0.0002]
            m.geom_rgba[gid, 3] = 1.0

    def random_defect(self, rng, t=None, face=None):
        t = t or DEFECT_TYPES[rng.integers(3)]
        face = int(rng.integers(6)) if face is None else face
        lim = 0.009
        if t == "chip":
            u, v = rng.choice([-1, 1]) * 0.014, rng.uniform(-lim, lim)
        else:
            u, v = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
        return dict(type=t, face=face, u=float(u), v=float(v), ang=float(rng.uniform(0, np.pi)),
                    scale=float(rng.uniform(0, 1)))

    def defects_on_face(self, i, face):
        return [sp["type"] for sp in self.defect_list[i] if sp["face"] == face]

    def is_bad(self, i):
        return len(self.defect_list[i]) > 0

    # -------------------------------------------------------------- reset
    def reset(self, seed=None, defect_rate=0.3, n_defect=None, continuous=False):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr:self.finger_qadr + 2] = 0.04
        if n_defect is None:
            bad = self.rng.random(N_PARTS) < defect_rate
        else:
            bad = np.zeros(N_PARTS, bool); bad[self.rng.choice(N_PARTS, n_defect, replace=False)] = True
        self.continuous = continuous
        y = PICK_Y - 0.45
        for i in range(N_PARTS):
            a = self.part_qadr[i]
            if continuous:   # as parts come from the previous machine: uneven gaps, side offset, any angle
                yaw = self.rng.uniform(-np.pi / 4, np.pi / 4)
                x = LX + self.rng.uniform(-0.015, 0.015)
                if i:
                    y -= self.rng.uniform(0.56, 0.68)
            else:
                yaw = self.rng.uniform(-0.2, 0.2)
                x = LX + self.rng.uniform(-0.006, 0.006)
                y = PICK_Y - 0.45 - i * PITCH
            d.qpos[a:a + 3] = [x, y, PART_Z + 0.0005]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.set_defect(i, [self.random_defect(self.rng)] if bad[i] else None)
        d.ctrl[:7] = HOME_Q
        d.ctrl[7] = OPEN
        d.eq_active[self.hold_eq] = 0
        mujoco.mj_forward(m, d)
        for _ in range(int(round(0.2 / m.opt.timestep))):   # let the parts settle for 0.2 s
            mujoco.mj_step(m, d)
        self.time, self.odo = 0.0, 0.0
        self.lane_running, self.held = True, None
        self.picked, self.gated = set(), set()
        m.geom_pos[self.mark_gid, 1] = self.mark_y0
        return bad

    # ---------------------------------------------------------------- lane
    def on_belt(self, i):
        p = self.data.xpos[self.part_body[i]]
        return (abs(p[0] - LX) < 0.06 and LANE_Y0 < p[1] < LANE_Y1 and p[2] < PART_Z + 0.006
                and i != self.held and self.part_body[i] not in self._pinched())

    def _pinched(self):
        """Parts the fingers are touching: the belt slips under them, the hand moves them."""
        m, d = self.model, self.data
        out = set()
        for j in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[j].geom1], m.geom_bodyid[d.contact[j].geom2]
            if b1 in self.finger_body:
                out.add(b2)
            elif b2 in self.finger_body:
                out.add(b1)
        return out

    def _convey(self):
        d, m = self.data, self.model
        if self.lane_running and not self.continuous:
            # Stop the line when an unpicked part reaches the pick point.
            for i in range(N_PARTS):
                if i not in self.picked and self.on_belt(i) and d.qpos[self.part_qadr[i] + 1] >= PICK_Y:
                    self.lane_running = False
        if not self.lane_running:
            return
        dy = V_LANE * CTRL_DT
        self.odo += dy
        for i in range(N_PARTS):
            if self.on_belt(i):
                a = self.part_qadr[i]
                y_prev = d.qpos[a + 1]
                d.qpos[a + 1] = y_prev + dy
                d.qvel[self.part_dadr[i]:self.part_dadr[i] + 6] = 0.0
                if y_prev < GATE_Y <= d.qpos[a + 1] and i not in self.gated:
                    self.gated.add(i)
                    mujoco.mj_forward(m, d)
                    if self.gate_hook is not None:
                        self.gate_hook(i)
        # Scroll the belt stripes.
        y = self.mark_y0 + (self.odo % 0.07)
        m.geom_pos[self.mark_gid, 1] = y

    def step(self, ctrl):
        self.data.ctrl[:7] = np.clip(ctrl[:7], self.jlo, self.jhi)
        self.data.ctrl[7] = np.clip(ctrl[7], CLOSED, OPEN)
        self._convey()
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.time += CTRL_DT
        if self.sensor_hook is not None:
            self.sensor_hook(self)
        if self.frame_hook is not None:
            self.frame_hook(self)

    # ------------------------------------------------------- grasp assist
    def attach(self, i):
        if not GRASP_ASSIST:                 # the rubber-padded fingers alone hold the part
            self.held = i
            return
        m, d = self.model, self.data
        hp, hq = d.xpos[self.hand_body], d.xquat[self.hand_body]
        pp, pq = d.xpos[self.part_body[i]], d.xquat[self.part_body[i]]
        inv = np.zeros(4); mujoco.mju_negQuat(inv, hq)
        rel_p = np.zeros(3); mujoco.mju_rotVecQuat(rel_p, pp - hp, inv)
        rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, inv, pq)
        eq = self.hold_eq[i]
        m.eq_data[eq, 0:3] = 0.0; m.eq_data[eq, 3:6] = rel_p; m.eq_data[eq, 6:10] = rel_q; m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self.held = i

    def detach_all(self):
        self.data.eq_active[self.hold_eq] = 0
        self.held = None

    # ------------------------------------------------------------- queries
    def render(self, cam):
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def render_video(self, cam="video"):
        self.video_renderer.update_scene(self.data, camera=cam)
        return self.video_renderer.render().copy()

    def part_pos(self, i):
        return self.data.xpos[self.part_body[i]].copy()

    def part_mat(self, i):
        return self.data.xmat[self.part_body[i]].reshape(3, 3).copy()

    def part_yaw(self, i):
        r = self.part_mat(i)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def tcp_pose(self):
        return self.data.site_xpos[self.tcp].copy(), self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def facing_face(self, i, direction):
        r = self.part_mat(i)
        normals = (r @ FACE_NORMALS.T).T
        return int(np.argmax(normals @ (direction / np.linalg.norm(direction))))

    def in_hand(self, i):
        """The part touches both fingers and sits between the pads (check for the friction-only hold)."""
        m, d = self.model, self.data
        pb, touch = self.part_body[i], set()
        for j in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[j].geom1], m.geom_bodyid[d.contact[j].geom2]
            if b1 == pb and b2 in self.finger_body:
                touch.add(b2)
            elif b2 == pb and b1 in self.finger_body:
                touch.add(b1)
        tcp, rot = self.tcp_pose()
        rel = rot.T @ (self.part_pos(i) - tcp)
        return len(touch) == 2 and float(np.abs(rel).max()) < 0.02

    def in_ng_box(self, i):
        p = self.part_pos(i)
        return abs(p[0] - NG_BOX[0]) < 0.085 and abs(p[1] - NG_BOX[1]) < 0.085 and p[2] > NG_TOP

    def ik(self, pos, rot, q_init, iters=60, tol=1e-4):
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.arm_qadr] = q_init
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_fwdPosition(m, d)
            cp = d.site_xpos[self.tcp]; cr = d.site_xmat[self.tcp].reshape(3, 3)
            err = np.concatenate([pos - cp, 0.5 * sum(np.cross(cr[:, k], rot[:, k]) for k in range(3))])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            j = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(6), err)
            nproj = np.eye(7) - np.linalg.pinv(j) @ j
            dq += nproj @ (0.05 * (HOME_Q - d.qpos[self.arm_qadr]))
            d.qpos[self.arm_qadr] = np.clip(d.qpos[self.arm_qadr] + dq, self.jlo, self.jhi)
        return d.qpos[self.arm_qadr].copy()
