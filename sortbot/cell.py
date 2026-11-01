"""Inspection cell: Franka Panda, 10 parts on a 2x5 grid, defects on any face, reject stand, 3 inspection cameras."""
import mujoco
import numpy as np

from .env import CLOSED, CTRL_DT, OPEN, PANDA_XML, grasp_rot, lookat_quat

HALF = 0.02                       # part half size (4 cm cube)
GRID_X = [0.42, 0.56]
GRID_Y = [-0.22, -0.11, 0.0, 0.11, 0.22]
SLOTS = [np.array([x, y]) for x in GRID_X for y in GRID_Y]       # 10 home positions
STATION = np.array([0.36, -0.44, 0.26])                          # TCP pose for bottom/side views
REJECT = np.array([0.30, 0.42])                                  # reject stand centre
REJECT_TOP = 0.04
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, 0.785])
DEFECT_TYPES = ["scratch", "stain", "chip"]
PART_RGBA = [0.70, 0.72, 0.75, 1.0]
FACE_NORMALS = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
FACE_NAMES = ["top", "bottom", "+x", "-x", "+y", "-y"]           # in the part's own frame


def quat_from_mat(r):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(r, float).reshape(-1))
    return q


def frame_on_face(n, ang):
    """Rotation whose z axis is the face normal n, spun by ang around it."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, n); x /= np.linalg.norm(x)
    y = np.cross(n, x)
    c, s = np.cos(ang), np.sin(ang)
    x2, y2 = c * x + s * y, -s * x + c * y
    return np.stack([x2, y2, n], axis=1)


def build_cell():
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):
        spec.delete(key)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    grip = spec.actuator("actuator8")
    gp = list(grip.gainprm); gp[0] = 0.1568627451; grip.gainprm = gp
    bp = list(grip.biasprm); bp[1], bp[2] = -1000.0, -40.0; grip.biasprm = bp
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 1, 0])
    hand.add_camera(name="wrist", pos=[0.06, 0.0, 0.02], quat=[0, 1, 0, 0], fovy=34)
    # LED ring light next to the wrist camera, pointing along the gripper.
    hand.add_light(name="wrist_light", pos=[0.06, 0.0, 0.03], dir=[0, 0, 1], diffuse=[0.45, 0.45, 0.45],
                   castshadow=False, cutoff=40)

    wb = spec.worldbody
    wb.add_light(pos=[0.45, 0.0, 1.8], dir=[0, 0, -1], diffuse=[0.75, 0.75, 0.75], castshadow=True)
    wb.add_light(pos=[-0.4, 0.5, 1.4], dir=[0.5, -0.4, -1], diffuse=[0.25, 0.25, 0.25], castshadow=False)
    # Ring-light style illumination at the inspection station.
    wb.add_light(pos=[STATION[0] + 0.15, STATION[1] - 0.15, STATION[2] + 0.10],
                 dir=[-0.6, 0.6, -0.4], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    wb.add_light(pos=[STATION[0], STATION[1], 0.02], dir=[0, 0, 1], diffuse=[0.3, 0.3, 0.3], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], rgba=[0.50, 0.44, 0.37, 1])
    wb.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.49, 0, 0.0005], size=[0.13, 0.30, 0.0005],
                rgba=[0.20, 0.24, 0.30, 1], contype=0, conaffinity=0)
    # Reject stand (red) and inspection station marker.
    st = wb.add_body(name="reject_stand", pos=[REJECT[0], REJECT[1], 0])
    st.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, REJECT_TOP / 2], size=[0.09, 0.09, REJECT_TOP / 2],
                rgba=[0.80, 0.15, 0.12, 1])
    wb.add_geom(name="station_base", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[STATION[0], STATION[1], 0.02],
                size=[0.06, 0.06, 0.02], rgba=[0.15, 0.15, 0.17, 1], contype=0, conaffinity=0)

    for i, s in enumerate(SLOTS):
        b = wb.add_body(name=f"part{i}", pos=[s[0], s[1], HALF])
        b.add_freejoint(name=f"part{i}_joint")
        b.add_geom(name=f"part{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, rgba=PART_RGBA,
                   mass=0.05, friction=[2.0, 0.05, 0.01], condim=6)
        # Visual-only details: a threaded hole on the top face, and defect slots (one shown per bad part).
        common = dict(contype=0, conaffinity=0, density=0, group=1)
        b.add_geom(name=f"part{i}_hole", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, HALF + 0.0002],
                   size=[0.0055, 0.0002, 0], rgba=[0.18, 0.18, 0.20, 1], **common)
        b.add_geom(name=f"part{i}_scratch", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.0009, 0.0002],
                   rgba=[0.08, 0.08, 0.09, 0], **common)
        b.add_geom(name=f"part{i}_stain", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.005, 0.0002, 0],
                   rgba=[0.38, 0.22, 0.08, 0], **common)
        b.add_geom(name=f"part{i}_chip", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.004, 0.0002],
                   rgba=[0.25, 0.25, 0.27, 0], **common)

    # Grasp assist: a weld per part, switched on once the fingers have closed on it.
    for i in range(len(SLOTS)):
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"hold{i}", name1="hand", name2=f"part{i}",
                          objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
    # Cameras at the station: one looking up (bottom face) and two facing each other along x (side faces).
    tgt = STATION.copy()
    wb.add_camera(name="bottom", pos=[tgt[0], tgt[1], 0.12], quat=[0, 1, 0, 0], fovy=32)
    for name, dx in [("sideF", 0.20), ("sideB", -0.20)]:
        eye = np.array([tgt[0] + dx, tgt[1], tgt[2]])
        cam = wb.add_camera(name=name, pos=eye.tolist(), fovy=21)
        cam.quat = lookat_quat(eye, tgt)
        # Visual housing for the camera (no collision).
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[eye[0] + np.sign(dx) * 0.025, eye[1], eye[2]],
                    size=[0.025, 0.02, 0.02], rgba=[0.12, 0.12, 0.14, 1], contype=0, conaffinity=0)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[eye[0] + np.sign(dx) * 0.025, eye[1], eye[2] / 2],
                    size=[0.008, eye[2] / 2 - 0.02, 0], rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[tgt[0], tgt[1], 0.08], size=[0.018, 0.04, 0],
                rgba=[0.12, 0.12, 0.14, 1], contype=0, conaffinity=0)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[tgt[0], tgt[1], 0.121], size=[0.03, 0.002, 0],
                rgba=[0.95, 0.95, 0.9, 1], contype=0, conaffinity=0)   # ring light
    wb.add_camera(name="overview", pos=[0.45, 0.0, 1.05], quat=[0.7071068, 0, 0, -0.7071068], fovy=62)
    eye = np.array([1.30, 0.06, 1.02])
    cam = wb.add_camera(name="video", pos=eye.tolist(), fovy=52)
    cam.quat = lookat_quat(eye, np.array([0.28, -0.02, 0.14]))
    return spec.compile()


class InspectionCell:
    def __init__(self, img_size=128, video_size=(960, 540)):
        self.model = build_cell()
        m = self.model
        self.data = mujoco.MjData(m)
        self.ik_data = mujoco.MjData(m)
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.jlo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.jhi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self.tcp = m.site("tcp").id
        self.part_qadr = [m.jnt_qposadr[m.joint(f"part{i}_joint").id] for i in range(10)]
        self.part_body = [m.body(f"part{i}").id for i in range(10)]
        self.defect_gid = [{t: m.geom(f"part{i}_{t}").id for t in DEFECT_TYPES} for i in range(10)]
        # Defect geoms are created at the body origin, so the compiler marks them "same frame as body"
        # and kinematics would ignore the per-episode pose we write; turn that optimisation off.
        for g in self.defect_gid:
            for gid in g.values():
                m.geom_sameframe[gid] = 0
        self.n_sub = int(round(CTRL_DT / m.opt.timestep))
        self.hold_eq = [m.equality(f"hold{i}").id for i in range(10)]
        self.hand_body = m.body("hand").id
        self.renderer = mujoco.Renderer(m, img_size, img_size)
        self.video_renderer = mujoco.Renderer(m, video_size[1], video_size[0])
        self.img_size = img_size
        self.rng = np.random.default_rng(0)
        self.defects = [None] * 10
        self.time = 0.0
        self.frame_hook = None

    # ----------------------------------------------------------- defects
    def set_defect(self, i, spec):
        """spec: None (good part), one defect dict(type, face, u, v, ang, scale), or a list of them."""
        m = self.model
        for t, gid in self.defect_gid[i].items():
            m.geom_rgba[gid, 3] = 0.0
        specs = [] if spec is None else (spec if isinstance(spec, list) else [spec])
        self.defects[i] = specs[0] if len(specs) == 1 else (specs or None)
        self.defect_list = getattr(self, "defect_list", [[] for _ in range(10)])
        self.defect_list[i] = specs
        for sp in specs:
            self._apply_defect(i, sp)

    def _apply_defect(self, i, spec):
        m = self.model
        gid = self.defect_gid[i][spec["type"]]
        n = FACE_NORMALS[spec["face"]]
        r = frame_on_face(n, spec["ang"])
        # Two in-face axes for the offset.
        pos = n * (HALF + 0.0003) + r[:, 0] * spec["u"] + r[:, 1] * spec["v"]
        m.geom_pos[gid] = pos
        m.geom_quat[gid] = quat_from_mat(r)
        sc = spec["scale"]
        if spec["type"] == "scratch":
            m.geom_size[gid] = [0.006 + 0.008 * sc, 0.0009 + 0.0006 * sc, 0.0002]
        elif spec["type"] == "stain":
            m.geom_size[gid] = [0.0030 + 0.0035 * sc, 0.0002, 0]
        else:
            m.geom_size[gid] = [0.003 + 0.003 * sc, 0.003 + 0.003 * sc, 0.0002]
        m.geom_rgba[gid, 3] = 1.0

    def defects_on_face(self, i, face):
        return [sp["type"] for sp in getattr(self, "defect_list", [[]] * 10)[i] if sp["face"] == face]

    def random_defect(self, rng, t=None):
        t = t or DEFECT_TYPES[rng.integers(3)]
        face = int(rng.integers(6))
        sc = rng.uniform(0, 1)
        lim = 0.009
        if t == "chip":   # chips sit near an edge
            u, v = rng.choice([-1, 1]) * 0.014, rng.uniform(-lim, lim)
        else:
            u, v = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
        return dict(type=t, face=face, u=float(u), v=float(v), ang=float(rng.uniform(0, np.pi)), scale=float(sc))

    # ------------------------------------------------------------- reset
    def reset(self, seed=None, defect_rate=0.3, n_defect=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr:self.finger_qadr + 2] = 0.04
        if n_defect is None:
            bad = self.rng.random(10) < defect_rate
        else:
            bad = np.zeros(10, bool); bad[self.rng.choice(10, n_defect, replace=False)] = True
        for i, s in enumerate(SLOTS):
            a = self.part_qadr[i]
            yaw = self.rng.uniform(-0.25, 0.25)
            jitter = self.rng.uniform(-0.008, 0.008, 2)
            d.qpos[a:a + 3] = [s[0] + jitter[0], s[1] + jitter[1], HALF + 0.0005]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.set_defect(i, self.random_defect(self.rng) if bad[i] else None)
        d.ctrl[:7] = HOME_Q
        d.ctrl[7] = OPEN
        d.eq_active[self.hold_eq] = 0
        mujoco.mj_forward(m, d)
        for _ in range(100):
            mujoco.mj_step(m, d)
        self.time = 0.0
        return bad

    # -------------------------------------------------------------- step
    def step(self, ctrl):
        self.data.ctrl[:7] = np.clip(ctrl[:7], self.jlo, self.jhi)
        self.data.ctrl[7] = np.clip(ctrl[7], CLOSED, OPEN)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.time += CTRL_DT
        if self.frame_hook is not None:
            self.frame_hook(self)

    def render(self, cam):
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def render_video(self, cam="video"):
        self.video_renderer.update_scene(self.data, camera=cam)
        return self.video_renderer.render().copy()

    # ------------------------------------------------------ grasp assist
    def attach(self, i):
        """Freeze the current hand->part pose into the weld and switch it on."""
        m, d = self.model, self.data
        hp, hq = d.xpos[self.hand_body], d.xquat[self.hand_body]
        pp, pq = d.xpos[self.part_body[i]], d.xquat[self.part_body[i]]
        inv = np.zeros(4); mujoco.mju_negQuat(inv, hq)
        rel_p = np.zeros(3); mujoco.mju_rotVecQuat(rel_p, pp - hp, inv)
        rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, inv, pq)
        eq = self.hold_eq[i]
        m.eq_data[eq, 0:3] = 0.0
        m.eq_data[eq, 3:6] = rel_p
        m.eq_data[eq, 6:10] = rel_q
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1

    def detach_all(self):
        self.data.eq_active[self.hold_eq] = 0

    # ----------------------------------------------------------- queries
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
        """Index of the part face whose world normal is closest to `direction` (world)."""
        r = self.part_mat(i)
        normals = (r @ FACE_NORMALS.T).T
        return int(np.argmax(normals @ (direction / np.linalg.norm(direction))))

    def on_reject(self, i):
        p = self.part_pos(i)
        return abs(p[0] - REJECT[0]) < 0.09 and abs(p[1] - REJECT[1]) < 0.09 and p[2] > REJECT_TOP

    def in_slot(self, i):
        p = self.part_pos(i)
        return np.linalg.norm(p[:2] - SLOTS[i]) < 0.03 and p[2] < 0.03

    # --------------------------------------------------------------- IK
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
