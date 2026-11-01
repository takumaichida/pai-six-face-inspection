"""MuJoCo tabletop sorting scene: Franka Panda, three coloured cubes, two trays."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

COLORS = {
    "red": [0.85, 0.12, 0.10, 1.0],
    "green": [0.10, 0.65, 0.20, 1.0],
    "blue": [0.12, 0.30, 0.90, 1.0],
}
COLOR_NAMES = list(COLORS)
BINS = {"left": np.array([0.40, 0.34]), "right": np.array([0.40, -0.34])}
BIN_NAMES = list(BINS)
TASKS = [(c, b) for c in COLOR_NAMES for b in BIN_NAMES]  # 6 tasks

CUBE_HALF = 0.02
BIN_HALF = 0.075  # inner half-width of a tray
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, 0.785])
CTRL_DT = 0.1  # policy / expert control period [s]
OPEN, CLOSED = 255.0, 0.0


def build_model():
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):
        spec.delete(key)
    spec.option.timestep = 0.002
    # Soft contacts let a grasped cube creep out of the fingers; these settings make grasps hold.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    # Stiffer gripper servo: the stock gains squeeze a 4 cm cube with only ~2 N, which lets it slip in transit.
    grip = spec.actuator("actuator8")
    gp = list(grip.gainprm); gp[0] = 0.1568627451; grip.gainprm = gp
    bp = list(grip.biasprm); bp[1], bp[2] = -1000.0, -40.0; grip.biasprm = bp

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0.005, 0.005], rgba=[1, 0, 1, 0])
    # Wrist camera looking along the gripper (hand +z), mounted on the hand's side.
    hand.add_camera(name="wrist", pos=[0.06, 0.0, 0.02], quat=[0, 1, 0, 0], fovy=75)

    wb = spec.worldbody
    wb.add_light(pos=[0.4, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)
    wb.add_light(pos=[-0.5, 0.6, 1.5], dir=[0.5, -0.4, -1], diffuse=[0.3, 0.3, 0.3])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], rgba=[0.55, 0.47, 0.38, 1])
    # Table top marking the workspace (visual only).
    wb.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.45, 0, 0.0005], size=[0.33, 0.52, 0.0005],
                rgba=[0.86, 0.84, 0.80, 1], contype=0, conaffinity=0)

    # Two trays (floor + four low walls).
    tray_rgba = {"left": [0.95, 0.75, 0.15, 1], "right": [0.55, 0.30, 0.70, 1]}
    for name, (bx, by) in BINS.items():
        t = wb.add_body(name=f"bin_{name}", pos=[bx, by, 0])
        t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.002], size=[BIN_HALF, BIN_HALF, 0.002],
                   rgba=tray_rgba[name])
        h, th = 0.016, 0.005
        for sx, sy, lx, ly in [(1, 0, th, BIN_HALF + th), (-1, 0, th, BIN_HALF + th),
                               (0, 1, BIN_HALF + th, th), (0, -1, BIN_HALF + th, th)]:
            t.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[sx * (BIN_HALF + th), sy * (BIN_HALF + th), h],
                       size=[lx, ly, h], rgba=tray_rgba[name])

    for cname, rgba in COLORS.items():
        b = wb.add_body(name=f"cube_{cname}", pos=[0.5, 0, CUBE_HALF])
        b.add_freejoint(name=f"cube_{cname}_joint")
        b.add_geom(name=f"cube_{cname}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                   rgba=rgba, mass=0.05, friction=[2.0, 0.02, 0.002], condim=4)

    # Fixed cameras: top-down for the policy, over-the-shoulder for the video.
    wb.add_camera(name="top", pos=[0.42, 0.0, 0.82], quat=[0.7071068, 0, 0, -0.7071068], fovy=62)
    cam = wb.add_camera(name="video", pos=[-0.12, 0.0, 1.30], fovy=58)
    cam.quat = lookat_quat(np.array([-0.12, 0.0, 1.30]), np.array([0.46, 0.0, 0.0]))
    return spec.compile()


def lookat_quat(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Quaternion for a MuJoCo camera (looks along -z, image-up +y) at eye looking at target."""
    f = target - eye
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1).reshape(-1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def grasp_rot(yaw):
    """Gripper pointing down (-z) with fingers closing along the yawed y axis."""
    rx = np.diag([1.0, -1.0, -1.0])
    return rotz(yaw) @ rx


class SortEnv:
    def __init__(self, img_size=128, video_size=(640, 360)):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.tcp = m.site("tcp").id
        self.cube_qadr = {c: m.jnt_qposadr[m.joint(f"cube_{c}_joint").id] for c in COLOR_NAMES}
        self.cube_body = {c: m.body(f"cube_{c}").id for c in COLOR_NAMES}
        self.jlo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.jhi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        self.n_sub = int(round(CTRL_DT / m.opt.timestep))
        self.img_size = img_size
        self.renderer = mujoco.Renderer(m, img_size, img_size)
        self.video_renderer = mujoco.Renderer(m, video_size[1], video_size[0])
        self.rng = np.random.default_rng(0)
        self.task = TASKS[0]

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, task=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = task if task is not None else TASKS[self.rng.integers(len(TASKS))]
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr:self.finger_qadr + 2] = 0.04
        # Cubes: random positions with minimum separation, random yaw.
        pts = []
        while len(pts) < 3:
            p = np.array([self.rng.uniform(0.33, 0.58), self.rng.uniform(-0.17, 0.17)])
            if all(np.linalg.norm(p - q) > 0.085 for q in pts):
                pts.append(p)
        for c, p in zip(COLOR_NAMES, pts):
            a = self.cube_qadr[c]
            yaw = self.rng.uniform(-np.pi / 4, np.pi / 4)
            d.qpos[a:a + 3] = [p[0], p[1], CUBE_HALF + 0.001]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.ctrl[:7] = HOME_Q
        d.ctrl[7] = OPEN
        mujoco.mj_forward(m, d)
        for _ in range(50):
            mujoco.mj_step(m, d)
        return self.obs()

    # ------------------------------------------------------------------- step
    def step(self, ctrl):
        ctrl = np.asarray(ctrl, dtype=float)
        self.data.ctrl[:7] = np.clip(ctrl[:7], self.jlo, self.jhi)
        self.data.ctrl[7] = np.clip(ctrl[7], CLOSED, OPEN)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        return self.obs()

    def state(self):
        q = self.data.qpos[self.arm_qadr].copy()
        w = self.data.qpos[self.finger_qadr] + self.data.qpos[self.finger_qadr + 1]
        return np.concatenate([q, [w]]).astype(np.float32)

    def render(self, cam):
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def render_video(self):
        self.video_renderer.update_scene(self.data, camera="video")
        return self.video_renderer.render().copy()

    def obs(self):
        return {"top": self.render("top"), "wrist": self.render("wrist"), "state": self.state()}

    # ---------------------------------------------------------------- queries
    def cube_pos(self, c):
        return self.data.xpos[self.cube_body[c]].copy()

    def cube_yaw(self, c):
        mat = self.data.xmat[self.cube_body[c]].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def tcp_pose(self):
        return self.data.site_xpos[self.tcp].copy(), self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def in_bin(self, c):
        p = self.cube_pos(c)
        for b, (bx, by) in BINS.items():
            if abs(p[0] - bx) < BIN_HALF and abs(p[1] - by) < BIN_HALF and p[2] < 0.06:
                return b
        return None

    def success(self):
        color, bin_name = self.task
        if self.in_bin(color) != bin_name:
            return False
        return all(self.in_bin(c) is None for c in COLOR_NAMES if c != color)

    # --------------------------------------------------------------------- IK
    def ik(self, pos, rot, q_init, iters=60, tol=1e-4):
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.arm_qadr] = q_init
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_fwdPosition(m, d)
            cp = d.site_xpos[self.tcp]
            cr = d.site_xmat[self.tcp].reshape(3, 3)
            ep = pos - cp
            er = 0.5 * sum(np.cross(cr[:, i], rot[:, i]) for i in range(3))
            err = np.concatenate([ep, er])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            j = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])
            lam = 1e-3
            dq = j.T @ np.linalg.solve(j @ j.T + lam * np.eye(6), err)
            # Null-space pull toward the home posture keeps the elbow tidy.
            n = np.eye(7) - np.linalg.pinv(j) @ j
            dq += n @ (0.05 * (HOME_Q - d.qpos[self.arm_qadr]))
            d.qpos[self.arm_qadr] = np.clip(d.qpos[self.arm_qadr] + dq, self.jlo, self.jhi)
        return d.qpos[self.arm_qadr].copy()
