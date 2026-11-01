"""Synthetic face images for the conveyor line: the top camera on the lane, and the bottom and side cameras
at the station, rendered in the same poses the line uses, with lighting / pose / colour randomisation."""
import mujoco
import numpy as np

from .cell import DEFECT_TYPES, HALF, PART_RGBA
from .env import grasp_rot
from .lane import GATE_Y, HOME_Q, LX, N_PARTS, PART_Z, PICK_Y, PITCH, STATION, LaneCell
from .lane_routine import Z_GRASP

VIEWS = ["gateTop", "stBottom", "stSide"]
VIEW_DIR = {"gateTop": [0, 0, 1.0], "stBottom": [0, 0, -1.0], "stSide": [-1.0, 0, 0]}
VIEW_P = [0.3, 0.2, 0.5]  # the side camera sees four of the six faces
CLASSES = ["ok"] + DEFECT_TYPES


def quat_z(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


class LaneFaceSampler:
    def __init__(self, seed=0, defect_part_rate=0.7):
        self.c = LaneCell(video_size=(320, 180))
        self.rng = np.random.default_rng(seed)
        self.rate = defect_part_rate
        self.base_light = self.c.model.light_diffuse.copy()
        c = self.c
        c.reset(seed=seed, n_defect=0)
        q = c.ik(STATION, grasp_rot(0.0), HOME_Q, iters=200)
        q_spun = q.copy(); q_spun[6] += -np.pi if q[6] > 0 else np.pi  # the wrist turned 180 deg
        self.q_station = [q, q_spun]
        self.q_ready = c.ik(np.array([LX, PICK_Y, PART_Z + 0.17]), grasp_rot(np.pi / 2), HOME_Q, iters=200)

    def _random_defects(self):
        if self.rng.random() > self.rate:
            return None
        n = self.rng.choice([1, 1, 1, 2, 3])
        types = list(self.rng.permutation(DEFECT_TYPES)[:n])
        faces = self.rng.permutation(6)[:n]
        out = []
        for t, f in zip(types, faces):
            out.append(self.c.random_defect(self.rng, t=t, face=int(f)))
        return out

    def sample(self):
        c, rng = self.c, self.rng
        m, d = c.model, c.data
        m.light_diffuse[:] = self.base_light * rng.uniform(0.75, 1.25)
        tone = rng.uniform(-0.05, 0.05)
        for j in range(N_PARTS):
            m.geom_rgba[m.geom(f"part{j}_geom").id, :3] = np.clip(np.array(PART_RGBA[:3]) + tone, 0, 1)
        i = int(rng.integers(N_PARTS))
        for j in range(N_PARTS):
            c.set_defect(j, self._random_defects())
        view = VIEWS[rng.choice(len(VIEWS), p=VIEW_P)]
        c.detach_all()
        # Other parts spread along the lane at the belt pitch.
        for j in range(N_PARTS):
            a = c.part_qadr[j]
            d.qpos[a:a + 3] = [LX, PICK_Y - 0.6 - j * PITCH, PART_Z]
            d.qpos[a + 3:a + 7] = quat_z(rng.uniform(-0.2, 0.2))
        a = c.part_qadr[i]
        if view.startswith("gate"):
            d.qpos[c.arm_qadr] = self.q_ready + rng.normal(0, 0.02, 7)
            d.qpos[c.finger_qadr:c.finger_qadr + 2] = 0.04
            d.qpos[a:a + 3] = [LX + rng.uniform(-0.006, 0.006), GATE_Y + rng.uniform(-0.004, 0.004), PART_Z]
            d.qpos[a + 3:a + 7] = quat_z(rng.uniform(-0.2, 0.2))
            # Neighbours one pitch up/down the lane.
            for k, j in enumerate([x for x in range(N_PARTS) if x != i][:2]):
                aj = c.part_qadr[j]
                d.qpos[aj:aj + 3] = [LX, GATE_Y + (PITCH if k == 0 else -PITCH), PART_Z]
        else:
            d.qpos[c.arm_qadr] = self.q_station[rng.integers(2)] + rng.normal(0, 0.004, 7)
            d.qpos[c.finger_qadr:c.finger_qadr + 2] = 0.02
            mujoco.mj_forward(m, d)
            tcp, rot = c.tcp_pose()
            hand_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
            # Before the regrasp the part sits at +90 deg to the hand, after it at 0 (any half turn).
            part_yaw = hand_yaw + rng.integers(4) * np.pi / 2 + rng.normal(0, 0.04)
            d.qpos[a:a + 3] = tcp + np.array([0, 0, -(Z_GRASP - PART_Z)]) + rng.normal(0, 0.004, 3)
            d.qpos[a + 3:a + 7] = quat_z(part_yaw)
        mujoco.mj_forward(m, d)
        img = c.render(view)
        face = c.facing_face(i, np.array(VIEW_DIR[view]))
        types = c.defects_on_face(i, face)
        return img, (CLASSES.index(types[0]) if types else 0), view, face


def generate(n, seed, out_path):
    s = LaneFaceSampler(seed=seed)
    imgs = np.zeros((n, s.c.img_size, s.c.img_size, 3), np.uint8)
    labels = np.zeros(n, np.int64); views = np.zeros(n, np.int64)
    for k in range(n):
        img, lab, view, _ = s.sample()
        imgs[k], labels[k], views[k] = img, lab, VIEWS.index(view)
    np.savez_compressed(out_path, imgs=imgs, labels=labels, views=views)
    return out_path
