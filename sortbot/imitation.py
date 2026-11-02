"""Imitation-learned grasping of the moving part from the wrist camera (ACT-style action chunking).

Once the robot hovers above a part that camera 1 has tracked, a learned policy takes over: every control step
(10 Hz) it sees the wrist-camera image and the hand state, and predicts the hand motion and the gripper command
for the next CHUNK steps. As in ACT (Action Chunking with Transformers), overlapping chunks are averaged with
exponential weights (temporal ensembling) and the first action is executed.

The teacher is the robot program's own grasp (the phases and speed limits of LaneProgram._grasp_at), given the
true part position from the simulator: a privileged teacher. The student policy sees only the wrist image and
the hand state, so at run time it needs neither the true position nor the encoder.

Actions are in the hand frame (rotated by the hand yaw): the hand target for the next step relative to the
current tool point, and the probability of closing the fingers. The belt's own motion over one step (known from
the encoder rate) is added outside the network, so the policy learns only the correction on top of it; otherwise
it has to learn how the belt direction turns with the hand yaw, and it lagged behind parts gripped at large yaws.
"""
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .env import CLOSED, CTRL_DT, OPEN, rotz
from .lane import PART_Z, V_LANE
from .lane_routine import V_MAX, grasp_ok

CHUNK = 10                 # predicted future steps (1 s)
WRIST_PX = 96              # wrist image size
ACT_SCALE = 0.03           # hand motion per step is normalised by this [m]
MAX_STEPS = 45             # the policy has this many steps to close the fingers
CLOSE_STEPS = 6            # steps with the fingers closed before the grasp is checked
LEAD = 0.1                 # the teacher aims one control step ahead of the moving part [s]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def wrist_image(c):
    if not hasattr(c, "_wrist_renderer"):
        c._wrist_renderer = mujoco.Renderer(c.model, WRIST_PX, WRIST_PX)
    c._wrist_renderer.update_scene(c.data, camera="wrist")
    return c._wrist_renderer.render().copy()


def hand_state(c, r):
    """Height of the tool point above the part centre, finger opening, hand yaw."""
    tcp, _ = c.tcp_pose()
    return np.array([(tcp[2] - PART_Z) / 0.1, c.data.qpos[c.finger_qadr] / 0.04,
                     np.cos(r.t_yaw), np.sin(r.t_yaw)], np.float32)


def belt_step(c):
    """Belt motion over one control step (world frame), from the encoder rate."""
    return np.array([0.0, V_LANE * CTRL_DT if c.lane_running else 0.0, 0.0])


def to_hand(yaw, v):
    out = np.array(v, float)
    out[:2] = rotz(yaw)[:2, :2].T @ out[:2]
    return out


def from_hand(yaw, a):
    out = np.array(a, float)
    out[:2] = rotz(yaw)[:2, :2] @ out[:2]
    return out


class Teacher:
    """The program's grasp, one control step at a time: approach to 12 cm above the part, descend to grip
    height (settle 5 steps), close the fingers while following the part (6 steps). goal() = part centre."""
    PLAN = [(0.12, 2, False), (0.004, 5, False), (0.004, CLOSE_STEPS, True)]

    def __init__(self, goal):
        self.goal, self.k, self.settle, self.moving = goal, 0, 0, True

    def step(self, t_pos):
        """Returns (next hand target, close?, done)."""
        dz, n_settle, close = self.PLAN[self.k]
        g = np.asarray(self.goal(), float) + [0, 0, dz]
        if close or not self.moving:
            self.settle += 1
            out = g
            if self.settle >= n_settle:
                self.k += 1; self.settle = 0; self.moving = True
            return out, close, self.k >= len(self.PLAN)
        d = g - t_pos
        dist, step = np.linalg.norm(d), V_MAX * CTRL_DT
        if dist <= step:
            self.moving = False
            return g, False, False
        return t_pos + d / dist * step, False, False


class ChunkPolicy(nn.Module):
    """ACT-style policy: wrist-image tokens (ResNet-18 feature map) and a hand-state token go through a
    transformer encoder; CHUNK learned queries decode the next CHUNK actions (dx, dy, dz, close logit)."""

    def __init__(self, d=256, chunk=CHUNK, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        rn = torchvision.models.resnet18(weights=w)
        self.backbone = nn.Sequential(*list(rn.children())[:-2])
        self.proj = nn.Conv2d(512, d, 1)
        self.state = nn.Linear(4, d)
        n_tok = (WRIST_PX // 32) ** 2 + 1
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True), 2)
        self.query = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 8, 4 * d, 0.1, batch_first=True), 2)
        self.head = nn.Linear(d, 4)

    def forward(self, img, state):
        f = self.proj(self.backbone(img)).flatten(2).transpose(1, 2)
        mem = self.enc(torch.cat([f, self.state(state)[:, None]], 1) + self.pos)
        return self.head(self.dec(self.query.expand(len(img), -1, -1), mem))


def to_tensor(imgs):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


class GraspPolicy:
    """Runs the learned policy with ACT's temporal ensembling (older predictions weigh more)."""

    def __init__(self, weights="models/grasp_policy.pt", device="cpu", m=0.1):
        self.dev = torch.device(device)
        self.net = ChunkPolicy(pretrained=False)
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net = self.net.to(self.dev).eval()
        self.m = m
        self.reset()

    def reset(self):
        self.hist, self.t = [], 0

    def __call__(self, img, state):
        with torch.no_grad():
            out = self.net(to_tensor(img[None]).to(self.dev), torch.from_numpy(state[None]).to(self.dev))
        self.hist = [(t0, p) for t0, p in self.hist if self.t - t0 < CHUNK] + [(self.t, out[0].cpu().numpy())]
        cand = np.array([p[self.t - t0] for t0, p in self.hist])
        w = np.exp(-self.m * np.arange(len(cand))); w /= w.sum()
        a = (w[:, None] * cand).sum(0)
        self.t += 1
        return a[:3] * ACT_SCALE, float(a[3])


def policy_grasp(prog, i, yaw, policy, on_step=None):
    """Let the learned policy grip part i from wherever the hand hovers. Same check, hold and lift as the
    program's grasp. Returns True if gripped."""
    c, r = prog.c, prog.r
    r.t_yaw = yaw
    policy.reset()
    closed = 0
    for _ in range(MAX_STEPS):
        img = wrist_image(c)
        if on_step is not None:
            on_step(img)
        a, logit = policy(img, hand_state(c, r))
        tcp, _ = c.tcp_pose()
        r.t_pos = tcp + from_hand(r.t_yaw, a) + belt_step(c)
        if logit > 0 or closed:                       # once the policy closes the fingers they stay closed
            r.grip = CLOSED; closed += 1
        else:
            r.grip = OPEN
        r.tick()
        if closed >= CLOSE_STEPS:
            break
    if closed < CLOSE_STEPS or not grasp_ok(c, i):
        r.gripper(OPEN, steps=4)
        r.move(r.t_pos + [0, 0, 0.10], settle=0)
        return False
    from . import lane as lane_cfg
    from .lane_routine import HOLD
    prog.grip_at = (c.tcp_pose()[0], c.odo)
    c.attach(i); r.grip = HOLD
    r.move(r.t_pos + [0, 0, 0.116], yaw, settle=0)
    if not lane_cfg.GRASP_ASSIST and not c.in_hand(i):   # slipped out while lifting
        c.detach_all(); r.gripper(OPEN, steps=4)
        return False
    return True


def teacher_goal(c, i):
    return lambda: c.part_pos(i) + [0, V_LANE * LEAD, 0] - [0, 0, c.part_pos(i)[2] - PART_Z]


__all__ = ["CHUNK", "WRIST_PX", "ACT_SCALE", "Teacher", "ChunkPolicy", "GraspPolicy", "policy_grasp", "belt_step",
           "wrist_image", "hand_state", "to_hand", "from_hand", "to_tensor", "teacher_goal", "OPEN"]
