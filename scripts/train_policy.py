"""Train the imitation-learned grasp policy (ACT-style action chunks) on the teacher demonstrations.

Targets for each step are the teacher's next CHUNK actions; steps past the end of an episode are masked out, as
in ACT. The hand-motion target is the teacher's motion minus the belt's own motion (sortbot.imitation.belt_step;
the demonstrations are all on the running belt). Loss: L1 on the hand motion, binary cross-entropy on closing the
fingers. Episodes are split 90/10.

usage: train_policy.py <epochs> -> models/grasp_policy.pt, results/policy_train.json
"""
import glob
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from sortbot.env import CTRL_DT  # noqa: E402
from sortbot.imitation import ACT_SCALE, CHUNK, ChunkPolicy, to_tensor  # noqa: E402
from sortbot.lane import V_LANE  # noqa: E402


def load():
    parts = [np.load(f) for f in sorted(glob.glob("data/demos_*.npz"))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("img", "state", "act", "close", "ep", "step")}
    good = {int(e[0]) for p in parts for e in p["episodes"] if e[1]}
    keep = np.isin(cat["ep"], list(good))
    cat = {k: v[keep] for k, v in cat.items()}
    n = len(cat["ep"])
    yaw = np.arctan2(cat["state"][:, 3], cat["state"][:, 2])            # hand yaw of each step
    belt = V_LANE * CTRL_DT
    cat["act"] = cat["act"] - np.stack([np.sin(yaw) * belt, np.cos(yaw) * belt, np.zeros(n)], 1).astype(np.float32)
    tgt = np.zeros((n, CHUNK, 4), np.float32); mask = np.zeros((n, CHUNK), np.float32)
    start = np.r_[0, np.flatnonzero(np.diff(cat["ep"])) + 1, n]
    for a, b in zip(start[:-1], start[1:]):
        for t in range(a, b):
            m = min(CHUNK, b - t)
            tgt[t, :m, :3] = cat["act"][t:t + m] / ACT_SCALE
            tgt[t, :m, 3] = cat["close"][t:t + m]
            mask[t, :m] = 1
    return cat, tgt, mask


def augment(x):
    """Random shift of up to 4 px and brightness / contrast jitter."""
    b = len(x)
    pad = F.pad(x, (4, 4, 4, 4), mode="replicate")
    i, j = np.random.randint(0, 9, 2)
    x = pad[:, :, i:i + x.shape[2], j:j + x.shape[3]]
    return x * (1 + 0.15 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)) + 0.2 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)


def main(epochs):
    cat, tgt, mask = load()
    eps = np.unique(cat["ep"]); rng = np.random.default_rng(0); rng.shuffle(eps)
    val_eps = set(eps[: len(eps) // 10].tolist())
    is_val = np.isin(cat["ep"], list(val_eps))
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{len(eps)} episodes, {len(tr)} train / {len(va)} val steps, device {dev}")
    net = ChunkPolicy().to(dev)
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": 1e-4},
                             {"params": [p for n, p in net.named_parameters() if not n.startswith("backbone")], "lr": 3e-4}],
                            weight_decay=1e-4)
    steps = epochs * (len(tr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 3e-4], total_steps=steps, pct_start=0.1)
    img_t, st_t = cat["img"], torch.from_numpy(cat["state"])
    tgt_t, mask_t = torch.from_numpy(tgt), torch.from_numpy(mask)

    def batch_loss(idx, train):
        x = to_tensor(img_t[idx]).to(dev)
        if train:
            x = augment(x)
        out = net(x, st_t[idx].to(dev))
        y, m = tgt_t[idx].to(dev), mask_t[idx].to(dev)
        l_mov = ((out[..., :3] - y[..., :3]).abs().mean(-1) * m).sum() / m.sum()
        l_cls = (F.binary_cross_entropy_with_logits(out[..., 3], y[..., 3], reduction="none") * m).sum() / m.sum()
        acc = (((out[..., 3] > 0).float() == y[..., 3]).float() * m).sum() / m.sum()
        return l_mov + l_cls, l_mov, l_cls, acc, m.sum()

    log = []
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        for b in range(len(tr) // 128):
            loss = batch_loss(perm[b * 128:(b + 1) * 128], True)[0]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            r = np.array([torch.stack(batch_loss(va[k:k + 256], False)).cpu().numpy() for k in range(0, len(va), 256)])
        r = (r[:, :4] * r[:, 4:5]).sum(0) / r[:, 4].sum()       # weighted by the number of unmasked actions
        log.append(dict(epoch=ep + 1, val_loss=float(r[0]), val_move_l1_mm=float(r[1] * ACT_SCALE * 1000),
                        val_close_bce=float(r[2]), val_close_acc=float(r[3]), sec=round(time.time() - t0)))
        print(log[-1], flush=True)
        torch.save(net.state_dict(), "models/grasp_policy.pt")
    json.dump(dict(episodes=int(len(eps)), train_steps=int(len(tr)), val_steps=int(len(va)), chunk=CHUNK, log=log),
              open("results/policy_train.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
