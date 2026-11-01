"""Generate camera-1 images of parts at random positions / angles under the camera and train the pose network.

usage: train_pose.py <images> [epochs]  -> models/pose.pt, results/pose_test.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')


def render_batch(args):
    n, seed = args
    import mujoco
    from sortbot.cell import PART_RGBA
    from sortbot.lane import GATE_Y, LX, N_PARTS, PART_Z, LaneCell
    c = LaneCell(video_size=(320, 180))
    rng = np.random.default_rng(seed)
    c.reset(seed=seed, n_defect=0)
    m, d = c.model, c.data
    base = m.light_diffuse.copy()
    imgs = np.zeros((n, c.img_size, c.img_size, 3), np.uint8); lab = np.zeros((n, 3), np.float32)
    for k in range(n):
        m.light_diffuse[:] = base * rng.uniform(0.75, 1.25)
        m.geom_pos[c.mark_gid, 1] = c.mark_y0 + rng.uniform(0, 0.07)     # the belt stripes move
        tone = rng.uniform(-0.05, 0.05)
        for j in range(N_PARTS):
            m.geom_rgba[m.geom(f"part{j}_geom").id, :3] = np.clip(np.array(PART_RGBA[:3]) + tone, 0, 1)
            a = c.part_qadr[j]
            d.qpos[a:a + 3] = [LX, -3.0 - 0.3 * j, PART_Z]
        i = int(rng.integers(N_PARTS))
        c.set_defect(i, [c.random_defect(rng)] if rng.random() < 0.4 else None)
        dx, dy, yaw = rng.uniform(-0.02, 0.02), rng.uniform(-0.04, 0.04), rng.uniform(-np.pi, np.pi)
        a = c.part_qadr[i]
        d.qpos[a:a + 3] = [LX + dx, GATE_Y + dy, PART_Z]
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        imgs[k] = c.render("gateTop")
        lab[k] = [dx, dy, yaw]
    return imgs, lab


def main(n_img=6000, epochs=5, workers=8):
    import torch
    from sortbot.pose import SCALE, build_model, decode, to_tensor
    t0 = time.time()
    with Pool(workers) as p:
        parts = p.map(render_batch, [(n_img // workers, 300 + w) for w in range(workers)])
    imgs = np.concatenate([a for a, _ in parts]); lab = np.concatenate([b for _, b in parts])
    print('images', imgs.shape, round(time.time() - t0), 's', flush=True)
    target = np.stack([lab[:, 0], lab[:, 1], np.cos(4 * lab[:, 2]), np.sin(4 * lab[:, 2])], 1) / SCALE
    n_te = len(imgs) // 10
    xtr, ytr, xte, yte, lte = imgs[n_te:], target[n_te:], imgs[:n_te], target[:n_te], lab[:n_te]
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    bs = 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * (len(xtr) // bs))
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(xtr)); tot = 0.0
        for k in range(0, len(perm) - bs + 1, bs):
            idx = perm[k:k + bs]
            x = to_tensor(xtr[idx]); x = x * torch.empty(len(idx), 1, 1, 1).uniform_(0.9, 1.1)
            loss = torch.nn.functional.mse_loss(model(x.to(dev)), torch.from_numpy(ytr[idx]).float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach())
        print(f'epoch {ep + 1} loss {tot / (len(perm) // bs):.4f} ({time.time() - t0:.0f}s)', flush=True)
    model.eval()
    with torch.no_grad():
        out = np.concatenate([model(to_tensor(xte[k:k + 128]).to(dev)).cpu().numpy() for k in range(0, len(xte), 128)])
    dx, dy, yaw = decode(out)
    yerr = np.abs((yaw - lte[:, 2] + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    inside = np.abs(lte[:, 1]) < 0.02          # part fully under the camera
    r = dict(test_images=int(n_te), err_x_mm=float(np.mean(np.abs(dx - lte[:, 0])[inside]) * 1000),
             err_y_mm=float(np.mean(np.abs(dy - lte[:, 1])[inside]) * 1000),
             err_yaw_deg=float(np.degrees(np.mean(yerr[inside]))),
             p95_xy_mm=float(np.percentile(np.hypot(dx - lte[:, 0], dy - lte[:, 1])[inside], 95) * 1000))
    torch.save(model.cpu().state_dict(), 'models/pose.pt')
    json.dump(r, open('results/pose_test.json', 'w'), indent=1)
    print(r)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
