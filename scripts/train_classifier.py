"""Fine-tune ResNet-18 to classify one face image as ok / scratch / stain / chip."""
import glob
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, '.')
from sortbot.lane_dataset import CLASSES, VIEWS  # noqa: E402

DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(pattern):
    xs, ys, vs = [], [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        xs.append(z['imgs']); ys.append(z['labels']); vs.append(z['views'])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)


def to_tensor(x):
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


def build_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def augment(x):
    # x: float tensor (B,3,H,W) normalised. Brightness/contrast jitter, flips, small shifts.
    b = x.shape[0]
    gain = torch.empty(b, 1, 1, 1).uniform_(0.85, 1.15)
    bias = torch.empty(b, 1, 1, 1).uniform_(-0.15, 0.15)
    x = x * gain + bias
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[2])
    dx, dy = np.random.randint(-6, 7, 2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    return x


def evaluate(model, x, y, v, bs=256):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for k in range(0, len(x), bs):
            out = model(to_tensor(x[k:k + bs]).to(DEV))
            p = torch.softmax(out, 1).cpu().numpy()
            probs.append(p); preds.append(p.argmax(1))
    preds = np.concatenate(preds); probs = np.concatenate(probs)
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y, preds):
        conf[t, p] += 1
    is_def, pred_def = y > 0, preds > 0
    tp = int((is_def & pred_def).sum()); fn = int((is_def & ~pred_def).sum()); fp = int((~is_def & pred_def).sum())
    per_view = {VIEWS[k]: float((preds[v == k] == y[v == k]).mean()) for k in range(len(VIEWS))}
    return dict(acc=float((preds == y).mean()), defect_recall=tp / max(1, tp + fn),
                defect_precision=tp / max(1, tp + fp), confusion=conf.tolist(), per_view_acc=per_view)


def main(epochs=6, bs=128, lr=3e-4, exclude=None):
    xtr, ytr, vtr = load('data/lane_train_*.npz')
    if exclude:  # leave one defect type out of training to test an unseen kind of defect
        keep = ytr != CLASSES.index(exclude)
        xtr, ytr, vtr = xtr[keep], ytr[keep], vtr[keep]
    xte, yte, vte = load('data/lane_test_*.npz')
    print('train', xtr.shape, np.bincount(ytr, minlength=4).tolist(), 'test', xte.shape, np.bincount(yte, minlength=4).tolist())
    model = build_model().to(DEV)
    # Up-weight defect classes so the rare positives matter.
    counts = np.maximum(np.bincount(ytr, minlength=len(CLASSES)), 1).astype(float)
    w = torch.tensor((counts.sum() / (len(CLASSES) * counts)) ** 0.5, dtype=torch.float32).to(DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.time(); step = 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(xtr))
        tot = 0.0
        for k in range(0, len(perm) - bs + 1, bs):
            idx = perm[k:k + bs]
            xb = augment(to_tensor(xtr[idx])).to(DEV)
            yb = torch.from_numpy(ytr[idx]).to(DEV)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            tot += float(loss.detach())
        r = evaluate(model, xte, yte, vte)
        print(f'epoch {ep + 1} loss {tot / (len(perm) // bs):.4f} test acc {r["acc"]:.4f} '
              f'recall {r["defect_recall"]:.3f} precision {r["defect_precision"]:.3f} ({time.time() - t0:.0f}s)', flush=True)
    tag = f'_no_{exclude}' if exclude else ''
    torch.save(model.state_dict(), f'models/face_cls{tag}.pt')
    r = evaluate(model, xte, yte, vte)
    if exclude:
        c = CLASSES.index(exclude)
        r['unseen_type'] = exclude
        r['unseen_detect'] = float(np.array(r['confusion'])[c, 1:].sum() / np.array(r['confusion'])[c].sum())
    json.dump(r, open(f'results/classifier_test{tag}.json', 'w'), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True); os.makedirs('results', exist_ok=True)
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 6, exclude=sys.argv[2] if len(sys.argv) > 2 else None)
