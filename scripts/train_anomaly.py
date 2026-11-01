"""Build the good-only detector from good faces only and test it.

Memory = ok images of the synthetic dataset + images of good parts collected from line runs
(data/line_good.npz, from scripts/gen_line_good.py). The threshold is set on good line images not in the memory.

Writes models/anomaly.pt and results/anomaly_test.json (per-view AUROC, detection rate per defect type,
false alarm rate on good faces at the calibrated threshold).
"""
import glob
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from sortbot.anomaly import AnomalyDetector  # noqa: E402
from sortbot.lane_dataset import CLASSES, VIEWS  # noqa: E402

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(pattern):
    xs, ys, vs = [], [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        xs.append(z['imgs']); ys.append(z['labels']); vs.append(z['views'])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)


def auroc(neg, pos):
    s = np.concatenate([neg, pos]); r = s.argsort().argsort() + 1
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main(n_fit=1200, n_cal=300):
    xtr, ytr, vtr = load('data/lane_train_*.npz')
    xte, yte, vte = load('data/lane_test_*.npz')
    det = AnomalyDetector(DEV)
    rng = np.random.default_rng(0)
    t0 = time.time()
    out = dict(per_view={}, per_type={}, fit_images_per_view=n_fit)
    lg = np.load('data/line_good.npz')
    lg_fit = lg['lot'] < 28                       # lots 28.. are held out for the threshold
    scores = np.zeros(len(xte))
    for k, view in enumerate(VIEWS):
        good = rng.permutation(np.where((ytr == 0) & (vtr == k))[0])
        fit = np.concatenate([xtr[good[:n_fit]], lg['imgs'][lg_fit & (lg['views'] == k)]])
        cal = np.concatenate([xtr[good[n_fit:n_fit + n_cal]], lg['imgs'][~lg_fit & (lg['views'] == k)]])
        det.fit(view, fit)
        det.calibrate(view, cal)
        m = vte == k
        scores[m] = det.score(view, xte[m])
        s, y = scores[m], yte[m]
        out["per_view"][view] = dict(auroc=auroc(s[y == 0], s[y > 0]), threshold=det.th[view],
                                     detect=float((s[y > 0] > det.th[view]).mean()),
                                     false_alarm=float((s[y == 0] > det.th[view]).mean()))
        print(view, out["per_view"][view], round(time.time() - t0), 's', flush=True)
    th = np.array([det.th[VIEWS[k]] for k in vte])
    pred = scores > th
    for c in range(1, len(CLASSES)):
        out["per_type"][CLASSES[c]] = float(pred[yte == c].mean())
    out["detect"] = float(pred[yte > 0].mean()); out["false_alarm"] = float(pred[yte == 0].mean())
    out["auroc"] = auroc(scores[yte == 0] / th[yte == 0], scores[yte > 0] / th[yte > 0])
    det.save('models/anomaly.pt')
    json.dump(out, open('results/anomaly_test.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
