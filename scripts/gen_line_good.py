"""Collect face images of good parts from running lots on the line (as a factory would during start-up).

usage: gen_line_good.py <lots> [workers]  -> data/line_good.npz (imgs, views)
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')


def lot(seed):
    from sortbot.lane import LaneCell
    from sortbot.lane_dataset import VIEWS
    from sortbot.lane_routine import LaneProgram
    c = LaneCell(video_size=(320, 180))
    c.reset(seed=seed, n_defect=0)
    imgs, views = [], []

    def keep(cam, img, part):
        imgs.append(img); views.append(VIEWS.index(cam))
        return 0.0
    LaneProgram(c, keep).run()
    return np.stack(imgs), np.array(views)


if __name__ == '__main__':
    lots, workers = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    with Pool(workers) as p:
        out = p.map(lot, [1000 + k for k in range(lots)])
    imgs = np.concatenate([o[0] for o in out]); views = np.concatenate([o[1] for o in out])
    lot_id = np.concatenate([np.full(len(o[1]), k) for k, o in enumerate(out)])
    np.savez_compressed('data/line_good.npz', imgs=imgs, views=views, lot=lot_id)
    print('images', len(imgs), np.bincount(views).tolist(), round(time.time() - t0), 's')
