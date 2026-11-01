"""Run one lot on the never-stopping lane with camera-based picking (ground-truth judge) and print what happened.

usage: test_sync.py [seed] [latency_s] [compensate 1/0] [video 1/0]
"""
import sys
import time

import numpy as np

sys.path.insert(0, '.')
from sortbot.cell import FACE_NAMES  # noqa: E402
from sortbot.lane import LX, PART_Z, LaneCell  # noqa: E402
from sortbot.pose import PoseEstimator  # noqa: E402
from sortbot.sync_routine import SyncProgram  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
lat = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
comp = (sys.argv[3] != '0') if len(sys.argv) > 3 else True
video = len(sys.argv) > 4 and sys.argv[4] == '1'
c = LaneCell(video_size=(960, 540))
c.reset(seed=seed, n_defect=3, continuous=True)
dirs = {'gateTop': [0, 0, 1], 'stBottom': [0, 0, -1], 'stSide': [-1, 0, 0]}
oracle = lambda cam, img, i: 1.0 if c.defects_on_face(i, c.facing_face(i, np.array(dirs[cam], float))) else 0.0  # noqa: E731
prog = SyncProgram(c, oracle, PoseEstimator(), latency=lat, compensate=comp)
prog.debug = True
frames = []
if video:
    import imageio
    c.frame_hook = lambda cell: frames.append(cell.render_video()) if int(round(cell.time / 0.1)) % 3 == 0 else None
t0 = time.time()
res = prog.run()
print('wall', round(time.time() - t0, 1), 'sim', round(c.time, 1), 'infer ms', round(np.mean(prog.infer_ms), 1))
for r, tr in zip(res, prog.tracks):
    i = r['part']; p = c.part_pos(i)
    where = 'NGbox' if c.in_ng_box(i) else ('belt' if abs(p[0] - LX) < 0.045 and abs(p[2] - PART_Z) < 0.01 else 'ELSE')
    tt = tr.get('truth', {})
    print(i, [d['type'] + '@' + FACE_NAMES[d['face']] for d in c.defect_list[i]], '->', r['decision'], 'views', r['n_views'],
          'grip', r.get('grip'), where, 'err_x_mm', round((tr['x'] - tt.get('x', np.nan)) * 1000, 1),
          'yaw_err_deg', round(np.degrees(((tr['yaw'] - tt.get('yaw', np.nan)) + np.pi / 4) % (np.pi / 2) - np.pi / 4), 1),
          'v', round(tr['v'], 4))
print([(e['part'], e['view'], FACE_NAMES[e['face']]) for e in prog.log if e['part'] in (1, 2)])
print([np.round(c.part_pos(i), 3).tolist() for i in (1, 2)])
if video:
    imageio.mimwrite('out/test_sync.mp4', frames, fps=10, quality=7, macro_block_size=1)
