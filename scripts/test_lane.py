import sys, time
import numpy as np, imageio
sys.path.insert(0, '.')
from sortbot.lane import LaneCell, N_PARTS, LX, PART_Z
from sortbot.cell import FACE_NAMES
from sortbot.lane_routine import LaneProgram
c = LaneCell(video_size=(960, 540))
bad = c.reset(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 3, n_defect=3)
def oracle(cam, img, i):
    dirs = {'gateTop': [0,0,1], 'stBottom': [0,0,-1], 'stSide': [-1,0,0]}
    f = c.facing_face(i, np.array(dirs[cam], float))
    return 1.0 if c.defects_on_face(i, f) else 0.0
frames = []
c.frame_hook = lambda cell: frames.append(cell.render_video()) if int(round(cell.time / 0.1)) % 3 == 0 else None
prog = LaneProgram(c, oracle)
t0 = time.time()
res = prog.run()
print('wall', round(time.time() - t0, 1), 'sim', round(c.time, 1))
for r in res:
    i = r['part']; p = c.part_pos(i)
    where = 'NGbox' if c.in_ng_box(i) else ('belt' if abs(p[0]-LX) < 0.045 and abs(p[2]-PART_Z) < 0.01 else 'ELSE')
    print(i, 'gt', [d['type']+'@'+FACE_NAMES[d['face']] for d in c.defect_list[i]], '->', r['decision'], 'views', r['n_views'], 'cyc', round(r['t_done']-r['t_pick'],1), where, np.round(p,3))
print([(e['part'], e['view'], FACE_NAMES[e['face']]) for e in prog.log][:12])
imageio.mimwrite('out/test_lane.mp4', frames, fps=10, quality=7, macro_block_size=1)
imageio.imwrite('out/test_lane_end.png', c.render_video())
