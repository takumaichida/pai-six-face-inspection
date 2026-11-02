"""End-to-end check of the imitation-learned grasp: full lots on the never-stopping lane with the face judge,
six-face inspection and sorting, where every grip is made by the learned policy (0.3 s latency, compensated).

usage: check_policy_line.py <lots> [workers] -> results/policy_line.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')


def job(seed):
    import torch
    torch.set_num_threads(1)
    from sortbot.imitation import GraspPolicy
    from sortbot.judge import FaceJudge
    from sortbot.lane import LaneCell
    from sortbot.pose import PoseEstimator
    from sortbot.sync_routine import SyncProgram
    c = LaneCell(video_size=(64, 64))
    c.reset(seed=seed, n_defect=3, continuous=True)
    prog = SyncProgram(c, FaceJudge(), PoseEstimator(), latency=0.3, compensate=True, grasp_policy=GraspPolicy())
    res = prog.run()
    return [dict(seed=seed, part=r["part"], decision=r["decision"], defect=bool(c.is_bad(r["part"])),
                 grip=bool(r.get("grip"))) for r in res]


if __name__ == "__main__":
    lots = int(sys.argv[1])
    t0 = time.time()
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else lots) as p:
        rows = [e for r in p.map(job, [400 + k for k in range(lots)]) for e in r]
    bad = [e for e in rows if e["defect"]]
    good = [e for e in rows if not e["defect"]]
    summ = dict(lots=lots, parts=len(rows), grips=sum(e["grip"] for e in rows), dropped=sum(e["decision"] == "DROPPED" for e in rows),
                ng_parts=len(bad), ng_caught=sum(e["decision"] == "NG" for e in bad),
                good_parts=len(good), good_ok=sum(e["decision"] == "OK" for e in good))
    json.dump(dict(summary=summ, rows=rows), open("results/policy_line.json", "w"), indent=1)
    print(json.dumps(summ), f"{time.time() - t0:.0f} s")
