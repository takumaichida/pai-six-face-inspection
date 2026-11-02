"""Evaluate the imitation-learned grasp.

Part A (hand-over test): the same start states as the demonstrations, on new seeds: the hand hovers above a moving
part with a random offset (up to 6 cm behind) and the policy grips it alone.
Part B (on the line): the full never-stopping lane, camera 1 + pose AI + encoder bring the hand above the part,
then the policy grips it. The camera estimate arrives 0-1.5 s late and is NOT compensated, so the hand starts
behind the part by belt speed x latency; the policy has to catch up from the wrist image. The program's grasp
under the same conditions is in results/sync_eval.json (B_<lat>_0).

usage: evaluate_policy.py <handover episodes> <workers> [lots per latency] -> results/policy_eval.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')
LATENCIES = [0.0, 0.3, 0.6, 1.0, 1.5]


def handover(seeds):
    import mujoco
    import torch
    torch.set_num_threads(1)
    from sortbot.env import grasp_rot
    from sortbot.imitation import GraspPolicy, policy_grasp, teacher_goal
    from sortbot.lane import HOME_Q, LX, PART_Z, PICK_Y, LaneCell
    from sortbot.lane_routine import Z_GRASP, LaneRobot
    c = LaneCell(img_size=64, video_size=(64, 64))
    pol = GraspPolicy()
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        c.reset(seed=s, continuous=True, n_defect=0)
        c.set_defect(0, [c.random_defect(rng)] if rng.random() < 0.4 else None)
        d = c.data; a = c.part_qadr[0]
        yaw_p = rng.uniform(-np.pi / 4, np.pi / 4)
        d.qpos[a:a + 3] = [LX + rng.uniform(-0.015, 0.015), PICK_Y - rng.uniform(0.02, 0.07), PART_Z + 0.0005]
        d.qpos[a + 3:a + 7] = [np.cos(yaw_p / 2), 0, 0, np.sin(yaw_p / 2)]
        c.odo = rng.uniform(0, 0.07)
        mujoco.mj_forward(c.model, d)
        r = LaneRobot(c)
        p = c.part_pos(0)
        yaw = r.best_yaw([p[0], PICK_Y, Z_GRASP], c.part_yaw(0) + np.pi / 2) + rng.normal(0, np.radians(1.5))
        off = np.array([np.clip(rng.normal(0, 0.01), -0.025, 0.025), rng.uniform(-0.06, 0.02), rng.uniform(-0.02, 0.03)])
        goal = teacher_goal(c, 0)
        hover = lambda: goal() + [0, 0, 0.17] + off  # noqa: E731
        if rng.random() < 0.5:          # the other yaw that grips the same faces, as in the demonstrations
            alt = yaw - np.pi if yaw > 0 else yaw + np.pi
            q7 = c.ik(hover(), grasp_rot(alt), HOME_Q, iters=300)[6]
            if c.jlo[6] + 0.2 < q7 < c.jhi[6] - 0.2:
                yaw = alt
        q = c.ik(hover(), grasp_rot(yaw), HOME_Q, iters=300)
        d.qpos[c.arm_qadr] = q; d.ctrl[:7] = q; d.qvel[:] = 0
        mujoco.mj_forward(c.model, d)
        r.q_cmd, r.t_yaw = q.copy(), yaw
        for _ in range(3):
            r.t_pos = hover(); r.tick()
        prog = type("P", (), {})(); prog.c, prog.r = c, r
        t0 = c.time
        ok = policy_grasp(prog, 0, yaw, pol)
        out.append(dict(seed=int(s), ok=bool(ok), off_y_mm=float(off[1] * 1000), off_x_mm=float(off[0] * 1000),
                        steps=int(round((c.time - t0) / 0.1))))
    return out


def line(args):
    latency, comp, seed, grasp = args
    import torch
    torch.set_num_threads(1)
    from sortbot.imitation import GraspPolicy
    from sortbot.lane import LaneCell
    from sortbot.pose import PoseEstimator
    from sortbot.sync_routine import SyncProgram
    c = LaneCell(video_size=(64, 64))
    c.reset(seed=seed, n_defect=3, continuous=True)
    prog = SyncProgram(c, lambda cam, img, i: 0.0, PoseEstimator(), latency=latency, compensate=comp,
                       grasp_policy=GraspPolicy() if grasp == "policy" else None, grasp_only=True)
    res = prog.run()
    return [dict(latency=latency, comp=comp, seed=seed, grasp=grasp, part=r["part"], grip=bool(r.get("grip"))) for r in res]


if __name__ == "__main__":
    n, w = int(sys.argv[1]), int(sys.argv[2])
    lots = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    t0 = time.time()
    with Pool(w) as pool:
        seeds = np.arange(50000, 50000 + n)
        jobs_b = ([(lat, False, 300 + k, "policy") for lat in LATENCIES for k in range(lots)]
                  + [(0.3, True, 300 + k, "policy") for k in range(lots)]
                  + [(lat, False, 300 + k, "program") for lat in LATENCIES for k in range(lots)])
        rb = pool.map_async(line, jobs_b)
        ra = pool.map(handover, [seeds[k::w] for k in range(w)])
        rb = rb.get()
    a = [e for r in ra for e in r]
    b = [e for r in rb for e in r]
    summ = dict(handover=dict(episodes=len(a), success=float(np.mean([e["ok"] for e in a])),
                              success_far=float(np.mean([e["ok"] for e in a if e["off_y_mm"] < -30] or [np.nan])),
                              mean_steps=float(np.mean([e["steps"] for e in a]))))
    for lat in LATENCIES:
        for comp in (False, True):
            g = [e["grip"] for e in b if e["latency"] == lat and e["comp"] == comp and e["grasp"] == "policy"]
            if g:
                summ[f"line_{lat}_{int(comp)}"] = dict(parts=len(g), grip_success=float(np.mean(g)))
        g = [e["grip"] for e in b if e["latency"] == lat and e["grasp"] == "program"]
        if g:
            summ[f"prog_{lat}_0"] = dict(parts=len(g), grip_success=float(np.mean(g)))
    json.dump(dict(summary=summ, handover=a, line=b), open("results/policy_eval.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), f"\n{time.time() - t0:.0f} s")
