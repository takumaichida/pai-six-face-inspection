"""Collect teacher demonstrations of gripping a moving part (for the imitation-learned grasp).

Each episode puts a part just before the pick zone (random side offset, angle, defects, belt stripe phase) and
the hand above it with a random offset (up to 6 cm behind, as when the camera estimate arrives late without
compensation). Half of the episodes use the other of the two hand yaws that grip the same faces (180 deg apart),
since on the line the arm reaches the part with either. The teacher (the program's grasp with the true part position) then grips the part. Small noise
is added to the executed motion (DART), while the recorded label is always the teacher's clean action, so the
student also learns to correct its own drift.

usage: collect_demos.py <episodes> <workers> -> data/demos_<k>.npz
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')


def worker(args):
    k, seeds = args
    import mujoco
    from sortbot.env import CLOSED, OPEN, grasp_rot
    from sortbot.imitation import Teacher, hand_state, teacher_goal, to_hand, wrist_image
    from sortbot.lane import HOME_Q, LX, PART_Z, PICK_Y, LaneCell
    from sortbot.lane_routine import Z_GRASP, LaneRobot, grasp_ok
    c = LaneCell(img_size=64, video_size=(64, 64))
    imgs, states, acts, closes, ep_id, step_id, ok_eps = [], [], [], [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        c.reset(seed=s, continuous=True, n_defect=0)
        c.set_defect(0, [c.random_defect(rng)] if rng.random() < 0.4 else None)
        d = c.data
        a = c.part_qadr[0]
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
        if rng.random() < 0.5:          # the other yaw that grips the same faces: on the line the arm uses both
            alt = yaw - np.pi if yaw > 0 else yaw + np.pi
            q7 = c.ik(hover(), grasp_rot(alt), HOME_Q, iters=300)[6]
            if c.jlo[6] + 0.2 < q7 < c.jhi[6] - 0.2:
                yaw = alt
        q = c.ik(hover(), grasp_rot(yaw), HOME_Q, iters=300)
        d.qpos[c.arm_qadr] = q; d.ctrl[:7] = q; d.qvel[:] = 0
        mujoco.mj_forward(c.model, d)
        r.q_cmd, r.t_yaw = q.copy(), yaw
        for _ in range(3):                                  # hover and follow the belt, as before the hand-over
            r.t_pos = hover(); r.tick()
        teach = Teacher(goal)
        n0 = len(imgs)
        for t in range(60):
            img, st = wrist_image(c), hand_state(c, r)
            nxt, close, done = teach.step(r.t_pos)
            tcp, _ = c.tcp_pose()
            imgs.append(img); states.append(st); acts.append(to_hand(r.t_yaw, nxt - tcp)); closes.append(close)
            ep_id.append(s); step_id.append(t)
            noise = np.r_[rng.normal(0, 0.003, 2), 0] if not close and rng.random() < 0.7 else 0.0
            r.t_pos = nxt + noise
            r.grip = CLOSED if close else OPEN
            r.tick()
            if done:
                break
        ok = grasp_ok(c, 0)
        ok_eps.append((s, ok, len(imgs) - n0))
    np.savez_compressed(f"data/demos_{k}.npz", img=np.array(imgs, np.uint8), state=np.array(states, np.float32),
                        act=np.array(acts, np.float32), close=np.array(closes, bool), ep=np.array(ep_id),
                        step=np.array(step_id), episodes=np.array(ok_eps))
    return ok_eps


if __name__ == "__main__":
    n, w = int(sys.argv[1]), int(sys.argv[2])
    seeds = np.arange(10000, 10000 + n)
    t0 = time.time()
    with Pool(w) as pool:
        res = pool.map(worker, [(k, seeds[k::w]) for k in range(w)])
    eps = [e for r in res for e in r]
    print(f"{len(eps)} episodes, teacher grip success {np.mean([e[1] for e in eps]):.3f}, "
          f"mean length {np.mean([e[2] for e in eps]):.1f} steps, {time.time() - t0:.0f} s")
