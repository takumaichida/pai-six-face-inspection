"""Evaluate the never-stopping lane with camera-synchronised picking.

Part A: inspection methods (top only / lift without regrasp / six faces) at 0.3 s latency with compensation.
Part B: grip success when the camera estimate arrives late (0 to 1.5 s), with and without latency compensation.

usage: evaluate_sync.py <lots per setting> [workers]  -> results/sync_eval.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')
LATENCIES = [0.0, 0.3, 0.6, 1.0, 1.5]


def job(args):
    tag, mode, latency, comp, seed = args
    import torch
    torch.set_num_threads(1)
    from sortbot.judge import FaceJudge
    from sortbot.lane import LaneCell
    from sortbot.pose import PoseEstimator
    from sortbot.sync_routine import SyncProgram
    c = LaneCell(video_size=(320, 180))
    c.reset(seed=seed, n_defect=3, continuous=True)
    prog = SyncProgram(c, FaceJudge(), PoseEstimator(), latency=latency, compensate=comp, mode=mode)
    res = prog.run()
    rows = []
    for r in res:
        i = r["part"]
        tr = prog.tracks[i] if i < len(prog.tracks) else {}
        t = tr.get("truth", {})
        rows.append(dict(seed=seed, part=i, bad=bool(c.defect_list[i]), top_defect=any(d["face"] == 0 for d in c.defect_list[i]),
                         decision=r["decision"], grip=r.get("grip"), n_views=r["n_views"], cycle=r["t_done"] - r["t_pick"],
                         in_ng_box=bool(c.in_ng_box(i)),
                         err_x_mm=(tr["x"] - t["x"]) * 1000 if t else None,
                         err_yaw_deg=float(np.degrees((tr["yaw"] - t["yaw"] + np.pi / 4) % (np.pi / 2) - np.pi / 4)) if t else None))
    return tag, rows, c.time, float(np.mean(prog.infer_ms)) if prog.infer_ms else None


def summarise(rows, sim_time):
    bad = [r for r in rows if r["bad"]]; good = [r for r in rows if not r["bad"]]
    hidden = [r for r in bad if not r["top_defect"]]
    tried = [r for r in rows if r["grip"] is not None]
    ng = lambda rs: sum(r["decision"] == "NG" for r in rs)  # noqa: E731
    ex = [abs(r["err_x_mm"]) for r in rows if r["err_x_mm"] is not None]
    ey = [abs(r["err_yaw_deg"]) for r in rows if r["err_yaw_deg"] is not None]
    return dict(parts=len(rows), ng_parts=len(bad), hidden_ng_parts=len(hidden),
                recall=ng(bad) / max(1, len(bad)), recall_hidden_faces=ng(hidden) / max(1, len(hidden)),
                false_reject_rate=ng(good) / max(1, len(good)), missed_defects=len(bad) - ng(bad),
                grips_tried=len(tried), grip_success=sum(bool(r["grip"]) for r in tried) / max(1, len(tried)),
                escaped_uninspected=sum(r["decision"] == "MISSED" for r in rows),
                mean_cycle_s=float(np.mean([r["cycle"] for r in rows if r["grip"]] or [0])),
                pose_err_x_mm=float(np.mean(ex)) if ex else None, pose_err_yaw_deg=float(np.mean(ey)) if ey else None,
                parts_per_min=60 * len(rows) / sim_time)


def main():
    lots = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(f"A_{m}", m, 0.3, True, 200 + s) for m in ("top_only", "no_regrasp", "six_face") for s in range(lots)]
    jobs += [(f"B_{lat}_{int(cp)}", "six_face", lat, cp, 300 + s) for lat in LATENCIES for cp in (True, False)
             for s in range(max(2, lots // 2))]
    t0 = time.time()
    with Pool(workers) as p:
        done = p.map(job, jobs)
    groups = {}
    for tag, rows, sim_t, inf in done:
        g = groups.setdefault(tag, dict(rows=[], time=0.0, infer=[]))
        g["rows"] += rows; g["time"] += sim_t
        if inf is not None:
            g["infer"].append(inf)
    out = {tag: dict(summarise(g["rows"], g["time"]), infer_ms=float(np.mean(g["infer"]))) for tag, g in groups.items()}
    json.dump(dict(summary=out, latencies=LATENCIES), open("results/sync_eval.json", "w"), indent=1, ensure_ascii=False)
    for k in sorted(out):
        s = out[k]
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in s.items()})
    print("wall", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
