"""Run the conveyor line with the trained classifier and compare inspection methods.

  top_only     camera 1 only (the part stays on the lane)
  no_regrasp   camera 1 + lift: bottom and the two free side faces (the gripped faces stay hidden)
  six_face     camera 1 + lift + regrasp: all six faces, stop imaging once a defect is found
  six_face_all the same, but always image all six faces (no early stop)
  *_combined   six_face judged by classifier + good-only anomaly detector
  *_no_chip    the classifier never saw a chipped corner in training (an unseen kind of defect)

usage: evaluate_lane.py <runs> [workers]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')
METHODS = [("top_only", "top_only", True, "cls"), ("no_regrasp", "no_regrasp", True, "cls"),
           ("six_face", "six_face", True, "cls"), ("six_face_all", "six_face", False, "cls"),
           ("six_face_combined", "six_face", True, "combined"),
           ("six_face_no_chip", "six_face", True, "cls_no_chip"),
           ("six_face_no_chip_combined", "six_face", True, "combined_no_chip")]


def make_judge(kind):
    from sortbot.judge import CombinedJudge, FaceJudge
    w = "models/face_cls_no_chip.pt" if kind.endswith("no_chip") else "models/face_cls.pt"
    return CombinedJudge(w) if kind.startswith("combined") else FaceJudge(w)


def job(args):
    name, mode, early, kind, seed = args
    import torch
    torch.set_num_threads(1)
    from sortbot.lane import LaneCell, N_PARTS
    from sortbot.lane_routine import LaneProgram
    c = LaneCell(video_size=(320, 180))
    c.reset(seed=seed, n_defect=3)
    prog = LaneProgram(c, make_judge(kind), mode=mode, early_exit=early)
    t0 = c.time
    res = prog.run()
    out = []
    for r in res:
        i = r["part"]
        faces = sorted({d["face"] for d in c.defect_list[i]})
        types = sorted({d["type"] for d in c.defect_list[i]})
        out.append(dict(seed=seed, part=i, bad=bool(c.defect_list[i]), defect_faces=faces, defect_types=types,
                        decision=r["decision"], n_views=r["n_views"], cycle=r["t_done"] - r["t_pick"],
                        in_ng_box=bool(c.in_ng_box(i))))
    return name, seed, out, c.time - t0, N_PARTS


def summarise(rows, total_time, n_parts):
    bad = [r for r in rows if r["bad"]]
    hidden = [r for r in bad if 0 not in r["defect_faces"]]   # no defect on the top face
    chip = [r for r in bad if "chip" in r["defect_types"]]
    good = [r for r in rows if not r["bad"]]
    ng = lambda rs: sum(r["decision"] == "NG" for r in rs)  # noqa: E731
    cyc = [r["cycle"] for r in rows]
    return dict(parts=len(rows), ng_parts=len(bad), hidden_ng_parts=len(hidden),
                recall=ng(bad) / max(1, len(bad)), recall_hidden_faces=ng(hidden) / max(1, len(hidden)),
                false_reject_rate=ng(good) / max(1, len(good)), missed=len(bad) - ng(bad),
                chip_parts=len(chip), recall_chip=ng(chip) / max(1, len(chip)),
                mean_cycle_s=float(np.mean(cyc)), mean_cycle_good_s=float(np.mean([r["cycle"] for r in good])),
                mean_views=float(np.mean([r["n_views"] for r in rows])),
                ng_sorted_into_box=sum(r["in_ng_box"] for r in rows if r["decision"] == "NG") / max(1, ng(rows)),
                parts_per_min=60 * n_parts / total_time)


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(name, mode, early, kind, 100 + s) for name, mode, early, kind in METHODS for s in range(runs)]
    t0 = time.time()
    with Pool(workers) as p:
        done = p.map(job, jobs)
    per = {m[0]: dict(rows=[], time=0.0, parts=0) for m in METHODS}
    for name, seed, rows, sim_t, n in done:
        per[name]["rows"] += rows; per[name]["time"] += sim_t; per[name]["parts"] += n
    summary = {k: summarise(v["rows"], v["time"], v["parts"]) for k, v in per.items()}
    os.makedirs("results", exist_ok=True)
    json.dump(dict(summary=summary, rows={k: v["rows"] for k, v in per.items()}),
              open("results/lane_eval.json", "w"), indent=1, ensure_ascii=False)
    for k, s in summary.items():
        print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in s.items()}, ensure_ascii=False))
    print("wall", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
