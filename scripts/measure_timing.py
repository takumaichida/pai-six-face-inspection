"""Timing checks in the style of lecture 3: does each AI step fit the camera's frame budget, and are the stamps sound?

1. One lot runs on the never-stopping lane; every camera-1 frame the pose network saw and every face image the
   classifier judged are kept, with the capture time and the encoder reading of each camera-1 frame.
2. The pose network and the face classifier are timed with time.perf_counter: a warm-up first, then every image
   x 3 repeats in shuffled order; median and p95. Rendering and the result check stay outside the timed region,
   and repeats of the same image must give the same output.
3. The stamps: the interval between camera-1 frames (a dropped frame shows up as a gap), the encoder step per frame
   (must equal belt speed x interval), and that the kept images are copies, not views of one render buffer.

Times are measured on this PC's CPU, not on a robot controller.

usage: measure_timing.py [seed] -> results/timing.json, results/fig_timing.png
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '.')
from sortbot.judge import FaceJudge  # noqa: E402
from sortbot.lane import V_LANE, LaneCell  # noqa: E402
from sortbot.pose import PoseEstimator  # noqa: E402
from sortbot.sync_routine import SyncProgram  # noqa: E402

FRAME_HZ = 10
BUDGET_MS = 1000 / FRAME_HZ         # one camera-1 frame interval
REPEATS = 3


def bench(fn, imgs, repeats=REPEATS, seed=7):
    """Time fn on every image `repeats` times in shuffled order (after a warm-up). Returns times [ms]."""
    fn(imgs[0])
    jobs = [(i, k) for i in range(len(imgs)) for k in range(repeats)]
    np.random.default_rng(seed).shuffle(jobs)
    ms, first = [], {}
    for i, _ in jobs:
        t0 = time.perf_counter()
        out = fn(imgs[i])
        ms.append((time.perf_counter() - t0) * 1000)
        out = np.asarray(out, float)                    # checked outside the timed region
        if i in first:
            assert np.allclose(first[i], out, atol=1e-5), "the same image gave a different result"
        first[i] = out
    return np.array(ms)


def stats(ms):
    return dict(n=int(len(ms)), median_ms=float(np.median(ms)), p95_ms=float(np.quantile(ms, 0.95)),
                max_ms=float(ms.max()), share_over_budget=float((ms > BUDGET_MS).mean()))


def main(seed):
    c = LaneCell(video_size=(320, 180))
    c.reset(seed=seed, n_defect=3, continuous=True)
    judge, pose = FaceJudge(), PoseEstimator()
    prog = SyncProgram(c, judge, pose, latency=0.3)
    cam1, faces, cur = [], [], dict(lst=None, n=0, track=-1)
    on_frame = prog.on_frame

    def keep_frame(cell):                               # every frame the pose network saw
        on_frame(cell)
        fr = prog.frames
        if fr and fr is not cur["lst"]:
            cur.update(lst=fr, n=0, track=cur["track"] + 1)
        if fr and len(fr) > cur["n"]:
            t, dx, dy, yaw, img, frac, odo = fr[-1]
            cam1.append(dict(track=cur["track"], t=t, odo=odo, img=img))
            cur["n"] = len(fr)
    c.sensor_hook = keep_frame
    prog.capture_listeners.append(lambda i, cam, img, label, p: faces.append(img))
    t0 = time.time()
    res = prog.run()
    print(f"lot ran in {time.time() - t0:.0f} s: {len(cam1)} camera-1 frames, {len(faces)} face images,",
          sum(bool(r.get('grip')) for r in res), "grips")

    # 2) processing time against the frame budget
    pose_ms = bench(lambda img: pose(img), [f["img"] for f in cam1])
    face_ms = bench(lambda img: judge.probs(img), faces)
    p, f = stats(pose_ms), stats(face_ms)
    print(f"pose  median {p['median_ms']:.1f} ms  p95 {p['p95_ms']:.1f} ms  (frame budget {BUDGET_MS:.0f} ms)")
    print(f"face  median {f['median_ms']:.1f} ms  p95 {f['p95_ms']:.1f} ms")

    # 3) capture stamps: interval, dropped frames, encoder step, copies
    dts, enc_err, dropped = [], [], 0
    for k in sorted({x["track"] for x in cam1}):
        fr = [x for x in cam1 if x["track"] == k]
        t = np.array([x["t"] for x in fr]); o = np.array([x["odo"] for x in fr])
        dt = np.diff(t)
        dts += dt.tolist()
        dropped += int(np.sum(np.round(dt * FRAME_HZ) - 1))
        enc_err += (np.abs(np.diff(o) - V_LANE * dt) * 1000).tolist()
    dts = np.array(dts)
    imgs = [x["img"] for x in cam1]
    shared = sum(np.shares_memory(imgs[i], imgs[i + 1]) for i in range(len(imgs) - 1))

    out = dict(seed=seed, torch_threads=torch.get_num_threads(), frame_hz=FRAME_HZ, budget_ms=BUDGET_MS,
               repeats=REPEATS, pose=p, face=f, frames=len(cam1), face_images=len(faces),
               interval_ms=dict(median=float(np.median(dts) * 1000), min=float(dts.min() * 1000),
                                max=float(dts.max() * 1000)),
               dropped_frames=dropped, encoder_step_err_mm_max=float(max(enc_err)), shared_buffers=int(shared),
               grips=int(sum(bool(r.get('grip')) for r in res)), parts=len(res))
    json.dump(out, open("results/timing.json", "w"), indent=1)

    # self-checks
    assert dropped == 0, f"{dropped} camera-1 frames are missing"
    assert out["encoder_step_err_mm_max"] < 1e-6, "encoder step does not match belt speed x interval"
    assert shared == 0, "kept frames share one buffer (views of the render output)"
    assert p["p95_ms"] < BUDGET_MS, f"pose p95 {p['p95_ms']:.0f} ms exceeds the {BUDGET_MS:.0f} ms frame interval"
    print("self-check: no dropped frames, encoder matches the stamps, frames are copies, pose fits the budget")
    figure(pose_ms, face_ms, dts)


def figure(pose_ms, face_ms, dts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    names = ["位置推定AI\n（カメラ1の毎フレーム）", "面の判定AI\n（検査の1枚ごと）"]
    med = [np.median(pose_ms), np.median(face_ms)]
    p95 = [np.quantile(pose_ms, 0.95), np.quantile(face_ms, 0.95)]
    x = np.arange(2)
    ax.bar(x - 0.18, med, 0.36, label="中央値", color="#2f7fd8")
    ax.bar(x + 0.18, p95, 0.36, label="p95", color="#9ec5f0")
    for xx, a, b in zip(x, med, p95):
        ax.text(xx - 0.18, a + 2, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(xx + 0.18, b + 2, f"{b:.0f}", ha="center", fontsize=9)
    ax.axhline(BUDGET_MS, color="#d0533f", ls="--", lw=1.4, label=f"カメラ1の撮影間隔 {BUDGET_MS:.0f} ms（10 Hz）")
    ax.set_xticks(x, names); ax.set_ylabel("処理時間 [ms]"); ax.set_ylim(0, BUDGET_MS * 1.3)
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("1枚あたりの処理時間（このPCのCPU）", fontsize=10)
    ax = axes[1]
    ax.hist(dts * 1000, bins=np.arange(50, 251, 10), color="#2f7fd8")
    ax.set_xlabel("カメラ1の撮影時刻の間隔 [ms]"); ax.set_ylabel("フレーム数")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("撮影時刻の間隔（抜けがあると200 ms以上に出る）", fontsize=10)
    fig.tight_layout(); fig.savefig("results/fig_timing.png"); plt.close(fig)
    print("wrote results/timing.json, results/fig_timing.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
