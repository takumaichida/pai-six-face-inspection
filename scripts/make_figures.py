"""Figures for the report and the video from results/lane_eval.json and results/classifier_test.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
from sortbot.lane import PITCH, V_LANE  # noqa: E402

for p in ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break

ev = json.load(open("results/lane_eval.json"))["summary"]
cl = json.load(open("results/classifier_test.json"))
C = {"top_only": "#9aa3ad", "no_regrasp": "#7bb3ef", "six_face": "#2f7fd8", "six_face_all": "#1d4f8f"}
NAME = {"top_only": "上面だけ（カメラ1）", "no_regrasp": "持ち上げのみ（持ち替えなし）",
        "six_face": "6面（持ち替えあり・不良で打ち切り）", "six_face_all": "6面（打ち切りなし）"}

# 1) Detection.
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
labels = ["不良品の検出率", "上面以外にだけ\n不良がある部品の検出率", "良品を誤ってNG\n（過検出率）"]
keys = ["top_only", "no_regrasp", "six_face"]
x = np.arange(3); w = 0.26
for k, m in enumerate(keys):
    v = [ev[m]["recall"], ev[m]["recall_hidden_faces"], ev[m]["false_reject_rate"]]
    bars = ax.bar(x + (k - 1) * w, v, w, label=NAME[m], color=C[m])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height() * 100:.0f}%",
                ha="center", fontsize=9)
ax.set_xticks(x, labels); ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
ax.legend(loc="upper right", fontsize=8.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
s = ev["six_face"]
ax.set_title(f"検査方式の比較（{s['parts']}個・うち不良{s['ng_parts']}個、上面以外だけの不良{s['hidden_ng_parts']}個）",
             fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_detection.png"); plt.close(fig)

# 2) Cycle time per part, and line throughput as arms are added.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150, gridspec_kw=dict(width_ratios=[1.2, 1]))
keys = ["top_only", "no_regrasp", "six_face", "six_face_all"]
vals = [ev[k]["mean_cycle_s"] for k in keys]
bars = a1.barh([NAME[k] for k in keys], vals, color=[C[k] for k in keys])
for b, v in zip(bars, vals):
    a1.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f} 秒", va="center", fontsize=9)
a1.set_xlabel("1個あたりのアーム作業時間（平均）"); a1.invert_yaxis()
a1.spines[["top", "right"]].set_visible(False); a1.set_xlim(0, max(vals) * 1.3)
lane_max = 60 * V_LANE / PITCH
one = ev["six_face"]["parts_per_min"]
arms = np.arange(1, 5)
est = np.minimum(arms * one, lane_max)
a2.plot(arms, arms * one, "--", color="#9aa3ad", label="アーム台数に比例")
a2.plot(arms, est, "o-", color=C["six_face"], label="推定（搬送能力で頭打ち）")
a2.axhline(lane_max, color="#d0533f", lw=1, ls=":")
a2.text(1.05, lane_max + 0.25, f"レーンの搬送能力 {lane_max:.1f} 個/分", color="#d0533f", fontsize=8.5)
for a, e in zip(arms, est):
    a2.text(a, e - 0.75, f"{e:.1f}", ha="center", fontsize=9)
a2.set_xticks(arms, [f"{a}台" for a in arms]); a2.set_ylabel("6面検査の処理能力（個/分）")
a2.set_ylim(0, max(arms * one) * 1.1); a2.legend(fontsize=8, frameon=False, loc="upper left")
a2.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/fig_cycle.png"); plt.close(fig)

# 3) Classifier confusion matrix.
conf = np.array(cl["confusion"]); names = ["良品", "打痕・傷", "油ジミ", "角欠け"]
fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
norm = conf / conf.sum(1, keepdims=True)
ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{conf[i, j]}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
ax.set_xticks(range(4), names); ax.set_yticks(range(4), names)
ax.set_xlabel("AIの判定"); ax.set_ylabel("正解")
ax.set_title(f"面画像の判定（テスト{conf.sum()}枚, 正解率{cl['acc'] * 100:.1f}%）", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_confusion.png"); plt.close(fig)
print("figures written; lane max", round(lane_max, 2), "one arm", round(one, 2))

# 4) Unseen kinds of defect: classifier that never saw the type vs the good-only detector.
an = json.load(open("results/anomaly_test.json"))
types = [("chip", "角欠け"), ("stain", "油ジミ"), ("scratch", "打痕・傷")]
cls_u = [json.load(open(f"results/classifier_test_no_{t}.json"))["unseen_detect"] for t, _ in types]
an_u = [an["per_type"][t] for t, _ in types]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
x = np.arange(len(types)); w = 0.36
b1 = ax.bar(x - w / 2, cls_u, w, color="#9aa3ad", label="分類器（その不良を学習していない）")
b2 = ax.bar(x + w / 2, an_u, w, color="#2f9e6a", label="異常検知（良品の画像だけで学習）")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height() * 100:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x, [f"{j}\n（学習から外した）" for _, j in types]); ax.set_ylim(0, 1.42)
ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"]); ax.set_ylabel("面画像での検出率")
ax.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"学習していない種類の不良を見つけられるか（異常検知の誤報 {an['false_alarm'] * 100:.1f}%/面）", fontsize=10.5)
fig.tight_layout(); fig.savefig("results/fig_unseen.png"); plt.close(fig)
print("unseen figure written")
