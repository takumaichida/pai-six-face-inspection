"""Figures for the report and the video from results/sync_eval.json and classifier_test.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

for p in ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break

z = json.load(open("results/sync_eval.json"))
ev, lats = z["summary"], z["latencies"]
cl = json.load(open("results/classifier_test.json"))
C = {"A_top_only": "#9aa3ad", "A_no_regrasp": "#7bb3ef", "A_six_face": "#2f7fd8"}
NAME = {"A_top_only": "上面だけ（カメラ1）", "A_no_regrasp": "持ち上げのみ（持ち替えなし）", "A_six_face": "6面（持ち替えあり）"}

# 1) Detection by inspection method (all on the never-stopping lane with camera-synchronised picking).
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
labels = ["不良品の検出率", "上面以外にだけ\n不良がある部品の検出率", "良品を誤ってNG\n（過検出率）"]
keys = ["A_top_only", "A_no_regrasp", "A_six_face"]
x = np.arange(3); w = 0.26
for k, m in enumerate(keys):
    v = [ev[m]["recall"], ev[m]["recall_hidden_faces"], ev[m]["false_reject_rate"]]
    bars = ax.bar(x + (k - 1) * w, v, w, label=NAME[m], color=C[m])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height() * 100:.0f}%",
                ha="center", fontsize=9)
ax.set_xticks(x, labels); ax.set_ylim(0, 1.18); ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
ax.legend(loc="upper right", fontsize=8.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
s = ev["A_six_face"]
ax.set_title(f"検査方式の比較（{s['parts']}個・うち不良{s['ng_parts']}個、上面以外だけの不良{s['hidden_ng_parts']}個）",
             fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_detection.png"); plt.close(fig)

# 2) Grip success vs latency of the camera estimate, with and without compensation.
fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
on = [ev[f"B_{lat}_1"]["grip_success"] for lat in lats]
off = [ev[f"B_{lat}_0"]["grip_success"] for lat in lats]
ax.plot(lats, on, "o-", color="#2f7fd8", lw=2, label="遅れを補正（撮影時刻の回転計の値と組み合わせる）")
ax.plot(lats, off, "s--", color="#d0533f", lw=2, label="補正なし（届いた時刻の値と組み合わせる）")
for xx, a, b in zip(lats, on, off):
    ax.text(xx, a + 0.04, f"{a * 100:.0f}%", ha="center", fontsize=9, color="#2f7fd8")
    ax.text(xx, b - 0.09, f"{b * 100:.0f}%", ha="center", fontsize=9, color="#d0533f")
ax.set_xlabel("カメラの推定がアームに届くまでの遅れ（秒）"); ax.set_ylabel("流れる部品をつかめた割合")
ax.set_ylim(-0.12, 1.15); ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
ax.set_xticks(lats); ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("遅れの補正と、止まらないレーンでのつかみ成功率", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_latency.png"); plt.close(fig)

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
ax.set_title(f"面画像の判定（テスト{conf.sum()}枚）", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_confusion.png"); plt.close(fig)

# 4) Imitation-learned grasp vs the program, both without latency compensation.
pe = json.load(open("results/policy_eval.json"))["summary"]
fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
prog = [ev[f"B_{lat}_0"]["grip_success"] for lat in lats]
pol = [pe[f"line_{lat}_0"]["grip_success"] for lat in lats]
ax.plot(lats, pol, "o-", color="#e08a1e", lw=2, label="模倣学習した方策（手先カメラで追いかける）")
ax.plot(lats, prog, "s--", color="#d0533f", lw=2, label="プログラム（届いた推定のまま狙う）")
for xx, a, b in zip(lats, pol, prog):
    ax.text(xx, a + 0.04, f"{a * 100:.0f}%", ha="center", fontsize=9, color="#b86a0c")
    ax.text(xx, b - 0.09, f"{b * 100:.0f}%", ha="center", fontsize=9, color="#d0533f")
ax.set_xlabel("カメラ1の推定がアームに届くまでの遅れ（秒）"); ax.set_ylabel("流れる部品をつかめた割合")
ax.set_ylim(-0.12, 1.15); ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
ax.set_xticks(lats); ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("遅れを補正しない条件での、つかみ成功率", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig_policy.png"); plt.close(fig)
print("figures written")
