"""Fill report.html with the numbers in results/*.json and print it to PDF with headless Chrome.

usage: python report/build.py <output.pdf>
"""
import json
import os
import subprocess
import sys

z = json.load(open("results/sync_eval.json"))
ev, lats = z["summary"], z["latencies"]
cl = json.load(open("results/classifier_test.json"))
po = json.load(open("results/pose_test.json"))
tm = json.load(open("results/timing.json"))
pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
v = {}
for tag, key in [("top", "A_top_only"), ("nr", "A_no_regrasp"), ("six", "A_six_face")]:
    s = ev[key]
    v[f"{tag}_recall"] = pct(s["recall"]); v[f"{tag}_hidden"] = pct(s["recall_hidden_faces"])
    v[f"{tag}_frr"] = pct(s["false_reject_rate"])
    v[f"{tag}_grip"] = f"{pct(s['grip_success'])}（{round(s['grip_success'] * s['grips_tried'])}/{s['grips_tried']}）"
    v[f"{tag}_cyc"] = f"{s['mean_cycle_s']:.1f}"
s = ev["A_six_face"]
v.update(drop_all=str(sum(ev[k].get("dropped", 0) for k in ("A_top_only", "A_no_regrasp", "A_six_face"))),
         six_parts=str(s["parts"]), six_ng=str(s["ng_parts"]), a_lots=str(s["parts"] // 8),
         pose_x=f"{po['err_x_mm']:.1f}", pose_y=f"{po['err_y_mm']:.1f}", pose_yaw=f"{po['err_yaw_deg']:.1f}",
         infer_ms=f"{s['infer_ms']:.0f}", cl_n=str(sum(map(sum, cl["confusion"]))), cl_acc=f"{cl['acc'] * 100:.1f}%",
         lat_max=f"{max(lats):.1f}", grip_on_max=pct(ev[f"B_{max(lats)}_1"]["grip_success"]),
         grip_off_max=pct(ev[f"B_{max(lats)}_0"]["grip_success"]), lat_parts=str(ev[f"B_{lats[0]}_1"]["grips_tried"]),
         lat_head="".join(f"<th>{lat:.1f}</th>" for lat in lats),
         lat_on="".join(f'<td class="num">{pct(ev[f"B_{lat}_1"]["grip_success"])}</td>' for lat in lats),
         lat_off="".join(f'<td class="num">{pct(ev[f"B_{lat}_0"]["grip_success"])}</td>' for lat in lats),
         tm_frames=str(tm["frames"]), tm_faces=str(tm["face_images"]), tm_rep=str(tm["repeats"]),
         tm_budget=f"{tm['budget_ms']:.0f}", tm_dt=f"{tm['interval_ms']['median']:.0f}", tm_drop=str(tm["dropped_frames"]),
         pose_med=f"{tm['pose']['median_ms']:.1f}", pose_p95=f"{tm['pose']['p95_ms']:.1f}",
         face_med=f"{tm['face']['median_ms']:.1f}", face_p95=f"{tm['face']['p95_ms']:.1f}")
pe = json.load(open("results/policy_eval.json"))["summary"]
pt = json.load(open("results/policy_train.json"))
v.update(il_eps=f"{pt['episodes']:,}", il_steps=f"{pt['train_steps'] + pt['val_steps']:,}",
         ho_eps=str(pe["handover"]["episodes"]), ho_succ=pct(pe["handover"]["success"]),
         ho_far=pct(pe["handover"]["success_far"]), pol_parts=str(pe[f"line_{lats[0]}_0"]["parts"]),
         pol_off="".join(f'<td class="num">{pct(pe[f"line_{lat}_0"]["grip_success"])}</td>' for lat in lats),
         pol_off_max=pct(pe[f"line_{max(lats)}_0"]["grip_success"]),
         prog_off="".join(f'<td class="num">{pct(pe.get(f"prog_{lat}_0", ev[f"B_{lat}_0"])["grip_success"])}</td>' for lat in lats),
         prog_off_max=pct(pe.get(f"prog_{max(lats)}_0", ev[f"B_{max(lats)}_0"])["grip_success"]),
         prog_parts=str(pe.get(f"prog_{lats[0]}_0", ev[f"B_{lats[0]}_0"]).get("parts", 24)))
pl = json.load(open("results/policy_line.json"))["summary"]
v.update({f"pl_{k}": str(val) for k, val in pl.items()})
html = open("report/report.html", encoding="utf-8").read()
for k, val in v.items():
    html = html.replace("{{" + k + "}}", val)
assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
open("report/report_filled.html", "w", encoding="utf-8").write(html)
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "report/report.pdf")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                "file://" + os.path.abspath("report/report_filled.html")], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", out, v)
