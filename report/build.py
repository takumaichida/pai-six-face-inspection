"""Fill report.html with the numbers in results/*.json and print it to PDF with headless Chrome.

usage: python report/build.py <output.pdf>
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, ".")
from sortbot.lane import PITCH, V_LANE  # noqa: E402

ev = json.load(open("results/lane_eval.json"))["summary"]
cl = json.load(open("results/classifier_test.json"))
pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
v = {}
for tag, key in [("top", "top_only"), ("nr", "no_regrasp"), ("six", "six_face"), ("all", "six_face_all"),
                 ("comb", "six_face_combined"), ("nc", "six_face_no_chip"), ("ncc", "six_face_no_chip_combined")]:
    s = ev[key]
    v[f"{tag}_chip"] = pct(s["recall_chip"])
    v[f"{tag}_recall"] = pct(s["recall"]); v[f"{tag}_hidden"] = pct(s["recall_hidden_faces"])
    v[f"{tag}_frr"] = pct(s["false_reject_rate"]); v[f"{tag}_missed"] = str(s["missed"])
    v[f"{tag}_cyc"] = f"{s['mean_cycle_s']:.1f}"; v[f"{tag}_ppm"] = f"{s['parts_per_min']:.1f}"
lane_max = 60 * V_LANE / PITCH
one = ev["six_face"]["parts_per_min"]
n = 1
while n * one < lane_max * 0.97:  # within 3% of the lane limit counts as reaching it
    n += 1
v.update(lane_max=f"{lane_max:.1f}", arms_sat=str(n), arm2=f"{min(2 * one, lane_max):.1f}",
         save_s=f"{ev['six_face_all']['mean_cycle_s'] - ev['six_face']['mean_cycle_s']:.1f}",
         cl_n=str(sum(map(sum, cl["confusion"]))), cl_acc=f"{cl['acc'] * 100:.1f}%",
         ub95=f"{(1 - 0.05 ** (1 / ev['six_face']['ng_parts'])) * 100:.0f}%")
an = json.load(open("results/anomaly_test.json"))
for c in ["chip", "stain", "scratch"]:
    v[f"u_{c}_cls"] = pct(json.load(open(f"results/classifier_test_no_{c}.json"))["unseen_detect"])
    v[f"u_{c}_an"] = pct(an["per_type"][c])
v["an_fa"] = f"{an['false_alarm'] * 100:.1f}%"
v["chip_parts"] = str(ev["six_face"]["chip_parts"])
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
