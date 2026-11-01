"""Record the conveyor line with the trained judge and Japanese overlays.

The simulation runs at 10 control steps per second; frames are written at 20 fps, so the clip plays at 2x.
Writes out/lane_demo.mp4, out/lane_demo_events.json and a still out/lane_demo_last.png.
"""
import json
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
from sortbot.cell import FACE_NAMES  # noqa: E402
from sortbot.judge import FaceJudge  # noqa: E402
from sortbot.lane import N_PARTS, LaneCell  # noqa: E402
from sortbot.lane_routine import LaneProgram  # noqa: E402

W, H = 1280, 720
FONT_PATHS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]
FONT = next(p for p in FONT_PATHS if os.path.exists(p))
CAMS = [("gateTop", "カメラ1 上面"), ("stBottom", "カメラ2 底面"), ("stSide", "カメラ3 側面")]
FACE_JA = {"top": "上面", "bottom": "底面", "+x": "側面1", "-x": "側面2", "+y": "側面3", "-y": "側面4"}
CLASSES_JA = ["良", "傷", "油ジミ", "欠け"]
TYPE_JA = {"scratch": "打痕・傷", "stain": "油ジミ", "chip": "角欠け"}


def font(sz):
    return ImageFont.truetype(FONT, sz)


class Recorder:
    def __init__(self, cell, prog, judge, path):
        self.c, self.prog, self.judge = cell, prog, judge
        self.writer = imageio.get_writer(path, fps=20, quality=8, macro_block_size=1)
        self.thumbs = {}
        self.part = None
        self.n_done = self.n_ng = self.n_back = 0
        self.frame = 0
        self.f_title, self.f_big, self.f_mid, self.f_small = font(26), font(24), font(17), font(14)

    def on_capture(self, i, cam, img, label, p):
        self.thumbs[cam] = (img, self.judge.last["probs"], self.judge.last.get("by", "classifier"), p)

    def face_status(self):
        out = {}
        for e in self.prog.log:
            if e["part"] == self.part:
                out[FACE_NAMES[e["face"]]] = e["p"]
        return out

    def compose(self):
        c = self.c
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        canvas.paste(Image.fromarray(c.render_video()), (0, 56))
        d = ImageDraw.Draw(canvas)
        d.text((16, 13), "ロボットによる6面外観検査ライン：アルミ切削ハウジングの出荷前全数検査（MuJoCo）",
               font=self.f_title, fill=(240, 240, 240))
        d.rounded_rectangle([860, 66, 950, 96], 6, fill=(200, 60, 50))
        d.text((870, 70), "2倍速", font=self.f_mid, fill=(255, 255, 255))
        d.rectangle([0, 596, 960, 720], fill=(28, 31, 38))
        d.text((18, 606), self.prog.status or "", font=self.f_big, fill=(255, 220, 120))
        d.text((18, 650), f"検査済み {self.n_done}/{N_PARTS}　NG箱へ {self.n_ng}　良品をレーンへ戻す {self.n_back}",
               font=self.f_mid, fill=(200, 205, 215))
        d.text((18, 682), f"シミュレーション時間 {c.time:5.1f} 秒", font=self.f_small, fill=(150, 155, 165))
        # Camera thumbnails with the AI verdict.
        d.text((972, 62), "検査カメラ画像とAI判定", font=self.f_mid, fill=(230, 230, 230))
        for k, (cam, name) in enumerate(CAMS):
            x0, y0 = 970 + k * 102, 90
            d.rectangle([x0, y0, x0 + 98, y0 + 98], outline=(70, 70, 80), width=2)
            if cam in self.thumbs:
                img, p, by, pd = self.thumbs[cam]
                canvas.paste(Image.fromarray(img).resize((94, 94)), (x0 + 2, y0 + 2))
                col = (230, 60, 50) if pd > 0.5 else (60, 200, 90)
                d.rectangle([x0, y0, x0 + 98, y0 + 98], outline=col, width=4)
                lab = ("NG 異常" if by == "anomaly" else f"NG {CLASSES_JA[int(np.argmax(p[1:]) + 1)]}") if pd > 0.5 else "OK"
                d.text((x0 + 2, y0 + 101), f"{lab} {pd:.2f}", font=self.f_small, fill=col)
            d.text((x0 + 2, y0 + 120), name, font=self.f_small, fill=(170, 175, 185))
        # Which of the six faces of the current part have been seen.
        x0, y0 = 970, 236
        d.text((x0, y0), f"部品{(self.part or 0) + 1}の6面", font=self.f_mid, fill=(230, 230, 230))
        st = self.face_status()
        for k, f in enumerate(FACE_NAMES):
            bx, by = x0 + (k % 3) * 102, y0 + 28 + (k // 3) * 36
            col = ((200, 55, 45) if st[f] > 0.5 else (45, 150, 75)) if f in st else (55, 58, 66)
            d.rounded_rectangle([bx, by, bx + 96, by + 30], 5, fill=col)
            d.text((bx + 26, by + 6), FACE_JA[f], font=self.f_small, fill=(245, 245, 245))
        d.text((x0, y0 + 102), "緑=良　赤=不良　灰=未撮影", font=self.f_small, fill=(150, 155, 165))
        # Close-up of the inspection station.
        d.text((x0, 368), "検査台のアップ", font=self.f_mid, fill=(230, 230, 230))
        canvas.paste(Image.fromarray(c.render_video("video_station")).resize((302, 170)), (x0, 394))
        d.text((x0, 572), "AI = 面画像の分類器（ResNet-18）", font=self.f_small, fill=(150, 155, 165))
        d.text((x0, 592), "数値 = 不良の確率", font=self.f_small, fill=(150, 155, 165))
        d.text((x0, 612), "0.5を超えた面が1つでもあれば NG", font=self.f_small, fill=(150, 155, 165))
        return np.asarray(canvas)

    def hook(self, cell):
        self.writer.append_data(self.compose())
        self.frame += 1


def main(seed=21, path="out/lane_demo.mp4"):
    c = LaneCell(video_size=(960, 540))
    c.reset(seed=seed, n_defect=0)
    # Showcase: a stain on the top face (camera 1), a scratch on the bottom (camera 2),
    # and a chipped corner on a face the fingers cover at first (found only after the regrasp).
    rng = np.random.default_rng(seed)
    showcase = {1: ("stain", 0), 3: ("scratch", 1), 5: ("chip", 2)}
    for i, (t, face) in showcase.items():
        sp = c.random_defect(rng, t=t, face=face); sp["scale"] = 0.8
        c.set_defect(i, [sp])
    judge = FaceJudge()
    prog = LaneProgram(c, judge)
    rec = Recorder(c, prog, judge, path)
    prog.capture_listeners.append(rec.on_capture)
    orig_capture = prog.capture

    def capture(i, cam, face_dir, label):
        p = orig_capture(i, cam, face_dir, label)
        if i == rec.part:
            for _ in range(6):                     # hold so the verdict can be read
                rec.hook(c)
        return p
    prog.capture = capture
    c.frame_hook = rec.hook
    events = []
    for i in range(N_PARTS):
        rec.part = i
        rec.thumbs = {k: v for k, v in rec.thumbs.items() if k == "gateTop"}
        f0 = rec.frame
        r = prog.run_part(i)
        rec.n_done += 1
        rec.n_ng += r["decision"] == "NG"
        rec.n_back += r["decision"] == "OK"
        truth = [f"{TYPE_JA[s['type']]}@{FACE_JA[FACE_NAMES[s['face']]]}" for s in c.defect_list[i]]
        events.append(dict(part=i, f0=f0, f1=rec.frame, decision=r["decision"], n_views=r["n_views"],
                           truth=truth or "良品", cycle=r["t_done"] - r["t_pick"]))
        print(events[-1], flush=True)
    prog.status = "全数検査完了：良品はレーンを流れ、不良品だけがNG箱に入っている"
    prog.r.move([0.37, 0.10, 0.35], settle=0)
    for _ in range(40):
        rec.hook(c)
    rec.writer.close()
    json.dump(events, open('out/lane_demo_events.json', 'w'), ensure_ascii=False, indent=1)
    imageio.imwrite('out/lane_demo_last.png', rec.compose())


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 21)
