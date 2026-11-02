"""Record the never-stopping line (camera-synchronised picking) with Japanese overlays.

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
from sortbot.cell import FACE_NAMES, HALF  # noqa: E402
from sortbot.judge import FaceJudge  # noqa: E402
from sortbot.lane import LX, N_PARTS, V_LANE, LaneCell  # noqa: E402
from sortbot.pose import PoseEstimator  # noqa: E402
from sortbot.imitation import GraspPolicy  # noqa: E402
from sortbot.sync_routine import SyncProgram  # noqa: E402

W, H = 1280, 720
LATENCY = 0.3
FONT_PATHS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]
FONT = next(p for p in FONT_PATHS if os.path.exists(p))
CAMS = [("gateTop", "カメラ1 上面"), ("stBottom", "カメラ2 底面"), ("stSide", "カメラ3 側面")]
FACE_JA = {"top": "上面", "bottom": "底面", "+x": "側面1", "-x": "側面2", "+y": "側面3", "-y": "側面4"}
CLASSES_JA = ["良", "傷", "油ジミ", "欠け"]
TYPE_JA = {"scratch": "打痕・傷", "stain": "油ジミ", "chip": "角欠け"}
PX_PER_M = 1667.0          # camera-1 image scale at the part's top face (128 px image)


def font(sz):
    return ImageFont.truetype(FONT, sz)


def cam1_corners(dx, dy, yaw, size):
    """Corners of the part's top face in the camera-1 image (world +x -> image up, world +y -> image left)."""
    k = size / 128.0
    out = []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        cx = dx + HALF * (sx * np.cos(yaw) - sy * np.sin(yaw))
        cy = dy + HALF * (sx * np.sin(yaw) + sy * np.cos(yaw))
        out.append(((63.5 - cy * PX_PER_M) * k, (63.5 - cx * PX_PER_M) * k))
    return out


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
        self.thumbs[cam] = (img, self.judge.last["probs"], p)

    def face_status(self):
        return {FACE_NAMES[e["face"]]: e["p"] for e in self.prog.log if e["part"] == self.part}

    def stream_panel(self, canvas, d, x0, y0, size=150):
        prog = self.prog
        d.text((x0, y0 - 26), "カメラ1（1秒に10枚）＋AIの位置推定", font=self.f_mid, fill=(230, 230, 230))
        canvas.paste(Image.fromarray(self.c.render("gateTop")).resize((size, size)), (x0, y0))
        d.rectangle([x0, y0, x0 + size, y0 + size], outline=(70, 110, 170), width=2)
        tx = x0 + size + 10
        if prog.in_view and prog.frames:
            _, dx, dy, yaw = prog.frames[-1][:4]
            pts = [(x0 + a, y0 + b) for a, b in cam1_corners(dx, dy, yaw, size)]
            d.line(pts + [pts[0]], fill=(255, 150, 40), width=3)
            d.text((tx, y0 + 4), "推定中", font=self.f_mid, fill=(255, 170, 70))
            d.text((tx, y0 + 30), f"横 {dx * 1000:+.1f} mm", font=self.f_small, fill=(220, 220, 220))
            d.text((tx, y0 + 50), f"角度 {np.degrees(yaw):+.0f}°", font=self.f_small, fill=(220, 220, 220))
        elif prog.tracks:
            tr = prog.tracks[-1]
            d.text((tx, y0 + 4), f"部品{tr['part'] + 1} 推定済み", font=self.f_small, fill=(120, 200, 140))
            d.text((tx, y0 + 26), f"横 {(tr['x'] - LX) * 1000:+.1f} mm", font=self.f_small, fill=(220, 220, 220))
            d.text((tx, y0 + 46), f"角度 {np.degrees(tr['yaw']):+.0f}°", font=self.f_small, fill=(220, 220, 220))
        else:
            d.text((tx, y0 + 4), "待機中", font=self.f_small, fill=(160, 160, 170))
        d.text((tx, y0 + 80), f"遅れ {LATENCY:.1f} 秒を", font=self.f_small, fill=(160, 165, 175))
        d.text((tx, y0 + 98), "撮影時刻で補正", font=self.f_small, fill=(160, 165, 175))
        d.text((tx, y0 + 122), "位置は回転計で追跡", font=self.f_small, fill=(160, 165, 175))

    def compose(self):
        c = self.c
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        canvas.paste(Image.fromarray(c.render_video()), (0, 56))
        d = ImageDraw.Draw(canvas)
        d.text((16, 13), "ロボットによる6面外観検査ライン：止まらないレーンでカメラとアームを同期（MuJoCo）",
               font=self.f_title, fill=(240, 240, 240))
        d.rounded_rectangle([860, 66, 950, 96], 6, fill=(200, 60, 50))
        d.text((870, 70), "2倍速", font=self.f_mid, fill=(255, 255, 255))
        # Station close-up as picture-in-picture.
        canvas.paste(Image.fromarray(c.render_video("video_station")).resize((256, 144)), (8, 64))
        d.rectangle([8, 64, 264, 208], outline=(90, 95, 105), width=2)
        d.text((14, 188), "検査台のアップ", font=self.f_small, fill=(240, 240, 240))
        if self.prog.policy_active and self.prog.wrist_img is not None:
            canvas.paste(Image.fromarray(self.prog.wrist_img).resize((144, 144)), (276, 64))
            d.rectangle([276, 64, 420, 208], outline=(255, 150, 40), width=3)
            d.rectangle([276, 210, 560, 252], fill=(18, 20, 24))
            d.text((282, 212), "手先カメラ（方策への入力）", font=self.f_small, fill=(255, 190, 110))
            d.text((282, 231), "模倣学習した方策が操作中", font=self.f_small, fill=(255, 190, 110))
        d.rectangle([0, 596, 960, 720], fill=(28, 31, 38))
        d.text((18, 606), self.prog.status or "", font=self.f_big, fill=(255, 220, 120))
        d.text((18, 650), f"検査済み {self.n_done}/{N_PARTS}　NG箱へ {self.n_ng}　良品をレーンへ戻す {self.n_back}",
               font=self.f_mid, fill=(200, 205, 215))
        d.text((18, 682), f"レーンは止まらない（{V_LANE * 100:.0f} cm/秒）　シミュレーション時間 {c.time:5.1f} 秒",
               font=self.f_small, fill=(150, 155, 165))
        x0 = 970
        self.stream_panel(canvas, d, x0, 90)
        d.text((x0, 256), "検査カメラ画像とAI判定", font=self.f_mid, fill=(230, 230, 230))
        for k, (cam, name) in enumerate(CAMS):
            bx, by = x0 + k * 102, 282
            d.rectangle([bx, by, bx + 98, by + 98], outline=(70, 70, 80), width=2)
            if cam in self.thumbs:
                img, p, pd = self.thumbs[cam]
                canvas.paste(Image.fromarray(img).resize((94, 94)), (bx + 2, by + 2))
                col = (230, 60, 50) if pd > 0.5 else (60, 200, 90)
                d.rectangle([bx, by, bx + 98, by + 98], outline=col, width=4)
                lab = f"NG {CLASSES_JA[int(np.argmax(p[1:]) + 1)]}" if pd > 0.5 else "OK"
                d.text((bx + 2, by + 101), f"{lab} {pd:.2f}", font=self.f_small, fill=col)
            d.text((bx + 2, by + 120), name, font=self.f_small, fill=(170, 175, 185))
        y6 = 432
        d.text((x0, y6), f"部品{(self.part or 0) + 1}の6面", font=self.f_mid, fill=(230, 230, 230))
        st = self.face_status()
        for k, f in enumerate(FACE_NAMES):
            bx, by = x0 + (k % 3) * 102, y6 + 28 + (k // 3) * 36
            col = ((200, 55, 45) if st[f] > 0.5 else (45, 150, 75)) if f in st else (55, 58, 66)
            d.rounded_rectangle([bx, by, bx + 96, by + 30], 5, fill=col)
            d.text((bx + 26, by + 6), FACE_JA[f], font=self.f_small, fill=(245, 245, 245))
        d.text((x0, y6 + 104), "緑=良　赤=不良　灰=未撮影", font=self.f_small, fill=(150, 155, 165))
        d.text((x0, 600), "AI①：カメラ1の画像から位置と向き", font=self.f_small, fill=(150, 155, 165))
        d.text((x0, 620), "AI②：各面の画像から良・不良", font=self.f_small, fill=(150, 155, 165))
        d.text((x0, 640), "数値 = 不良の確率（0.5超でNG）", font=self.f_small, fill=(150, 155, 165))
        return np.asarray(canvas)

    def hook(self, cell):
        self.writer.append_data(self.compose())
        self.frame += 1


def main(seed=23, path="out/lane_demo.mp4"):
    c = LaneCell(video_size=(960, 540))
    c.reset(seed=seed, n_defect=0, continuous=True)
    # Showcase: a stain on the top face (camera 1), a scratch on the bottom (camera 2),
    # and a chipped corner on a face the fingers cover at first (found only after the regrasp).
    rng = np.random.default_rng(seed)
    showcase = {1: ("stain", 0), 3: ("scratch", 1), 5: ("chip", 2)}
    for i, (t, face) in showcase.items():
        sp = c.random_defect(rng, t=t, face=face); sp["scale"] = 0.8
        c.set_defect(i, [sp])
    judge = FaceJudge()
    prog = SyncProgram(c, judge, PoseEstimator(), latency=LATENCY, compensate=True, grasp_policy=GraspPolicy())
    rec = Recorder(c, prog, judge, path)
    prog.capture_listeners.append(rec.on_capture)
    orig_capture = prog.capture

    def capture(i, cam, face_dir, label):
        p = orig_capture(i, cam, face_dir, label)
        for _ in range(6):                     # hold so the verdict can be read
            rec.hook(c)
        return p
    prog.capture = capture
    c.frame_hook = rec.hook
    events = []
    for i in range(N_PARTS):
        rec.part = i
        rec.thumbs = {}
        f0 = rec.frame
        r = prog.run_part(i)
        rec.n_done += 1
        rec.n_ng += r["decision"] == "NG"
        rec.n_back += r["decision"] == "OK"
        truth = [f"{TYPE_JA[s['type']]}@{FACE_JA[FACE_NAMES[s['face']]]}" for s in c.defect_list[i]]
        events.append(dict(part=i, f0=f0, f1=rec.frame, decision=r["decision"], grip=r.get("grip"),
                           n_views=r["n_views"], truth=truth or "良品", cycle=r["t_done"] - r["t_pick"]))
        print(events[-1], flush=True)
    prog.status = "全数検査完了：良品はレーンを流れ、不良品だけがNG箱に入っている"
    prog.r.move([0.37, 0.10, 0.35], settle=0)
    for _ in range(40):
        rec.hook(c)
    rec.writer.close()
    json.dump(events, open('out/lane_demo_events.json', 'w'), ensure_ascii=False, indent=1)
    imageio.imwrite('out/lane_demo_last.png', rec.compose())


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 23)
