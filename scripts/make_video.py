"""Assemble the submission video: title and problem cards, the 2x demo, and the result cards.

usage: make_video.py <demo.mp4> <out.mp4>
"""
import json
import os
import subprocess
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

W, H, FPS = 1280, 720, 20
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
F = lambda s: ImageFont.truetype(FONT, s)  # noqa: E731
BG = (18, 20, 24)
Z = json.load(open("results/sync_eval.json"))
EV, LATS = Z["summary"], Z["latencies"]
PCT = lambda k, f="recall": f"{EV[k][f] * 100:.0f}%"  # noqa: E731


def fit(path, w, h):
    im = Image.open(path).convert("RGB")
    r = min(w / im.width, h / im.height)
    return im.resize((int(im.width * r), int(im.height * r)))


def lines(d, x, y, items, size=26, gap=12, fill=(230, 232, 238)):
    for t in items:
        col = fill
        if isinstance(t, tuple):
            t, col = t
        d.text((x, y), t, font=F(size), fill=col)
        y += size + gap
    return y


def card_title():
    im = fit("concept/site_layout.png", W, H)
    bg = Image.new("RGB", (W, H), BG)
    bg.paste(ImageEnhance.Brightness(im).enhance(0.16), ((W - im.width) // 2, (H - im.height) // 2))
    d = ImageDraw.Draw(bg)
    d.text((70, 210), "ロボットによる6面外観検査ライン", font=F(54), fill=(255, 255, 255))
    d.text((70, 290), "止まらないレーンで、カメラとアームを同期させる", font=F(38), fill=(255, 220, 120))
    d.text((70, 370), "アルミ切削ハウジング（40 mm角）の出荷前全数検査", font=F(28), fill=(220, 225, 235))
    d.text((70, 410), "MuJoCo シミュレーション ＋ ROS 2 ＋ 画像認識AI（ResNet-18）", font=F(28), fill=(220, 225, 235))
    d.text((70, 620), "Physical AI 応用編1 最終課題", font=F(22), fill=(180, 185, 195))
    return bg


def card_problem():
    bg = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(bg)
    d.text((60, 36), "難しい点と、このラインの解き方", font=F(36), fill=(255, 255, 255))
    im = fit("concept/top_camera_flow.png", 600, 340)
    bg.paste(im, (650, 120))
    lines(d, 60, 116, [
        ("① 部品は止まらない・位置も向きもばらばら", (230, 120, 110)),
        "→ カメラ1を撮り続け、AIが位置と向きを推定",
        "→ ベルトの回転計と同期して追いかけてつかむ",
        "",
        ("② カメラからアームまで遅れがある", (230, 120, 110)),
        "→ 撮影時刻の回転計の値と組み合わせて補正",
        "",
        ("③ つかむと指の2面が隠れる", (230, 120, 110)),
        "→ 手首180°回転＋持ち替えで6面すべて撮る",
        "",
        "良品は流れているレーンの元の位置へ戻す",
    ], size=24, gap=9)
    d.text((650, 470), "撮影の流れ（生成AIで作成したイメージ）", font=F(16), fill=(160, 165, 175))
    return bg


def card_sync():
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(bg)
    d.text((50, 30), "結果①：遅れを補正すると、流れる部品を確実につかめる", font=F(34), fill=(20, 24, 30))
    im = fit("results/fig_latency.png", 800, 450)
    bg.paste(im, (30, 105))
    mx = max(LATS)
    lines(d, 860, 150, [
        "カメラの推定がアームに届くまで",
        f"{mx:.1f} 秒遅れたとき",
        "",
        ("補正あり", (30, 90, 200)), f"つかみ成功 {PCT(f'B_{mx}_1', 'grip_success')}",
        "",
        ("補正なし", (200, 70, 60)), f"つかみ成功 {PCT(f'B_{mx}_0', 'grip_success')}",
    ], size=24, gap=10, fill=(30, 34, 40))
    d.text((50, 600), "補正なしでは、遅れの間に進んだ分（3 cm/秒 × 遅れ）だけ後ろを狙ってしまう", font=F(24), fill=(80, 85, 95))
    d.text((50, 640), "撮影した瞬間の回転計の値と組み合わせることで、遅れを打ち消す", font=F(26), fill=(30, 90, 200))
    return bg


def card_result():
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(bg)
    d.text((50, 30), "結果②：持ち替えまで行うと、見逃しは0個", font=F(36), fill=(20, 24, 30))
    im = fit("results/fig_detection.png", 820, 460)
    bg.paste(im, (30, 110))
    s = EV["A_six_face"]
    lines(d, 880, 140, [
        f"{s['parts']}個（不良{s['ng_parts']}個）で比較",
        "",
        ("上面だけ", (110, 115, 125)), f"検出 {PCT('A_top_only')}（見逃し{EV['A_top_only']['missed_defects']}個）",
        ("持ち上げのみ", (60, 130, 220)), f"検出 {PCT('A_no_regrasp')}（見逃し{EV['A_no_regrasp']['missed_defects']}個）",
        ("6面（持ち替えあり）", (30, 90, 200)), f"検出 {PCT('A_six_face')}（見逃し{s['missed_defects']}個）",
        f"過検出 {PCT('A_six_face', 'false_reject_rate')}",
    ], size=24, gap=10, fill=(30, 34, 40))
    d.text((50, 640), "すべて止まらないレーンで、カメラと同期してつかんだ結果", font=F(24), fill=(80, 85, 95))
    return bg


def card_end():
    bg = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(bg)
    lines(d, 80, 200, [
        ("まとめ", (255, 220, 120)),
        "止まらないレーンで、AIの位置推定と回転計を同期させて追いかけてつかむ",
        "遅れを撮影時刻で補正し、持ち替えで6面すべてをAIで判定",
        "良品は流れているレーンの元の位置へ戻し、不良品だけをNG箱へ",
        "撮影するラインと判定AIは ROS 2 で別ノードに分離",
        "",
        ("コード・説明資料：GitHub（概要欄のリンク）", (200, 205, 215)),
        ("ロボットモデル：MuJoCo Menagerie Franka Emika Panda（Apache-2.0）", (160, 165, 175)),
    ], size=26, gap=14)
    return bg


def still(img, sec, w):
    a = np.asarray(img.convert("RGB"))
    for _ in range(int(sec * FPS)):
        w.append_data(a)


def main(demo, out):
    tmp = out + ".cards.mp4"
    w = imageio.get_writer(tmp, fps=FPS, quality=8, macro_block_size=1)
    still(card_title(), 4, w); still(card_problem(), 11, w)
    w.close()
    tmp2 = out + ".end.mp4"
    w = imageio.get_writer(tmp2, fps=FPS, quality=8, macro_block_size=1)
    still(card_sync(), 9, w); still(card_result(), 9, w); still(card_end(), 6, w)
    w.close()
    lst = out + ".txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in [tmp, demo, tmp2]))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", str(FPS), out], check=True)
    for p in (tmp, tmp2, lst):
        os.remove(p)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
