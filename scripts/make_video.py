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
EV = json.load(open("results/lane_eval.json"))["summary"]
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
    im = fit("concept/scene_lane.png", W, H)
    bg = Image.new("RGB", (W, H), BG)
    bg.paste(ImageEnhance.Brightness(im).enhance(0.16), ((W - im.width) // 2, (H - im.height) // 2))
    d = ImageDraw.Draw(bg)
    d.text((70, 220), "ロボットによる6面外観検査ライン", font=F(54), fill=(255, 255, 255))
    d.text((70, 300), "持ち替えで「つかんだ面」まで見る", font=F(40), fill=(255, 220, 120))
    d.text((70, 380), "アルミ切削ハウジング（40 mm角）の出荷前全数検査", font=F(28), fill=(220, 225, 235))
    d.text((70, 420), "MuJoCo シミュレーション ＋ 画像認識AI（ResNet-18）", font=F(28), fill=(220, 225, 235))
    d.text((70, 620), "Physical AI 応用編1 最終課題", font=F(22), fill=(180, 185, 195))
    return bg


def card_problem():
    bg = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(bg)
    d.text((60, 40), "課題：つかむと、指が触れる2面が見えなくなる", font=F(36), fill=(255, 255, 255))
    im = fit("concept/scene_lane.png", 600, 400)
    bg.paste(im, (640, 130))
    lines(d, 60, 130, [
        ("現場", (255, 220, 120)),
        "切削・洗浄後、梱包前の全数外観検査",
        "不良：打痕・傷／油ジミ／角欠け（6面どこでも）",
        "",
        ("上から撮るだけ → 底面・側面が写らない", (230, 120, 110)),
        ("持ち上げるだけ → 指の2面が写らない", (230, 120, 110)),
        "",
        ("このラインの解き方", (255, 220, 120)),
        "カメラ3台：上面（レーン）・底面・側面",
        "手首を180°回して側面の表裏を1台で撮る",
        "持ち替え台で90°つかみ直し、残り2面を撮る",
        "良品は動いているレーンの元の位置へ戻す",
    ], size=24, gap=10)
    d.text((640, 545), "構想図（生成AIで作成）。実装はこの配置をMuJoCoで再現", font=F(16), fill=(160, 165, 175))
    return bg


def card_result():
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(bg)
    d.text((50, 30), "結果：持ち替えまで行うと、見逃しは0個", font=F(36), fill=(20, 24, 30))
    im = fit("results/fig_detection.png", 820, 460)
    bg.paste(im, (30, 110))
    s = EV["six_face"]
    lines(d, 880, 140, [
        f"{s['parts']}個（不良{s['ng_parts']}個）で比較",
        "",
        ("上面だけ", (110, 115, 125)), f"検出 {PCT('top_only')}（見逃し{EV['top_only']['missed']}個）",
        ("持ち上げのみ", (60, 130, 220)), f"検出 {PCT('no_regrasp')}（見逃し{EV['no_regrasp']['missed']}個）",
        ("6面（持ち替えあり）", (30, 90, 200)), f"検出 {PCT('six_face')}（見逃し{s['missed']}個）",
        f"過検出 {PCT('six_face', 'false_reject_rate')}",
    ], size=24, gap=10, fill=(30, 34, 40))
    d.text((50, 640), "見逃しの差は、AIの精度より「撮っていない面があるか」で決まった", font=F(26), fill=(200, 60, 50))
    return bg


def card_ai():
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(bg)
    d.text((50, 30), "追加実験：見たことのない不良が来たら", font=F(36), fill=(20, 24, 30))
    im = fit("results/fig_unseen.png", 760, 430)
    bg.paste(im, (30, 110))
    nc, ncc = EV["six_face_no_chip"], EV["six_face_no_chip_combined"]
    lines(d, 820, 130, [
        ("良品の画像だけで学ぶ", (40, 140, 90)), ("異常検知を重ねると", (40, 140, 90)),
        "",
        "ラインでの角欠けの検出",
        f"{nc['recall_chip'] * 100:.0f}% → {ncc['recall_chip'] * 100:.0f}%",
        "",
        "良品の誤NG",
        f"0% → {ncc['false_reject_rate'] * 100:.0f}%",
    ], size=24, gap=10, fill=(30, 34, 40))
    d.text((50, 600), "本ラインの判定は分類器のみ（誤NG 0%）", font=F(26), fill=(30, 34, 40))
    d.text((50, 640), "異常検知だけが反応した部品は「再検査」へ回す運用が向く", font=F(26), fill=(200, 60, 50))
    return bg


def card_throughput():
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(bg)
    d.text((50, 30), "処理能力：カメラを増やさず、同じセルを並べる", font=F(36), fill=(20, 24, 30))
    im = fit("results/fig_cycle.png", 1180, 440)
    bg.paste(im, ((W - im.width) // 2, 100))
    one = EV["six_face"]["parts_per_min"]; lane = 60 * 0.03 / 0.28
    n = next(k for k in range(1, 10) if k * one >= lane * 0.97)
    lines(d, 60, 560, [
        f"6面検査は1台で約{one:.1f}個/分（良品1個あたりアーム作業 約{EV['six_face']['mean_cycle_good_s']:.0f}秒）",
        f"{n}台でレーンの搬送能力（約{lane:.1f}個/分）にほぼ達する試算",
    ], size=26, gap=12, fill=(30, 34, 40))
    return bg


def card_end():
    bg = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(bg)
    lines(d, 80, 210, [
        ("まとめ", (255, 220, 120)),
        "カメラ3台＋アーム1台＋持ち替えで、6面すべてをAIで判定",
        "良品は流れているレーンへ戻し、不良品だけをNG箱へ",
        "ROS 2 でライン（撮影）ノードと判定AIノードを分離",
        ("  /inspection/image → inspector → /inspection/verdict", (160, 165, 175)),
        "",
        ("コード・説明資料：GitHub（概要欄のリンク）", (200, 205, 215)),
        ("ロボットモデル：MuJoCo Menagerie Franka Emika Panda（Apache-2.0）", (160, 165, 175)),
    ], size=28, gap=16)
    return bg


def still(img, sec, w):
    a = np.asarray(img.convert("RGB"))
    for _ in range(int(sec * FPS)):
        w.append_data(a)


def main(demo, out):
    tmp = out + ".cards.mp4"
    w = imageio.get_writer(tmp, fps=FPS, quality=8, macro_block_size=1)
    still(card_title(), 4, w); still(card_problem(), 10, w)
    w.close()
    tmp2 = out + ".end.mp4"
    w = imageio.get_writer(tmp2, fps=FPS, quality=8, macro_block_size=1)
    still(card_result(), 9, w); still(card_ai(), 9, w); still(card_throughput(), 7, w); still(card_end(), 5, w)
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
