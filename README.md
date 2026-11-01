# ロボットによる6面外観検査ライン：止まらないレーンで、カメラとアームを同期させる（MuJoCo シミュレーション）

Physical AI 応用編1 最終課題。

精密切削工場の出荷前工程を想定し、アルミ削り出しのセンサーハウジング（40 mm 角、ねじ穴 1 か所）の
**6面全数外観検査**を、止まらないコンベア・カメラ3台・ロボットアーム1台で自動化するラインを MuJoCo 上に構築しました。

![構想図](concept/site_layout.png)

## 何をしているか

1. **カメラ1が撮り続ける**：部品は止まらずに流れてくる（間隔・横ずれ ±15 mm・向きはばらばら）。カメラ1は1秒に10枚撮り、
   AI（ResNet-18）が毎フレーム部品の位置と向きを推定する。いちばん中央に写ったフレームで上面を検査する。
2. **カメラとアームを同期する**：各フレームに撮影した瞬間のベルトの回転計（エンコーダ）の値を付け、部品がベルトのどこに載っているかを求める。
   その後の位置は回転計で追う。推定がアームに届くまでの遅れ（推論＋通信）は、撮影時刻の値と組み合わせて打ち消す。
3. **追いかけてつかむ**：部品が「つかむ区間」に入ったら、ベルトと同じ速さで追いかけながらつかむ。指の開きと部品の位置で、つかめたかを確かめる。
4. **6面を見せる**：検査台でカメラ2が底面、カメラ3（横カメラ1台）が側面を撮る。手首（第7関節）だけを180°回して反対側も撮り、
   持ち替え台で90°つかみ直して、指で隠れていた2面も撮る。
5. **仕分け**：1面でも不良なら NG 箱へ。6面とも良なら、流れているレーンの元の位置へ追いかけて戻す。

![上部カメラの流れ](concept/top_camera_flow.png)

## 結果

推定がアームに届くまでの遅れと、流れる部品をつかめた割合（各条件 24 個）：

| 遅れ（秒） | 0.0 | 0.3 | 0.6 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| 補正あり | 100% | 100% | 100% | 100% | 100% |
| 補正なし | 100% | 92% | 8% | 0% | 0% |

検査方式の比較（止まらないレーン、48個・うち不良18個）：

| 方式 | 不良品の検出率 | 上面以外にだけ不良がある部品 | 良品の誤NG | つかみ成功率 |
|---|---|---|---|---|
| 上面だけ（カメラ1） | 11% | 0% | 0% | 100%（2/2） |
| 持ち上げのみ（持ち替えなし） | 61% | 56% | 0% | 100%（48/48） |
| **6面（持ち替えあり）** | **100%** | **100%** | 0% | 100%（48/48） |

上面だけの方式は、上面で不良が見つかった部品だけをつかんで NG 箱へ運ぶため、つかんだ回数が少なくなります。

処理時間と撮影時刻の確認（第3回の演習にならった計測、`scripts/measure_timing.py`、このPCのCPU）：

| AI | 中央値 | p95 | 予算 |
|---|---|---|---|
| 位置・向きの推定（カメラ1の毎フレーム、816回） | 8.6 ms | 9.5 ms | 100 ms（カメラ1の撮影間隔、10 Hz） |
| 面の判定（検査の1枚ごと、129回） | 8.5 ms | 9.0 ms | ― |

1ロットで実際に使った画像を、ウォームアップの後に3回ずつ順番を混ぜて推論し、描画と結果の照合は計測から外しました。
撮影時刻の間隔はすべて 100 ms で、抜けたフレームはありませんでした。各フレームの回転計の進みはベルト速度×間隔と一致し、
保存した画像は描画結果のコピーで、同じメモリを共有していないことも確かめています。
評価（`evaluate_sync.py`、7プロセスを並列・各1スレッド）では、位置推定は平均 56 ms でした。

図は `results/`、詳しい説明は `report/report.pdf` にあります。

## ROS 2 での実行

ライン（シミュレーション＋ロボット＋位置推定）と、面の判定AIを、ROS 2 の2つのノードに分けて動かせます。

```
conveyor_line ──/inspection/image (sensor_msgs/Image)──▶ inspector
conveyor_line ◀──/inspection/verdict (std_msgs/String)── inspector
conveyor_line ──/camera1/image, /camera1/pose_estimate, /joint_states, /line/status, /line/result──▶
```

macOS では RoboStack で ROS 2 Humble を入れました。

```bash
CONDA_SUBDIR=osx-arm64 conda create -n pai_ros -c conda-forge -c robostack-humble python=3.11 ros-humble-ros-base pytorch torchvision
conda activate pai_ros && pip install mujoco pillow imageio
bash ros2/run_demo.sh classifier
```

## 再現手順

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
mkdir -p data models results out                      # データ・重み・結果・動画の置き場所
./scripts/setup_menagerie.sh                          # Panda モデルを取得
./.venv/bin/python scripts/gen_lane_dataset.py 2000 8 # 面画像を合成（判定AI用）
./.venv/bin/python scripts/train_classifier.py 6      # 面の判定AIを学習
./.venv/bin/python scripts/train_pose.py 8000 5       # カメラ1の位置・向き推定AIを学習
./.venv/bin/python scripts/test_sync.py 3 0.3 1       # 止まらないレーンで1ロット流して確認（遅れ0.3秒・補正あり）
./.venv/bin/python scripts/evaluate_sync.py 6         # 検査方式の比較と、遅れの実験
./.venv/bin/python scripts/measure_timing.py 3        # AI の処理時間と撮影時刻の確認
./.venv/bin/python scripts/make_figures.py            # 図を作成
./.venv/bin/python scripts/make_lane_demo.py          # デモ映像（2倍速）を記録
./.venv/bin/python scripts/make_video.py out/lane_demo.mp4 out/video.mp4  # 提出用の動画にまとめる
./.venv/bin/python report/build.py report/report.pdf  # 説明資料を PDF にする
```

macOS（Apple Silicon）、Python 3.11、MuJoCo 3.17 で動作確認しました。学習済みの重み（`models/`）は容量が大きいためリポジトリに含めず、上の手順で作成します。

## ディレクトリ

```
sortbot/lane.py          ライン（止まらないコンベア・回転計・カメラ3台・検査台・持ち替え台・NG箱・部品8個）
sortbot/sync_routine.py  カメラとアームの同期（撮り続け→位置推定→回転計で追跡→遅れ補正→追いかけてつかむ）
sortbot/lane_routine.py  6面を見せる動き（手首180°、持ち替え、レーンの元の位置へ戻す）と、レーンを止める方式
sortbot/pose.py          カメラ1の画像から位置と向きを推定するAI
sortbot/judge.py         面ごとの良否を判定するAI
sortbot/lane_dataset.py  判定AI用の面画像の合成（ラインと同じカメラ・姿勢＋照明や位置のランダム化）
sortbot/anomaly.py       良品だけで作る異常検知（追加実験）
sortbot/cell.py, env.py  部品形状・不良の定義、共通の補助関数
ros2/                    ROS 2 ノード（ライン、判定AI）と起動スクリプト
scripts/                 データ生成・学習・評価・図・デモ映像
report/                  説明資料（HTML と PDF の生成スクリプト）
results/                 評価結果（JSON・図）
concept/                 構想図
research/report.md       6面検査の先行事例と検証方法の調査メモ
```

## 前提と限界

- **把持の補助**：つかんだ後は、部品を手先に固定する拘束で運びます。接触の摩擦だけだと、シミュレーション上で部品が指の間で滑ったためです。
  つかめたかどうかは、指の開きと部品の位置で別に判定しています。
- 学習も評価もシミュレーションの合成画像です。実機では金属面の反射や照明、微小な傷への対応（sim-to-real）が課題になります。
- 遅れの実験の 0〜1.5 秒は、シミュレーション上で与えた一定の値です。ROS 2 の通信遅延の実測や、遅れの揺らぎは含めていません。
- ロボットの動作は逆運動学と手順で設計しています。AI は位置・向きの推定と良否判定に使っています。
  次の段階として、この動作をお手本のデータにして、模倣学習（ACT など）で動作そのものを学習させることを考えています。
- 構想図（`concept/`）と先行事例の調査（`research/report.md`）には生成AIを使いました。コードの作成にもAIアシスタントを使っています。

## ライセンス

本リポジトリのコードは MIT License。`third_party/` に取得する MuJoCo Menagerie の Panda モデルは Apache-2.0 です。
