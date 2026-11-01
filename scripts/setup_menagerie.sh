#!/usr/bin/env bash
# Fetch only the Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d third_party/mujoco_menagerie/franka_emika_panda ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
fi
echo "menagerie ready"
