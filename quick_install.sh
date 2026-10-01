#!/usr/bin/env bash
# Quick install: sets up a brand-new environment for this project from
# scratch using uv, the pinned Isaac Lab / asimov-1 submodules, and this
# extension. For a custom Isaac Lab checkout or a conda + pip setup instead,
# follow the "Advanced install" section in README.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

sudo apt-get update && sudo apt-get install -y cmake build-essential libglu1-mesa

git submodule update --init third_party/IsaacLab
git submodule update --init --filter=blob:none third_party/asimov-1
git -C third_party/asimov-1 sparse-checkout set sim-model

uv venv --seed --python 3.11
source .venv/bin/activate

uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

(cd third_party/IsaacLab && ./isaaclab.sh --install rsl_rl)

uv pip install -e "${ROOT}/source/cyclotron"

echo
echo "Install complete. Activate the environment with:"
echo "    source .venv/bin/activate"
