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

# Only sim-model (URDF + STL meshes) is needed from asimov-1. Set up the sparse
# checkout before the first checkout; otherwise git-lfs downloads a ~400 MB CAD
# file and git fetches every other CAD file, all of which sparse-checkout deletes.
echo
echo "Downloading asimov-1 STL files. This may take a while..."
if [ ! -e third_party/asimov-1/.git ]; then
    git clone --filter=blob:none --no-checkout https://github.com/menloresearch/asimov-1.git third_party/asimov-1
fi
git -C third_party/asimov-1 sparse-checkout set sim-model
git -C third_party/asimov-1 checkout --quiet "$(git rev-parse HEAD:third_party/asimov-1)"
git submodule absorbgitdirs third_party/asimov-1
git submodule init third_party/asimov-1

uv venv --seed --python 3.12
source .venv/bin/activate

uv pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

(cd third_party/IsaacLab && ./isaaclab.sh -i 'rl[rsl-rl]')
# --video records with MoviePy, which Isaac Lab doesn't install by default.
uv pip install "moviepy>=1.0.3,<2.0.0.dev0"

uv pip install -e "${ROOT}/source/cyclotron"

echo
echo "Install complete. Activate the environment with:"
echo "    source .venv/bin/activate"
echo "The first run of Isaac Sim asks you to accept the NVIDIA Omniverse EULA"
echo "(set OMNI_KIT_ACCEPT_EULA=yes to accept it non-interactively) and can take"
echo "over ten minutes while it sets up its extensions."
