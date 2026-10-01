# Advanced Install

Use this instead of `quick_install.sh` if you already have your own Isaac
Lab checkout you want to reuse, want conda instead of uv, or just want to
understand what each install step does. Pick one of the two approaches
below; both end with the same "Install this extension" step.

**Contents**
- [Using your own Isaac Lab checkout](#using-your-own-isaac-lab-checkout)
- [Install with uv](#install-with-uv)
- [Install with conda and pip](#install-with-conda-and-pip)
- [Robot model](#robot-model)
- [Install this extension](#install-this-extension)

### Using your own Isaac Lab checkout

If you already have Isaac Lab cloned somewhere, skip the
`git submodule update --init third_party/IsaacLab` step in whichever
approach you follow below, and substitute your own path wherever
`third_party/IsaacLab` appears (for example,
`(cd /path/to/your/IsaacLab && ./isaaclab.sh --install rsl_rl)`). Everything
else — the environment setup, the robot model, and installing this
extension — stays the same. This project pins RSL-RL 5.0.1 and expects
Isaac Lab `main` (currently reporting 2.3.2, with RSL-RL 5 support); other
versions may not be compatible.

### Install with uv

These steps target Ubuntu 22.04+ (x86_64), with
[uv](https://docs.astral.sh/uv/) and a compatible NVIDIA driver installed.

1. Install build tools needed to compile some Isaac Lab dependencies, and
   `libglu1-mesa`, which Isaac Sim's `omni.iray` extension needs at runtime:

   ```bash
   sudo apt-get update && sudo apt-get install -y cmake build-essential libglu1-mesa
   ```

2. Fetch Isaac Lab, pinned to a known-working commit (skip this if using
   your own checkout — see above):

   ```bash
   git submodule update --init third_party/IsaacLab
   ```

3. Create a Python 3.11 virtual environment and activate it. `--seed`
   installs `pip` into the venv, since Isaac Lab's own installer shells out
   to it:

   ```bash
   uv venv --seed --python 3.11
   source .venv/bin/activate
   ```

4. Install Isaac Sim and CUDA-enabled PyTorch into the active environment:

   ```bash
   uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
   uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   ```

5. Install Isaac Lab's RSL-RL extras. This installs `isaaclab`,
   `isaaclab_rl`, and `isaaclab_tasks` in editable mode from the checkout,
   plus `rsl-rl-lib`:

   ```bash
   (cd third_party/IsaacLab && ./isaaclab.sh --install rsl_rl)
   ```

See the
[official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
for prerequisites and troubleshooting.

### Install with conda and pip

These steps target Ubuntu 22.04+ (x86_64), with Conda and a compatible
NVIDIA driver installed.

1. Install build tools needed to compile some Isaac Lab dependencies, and
   `libglu1-mesa`, which Isaac Sim's `omni.iray` extension needs at runtime:

   ```bash
   sudo apt-get update && sudo apt-get install -y cmake build-essential libglu1-mesa
   ```

2. Fetch Isaac Lab, pinned to a known-working commit (skip this if using
   your own checkout — see above):

   ```bash
   git submodule update --init third_party/IsaacLab
   ```

3. Create a Python 3.11 conda environment and activate it:

   ```bash
   conda create -n env_isaaclab python=3.11 -y
   conda activate env_isaaclab
   python -m pip install --upgrade pip
   ```

4. Install Isaac Sim and CUDA-enabled PyTorch into the active environment:

   ```bash
   python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
   python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
   ```

5. Install Isaac Lab's RSL-RL extras. This installs `isaaclab`,
   `isaaclab_rl`, and `isaaclab_tasks` in editable mode from the checkout,
   plus `rsl-rl-lib`:

   ```bash
   (cd third_party/IsaacLab && ./isaaclab.sh --install rsl_rl)
   ```

Keep `env_isaaclab` active for the remaining steps. See the
[official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
for prerequisites and troubleshooting.

### Robot model

The `asimov-1` submodule provides the robot's URDF and meshes at
`third_party/asimov-1/sim-model`; the robot configuration uses this location
by default, so no path changes are needed. Its repository is large (mostly
unrelated CAD/fabrication files), so this fetches only the `sim-model`
subtree via sparse-checkout:

```bash
git submodule update --init --filter=blob:none third_party/asimov-1
git -C third_party/asimov-1 sparse-checkout set sim-model
```

To use a robot model stored elsewhere, set `ASIMOV_1_MODEL_DIR` to a
`sim-model` directory containing `urdf/asimov_1.urdf`. This is also needed
for non-editable installs, since the default path is resolved relative to
this repository's checkout:

```bash
export ASIMOV_1_MODEL_DIR=/path/to/asimov-1/sim-model
```

### Install this extension

With the environment from either approach above still active, install this
extension in editable mode:

```bash
./isaac_asimov.sh --install
```

### Policy viewer (optional)

`./isaac_asimov.sh --view` runs in the browser through the
`third_party/humanoid-policy-viewer` submodule and needs only Node.js 20 or
newer, not this Python environment. It fetches the submodule, its npm
dependencies and its own copy of the robot model on first use, so nothing
needs to be installed ahead of time.
