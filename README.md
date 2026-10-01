# Asimov 1 Locomotion

Asimov 1 is an open-source humanoid robot developed by
[Menlo Research](https://menlo.ai/). This repository provides the
training and evaluation code for its locomotion policies, so you can
train policies in simulation and deploy them to the real robot.

Get your own Asimov 1.
[Order now](https://menlo.ai/order).

## Isaac Asimov

Standalone Isaac Lab extension for training Asimov-1 locomotion policies with
PPO and adversarial motion priors (AMP).

## Quick Install

For a brand-new machine with no existing Isaac Lab setup, run the install
script from the project root. It sets up a uv-managed environment, pulls the
pinned Isaac Lab and `asimov-1` submodules, and installs everything needed
to train and play. Prerequisites: Ubuntu 22.04+ (x86_64), a compatible
NVIDIA driver, [uv](https://docs.astral.sh/uv/) installed, and `sudo` access
(used to install `cmake`/`build-essential`).

```bash
git clone https://github.com/menloresearch/isaac_asimov.git
cd isaac_asimov
./quick_install.sh
```

### Advanced Install

If you already have your own Isaac Lab checkout you want
to reuse, want conda instead of uv, or just want to understand what each
install step does: **[Advanced Install](INSTALL.md)**.

## Train

Before starting a training run, run a quick test to ensure the full pipeline is functional. The following code will fire off a short training run with a small number of environments which should take ~10 minutes on a 4090.

**Quick Test**

This is a small job to see if the full training code is working. These settings should work for most GPUs and finish relatively quickly.

```bash
./isaac_asimov.sh --train \
    --task Asimov1-Velocity-AMP-v0 --num_envs 128 --headless --max_iterations 100
```

### Single GPU Training Run

Use this code to replicate the training run for our baseline policy using a single GPU.

**AMP (recommended)**

```bash
./isaac_asimov.sh --train \
    --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless
```

**Plain PPO baseline**

```bash
./isaac_asimov.sh --train \
    --task Asimov1-Velocity-v0 --num_envs 4096 --headless
```

Useful flags: `--max_iterations <n>`, `--seed <n>`, `--video` (record rollout
clips during training).

Note: We use 4096 `num_envs` to train our baseline locomotion policy using A6000 or pro 6000. If you hit any out of memory errors, consider lowering the `num_envs`. However, this means that the policy may take longer to converge or may be less stable for the same number of iterations.

### Multi-GPU Training Run

This code runs training via `--distributed` with two GPUs and 4096 environments per GPU. You should adjust the parameters according to the compute available to you.

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/rsl_rl/train.py \
    --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless --distributed
```

## Play / Evaluate

Load the latest checkpoint and visualize the trained policy:

```bash
./isaac_asimov.sh --play \
    --task Asimov1-Velocity-AMP-Play-v0 --num_envs 32
```

Use `--checkpoint <path>` to select a specific checkpoint (`--target` is an
alias), or `--onnx-output <path>` for an extra ONNX export.

Checkpoints and logs are written to `logs/rsl_rl/<experiment_name>/<run>/`.

## Share and View on Hugging Face

Share a trained policy on the [Hugging Face Hub](https://huggingface.co/models?other=asimov),
where anyone can run it in the browser-based
[humanoid-policy-viewer](https://github.com/menloresearch/humanoid-policy-viewer),
with no GPU needed. A policy repo needs three files from your run, plus a model card:

| File | Where it comes from |
|---|---|
| `policy.onnx` | `<run>/exported/policy.onnx`, written by `--play` |
| `env.yaml` | `<run>/params/env.yaml`, written at the start of training. Required: the viewer reads the gains, action scale and torque limits from it. |
| `agent.yaml` | `<run>/params/agent.yaml`, a record of the training settings |
| `README.md` | The model card. Its metadata must include `library_name: asimov`. |

```bash
RUN=logs/rsl_rl/asimov_velocity_amp/<run>
mkdir -p hub_upload
cp "$RUN/exported/policy.onnx" "$RUN/params/env.yaml" "$RUN/params/agent.yaml" hub_upload/
cat > hub_upload/README.md <<'EOF'
---
library_name: asimov
pipeline_tag: robotics
license: bsd-3-clause
---

# Asimov 1 locomotion policy

Trained with [isaac_asimov](https://github.com/menloresearch/isaac_asimov).
EOF

pip install -U huggingface_hub   # provides the hf command
hf auth login
hf upload <your-username>/<repo-name> hub_upload .
```

Then try it in the viewer:

```bash
git clone https://github.com/menloresearch/humanoid-policy-viewer
cd humanoid-policy-viewer
npm run hf <your-username>/<repo-name>
```

The viewer runs policies with the default observations and 23 joint
actions configured in this repo. See its
[supported policies](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/huggingface.md#supported-policies)
doc before changing the observation or action terms.

## Troubleshooting
The training code has been tested on the following GPUs:
- NVIDIA RTX A6000
- NVIDIA RTX PRO 6000
- NVIDIA RTX 4090
- NVIDIA RTX 3090

## Acknowledgement

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Providing powerful simulation functionalities.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
- [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP): Referenced for AMP-based motion imitation.
- [mjlab](https://github.com/mujocolab/mjlab): MuJoCo-based training utilities and references.

## Community

We're planning community livestreams where we’ll test policies
contributed by developers on the real Asimov 1. [Join the community
to share your work](https://discord.gg/3wTVbHabtn), discuss experiments, and hear about upcoming sessions.
