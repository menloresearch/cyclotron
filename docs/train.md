# Training policies

`./cyclotron.sh --train` trains an Asimov-1 locomotion policy in Isaac Lab
with RSL-RL, as PPO or as AMP. It is the first stage of the pipeline: the
checkpoints it writes are what [`--play`](play.md), [`--export`](export.md),
[`--share`](share.md) and [`--view`](view.md) consume.

## Quick start

Run a short test first to check the full pipeline. 128 environments and 100
iterations finish in about 10 minutes on a 4090:

```bash
./cyclotron.sh --train \
    --task Asimov1-Velocity-AMP-v0 --num_envs 128 --headless --max_iterations 100
```

The baseline policy is trained with AMP on 4096 environments:

```bash
./cyclotron.sh --train \
    --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless
```

If you hit an out-of-memory error, lower `--num_envs`; the policy may then
need more iterations to converge.

Two tasks ship with the repo:

| Task | Algorithm | Experiment name |
| --- | --- | --- |
| `Asimov1-Velocity-AMP-v0` | PPO + adversarial motion priors (recommended) | `asimov_velocity_amp` |
| `Asimov1-Velocity-v0` | Plain PPO | `asimov1_velocity` |

The stock tasks are the starting point, not the destination: the point of
this repo is to define your own task and train a policy for it.
[Designing your own task](designing-your-own-task.md) walks through building
one from the velocity task.

## Configuration

Every flag of `scripts/rsl_rl/train.py`, all optional:

| Flag | Meaning |
| --- | --- |
| `--task <id>` | The gym task to train, from the table above or one you registered. |
| `--num_envs <n>` | Number of parallel environments. Defaults to the task's setting; the main lever for GPU memory. |
| `--max_iterations <n>` | Training iterations. Both stock tasks default to 10000. |
| `--seed <n>` | RNG seed for the environment and agent. `-1` picks a random seed. |
| `--headless` | Run Isaac Sim without a window. Use it for every real run. |
| `--device <dev>` | Simulation and training device, e.g. `cuda:1`. Defaults to the task's setting. |
| `--video` | Record rollout clips during training (implies cameras, see below). |
| `--video_length <n>` | Length of each clip in steps (default 200). |
| `--video_interval <n>` | Steps between clips (default 2000). |
| `--experiment_name <name>` | Log folder under `logs/rsl_rl/`. Defaults to the task's experiment name. |
| `--run_name <name>` | Suffix appended to the timestamped run directory. |
| `--resume` | Resume from a checkpoint (see below). |
| `--load_run <run>` | The run folder to resume from. Without it, the latest run. |
| `--checkpoint <path>` | Checkpoint to start from: a full path to a `.pt` file, or a filename such as `model_500.pt` together with `--load_run`. Implies `--resume`. |
| `--logger {tensorboard,wandb,neptune}` | Logger module. Defaults to the task's setting (tensorboard). |
| `--log_project_name <name>` | Project name for wandb or neptune. |
| `--distributed` | Multi-GPU training via `torch.distributed.run` (see below). Needs a GPU device. |
| `--agent <entry_point>` | RL agent configuration entry point (default `rsl_rl_cfg_entry_point`). |
| `--export_io_descriptors` | Export the environment's IO descriptors alongside the run. |
| `--ray-proc-id <n>` | Set automatically by the Ray integration; leave it alone. |

`--headless` and `--device` come from Isaac Lab's `AppLauncher`, which adds
further simulator flags; `./cyclotron.sh --train --help` lists them all.

### Hydra overrides

Unknown arguments pass through to Hydra, so any value in the task's env or
agent config can be overridden by its dotted path:

```bash
./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless \
    env.observations.policy.base_ang_vel.scale=0.5 agent.max_iterations=2000
```

These are the same paths the
[code-change check](../README.md#code-changes-since-training) prints, so a
difference it reports can be passed straight back on the command line.

### What a run writes

Each run gets its own directory, `logs/rsl_rl/<experiment_name>/<run>/`,
where `<run>` is a timestamp plus the run name:

| File | Contents |
| --- | --- |
| `model_<n>.pt` | Checkpoints, written every `save_interval` iterations (500 for the stock tasks) and at the end. |
| `params/env.yaml`, `params/agent.yaml` | The task and training settings the run actually used, overrides included. |
| `params/code_state.yaml` | Hashes of the code the run was trained with; what [`--export` and `--play` compare against](../README.md#code-changes-since-training). Also the sha256 of `env.yaml` and `agent.yaml`, so `--export` and `--share` refuse them once edited. |
| `git/` | rsl_rl's own records: the repo's commit and diff at training time. |
| `events.out.tfevents.*` | Tensorboard event files. Watch them with `tensorboard --logdir logs/rsl_rl/<experiment_name>`. |

### Resuming

Training prints the run directory and a ready-made resume command at the
start, and again when it ends or is interrupted. Resuming does not append to
the old run: it starts a **new** run directory, initialised from the
checkpoint, with its own logs and checkpoints. `--resume` alone picks the
latest checkpoint of the latest run; `--load_run` and `--checkpoint` narrow
that down.

## Advanced examples

### Multi-GPU training

Launch `train.py` through `torch.distributed.run` with `--distributed`.
`--num_envs` is per GPU; each rank gets its own seed:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/rsl_rl/train.py \
    --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless --distributed
```

### Resume or fine-tune from a specific checkpoint

A full `--checkpoint` path is enough; it implies `--resume`:

```bash
./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless \
    --checkpoint logs/rsl_rl/asimov_velocity_amp/<run>/model_5000.pt
```

To fine-tune under a different name, add `--run_name` (or
`--experiment_name` for a separate log folder):

```bash
./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless \
    --checkpoint logs/rsl_rl/asimov_velocity_amp/<run>/model_5000.pt \
    --run_name finetune
```

### Record training videos

`--video` records clips to `<run>/videos/train/` while training. It implies
`--enable_cameras`, so it costs speed; the defaults record 200 steps every
2000 steps:

```bash
./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless \
    --video --video_length 200 --video_interval 2000
```

### Log to Weights & Biases

```bash
./cyclotron.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless \
    --logger wandb --log_project_name asimov1-locomotion
```

Tensorboard event files are written either way.

### Override a config value with Hydra

Any dotted path from `env.yaml` or `agent.yaml` works as an override; here a
shorter run with a lower entropy bonus:

```bash
./cyclotron.sh --train --task Asimov1-Velocity-v0 --num_envs 4096 --headless \
    agent.max_iterations=3000 agent.algorithm.entropy_coef=0.002
```
