# Playing a checkpoint

`./cyclotron.sh --play` runs a training checkpoint live in Isaac Sim so you can
watch and evaluate it. It is the tool of the person training: a quick look at
the latest checkpoint while a run is still going, or a closer evaluation after
it finishes.

## Quick start

Play the latest checkpoint of the latest run:

```bash
./cyclotron.sh --play \
    --task Asimov1-Velocity-AMP-Play-v0 --num_envs 32
```

Use the `-Play-v0` variant of the task the run was trained with:
`Asimov1-Velocity-AMP-Play-v0` for `Asimov1-Velocity-AMP-v0`,
`Asimov1-Velocity-Play-v0` for `Asimov1-Velocity-v0`. The play variants keep
the policy's inputs identical but make the environment watchable: 32
environments, effectively infinite episodes, observation corruption off, no
pushes, no randomization of initial joint positions, PD gains, base CoM or
foot friction, zero reset randomization of base pose and velocity, a narrowed
velocity command range (forward 0.6–0.8 m/s), and a small 5x5 cobblestone-road
terrain.

Three things to know before loading anything else:

- **Checkpoint selection.** Without `--checkpoint`, the latest checkpoint of
  the latest run of the task's experiment is played. Pass `--checkpoint` with
  a full path to a `.pt` file, or a filename such as `model_500.pt` together
  with `--load_run <run>`. `--target` is an alias for `--checkpoint`.
- **Play writes no ONNX.** It runs the checkpoint itself; use
  [`--export`](export.md) to produce `policy.onnx`. The removed flags
  `--export-only`, `--onnx-output` and `--onnx-filename` are refused before
  Isaac Sim starts, with a pointer to `--export`.
- **Code changes since training.** Before loading the checkpoint, play prints
  what changed in the code since the run was trained; see
  [Code changes since training](../README.md#code-changes-since-training).
  Changes are warnings (`--strict` turns them into a stop). Two cases always
  stop: a checkpoint that no longer fits the network the current code builds,
  and a changed actor class or activation, which would load the old weights
  but compute something else. Only the actor is loaded; the critic and the
  AMP discriminator are training-only state.

## Configuration

Checkpoint and run selection:

| Flag | Meaning |
| --- | --- |
| `--task <id>` | The gym task to play. Use the `-Play-v0` variants. |
| `--checkpoint <path or file>` | Checkpoint to load: a full path to a `.pt` file, or a filename inside the `--load_run` folder. Default: the latest checkpoint. |
| `--target <path>` | Alias for `--checkpoint`. |
| `--load_run <run>` | Run folder to pick the checkpoint from. Default: the latest run. |
| `--experiment_name <name>` | Experiment folder under `logs/rsl_rl/` to look in. Default: the task's configured experiment. |
| `--use_pretrained_checkpoint` | Use the pre-trained checkpoint from Nucleus instead, where one is published for the task. |

Playback:

| Flag | Meaning |
| --- | --- |
| `--num_envs <n>` | Number of environments to simulate. Default: the task's (32 for the play variants). |
| `--real-time` | Sleep each step so the simulation runs at real-time speed, if possible. |
| `--video` | Record a video of the first `--video_length` steps (enables cameras). Written to `videos/play/` inside the run folder; play exits when the clip is done. |
| `--video_length <steps>` | Length of the recorded video in steps. Default: 200. |
| `--seed <n>` | Seed for the environment (`-1` picks a random one). |
| `--strict` | Stop, instead of warning, if the code changed since the run was trained. |
| `--disable_fabric` | Disable fabric and use USD I/O operations. |
| `--agent <entry point>` | RL agent configuration entry point. Default: `rsl_rl_cfg_entry_point`. |

Isaac Sim's AppLauncher flags also apply, most usefully `--headless` (no
viewer window; combine with `--video`) and `--device` (e.g. `cuda:1`). The
remaining RSL-RL flags (`--run_name`, `--resume`, `--logger`,
`--log_project_name`) are accepted for config compatibility but matter for
`--train`, not here.

Unknown flags are passed to Hydra as config overrides, so a trained value a
code-change warning names (for example
`env.observations.policy.base_ang_vel.scale=0.25`) can be passed back on the
command line.

## Advanced examples

Play a specific old checkpoint by full path:

```bash
./cyclotron.sh --play \
    --task Asimov1-Velocity-AMP-Play-v0 \
    --checkpoint logs/rsl_rl/asimov_velocity_amp/2026-09-30_14-12-03/model_1500.pt
```

Record a 400-step clip without opening a window:

```bash
./cyclotron.sh --play \
    --task Asimov1-Velocity-AMP-Play-v0 \
    --headless --video --video_length 400
```

Watch at real-time speed instead of as fast as the GPU allows:

```bash
./cyclotron.sh --play \
    --task Asimov1-Velocity-Play-v0 --real-time
```

Run strictly, refusing any code change since the run was trained:

```bash
./cyclotron.sh --play \
    --task Asimov1-Velocity-AMP-Play-v0 \
    --load_run 2026-09-30_14-12-03 --checkpoint model_500.pt \
    --strict
```
