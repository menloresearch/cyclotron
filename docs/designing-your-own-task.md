# Designing your own task

You have trained `Asimov1-Velocity-v0` once with [`--train`](train.md). This
tutorial walks through making a task of your own. The worked example is a
**slow-walk variant**: the same robot and rewards, but commands capped at
walking speed and a stronger posture reward. The same steps apply to any
variant you want to train.

Before changing anything, read [the constraints](#constraints-that-keep-the-pipeline-working)
at the end. They are what keep `--export`, `--view` and the robot runtime
working with your new task.

## Where a task lives

A task is five pieces, all small:

| Piece | Example | File |
| --- | --- | --- |
| Environment config | `Asimov1VelocityEnvCfg` | `source/cyclotron/cyclotron/tasks/locomotion/velocity_env_cfg.py` |
| Play variant | `Asimov1VelocityEnvCfg_PLAY` | same file |
| Agent (runner) config | `Asimov1PPORunnerCfg` | `source/cyclotron/cyclotron/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` |
| Gym registration | `Asimov1-Velocity-v0`, `-Play-v0` | `source/cyclotron/cyclotron/tasks/locomotion/__init__.py` |
| Experiment-to-task entry | `EXPERIMENT_TASKS` | `source/cyclotron/cyclotron/hub.py` |

The env config says what the robot sees, does and is rewarded for. The runner
config says how PPO trains it. The gym ids tie the two together, and the
`EXPERIMENT_TASKS` entry lets `--export` and `--share` find the task from a
checkpoint alone.

## 1. Subclass the env config

Create your variant as a subclass of `Asimov1VelocityEnvCfg`. For a small
variant, put it in `velocity_env_cfg.py` below the classes it extends; a
bigger task deserves its own `<name>_env_cfg.py` next to it (that is exactly
what `amp_env_cfg.py` is).

```python
@configclass
class Asimov1SlowWalkEnvCfg(Asimov1VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Walking speed only.
        self.commands.twist.ranges.lin_vel_x = (-0.3, 0.4)
        self.commands.twist.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.twist.ranges.ang_vel_z = (-0.5, 0.5)
        # Hold the pose more tightly at low speed.
        self.rewards.pose.weight = 2.0
```

Make the changes in `__post_init__`, after `super().__post_init__()`.
Isaac Lab's `@configclass` resolves the config tree after construction, so
`__post_init__` is where you get fully built sub-configs you can edit in
place: `self.commands.twist` is the `UniformVelocityCommandCfg` from
`CommandsCfg`, `self.rewards.pose` the `RewTerm` from `RewardsCfg`. Overriding
class attributes instead would replace whole groups and silently drop the base
task's tuning.

Anything in the base config is fair game the same way:

- `self.rewards.<term>.weight` or `.params[...]` to retune a reward, or
  `self.rewards.<term> = None` to drop it.
- `self.events.<term> = None` to drop a randomization
  (`Asimov1VelocityEnvCfg_PLAY` does this for `push_robot` and friends).
- `self.scene.terrain.terrain_generator` for different ground.
- A new `RewTerm` assigned to a fresh attribute to add a reward.

## 2. Add the play variant

Every task needs a play twin: fewer envs, no randomization, deterministic
resets, so [`--play`](play.md) shows the policy instead of the curriculum.
Subclass your variant and mirror what `Asimov1VelocityEnvCfg_PLAY` disables —
or inherit from it directly if your changes and its changes compose:

```python
@configclass
class Asimov1SlowWalkEnvCfg_PLAY(Asimov1SlowWalkEnvCfg, Asimov1VelocityEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # _PLAY resets the command ranges for the stock task; restore ours.
        self.commands.twist.ranges.lin_vel_x = (0.2, 0.4)
```

With this ordering Python runs `Asimov1VelocityEnvCfg_PLAY.__post_init__`
(32 envs, corruption off, pushes and randomization events removed, zeroed
resets, a small terrain) on top of your variant. Check what the play base
sets against what you changed: it overwrites `commands.twist.ranges`, so the
example puts the slow ranges back. If the interaction gets confusing, skip
the double inheritance and copy the `_PLAY` body into a plain subclass of
your task.

## 3. Add a runner config with a new experiment name

In `agents/rsl_rl_ppo_cfg.py`, subclass `Asimov1PPORunnerCfg` and give it a
**new** `experiment_name`:

```python
@configclass
class Asimov1SlowWalkPPORunnerCfg(Asimov1PPORunnerCfg):

    experiment_name = "asimov1_slow_walk"
    run_name = "ppo"
```

The experiment name matters more than it looks:

- Every run of the task lands in `logs/rsl_rl/<experiment_name>/`. Reusing
  `asimov1_velocity` would interleave your runs with the stock task's.
- `--export` and `--share` infer the task from the `experiment_name` saved in
  a run's `agent.yaml`, via `EXPERIMENT_TASKS` (step 5). A shared name maps
  two tasks to one key, and the inference picks the wrong one.

Network sizes, PPO hyperparameters and `max_iterations` are inherited; change
them here if the variant needs it.

## 4. Register both gym ids

In `source/cyclotron/cyclotron/tasks/locomotion/__init__.py`, register the
train and play ids, following the existing entries:

```python
gym.register(
    id="Asimov1-SlowWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Asimov1SlowWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1SlowWalkPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-SlowWalk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Asimov1SlowWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1SlowWalkPPORunnerCfg",
    },
)
```

The entry points are strings, so a typo here surfaces only when the task is
first made — the smoke train in step 6 catches it.

## 5. Map the experiment to the task

In `source/cyclotron/cyclotron/hub.py`, add your experiment name to
`EXPERIMENT_TASKS`:

```python
EXPERIMENT_TASKS = {
    "asimov1_velocity": "Asimov1-Velocity-v0",
    "asimov_velocity_amp": "Asimov1-Velocity-AMP-v0",
    "asimov1_slow_walk": "Asimov1-SlowWalk-v0",
}
```

This is how `--export` and `--share` turn a bare `--checkpoint` path into the
right task. Without the entry they stop with "Cannot infer the task" and you
have to pass `--task Asimov1-SlowWalk-v0` by hand every time.

## 6. Smoke-train it

Before committing a GPU for hours, run a short training to shake out config
errors:

```bash
./cyclotron.sh --train --task Asimov1-SlowWalk-v0 \
    --headless --num_envs 256 --max_iterations 100
```

This catches bad entry-point strings, term params that do not resolve, and
shape mismatches, and writes a first run under
`logs/rsl_rl/asimov1_slow_walk/`. Reward curves after 100 iterations mean
little; a crash means everything. See [train.md](train.md) for the full
training workflow and multi-GPU runs.

## 7. Play, export, view

The rest of the pipeline now works unchanged, because the pieces from steps
3-5 tell it where everything is:

```bash
# Watch the checkpoint in Isaac Sim (uses Asimov1-SlowWalk-Play-v0).
./cyclotron.sh --play --task Asimov1-SlowWalk-Play-v0 --num_envs 16

# Export the latest checkpoint; the task is inferred from agent.yaml.
./cyclotron.sh --export \
    --checkpoint logs/rsl_rl/asimov1_slow_walk/<run>/model_100.pt

# Run the export in the browser viewer.
./cyclotron.sh --view logs/rsl_rl/asimov1_slow_walk/<run>
```

Each stage has its own doc: [play.md](play.md), [export.md](export.md),
[view.md](view.md).

## Writing new MDP terms

When the built-in terms are not enough, write your own under
`source/cyclotron/cyclotron/tasks/locomotion/mdp/`:

- `observations.py` for observation terms,
- `rewards.py` for reward terms,
- `events.py` for randomization and reset events.

All three are re-exported through `mdp/__init__.py` alongside everything from
`isaaclab.envs.mdp`, so a term you add is immediately available in the config
as `mdp.<your_term>` — same namespace as `mdp.joint_pos_rel` or
`mdp.time_out`.

A term is a function taking the env first and its `params` as keyword
arguments, returning one row per env. Copy the signature pattern from an
existing term, for example `track_linear_velocity` in `rewards.py`:

```python
def track_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ...
```

Rewards return a float tensor of shape `(num_envs,)`; observations return
`(num_envs, dim)` (see `foot_height` in `observations.py`). Everything comes
from `env`: assets via `env.scene[asset_cfg.name]`, sensors via
`env.scene.sensors[sensor_name]`, commands via `env.command_manager`. A term
that needs per-env state across steps is a class subclassing
`ManagerTermBase` with `__init__`, `reset` and `__call__` — `delayed_obs` in
`observations.py` is the template.

## Constraints that keep the pipeline working

A task config can change almost anything, but four things are load-bearing
for everything downstream of training. Break them knowingly or not at all.

1. **Keep `sim.dt x decimation = 0.02 s`.** `Asimov1VelocityEnvCfg.__post_init__`
   sets `decimation = 4` and `sim.dt = 0.005`: the policy acts at 50 Hz. The
   robot and the [viewer](view.md) run policies at 50 Hz, `--export` attaches
   the rate to `policy.onnx` as deploy metadata, and the viewer refuses other
   rates. Change the product only for a policy that will never leave Isaac Sim.
2. **Keep policy observations viewer-computable if `--view` should work.**
   The viewer recomputes the policy group's terms from its own MuJoCo
   simulation, from the `observation_names` and recipes in the export. Stick
   to terms it supports, and leave `clip` and `modifiers` off policy
   `ObsTerm`s (noise, `scale` and the stock terms are fine — the training
   config already uses them). The supported list lives in the viewer's
   [supported-policies doc](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/huggingface.md#supported-policies);
   see [view.md](view.md). An unsupported policy group still trains, plays
   and deploys — it just cannot be watched in the browser.
3. **Keep `actions.joint_pos` as it is.** `preserve_order=True` makes the
   action order the declared `ASIMOV_1_JOINT_NAMES` order, and
   `use_default_offset=True` makes raw actions offsets from the default pose
   — both are baked into the exported `joint_names`, `action_scale` and
   `action_offset` metadata the robot runtime handshakes against. Reordering
   or re-centering actions silently scrambles a deployed robot.
4. **Changing the policy's observations or actions changes what existing
   checkpoints are compatible with.** A checkpoint stores the config it was
   trained with; after you edit the task, `--play` and `--export` diff the
   run's saved config against your checkout and name exactly what changed —
   see [Code changes since training](../README.md#code-changes-since-training).
   That is the reason to make a new task instead of editing
   `Asimov1VelocityEnvCfg` in place: old runs keep their config, your variant
   gets its own.

## AMP variants

An AMP version of your task is the same recipe with one more observation
group and a different runner base:

1. Subclass your env config and add an `amp` group in `__post_init__`,
   exactly as `Asimov1AmpEnvCfg` does:

   ```python
   @configclass
   class Asimov1SlowWalkAmpEnvCfg(Asimov1SlowWalkEnvCfg):
       def __post_init__(self):
           super().__post_init__()
           self.observations.amp = AmpObsCfg()
   ```

   `AmpObsCfg` (in `amp_env_cfg.py`) is what the AMP discriminator sees:
   `joint_pos` and `joint_vel` over all joints, order-preserved. Its
   `__post_init__` asserts the term names against `ASIMOV_1_AMP_OBS_TERMS`,
   because the expert features in the motion dataset are built with the same
   term list — if you change the group, change `ASIMOV_1_AMP_OBS_TERMS` and
   the dataset config together, or the discriminator compares misaligned
   features.
2. Base the runner on `Asimov1AMPRunnerCfg` instead of `Asimov1PPORunnerCfg`
   (again with a new `experiment_name`). It swaps the algorithm for
   `AMPPPOAlgorithmCfg`, whose `amp_data` points the `MotionDatasetCfg` at
   the reference motion files; give it your own motions if the stock
   slow-walk clip is not the style you want.
3. Register the ids and add the `EXPERIMENT_TASKS` entry as before —
   `Asimov1-Velocity-AMP-v0` and `asimov_velocity_amp` are the pattern to
   follow.
