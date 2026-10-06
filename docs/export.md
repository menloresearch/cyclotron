# Exporting policies to ONNX

`./cyclotron.sh --export` turns a training checkpoint into a folder that can be
run in the [policy viewer](../README.md#view-a-policy), shared on the Hugging
Face Hub with [`--share`](../README.md#share-your-policy), or deployed on the
robot:

```bash
./cyclotron.sh --export \
    --checkpoint logs/rsl_rl/<experiment_name>/<run>/model_<n>.pt
```

## What an export contains

The export is written to `exported/` next to the checkpoint (`--output
<folder>` writes somewhere else):

| File | Contents |
| --- | --- |
| `policy.onnx` | The policy, exported by rsl_rl's own exporter, with the deploy metadata below attached. |
| `policy.pt` | The same policy as TorchScript. |
| `env.yaml`, `agent.yaml` | Unchanged copies of the run's `params/`: the task and training settings. |
| `code_state.yaml` | The code the run was trained with, for runs whose training recorded one. |

The ONNX graph is whatever rsl_rl's exporter produces for the actor class: a
plain MLP is `obs -> actions`, an LSTM or GRU carries its state as extra
inputs and outputs. The observation normalizer and the deterministic output
head are baked into the graph.

## What `--export` does

1. **Resolves the checkpoint and task.** A full `--checkpoint` path is enough
   for runs trained here: the task is read from the run's `agent.yaml`.
   Otherwise pass `--task`, and pick the checkpoint like `--play` does
   (`--load_run`, `--experiment_name`, or the latest run by default).
2. **Checks the run against the current code.** Before loading anything it
   compares the run's saved `env.yaml`, `agent.yaml` and `code_state.yaml`
   with the code you have checked out and prints what changed; see
   [Code changes since training](../README.md#code-changes-since-training).
   Changes are warnings (`--strict` turns them into a stop); a policy network
   whose loaded weights would compute something else is always a stop.
3. **Loads only the actor.** The critic, the AMP discriminator and the
   optimizer are training-only state, so a run whose critic no longer matches
   the current code still exports.
4. **Exports and bundles.** Writes `policy.onnx` and `policy.pt` with rsl_rl's
   exporter and copies the run's yaml files next to them.
5. **Attaches the deploy metadata** described below to `policy.onnx`.
6. **Checks the export.** Feeds the same observations to the checkpoint and to
   `policy.onnx` (as attached, through onnxruntime) and fails if any action
   differs by more than 1e-4, which would mean a broken export, for example a
   dropped observation normalizer. Recurrent policies skip this check.

## Deploy metadata

`policy.onnx` carries a small deployment contract in the ONNX file's
`metadata_props` (string key-value pairs, readable with `onnx` or any
onnxruntime binding's model-metadata API). It exists so a robot runtime can
check its assumptions about the file before driving a robot with it.

It is a handshake, not configuration. The runtime should compare these values
against its own settings and refuse the policy on any mismatch, never
configure itself from them: a wrong or tampered file must not be able to
retune a robot. Torque limits and the actuator model are deliberately absent;
they belong to the runtime's own configuration (and to `env.yaml`). Runtimes
that read no metadata run the file unchanged, since the graph is untouched.

| Key | Value (all stored as strings) | Meaning |
| --- | --- | --- |
| `deploy_metadata_version` | `"1"` | Schema version. Refuse versions you do not know. |
| `obs_dim`, `action_dim` | int | Width of the graph's `obs` input and `actions` output, read from the graph itself. |
| `joint_names` | JSON list of `action_dim` strings | The joint each action drives, in action order. The single most safety-critical check: a joint-order mismatch silently scrambles the robot. |
| `action_scale`, `action_offset` | JSON lists of `action_dim` floats | The affine that turns a raw action into a position target: `target = action * scale + offset`. The offset is the trained default pose for tasks that use `use_default_offset`. |
| `action_clip` | JSON: `null`, or `action_dim` pairs `[low, high]` (`null` for an unclipped side) | Clip applied to the targets after the affine. |
| `joint_stiffness`, `joint_damping` | JSON lists of `action_dim` floats | The PD gains (kp/kd) the targets were trained to be tracked with, from the task's configured actuators (not the simulated values, which randomization events can perturb). |
| `sim_dt`, `decimation`, `policy_rate_hz` | float, int, float | The policy step: it was trained to act every `sim_dt x decimation` seconds. |
| `observation_names` | JSON list of strings | The policy's input terms in order, as named in `env.yaml` (`<group>/<term>` when the actor reads several groups). The per-term recipe stays in `env.yaml`. |
| `trained_commit` | string, absent when unknown | The commit the run was trained with, from `code_state.yaml` (or the run's `git/` records), for traceability. |

The metadata is resolved from the live environment at export time, so patterns
in the config (joint regexes, per-joint scales) arrive as concrete per-joint
values. It is attached for manager-based tasks whose action terms are joint
actions; for anything else (direct-workflow envs, non-joint action terms)
`--export` prints a warning and writes the file without metadata rather than
writing a wrong contract.

## Exporting a model that was not trained with cyclotron

`--export` works for any Isaac Lab run trained with rsl_rl, not only this
repo's tasks. It needs three things:

1. **The run directory layout Isaac Lab writes**: the checkpoint
   (`model_<n>.pt`) with `params/agent.yaml` and `params/env.yaml` next to it
   (Isaac Lab's `train.py` writes these). The yaml copies and the
   code-change check come from there.
2. **The task, passed explicitly**: task inference only knows this repo's
   experiments, so pass the gym id the run was trained with, e.g.

   ```bash
   ./cyclotron.sh --export --task Isaac-Velocity-Rough-Anymal-C-v0 \
       --checkpoint /path/to/run/model_1500.pt
   ```

   Every task that `isaaclab_tasks` registers is available. A task from a
   third-party extension is not: `export.py` only imports `isaaclab_tasks`
   and `cyclotron.tasks`, so a task registered by another package would need
   that package imported (install it and add the import) before `gym.make`
   can find it.
3. **An rsl_rl checkpoint this rsl_rl version can read.** The pinned
   `rsl-rl-lib` (see `source/cyclotron/setup.py`) loads its own layout
   (`actor_state_dict` etc.); checkpoints from recent earlier versions are
   converted on the fly by Isaac Lab's compatibility shim. If loading fails
   with a layout error, re-export from the rsl_rl version that trained it.

What to expect with a foreign run:

- The code-change check cannot find `params/code_state.yaml` and falls back
  to the `git/*.diff` records rsl_rl wrote. Commits from another repository
  are not in this clone, so it reports that changes "can't be listed" — a
  warning, and the export proceeds (`--strict` would stop on it).
- The deploy metadata is attached as long as the task is manager-based with
  joint actions, Anymal and friends included; `trained_commit` is only
  present when the run recorded a single commit.
- The export check (ONNX vs checkpoint) runs the same way as for local runs.
- [`--view`](../README.md#view-a-policy) is Asimov-specific: the viewer
  simulates the Asimov 1 robot, so a policy for another robot exports fine
  but cannot be watched there. Use the robot's own tooling, or Isaac Lab's
  `play.py`, to see it move.
