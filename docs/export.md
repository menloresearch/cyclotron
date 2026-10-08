# Exporting policies to ONNX

`./cyclotron.sh --export` turns a training checkpoint into a folder that can be
run in the [policy viewer](view.md), shared on the Hugging
Face Hub with [`--share`](share.md), or deployed on the
robot:

## Quick start

```bash
./cyclotron.sh --export \
    --checkpoint logs/rsl_rl/<experiment_name>/<run>/model_<n>.pt
```

One command is enough for a run trained here: the task is read from the run's
`agent.yaml`. The export lands in `exported/` next to the checkpoint; watch it
with [`--view`](view.md) or publish it with [`--share`](share.md).

## What an export contains

The export is written to `exported/` next to the checkpoint (`--output
<folder>` writes somewhere else):

| File | Contents |
| --- | --- |
| `policy.onnx` | The policy, exported by rsl_rl's own exporter, with the deploy metadata below attached. |
| `policy.pt` | The same policy as TorchScript. |
| `env.yaml`, `agent.yaml` | Unchanged copies of the run's `params/`: the task and training settings. |
| `code_state.yaml` | The code the run was trained with, for runs whose training recorded one. |
| `export.log` | Append-only history of every export written here: which checkpoint, when, and what it printed. |

The ONNX graph is whatever rsl_rl's exporter produces for the actor class: a
plain MLP is `obs -> actions`, an LSTM or GRU carries its state as extra
inputs and outputs. The observation normalizer and the deterministic output
head are baked into the graph.

## What `--export` does

1. **Resolves the checkpoint and task.** A full `--checkpoint` path is enough
   for runs trained here: the task is read from the run's `agent.yaml`.
   Otherwise pass `--task`, and pick the checkpoint like `--play` does
   (`--load_run`, `--experiment_name`, or the latest run by default).
2. **Rebuilds the run's policy settings.** Like a restart of the run, it sets
   what the policy sees and does back to the values in the run's `env.yaml` and
   `agent.yaml` before building the environment: the actor's observation terms
   (functions, scales, clipping, history), the actions, the robot's default
   pose and actuators, `sim.dt`, decimation and the actor network (an LSTM run
   exports as an LSTM even if the task now trains an MLP). The policy and its
   deploy metadata therefore describe the run as it was trained, whatever the
   task's config says today. The two files are the only source of these
   settings: export doesn't run Hydra, and refuses setting overrides
   (`env.…=`, `agent.…=`) before Isaac Sim starts. Everything else (rewards,
   terrain, events, commands, the robot model file) stays as the code has it:
   it only shapes training, or depends on the machine. If the current code
   can't take a saved setting (an observation term or actuator group it no
   longer has, or a function it can't find), the export stops and names it.

   Training records the sha256 of both files in `code_state.yaml`. If either
   no longer matches, because it was edited by hand or deleted, the export
   stops: the files are the record of how the run was trained. Runs trained
   before the hashes were recorded say so in one line, since edits to them
   can't be detected.

   A run without `env.yaml` or `agent.yaml` (one trained elsewhere, or with
   the file deleted) gets a `[y/N]` question in the terminal: yes writes the
   missing file into the run's `params/` from the current code, with a first
   line saying it was generated, from which commit and when; no, or no answer,
   stops the export. Later exports of that run warn that the file came from
   the code, and `--strict` (so `--share`) stops on such a run instead of
   asking.
3. **Checks the rebuilt settings against the run.** Once the environment is
   built, it compares the policy settings with the run's `env.yaml` and
   `agent.yaml` again. Any difference means the rebuild failed and stops the
   export. A setting the current code has but the run didn't save (added since
   training) keeps the code's value and is listed as a warning; `--strict`
   stops on it, since it doesn't come from the run.
4. **Reports changes to the code behind the settings.** It compares the run's
   `code_state.yaml` (or rsl_rl's `git/` records) with the code you have
   checked out, but only what can change the exported policy: the cyclotron
   files that define the functions and classes the policy settings name (the
   file a function is defined in, not helpers it imports), the robot model,
   Isaac Lab's commit and the `isaacsim`, `isaaclab`, `rsl-rl-lib` and `torch`
   versions. Rewards, the training algorithm, configs and docs can't change an
   export and aren't listed (`--play` still lists them; see
   [Code changes since training](../README.md#code-changes-since-training)).
   Changes are warnings, with the commit to check out for the exact training
   code; `--strict` turns them into a stop. Export doesn't rebuild the
   training code itself: Isaac Lab and rsl_rl are installed packages a
   checkout can't bring back, so it names what changed and leaves the call to
   you.
5. **Loads only the actor.** The critic, the AMP discriminator and the
   optimizer are training-only state, so a run whose critic no longer matches
   the current code still exports. A checkpoint whose weights don't fit the
   network built from the run's settings, for example because an observation
   function now returns more values, stops the export.
6. **Exports and bundles.** Writes `policy.onnx` and `policy.pt` with rsl_rl's
   exporter and copies the run's yaml files next to them.
7. **Attaches the deploy metadata** described below to `policy.onnx`.
8. **Checks the export.** Feeds the same observations to the checkpoint and to
   `policy.onnx` (as attached, through onnxruntime) and fails if any action
   differs by more than 1e-4, which would mean a broken export, for example a
   dropped observation normalizer. A recurrent policy (LSTM or GRU) is run for
   3 steps from an empty memory on both sides, the ONNX file getting zeros as
   `h_in`/`c_in` and then its own `h_out`/`c_out` back, as a robot runtime
   does; the memory it returns is compared too. One step would not be enough:
   from an empty memory, the weights that carry memory between steps multiply
   zeros.

rsl_rl's own exporter checks nothing: it traces the actor once on zero inputs
and writes the file. Every check above, and the deploy metadata, are
cyclotron's; rsl_rl's only safeguard is PyTorch's strict weight loading, which
step 5 turns into a message naming the layers and sizes that don't fit.

## Options

| Flag | Meaning |
| --- | --- |
| `--task` | Task used to build the policy. Inferred from the run's `agent.yaml` when `--checkpoint` is a full path, or from `--experiment_name`. |
| `--checkpoint` | Checkpoint to export: a full path to a `.pt` file, or a filename inside the `--load_run` folder. |
| `--load_run` | Run folder to export from. Defaults to the latest. |
| `--experiment_name` | Experiment folder under `logs/rsl_rl/`. Defaults to the task's. |
| `--output` | Folder to write to. Defaults to `<checkpoint folder>/exported`. |
| `--strict` | Stop, instead of warning, if code that can change the exported policy changed since the run was trained (step 4) or the code has policy settings the run didn't save, and stop instead of asking when the run has no `env.yaml` or `agent.yaml` (or has one an earlier export generated). |
| `--device` | Device to run the export on (an AppLauncher flag; the export always runs headless). |

Nothing else is accepted: setting overrides such as `env.actions.joint_pos.scale=0.3` are refused, since the run's
`env.yaml` and `agent.yaml` are the only source of its settings.

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
| `action_scale`, `action_offset` | JSON lists of `action_dim` floats | The affine that turns a raw action into a position target: `target = action * scale + offset`. The offset is the run's default pose (`init_state.joint_pos` in its `env.yaml`) for tasks that use `use_default_offset`, without the per-environment jitter that `randomize_joint_default_pos` adds during training. |
| `action_clip` | JSON: `null`, or `action_dim` pairs `[low, high]` (`null` for an unclipped side) | Clip applied to the targets after the affine. |
| `joint_stiffness`, `joint_damping` | JSON lists of `action_dim` floats | The PD gains (kp/kd) the targets were trained to be tracked with, from the run's configured actuators (not the simulated values, which randomization events can perturb). |
| `sim_dt`, `decimation`, `policy_rate_hz` | float, int, float | The policy step: it was trained to act every `sim_dt x decimation` seconds. |
| `observation_names` | JSON list of strings | The policy's input terms in order, as named in `env.yaml` (`<group>/<term>` when the actor reads several groups). The per-term recipe stays in `env.yaml`. |
| `trained_commit` | string, absent when unknown | The commit the run was trained with, from `code_state.yaml` (or the run's `git/` records), for traceability. |

The metadata is resolved from the live environment at export time, so patterns
in the config (joint regexes, per-joint scales) arrive as concrete per-joint
values. It is attached for manager-based tasks whose action terms are joint
actions; for anything else (direct-workflow envs, non-joint action terms)
`--export` prints a warning and writes the file without metadata rather than
writing a wrong contract.

### Inspecting the metadata

Print every key of an exported file with the `onnx` package, which the
cyclotron environment already has:

```bash
python -c '
import json, sys, onnx
model = onnx.load(sys.argv[1], load_external_data=False)
for prop in model.metadata_props:
    try:
        value = json.loads(prop.value)
    except ValueError:
        value = prop.value
    print(f"{prop.key}: {value}")
' logs/rsl_rl/<experiment>/<run>/exported/policy.onnx
```

A file without metadata (e.g. one exported by an older cyclotron, or with a
non-joint action term) prints nothing. On the robot side, onnxruntime reads the
same map without the `onnx` package, which is how a runtime would check it:

```python
import onnxruntime as ort

metadata = ort.InferenceSession("policy.onnx").get_modelmeta().custom_metadata_map
```

[Netron](https://netron.app) also lists the keys under the model's properties
when you open the file.

## Advanced: exporting a model that was not trained with cyclotron

`--export` works for any Isaac Lab run trained with rsl_rl, not only this
repo's tasks. It needs three things:

1. **The run directory layout Isaac Lab writes**: the checkpoint
   (`model_<n>.pt`) with `params/agent.yaml` and `params/env.yaml` next to it
   (Isaac Lab's `train.py` writes these). The policy settings, the yaml
   copies and the code-change check come from there.
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
- The policy settings are rebuilt from the run's `env.yaml` and `agent.yaml`
  as for local runs. A function the run used from a package that isn't
  installed here, and that this checkout has under no other module, stops the
  export.
- The deploy metadata is attached as long as the task is manager-based with
  joint actions, Anymal and friends included; `trained_commit` is only
  present when the run recorded a single commit.
- The export check (ONNX vs checkpoint) runs the same way as for local runs.
- [`--view`](view.md) is Asimov-specific: the viewer
  simulates the Asimov 1 robot, so a policy for another robot exports fine
  but cannot be watched there. Use the robot's own tooling, or Isaac Lab's
  `play.py`, to see it move.
