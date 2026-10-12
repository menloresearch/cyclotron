# Viewing policies in the browser

`./cyclotron.sh --view` runs an ONNX policy in your browser with
[humanoid-policy-viewer](https://github.com/menloresearch/humanoid-policy-viewer)
(MuJoCo + onnxruntime in WebAssembly): no Isaac Sim, no GPU, and — since the
physics is MuJoCo, not PhysX — a quick sim2sim check in one command. Anyone
can use it to try a policy, shared on the Hugging Face Hub or exported with
[`--export`](export.md), including people without a training machine.

## Quick start

View a policy from the Hub by its repo id:

```bash
./cyclotron.sh --view <org>/<model>
```

Or view a run you exported with [`--export`](export.md):

```bash
./cyclotron.sh --view logs/rsl_rl/<experiment_name>/<run>
```

The only requirement is [Node.js](https://nodejs.org) 20 or newer. The first
run fetches the `third_party/humanoid-policy-viewer` submodule, its npm
dependencies and the Asimov 1 robot model; after a pull that moves the
submodule, the next run updates it and its dependencies (unless it has
uncommitted changes, which are left alone with a warning). The viewer then checks the policy,
starts a local server and opens the browser with the policy selected (over SSH
or without a display it prints a link instead; see
[Advanced examples](#advanced-examples)). Use the sliders to command a
velocity and push the robot.

## Configuration

`--view` takes one argument, the policy, in two forms:

| Argument | What runs |
| --- | --- |
| `<org>/<model>` | A Hub repo, for example one uploaded with [`--share`](share.md). `policy.onnx`, `env.yaml` and `agent.yaml` are downloaded to `~/.cache/humanoid-policy-viewer/hf/` and checked. |
| A local path | Runs in place, with no download: a run folder (its `exported/`, written by [`--export`](export.md)), any folder holding `policy.onnx` and `env.yaml`, or an `.onnx` file. |

A run folder argument descends into `<run>/exported` when the folder itself
has no top-level `.onnx`, so for a fresh run, run `--export` first. A folder
without `env.yaml` next to the `.onnx` is refused: the viewer will not guess
the policy's gains. The policy's inputs, gains, action scale, default pose and
torque limits all come from `env.yaml`.

Options after the argument:

| Flag | Meaning |
| --- | --- |
| `--revision <ref>` | Branch, tag or commit of a Hub repo (default: `main`). Ignored for local paths. |
| `--port <n>` | Dev server port (default: 3000, or the next free one). |
| `--no-open` | Do not open a browser window. Added automatically over SSH or without a display. |

Private or gated Hub repos need `HF_TOKEN` set.

### What the viewer runs, and what it refuses

The viewer builds the policy's input term by term from `observations.policy`
in `env.yaml`. Recurrent policies (an rsl_rl LSTM or GRU) and stacked-history
policies run as long as every term is computable. The output is fixed by the
robot: 23 joint position targets at 50 Hz. A policy the viewer would run
wrongly is refused, naming the difference, and `--view` warns about it before
the server starts:

- an observation term the viewer can't compute, such as base linear velocity,
  foot contacts or a height scan, or a term with `clip` or `modifiers` set
- a policy trained at another rate than 50 Hz (`sim.dt` x `decimation` in
  `env.yaml`)
- actions sent to the joints in another order, without `preserve_order` or
  `use_default_offset`, or clipped

The viewer's
[supported policies](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/huggingface.md#supported-policies)
list every term it can compute.

## Advanced examples

### Viewing on a remote GPU machine

The server listens on localhost only, and over SSH `--view` opens no browser:
it prints a link whose `?policy=` selects your policy. Forward the port and
open the printed link on your own machine:

```bash
# on the remote machine
./cyclotron.sh --view logs/rsl_rl/<experiment_name>/<run>

# on your own machine
ssh -L 3000:localhost:3000 <remote>
```

### Pinning a Hub revision

Run the exact version you mean, not whatever `main` points to today:

```bash
./cyclotron.sh --view <org>/<model> --revision v1.2
```

### A private repo

```bash
HF_TOKEN=hf_... ./cyclotron.sh --view <org>/<private-model>
```

### Sim2sim sanity check

[`--play`](play.md) runs the checkpoint in Isaac Sim (PhysX); `--view` runs
the exported ONNX in MuJoCo. Running both on the same run is a quick sim2sim
check: a policy that walks in one and stumbles in the other has overfit its
training physics.

```bash
./cyclotron.sh --play --task Asimov1-Velocity-AMP-Play-v0 --num_envs 32 \
    --load_run <run>
./cyclotron.sh --export --checkpoint logs/rsl_rl/<experiment_name>/<run>/model_<n>.pt
./cyclotron.sh --view logs/rsl_rl/<experiment_name>/<run>
```
