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

Or every exported run at once, the newest selected, to compare them in one
viewer:

```bash
./cyclotron.sh --view logs/rsl_rl
```

Requirements are Python 3.10+ with `pip install -r requirements-eval.txt`
and [Node.js](https://nodejs.org) 20 or newer with npm. No training environment
is needed; see [standalone installation](../INSTALL.md#viewing-and-benchmarking-only-optional).
The first run fetches the `third_party/humanoid-policy-viewer` submodule, its
npm dependencies and the Asimov 1 robot model; after a pull that moves the
submodule, the next run updates it, its robot model and its dependencies (unless it has
uncommitted changes, which are left alone with a warning). The viewer then checks the policy,
starts a local server and opens the browser with the policy selected (over SSH
or without a display it prints a link instead; see
[Advanced examples](#advanced-examples)). Use the sliders to command a
velocity and push the robot.

## Configuration

`--view` takes one or more policies, in these forms:

| Argument | What runs |
| --- | --- |
| `<org>/<model>[@revision]` | A Hub repo, for example one uploaded with [`--share`](share.md). Cyclotron resolves its commit and downloads the ONNX and configuration files into its cache, then passes a local path to HPV. |
| A local path | Runs in place, with no download: a run folder (its `exported/`, written by [`--export`](export.md)), any folder holding `policy.onnx` and `env.yaml`, or an `.onnx` file. |
| A folder of runs | Every exported run below it, such as `logs/rsl_rl` or `logs/rsl_rl/<experiment_name>`: each `<run>/exported/` holding `policy.onnx` and `env.yaml`. |

A run folder argument descends into `<run>/exported` when the folder itself
has no top-level `.onnx`, so for a fresh run, run `--export` first. A folder
without `env.yaml` next to the `.onnx` is refused: the viewer will not guess
the policy's gains. The policy's inputs, gains, action scale, default pose and
torque limits all come from `env.yaml`.

All the policies go into the viewer's policy list, each labelled by the
shortest end of its path that tells it apart (`<run>/exported` for runs), and
the first one is selected. A folder that is not a policy itself is searched
for runs, newest first by the timestamp that starts a run's folder name (or
the folder's modification time), so the newest run is selected. Cyclotron
prints what it found, naming a few runs that have no export yet:

```text
Found 12 runs under logs/rsl_rl: 9 exported, 3 not (export them with ./cyclotron.sh --export --checkpoint <run>/model_<n>.pt)
  Not exported: asimov/2026-10-21_09-12-40, asimov/2026-10-20_17-03-11 and 1 more
```

A folder with no exported run is an error that says so.

Options after the policies:

| Flag | Meaning |
| --- | --- |
| `--revision <ref>` | Branch, tag or commit of the first Hub policy given (default: `main`). Other policies use `@<ref>`. |
| `--port <n>` | Dev server port (default: 3000, or the next free one). |
| `--no-open` | Do not open a browser window. Added automatically over SSH or without a display. |
| `--benchmark <folder or datasets/org/name[@rev]>` | Open this benchmark in HPV's existing editor (default: cyclotron's `benchmarks/default`). A local folder is edited in place; a Hub dataset opens read-only ([pull it](#edit-a-benchmark-in-hpv) to edit). |
| `--harness <name>` | Select a registered harness (currently `humanoid-policy-viewer`); also configurable through `CYCLOTRON_HARNESS`. |

Run options such as `--jobs`, `--out`, `--watch` or `--verbose` belong to
[`--benchmark`](benchmark.md#options-and-harness-selection); `--view` refuses them.

Private or gated Hub repos need `hf auth login` or `HF_TOKEN` set. Cyclotron's
Python SDK handles authentication, not HPV, which runs with no Hub token or Hub
access. `HF_ENDPOINT` selects another Hub.

Downloads live under `~/.cache/cyclotron/` (or `$XDG_CACHE_HOME/cyclotron`),
overridden by `CYCLOTRON_CACHE_DIR`. Inputs are separated by endpoint, repo
type, repository and resolved commit, and verified before reuse. A corrupt
cache causes an error; move the named revision folder aside and retry.

### Edit a benchmark in HPV

```bash
./cyclotron.sh --view <org>/<model> --benchmark ./my-benchmark
```

The folder must already contain `benchmark.yaml` and its `data/` files.
HPV provides all editing, test creation, task configuration and preview UI,
and saves directly into that folder. Cyclotron does not replace the editor.
A test's Environment tab also sets where the robot starts and whether it drops
and settles first (see [start poses](benchmark.md#start-poses-and-their-hub-form)).
To start a new benchmark, use the viewer's **New benchmark** UI. New folders
are saved in `benchmarks/<name>/`, beside cyclotron's `benchmarks/default/`,
which they start as a copy of. Git ignores them (`benchmarks/.gitignore`), so
they never make the checkout dirty; to commit one, add a `!/<name>/` line there.

Without `--benchmark`, the viewer opens cyclotron's default suite,
`benchmarks/default/`, in place: saving in the UI changes tracked files in this
repository (see [the default benchmark](benchmark.md#the-default-benchmark)).

A Hub benchmark opens read-only, straight from its pinned download in
cyclotron's cache, and cyclotron prints how to edit it. To edit one, pull it
into `benchmarks/` first:

```bash
./cyclotron.sh --benchmark pull datasets/<org>/<dataset>[@<revision>] [<folder>]
./cyclotron.sh --view <org>/<model> --benchmark benchmarks/<dataset>
```

`pull` copies the benchmark's files into `benchmarks/<dataset>/` (or the folder
given), never over an existing folder. It leaves out `eval.yaml`, which
publishing builds. The copy records the commit it was pulled from, so
publishing it checks that the dataset's main has not moved since (see
[publishing](benchmark.md#publish-a-benchmark-folder)).

Publish the saved folder with cyclotron, not the viewer's HF commands:

```bash
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --create --dry-run
```

See [benchmark publication](benchmark.md#publish-a-benchmark-folder) for
validation, licensing, versioning and live publication. Browser-generated
results are saved under cyclotron's results directory, but are currently
local-only: use the cyclotron benchmark command for submittable run provenance.

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

### Comparing policies

Give several policies, from the Hub or on disk, and switch between them in
the viewer's policy list; the first is selected:

```bash
./cyclotron.sh --view logs/rsl_rl/<experiment_name> <org>/<model>@v1.2
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
