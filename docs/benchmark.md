# Benchmarking and publication

Cyclotron owns Hugging Face access, input storage, provenance and publication.
The selected harness owns simulation, scoring and its editing UI. The first
harness is
[humanoid-policy-viewer (HPV)](https://github.com/menloresearch/humanoid-policy-viewer).
It receives local paths only, even when you specify Hub repositories.

## Installation

Use Python 3.10+ and `pip install -r requirements-eval.txt`, plus Node.js 20+
with npm for HPV. No Isaac Lab, torch, GPU or full training install is needed.
Cyclotron automatically fetches the pinned harness and npm dependencies. HPV
fetches its robot assets and installs Playwright Chromium on the first run.
Node itself and system libraries are not installed automatically.
See [installation](../INSTALL.md#viewing-and-benchmarking-only-optional).

`cyclotron.sh` simply calls `scripts/view.py` or `scripts/benchmark.py`.
There is no TUI. You can call those Python scripts directly, just as you can
continue to call the original Isaac Lab train/play/export scripts directly.

## Run a benchmark

```bash
# The default suite (benchmarks/default); exported/ is selected when given an exported run
./cyclotron.sh --benchmark logs/rsl_rl/<experiment>/<run> --jobs 2

# A benchmark authored locally in HPV
./cyclotron.sh --benchmark <org>/<model> ./my-benchmark --jobs 2

# A pinned published suite and model
./cyclotron.sh --benchmark <org>/<model>@<model-commit> \
    datasets/<org>/<benchmark>@v1 --jobs 2

# Compare multiple policies
./cyclotron.sh --benchmark ./policy-a ./policy-b ./my-benchmark jobs=2
```

The default output is `logs/benchmark_results/<timestamp>/`:

- `result.json`: unchanged native HPV result, with cells, tasks and runtime details.
- `report.html`: HPV's report.
- `cyclotron.json`: resolved Hub identities, input hashes, code revisions,
  benchmark inspection, completion status and native-result hash.

`--out <directory>` selects another **new** directory; existing output folders
are never overwritten. Use letters, digits, `_` and `-` in the final folder
name so HPV can open it in the results viewer.

### The default benchmark

Without a benchmark argument, `--benchmark` runs and `--view` opens
`benchmarks/default/`, cyclotron's own default locomotion suite. It started as
a copy of HPV's `benchmarks/default` at `a90090b` and is maintained here from
then on: an HPV upgrade does not change it. With HPV `4e62f70` it moved to
protocol 2 (version 2, task IDs `survival_rate_v2` and `survival_time_v2`); its
tests are unchanged.

`--view` without `--benchmark` edits this folder in place, like any local
folder, so a save in HPV's UI changes tracked files. Review them with
`git diff benchmarks/default`, then commit or revert them; until then the dirty
checkout blocks `submit` and `publish`. Bump `version` in `benchmark.yaml` when
a change can move a score. Like any local folder, its results are local-only.

It is only an example: HPV's **New benchmark** button does not copy it (see
[editing a benchmark](view.md#edit-a-benchmark-in-hpv)). `tests/test_evaluation.py`
validates this folder against the pinned HPV, so an upgrade that changes the
format or protocol fails there.

### Scoring protocol 2

The pinned HPV (`5c4e24e`) implements scoring protocol 2, and runs a benchmark
only when its `harness.protocol` is 2 (`--force` runs others, locally only).
Protocol 2 adds metrics that judge a run by how it ends, so a robot that falls
and gets back up survives:

| Metric | One run (test x repeat) |
| --- | --- |
| `survival_rate@2` | 1 if the robot is up in the last frame (and reached the end zone, if the test has one), else 0 |
| `survival_time@2` | 1 if it is up at the end, else the fraction of the test before it last went down (0 if it never stood) |
| `survival_rate@1`, `survival_time@1` | Kept: any fall fails the run, and time counts to the first fall |

A fall is the base below 0.35 m or the torso tilted past 78.5°. The `@2`
metrics work whatever the test's start pose: a robot that starts lying down is
"down" from the first frame, which `@1` counts as a fall at t = 0. To move a
protocol-1 benchmark over, set `harness.protocol: 2`, switch its tasks' metrics
to `@2` and bump `version`, as `benchmarks/default` did. See HPV's
[metrics](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/benchmarks.md#metrics-and-repeats).

### Options and harness selection

Known dashed flags are translated into HPV's words-only CLI; its word syntax
also works. For example, `--jobs 2` and `jobs=2` are equivalent.

| Option | Meaning |
| --- | --- |
| `--benchmark <target>` | Explicit local folder or `datasets/org/name[@rev]` |
| `--policy <target>` | Additional policy; repeatable (`policy=` also works) |
| `--revision <ref>` | Revision of the first positional Hub policy; local paths ignore it |
| `--out <directory>` | New output folder |
| `--jobs`, `--robots` | Worker/page and robot batching settings |
| `--tests`, `--repeats`, `--force` | Local experimentation; blocks submission |
| `--shard i/n` | Partial run; cyclotron does not yet finalize merged provenance |
| `--watch`, `--verbose` | HPV display/debug options |
| `--harness <name>` | Explicit registered harness; `CYCLOTRON_HARNESS` is an alternative |

Harness selection otherwise follows the `harness` that a published benchmark's
`eval.yaml` tasks name (see [the harness block](#the-harness-in-evalyaml)), then
`harness.name` or a recognized `harness.repo` in `benchmark.yaml`. Legacy HPV
repository identities still work. Conflicting
selections and unknown harnesses are rejected. Dataset URLs never install plugins.
Saved results use their recorded harness.

Other `key=value` words (a lowercase key, as HPV writes its options) are passed
to the harness for validation, including after `--`; any other unknown word,
such as `--seed`, `org/m@main=x` or a Hub URL containing `=`, is an error, so
no input reaches the harness unresolved. Put positional inputs before `--`;
cyclotron must resolve all inputs, and an input whose name contains `=` is
given as `policy=<input>` or `benchmark=<input>`. Reserved `policy=`,
`benchmark=` and `out=` aliases are intercepted even after the separator, and
conflicting duplicate options are errors. HPV will reject options it does not
recognize. See its
[benchmark documentation](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/benchmarking.md)
for simulation details.

## View results

```bash
./cyclotron.sh --benchmark view logs/benchmark_results/<timestamp> --no-open
```

Native HPV/browser results remain viewable without a cyclotron manifest.
The server binds to localhost and prints a URL; use SSH port forwarding on
a remote machine. You can also open `report.html` directly.

## Submit evaluation results

```bash
./cyclotron.sh --benchmark submit <result-directory> --dry-run
./cyclotron.sh --benchmark submit <result-directory>
# Or explicitly propose a community contribution
./cyclotron.sh --benchmark submit <result-directory> --pr
```

Submission requires:

- A complete cyclotron-generated manifest, unchanged result and matching inputs.
- Both policy and dataset resolved from the Hub at immutable commits; local
  folders, pulled ones included, are not made eligible by attaching a repo ID
  afterward.
- A dataset commit in the dataset's main history: a pending PR's revision
  (`@refs/pr/<n>`) is refused until the PR is merged.
- Known, clean cyclotron and harness checkouts when the run was made.
  Untracked files inside the HPV submodule do not make cyclotron's checkout
  dirty; the submodule off its pin, edits to its tracked files and cyclotron's
  own untracked files do.
- Matching harness protocol, valid benchmark, full unique cell set, no failures,
  no test/repeat/force overrides and no unmerged shard.
- Published task IDs in the pinned dataset's `eval.yaml`, declaring
  `evaluation_framework: cyclotron`.

Cyclotron verifies pinned Hub bytes again during submission, including model
configuration and tracking overrides. It records provenance alongside HPV's
`source: local` result rather than changing that field.

Known limitation: unlike `publish`, `submit` does not require the recorded
cyclotron and harness commits to be on a public upstream branch, only known
and clean, so a result can name a commit others cannot check out.

Scores go to `.eval_results/<org>__<benchmark>.yaml` in each evaluated model
repo. Existing entries for other datasets/tasks are kept. Dates are strings,
and notes identify the cyclotron/harness/policy commits and scoring context.
Nothing receives a verified badge or verification token automatically.

Commits use the evaluated model SHA as `parent_commit`: if main has moved,
a conflict is reported, not silently rebased onto a different policy. Direct
commits are attempted unless `--pr` is requested; an explicit permission
denial falls back to a PR. Authentication/network/read failures are not
interpreted as an absent results file. For multi-model runs, each model is a
separate commit; success is printed per model, so a later failure can be
retried without mistaking it for a fully successful submission.

A sidecar prevents accidental misattribution; it is **not cryptographic proof**
of execution. Browser-generated results and results made outside cyclotron
are local-only in this iteration. To obtain submittable results, publish the
benchmark, then rerun using its Hub ID rather than the editable folder.

## Publish a benchmark folder

Author the benchmark in HPV's existing UI:

```bash
./cyclotron.sh --view <org>/<model> --benchmark ./my-benchmark
```

HPV edits and saves that exact folder. New benchmarks created in its UI are
saved in `benchmarks/<name>/` and start at `version: 1` under the pinned HPV's protocol; to
change a published benchmark, `--benchmark pull` it into `benchmarks/` first.
See [editing a benchmark](view.md#edit-a-benchmark-in-hpv). Cyclotron does not
reimplement test creation, task configuration or visualization.

Git ignores the benchmarks in `benchmarks/` (see `benchmarks/.gitignore`), so
an unpublished one never makes the checkout dirty. To commit one on purpose,
add a `!/<name>/` line for it to that file; its files then track like any other.

After saving, publish the folder with cyclotron:

```bash
# New dataset: preview, then create a PRIVATE repo and publish
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --create --dry-run
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --create

# Existing dataset: optimistic base check and optional PR
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --base <commit> --pr

# A published benchmark: pull it, edit it, then publish onto the commit it was pulled from
./cyclotron.sh --benchmark pull datasets/<org>/<dataset>
./cyclotron.sh --benchmark publish benchmarks/<dataset> <org>/<dataset>

# After a PR is merged, tag its merged revision
./cyclotron.sh --benchmark release <org>/<dataset> <merged-commit>
```

Before writing remotely, cyclotron:

1. Stages a coherent copy of the folder and detects concurrent edits.
2. Runs HPV's offline inspect command (see [adding a harness](#storage-and-adding-a-harness)):
   its own validation, run plan and exact scored-content hash.
3. Checks protocol compatibility and a clean, public harness commit, and a
   clean cyclotron checkout. Bump `benchmark.yaml`'s version when scored content
   changes; versions cannot decrease. "Public" means on a fetched upstream
   branch, so cyclotron first fetches the harness's history (HPV is cloned with
   `--depth 1`, which only knows its own commit); if that fetch fails, the
   blocker gives the `git fetch` command to run.
4. Records the harness's commit in `harness.tested_commit`, writes each test
   row in its Hub form (see [start poses](#start-poses-and-their-hub-form)),
   builds `eval.yaml` with `evaluation_framework: cyclotron` and each task's
   `harness`, preserves historical task IDs, and builds the dataset card while
   retaining README prose and unrelated metadata.
5. Uses an optimistic parent commit, removes obsolete benchmark files, and
   tags successful direct version releases. PRs are tagged only after merging.

The parent is the dataset's current main, or `--base` if given. A folder made
by `--benchmark pull` records the commit it was pulled from in
`.cyclotron-origin.json` (git ignores it), and each direct publication from it
moves the record to the new commit. Without `--base`, publishing it to the same
dataset (and endpoint) uses that commit as the base, so if main has moved
since, `publish` refuses rather than overwrite those changes. Pull the current
main into a new folder and carry your edits over, or pass
`--base <current main>` to publish over them.

### Start poses and their Hub form

A test row can say where the robot starts. `start` takes the shape of the
robot's `init_state` in `env.yaml`: `pos`, `rot` (w, x, y, z), `lin_vel`,
`ang_vel`, and `joint_pos`/`joint_vel` keyed by joint names or patterns. The
keys it gives replace the policy's own `init_state`; `null` starts from the
policy's own. `drop_settle_s` (seconds, at most 10, or `null` for off) leaves
the robot limp until it comes to rest before the test's clock starts.

Author both in HPV's editor: the test's Environment tab offers the start poses
Policy default, Standing and Lying on back, and a Drop and settle checkbox with
its duration. Anything finer is written into the row by hand; see HPV's
[test row](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/benchmarks.md#a-test-one-row).

Locally, `test.jsonl` holds `start` as a JSON object. On the Hub it is JSON
text, because a dataset card cannot declare a map keyed by joint patterns, so
the card declares the column a string. `publish` uploads each row in that form:
the harness gives cyclotron every file as it is to be published. HPV reads both
forms, so a published benchmark has the same scored-content hash as the folder
it came from.

### The harness in eval.yaml

Cyclotron is the evaluation framework; the harness runs and scores a
benchmark (HPV is the first). The
[eval.yaml spec](https://huggingface.co/docs/hub/en/eval-results) fixes the
top-level fields, and keeps framework-specific settings in each task, as with
inspect-ai's `solvers` and `scorers`. So each task names its harness:

```yaml
evaluation_framework: cyclotron
tasks:
- id: survival_rate_v2
  config: all
  split: test
  harness:
    name: humanoid-policy-viewer
    repo: https://github.com/menloresearch/humanoid-policy-viewer
    protocol: 2
    commit: 5c4e24efae725205c667f0c73af237827e11b985
```

`name` is the registered harness, `protocol` the scoring protocol the results
are comparable under, and `commit` the harness commit the benchmark was checked
with. Running a published benchmark selects that harness, and `submit` refuses a
result made with a different one. Tasks from earlier versions keep the block
they were published with. The card is tagged `cyclotron` plus the benchmark's
own `tags`, not the harness's name.

Set `license:` in `benchmark.yaml` to a Hub license ID or a publishable license
filename such as `LICENSE` or `LICENSE.txt`. The named file must be included in
the upload. An explicit blank means no license; an existing card's license can be retained.
Whatever the card's `license_link` ends up naming, one kept from the existing
card included, must be a file in the upload (a URL is fine), so a linked
license is never deleted as obsolete.
There is no silent license default: when a benchmark names none, `publish`
asks in a terminal whether to add `license: bsd-3-clause` under `version:`, and
otherwise (including `--dry-run`, which never edits the folder) stops and names
the line to add. HPV task IDs remain `<task>_v<version>`,
with the same config mapping and `test` split.

Dry-run makes no remote writes and does not change your benchmark folder. It
prints generated metadata and publication blockers, and exits with status 1
when there are blockers (0 when the folder is ready to publish). Generated files, the
tested commit and Hub-form rows are staged for upload, not written over the
folder being edited.
If tagging fails after a successful commit, the error includes a release
command to retry. Existing tags are never moved.

Dedicated diff/baseline commands are not part of this integration: version
comparison is built into publication. HPV itself only reads from the Hub; it
has no publish or submit commands.

### Hub registration (external prerequisite)

The Hub must recognize `cyclotron` as an evaluation framework. This repository
cannot make that registration locally. Before live benchmark publication:

1. Propose `cyclotron` in the framework registry at
   [huggingface.js/packages/tasks/src/eval.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/eval.ts),
   following the current entry type and linking to this project's evaluation
   documentation. The framework identity is cyclotron; HPV is a harness.
2. Obtain any benchmark-dataset beta/allow-list approval required by the Hub.
3. Test registration on a private dataset before a public release, including
   that the per-task `harness` block passes the Hub's push-time validation
   (other frameworks' per-task fields, such as `lower_is_better`, already do).

No registration PR, approval or live publication is implied by this code.
Local editing/runs and metadata dry-runs work independently; the Hub may reject
`eval.yaml` until registration is available. See the
[Hub eval-results specification](https://huggingface.co/docs/hub/en/eval-results).

## Storage and adding a harness

`CYCLOTRON_CACHE_DIR` overrides `~/.cache/cyclotron/` (which otherwise honors
`XDG_CACHE_HOME`). Hub snapshots are separated by endpoint, repository type,
repo and commit, and recorded under the Hub's own spelling of the repo ID (its
case, or its new name after a rename), not the one typed. Cached bytes are
checked against pinned Git/LFS metadata and materialized as real files for the
viewer, filenames containing `[`, `*` or `?` included. The cache is never edited:
the viewer opens a Hub benchmark from it read-only, and `--benchmark pull`
copies one into `benchmarks/` to edit.
Private repositories use `hf auth login` or `HF_TOKEN`; `HF_ENDPOINT` selects
the Hub. Local run/view paths do not need Hub authentication. HPV itself runs
without Hub credentials or access: cyclotron removes `HF_TOKEN`, points
`HF_TOKEN_PATH` at a file that does not exist and `HF_ENDPOINT` at an
unroutable local address for it. HPV's own files
(its download cache, temporary policy libraries and the viewer's debug logs)
go under `harnesses/humanoid-policy-viewer/` in the cyclotron cache, so the
HPV checkout stays clean.

Adapters live in `source/cyclotron/cyclotron/evaluation/harnesses/`. Add a
trusted adapter and explicit registry entry/legacy repository alias in
`harnesses/base.py`. An adapter owns bootstrap, file selection, view/run
commands, benchmark inspection/task descriptors, result interpretation and
harness-specific blockers. Cyclotron owns all Hub access and provenance. The
HPV adapter reads benchmarks and policies through HPV's own offline inspect
command, which needs only its npm dependencies (no robot model, browser or
network): `node scripts/inspect.mjs benchmark <dir> [<commit>]` (schema
`hpv-inspection/1`: validation, run plan, scored-content hash, card and each
file as published) and `node scripts/inspect.mjs policy <path>` (schema
`hpv-policy-inspection/1`: the ONNX to run, the hashes a result records, and
errors that stop a run). The adapter checks both schema names. A future harness
need not use Node.
