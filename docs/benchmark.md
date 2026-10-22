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

The default output is `logs/benchmarks/humanoid-policy-viewer/<timestamp>/`:

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

HPV's **New benchmark** button still starts new suites from HPV's own bundled
default, not this one. `tests/test_evaluation.py` validates this folder against
the pinned HPV, so an upgrade that changes the format or protocol fails there.

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

Unknown options are passed to the harness for validation, including after `--`.
Put positional inputs before `--`; cyclotron must resolve all inputs. Reserved
`policy=`, `benchmark=` and `out=` aliases are intercepted even after the
separator, and conflicting duplicate options are errors. HPV will reject
options it does not recognize. See its
[benchmark documentation](https://github.com/menloresearch/humanoid-policy-viewer/blob/main/docs/benchmarking.md)
for simulation details.

## View results

```bash
./cyclotron.sh --benchmark view logs/benchmarks/humanoid-policy-viewer/<timestamp> --no-open
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
  working folders are not made eligible by attaching a repo ID afterward.
- Known, clean cyclotron and harness checkouts when the run was made.
- Matching harness protocol, valid benchmark, full unique cell set, no failures,
  no test/repeat/force overrides and no unmerged shard.
- Published task IDs in the pinned dataset's `eval.yaml`, declaring
  `evaluation_framework: cyclotron`.

Cyclotron verifies pinned Hub bytes again during submission, including model
configuration and tracking overrides. It records provenance alongside HPV's
`source: local` result rather than changing that field. HPV's own printed
local-source upload hint does not determine cyclotron eligibility.

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

HPV edits and saves that exact folder. New benchmarks created in its UI live
under `logs/benchmark_workspaces/humanoid-policy-viewer/`. See
[editing a benchmark](view.md#edit-a-benchmark-in-hpv). Cyclotron does not
reimplement test creation, task configuration or visualization.

After saving, publish the folder with cyclotron:

```bash
# New dataset: preview, then create a PRIVATE repo and publish
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --create --dry-run
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --create

# Existing dataset: optimistic base check and optional PR
./cyclotron.sh --benchmark publish ./my-benchmark <org>/<dataset> --base <commit> --pr

# After a PR is merged, tag its merged revision
./cyclotron.sh --benchmark release <org>/<dataset> <merged-commit>
```

Before writing remotely, cyclotron:

1. Stages a coherent copy of the folder and detects concurrent edits.
2. Calls HPV's offline validation and exact scored-content hash implementation.
3. Checks protocol compatibility and a clean, public harness commit, and a
   clean cyclotron checkout. Bump `benchmark.yaml`'s version when scored content
   changes; versions cannot decrease.
4. Records the harness's commit in `harness.tested_commit`, builds `eval.yaml` with
   `evaluation_framework: cyclotron` and each task's `harness`, preserves
   historical task IDs, and builds the dataset card while retaining README prose
   and unrelated metadata.
5. Uses an optimistic parent commit, removes obsolete benchmark files, and
   tags successful direct version releases. PRs are tagged only after merging.

### The harness in eval.yaml

Cyclotron is the evaluation framework; the harness runs and scores a
benchmark (HPV is the first). The
[eval.yaml spec](https://huggingface.co/docs/hub/en/eval-results) fixes the
top-level fields, and keeps framework-specific settings in each task, as with
inspect-ai's `solvers` and `scorers`. So each task names its harness:

```yaml
evaluation_framework: cyclotron
tasks:
- id: survival_rate_v1
  config: all
  split: test
  harness:
    name: humanoid-policy-viewer
    repo: https://github.com/menloresearch/humanoid-policy-viewer
    protocol: 1
    commit: a90090b716d2f121989803fe4afc7be8a45aa6c7
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
There is no silent license default. HPV task IDs remain `<task>_v<version>`,
with the same config mapping and `test` split.

Dry-run makes no remote writes and does not change your working folder. It
prints generated metadata and publication blockers. Generated files and the
tested commit are staged for upload, not written over the folder being edited.
If tagging fails after a successful commit, the error includes a release
command to retry. Existing tags are never moved.

Dedicated pull/diff/baseline commands are not part of this integration. Hub
working copies are created when opening a dataset in the viewer, and version
comparison is built into publication. Standalone HPV commands still exist,
but cyclotron does not call their HF operations.

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
repo and commit. Cached bytes are checked against pinned Git/LFS metadata and
materialized as real files for the viewer. Writable workspaces are separate;
opening the same Hub reference preserves edits rather than overwriting them.
Private repositories use `hf auth login` or `HF_TOKEN`; `HF_ENDPOINT` selects
the Hub. Local run/view paths do not need Hub authentication.

Adapters live in `source/cyclotron/cyclotron/evaluation/harnesses/`. Add a
trusted adapter and explicit registry entry/legacy repository alias in
`harnesses/base.py`. An adapter owns bootstrap, file selection, view/run
commands, benchmark inspection/task descriptors, result interpretation and
harness-specific blockers. Cyclotron owns all Hub access and provenance. HPV's
thin offline JS bridge is `scripts/harnesses/hpv-inspect.mjs`; it reuses HPV's
format validation and hash functions without calling its HF workflows. A future
harness need not use Node.
