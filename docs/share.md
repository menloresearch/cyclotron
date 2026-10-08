# Sharing policies on the Hugging Face Hub

`./cyclotron.sh --share` publishes a finished run to the Hugging Face Hub:
the exported policy, its training settings and a generated model card, in one
repo anyone can [view](view.md) or deploy. Use it when a
policy is ready to hand to others.

## Quick start

Log in once:

```bash
hf auth login
```

Then share a run:

```bash
./cyclotron.sh --share logs/rsl_rl/<experiment_name>/<run> \
    --repo-id <user_or_org>/<repo_name>
```

## What a share uploads

| File | Contents |
| --- | --- |
| `policy.onnx` | The policy, freshly exported from the checkpoint (see below). |
| `env.yaml`, `agent.yaml` | The run's task and training settings. |
| `code_state.yaml` | The code the run was trained with, for runs whose training recorded one. |
| `README.md` | The generated model card: BSD-3-Clause license, `library_name: asimov`, `pipeline_tag: robotics`. |

`policy.pt` and `export.log` never leave the machine. The uploaded yaml
copies have the training machine's file paths reduced to file names; the run
directory itself is not changed.

## What `--share` does

1. **Checks the run's settings files.** A run whose `env.yaml` or
   `agent.yaml` is missing (`Missing files`), or no longer matches the sha256
   its `code_state.yaml` recorded at training, is refused before anything
   runs, also with `--onnx`. A file generated from the code by an earlier
   export is published with a warning; its first line says it was generated. The uploaded
   `code_state.yaml` records the sha256 of the uploaded copies, whose local
   paths are reduced to file names.
2. **Re-exports the checkpoint.** The checkpoint is always exported first
   through [`--export`](export.md) with `--strict`; a pre-existing
   `exported/policy.onnx` of unknown vintage is never uploaded. Strict turns
   the warnings `--export` alone prints (code that can change the exported
   policy changed since training, or the code has policy settings the run
   didn't save) into stops. A `policy.onnx` whose
   actions differ from the checkpoint is always a stop. `--onnx <file>` is the
   only way to upload an existing ONNX file as is.
3. **Generates the model card** from `--title` and `--summary`.
4. **Uploads everything as one git commit** on the Hub repo, with the message
   `Upload Asimov policy from <run folder name>`. The repo is created if it
   does not exist (`--private` makes it private).

## Configuration

| Flag | Meaning |
| --- | --- |
| `run_dir` (positional) | The run to share, e.g. `logs/rsl_rl/<experiment_name>/<run>`. |
| `--repo-id` (required) | Target repo, e.g. `user/name`. Created if it does not exist. |
| `--title` | Title of the model card. |
| `--summary` | Paragraph shown under the model card title, e.g. describing the training method. |
| `--checkpoint` | Checkpoint to export and upload: a full path to a `.pt` file, or a filename inside `run_dir`. Defaults to the latest `model_*.pt` in the run. |
| `--onnx` | Upload this ONNX file as is, instead of exporting the checkpoint. |
| `--task` | Task used to export the policy. Inferred from the experiment name in `agent.yaml` if omitted. |
| `--private` | Create the repo as private if it does not exist. |
| `--dry-run` | Print what would be uploaded and the model card, then exit. |

## Repos are git repos

Hugging Face model repos are git repositories, and each share is one commit.
Sharing the same `--repo-id` again adds a commit on top, so a repo builds a
history you can diff, pin by revision hash, or roll back. The sane convention
is one repo per experiment; the history is a safety net, not an
organizational scheme.

## Advanced examples

Share a specific checkpoint instead of the latest:

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_rough/2026-10-01_12-00-00 \
    --repo-id menlo/asimov-rough --checkpoint model_1500.pt
```

Preview the model card and the upload list without touching the Hub:

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_rough/2026-10-01_12-00-00 \
    --repo-id menlo/asimov-rough \
    --title "Asimov 1 rough-terrain policy" \
    --summary "Trained with PPO + AMP on rough terrain with pushes." \
    --dry-run
```

Upload a hand-picked ONNX file as is, skipping the re-export (you vouch for
the file matching the run's yaml settings):

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_rough/2026-10-01_12-00-00 \
    --repo-id menlo/asimov-rough --onnx /path/to/policy.onnx
```

Update an already-shared policy by sharing to the same repo again:

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_rough/2026-10-05_09-30-00 \
    --repo-id menlo/asimov-rough
```

The second share is a second commit. Consumers who load
`menlo/asimov-rough` get the new policy; anyone who pinned the previous
revision hash keeps the old one, and the first commit stays in the history
for comparison or rollback.
