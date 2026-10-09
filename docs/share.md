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
| `policy.pt` | The same policy as TorchScript, from the same export. Not uploaded with `--onnx`. |
| `env.yaml`, `agent.yaml` | The run's task and training settings. |
| `code_state.yaml` | The code the run was trained with, for runs whose training recorded one. |
| `README.md` | The generated model card: BSD-3-Clause license, `library_name: asimov`, `pipeline_tag: robotics`. |

`export.log` never leaves the machine. The uploaded yaml
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
   through [`--export`](export.md) with `--strict`, into a temporary folder
   that is deleted after the upload; a pre-existing `exported/` folder of
   unknown vintage is never uploaded, and the run's `exported/` is left as is. Strict turns
   the warnings `--export` alone prints (the code of the policy's network
   changed since training, or the code has policy settings the run didn't
   save) into stops. A run that recorded no network code (an older run, or one
   trained outside cyclotron) is only warned about and can still be shared. A
   `policy.onnx` or `policy.pt` whose actions differ from the checkpoint is
   always a stop. `--onnx <file>` is the only way to upload an existing ONNX
   file as is; it uploads no `policy.pt`.
3. **Generates the model card** from `--title` and `--summary`.
4. **Uploads everything as one git commit** on the Hub repo, with the message
   `Upload Asimov policy from <run folder name>`. The repo is created if it
   does not exist (`--private` makes it private). Any of the files in the
   table above that an earlier share put in the repo and this one doesn't
   upload (a `code_state.yaml` from a run that had one, or a `policy.pt`
   before an `--onnx` share) is removed in the same commit, so the repo never
   mixes two runs; other files you added to the repo are left alone.

## Configuration

| Flag | Meaning |
| --- | --- |
| `run_dir` (positional) | The run to share, e.g. `logs/rsl_rl/<experiment_name>/<run>`. |
| `--repo-id` (required) | Target repo, e.g. `user/name`. Created if it does not exist. |
| `--title` | Title of the model card. |
| `--summary` | Paragraph shown under the model card title, e.g. describing the training method. |
| `--checkpoint` | Checkpoint to export and upload: a filename inside `run_dir`, or a full path to one. Defaults to the latest `model_*.pt` in the run. A checkpoint from another run is refused, since the run's settings are uploaded with it; share that run instead. |
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
./cyclotron.sh --share logs/rsl_rl/asimov_velocity_amp/2026-10-01_12-00-00 \
    --repo-id menlo/asimov-velocity-amp --checkpoint model_1500.pt
```

Preview the model card and the upload list without touching the Hub:

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_velocity_amp/2026-10-01_12-00-00 \
    --repo-id menlo/asimov-velocity-amp \
    --title "Asimov 1 AMP walking policy" \
    --summary "Trained with PPO + AMP, with pushes and randomized PD gains." \
    --dry-run
```

Upload a hand-picked ONNX file as is, skipping the re-export (you vouch for
the file matching the run's yaml settings). No `policy.pt` is uploaded, and
one an earlier share put in the repo is removed:

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_velocity_amp/2026-10-01_12-00-00 \
    --repo-id menlo/asimov-velocity-amp --onnx /path/to/policy.onnx
```

Update an already-shared policy by sharing to the same repo again:

```bash
./cyclotron.sh --share logs/rsl_rl/asimov_velocity_amp/2026-10-05_09-30-00 \
    --repo-id menlo/asimov-velocity-amp
```

The second share is a second commit. Consumers who load
`menlo/asimov-velocity-amp` get the new policy; anyone who pinned the previous
revision hash keeps the old one, and the first commit stays in the history
for comparison or rollback.
