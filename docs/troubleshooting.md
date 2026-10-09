# Troubleshooting

Known failures, by symptom. Each entry says what actually went wrong and the
fix. If yours is not here, the [tested hardware](../README.md#tested-hardware)
list is worth a check before anything else.

## Training

### Out-of-memory during training

Isaac Sim's memory use scales with `--num_envs`. Lower it until the run fits;
the policy may then need more iterations to converge or be less stable for
the same number of iterations. The baseline uses 4096 environments on an
A6000-class GPU.

### Crash naming missing sensors, e.g. "No contact sensors", when another run is active

Two Isaac Sim processes on the same machine race on the shared USD cache
under `/tmp/IsaacLab/`, and the loser reads a half-written robot model. The
error blames whatever happened to be missing — contact sensors, usually — not
the race. Run `--train`, `--play` and `--export` sequentially, not in
parallel.

### Training won't stop on SIGTERM

Isaac Sim's loop ignores SIGTERM. Interrupt with Ctrl-C (SIGINT); from a
script, `timeout -s INT -k <grace> <seconds> ...` instead of plain `timeout`.
Training saves a checkpoint and prints the resume command when interrupted.

## Export and share

### "Cannot infer the task"

`--export` and `--share` map the run's `experiment_name` (from its
`agent.yaml`) to a task via `EXPERIMENT_TASKS` in
`source/cyclotron/cyclotron/hub.py`. A new experiment is not in the map yet:
pass `--task <id>` by hand, or add the entry — step 5 of
[Designing your own task](designing-your-own-task.md).

Each `--export` entry below names its check in [export.md](export.md#checks-step-by-step),
where the diagram of its stage shows where it stops; [Examples, check by check](export.md#examples-check-by-check)
tells what each one catches as a short story. The run folder's `exported/export.log`
keeps the full output of the last export.

### "Stopped by --strict: the code of the policy's network changed since training" (export, check 11)

Not a malfunction: the code that builds the policy's network (rsl_rl's
`models` and `modules`: the network classes, the MLP, the normalizer, the
memory module, the action distributions; or a custom network's own module)
differs from what the run recorded at training. The warning above the
stop names each module. The weights still load and the ONNX file still agrees
with PyTorch, because both use the changed code, so nothing later in the
export would notice. Check out the commit the warning names (a separate
worktree works well) and export from there. Or, if you know the change
doesn't alter what the network computes (a comment, a refactor), export
without `--strict`; the file is then tagged `actor_code: changed`. `--share`
always uses `--strict`, so share from the training commit. Nothing else in
the library, this package or Isaac Lab stops an export; the robot model has
checks of its own (10c and 10d, below).

### "records no network code" warning (export, check 11)

The run's `code_state.yaml` predates training recording the network's code,
so changes to it can't be detected. A warning only, also with `--strict` and
`--share`; the exported file is tagged `actor_code: unrecorded`. A run with no
`code_state.yaml` at all gets the "The run has no code_state.yaml" warning instead.
Nothing to fix; runs trained with the current code record it.

### "Stopped by --strict: the code changed since the run was trained" (play)

`--play` with `--strict` stops on any change it checks: the policy settings,
the joints and gains they resolve to, the robot model's URDF, and the code of
the policy's network. The message above the stop names each difference, and
`code_state.yaml` names the trained commit. Check that commit out and re-run,
decide the changes are harmless and run without `--strict`, or pass changed
policy settings back as overrides. Background in
[Code changes since training](../README.md#code-changes-since-training).

### "The current code can't rebuild the policy settings this run was trained with" (export, check 9)

`--export` sets the policy settings back to the run's saved `env.yaml` and
`agent.yaml` before building the environment, and the current code has no
place for some of them: an observation term, actuator group or config field
it no longer has, or a function it can't import under the saved name or any
other module. The lines under the error name each one. Check out the commit
the stop names (a separate worktree works well) and export from there, or add
back what was removed. A function that only moved to another module, for
example from the package's old name `isaac_asimov`, is found under its new
module and doesn't stop the export.

### "The rebuilt policy settings don't match the run's env.yaml and agent.yaml" (export, check 10a)

`--export` set the policy settings back to the run's files, built the
environment, and found a saved setting that ended up with a different value:
the lines under the error name each one. Either the current code changes it
while building the environment (a config that computes a value from others,
for example), or the rebuild in `source/cyclotron/cyclotron/run_config.py`
missed it. Either way the exported policy wouldn't see what it was trained
with. Export from a checkout of the commit the error names, the one the run
was trained with. If that commit's code exports the run cleanly and the
current code doesn't, report it as a rebuild bug with `exported/export.log`.

### "The current code has policy settings the run didn't save" (export, check 10b)

The current code has settings the run's `env.yaml` and `agent.yaml` don't
mention, because they were added to the code after the run was trained. They
can't be checked against the run, so they keep the current code's value; the
lines under the warning name each one with that value. Without `--strict` this
is a warning and the export goes on; with `--strict`, and so always with
`--share`, it stops. To fix it:

- Export from the commit the run was trained with (the warning names it). The
  settings don't exist there, so the question doesn't come up.
- Or check each listed value. A new setting whose default keeps the old
  behaviour (say a new noise option that defaults to off) is harmless; export
  without `--strict`. If a value changes what the policy sees or does, the
  export isn't the trained policy: use the training commit, or train a new
  run with the current code.

### "The joints and gains the policy's inputs and outputs resolve to changed since training" (export, check 10c)

The run's settings name joints by pattern, and the robot model the current
code loads resolves them to other joints, another order, or other gains than
at training: the lines under the error name each one (`action joint 4:
left_knee -> left_hip_pitch`, `stiffness right_ankle: 40 -> 35`). The
exported policy would send its actions to the wrong motors or track them with
the wrong gains, and no later check would notice, since the checkpoint and
`policy.onnx` run in the same environment. Always a stop, also without
`--strict`. Load the robot model the run was trained with: check out the
commit the error names, or the robot model repository's commit recorded under
`robot_model` in the run's `code_state.yaml`. If the robot really changed,
train a new run on it.

### "The robot model changed since training" (export, check 10d)

The URDF the current code loads has a different sha256 than the one the run
was trained on; the warning names both files and their commits. When the
export didn't stop at 10c, the policy still reaches the same joints with the
same gains, but the simulated robot (masses, limits, geometry) changed. A
warning without `--strict`, and the file is tagged `robot_model_check:
changed`; with `--strict`, and so always with `--share`, it stops. Export with
the robot model's recorded commit checked out, or, if you know the change
doesn't matter to the policy, export without `--strict`. A run that records
neither the robot model's sha256 nor its joints and gains gets a note that
which joints the actions reach can't be checked.

### "The checkpoint's policy doesn't fit the network the current code builds" (export, check 12)

The weights in the checkpoint have different layers or sizes than the network
the run's `agent.yaml` builds with the current code; the lines under the error
name each one. Usually the policy's inputs changed (an observation term now
returns a different width) or the network code did (see check 11's warning
above it). Always a stop. Export from the commit the error names.

### "not a joint position action; not attaching deploy metadata" (export, check 13)

The ONNX file is written, but without the [deploy metadata](export.md#deploy-metadata):
it only describes actions that are joint position targets, and this task's
actions are something else (velocity, effort, relative position) or the task
has no action manager (a direct-workflow environment). The file still runs;
a deployer must take the action contract from the task's code, and tools
that read the metadata, such as the [viewer](view.md), can't run it.

### "env.yaml changed or went missing since training" (export, check 6)

Training recorded the sha256 of the run's `env.yaml` and `agent.yaml` in
`code_state.yaml`, and the file no longer matches: it was edited by hand,
replaced, or deleted. `--export` and `--share` refuse it, since these files are
the only record of the settings the run was trained with. Restore the
original (from a backup, or from the run's Hub repo, whose `code_state.yaml`
records the hashes of its uploaded copies), or train a new run with the
settings you want.

### "params has no env.yaml" (`Generate … from the current code? [y/N]`) (export, check 7)

The run doesn't have the file every export ships with, so the settings it was
trained with are unknown. Answering `y` generates it from the current code
and writes it into the run's `params/` once the export succeeds; its first
line says it was generated, from which commit and when, and every later
export of the run warns about it. Only say
yes if the current code is what the run was trained with. Anything else stops
the export, as does running without a terminal to answer in. `--strict` stops
on such a run without asking, and `--share` refuses it before exporting with
`Missing files`. Once generated, the file no longer stops either: `--strict`
and `--share` warn and go on, and the file is uploaded with its first line.

### "policy.onnx (or policy.pt) gives different actions than the checkpoint" (export, check 14)

The export self-check failed: the new `policy.onnx` or `policy.pt` does not
reproduce the checkpoint's actions (for a recurrent policy, its actions or the memory it
returns over the first 3 steps), so the file is wrong (for example, a dropped
observation normalizer), not your run. The new files were discarded; the
previous export in the output folder, if any, is unchanged. This points at an
exporter bug — please report it with the run's `exported/export.log`.

### "Not logged in to Hugging Face"

`--share` needs credentials: run `hf auth login` once, or set `HF_TOKEN`.

## Viewing

### "--view needs Node.js 20 or newer"

Ubuntu 22.04's system Node is 18. Install a current Node from
[nodejs.org](https://nodejs.org) or via nvm; no other part of the pipeline
needs it.

### The viewer refuses the policy

By design: the viewer only runs a policy it can run exactly as trained, and
the refusal names the difference — an observation term it cannot compute, a
policy rate other than 50 Hz, or actions in another order. The rules and the
supported-term list are in [view.md](view.md); the constraints that keep a
new task viewable are in
[Designing your own task](designing-your-own-task.md).

### The browser shows the bundled example policy, not mine

You opened the server's bare URL. The link `--view` prints carries a
`?policy=` parameter that selects your policy — open that exact link,
including over SSH port forwarding ([view.md](view.md) shows the
`ssh -L` setup).
