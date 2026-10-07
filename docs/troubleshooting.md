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

### "Stopped by --strict: the code changed since the run was trained"

Not a malfunction: the checkout differs from the code that trained the run,
and strict mode (`--share` always, `--export`/`--play` with `--strict`)
refuses to continue. The message above the stop names each difference, and
`code_state.yaml` names the trained commit. Check that commit out (a separate
worktree works well) and re-run, or decide the changes are harmless and run
without `--strict`. For `--play`, changed policy settings can also be passed
back as overrides; `--export` restores them itself. Background in
[Code changes since training](../README.md#code-changes-since-training).

### "The current code can't rebuild the policy settings this run was trained with"

`--export` sets the policy settings back to the run's saved `env.yaml` and
`agent.yaml` before building the environment, and the current code has no
place for some of them: an observation term, actuator group or config field
it no longer has, or a function it can't import under the saved name or any
other module. The lines under the error name each one. Check out the commit
the stop names (a separate worktree works well) and export from there, or add
back what was removed. A function that only moved to another module, for
example from the package's old name `isaac_asimov`, is found under its new
module and doesn't stop the export.

### "policy.onnx gives different actions than the checkpoint"

The export self-check failed: the written ONNX does not reproduce the
checkpoint's actions, so the file is wrong (for example, a dropped
observation normalizer), not your run. Nothing was verified, so do not deploy
or share the file. This points at an exporter bug — please report it with the
run's `exported/export.log`.

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
