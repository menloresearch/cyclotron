# Code changes since training

These checks apply to [`--play`](play.md) and [`--export`](export.md),
and therefore to [`--share`](share.md), which exports before uploading.

`--play` rebuilds the policy from the code you have checked out now, so a run
trained with different code can load wrongly or behave differently. `--export`
first sets the policy settings back to the ones the run saved in `env.yaml` and
`agent.yaml`, like a restart of the run, with no overrides (see
[What `--export` does](export.md#what---export-does)), but the code behind
them, such as the observation functions and the robot model, is still the
current one. Training therefore writes `params/code_state.yaml` next to
`env.yaml` and `agent.yaml`, with only what the checks below use or a reader
needs to find the training code again:

- the git commit and branch, and whether there were uncommitted changes, plus
  the Isaac Lab commit and the `isaacsim`, `isaaclab`, `rsl-rl-lib` and
  `torch` versions (recorded, not compared);
- the robot model: its URDF's sha256 and the commit of the repository it
  comes from (`robot_model`);
- the sha256 of the code behind the policy's network: every module of
  rsl_rl's `models` and `modules`, and a custom network's own module
  (`actor_code`);
- the joints and gains the policy's settings resolved to on that robot model:
  each action joint in order with its scale, offset, clip, stiffness and
  damping, and the joints or bodies each observation term reads
  (`policy_io`);
- the sha256 of the `env.yaml` and `agent.yaml` written next to it: they are
  the record of how the run was trained, so `--export` and `--share` refuse a
  run whose files no longer match (edited by hand, or deleted).

It holds hashes and names, not code or local paths.

Before loading a checkpoint, export and play compare the run with the current
code and print what changed:

- **Policy settings**: what the policy sees and does. That is the actor's
  observation terms (order, functions, parameters, scales, clipping, history),
  the actions, the robot's default pose and actuators, the control rate and
  the actor network. For `--play` each difference is named by its Hydra
  override path, for example
  `env.observations.policy.base_ang_vel.scale: 0.25 -> 0.5`, so you can pass
  the trained value back on the command line. `--export` has set them back
  already, so for it a saved setting that still differs means the current
  code changes it while building, and stops the export; settings the run
  didn't save (added to the code since training) are warnings. Critic and
  AMP inputs, rewards, events, terrain, command ranges and observation noise
  are not compared: they only shape training, and the play tasks change some
  of them.
- **Joints and gains**: what the settings resolve to on the robot model the
  current code loads, compared with `policy_io`, one line per joint
  (`action joint 4: left_knee -> left_hip_pitch`). Settings that name joints
  by pattern reach other joints, or other gains, when the robot model
  changes, and no later check notices. `--export` always stops on these
  (check 10c in [export.md](export.md#after-building-the-environment)).
- **Robot model**: whether its URDF still has the recorded sha256 (check
  10d).
- **Network code**: the modules in `actor_code` that changed. The weights
  still load and the ONNX file agrees with the changed code, so this is the
  only way to notice (check 11).

  Nothing else is compared: not the rest of this package, Isaac Lab or the
  package versions. Runs trained before `code_state.yaml` existed have only
  the commit rsl_rl logged in the run's `git/` folder, used to name the
  commit to check out.

Code changes are warnings, since experiments change code on purpose; add
`--strict` to stop instead. `--export` stops on changed joints and gains
without it. A checkpoint whose policy no longer fits the
network always stops, for example because an observation term was added and
the input size changed. For `--play`, so does a changed actor class or
activation, which would load the old weights but compute something else;
`--export` takes those from the run's `agent.yaml`, so they can't change.
[Examples, check by check](export.md#examples-check-by-check) shows what
each export check catches.

Export and play load only the policy from the checkpoint. The critic and the
AMP discriminator are only used in training, so a run whose critic saw extra
privileged inputs still exports and plays.
