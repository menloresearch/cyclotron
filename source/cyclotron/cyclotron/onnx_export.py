"""Helpers for ``./cyclotron.sh --export`` that need PyTorch but not Isaac Sim, so they can be tested on their own."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
from tensordict import TensorDict

from cyclotron.code_state import CODE_STATE_FILE, RUN_CONFIGS, urdf_filepath
from cyclotron.hub import checkpoints

# The run's training config, copied next to policy.onnx so the export folder has the same files as a shared Hub repo.
BUNDLE_YAMLS = RUN_CONFIGS
# The record of the code the run was trained with; runs trained before it existed don't have one.
OPTIONAL_BUNDLE_YAMLS = (CODE_STATE_FILE,)

# Schema of the deploy metadata attached to policy.onnx; firmware should refuse versions it does not know.
DEPLOY_METADATA_VERSION = "1"

# Steps the export check runs a recurrent policy for, from an empty memory: enough to use the memory it carries.
RECURRENT_STEPS = 3


def deploy_metadata(
    joint_names: list[str],
    raw_action_clip: float | None,
    action_scale: list[float],
    action_offset: list[float],
    action_clip: list[list[float]] | None,
    joint_stiffness: list[float],
    joint_damping: list[float],
    sim_dt: float,
    decimation: int,
    observation_names: list[str],
    trained_commit: str | None,
    robot_model: dict | None = None,
    trained_outside_cyclotron: bool = False,
) -> dict[str, str]:
    """The deployment contract of a policy, as the strings stored in ONNX metadata.

    Only what a runtime must agree with the policy about before driving a robot with it: the joint order the actions
    are in, how raw actions become position targets (clamped to ``[-raw_action_clip, raw_action_clip]`` when the run
    set ``clip_actions``, then ``target = action * scale + offset``, then an optional ``[low, high]`` clip per joint),
    the PD gains the targets were trained to be tracked with, the rate the policy was trained to run at, the ordered
    observation terms its input is built from, and, for traceability, the training commit and the robot model the run
    was trained with (``robot_model`` as ``code_state.yaml`` records it). Training settings stay in the yaml files next
    to the ONNX; they are not deployment inputs.
    """
    per_joint = [action_scale, action_offset, joint_stiffness, joint_damping]
    if any(len(values) != len(joint_names) for values in per_joint):
        raise ValueError("action_scale, action_offset, joint_stiffness and joint_damping need one entry per action")
    if action_clip is not None and len(action_clip) != len(joint_names):
        raise ValueError("action_clip must have one [low, high] pair per action")
    metadata = {
        "deploy_metadata_version": DEPLOY_METADATA_VERSION,
        "joint_names": json.dumps(joint_names),
        "raw_action_clip": json.dumps(None if raw_action_clip is None else float(raw_action_clip)),
        "action_scale": json.dumps(action_scale),
        "action_offset": json.dumps(action_offset),
        "action_clip": json.dumps(action_clip),
        "joint_stiffness": json.dumps(joint_stiffness),
        "joint_damping": json.dumps(joint_damping),
        "sim_dt": repr(float(sim_dt)),
        "decimation": str(int(decimation)),
        "policy_rate_hz": repr(1.0 / (float(sim_dt) * int(decimation))),
        "observation_names": json.dumps(observation_names),
    }
    if trained_commit:
        metadata["trained_commit"] = trained_commit
    if robot_model:
        # Check out the commit of the repository and hash the urdf there to get the exact model back.
        fields = {
            "name": robot_model.get("name"),
            "repo": robot_model.get("repo"),
            "urdf_filepath": urdf_filepath(robot_model),
            "sha256": robot_model.get("sha256"),
            "commit": robot_model.get("commit"),
        }
        metadata.update({f"robot_model_{key}": value for key, value in fields.items() if value})
        if robot_model.get("dirty") is not None:
            metadata["robot_model_dirty"] = json.dumps(bool(robot_model["dirty"]))
    if trained_outside_cyclotron:
        metadata["trained_outside_cyclotron"] = "true"
    return metadata


def attach_deploy_metadata(onnx_path: str, metadata: dict[str, str]) -> None:
    """Store ``metadata`` plus the graph's own input and output widths in the ONNX file's metadata_props.

    The graph itself is untouched; runtimes that do not read metadata run the file unchanged.
    """
    import onnx

    model = onnx.load(onnx_path)
    dims = {
        name: [d.dim_value for d in value.type.tensor_type.shape.dim]
        for name, value in (("obs", model.graph.input[0]), ("actions", model.graph.output[0]))
    }
    entries = {**metadata, "obs_dim": str(dims["obs"][-1]), "action_dim": str(dims["actions"][-1])}
    kept = [entry for entry in model.metadata_props if entry.key not in entries]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    for key, value in entries.items():
        model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


EXPORT_LOG = "export.log"


def export_log(output_dir: str, header: str):
    """Print and append an export's messages to ``export.log`` next to its artifacts; returns the log function.

    The log is append-only, so overwritten exports leave their history behind: which checkpoint was exported when,
    and everything that export printed. ``header`` opens the invocation's section in the file without being printed.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, EXPORT_LOG)
    with open(path, "a") as f:
        f.write(f"--- {datetime.now().isoformat(timespec='seconds')} {header}\n")

    def log(message: str) -> None:
        # Flushed: Isaac Sim exits without flushing a stdout that is redirected to a file or a pipe.
        print(message, flush=True)
        with open(path, "a") as f:
            f.write(message + "\n")

    return log


def existing_export_note(run_dir: str, output_dir: str) -> str | None:
    """What ``--export`` is about to overwrite, or None when the output folder has no policy.onnx yet.

    Export writes its artifacts together, so the existing policy.onnx's file time is when that export was made;
    checkpoints written after it mean the old export was not of the run's latest checkpoint.
    """
    existing = os.path.join(output_dir, "policy.onnx")
    if not os.path.isfile(existing):
        return None
    exported_at = os.path.getmtime(existing)
    newer = [name for name in checkpoints(run_dir) if os.path.getmtime(os.path.join(run_dir, name)) > exported_at]
    if not newer:
        return "Overwriting the run's existing export."
    return f"Overwriting an export made before {newer[-1]} was written, so it was not of the run's latest checkpoint."


def replace_export(staging: str, output_dir: str) -> None:
    """Move a checked export from ``staging`` into ``output_dir``, replacing the previous export's files, and remove
    ``staging``. Export writes into a staging folder first so that a failed check leaves the previous export as is."""
    for name in sorted(os.listdir(staging)):
        os.replace(os.path.join(staging, name), os.path.join(output_dir, name))
    os.rmdir(staging)


def copy_run_yamls(run_dir: str, output_dir: str) -> list[str]:
    """Copy ``params/env.yaml``, ``params/agent.yaml`` and, if present, ``params/code_state.yaml`` from the run into
    the export folder.

    The copies are unchanged; ``--share`` strips the training machine's file paths only from what it uploads.
    Returns the names of the required yaml files the run does not have.
    """
    missing = []
    for name in BUNDLE_YAMLS + OPTIONAL_BUNDLE_YAMLS:
        source = os.path.join(run_dir, "params", name)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(output_dir, name))
        elif name in BUNDLE_YAMLS:
            missing.append(name)
    return missing


@contextmanager
def _full_float32():
    """Run PyTorch in full float32, like onnxruntime. GPUs since Ampere let cuDNN run an LSTM in TF32 by default,
    which drifts ~1e-3 from float32 within a few steps without the exported file being wrong."""
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def max_onnx_difference(
    policy: torch.nn.Module, obs: TensorDict, onnx_path: str, num_samples: int = 64, seed: int = 0
) -> float:
    """Run the same observations through the PyTorch policy and the exported ONNX file; return the largest
    difference between their actions.

    The first observation is used as is, and ``num_samples - 1`` noisy copies of it widen the check beyond a
    single input. A large difference means the export is wrong, e.g. it dropped the observation normalizer.

    A recurrent policy (an LSTM or GRU) is checked over ``RECURRENT_STEPS`` steps from an empty memory, the way a
    robot starts: the PyTorch policy's memory is cleared, the ONNX file gets zeros as ``h_in`` (and ``c_in``), and
    the memory it returns (``h_out``, ``c_out``) is fed back in, as a runtime does. The returned memory is compared
    too. The first step alone would not do: from an empty memory, the weights that carry memory between steps
    multiply zeros.
    """
    import onnxruntime as ort

    groups = list(policy.obs_groups)
    first = TensorDict({group: obs[group][:1].float().cpu() for group in groups}, batch_size=[1])
    flat = torch.cat([first[group] for group in groups], dim=-1)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((num_samples - 1, flat.shape[-1]), generator=generator)
    samples = torch.cat([flat, flat + noise])

    sizes = [first[group].shape[-1] for group in groups]
    device = next(policy.parameters()).device
    batch = TensorDict(dict(zip(groups, samples.split(sizes, dim=-1))), batch_size=[num_samples]).to(device)
    # The exported graph has a fixed batch size of 1, so the ONNX file runs the samples one at a time. A recurrent
    # graph's inputs after obs are its memory, matching its outputs after the actions in order.
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    obs_input, *memory_inputs = session.get_inputs()
    empty = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in memory_inputs}
    memories = [empty] * num_samples

    recurrent = policy.is_recurrent
    if recurrent:
        policy.reset()
    difference = 0.0
    for _ in range(RECURRENT_STEPS if recurrent else 1):
        with torch.inference_mode(), _full_float32():
            expected = [policy(batch).cpu().numpy()]
            if recurrent:
                memory = policy.get_hidden_state()
                # (layers, samples, size) per state: the hidden state, and the cell state of an LSTM.
                expected += [state.cpu().numpy() for state in (memory if isinstance(memory, tuple) else (memory,))]
        outputs = [
            session.run(None, {obs_input.name: sample[None].numpy(), **memories[i]}) for i, sample in enumerate(samples)
        ]
        difference = max(difference, np.abs(expected[0] - np.stack([out[0][0] for out in outputs])).max())
        for index, state in enumerate(expected[1:], start=1):
            actual = np.stack([out[index][:, 0] for out in outputs], axis=1)
            difference = max(difference, np.abs(state - actual).max())
        memories = [{i.name: out[1 + n] for n, i in enumerate(memory_inputs)} for out in outputs]
    if recurrent:
        policy.reset()
    return float(difference)
