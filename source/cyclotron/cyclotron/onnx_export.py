"""Helpers for ``./cyclotron.sh --export`` that need PyTorch but not Isaac Sim, so they can be tested on their own."""

from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np
import torch
from tensordict import TensorDict

# The run's training config, copied next to policy.onnx so the export folder has the same files as a shared Hub repo.
BUNDLE_YAMLS = ("env.yaml", "agent.yaml")
# The record of the code the run was trained with; runs trained before it existed don't have one.
OPTIONAL_BUNDLE_YAMLS = ("code_state.yaml",)

# Schema of the deploy metadata attached to policy.onnx; firmware should refuse versions it does not know.
DEPLOY_METADATA_VERSION = "1"


def deploy_metadata(
    joint_names: list[str],
    action_scale: list[float],
    action_offset: list[float],
    action_clip: list[list[float]] | None,
    joint_stiffness: list[float],
    joint_damping: list[float],
    sim_dt: float,
    decimation: int,
    observation_names: list[str],
    trained_commit: str | None,
) -> dict[str, str]:
    """The deployment contract of a policy, as the strings stored in ONNX metadata.

    Only what a runtime must agree with the policy about before driving a robot with it: the joint order the actions
    are in, the affine that turns raw actions into position targets (``target = action * scale + offset``, then an
    optional ``[low, high]`` clip per joint), the PD gains the targets were trained to be tracked with, the rate the
    policy was trained to run at, the ordered observation terms its input is built from, and the training commit for
    traceability. Training settings stay in the yaml files next to the ONNX; they are not deployment inputs.
    """
    per_joint = [action_scale, action_offset, joint_stiffness, joint_damping]
    if any(len(values) != len(joint_names) for values in per_joint):
        raise ValueError("action_scale, action_offset, joint_stiffness and joint_damping need one entry per action")
    if action_clip is not None and len(action_clip) != len(joint_names):
        raise ValueError("action_clip must have one [low, high] pair per action")
    metadata = {
        "deploy_metadata_version": DEPLOY_METADATA_VERSION,
        "joint_names": json.dumps(joint_names),
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


def existing_export_note(run_dir: str, output_dir: str) -> str | None:
    """What ``--export`` is about to overwrite, or None when the output folder has no policy.onnx yet.

    Export writes its artifacts together, so the existing policy.onnx's file time is when that export was made;
    checkpoints written after it mean the old export was not of the run's latest checkpoint.
    """
    existing = os.path.join(output_dir, "policy.onnx")
    if not os.path.isfile(existing):
        return None
    exported_at = os.path.getmtime(existing)
    newer = [
        name
        for name in os.listdir(run_dir)
        if re.fullmatch(r"model_\d+\.pt", name) and os.path.getmtime(os.path.join(run_dir, name)) > exported_at
    ]
    if not newer:
        return "Overwriting the run's existing export."
    latest = max(newer, key=lambda name: int(name[len("model_") : -len(".pt")]))
    return f"Overwriting an export made before {latest} was written, so it was not of the run's latest checkpoint."


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


def max_onnx_difference(
    policy: torch.nn.Module, obs: TensorDict, onnx_path: str, num_samples: int = 64, seed: int = 0
) -> float:
    """Run the same observations through the PyTorch policy and the exported ONNX file; return the largest
    difference between their actions.

    The first observation is used as is, and ``num_samples - 1`` noisy copies of it widen the check beyond a
    single input. A large difference means the export is wrong, e.g. it dropped the observation normalizer.
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
    with torch.inference_mode():
        expected = policy(batch).cpu().numpy()

    # The exported graph has a fixed batch size of 1, so run the samples one at a time.
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    actual = [session.run(None, {input_name: sample[None].numpy()})[0][0] for sample in samples]
    return float(np.abs(expected - np.stack(actual)).max())
