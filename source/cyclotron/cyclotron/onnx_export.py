"""Helpers for ``./cyclotron.sh --export`` that need PyTorch but not Isaac Sim, so they can be tested on their own."""

from __future__ import annotations

import os
import shutil

import numpy as np
import torch
from tensordict import TensorDict

# The run's training config, copied next to policy.onnx so the export folder has the same files as a shared Hub repo.
BUNDLE_YAMLS = ("env.yaml", "agent.yaml")
# The record of the code the run was trained with; runs trained before it existed don't have one.
OPTIONAL_BUNDLE_YAMLS = ("code_state.yaml",)


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
