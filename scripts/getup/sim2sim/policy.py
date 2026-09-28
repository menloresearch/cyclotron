# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""ONNX policy runtime, plus zero/random fallbacks so the rest of the pipeline is testable without a checkpoint.

Also reads the deploy-style ONNX custom metadata (``scripts/getup/export_onnx.py``'s get-up
contract) if present. For a get-up checkpoint this is the *authoritative* source for the action term's live
``s_j``/``beta`` -- the exporter bakes the trained ``bound_scale`` directly into ``action_s_j``
(``s_j = scale_torque_factor * bound_scale * tau_max_nominal / kp_nominal``), so reading ``action_s_j`` +
``action_beta`` from the metadata is correct *without* needing to separately know or apply ``bound_scale`` --
see ``export_onnx.py``. Reading them from the metadata avoids assuming a bound the checkpoint was not trained with.
Values are reordered to :data:`constants.ASIMOV_1_JOINT_NAMES` order via the metadata's own
``joint_order`` key if present and different (mirrors the same defensive reordering `compare_isaac.py` already
does for Isaac's own articulation order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C


@dataclass
class OnnxMetadata:
    joint_stiffness: np.ndarray | None = None
    joint_damping: np.ndarray | None = None
    default_joint_pos: np.ndarray | None = None
    action_scale: np.ndarray | None = None
    action_s_j: np.ndarray | None = None  # trained s_j, bound_scale already baked in -- see module docstring
    action_beta: float | None = None
    action_lpf_alpha: float | None = None
    action_mode: str | None = None
    obs_history_length: int | None = None
    raw: dict[str, str] | None = None

    def has_action_contract(self) -> bool:
        return self.action_s_j is not None and self.action_beta is not None


def _parse_meta(raw: dict[str, str]) -> OnnxMetadata:
    joint_order = raw.get("joint_order")
    order = joint_order.split(",") if joint_order else None

    def _floats(key: str) -> np.ndarray | None:
        if key not in raw:
            return None
        vals = np.array([float(x) for x in raw[key].split(",")], dtype=float)
        if order is not None and len(vals) == C.NUM_JOINTS and order != C.ASIMOV_1_JOINT_NAMES:
            idx = [order.index(n) for n in C.ASIMOV_1_JOINT_NAMES]
            vals = vals[idx]
        return vals

    return OnnxMetadata(
        joint_stiffness=_floats("joint_stiffness"),
        joint_damping=_floats("joint_damping"),
        default_joint_pos=_floats("default_joint_pos"),
        action_scale=_floats("action_scale"),
        action_s_j=_floats("action_s_j"),
        action_beta=float(raw["action_beta"]) if "action_beta" in raw else None,
        action_lpf_alpha=float(raw["action_lpf_alpha"]) if "action_lpf_alpha" in raw else None,
        action_mode=raw.get("action_mode"),
        obs_history_length=int(raw["obs_history_length"]) if "obs_history_length" in raw else None,
        raw=dict(raw),
    )


class OnnxPolicy:
    """Wraps ``onnxruntime.InferenceSession`` for a single-input/single-output policy (the deploy
    contract: exactly 1 input tensor, 1 output tensor)."""

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"expected exactly 1 input/1 output, got {len(inputs)} inputs, {len(outputs)} outputs")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_dim = inputs[0].shape[-1]
        self.output_dim = outputs[0].shape[-1]
        self.metadata = _parse_meta(dict(self.session.get_modelmeta().custom_metadata_map))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        (out,) = self.session.run([self.output_name], {self.input_name: obs})
        return out[0].astype(float)


class ZeroPolicy:
    """Always outputs zero action. Useful to check the sim/actuator/obs pipeline holds a standing pose without
    exploding, independent of any checkpoint (a "stands / does not explode" pipeline check)."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=float)


class RandomPolicy:
    """Small-amplitude random actions, for exercising delay/noise/LPF code paths without a checkpoint."""

    def __init__(self, action_dim: int, scale: float = 0.05, rng: np.random.Generator | None = None):
        self.action_dim = action_dim
        self.scale = scale
        self.rng = rng or np.random.default_rng()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-self.scale, self.scale, size=self.action_dim)


def load_policy(onnx_path: str | None, action_dim: int, kind: str = "onnx", **kwargs):
    if onnx_path:
        return OnnxPolicy(onnx_path)
    if kind == "zero":
        return ZeroPolicy(action_dim)
    if kind == "random":
        return RandomPolicy(action_dim, **kwargs)
    raise ValueError("no --onnx given and --fallback-policy is not 'zero'/'random'")
