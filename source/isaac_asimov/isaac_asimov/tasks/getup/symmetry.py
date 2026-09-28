# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Left/right mirror symmetry for the Asimov 1 get-up task.

Entry point for rsl-rl 5.0.1 symmetry augmentation::

    RslRlSymmetryCfg(use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=mirror_getup)

rsl-rl calls ``mirror_getup(env=<RslRlVecEnvWrapper>, obs=<TensorDict[B]> | None, actions=<Tensor[B, 23]> | None)``
(``rsl_rl/algorithms/ppo.py:241-245, 321-323, 333-335``) and expects ``(obs_aug, actions_aug)`` of batch ``2B``
with the originals first. Every observation group in the TensorDict is mirrored (policy *and* critic), because the
critic is evaluated on the augmented batch (``ppo.py:263``).

Mirror conventions (sagittal-plane reflection ``M = diag(1, -1, 1)`` in the pelvis frame, x fwd / y left / z up):

* Joints: **every** joint of this URDF mirrors as ``q'[i] = -q[perm[i]]``, where ``perm`` swaps left/right names and
  maps ``waist_yaw_joint`` to itself. Verified from ``third_party/asimov-1/sim-model/urdf/asimov_1.urdf``: for each
  L/R pair the reflected axis ``M a_L`` equals ``a_R`` (and ``M a = a`` for the self-mapped waist yaw axis), hence
  ``R(a_R, q_R) = M R(a_L, q_L) M`` iff ``q_R = -q_L``. ``tests/getup/test_symmetry.py`` re-derives this from the URDF.
  Applies equally to absolute and default-relative positions, velocities, torques and actions (the default standing
  pose is mirror-invariant).
* Polar vectors (gravity, linear velocity, forces): ``(x, -y, z)``. Axial vectors (angular velocity, torques):
  ``(-x, y, -z)``. Quaternion ``(w, x, y, z)``: ``(w, -x, y, -z)``. Scalars such as heights: unchanged.
* Per-body L/R features: swap the left and right bodies (and mirror each body's vector components if any).

Observation layout
------------------
Offsets are computed at run time from the environment's ``ObservationManager`` (term names, flattened term dims and
term ``history_length``), never hard-coded. Isaac Lab flattens a term with history as ``buffer.reshape(N, -1)`` of a
``(N, H, d)`` buffer, oldest frame first (``isaaclab/managers/observation_manager.py:423-424``, ``circular_buffer.py:80-90``),
so term ``k`` occupies ``H*d`` consecutive entries laid out ``[t-H+1 (d) | ... | t (d)]``. The per-frame mirror map is
tiled over the ``H`` frames. Terms are concatenated along the last dim in ``active_terms`` order.

Mirror-rule registry
--------------------
Every observation term in every group must have a rule, looked up **by term name**; an unknown name raises
``KeyError`` at the first augmentation call (fail loudly rather than silently mis-mirror). Register new terms at
import time of the module that defines them, e.g. in ``tasks/getup/mdp/observations.py``::

    from isaac_asimov.tasks.getup import symmetry as sym
    sym.register_mirror_rule("my_scalar", sym.Invariant())
    sym.register_mirror_rule("my_base_frame_vec3", sym.SignedPerm(sign=(1, -1, 1)))
    sym.register_mirror_rule("my_per_joint_signed", sym.JointMirror(sign=-1.0))   # names from params["asset_cfg"]
    sym.register_mirror_rule("my_per_body_norm", sym.BodySwap())                 # names from params["sensor_cfg"]

Rule classes (all operate on ONE history frame of dimension ``d``):

* ``Invariant()``: identity.
* ``SignedPerm(sign, perm=None)``: ``out[i] = sign[i] * x[perm[i]]`` (``perm=None`` is identity).
* ``JointMirror(sign=-1.0, default_joint_names=None, from_action=False)``: per-joint vector. Joint order is taken
  from the term's resolved ``params["asset_cfg"]`` (``asset.joint_names[joint_ids]``, i.e. the true output order),
  or from the action term when ``from_action=True``, else ``default_joint_names``, else ``ASIMOV_1_JOINT_NAMES`` if
  ``d == 23``. ``sign=-1`` for signed joint quantities (q, dq, tau, actions); ``sign=+1`` for non-negative per-joint
  quantities (|tau|/tau_max, thermal load).
* ``BodySwap(component_signs=None, default_body_names=None)``: per-body features, body-major layout
  ``[b0c0, b0c1, .., b1c0, ..]``. Body order from the term's resolved ``params["sensor_cfg"]`` (contact sensor) or
  ``params["asset_cfg"]`` body ids, else ``default_body_names``. ``component_signs`` defaults to ``(1,)`` for 1
  component and ``(1, -1, 1)`` for 3 (polar vector); pass ``(-1, 1, -1)`` for axial vectors.
* ``ByDim({d: rule, ...}, default=None)``: choose the rule by per-frame dim.

Built-in rules cover the ``policy`` group and the ``critic`` terms (see ``_register_defaults``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

__all__ = [
    "ASIMOV_1_JOINT_NAMES",
    "SLOT_0_1",
    "SLOT_2_3",
    "SLOT_4_5",
    "GETUP_CATEGORIES",
    "FEET_BODY_NAMES",
    "TermContext",
    "MirrorRule",
    "Invariant",
    "SignedPerm",
    "JointMirror",
    "BodySwap",
    "ByDim",
    "mirror_name",
    "lr_swap_perm",
    "register_mirror_rule",
    "get_mirror_rule",
    "registered_terms",
    "build_term_map",
    "build_group_map",
    "action_map",
    "apply_map",
    "mirror_getup",
    "mirror_obs_group",
]

# ---------------------------------------------------------------------------------------------------------------------
# Robot constants. Copied (this module must import without Isaac Lab for CPU unit tests); tests/getup/test_symmetry.py
# asserts they equal assets/robots/asimov_1.py and tasks/locomotion/velocity_env_cfg.py.
# ---------------------------------------------------------------------------------------------------------------------

ASIMOV_1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
)

SLOT_0_1: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
)  # fmt: skip
SLOT_2_3: tuple[str, ...] = (
    "left_hip_yaw_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
)  # fmt: skip
SLOT_4_5: tuple[str, ...] = (
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "right_wrist_yaw_joint", "left_wrist_yaw_joint",
)  # fmt: skip

GETUP_CATEGORIES: tuple[str, ...] = (
    "supine", "prone", "side_left", "side_right", "sitting", "kneeling", "mid_fall", "standing",
)  # fmt: skip
"""Fixed category key order (logged keys and the category one-hot depend on it). ``side_left`` and ``side_right``
swap under mirroring."""

FEET_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")


# ---------------------------------------------------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------------------------------------------------

_LR_TOKEN = re.compile(r"(?<![A-Za-z0-9])(left|right|Left|Right|LEFT|RIGHT)(?![A-Za-z0-9])")
_LR_SWAP = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def mirror_name(name: str) -> str:
    """Swap left/right tokens in a joint/body name (tokens delimited by non-alphanumerics, e.g. ``left_knee_link``)."""
    return _LR_TOKEN.sub(lambda m: _LR_SWAP[m.group(1)], name)


def lr_swap_perm(names: Sequence[str]) -> list[int]:
    """Index permutation that maps each name to its mirrored counterpart within ``names``.

    Names without a left/right token map to themselves. Raises ``ValueError`` if a mirrored name is missing.
    """
    names = list(names)
    index = {n: i for i, n in enumerate(names)}
    if len(index) != len(names):
        raise ValueError(f"Duplicate names in {names}")
    perm = []
    for n in names:
        m = mirror_name(n)
        if m not in index:
            raise ValueError(f"Mirror partner '{m}' of '{n}' is not in {names}")
        perm.append(index[m])
    return perm


# ---------------------------------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TermContext:
    """What is known about one observation term when building its mirror map.

    Attributes:
        name: Term name (registry key).
        dim: Total flattened size of the term inside the group (history included).
        history: Number of history frames ``H`` (1 when the term has no history).
        joint_names: Resolved joint order of the term's output, if the term is joint-indexed.
        body_names: Resolved body order of the term's output, if the term is body-indexed.
        action_joint_names: Joint order of the action term (used by ``JointMirror(from_action=True)``).
        robot_joint_names: Articulation joint order (fallback for full-vector joint terms without ``asset_cfg``).
    """

    name: str
    dim: int
    history: int = 1
    joint_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None
    action_joint_names: tuple[str, ...] | None = None
    robot_joint_names: tuple[str, ...] | None = None


FrameMap = tuple[list[int], list[float]]


class MirrorRule:
    """Base class: maps one history frame of size ``frame_dim`` to ``(perm, sign)`` with ``out[i] = sign[i]*x[perm[i]]``."""

    def frame_map(self, ctx: TermContext, frame_dim: int) -> FrameMap:
        raise NotImplementedError


@dataclass(frozen=True)
class Invariant(MirrorRule):
    """Mirror-invariant quantity (heights, magnitudes, timers, flags)."""

    def frame_map(self, ctx: TermContext, frame_dim: int) -> FrameMap:
        return list(range(frame_dim)), [1.0] * frame_dim


@dataclass(frozen=True)
class SignedPerm(MirrorRule):
    """Fixed signed permutation of the frame: ``out[i] = sign[i] * x[perm[i]]``."""

    sign: tuple[float, ...]
    perm: tuple[int, ...] | None = None

    def frame_map(self, ctx: TermContext, frame_dim: int) -> FrameMap:
        perm = list(self.perm) if self.perm is not None else list(range(len(self.sign)))
        if len(self.sign) != frame_dim or len(perm) != frame_dim:
            raise ValueError(
                f"Mirror rule for '{ctx.name}' expects a frame of {len(self.sign)} values, got {frame_dim}"
                f" (term dim {ctx.dim}, history {ctx.history})."
            )
        return perm, [float(s) for s in self.sign]


@dataclass(frozen=True)
class JointMirror(MirrorRule):
    """Per-joint vector: swap L/R joints and multiply by ``sign`` (-1 for signed joint quantities on Asimov 1)."""

    sign: float = -1.0
    default_joint_names: tuple[str, ...] | None = None
    from_action: bool = False

    def frame_map(self, ctx: TermContext, frame_dim: int) -> FrameMap:
        # priority: resolved term order > rule default > articulation order (full vector) > ASIMOV_1_JOINT_NAMES
        names = ctx.action_joint_names if self.from_action else ctx.joint_names
        if names is None:
            names = self.default_joint_names
        if names is None and not self.from_action and ctx.robot_joint_names is not None:
            if len(ctx.robot_joint_names) == frame_dim:
                names = ctx.robot_joint_names
        if names is None and frame_dim == len(ASIMOV_1_JOINT_NAMES):
            names = ASIMOV_1_JOINT_NAMES
        if names is None:
            raise ValueError(f"Cannot resolve the joint order of term '{ctx.name}' (frame dim {frame_dim}).")
        if len(names) != frame_dim:
            raise ValueError(
                f"Term '{ctx.name}': {len(names)} joint names but a frame of {frame_dim} values"
                f" (term dim {ctx.dim}, history {ctx.history}). Names: {list(names)}"
            )
        return lr_swap_perm(names), [float(self.sign)] * frame_dim


@dataclass(frozen=True)
class BodySwap(MirrorRule):
    """Per-body features (body-major); swap L/R bodies and apply ``component_signs`` to each body's components."""

    component_signs: tuple[float, ...] | None = None
    default_body_names: tuple[str, ...] | None = None

    def frame_map(self, ctx: TermContext, frame_dim: int) -> FrameMap:
        names = ctx.body_names if ctx.body_names is not None else self.default_body_names
        if names is None:
            raise ValueError(f"Cannot resolve the body order of term '{ctx.name}' (frame dim {frame_dim}).")
        n = len(names)
        if frame_dim % n != 0:
            raise ValueError(f"Term '{ctx.name}': frame dim {frame_dim} is not a multiple of {n} bodies {list(names)}")
        c = frame_dim // n
        signs = self.component_signs
        if signs is None:
            if c == 1:
                signs = (1.0,)
            elif c == 3:
                signs = (1.0, -1.0, 1.0)
            else:
                raise ValueError(f"Term '{ctx.name}': give component_signs for {c} components per body.")
        if len(signs) != c:
            raise ValueError(f"Term '{ctx.name}': {len(signs)} component signs for {c} components per body.")
        body_perm = lr_swap_perm(names)
        perm, sign = [], []
        for b in range(n):
            for k in range(c):
                perm.append(body_perm[b] * c + k)
                sign.append(float(signs[k]))
        return perm, sign


@dataclass(frozen=True)
class ByDim(MirrorRule):
    """Select a rule by per-frame dim (for terms whose meaning depends on their size)."""

    rules: Mapping[int, MirrorRule] = field(default_factory=dict)
    default: MirrorRule | None = None

    def frame_map(self, ctx: TermContext, frame_dim: int) -> FrameMap:
        rule = self.rules.get(frame_dim, self.default)
        if rule is None:
            raise ValueError(
                f"Term '{ctx.name}': no mirror rule for frame dim {frame_dim} (known: {sorted(self.rules)})."
            )
        return rule.frame_map(ctx, frame_dim)


# ---------------------------------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------------------------------

_REGISTRY: dict[str, MirrorRule] = {}


def register_mirror_rule(term_name: str, rule: MirrorRule, *, override: bool = False) -> None:
    """Register ``rule`` for observation terms named ``term_name`` (in any group).

    Raises ``KeyError`` if the name is already registered with a different rule and ``override`` is False.
    Registering an identical rule again is a no-op (safe under module re-imports).
    """
    if not isinstance(rule, MirrorRule):
        raise TypeError(f"rule must be a MirrorRule, got {type(rule)}")
    existing = _REGISTRY.get(term_name)
    if existing is not None and existing != rule and not override:
        raise KeyError(f"Mirror rule for '{term_name}' already registered ({existing}); pass override=True.")
    _REGISTRY[term_name] = rule
    _invalidate_bound_maps()


def get_mirror_rule(term_name: str) -> MirrorRule:
    """Return the rule registered for ``term_name`` (``KeyError`` if none)."""
    try:
        return _REGISTRY[term_name]
    except KeyError:
        raise KeyError(
            f"No mirror rule for observation term '{term_name}'. Register one with"
            " isaac_asimov.tasks.getup.symmetry.register_mirror_rule(name, rule)."
            f" Registered: {sorted(_REGISTRY)}"
        ) from None


def registered_terms() -> list[str]:
    """Names with a registered mirror rule."""
    return sorted(_REGISTRY)


_POLAR = SignedPerm(sign=(1.0, -1.0, 1.0))
_AXIAL = SignedPerm(sign=(-1.0, 1.0, -1.0))
_QUAT_WXYZ = SignedPerm(sign=(1.0, -1.0, 1.0, -1.0))
_WRENCH_B = SignedPerm(sign=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0))  # [force (polar), torque (axial)]


def _register_defaults() -> None:
    # --- policy group and its noise-free critic copies
    _REGISTRY["base_ang_vel"] = _AXIAL
    _REGISTRY["projected_gravity"] = _POLAR
    for kind in ("pos", "vel"):
        _REGISTRY[f"joint_{kind}_slot01"] = JointMirror(-1.0, SLOT_0_1)
        _REGISTRY[f"joint_{kind}_slot23"] = JointMirror(-1.0, SLOT_2_3)
        _REGISTRY[f"joint_{kind}_slot45"] = JointMirror(-1.0, SLOT_4_5)
    for name in ("actions", "last_action", "filtered_actions"):
        _REGISTRY[name] = JointMirror(-1.0, from_action=True)
    # --- walking only (kept so the same function can mirror the walking layout)
    _REGISTRY["command"] = SignedPerm(sign=(1.0, -1.0, -1.0))  # (vx, vy, wz)
    # --- critic privileged terms
    for name in ("base_lin_vel", "root_lin_vel_b"):
        _REGISTRY[name] = _POLAR
    _REGISTRY["root_ang_vel_b"] = _AXIAL
    for name in ("root_quat", "root_quat_w", "base_quat"):
        _REGISTRY[name] = _QUAT_WXYZ  # Isaac Lab (w, x, y, z); world-frame reflection about the x-z plane
    for name in ("pelvis_height", "base_height", "root_height", "base_pos_z", "torso_height", "max_height_ep"):
        _REGISTRY[name] = Invariant()
    for name in ("joint_pos", "joint_vel", "joint_pos_rel", "joint_vel_rel", "joint_effort", "applied_torque"):
        _REGISTRY[name] = JointMirror(-1.0)
    # effort saturation: signed tau/tau_max flips with the joint; |tau|/tau_max only permutes
    _REGISTRY["effort_saturation"] = JointMirror(-1.0)
    _REGISTRY["effort_saturation_abs"] = JointMirror(+1.0)
    # thermal proxy: non-negative per-joint load, or a scalar summary
    _REGISTRY["thermal_proxy"] = ByDim({1: Invariant()}, default=JointMirror(+1.0))
    # per-body contact-force norms (feet, knees, forearms, wrists, torso, pelvis, head): L/R body swap
    for name in ("body_contact_norms", "contact_force_norms", "body_contact_forces", "body_contact"):
        _REGISTRY[name] = BodySwap()
    for name in ("foot_contact", "foot_height", "foot_air_time", "foot_contact_forces", "feet_contact"):
        _REGISTRY[name] = BodySwap(default_body_names=FEET_BODY_NAMES)
    # assist harness: scalar magnitude, vec3 force, or [force, torque] (base or world frame, reflection about x-z)
    _REGISTRY["assist_force"] = ByDim({1: Invariant(), 3: _POLAR, 6: _WRENCH_B})
    for name in ("assist_enabled", "assist_scale", "assist_target_height", "policy_active", "stand_timer",
                 "time_since_control", "effort_scale", "action_bound", "limp"):  # fmt: skip
        _REGISTRY[name] = Invariant()
    # category one-hot in the fixed order: side_left <-> side_right
    cat_perm = tuple(lr_swap_perm(GETUP_CATEGORIES))
    _REGISTRY["category_onehot"] = SignedPerm(sign=(1.0,) * len(GETUP_CATEGORIES), perm=cat_perm)


# ---------------------------------------------------------------------------------------------------------------------
# Pure map builders (no Isaac Lab needed; unit-tested)
# ---------------------------------------------------------------------------------------------------------------------


def build_term_map(ctx: TermContext, rule: MirrorRule | None = None) -> FrameMap:
    """Signed permutation over the full (history-stacked) term: the frame map tiled over ``ctx.history`` frames."""
    rule = rule if rule is not None else get_mirror_rule(ctx.name)
    h = max(int(ctx.history), 1)
    if ctx.dim % h != 0:
        raise ValueError(f"Term '{ctx.name}': dim {ctx.dim} is not divisible by history length {h}.")
    d = ctx.dim // h
    fperm, fsign = rule.frame_map(ctx, d)
    perm, sign = [], []
    for t in range(h):
        perm.extend(t * d + p for p in fperm)
        sign.extend(fsign)
    return perm, sign


def build_group_map(ctxs: Sequence[TermContext], rules: Mapping[str, MirrorRule] | None = None) -> FrameMap:
    """Signed permutation over a concatenated observation group, terms in ``ctxs`` order."""
    perm, sign, offset = [], [], 0
    for ctx in ctxs:
        rule = rules[ctx.name] if rules is not None and ctx.name in rules else None
        tperm, tsign = build_term_map(ctx, rule)
        perm.extend(offset + p for p in tperm)
        sign.extend(tsign)
        offset += ctx.dim
    _check_involution(perm, sign, "group")
    return perm, sign


def action_map(joint_names: Sequence[str] = ASIMOV_1_JOINT_NAMES) -> FrameMap:
    """Signed permutation for the joint-position action vector (every Asimov 1 joint flips sign)."""
    return lr_swap_perm(joint_names), [-1.0] * len(joint_names)


def apply_map(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """``out[..., i] = sign[i] * x[..., perm[i]]``."""
    return x[..., perm] * sign


def _check_involution(perm: Sequence[int], sign: Sequence[float], what: str) -> None:
    for i, p in enumerate(perm):
        if perm[p] != i or sign[i] * sign[p] != 1.0:
            raise ValueError(f"Mirror map for {what} is not an involution at index {i} (perm {p}).")


# ---------------------------------------------------------------------------------------------------------------------
# Environment binding (Isaac Lab ManagerBasedRLEnv); cached per env
# ---------------------------------------------------------------------------------------------------------------------

_BIND_ATTR = "_getup_symmetry_maps"
_BOUND_ENVS: list = []


def _invalidate_bound_maps() -> None:
    for env in _BOUND_ENVS:
        if hasattr(env, _BIND_ATTR):
            delattr(env, _BIND_ATTR)
    _BOUND_ENVS.clear()


def _unwrap(env):
    return getattr(env, "unwrapped", env)


def _resolved_names(all_names: Sequence[str], ids) -> tuple[str, ...]:
    if isinstance(ids, slice):
        return tuple(list(all_names)[ids])
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tuple(all_names[int(i)] for i in ids)


def _term_joint_names(env, params: Mapping) -> tuple[str, ...] | None:
    cfg = params.get("asset_cfg") if isinstance(params, Mapping) else None
    if cfg is None or getattr(cfg, "joint_ids", None) is None:
        return None
    try:
        asset = env.scene[cfg.name]
        return _resolved_names(asset.joint_names, cfg.joint_ids)
    except Exception:
        return None


def _term_body_names(env, params: Mapping) -> tuple[str, ...] | None:
    if not isinstance(params, Mapping):
        return None
    for key in ("sensor_cfg", "asset_cfg"):
        cfg = params.get(key)
        if cfg is None or getattr(cfg, "body_ids", None) is None:
            continue
        if key == "asset_cfg" and isinstance(cfg.body_ids, slice):
            continue  # default asset_cfg: not a body-indexed term
        try:
            entity = env.scene.sensors[cfg.name] if key == "sensor_cfg" else env.scene[cfg.name]
            return _resolved_names(entity.body_names, cfg.body_ids)
        except Exception:
            continue
    return None


DEFAULT_ACTION_TERM = "joint_pos"
"""Name of the get-up joint-position action term."""


def _default_action_term(am) -> str | None:
    """``joint_pos`` if present, else the first term with ``action_dim > 0`` (e.g. skips the 0-dim ``assist`` term)."""
    names = list(am.active_terms)
    if DEFAULT_ACTION_TERM in names:
        return DEFAULT_ACTION_TERM
    for name in names:
        try:
            if int(am.get_term(name).action_dim) > 0:
                return name
        except Exception:
            continue
    return None


def _action_joint_names(env, action_name: str | None = None) -> tuple[str, ...] | None:
    am = getattr(env, "action_manager", None)
    if am is None:
        return None
    try:
        if action_name is None:
            action_name = _default_action_term(am)
            if action_name is None:
                return None
        term = am.get_term(action_name)
    except Exception:
        return None
    names = getattr(term, "_joint_names", None)
    return tuple(names) if names is not None else None


def _group_contexts(env, group: str) -> list[TermContext]:
    om = env.observation_manager
    if group not in om.active_terms:
        raise KeyError(f"Observation group '{group}' not found in the observation manager ({list(om.active_terms)}).")
    if not om.group_obs_concatenate[group]:
        raise NotImplementedError(f"Mirroring needs concatenated observation groups; '{group}' is not.")
    concat_dim = getattr(om, "_group_obs_concatenate_dim", {}).get(group, -1)
    if concat_dim not in (-1, 1):
        raise NotImplementedError(f"Group '{group}' is concatenated along dim {concat_dim}; expected the last dim.")
    term_cfgs = om._group_obs_term_cfgs[group]  # private in Isaac Lab b0542fe; holds the resolved history_length
    try:
        robot_joint_names = tuple(env.scene["robot"].joint_names)
    except Exception:
        robot_joint_names = None
    ctxs = []
    for name, dim, cfg in zip(om.active_terms[group], om.group_obs_term_dim[group], term_cfgs):
        history = int(getattr(cfg, "history_length", 0) or 0)
        if history > 0 and not getattr(cfg, "flatten_history_dim", True):
            raise NotImplementedError(f"Term '{group}/{name}' has an unflattened history dim.")
        params = getattr(cfg, "params", {}) or {}
        ctxs.append(
            TermContext(
                name=name,
                dim=int(math.prod(dim)),
                history=max(history, 1),
                joint_names=_term_joint_names(env, params),
                body_names=_term_body_names(env, params),
                action_joint_names=_action_joint_names(env, params.get("action_name")),
                robot_joint_names=robot_joint_names,
            )
        )
    return ctxs


class _EnvMaps:
    """Mirror maps bound to one environment, with per-device tensor caches."""

    def __init__(self, env):
        self.env = env
        self.groups: dict[str, FrameMap] = {}
        names = _action_joint_names(env)
        if names is None:
            print("[GetUpSymmetry] WARNING: action joint order not resolvable; assuming ASIMOV_1_JOINT_NAMES.")
        self.action: FrameMap = action_map(names if names is not None else ASIMOV_1_JOINT_NAMES)
        self._tensors: dict[tuple[str, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def group_map(self, group: str) -> FrameMap:
        if group not in self.groups:
            ctxs = _group_contexts(self.env, group)
            self.groups[group] = build_group_map(ctxs)
            print(
                f"[GetUpSymmetry] group '{group}': "
                + ", ".join(f"{c.name}[{c.dim // c.history}x{c.history}]" for c in ctxs)
                + f" -> dim {sum(c.dim for c in ctxs)}"
            )
        return self.groups[group]

    def tensors(self, key: str, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        cache_key = (key, device, dtype)
        if cache_key not in self._tensors:
            perm, sign = self.action if key == "__actions__" else self.group_map(key)
            self._tensors[cache_key] = (
                torch.tensor(perm, dtype=torch.long, device=device),
                torch.tensor(sign, dtype=dtype, device=device),
            )
        return self._tensors[cache_key]


def _bound(env) -> _EnvMaps:
    env = _unwrap(env)
    maps = getattr(env, _BIND_ATTR, None)
    if maps is None:
        maps = _EnvMaps(env)
        setattr(env, _BIND_ATTR, maps)
        _BOUND_ENVS.append(env)
    return maps


# ---------------------------------------------------------------------------------------------------------------------
# rsl-rl entry points
# ---------------------------------------------------------------------------------------------------------------------


def mirror_obs_group(env, group: str, x: torch.Tensor) -> torch.Tensor:
    """Mirror one concatenated observation group tensor ``[..., D]``."""
    perm, sign = _bound(env).tensors(group, x.device, x.dtype)
    if x.shape[-1] != perm.numel():
        raise ValueError(f"Group '{group}' has {x.shape[-1]} features but the mirror map covers {perm.numel()}.")
    return apply_map(x, perm, sign)


@torch.no_grad()
def mirror_getup(env=None, obs=None, actions=None):
    """rsl-rl 5.0.1 symmetry data-augmentation function for the get-up task.

    Args:
        env: The rsl-rl VecEnv (``RslRlVecEnvWrapper``); its ``unwrapped`` env provides the managers.
        obs: ``TensorDict`` of observation groups with batch size ``[B]``, or None.
        actions: Action tensor ``[B, num_actions]``, or None.

    Returns:
        ``(obs_aug, actions_aug)``, each with batch ``2B`` (original first, mirrored second), or None where the
        corresponding input was None.
    """
    obs_aug = actions_aug = None
    if obs is not None:
        mirrored = obs.clone()
        for group in obs.keys():
            mirrored[group] = mirror_obs_group(env, group, obs[group])
        obs_aug = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        perm, sign = _bound(env).tensors("__actions__", actions.device, actions.dtype)
        if actions.shape[-1] != perm.numel():
            raise ValueError(f"Actions have {actions.shape[-1]} dims but the mirror map covers {perm.numel()}.")
        actions_aug = torch.cat([actions, apply_map(actions, perm, sign)], dim=0)
    return obs_aug, actions_aug


_register_defaults()
