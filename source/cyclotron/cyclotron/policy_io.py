"""What a policy's inputs and outputs mean on the robot, as the live environment resolves them.

The configs name joints and bodies by pattern (``joint_names=[".*_knee"]``); which joints those match, in which
order, and the gains they get come from the robot model the environment loads. Training records this resolution in
``code_state.yaml`` (``policy_io``) and ``--export`` resolves it again, so a robot model that reorders or renames
joints, or changes their gains, stops the export instead of sending the policy's actions to the wrong motors: the
check of the ONNX file against the checkpoint can't see this, since both run in the same environment.

Isaac Lab is only imported inside ``resolve_policy_io``, so the comparison runs without Isaac Sim and in tests.
"""

from __future__ import annotations

import math

# The per-joint action values compared by joint name, in the order they are reported (then the clip).
_ACTION_VALUES = ("scale", "offset", "stiffness", "damping")


def _per_joint(value, count: int) -> list[float]:
    """A term's resolved scale or offset (a float, or a tensor row per env) as one float per joint."""
    import torch

    if isinstance(value, torch.Tensor):
        return [float(v) for v in value[0]]
    return [float(value)] * count


def _configured_offset(term) -> list[float]:
    """A joint position action's offset as configured, one float per joint.

    With ``use_default_offset`` the live offset is the default pose of the one environment export builds, which
    startup events such as ``randomize_joint_default_pos`` perturb. Resolve the robot's configured
    ``init_state.joint_pos`` the way Isaac Lab builds the default pose instead.
    """
    if not term.cfg.use_default_offset:
        return _per_joint(term._offset, len(term._joint_names))
    from isaaclab.utils.string import resolve_matching_names_values

    asset = term._asset
    pose = [0.0] * asset.num_joints
    indices, _, values = resolve_matching_names_values(asset.cfg.init_state.joint_pos, asset.joint_names)
    for index, value in zip(indices, values):
        pose[index] = float(value)
    joint_ids = range(asset.num_joints) if isinstance(term._joint_ids, slice) else term._joint_ids
    return [pose[int(i)] for i in joint_ids]


def _joint_position_action():
    from isaaclab.envs.mdp.actions import JointPositionAction

    return JointPositionAction


def _resolve_actions(env) -> tuple[dict | None, str | None]:
    """The actions as joint position targets, one entry per action in order, or None and why they aren't."""
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return None, "The environment has no action manager (direct workflow)."
    joint_position_action = _joint_position_action()
    joint_names, scale, offset, clip, stiffness, damping = [], [], [], [], [], []
    clipped = False
    for name in manager.active_terms:
        term = manager.get_term(name)
        # Velocity, effort and relative position actions turn actions into something other than position targets.
        if not isinstance(term, joint_position_action):
            return None, f"Action term {name} ({type(term).__name__}) is not a joint position action."
        names = list(term._joint_names)
        joint_names += names
        scale += _per_joint(term._scale, len(names))
        offset += _configured_offset(term)
        # The configured gains, not the simulated ones, which startup randomization events can perturb.
        stiffness += [float(v) for v in term._asset.data.default_joint_stiffness[0, term._joint_ids]]
        damping += [float(v) for v in term._asset.data.default_joint_damping[0, term._joint_ids]]
        if term.cfg.clip is None:
            clip += [[None, None]] * len(names)
        else:
            clipped = True
            # None for an unclipped side: the resolver fills joints the config does not name with +-inf.
            clip += [[None if abs(side) == float("inf") else side for side in pair] for pair in term._clip[0].tolist()]
    actions = {"joint_names": joint_names, "scale": scale, "offset": offset, "clip": clip if clipped else None}
    return {**actions, "stiffness": stiffness, "damping": damping}, None


def _selected(ids, names, everything: list[str]) -> list[str] | None:
    """The names an entity config's resolved ``ids`` select (a slice is all of them, in the asset's order), or None
    when it selects none of this kind."""
    if names is None and ids == slice(None):
        return None
    return list(everything[ids]) if isinstance(ids, slice) else [everything[int(i)] for i in ids]


def _resolve_term(env, cfg) -> dict:
    """The joints and bodies an observation term reads, from its resolved entity configs (``SceneEntityCfg``)."""
    resolved = {}
    for param, value in (getattr(cfg, "params", None) or {}).items():
        if not hasattr(value, "joint_ids") or not hasattr(value, "body_ids"):
            continue
        asset = env.scene[value.name]
        found = {
            "joint_names": _selected(value.joint_ids, value.joint_names, list(getattr(asset, "joint_names", []))),
            "body_names": _selected(value.body_ids, value.body_names, list(getattr(asset, "body_names", []))),
        }
        found = {key: names for key, names in found.items() if names is not None}
        if found:
            resolved[param] = found
    return resolved


def _resolve_observations(env, agent: dict) -> dict:
    """The terms of the actor's observation groups in order, each with the joints and bodies it reads."""
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return {}
    model = "student" if agent.get("class_name") == "DistillationRunner" else "actor"
    groups = (agent.get("obs_groups") or {}).get(model) or ["policy"]
    terms = manager.active_terms
    return {
        group: {name: _resolve_term(env, cfg) for name, cfg in zip(terms[group], manager._group_obs_term_cfgs[group])}
        for group in groups
        if group in terms
    }


def resolve_policy_io(env, agent: dict) -> dict:
    """What the policy's inputs and outputs are on the robot, from the live environment (``env.unwrapped``).

    ``agent`` is the agent config as a dict; it names the actor's observation groups. Returns ``actions`` (joint
    names in action order, and per joint the scale, offset, ``[low, high]`` clip or None, stiffness and damping),
    or None with ``unsupported`` saying why, and ``observations``: per actor group, its terms in order with the joints
    and bodies each reads.
    """
    actions, unsupported = _resolve_actions(env)
    io = {"actions": actions, "observations": _resolve_observations(env, agent)}
    if unsupported:
        io["unsupported"] = unsupported
    return io


def observation_names(io: dict) -> list[str]:
    """The actor's observation terms in order, ``group/term`` when it has several groups."""
    groups = io.get("observations") or {}
    return [name if len(groups) == 1 else f"{group}/{name}" for group, terms in groups.items() for name in terms]


def _number(value) -> str:
    return "none" if value is None else f"{value:g}"


def _close(old, new) -> bool:
    if old is None or new is None:
        return old is new
    return math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-9)


def _order_changes(label: str, old: list[str], new: list[str], limit: int = 4) -> list[str]:
    """Lines describing how a list of joint or body names changed, position by position."""
    if old == new:
        return []
    if len(old) != len(new):
        removed = [name for name in old if name not in new]
        added = [name for name in new if name not in old]
        parts = [f"{len(old)} -> {len(new)}"]
        parts += [f"removed {', '.join(removed)}"] if removed else []
        parts += [f"added {', '.join(added)}"] if added else []
        return [f"{label}: {'; '.join(parts)}"]
    moved = [f"{label} {i}: {a} -> {b}" for i, (a, b) in enumerate(zip(old, new)) if a != b]
    return moved[:limit] + ([f"{label}: and {len(moved) - limit} more positions"] if len(moved) > limit else [])


def _by_joint(actions: dict, key: str) -> dict:
    names = actions.get("joint_names") or []
    values = actions.get(key)
    if key == "clip" and values is None:
        values = [[None, None]] * len(names)
    return dict(zip(names, values or []))


def _action_differences(old: dict | None, new: dict | None, unsupported: str | None) -> list[str]:
    if old is None:
        # Training couldn't describe them either; whether the action terms changed is for the configs to show.
        return []
    if new is None:
        return [f"actions: training resolved them as joint position targets, now: {unsupported}"]
    lines = _order_changes("action joint", old.get("joint_names") or [], new.get("joint_names") or [])
    # The values compared by joint name, so a reorder is reported once and not as every value changing.
    for key in (*_ACTION_VALUES, "clip"):
        before, after = _by_joint(old, key), _by_joint(new, key)
        for joint, value in before.items():
            if joint not in after:
                continue
            if key == "clip":
                if not all(_close(a, b) for a, b in zip(value, after[joint])):
                    old_clip, new_clip = (f"[{', '.join(map(_number, pair))}]" for pair in (value, after[joint]))
                    lines.append(f"clip {joint}: {old_clip} -> {new_clip}")
            elif not _close(value, after[joint]):
                lines.append(f"{key} {joint}: {_number(value)} -> {_number(after[joint])}")
    return lines


def policy_io_differences(recorded: dict, current: dict) -> list[str]:
    """How the policy's inputs and outputs the current environment resolves (``resolve_policy_io``) differ from the
    ones training recorded, one line each. Terms added or removed aren't reported here: that is a settings change,
    which the configs show."""
    lines = _action_differences(recorded.get("actions"), current.get("actions"), current.get("unsupported"))
    old_groups, new_groups = recorded.get("observations") or {}, current.get("observations") or {}
    for group, old_terms in old_groups.items():
        new_terms = new_groups.get(group) or {}
        for term, old_params in old_terms.items():
            if term not in new_terms:
                continue
            new_params = new_terms[term] or {}
            for param in dict.fromkeys([*(old_params or {}), *new_params]):
                old, new = (old_params or {}).get(param) or {}, new_params.get(param) or {}
                for key in ("joint_names", "body_names"):
                    label = f"observation {group}/{term} {param} {key.split('_')[0]}"
                    lines += _order_changes(label, old.get(key) or [], new.get(key) or [])
    return lines
