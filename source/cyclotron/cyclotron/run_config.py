"""Set a run's policy settings back to the values its ``env.yaml`` and ``agent.yaml`` saved, for ``--export``.

Export builds the environment from the task's config in the current code, which may have changed since the run was
trained. Before the environment is built, the settings that decide what the policy sees and does (the ones
``code_state.policy_interface`` compares: the actor's observations, the actions, the default pose, the actuators,
the timing and the actor network) are set back to the run's saved values, so the exported policy and its deploy
metadata describe the run as it was trained. Everything else (rewards, terrain, events, commands, the robot model
file) only shapes training or depends on the machine, and stays as the code has it.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import re

import yaml

from cyclotron.code_state import _IGNORED_KEYS


class _RunConfigLoader(yaml.SafeLoader):
    """Reads the yaml Isaac Lab's ``dump_yaml`` writes: plain values, tuples and slices. Other Python objects are
    refused, since a run can come from another machine."""


_RunConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True))
)
_RunConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)

# A function or class as Isaac Lab writes it, "module.path:name".
_FUNCTION = re.compile(r"([A-Za-z_][\w.]*):([A-Za-z_]\w*)")
# Marks a setting the run's yaml doesn't have, which then keeps the code's value.
_MISSING = object()


def load_run_configs(run_dir: str) -> tuple[dict, dict] | None:
    """The run's saved ``params/env.yaml`` and ``params/agent.yaml``, or None if it doesn't have both."""
    paths = [os.path.join(run_dir, "params", name) for name in ("env.yaml", "agent.yaml")]
    if not all(os.path.isfile(path) for path in paths):
        return None
    configs = []
    for path in paths:
        with open(path) as f:
            configs.append(yaml.load(f, Loader=_RunConfigLoader) or {})
    return configs[0], configs[1]


def _get(config: dict, *keys):
    for key in keys:
        if not isinstance(config, dict) or key not in config:
            return _MISSING
        config = config[key]
    return config


def _child(name: str, key) -> str:
    return f"{name}.{key}" if re.fullmatch(r"\w+", str(key)) else f"{name}[{key}]"


def _is_config(value) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _default(cls, key: str):
    """A config class's default for ``key``. Isaac Lab's ``configclass`` stores every default as a factory."""
    field = cls.__dataclass_fields__.get(key)
    if field is None:
        return None
    if field.default is not dataclasses.MISSING:
        return field.default
    return field.default_factory() if field.default_factory is not dataclasses.MISSING else None


class _Restorer:
    """Sets config objects back to saved values, collecting the saved settings the current code can't take."""

    def __init__(self, overridden):
        self.overridden = tuple(overridden)
        self.problems = []

    def _is_overridden(self, name: str) -> bool:
        return any(name == path or name.startswith((f"{path}.", f"{path}[")) for path in self.overridden)

    def value(self, current, saved, name: str):
        """The value to store at ``name``: ``saved``, merged into ``current`` where that is a config."""
        if saved is _MISSING or self._is_overridden(name):
            return current
        if _is_config(current) and isinstance(saved, dict):
            current = self._config_class(current, saved, name)
            for key, item in saved.items():
                if key in _IGNORED_KEYS:
                    continue
                if not hasattr(current, key):
                    self.problems.append(f"{_child(name, key)}: the run has this setting, the current code doesn't")
                    continue
                setattr(current, key, self.value(getattr(current, key), item, _child(name, key)))
            return current
        if isinstance(current, dict) and isinstance(saved, dict):
            # The run's entries replace the code's, e.g. the default pose per joint pattern. Configs held in a dict
            # (e.g. actuator groups) are merged, since one can't be built back from its yaml.
            holds_configs = any(_is_config(item) for item in current.values())
            restored = {}
            for key, item in saved.items():
                if key in current:
                    restored[key] = self.value(current[key], item, _child(name, key))
                elif holds_configs and isinstance(item, dict):
                    self.problems.append(f"{_child(name, key)}: the run has this entry, the current code doesn't")
                else:
                    restored[key] = item
            restored.update(
                {k: v for k, v in current.items() if k not in saved and self._is_overridden(_child(name, k))}
            )
            return restored
        if isinstance(saved, str) and _FUNCTION.fullmatch(saved) and (callable(current) or isinstance(current, str)):
            return self._function(current, saved, name)
        if isinstance(saved, dict) and ("func" in saved or "class_type" in saved):
            self.problems.append(f"{name}: the run used this term, the current code turned it off")
            return current
        if isinstance(saved, list) and isinstance(current, (list, tuple)) and any(_is_config(i) for i in current):
            if len(saved) != len(current):
                self.problems.append(f"{name}: the run has {len(saved)} entries, the current code {len(current)}")
                return current
            return type(current)(self.value(c, s, f"{name}[{i}]") for i, (c, s) in enumerate(zip(current, saved)))
        if isinstance(saved, list) and isinstance(current, tuple):
            return tuple(saved)
        return saved

    def terms(self, container, saved, name: str) -> None:
        """Restore an observation group or the actions; a term the run didn't have is turned off."""
        if saved is _MISSING or self._is_overridden(name):
            return
        self.value(container, saved, name)
        for key, term in list(vars(container).items()):
            if _is_config(term) and key not in saved and not self._is_overridden(_child(name, key)):
                setattr(container, key, None)

    def _config_class(self, current, saved: dict, name: str):
        """A config whose ``class_name`` changed since training (e.g. the actor from ``MLPModel`` to ``RNNModel``) is
        replaced by a new instance of the config class for the saved one, found among the current class's family."""
        wanted = saved.get("class_name")
        if not isinstance(wanted, str) or getattr(current, "class_name", wanted) == wanted:
            return current
        root = next(
            c for c in reversed(type(current).__mro__) if "class_name" in getattr(c, "__dataclass_fields__", {})
        )
        family, index = [root], 0
        while index < len(family):
            family += [c for c in family[index].__subclasses__() if c not in family]
            index += 1
        for cls in family:
            if _default(cls, "class_name") == wanted:
                return cls()
        self.problems.append(f"{name}.class_name: the run used {wanted}, which no config class in the current code has")
        return current

    def _function(self, current, saved: str, name: str):
        module, attr = _FUNCTION.fullmatch(saved).groups()
        try:
            found = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            found = None
        if found is not None:
            return saved if isinstance(current, str) else found
        current_name = current.split(":")[-1] if isinstance(current, str) else getattr(current, "__name__", None)
        if current_name == attr:
            # Moved or renamed since training, e.g. from the package's old name isaac_asimov.
            return current
        self.problems.append(f"{name}: the run used {saved}, which the current code doesn't have")
        return current


def restore_policy_settings(env_cfg, agent_cfg, saved_env: dict, saved_agent: dict, overridden=()) -> list[str]:
    """Set the policy settings of ``env_cfg`` and ``agent_cfg`` (the task's configs from the code) back to the run's
    saved ones, in place. Settings under ``overridden`` (Hydra override paths given on the command line, such as
    ``env.actions.joint_pos.scale``) keep the value they were given.

    Returns the saved settings the current code can't take, one line each: any of them means the current code can
    no longer build the run's policy.
    """
    restore = _Restorer(overridden)
    model = "student" if saved_agent.get("class_name") == "DistillationRunner" else "actor"
    groups = _get(saved_agent, "obs_groups", model)
    for group in ["policy"] if groups is _MISSING or not groups else groups:
        name = f"env.observations.{group}"
        if getattr(env_cfg.observations, group, None) is None:
            restore.problems.append(f"{name}: the run's policy reads this observation group, the current code doesn't")
            continue
        restore.terms(getattr(env_cfg.observations, group), _get(saved_env, "observations", group), name)
    restore.terms(env_cfg.actions, _get(saved_env, "actions"), "env.actions")

    robot = env_cfg.scene.robot
    saved_robot = _get(saved_env, "scene", "robot")
    robot.init_state.joint_pos = restore.value(
        robot.init_state.joint_pos, _get(saved_robot, "init_state", "joint_pos"), "env.scene.robot.init_state.joint_pos"
    )
    robot.actuators = restore.value(robot.actuators, _get(saved_robot, "actuators"), "env.scene.robot.actuators")
    env_cfg.sim.dt = restore.value(env_cfg.sim.dt, _get(saved_env, "sim", "dt"), "env.sim.dt")
    env_cfg.decimation = restore.value(env_cfg.decimation, _get(saved_env, "decimation"), "env.decimation")

    if groups is not _MISSING:
        obs_groups = dict(agent_cfg.obs_groups or {})
        obs_groups[model] = restore.value(obs_groups.get(model), groups, f"agent.obs_groups.{model}")
        agent_cfg.obs_groups = obs_groups
    if hasattr(agent_cfg, model):
        setattr(agent_cfg, model, restore.value(getattr(agent_cfg, model), _get(saved_agent, model), f"agent.{model}"))
    agent_cfg.clip_actions = restore.value(
        agent_cfg.clip_actions, _get(saved_agent, "clip_actions"), "agent.clip_actions"
    )
    return restore.problems
