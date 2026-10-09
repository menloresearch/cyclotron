"""A run's ``env.yaml`` and ``agent.yaml``: reading them, the policy settings in them, and setting those back for
``--export``.

Isaac Lab saves both configs into the run's ``params/`` at train time. The settings that decide what the policy sees
and does (``policy_interface``: the actor's observations, the actions, the default pose, the actuators, the timing and
the actor network) are compared with the current code before ``--export`` and ``--play`` use the run, so a changed
observation, action or network is reported instead of crashing or silently changing what the policy sees.

Export builds the environment from the task's config in the current code, which may have changed since the run was
trained. Before the environment is built, these settings are set back to the run's saved values, so the exported
policy and its deploy metadata describe the run as it was trained. Everything else (rewards, terrain, events,
commands, the robot model file) only shapes training or depends on the machine, and stays as the code has it. Nothing
is overridden on top: the run's files are the only source of these settings.

A run without these files can have them written from the current code, marked as generated, so a policy is always
exported with both.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import re
from collections import Counter

import yaml

# -- Reading the configs Isaac Lab saves ------------------------------------------------------------------------

RUN_CONFIGS = ("env.yaml", "agent.yaml")


class _ConfigLoader(yaml.SafeLoader):
    """Reads the yaml Isaac Lab's ``dump_yaml`` writes: tuples and slices are rebuilt; any other Python object is
    never constructed, since a run can come from another machine, and reads as its tag (``!!python/...``)."""


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True))
)
_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)
_ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: f"!!python/{suffix}")


def load_config(text: str) -> dict:
    return yaml.load(text, Loader=_ConfigLoader) or {}


def load_run_configs(run_dir: str) -> tuple[dict, dict]:
    """The run's saved ``params/env.yaml`` and ``params/agent.yaml``; an empty dict for a file it doesn't have."""
    configs = []
    for name in RUN_CONFIGS:
        path = os.path.join(run_dir, "params", name)
        if os.path.isfile(path):
            with open(path) as f:
                configs.append(load_config(f.read()))
        else:
            configs.append({})
    return configs[0], configs[1]


def normalize_config(config: dict) -> dict:
    """Pass a config from ``class_to_dict`` through the same dump as ``dump_yaml``, so it reads like a saved one."""
    return load_config(yaml.dump(config, default_flow_style=False, sort_keys=False))


# -- What the policy sees and does -------------------------------------------------------------------------------

# Settings inside the compared sections that only shape training, or that the play tasks change on purpose.
_IGNORED_KEYS = frozenset({"noise", "enable_corruption", "debug_vis", "init_std"})
# Changes that keep every weight's shape but make the same weights compute something else.
_NETWORK_KEYS = re.compile(r"agent\.(actor|student)\.(class_name|activation|rnn_type)")
# A function or class as Isaac Lab writes it, "module.path:name".
_FUNCTION = re.compile(r"([A-Za-z_][\w.]*):([A-Za-z_]\w*)")


def _short_names(value):
    """Keep only the name of each ``module.path:name``, so moving or renaming a module isn't a change."""
    if isinstance(value, str):
        match = _FUNCTION.fullmatch(value)
        return match.group(2) if match else value
    if isinstance(value, dict):
        return {key: _short_names(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_short_names(item) for item in value)
    return value


def _flatten(name: str, value, out: dict) -> None:
    if isinstance(value, dict) and value:
        for key, item in value.items():
            if key not in _IGNORED_KEYS:
                _flatten(f"{name}.{key}" if re.fullmatch(r"\w+", str(key)) else f"{name}[{key}]", item, out)
    else:
        out[name] = value


def policy_sections(agent: dict) -> list[tuple[str, bool]]:
    """Where the settings that decide what a run's policy sees and does live in its configs, as Hydra override paths
    (``env.actions``), each with whether it holds terms (an observation group or the actions). ``agent`` is the
    agent config as a dict; it names the actor's observation groups.

    Critic and AMP inputs, rewards, events, terrain and command ranges only shape training, and the play tasks change
    some of them, so they are left out. ``--export`` restores these sections from a run and compares them.
    """
    model = "student" if agent.get("class_name") == "DistillationRunner" else "actor"
    groups = (agent.get("obs_groups") or {}).get(model) or ["policy"]
    sections = [(f"env.observations.{group}", True) for group in groups] + [("env.actions", True)]
    paths = ["env.scene.robot.init_state.joint_pos", "env.scene.robot.actuators", "env.sim.dt", "env.decimation"]
    paths += [f"agent.obs_groups.{model}", f"agent.{model}", "agent.clip_actions"]
    return sections + [(path, False) for path in paths]


def lookup(configs: dict, path: str, default=None):
    """The value at a dotted ``path`` such as ``env.sim.dt`` in ``{"env": ..., "agent": ...}``, or ``default``."""
    value = configs
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def policy_interface(env: dict, agent: dict) -> dict:
    """The settings that decide what a run's policy sees and does (``policy_sections``), as ``{dotted.name: value}``.

    Names are Hydra override paths (``env.actions.joint_pos.scale``). ``env.observations.<group>`` holds the group's
    term names in order. Observation noise is left out too: it only shapes training, and the play tasks turn it off.
    """
    configs = {"env": env, "agent": agent}
    out = {}
    for path, _ in policy_sections(agent):
        value = lookup(configs, path)
        _flatten(path, value, out)
        if path.startswith("env.observations."):
            out[path] = [name for name, term in (value or {}).items() if isinstance(term, dict) and "func" in term]
    return _short_names(out)


def _unset(value) -> bool:
    return value is None or value == {} or value == [] or value == ()


def _text(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_text(item) for item in value) + "]"
    return str(value)


def _describe(old, new) -> str:
    if isinstance(old, list) and isinstance(new, list):
        before, after = Counter(map(repr, old)), Counter(map(repr, new))
        if before == after:
            return "same entries in a different order"
        if len(old) + len(new) > 8:
            removed = [item for item in old if repr(item) not in after]
            added = [item for item in new if repr(item) not in before]
            parts = [f"removed {_text(removed)}"] if removed else []
            parts += [f"added {_text(added)}"] if added else []
            return ", ".join(parts) or f"{_text(old)} -> {_text(new)}"
    return f"{_text(old)} -> {_text(new)}"


def _is_term_list(key: str) -> bool:
    return key.startswith("env.observations.") and key.count(".") == 2


def interface_differences(saved: dict, current: dict) -> tuple[list[str], list[str]]:
    """Describe how the current policy settings differ from the ones a run saved, one line each.

    Returns ``(errors, warnings)``: errors are network changes that keep every weight's shape (e.g. the
    activation), so the checkpoint would load but compute something else.
    """
    errors, warnings = [], []
    # Terms that were added or removed are reported once, not setting by setting.
    skipped = []
    keys = list(dict.fromkeys([*saved, *current]))
    for key in filter(_is_term_list, keys):
        before, after = saved.get(key) or [], current.get(key) or []
        for name in before:
            if name not in after:
                warnings.append(f"{key}: term {name} removed")
                skipped.append(f"{key}.{name}.")
        for name in after:
            if name not in before:
                warnings.append(f"{key}: term {name} added")
                skipped.append(f"{key}.{name}.")
        if [name for name in before if name in after] != [name for name in after if name in before]:
            warnings.append(f"{key}: terms reordered, {', '.join(before)} -> {', '.join(after)}")
    for key in keys:
        if _is_term_list(key) or key.startswith(tuple(skipped)):
            continue
        old, new = saved.get(key), current.get(key)
        if old == new or (_unset(old) and _unset(new)):
            continue
        (errors if _NETWORK_KEYS.fullmatch(key) else warnings).append(f"{key}: {_describe(old, new)}")
    return errors, warnings


def rebuild_differences(saved: tuple[dict, dict], env_cfg: dict, agent_cfg: dict) -> tuple[list[str], list[str]]:
    """Compare the policy settings ``--export`` rebuilt from a run's ``env.yaml`` and ``agent.yaml`` (``saved``, from
    ``load_run_configs``) with the saved ones. Returns ``(mismatches, new)``, one line each: settings whose value
    differs from the run's, which means the rebuild failed, and settings the run didn't save (added to the code
    since), which keep the current code's value.

    ``env_cfg`` and ``agent_cfg`` are the current configs as plain dicts (Isaac Lab's ``class_to_dict``), taken where
    training saves them: after the environment is created.
    """
    before = policy_interface(*saved)
    after = policy_interface(normalize_config(env_cfg), normalize_config(agent_cfg))
    mismatches, new = [], []
    for key in dict.fromkeys([*before, *after]):
        old, value = before.get(key), after.get(key)
        if old == value or (_unset(old) and _unset(value)):
            continue
        (mismatches if key in before else new).append(f"{key}: {_describe(old, value)}")
    return mismatches, new


# -- Setting a run's policy settings back for --export -----------------------------------------------------------

# Marks a setting the run's yaml doesn't have, which then keeps the code's value.
_MISSING = object()

# First line of a run config that --export wrote from the code because the run didn't save it.
GENERATED_HEADER = "# Generated by --export from the current code"


def missing_run_configs(run_dir: str) -> list[str]:
    """Which of ``env.yaml`` and ``agent.yaml`` the run's ``params/`` doesn't have."""
    return [name for name in RUN_CONFIGS if not os.path.isfile(os.path.join(run_dir, "params", name))]


def generated_run_configs(run_dir: str) -> list[str]:
    """Which of the run's ``env.yaml`` and ``agent.yaml`` an earlier export generated from the code."""
    generated = []
    for name in RUN_CONFIGS:
        path = os.path.join(run_dir, "params", name)
        if os.path.isfile(path):
            with open(path) as f:
                if f.readline().startswith(GENERATED_HEADER):
                    generated.append(name)
    return generated


def mark_generated(path: str, code: str, date: str) -> None:
    """Start a run config written from the code with a line saying so."""
    with open(path) as f:
        text = f.read()
    header = f"{GENERATED_HEADER} ({code}) on {date}: the run didn't save this file, so it may not match training.\n"
    with open(path, "w") as f:
        f.write(header + text)


def _member(obj, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _assign(obj, key: str, value) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _child(name: str, key) -> str:
    return f"{name}.{key}" if re.fullmatch(r"\w+", str(key)) else f"{name}[{key}]"


def _is_config(value) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _default(cls, key: str):
    """A config class's default for ``key``. Isaac Lab's ``configclass`` stores every default as a factory."""
    field = getattr(cls, "__dataclass_fields__", {}).get(key)
    if field is None:
        return None
    if field.default is not dataclasses.MISSING:
        return field.default
    return field.default_factory() if field.default_factory is not dataclasses.MISSING else None


class _Restorer:
    """Sets config objects back to saved values, collecting the saved settings the current code can't take."""

    def __init__(self):
        self.problems = []

    def value(self, current, saved, name: str):
        """The value to store at ``name``: ``saved``, merged into ``current`` where that is a config."""
        if saved is _MISSING:
            return current
        if isinstance(saved, str) and saved.startswith("!!python/"):
            self.problems.append(f"{name}: the run saved a Python object here that can't be read back ({saved})")
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
        if saved is _MISSING:
            return
        if not isinstance(saved, dict) or self.value(container, saved, name) is not container:
            self.problems.append(f"{name}: the run saved a different kind of section here")
            return
        for key, term in list(vars(container).items()):
            if _is_config(term) and key not in saved:
                setattr(container, key, None)

    def _config_class(self, current, saved: dict, name: str):
        """A config whose ``class_name`` changed since training (e.g. the actor from ``MLPModel`` to ``RNNModel``) is
        replaced by a new instance of the config class for the saved one, found among the current class's family."""
        wanted = saved.get("class_name")
        if not isinstance(wanted, str) or getattr(current, "class_name", wanted) == wanted:
            return current
        root = next(
            (c for c in reversed(type(current).__mro__) if "class_name" in getattr(c, "__dataclass_fields__", {})),
            type(current),
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


def restore_policy_settings(env_cfg, agent_cfg, saved_env: dict, saved_agent: dict) -> list[str]:
    """Set the policy settings of ``env_cfg`` and ``agent_cfg`` (the task's configs from the code) back to the run's
    saved ones, in place. An empty saved config leaves its side as the code has it.

    Returns the saved settings the current code can't take, one line each: any of them means the current code can
    no longer build the run's policy.
    """
    restore = _Restorer()
    saved_configs, configs = {"env": saved_env, "agent": saved_agent}, {"env": env_cfg, "agent": agent_cfg}
    for path, holds_terms in policy_sections(saved_agent):
        saved = lookup(saved_configs, path, _MISSING)
        if saved is _MISSING:
            continue
        *parents, key = path.split(".")
        parent = configs
        for name in parents:
            parent = None if parent is None else _member(parent, name)
        current = None if parent is None else _member(parent, key)
        if parent is None or (holds_terms and current is None):
            restore.problems.append(f"{path}: the run's policy uses this, the current code doesn't have it")
        elif holds_terms:
            restore.terms(current, saved, path)
        else:
            _assign(parent, key, restore.value(current, saved, path))
    return restore.problems
