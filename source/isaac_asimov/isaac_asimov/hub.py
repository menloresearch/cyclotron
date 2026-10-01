"""Load Asimov policies shared on the Hugging Face Hub with ``./isaac_asimov.sh --share``.

Plain Python with no Isaac Lab imports, so it can run before Isaac Sim starts and in tests.
"""

from __future__ import annotations

import dataclasses
import os
import re

import yaml

# Play task for each experiment name, as set in agents/rsl_rl_ppo_cfg.py.
PLAY_TASKS = {
    "asimov1_velocity": "Asimov1-Velocity-Play-v0",
    "asimov_velocity_amp": "Asimov1-Velocity-AMP-Play-v0",
}

REQUIRED_FILES = ("policy.onnx", "env.yaml")

# env.yaml sections that define how the policy sees and drives the robot. Everything else (num_envs, events, terrain,
# commands, ...) keeps the Play task's values.
POLICY_SECTIONS = (
    "sim.dt",
    "decimation",
    "scene.robot.init_state",
    "scene.robot.actuators",
    "actions",
    "observations.policy",
)

# Never copied from env.yaml: references to the training machine's code and files, and training-only noise.
SKIPPED_KEYS = {"func", "class_type", "prim_path", "noise", "enable_corruption"}


class HubError(Exception):
    """A shared model cannot be viewed. The message says why and what to do."""


class _Unsupported:
    """Placeholder for a YAML Python tag the loader does not construct."""

    def __init__(self, tag: str):
        self.tag = tag


class _EnvYamlLoader(yaml.SafeLoader):
    """Safe loader for Isaac Lab's dump_yaml output, which uses python/tuple and builtins.slice tags."""


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_slice(loader, node):
    return slice(*loader.construct_sequence(node, deep=True))


def _construct_unsupported(loader, tag_suffix, node):
    return _Unsupported(node.tag)


_EnvYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
_EnvYamlLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_slice)
_EnvYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_unsupported)


def load_yaml(path: str) -> dict:
    """Load a downloaded yaml file without executing any Python it names."""
    with open(path) as f:
        return yaml.load(f, Loader=_EnvYamlLoader) or {}


def download_asimov_model(repo_id: str) -> dict:
    """Check that ``repo_id`` is an Asimov model on the Hub and download its policy files.

    Returns the local paths: ``dir``, ``onnx``, ``env_yaml`` and ``agent_yaml`` (None if the repo has none).
    """
    if os.path.exists(repo_id) or repo_id.endswith((".onnx", ".pt")):
        raise HubError(
            f"'{repo_id}' is a local path. --view only loads models from the Hugging Face Hub for now;"
            " to watch your own training runs, use --play."
        )
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise HubError(f"Expected a Hugging Face model id like <org>/<model>, got '{repo_id}'.")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise HubError("huggingface_hub is not installed. Install it with: ./isaac_asimov.sh --install")

    try:
        info = HfApi().model_info(repo_id)
    except Exception as e:
        name = type(e).__name__
        if "GatedRepo" in name:
            raise HubError(f"'{repo_id}' is gated. Request access on the Hub, then run: huggingface-cli login")
        if "RepositoryNotFound" in name:
            raise HubError(
                f"Model '{repo_id}' not found on the Hugging Face Hub. If it is private, run: huggingface-cli login"
            )
        raise HubError(f"Could not read '{repo_id}' from the Hugging Face Hub: {e}")

    if info.library_name != "asimov":
        raise HubError(
            f"'{repo_id}' is not an Asimov model (library_name: {info.library_name}). Models shared with"
            " ./isaac_asimov.sh --share have library_name: asimov."
        )
    files = {sibling.rfilename for sibling in info.siblings or []}
    missing = [name for name in REQUIRED_FILES if name not in files]
    if missing:
        raise HubError(f"'{repo_id}' is missing {', '.join(missing)}, so it cannot be viewed.")

    local_dir = os.path.abspath(os.path.join("logs", "hf", repo_id.replace("/", "__")))
    print(f"[INFO] Downloading {repo_id} to: {local_dir}")
    snapshot_download(repo_id, allow_patterns=["policy.onnx", "*.yaml"], local_dir=local_dir)
    agent_yaml = os.path.join(local_dir, "agent.yaml")
    return {
        "dir": local_dir,
        "onnx": os.path.join(local_dir, "policy.onnx"),
        "env_yaml": os.path.join(local_dir, "env.yaml"),
        "agent_yaml": agent_yaml if "agent.yaml" in files else None,
    }


def infer_play_task(env_cfg_data: dict, agent_cfg_data: dict | None) -> str:
    """Pick the Play task for a shared model from its agent.yaml, or from env.yaml if there is no agent.yaml."""
    if agent_cfg_data and "experiment_name" in agent_cfg_data:
        experiment_name = agent_cfg_data["experiment_name"]
        if experiment_name not in PLAY_TASKS:
            raise HubError(
                f"Unknown experiment '{experiment_name}' in agent.yaml, so the task cannot be inferred."
                f" Known experiments: {', '.join(PLAY_TASKS)}."
            )
        return PLAY_TASKS[experiment_name]
    has_amp = "amp" in (env_cfg_data.get("observations") or {})
    return PLAY_TASKS["asimov_velocity_amp" if has_amp else "asimov1_velocity"]


def _is_config(obj) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _has_unsupported(value) -> bool:
    if isinstance(value, _Unsupported):
        return True
    if isinstance(value, dict):
        return any(_has_unsupported(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_unsupported(v) for v in value)
    return False


def _skip_key(key: str) -> bool:
    return key in SKIPPED_KEYS or key.endswith("_path")


def overlay_cfg(cfg, data: dict, path: str = "") -> tuple[list[str], list[str]]:
    """Copy the values in ``data`` onto the config object or dict ``cfg``.

    Config objects, and dicts holding config objects (e.g. actuator groups), are updated key by key. Plain dicts
    (e.g. joint_pos patterns, term params) are replaced as a whole. Keys ``cfg`` does not have are left out.

    Returns ``(changed, missing)``: dotted paths that got a new value, and paths ``cfg`` does not have.
    """
    changed, missing = [], []
    for key, value in data.items():
        key_path = f"{path}.{key}" if path else key
        if _skip_key(key) or _has_unsupported(value):
            continue
        if isinstance(cfg, dict):
            if key not in cfg:
                missing.append(key_path)
                continue
            current = cfg[key]
        else:
            if not hasattr(cfg, key):
                missing.append(key_path)
                continue
            current = getattr(cfg, key)

        if isinstance(value, dict) and _is_config(current):
            sub_changed, sub_missing = overlay_cfg(current, value, key_path)
        elif isinstance(value, dict) and isinstance(current, dict) and any(_is_config(v) for v in current.values()):
            sub_changed, sub_missing = overlay_cfg(current, value, key_path)
        elif (_is_config(current) and value is not None) or (current is None and isinstance(value, dict)):
            # A config object can only be turned off (None), and a turned-off one cannot be rebuilt from a dict.
            sub_changed, sub_missing = [], [key_path]
        else:
            if current != value:
                if isinstance(cfg, dict):
                    cfg[key] = value
                else:
                    setattr(cfg, key, value)
                sub_changed, sub_missing = [key_path], []
            else:
                sub_changed, sub_missing = [], []
        changed += sub_changed
        missing += sub_missing
    return changed, missing


def _lookup(data, dotted: str):
    for key in dotted.split("."):
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def apply_env_yaml(env_cfg, env_cfg_data: dict) -> tuple[list[str], list[str]]:
    """Overlay the policy-defining sections of a shared env.yaml, plus its seed, onto a Play task config."""
    changed, missing = [], []
    for section in POLICY_SECTIONS:
        value = _lookup(env_cfg_data, section)
        if value is None:
            continue
        parent_path, _, key = section.rpartition(".")
        parent = env_cfg
        for part in parent_path.split(".") if parent_path else []:
            parent = getattr(parent, part)
        sub_changed, sub_missing = overlay_cfg(parent, {key: value}, parent_path)
        changed += sub_changed
        missing += sub_missing
    if isinstance(env_cfg_data.get("seed"), int) and env_cfg.seed != env_cfg_data["seed"]:
        env_cfg.seed = env_cfg_data["seed"]
        changed.append("seed")
    # The task sets render_interval from decimation in __post_init__, which has already run.
    env_cfg.sim.render_interval = env_cfg.decimation
    return changed, missing
