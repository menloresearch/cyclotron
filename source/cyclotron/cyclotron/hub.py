"""Prepare a training run for ``./cyclotron.sh --export`` and ``--share``.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import ntpath
import os
import re

# Training task for each experiment name, so a run can be exported without passing --task.
EXPERIMENT_TASKS = {
    "asimov1_velocity": "Asimov1-Velocity-v0",
    "asimov_velocity_amp": "Asimov1-Velocity-AMP-v0",
}


def read_experiment_name(agent_yaml_path: str) -> str | None:
    """Return ``experiment_name`` from a run's ``params/agent.yaml``, without loading Isaac Lab's yaml tags."""
    with open(agent_yaml_path) as f:
        match = re.search(r"^experiment_name: *(\S+)", f.read(), re.MULTILINE)
    return match.group(1).strip("'\"") if match else None


def infer_task(run_dir: str) -> str:
    """Return the training task of a run directory, from the experiment name in its ``params/agent.yaml``."""
    agent_yaml_path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent_yaml_path):
        raise ValueError(f"Cannot infer the task: {agent_yaml_path} not found. Pass it with --task.")
    experiment_name = read_experiment_name(agent_yaml_path)
    if experiment_name not in EXPERIMENT_TASKS:
        raise ValueError(f"Cannot infer the task for experiment '{experiment_name}'. Pass it with --task.")
    return EXPERIMENT_TASKS[experiment_name]


# Keys holding files on the training machine, e.g. asset_path, cache_dir, motion_files. USD prim paths (prim_path,
# filter_prim_paths_expr, ...) also start with "/" but name scene objects, not files, so they are kept.
_LOCAL_PATH_KEY = re.compile(r"\w*(_path|_dir|_file|_files)")
_ABSOLUTE_PATH = re.compile(r"(/|~|[A-Za-z]:[\\/])")
_KEY_LINE = re.compile(r"( *)(- )?([\w.-]+): ?(.*?)(\s*)$")
_ITEM_LINE = re.compile(r"( *)- (.*?)(\s*)$")


def _strip_value(value: str, removed: list[str]) -> str:
    quote = value[0] if value[:1] in ("'", '"') and value[-1:] == value[:1] else ""
    path = value[1:-1] if quote else value
    if not _ABSOLUTE_PATH.match(path):
        return value
    removed.append(path)
    return quote + ntpath.basename(path.rstrip("/\\")) + quote


def strip_local_paths(text: str) -> tuple[str, list[str]]:
    """Replace training-machine file paths in a run's yaml with their file names, so the yaml can be shared.

    Works line by line rather than loading and dumping the yaml, so every other line, including Isaac Lab's
    Python tags, stays byte for byte the same. Returns the new text and the paths that were replaced.
    """
    lines, removed = [], []
    list_indent = None  # indent of a key like motion_files whose path list is being read
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        item = _ITEM_LINE.fullmatch(body)
        if list_indent is not None and item and len(item.group(1)) >= list_indent and ": " not in item.group(2):
            lines.append(item.group(1) + "- " + _strip_value(item.group(2), removed) + item.group(3) + newline)
            continue
        list_indent = None
        key = _KEY_LINE.fullmatch(body)
        if key and _LOCAL_PATH_KEY.fullmatch(key.group(3)) and "prim" not in key.group(3):
            indent, dash, name, value, trailing = key.groups()
            if not value:
                list_indent = len(indent) + len(dash or "")
            else:
                body = f"{indent}{dash or ''}{name}: {_strip_value(value, removed)}{trailing}"
        lines.append(body + newline)
    return "".join(lines), removed
