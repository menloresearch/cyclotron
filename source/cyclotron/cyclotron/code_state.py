"""Record the code a run was trained with, and report what changed before ``--export`` and ``--play`` use the run.

Training writes ``params/code_state.yaml``: the git commit, a hash of every training-code file in the cyclotron
package, the Isaac Lab commit, a hash of the robot model, the installed package versions, and the sha256 of the
``env.yaml`` and ``agent.yaml`` it wrote next to it, so editing those by hand later shows. Export and play compare the
run's saved ``env.yaml``, ``agent.yaml`` and ``code_state.yaml`` with the current code, so a changed observation,
action or network is reported instead of crashing or silently changing what the policy sees.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata as metadata
import importlib.util
import os
import re
import subprocess
from collections import Counter

import yaml

CODE_STATE_FILE = "code_state.yaml"
PACKAGES = ("isaacsim", "isaaclab", "rsl-rl-lib", "torch")
# Package files that export, share and these checks use, but training doesn't; editing them changes no policy.
NOT_TRAINING_CODE = frozenset({"code_state.py", "hub.py", "onnx_export.py", "run_config.py"})

# -- Recording the code at training time ------------------------------------------------------------------------


def package_dir() -> str:
    """Folder of the installed cyclotron package: the code the tasks are actually imported from."""
    import cyclotron

    return os.path.dirname(os.path.abspath(cyclotron.__file__))


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def hash_files(root: str) -> dict[str, str]:
    """Return the sha256 of every training-code file under ``root``, keyed by its ``/``-separated relative path."""
    hashes = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name != "__pycache__")
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if not name.endswith(".pyc") and relative not in NOT_TRAINING_CODE:
                hashes[relative] = _sha256(path)
    return hashes


def _git(cwd: str, *args: str) -> str | None:
    """Run git in ``cwd``; return its output, or None if git is missing or fails (e.g. not a repository)."""
    try:
        result = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _without_credentials(url: str | None) -> str | None:
    # https://user:token@github.com/org/repo.git -> https://github.com/org/repo.git
    return re.sub(r"^([a-z+]+://)[^/@]+@", r"\1", url) if url else url


def _git_state(path: str) -> dict:
    """Commit, branch, remote and dirty flag of the repository tracking ``path``; all None if git doesn't track it."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    # A package installed into site-packages can sit inside a repository without being tracked by it.
    if _git(folder, "ls-files", "--error-unmatch", os.path.basename(path) if folder != path else ".") is None:
        return {"commit": None, "branch": None, "remote": None, "dirty": None}
    branch = _git(folder, "rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": _git(folder, "rev-parse", "HEAD"),
        "branch": None if branch == "HEAD" else branch,
        "remote": _without_credentials(_git(folder, "remote", "get-url", "origin")),
        # File hashes are exact; this only says whether the commit alone describes the code in ``folder``: changed
        # files and new ones never added to git count, ignored ones and the rest of the repository don't.
        "dirty": bool(_git(folder, "status", "--porcelain", "--", ".")),
    }


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def robot_model_path(env: dict | None) -> str | None:
    """The robot model file a task loads (``scene.robot.spawn.asset_path``), if it is a local file."""
    robot = ((env or {}).get("scene") or {}).get("robot") or {}
    path = (robot.get("spawn") or {}).get("asset_path")
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _robot_model_record(robot: str) -> dict:
    """The robot model file, and where to find it again: its path in the model repository, that repository's commit,
    and whether the repository had uncommitted changes (the commit alone isn't the model then). The commit, path and
    dirty flag are None when the file isn't tracked by git."""
    folder = os.path.dirname(robot)
    commit = _git_state(robot)["commit"]
    tracked = commit is not None
    return {
        "file": os.path.basename(robot),
        "path": _git(folder, "ls-files", "--full-name", "--", os.path.basename(robot)) if tracked else None,
        "sha256": _sha256(robot),
        "commit": commit,
        # The whole repository, not just the urdf folder: the meshes the urdf loads sit next to it.
        "dirty": bool(_git(folder, "status", "--porcelain", "--", ":/")) if tracked else None,
    }


def record_code_state(env: dict | None = None, loaded_checkpoint: str | None = None, package: str | None = None):
    """Describe the code a run is trained with. It holds hashes, not code or local paths, so it can be shared.

    ``env`` is the task's config as a dict, used to find the robot model the task loads.
    """
    package = package or package_dir()
    state = {"cyclotron": {**_git_state(package), "files": hash_files(package)}}
    isaaclab = importlib.util.find_spec("isaaclab")
    state["isaaclab_commit"] = _git_state(isaaclab.origin)["commit"] if isaaclab and isaaclab.origin else None
    robot = robot_model_path(env)
    state["robot_model"] = _robot_model_record(robot) if robot else None
    state["packages"] = {name: _version(name) for name in PACKAGES}
    if loaded_checkpoint:
        state["loaded_checkpoint"] = loaded_checkpoint
    return state


def write_code_state(params_dir: str, state: dict) -> None:
    """Write ``code_state.yaml`` into a run's ``params/`` folder."""
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, CODE_STATE_FILE), "w") as f:
        yaml.safe_dump(state, f, sort_keys=False)


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


def hash_run_configs(params_dir: str) -> dict[str, str]:
    """The sha256 of the ``env.yaml`` and ``agent.yaml`` in ``params_dir``, which training records in
    ``code_state.yaml`` so that editing them afterwards shows."""
    paths = {name: os.path.join(params_dir, name) for name in RUN_CONFIGS}
    return {name: _sha256(path) for name, path in paths.items() if os.path.isfile(path)}


def edited_run_configs(run_dir: str) -> list[str] | None:
    """Which of the run's ``env.yaml`` and ``agent.yaml`` changed or went missing since training, by the sha256 its
    ``code_state.yaml`` recorded; None when the run didn't record them (trained before training did)."""
    path = os.path.join(run_dir, "params", CODE_STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        recorded = (yaml.safe_load(f) or {}).get("run_configs")
    if not recorded:
        return None
    current = hash_run_configs(os.path.dirname(path))
    return [name for name, digest in recorded.items() if current.get(name) != digest]


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


# -- What changed in the code -------------------------------------------------------------------------------------


def _short(commit: str | None) -> str:
    return commit[:7] if commit else "unknown"


def _where(code: dict) -> str:
    place = f"{code.get('branch') or 'a detached HEAD'} @ {_short(code.get('commit'))}"
    return place + (" with uncommitted changes" if code.get("dirty") else "")


def _listing(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def policy_code_files(env: dict, agent: dict) -> set[str]:
    """The package files, named as ``code_state.yaml`` names them (``tasks/locomotion/mdp/observations.py``), that
    define the functions and classes a run's policy settings (``policy_sections``) name: the package's code behind
    what the policy sees and does. Functions from other packages, such as Isaac Lab's, come with that package's
    version; helpers these files import aren't followed.

    A module of a package that is no longer installed, such as this package's old name ``isaac_asimov``, counts as
    this package's, by its path inside it.
    """
    package = os.path.basename(package_dir())
    files = set()

    def visit(value) -> None:
        if isinstance(value, str) and _FUNCTION.fullmatch(value):
            top, *path = value.split(":")[0].split(".")
            if top == package or importlib.util.find_spec(top) is None:
                files.add("/".join([*path, "__init__.py"]))
                if path:
                    files.add("/".join(path) + ".py")
        elif isinstance(value, dict):
            for key, item in value.items():
                if key not in _IGNORED_KEYS:
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for path, _ in policy_sections(agent):
        visit(lookup({"env": env, "agent": agent}, path))
    return files


def _among(path: str, files: set[str] | None) -> bool:
    """Whether a package file (``tasks/x.py``, or a repo path ending in it) is one of ``files``; None is all."""
    return files is None or path in files or any(path.endswith(f"/{name}") for name in files)


def compare_code_state(saved: dict, current: dict, files: set[str] | None = None) -> list[str]:
    """Describe, one line each, how the current code differs from a run's ``code_state.yaml``. ``files`` limits the
    package files compared to those (``policy_code_files``); the rest is always compared."""
    lines = []
    trained, now = saved.get("cyclotron") or {}, current["cyclotron"]
    before, after = trained.get("files") or {}, now["files"]
    changes = {
        "changed": [name for name in after if name in before and after[name] != before[name]],
        "added": [name for name in after if name not in before],
        "removed": [name for name in before if name not in after],
    }
    changes = {label: [name for name in names if _among(name, files)] for label, names in changes.items()}
    if any(changes.values()):
        lines.append(f"trained on {_where(trained)}, now {_where(now)}")
        lines += [f"cyclotron files {label}: {_listing(names)}" for label, names in changes.items() if names]
    if saved.get("isaaclab_commit") != current["isaaclab_commit"]:
        lines.append(f"Isaac Lab: {_short(saved.get('isaaclab_commit'))} -> {_short(current['isaaclab_commit'])}")
    robot_before, robot_now = saved.get("robot_model") or {}, current["robot_model"] or {}
    if robot_before.get("sha256") != robot_now.get("sha256"):
        lines.append(
            f"robot model changed: {robot_before.get('file') or 'unknown'} @ {_short(robot_before.get('commit'))}"
            f" -> {robot_now.get('file') or 'unknown'} @ {_short(robot_now.get('commit'))}"
        )
    for name, version in (saved.get("packages") or {}).items():
        if version != current["packages"].get(name):
            lines.append(f"{name}: {version} -> {current['packages'].get(name)}")
    return lines


def read_git_records(run_dir: str) -> list[dict]:
    """The commits rsl_rl logged in a run's ``git/*.diff`` files: one per repository, with branch and dirty flag."""
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "git", "*.diff"))):
        with open(path) as f:
            text = f.read()
        commit = re.search(r"--- git commit ---\s+([0-9a-f]{7,40})", text)
        if not commit:
            continue
        branch = re.search(r"^On branch (\S+)", text, re.MULTILINE)
        diff = text.split("--- git diff ---", 1)[1] if "--- git diff ---" in text else ""
        records.append(
            {
                "file": os.path.basename(path),
                "commit": commit.group(1),
                "branch": branch.group(1) if branch else None,
                "dirty": bool(diff.strip()),
                "diff_files": re.findall(r"^diff --git a/(\S+) ", diff, re.MULTILINE),
            }
        )
    return records


def files_changed_since(repo: str, commit: str) -> list[str] | None:
    """Training-code files under ``source/`` that differ between ``commit`` and the working tree, ignoring pure
    renames and the files in ``NOT_TRAINING_CODE``.

    Returns None when ``commit`` isn't in the repository.
    """
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        return None
    output = _git(repo, "diff", "--name-status", "-M", commit, "--", "source") or ""
    paths = [line.split("\t")[-1] for line in output.splitlines() if not line.startswith("R100")]
    return [path for path in paths if os.path.basename(path) not in NOT_TRAINING_CODE]


def _git_record_differences(run_dir: str, files: set[str] | None = None) -> list[str]:
    """For runs trained before ``code_state.yaml`` existed: compare with the commits rsl_rl logged in ``git/``.
    ``files`` limits the package files compared, as in ``compare_code_state``."""
    records = read_git_records(run_dir)
    if not records:
        return [
            "the run has no record of its code (no params/code_state.yaml or git/), so code changes can't be checked"
        ]
    repo = _git(package_dir(), "rev-parse", "--show-toplevel")
    lines = []
    for record in {record["commit"]: record for record in records}.values():
        trained = f"trained on {_where(record)} (git/{record['file']})"
        changed = files_changed_since(repo, record["commit"]) if repo else None
        if changed:
            changed = [path for path in changed if _among(path, files)]
        if not repo:
            lines.append(f"{trained}; the current code isn't in a git repository, so it can't be compared")
        elif changed is None:
            lines.append(f"{trained}; that commit isn't in this clone, so changes can't be listed")
        elif changed:
            lines.append(f"{trained}; changed under source/ since then: {_listing(changed)}")
        elif record["dirty"] and any(_among(path, files) for path in record["diff_files"]):
            lines.append(f"{trained}; it had uncommitted changes, so it can't be compared exactly")
    return lines


def code_differences(run_dir: str, current: dict, files: set[str] | None = None) -> list[str]:
    """Describe how the current code (``record_code_state()``) differs from the code a run was trained with.
    ``files`` limits the package files compared (``policy_code_files``); None compares them all."""
    path = os.path.join(run_dir, "params", CODE_STATE_FILE)
    if not os.path.isfile(path):
        return _git_record_differences(run_dir, files)
    with open(path) as f:
        return compare_code_state(yaml.safe_load(f) or {}, current, files)


def training_code(run_dir: str) -> str | None:
    """Where the code a run was trained with is, e.g. ``exp/drift @ 32aef5c``, or None if the run doesn't say.

    Runs trained before ``code_state.yaml`` existed can name several commits: rsl_rl logged one per repository.
    """
    path = os.path.join(run_dir, "params", CODE_STATE_FILE)
    if os.path.isfile(path):
        with open(path) as f:
            code = (yaml.safe_load(f) or {}).get("cyclotron") or {}
        return _where(code) if code.get("commit") else None
    records = {record["commit"]: record for record in read_git_records(run_dir)}.values()
    return " or ".join(f"{_where(record)} (git/{record['file']})" for record in records) or None


def training_commit(run_dir: str) -> str | None:
    """The commit a run was trained with, or None when the run doesn't record one (or names several). A run trained
    with uncommitted changes gets ``-dirty`` appended, as ``git describe --dirty`` does: the commit alone isn't its
    code."""
    path = os.path.join(run_dir, "params", CODE_STATE_FILE)
    if os.path.isfile(path):
        with open(path) as f:
            code = (yaml.safe_load(f) or {}).get("cyclotron") or {}
        records = [code] if code.get("commit") else []
    else:
        records = list({record["commit"]: record for record in read_git_records(run_dir)}.values())
    if len(records) != 1:
        return None
    return records[0]["commit"] + ("-dirty" if records[0].get("dirty") else "")


def training_robot_model(run_dir: str) -> dict | None:
    """The robot model a run was trained with, as ``code_state.yaml`` recorded it (file, path in the model repository,
    sha256, commit, dirty); None when the run doesn't record one."""
    path = os.path.join(run_dir, "params", CODE_STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("robot_model") or None


def check_out_hint(run_dir: str) -> str:
    trained = training_code(run_dir)
    if not trained:
        return "Train a new run, or check out the code the run was trained with (the run doesn't record its commit)."
    return f"Check out the code the run was trained with ({trained}), or train a new run."


# -- Putting it together for export and play ---------------------------------------------------------------------


def describe_changes(run_dir: str, env_cfg: dict, agent_cfg: dict, consequence: str) -> tuple[str, str]:
    """Compare a run with the current code. Returns the message to print and its level: ``"ok"``, ``"warning"``,
    or ``"error"`` when the policy's network changed in a way its weights can't show.

    ``env_cfg`` and ``agent_cfg`` are the current configs as plain dicts (Isaac Lab's ``class_to_dict``), taken where
    training saves them: after the environment is created.
    """
    params = os.path.join(run_dir, "params")
    if not all(os.path.isfile(os.path.join(params, name)) for name in RUN_CONFIGS):
        return (
            f"[WARNING] {params} has no env.yaml and agent.yaml, so changes since training can't be checked.",
            "warning",
        )
    saved = load_run_configs(run_dir)
    current = normalize_config(env_cfg), normalize_config(agent_cfg)
    errors, settings = interface_differences(policy_interface(*saved), policy_interface(*current))
    code = code_differences(run_dir, record_code_state(current[0]))
    if not (errors or settings or code):
        return "[INFO] Checked the run against the current code: nothing changed since training.", "ok"
    if errors:
        lines = ["[ERROR] The policy's network changed since this run was trained; its weights would compute something"]
        lines += ["  else in the network the current code builds."] + [f"    {line}" for line in errors]
    else:
        lines = ["[WARNING] The code has changed since this run was trained."]
    if settings:
        lines += ["  Policy settings (what the policy sees and does):"] + [f"    {line}" for line in settings]
    if code:
        lines += ["  Code:"] + [f"    {line}" for line in code]
    if errors:
        lines.append(f"  {check_out_hint(run_dir)}")
    else:
        lines.append(f"  {consequence}")
    return "\n".join(lines), "error" if errors else "warning"


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


def current_code() -> str:
    """Where the cyclotron package is imported from, e.g. ``exp/drift @ 32aef5c with uncommitted changes``."""
    return _where(_git_state(package_dir()))


# -- Loading only the policy -----------------------------------------------------------------------------------


def inference_load_cfg(runner_class: str) -> dict:
    """What ``runner.load`` must restore to run the policy: the actor, or the distilled student. The critic,
    optimizer and AMP discriminator are training-only, and may no longer fit after the code changed."""
    return {"student": True} if runner_class == "DistillationRunner" else {"actor": True}


def _shape(tensor) -> list[int]:
    return list(getattr(tensor, "shape", ()))


def policy_shape_errors(saved: dict, current: dict) -> list[str]:
    """Compare a checkpoint's policy weights with the network the current code builds; describe each mismatch."""
    lines = []
    missing = [name for name in current if name not in saved]
    unexpected = [name for name in saved if name not in current]
    if missing or unexpected:
        lines.append("the network's layers changed:")
        if unexpected:
            lines.append(f"  only in the checkpoint: {_listing(unexpected)}")
        if missing:
            lines.append(f"  only in the current code: {_listing(missing)}")
    # With the same layers, the first and last weight matrices hold the policy's input and output sizes.
    matrices = [] if lines else [name for name in saved if len(_shape(saved[name])) == 2]
    for name in saved:
        if name not in current or _shape(saved[name]) == _shape(current[name]):
            continue
        old, new = _shape(saved[name]), _shape(current[name])
        line = f"{name}: checkpoint {old}, current code {new}"
        if matrices and name == matrices[0] and old[1:] != new[1:]:
            line += f" (the policy takes {old[1]} inputs; the current observations give {new[1]})"
        elif matrices and name == matrices[-1] and old[:1] != new[:1]:
            line += f" (the policy gives {old[0]} actions; the current code expects {new[0]})"
        lines.append(line)
    return lines


def load_policy(runner, checkpoint: str, runner_class: str, run_dir: str) -> None:
    """Restore only the policy from ``checkpoint`` into ``runner``, after checking it fits the current network.

    Raises ``ValueError`` with a plain description when it doesn't, instead of PyTorch's size-mismatch error,
    naming the code the run in ``run_dir`` was trained with.
    """
    import torch

    key = "student_state_dict" if runner_class == "DistillationRunner" else "actor_state_dict"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False).get(key)
    if saved is None:
        # Not the layout this runner saves (e.g. distilling from an RL checkpoint): let rsl_rl decide what to load.
        runner.load(checkpoint)
        return
    errors = policy_shape_errors(saved, runner.alg.get_policy().state_dict())
    if errors:
        raise ValueError(
            "The checkpoint's policy doesn't fit the network the current code builds:\n"
            + "\n".join(f"  {line}" for line in errors)
            + "\nThe policy's inputs or network changed since training; see the warning above.\n"
            + check_out_hint(run_dir)
        )
    runner.load(checkpoint, load_cfg=inference_load_cfg(runner_class))
