"""Record what a run was trained with, and check it before ``--export`` and ``--play`` use the run.

Training writes ``params/code_state.yaml`` with only what those checks use, plus where to find the code again:

- ``cyclotron``: the git commit, branch, remote and dirty flag of the package, to name the code to check out;
- ``isaaclab_commit`` and ``packages`` (installed versions), for the record;
- ``robot_model``: the robot model's name, repository, urdf path, sha256, commit and dirty flag;
- ``actor_code``: the sha256 of the modules behind the network the run deploys;
- ``policy_io``: the joints, gains and offsets the policy's inputs and outputs resolve to (``policy_io``);
- ``run_configs``: the sha256 of the ``env.yaml`` and ``agent.yaml`` written next to it, so editing them shows.

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
from xml.etree import ElementTree

import yaml

from cyclotron.policy_io import policy_io_differences
from cyclotron.run_config import (
    RUN_CONFIGS,
    interface_differences,
    load_run_configs,
    normalize_config,
    policy_interface,
)

CODE_STATE_FILE = "code_state.yaml"
PACKAGES = ("isaacsim", "isaaclab", "rsl-rl-lib", "torch")
# The rsl_rl packages behind a deployed network: the models, and the layers, normalization and distributions they are
# built from. Exporting a policy runs nothing else of the library.
ACTOR_CODE_PACKAGES = ("rsl_rl.models", "rsl_rl.modules")

# -- Recording at training time -----------------------------------------------------------------------------------


def package_dir() -> str:
    """Folder of the installed cyclotron package: the code the tasks are actually imported from."""
    import cyclotron

    return os.path.dirname(os.path.abspath(cyclotron.__file__))


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(cwd: str, *args: str) -> str | None:
    """Run git in ``cwd``; return its output, or None if git is missing or fails (e.g. not a repository)."""
    try:
        result = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _without_credentials(url: str | None) -> str | None:
    # https://user:token@github.com/org/repo.git -> https://github.com/org/repo.git. Everything up to the last @ before
    # the host's slash goes, since git takes a password with a raw @ in it (https://user:p@ss@host).
    return re.sub(r"^([a-z+]+://)[^/]*@", r"\1", url) if url else url


def _git_state(path: str) -> dict:
    """Commit, branch, remote and dirty flag of the repository tracking ``path``; all None if git doesn't track it."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    # A package installed into site-packages can sit inside a repository without being tracked by it.
    if _git(folder, "ls-files", "--error-unmatch", os.path.basename(path) if folder != path else ".") is None:
        return {"commit": None, "branch": None, "remote": None, "dirty": None}
    branch = _git(folder, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(folder, "status", "--porcelain", "--", ".")
    return {
        "commit": _git(folder, "rev-parse", "HEAD"),
        "branch": None if branch == "HEAD" else branch,
        "remote": _without_credentials(_git(folder, "remote", "get-url", "origin")),
        # Whether the commit alone describes the code in ``folder``: changed files and new ones never added to git
        # count, ignored ones and the rest of the repository don't. None when git status failed: unknown, not clean.
        "dirty": None if status is None else bool(status),
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


def _repository_name(remote: str | None) -> str | None:
    # https://github.com/menloresearch/asimov-1.git or git@github.com:menloresearch/asimov-1 -> menloresearch/asimov-1
    match = re.search(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?/?$", remote) if remote else None
    return match.group(1) if match else None


def _urdf_robot_name(urdf: str) -> str | None:
    """The ``name`` of the urdf's ``<robot>`` element, or None when the file isn't a urdf with one."""
    try:
        return ElementTree.parse(urdf).getroot().get("name")
    except (OSError, ElementTree.ParseError):
        return None


def _robot_model_record(robot: str) -> dict:
    """The robot model, and where to find it again: the robot's name from the urdf, the model repository
    (``owner/name`` of its origin remote), the urdf's path from that repository's root, its sha256, the repository's
    commit, and whether the repository had uncommitted changes (the commit alone isn't the model then). When git
    doesn't track the urdf, the path is just its file name and the repository fields are None."""
    folder, name = os.path.split(robot)
    git = _git_state(robot)
    tracked = git["commit"] is not None
    # The whole repository, not just the urdf folder: the meshes the urdf loads sit next to it. None when git
    # status failed: unknown, not clean.
    status = _git(folder, "status", "--porcelain", "--", ":/") if tracked else None
    return {
        "name": _urdf_robot_name(robot),
        "repo": _repository_name(git["remote"]),
        "urdf_filepath": _git(folder, "ls-files", "--full-name", "--", name) if tracked else name,
        "sha256": _sha256(robot),
        "commit": git["commit"],
        "dirty": None if status is None else bool(status),
    }


def actor_class_name(agent: dict) -> str | None:
    """The ``class_name`` of the network a run deploys: the actor, or the distilled student."""
    model = "student" if agent.get("class_name") == "DistillationRunner" else "actor"
    name = (agent.get(model) or {}).get("class_name")
    return name if isinstance(name, str) else None


def _module_file(module: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec and spec.origin and spec.origin.endswith(".py") else None


def actor_code(agent: dict) -> dict[str, str] | None:
    """The sha256 of each module behind the network a run deploys, keyed by module name (``rsl_rl.models.mlp_model``):
    every module of ``ACTOR_CODE_PACKAGES``, and the module of each class the network's class is or inherits from
    outside rsl_rl and PyTorch (a custom network). The rest of the library (the training algorithm, runners, storage,
    utilities) isn't included, so changing it isn't a change to the exported policy.

    None when the agent config names no network class, or the class can't be imported.
    """
    name = actor_class_name(agent)
    if name is None:
        return None
    try:
        from rsl_rl.utils import resolve_callable

        cls = resolve_callable(name)
    except (ImportError, AttributeError, ValueError, TypeError):
        return None
    hashes = {}
    for package in ACTOR_CODE_PACKAGES:
        init = _module_file(package)
        for path in sorted(glob.glob(os.path.join(os.path.dirname(init), "*.py"))) if init else []:
            stem = os.path.basename(path)[: -len(".py")]
            hashes[package if stem == "__init__" else f"{package}.{stem}"] = _sha256(path)
    for module in {c.__module__ for c in cls.__mro__}:
        path = _module_file(module) if module.split(".")[0] not in ("builtins", "torch", "rsl_rl") else None
        if path:
            hashes[module] = _sha256(path)
    return dict(sorted(hashes.items()))


def record_code_state(
    env: dict | None = None,
    loaded_checkpoint: str | None = None,
    package: str | None = None,
    agent: dict | None = None,
    policy_io: dict | None = None,
):
    """Describe what a run is trained with. It holds hashes and names, not code or local paths, so it can be shared.

    ``env`` is the task's config as a dict, used to find the robot model the task loads. ``agent`` is the agent
    config as a dict, used to find the code of the network the run deploys (``actor_code``). ``policy_io`` is what the
    live environment resolves the policy's inputs and outputs to (``policy_io.resolve_policy_io``).
    """
    state = {"cyclotron": _git_state(package or package_dir())}
    isaaclab = importlib.util.find_spec("isaaclab")
    state["isaaclab_commit"] = _git_state(isaaclab.origin)["commit"] if isaaclab and isaaclab.origin else None
    state["packages"] = {name: _version(name) for name in PACKAGES}
    robot = robot_model_path(env)
    state["robot_model"] = _robot_model_record(robot) if robot else None
    actor = actor_code(agent) if agent else None
    if actor:
        state["actor_code"] = actor
    if policy_io:
        state["policy_io"] = policy_io
    if loaded_checkpoint:
        state["loaded_checkpoint"] = loaded_checkpoint
    return state


def write_code_state(params_dir: str, state: dict) -> None:
    """Write ``code_state.yaml`` into a run's ``params/`` folder."""
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, CODE_STATE_FILE), "w") as f:
        yaml.safe_dump(state, f, sort_keys=False)


def hash_run_configs(params_dir: str) -> dict[str, str]:
    """The sha256 of the ``env.yaml`` and ``agent.yaml`` in ``params_dir``, which training records in
    ``code_state.yaml`` so that editing them afterwards shows."""
    paths = {name: os.path.join(params_dir, name) for name in RUN_CONFIGS}
    return {name: _sha256(path) for name, path in paths.items() if os.path.isfile(path)}


# -- Reading the record -------------------------------------------------------------------------------------------


def read_code_state(run_dir: str) -> dict | None:
    """The run's ``params/code_state.yaml``, or None when it has none."""
    path = os.path.join(run_dir, "params", CODE_STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def code_state_missing(run_dir: str) -> bool:
    """Whether the run has no ``code_state.yaml``, which cyclotron's training always writes into ``params/``.

    Isaac Lab's own ``train.py`` doesn't, so this is true for a run from another codebase. It is also true for a
    cyclotron run trained before training recorded one, which can't be told apart.
    """
    return read_code_state(run_dir) is None


def edited_run_configs(run_dir: str) -> list[str] | None:
    """Which of the run's ``env.yaml`` and ``agent.yaml`` changed or went missing since training, by the sha256 its
    ``code_state.yaml`` recorded; None when the run didn't record them (trained before training did)."""
    recorded = (read_code_state(run_dir) or {}).get("run_configs")
    if not recorded:
        return None
    current = hash_run_configs(os.path.join(run_dir, "params"))
    return [name for name, digest in recorded.items() if current.get(name) != digest]


def _short(commit: str | None) -> str:
    return commit[:7] if commit else "unknown"


def _where(code: dict) -> str:
    if not code.get("commit"):
        # Not in git, or git failed: there is no branch to name, detached or not.
        return "code git doesn't track"
    place = f"{code.get('branch') or 'a detached HEAD'} @ {_short(code.get('commit'))}"
    if code.get("dirty") is None:
        return place + " (uncommitted changes unknown)"
    return place + (" with uncommitted changes" if code["dirty"] else "")


def _rsl_rl_commits(run_dir: str) -> list[dict]:
    """For runs trained before ``code_state.yaml`` existed: the commits rsl_rl logged in the run's ``git/*.diff``,
    one per repository, with its branch and whether it had uncommitted changes."""
    records = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "git", "*.diff"))):
        with open(path) as f:
            text = f.read()
        commit = re.search(r"--- git commit ---\s+([0-9a-f]{7,40})", text)
        branch = re.search(r"^On branch (\S+)", text, re.MULTILINE)
        diff = text.split("--- git diff ---", 1)[1] if "--- git diff ---" in text else ""
        if commit:
            record = {"commit": commit.group(1), "branch": branch and branch.group(1), "dirty": bool(diff.strip())}
            records.setdefault(commit.group(1), {**record, "file": os.path.basename(path)})
    return list(records.values())


def training_code(run_dir: str) -> str | None:
    """Where the code a run was trained with is, e.g. ``exp/drift @ 32aef5c``, or None if the run doesn't say.

    Runs trained before ``code_state.yaml`` existed can name several commits: rsl_rl logged one per repository.
    """
    state = read_code_state(run_dir)
    if state is not None:
        code = state.get("cyclotron") or {}
        return _where(code) if code.get("commit") else None
    return " or ".join(f"{_where(record)} (git/{record['file']})" for record in _rsl_rl_commits(run_dir)) or None


def training_commit(run_dir: str) -> str | None:
    """The commit a run was trained with, or None when the run doesn't record one (or names several). A run trained
    with uncommitted changes, or without knowing whether it had any, gets ``-dirty`` appended, as ``git describe
    --dirty`` does: the commit alone isn't known to be its code."""
    state = read_code_state(run_dir)
    if state is not None:
        code = state.get("cyclotron") or {}
        records = [code] if code.get("commit") else []
    else:
        records = _rsl_rl_commits(run_dir)
    if len(records) != 1:
        return None
    return records[0]["commit"] + ("" if records[0].get("dirty") is False else "-dirty")


def training_robot_model(run_dir: str) -> dict | None:
    """The robot model a run was trained with, as ``code_state.yaml`` recorded it (name, repo, urdf_filepath, sha256,
    commit, dirty); None when the run doesn't record one."""
    return (read_code_state(run_dir) or {}).get("robot_model") or None


def check_out_hint(run_dir: str) -> str:
    trained = training_code(run_dir)
    if not trained:
        return "Train a new run, or check out the code the run was trained with (the run doesn't record its commit)."
    return f"Check out the code the run was trained with ({trained}), or train a new run."


def current_code() -> str:
    """Where the cyclotron package is imported from, e.g. ``exp/drift @ 32aef5c with uncommitted changes``."""
    return _where(_git_state(package_dir()))


# -- Checking the current code against the record -----------------------------------------------------------------

ACTOR_CODE_UNCHANGED, ACTOR_CODE_CHANGED, ACTOR_CODE_UNRECORDED = "unchanged", "changed", "unrecorded"
ROBOT_MODEL_MATCHED, ROBOT_MODEL_CHANGED, ROBOT_MODEL_UNCHECKED = "matched", "changed", "unchecked"


def actor_code_differences(run_dir: str, agent: dict) -> tuple[str, list[str]]:
    """Compare the code of the network a run deploys with what its ``code_state.yaml`` recorded (``actor_code``).

    ``agent`` is the run's saved agent config. Returns ``(status, lines)``: ``"unchanged"``, ``"changed"`` with one
    line per module that differs, or ``"unrecorded"`` when the run didn't record it (trained before training did, or
    outside cyclotron), which can't be checked.
    """
    recorded = (read_code_state(run_dir) or {}).get("actor_code")
    if not recorded:
        return ACTOR_CODE_UNRECORDED, []
    current = actor_code(agent)
    if current is None:
        return ACTOR_CODE_CHANGED, [
            f"the network class {actor_class_name(agent)} can't be imported by the current code"
        ]
    lines = [f"{name}: changed" for name in recorded if name in current and recorded[name] != current[name]]
    lines += [f"{name}: added" for name in current if name not in recorded]
    lines += [f"{name}: removed" for name in recorded if name not in current]
    return (ACTOR_CODE_CHANGED, lines) if lines else (ACTOR_CODE_UNCHANGED, [])


def robot_model_check(run_dir: str, env: dict) -> tuple[str, str | None]:
    """Compare the robot model the current config loads with the one the run was trained with, by sha256.

    ``env`` is the current env config as a dict. Returns ``(status, line)``: ``"matched"``, ``"changed"`` with a line
    naming both, or ``"unchecked"`` when the run recorded no robot model or the current one isn't a local file.
    """
    trained = training_robot_model(run_dir) or {}
    path = robot_model_path(env)
    if not trained.get("sha256") or path is None:
        return ROBOT_MODEL_UNCHECKED, None
    if _sha256(path) == trained["sha256"]:
        return ROBOT_MODEL_MATCHED, None
    now = _robot_model_record(path)
    return ROBOT_MODEL_CHANGED, f"{_model_where(trained)} -> {_model_where(now)}"


def _model_where(record: dict) -> str:
    # The sha256 tells the two apart when the same commit has uncommitted edits to the URDF.
    place = f"{record.get('urdf_filepath') or 'unknown'} @ {_short(record.get('commit'))}"
    if record.get("commit") and record.get("dirty") is not False:
        place += " with uncommitted changes" if record.get("dirty") else " (uncommitted changes unknown)"
    return f"{place} (sha256 {record['sha256'][:12]})"


def describe_changes(
    run_dir: str, env_cfg: dict, agent_cfg: dict, consequence: str, policy_io: dict | None = None
) -> tuple[str, str]:
    """Compare a run with the current code. Returns the message to print and its level: ``"ok"``, ``"warning"``,
    or ``"error"`` when the policy's network changed in a way its weights can't show.

    ``env_cfg`` and ``agent_cfg`` are the current configs as plain dicts (Isaac Lab's ``class_to_dict``), taken where
    training saves them: after the environment is created. ``policy_io`` is what the live environment resolves the
    policy's inputs and outputs to (``policy_io.resolve_policy_io``), compared with what training recorded.
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
    recorded_io = (read_code_state(run_dir) or {}).get("policy_io")
    resolved = policy_io_differences(recorded_io, policy_io) if recorded_io and policy_io else []
    _, robot = robot_model_check(run_dir, current[0])
    _, network = actor_code_differences(run_dir, saved[1])
    if not (errors or settings or resolved or robot or network):
        return "[INFO] Checked the run against the current code: nothing changed since training.", "ok"
    if errors:
        lines = ["[ERROR] The policy's network changed since this run was trained; its weights would compute something"]
        lines += ["  else in the network the current code builds."] + [f"    {line}" for line in errors]
    else:
        lines = ["[WARNING] The code has changed since this run was trained."]
    sections = [
        ("Policy settings (what the policy sees and does)", settings),
        ("Joints and gains the policy's inputs and outputs resolve to", resolved),
        ("Robot model", [robot] if robot else []),
        ("Code of the policy's network", network),
    ]
    for title, changes in sections:
        if changes:
            lines += [f"  {title}:"] + [f"    {line}" for line in changes]
    lines.append(f"  {check_out_hint(run_dir) if errors else consequence}")
    return "\n".join(lines), "error" if errors else "warning"
