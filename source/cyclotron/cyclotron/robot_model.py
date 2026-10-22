"""Load the robot model a run was trained with, for ``--export`` and ``--play``.

Training records the robot model in ``code_state.yaml`` (``robot_model``: its repository, the urdf's path in it, the
urdf's sha256 and the repository's commit). When the urdf the current config loads has another sha256, as after the
robot model was updated, export and play load the run's own instead: the model folder the urdf sits in (``sim-model/``,
the folder ``ASIMOV_1_MODEL_DIR`` names, with its meshes) is fetched at the recorded commit into cyclotron's cache, and
the environment's robot is pointed at it. Check 10d then compares the run with the model it was trained on. When that
can't be done (no clean commit recorded, no network), the current model is loaded as before and 10d says why.

The cache holds one folder per repository, commit and model folder,
``robot-models/<owner>__<name>/<commit>__<model folder>/``, with that commit's model folder and the files at the
repository's root, and no git files. It is fetched into a staging folder next to it and renamed into place once its urdf
has the recorded sha256, under a lock, so concurrent exports fetch it once; a folder in place is never changed, and is
read without the lock. Its urdf is checked against the recorded sha256 each time it is used.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import fcntl
import os
import posixpath
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.parse

from cyclotron.code_state import _git, _sha256, _short, training_robot_model

MODEL_DIR_VARIABLE = "ASIMOV_1_MODEL_DIR"
# Where a run's recorded repository (owner/name of its origin remote) is fetched from.
REPOSITORY_URL = "https://github.com/{repo}.git"
GIT_TIMEOUT = 600

_REPO = re.compile(r"[\w-][\w.-]*/[\w-][\w.-]*")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class RobotModelUnavailable(Exception):
    """The robot model a run was trained with can't be loaded; the message says why."""


def cache_dir() -> str:
    """cyclotron's cache folder: ``CYCLOTRON_CACHE_DIR``, else ``$XDG_CACHE_HOME/cyclotron``, else
    ``~/.cache/cyclotron``."""
    if os.environ.get("CYCLOTRON_CACHE_DIR"):
        return os.path.abspath(os.path.expanduser(os.environ["CYCLOTRON_CACHE_DIR"]))
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache, "cyclotron")


def _model_folder(urdf_filepath: str) -> str:
    # sim-model/urdf/asimov_1.urdf -> sim-model: the folder ASIMOV_1_MODEL_DIR names, with the meshes the urdf loads.
    return posixpath.dirname(posixpath.dirname(urdf_filepath))


def _cache_key(commit: str, urdf_filepath: str) -> str:
    # <commit>__sim-model: runs at one commit can record different model folders. A / in it is written %2F, so each
    # model folder has a folder of its own.
    return f"{commit}__{urllib.parse.quote(_model_folder(urdf_filepath), safe='')}"


def _run_git(cwd: str, *args: str, check: bool = True) -> bool:
    """Run git in ``cwd``; True when it succeeds. Raises ``RobotModelUnavailable`` when git can't run, or, with
    ``check``, fails."""
    try:
        # A repository that asks for credentials (private, or renamed) fails instead of waiting for an answer.
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except OSError as error:
        raise RobotModelUnavailable(f"git can't run ({error.strerror})") from error
    except subprocess.TimeoutExpired as error:
        raise RobotModelUnavailable(f"git {args[0]} took more than {GIT_TIMEOUT} seconds") from error
    if result.returncode != 0 and check:
        lines = result.stderr.strip().splitlines()
        raise RobotModelUnavailable(f"git {args[0]}: {lines[-1] if lines else f'exit status {result.returncode}'}")
    return result.returncode == 0


def _check_fetchable(trained: dict) -> None:
    """Raise ``RobotModelUnavailable`` unless the record names a model folder at a commit of a repository."""
    commit, repo, path = trained.get("commit"), trained.get("repo"), trained.get("urdf_filepath")
    if not commit:
        raise RobotModelUnavailable("the run records no commit of it (git didn't track its urdf at training)")
    if trained.get("dirty") is not False:
        changes = "had uncommitted changes" if trained.get("dirty") else "may have had uncommitted changes"
        raise RobotModelUnavailable(f"its repository {changes} at training, so the commit alone isn't the model")
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise RobotModelUnavailable("the run records no repository to fetch it from")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise RobotModelUnavailable(f"the run records {commit!r} as its commit, not a full commit hash")
    # The path is joined to the cache folder, so it must stay inside it.
    parts = path.split("/") if isinstance(path, str) else []
    if len(parts) < 3 or any(part in ("", ".", "..") for part in parts):
        raise RobotModelUnavailable(f"its urdf ({path}) isn't in a <folder>/urdf/ folder of its repository")


def _check_urdf(urdf: str, trained: dict, where: str) -> None:
    digest = _sha256(urdf) if os.path.isfile(urdf) else None
    if digest != trained["sha256"]:
        found = f"has sha256 {digest[:12]}" if digest else "is missing"
        raise RobotModelUnavailable(f"{where} {found}, not the {trained['sha256'][:12]} the run recorded")


def _fetch(url: str, commit: str, folder: str, staging: str, objects: str | None) -> None:
    """Check out ``folder`` of ``commit`` from ``url`` into ``staging``, and remove git's own files. A sparse checkout
    of a partial clone downloads only the commit, its trees, and the files of ``folder`` that the local git object
    folder ``objects`` doesn't have; nothing when it has them all."""
    _run_git(staging, "init", "-q")
    if objects:
        # Git's alternates: objects found there are read from there, not downloaded.
        with open(os.path.join(staging, ".git", "objects", "info", "alternates"), "w") as f:
            f.write(objects + "\n")
    # Asked before the remote is a promisor: git would download a missing commit to answer.
    has_commit = _run_git(staging, "cat-file", "-e", f"{commit}^{{commit}}", check=False)
    _run_git(staging, "remote", "add", "origin", url)
    _run_git(staging, "config", "remote.origin.promisor", "true")
    _run_git(staging, "config", "remote.origin.partialclonefilter", "blob:none")
    _run_git(staging, "sparse-checkout", "set", folder)
    if not has_commit:
        _run_git(staging, "fetch", "-q", "--depth=1", "--filter=blob:none", "--no-tags", "origin", commit)
    # Downloads the files of the folder missing locally, in one batch.
    _run_git(staging, "checkout", "-q", "--detach", commit)
    shutil.rmtree(os.path.join(staging, ".git"))


def _fetch_into(folder: str, trained: dict, objects: str | None) -> None:
    """Fetch the run's model into the cache folder ``folder``; the caller holds its lock."""
    repo, commit, path = trained["repo"], trained["commit"], trained["urdf_filepath"]
    # Named after the folder, so one left by a fetch that was killed is removed here; and after the host, since hosts
    # sharing the cache may not see each other's lock.
    staging = f"{folder}.{socket.gethostname()}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    # The umask's mode, not tempfile.mkdtemp's 0700, so the other users of a shared cache can read it.
    os.mkdir(staging)
    try:
        try:
            _fetch(REPOSITORY_URL.format(repo=repo), commit, _model_folder(path), staging, objects)
        except RobotModelUnavailable as error:
            raise RobotModelUnavailable(f"fetching {repo} @ {_short(commit)} failed ({error})") from error
        _check_urdf(os.path.join(staging, path), trained, f"{path} at {repo} @ {_short(commit)}")
        try:
            os.rename(staging, folder)
        except OSError:
            # Put in place meanwhile by another host; its urdf is checked like any cached copy's.
            if not os.path.isdir(folder):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def materialize(trained: dict, objects: str | None = None) -> str:
    """The path of the urdf a run was trained with, in a copy of its model folder at the recorded commit in cyclotron's
    cache (``robot-models/<owner>__<name>/<commit>__<model folder>/``), fetched there first when missing.

    ``trained`` is the run's ``robot_model`` record (``training_robot_model``). ``objects`` is a local git object
    folder to take the commit's files from when it has them, so only what it lacks is downloaded. Raises
    ``RobotModelUnavailable`` saying why when the record names no clean commit of a repository, the fetch fails, or
    the urdf, fetched or cached, doesn't have the recorded sha256.
    """
    _check_fetchable(trained)
    repo, commit, path = trained["repo"], trained["commit"], trained["urdf_filepath"]
    parent = os.path.join(cache_dir(), "robot-models", repo.replace("/", "__"))
    folder = os.path.join(parent, _cache_key(commit, path))
    # A folder in place is complete and never changed, so it is read without the lock, which the other users of a
    # shared cache may not be able to take.
    if not os.path.isdir(folder):
        os.makedirs(parent, exist_ok=True)
        # One process fetches while the others wait for it, then find the folder in place.
        with open(folder + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.isdir(folder):
                _fetch_into(folder, trained, objects)
    urdf = os.path.join(folder, path)
    try:
        _check_urdf(urdf, trained, f"{path} in the cached copy {folder}")
    except RobotModelUnavailable as error:
        raise RobotModelUnavailable(f"{error} (delete that folder to fetch it again)") from error
    return urdf


def _local_objects(urdf: str) -> str | None:
    """The git object folder of the repository holding ``urdf``, if any. A run's model is usually an earlier commit of
    the current one, whose files that folder then already has."""
    folder = os.path.dirname(urdf)
    if not os.path.isdir(folder):
        return None
    return _git(folder, "rev-parse", "--path-format=absolute", "--git-path", "objects")


def _supply_hint(trained: dict) -> str:
    folder = posixpath.basename(_model_folder(trained.get("urdf_filepath") or ""))
    folder = folder if folder not in ("", ".", "..") else "model"
    if trained.get("repo") and trained.get("commit") and trained.get("dirty") is False:
        model = f"the {folder} folder of {trained['repo']} @ {_short(trained['commit'])}"
    else:
        model = f"the {folder} folder the run was trained with"
    return f"To load it, set {MODEL_DIR_VARIABLE}=<{model}>."


def load_training_robot_model(env_cfg, run_dir: str) -> tuple[str | None, str | None]:
    """Point ``env_cfg``'s robot at the robot model ``run_dir`` was trained with, before the environment is built, when
    the urdf it loads (``scene.robot.spawn.asset_path``) has another sha256 than the run recorded.

    ``env_cfg`` is the env config object. Left as it is when ``ASIMOV_1_MODEL_DIR`` is set, which names the model to
    load, or when the run's model can't be loaded (``materialize``); check 10d then compares the two.

    Returns ``(loaded, unavailable)``: the line naming the robot model when the run's is loaded, else why it isn't and
    how to supply it, for the robot model warning. Both are None when the run records no robot model.
    """
    trained = training_robot_model(run_dir) or {}
    if not trained.get("sha256"):
        return None, None
    spawn = env_cfg.scene.robot.spawn
    current = getattr(spawn, "asset_path", None)
    if not isinstance(current, str):
        return None, None
    loaded = f"Robot model: {trained.get('urdf_filepath')} @ {_short(trained.get('commit'))} (as trained)"
    if os.path.isfile(current) and _sha256(current) == trained["sha256"]:
        return loaded, None
    if MODEL_DIR_VARIABLE in os.environ:
        return None, (
            f"{MODEL_DIR_VARIABLE} is set, so the robot model it names is loaded instead of the run's; unset it to load"
            " the run's."
        )
    try:
        urdf = materialize(trained, _local_objects(current))
        # Isaac Lab converts the urdf into /tmp/IsaacLab/usd_<second>_<number from the seeded random module>, which
        # processes building their scene in the same second with the same seed share; its own folder keeps another
        # model's conversion out.
        usd_dir = os.path.join(tempfile.gettempdir(), "IsaacLab")
        os.makedirs(usd_dir, exist_ok=True)
        spawn.usd_dir = tempfile.mkdtemp(prefix="usd_", dir=usd_dir)
    except (RobotModelUnavailable, OSError) as error:
        return None, f"The robot model the run was trained with couldn't be loaded: {error}. {_supply_hint(trained)}"
    spawn.asset_path = urdf
    return loaded, None
