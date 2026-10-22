from __future__ import annotations

import errno
import hashlib
import os
import shutil
import socket
import stat
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cyclotron import robot_model
from cyclotron.code_state import robot_model_check, write_code_state
from cyclotron.robot_model import cache_dir, load_training_robot_model, materialize, overrides_robot_model

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

URDF = "sim-model/urdf/asimov_1.urdf"
OLD_URDF = '<robot name="asimov_1"><limit lower="-2.094395" upper="1"/></robot>\n'
NEW_URDF = '<robot name="asimov_1"><limit lower="-1.57" upper="1.57"/></robot>\n'
LITE = "sim-model-lite/urdf/asimov_1.urdf"
LITE_URDF = '<robot name="asimov_1"><!-- fewer meshes --></robot>\n'


def git(cwd, *args) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def model_repo(tmp_path, monkeypatch):
    """A model repository with two commits of a tiny model folder, served from a local bare repository the way
    GitHub serves menloresearch/asimov-1, and cyclotron's cache and temporary folder in tmp_path."""
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "test")
    git(source, "config", "commit.gpgsign", "false")
    (source / "sim-model" / "urdf").mkdir(parents=True)
    (source / "sim-model" / "assets" / "meshes").mkdir(parents=True)
    (source / "sim-model-lite" / "urdf").mkdir(parents=True)
    (source / "cad").mkdir()
    (source / URDF).write_text(OLD_URDF)
    (source / LITE).write_text(LITE_URDF)
    (source / "sim-model" / "assets" / "meshes" / "leg.STL").write_text("mesh")
    (source / "cad" / "frame.step").write_text("not part of the model")
    (source / "README.md").write_text("asimov-1\n")
    git(source, "add", ".")
    git(source, "commit", "-q", "-m", "old joint ranges")
    old = git(source, "rev-parse", "HEAD")
    (source / URDF).write_text(NEW_URDF)
    git(source, "commit", "-q", "-am", "joint ranges of the hardware")
    new = git(source, "rev-parse", "HEAD")

    remote = tmp_path / "remotes" / "menloresearch" / "asimov-1.git"
    remote.parent.mkdir(parents=True)
    git(tmp_path, "clone", "-q", "--bare", str(source), str(remote))
    # What GitHub allows: partial clones, and fetching a commit by its hash.
    git(remote, "config", "uploadpack.allowFilter", "true")
    git(remote, "config", "uploadpack.allowAnySHA1InWant", "true")
    monkeypatch.setattr(robot_model, "REPOSITORY_URL", f"file://{tmp_path}/remotes/{{repo}}.git")
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ASIMOV_1_MODEL_DIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    os.makedirs(tmp_path / "tmp")
    return SimpleNamespace(source=source, remote=remote, old=old, new=new, tmp=tmp_path)


def record(commit: str, text: str = OLD_URDF) -> dict:
    """The robot model a run trained on ``commit`` records, its urdf being ``text``."""
    robot = {"name": "asimov_1", "repo": "menloresearch/asimov-1", "urdf_filepath": URDF}
    robot.update(sha256=sha256(text), commit=commit, dirty=False)
    return robot


def trained_run(tmp_path, robot: dict | None) -> str:
    run = tmp_path / "run"
    write_code_state(str(run / "params"), {"robot_model": robot})
    return str(run)


def current_model(model_repo, commit: str):
    """A checkout of the model repository like third_party/asimov-1: a sparse, blob-less clone of ``commit``."""
    checkout = model_repo.tmp / "asimov-1"
    if not checkout.exists():
        url = f"file://{model_repo.remote}"
        git(model_repo.tmp, "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(checkout))
        git(checkout, "sparse-checkout", "set", "sim-model")
    git(checkout, "checkout", "-q", commit)
    return checkout / URDF


def env_cfg(urdf) -> SimpleNamespace:
    """The part of an env config object the robot model is loaded through."""
    spawn = SimpleNamespace(asset_path=str(urdf), usd_dir=None)
    return SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)))


def as_dict(cfg) -> dict:
    return {"scene": {"robot": {"spawn": {"asset_path": cfg.scene.robot.spawn.asset_path}}}}


def cached(model_repo, commit: str, folder: str = "sim-model"):
    return model_repo.tmp / "cache" / "robot-models" / "menloresearch__asimov-1" / f"{commit}__{folder}"


def test_cache_dir_is_cyclotron_cache_dir_else_xdg_cache_home_else_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "mine"))
    assert cache_dir() == str(tmp_path / "mine")
    monkeypatch.delenv("CYCLOTRON_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == str(tmp_path / "xdg" / "cyclotron")
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cache_dir() == str(tmp_path / "home" / ".cache" / "cyclotron")


def test_a_run_trained_on_another_commit_loads_that_commit_from_the_cache(model_repo):
    cfg = env_cfg(current_model(model_repo, model_repo.new))
    run = trained_run(model_repo.tmp, record(model_repo.old))
    assert robot_model_check(run, as_dict(cfg))[0] == "changed"

    lines = []
    loaded, unavailable = load_training_robot_model(cfg, run, lines.append)
    assert loaded == f"Robot model: {URDF} @ {model_repo.old[:7]} (as trained)" and unavailable is None
    assert lines == [f"[INFO] Fetching menloresearch/asimov-1 @ {model_repo.old[:7]} ..."]
    folder = cached(model_repo, model_repo.old)
    assert cfg.scene.robot.spawn.asset_path == str(folder / URDF)
    # Check 10d now compares the run with the model it was trained on.
    assert robot_model_check(run, as_dict(cfg)) == ("matched", None)
    # The model folder with its meshes, and the files at the repository's root; no other folder, no git files, no
    # staging folder left behind.
    assert (folder / "sim-model" / "assets" / "meshes" / "leg.STL").read_text() == "mesh"
    assert sorted(os.listdir(folder)) == ["README.md", "sim-model"]
    assert sorted(os.listdir(folder.parent)) == [folder.name, f"{folder.name}.lock"]
    # Converted to USD in a folder of its own, not one another process's robot model can share.
    usd_dir = cfg.scene.robot.spawn.usd_dir
    assert os.path.dirname(usd_dir) == str(model_repo.tmp / "tmp" / "IsaacLab") and os.listdir(usd_dir) == []


def test_the_cached_copy_is_used_again_without_fetching_and_checked_each_time(model_repo, monkeypatch):
    run = trained_run(model_repo.tmp, record(model_repo.old))
    cfg = env_cfg(current_model(model_repo, model_repo.new))
    load_training_robot_model(cfg, run)
    urdf = cfg.scene.robot.spawn.asset_path

    monkeypatch.setattr(robot_model, "REPOSITORY_URL", "file:///nonexistent/{repo}.git")
    cfg, lines = env_cfg(model_repo.source / URDF), []
    assert load_training_robot_model(cfg, run, lines.append)[1] is None and cfg.scene.robot.spawn.asset_path == urdf
    assert lines == []

    # A cached copy whose urdf was changed is refused, never fetched over.
    with open(urdf, "a") as f:
        f.write("<!-- edited -->\n")
    current = str(model_repo.source / URDF)
    cfg = env_cfg(current)
    loaded, unavailable = load_training_robot_model(cfg, run)
    assert loaded is None and cfg.scene.robot.spawn.asset_path == current and cfg.scene.robot.spawn.usd_dir is None
    assert f"{URDF} in the cached copy {cached(model_repo, model_repo.old)} has sha256" in unavailable
    assert "(delete that folder to fetch it again)" in unavailable


def test_the_current_model_repository_supplies_the_files_it_has_without_a_network(model_repo, monkeypatch):
    # third_party/asimov-1 checked out at the old pin, then moved to the new one: it has both commits' model files.
    current_model(model_repo, model_repo.old)
    urdf = current_model(model_repo, model_repo.new)
    monkeypatch.setattr(robot_model, "REPOSITORY_URL", "file:///nonexistent/{repo}.git")
    cfg = env_cfg(urdf)
    loaded, unavailable = load_training_robot_model(cfg, trained_run(model_repo.tmp, record(model_repo.old)))
    assert unavailable is None and cfg.scene.robot.spawn.asset_path == str(cached(model_repo, model_repo.old) / URDF)


def test_concurrent_loads_fetch_the_model_once(model_repo, monkeypatch):
    fetches = []
    fetch = robot_model._fetch
    monkeypatch.setattr(robot_model, "_fetch", lambda *args: fetches.append(args) or fetch(*args))
    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(lambda _: materialize(record(model_repo.old)), range(4)))
    assert len(fetches) == 1 and len(set(paths)) == 1


def test_runs_recording_other_model_folders_at_one_commit_have_a_folder_each(model_repo):
    lite = {**record(model_repo.old, LITE_URDF), "urdf_filepath": LITE}
    assert materialize(record(model_repo.old)) == str(cached(model_repo, model_repo.old) / URDF)
    assert materialize(lite) == str(cached(model_repo, model_repo.old, "sim-model-lite") / LITE)
    # Neither replaced the other.
    assert materialize(record(model_repo.old)) == str(cached(model_repo, model_repo.old) / URDF)
    assert not (cached(model_repo, model_repo.old) / "sim-model-lite").exists()
    assert robot_model._cache_key("0" * 40, "models/sim/urdf/a.urdf") == "0" * 40 + "__models%2Fsim"


def test_a_staging_folder_left_by_a_killed_fetch_is_replaced(model_repo):
    folder = cached(model_repo, model_repo.old)
    staging = folder.parent / f"{folder.name}.{socket.gethostname()}.staging"
    (staging / ".git").mkdir(parents=True)
    (staging / "half-fetched").write_text("")
    assert materialize(record(model_repo.old)) == str(folder / URDF)
    assert sorted(os.listdir(folder)) == ["README.md", "sim-model"] and not staging.exists()


def test_a_folder_another_host_put_in_place_meanwhile_is_used_and_checked(model_repo, monkeypatch):
    folder = cached(model_repo, model_repo.old)
    fetch = robot_model._fetch

    def fetch_while_another_host_does(url, commit, model_folder, staging, objects, edit=""):
        fetch(url, commit, model_folder, staging, objects)
        shutil.copytree(staging, folder)
        with open(folder / URDF, "a") as f:
            f.write(edit)

    monkeypatch.setattr(robot_model, "_fetch", fetch_while_another_host_does)
    assert materialize(record(model_repo.old)) == str(folder / URDF)
    assert sorted(os.listdir(folder.parent)) == [folder.name, f"{folder.name}.lock"]

    shutil.rmtree(folder)
    monkeypatch.setattr(robot_model, "_fetch", lambda *args: fetch_while_another_host_does(*args, edit="<!-- -->"))
    with pytest.raises(robot_model.RobotModelUnavailable, match="in the cached copy .* has sha256"):
        materialize(record(model_repo.old))


def test_the_cache_takes_the_umask_so_other_users_can_read_it(model_repo):
    umask = os.umask(0o002)
    try:
        urdf = materialize(record(model_repo.old))
    finally:
        os.umask(umask)
    folder = cached(model_repo, model_repo.old)
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o775
    assert stat.S_IMODE(os.stat(folder / "sim-model").st_mode) == 0o775
    assert stat.S_IMODE(os.stat(urdf).st_mode) == 0o664


def test_a_cache_that_cant_be_locked_is_a_note_and_a_folder_in_place_needs_no_lock(model_repo, monkeypatch):
    def flock(*args):
        raise OSError(errno.ENOLCK, "No locks available")

    current = str(current_model(model_repo, model_repo.new))
    run = trained_run(model_repo.tmp, record(model_repo.old))
    with monkeypatch.context() as patch:
        patch.setattr(robot_model.fcntl, "flock", flock)
        cfg = env_cfg(current)
        loaded, unavailable = load_training_robot_model(cfg, run)
        assert loaded is None and cfg.scene.robot.spawn.asset_path == current
        reason = f"couldn't be loaded: [Errno {errno.ENOLCK}] No locks available"
        assert unavailable.startswith(f"The robot model the run was trained with {reason}")
    materialize(record(model_repo.old))
    monkeypatch.setattr(robot_model.fcntl, "flock", flock)
    cfg = env_cfg(current)
    assert load_training_robot_model(cfg, run)[1] is None
    assert cfg.scene.robot.spawn.asset_path == str(cached(model_repo, model_repo.old) / URDF)


def test_a_run_on_the_current_model_loads_it_as_it_is(model_repo):
    urdf = current_model(model_repo, model_repo.old)
    cfg = env_cfg(urdf)
    loaded, unavailable = load_training_robot_model(cfg, trained_run(model_repo.tmp, record(model_repo.old)))
    # Only the urdf is known to be the run's.
    assert loaded == f"Robot model: {urdf} (same URDF as trained)" and unavailable is None
    assert cfg.scene.robot.spawn.asset_path == str(urdf) and cfg.scene.robot.spawn.usd_dir is None
    assert not (model_repo.tmp / "cache").exists()


def test_a_run_without_a_recorded_robot_model_is_left_alone(model_repo):
    cfg = env_cfg(current_model(model_repo, model_repo.new))
    assert load_training_robot_model(cfg, trained_run(model_repo.tmp, None)) == (None, None)
    assert load_training_robot_model(cfg, str(model_repo.tmp / "no-code-state")) == (None, None)
    assert cfg.scene.robot.spawn.asset_path == str(current_model(model_repo, model_repo.new))


def test_asimov_1_model_dir_names_the_model_to_load(model_repo, monkeypatch):
    urdf = current_model(model_repo, model_repo.new)
    monkeypatch.setenv("ASIMOV_1_MODEL_DIR", str(urdf.parent.parent))
    cfg = env_cfg(urdf)
    loaded, unavailable = load_training_robot_model(cfg, trained_run(model_repo.tmp, record(model_repo.old)))
    assert loaded is None and unavailable.startswith("ASIMOV_1_MODEL_DIR is set, so the robot model it names is loaded")
    assert cfg.scene.robot.spawn.asset_path == str(urdf) and not (model_repo.tmp / "cache").exists()
    # Naming the model the run was trained with loads it as trained.
    assert load_training_robot_model(cfg, trained_run(model_repo.tmp, record(model_repo.new, NEW_URDF)))[0]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"commit": None, "repo": None, "dirty": None}, "the run records no commit of it"),
        ({"dirty": True}, "its repository had uncommitted changes at training, so the commit alone isn't the model"),
        ({"dirty": None}, "its repository may have had uncommitted changes at training"),
        ({"repo": None}, "the run records no repository to fetch it from"),
        ({"commit": "732cc60"}, "the run records '732cc60' as its commit, not a full commit hash"),
        ({"urdf_filepath": "../urdf/asimov_1.urdf"}, "its urdf (../urdf/asimov_1.urdf) isn't in a <folder>/urdf/"),
        ({"repo": "menloresearch/missing"}, "fetching menloresearch/missing @ "),
        ({"commit": "0" * 40}, "fetching menloresearch/asimov-1 @ 0000000 failed (git fetch: "),
        ({"sha256": sha256("another urdf")}, f"{URDF} at menloresearch/asimov-1 @ "),
    ],
)
def test_a_model_that_cant_be_loaded_leaves_the_current_one_and_says_why(model_repo, changes, reason):
    current = str(current_model(model_repo, model_repo.new))
    cfg = env_cfg(current)
    run = trained_run(model_repo.tmp, {**record(model_repo.old), **changes})
    lines = []
    loaded, unavailable = load_training_robot_model(cfg, run, lines.append)
    assert loaded is None and not [line for line in lines if line.startswith("[WARNING]")]
    assert unavailable.startswith(f"The robot model the run was trained with couldn't be loaded: {reason}")
    assert unavailable.endswith(">.") and ". To load it, set ASIMOV_1_MODEL_DIR=<the " in unavailable
    assert cfg.scene.robot.spawn.asset_path == current and cfg.scene.robot.spawn.usd_dir is None
    # Nothing half-fetched is left to be found later.
    models = model_repo.tmp / "cache" / "robot-models"
    assert [path for path in models.glob("*/*") if path.is_dir()] == []


def test_why_it_cant_be_loaded_is_said_at_once_when_the_current_model_is_missing_too(model_repo):
    # Building the environment would stop first, saying only that the current urdf doesn't exist.
    cfg, lines = env_cfg(model_repo.tmp / "missing" / URDF), []
    run = trained_run(model_repo.tmp, {**record(model_repo.old), "commit": "0" * 40})
    loaded, unavailable = load_training_robot_model(cfg, run, lines.append)
    assert lines == ["[INFO] Fetching menloresearch/asimov-1 @ 0000000 ...", f"[WARNING] {unavailable}"]


def test_a_urdf_named_by_a_hydra_override_is_loaded_as_it_is():
    assert overrides_robot_model(["env.scene.num_envs=4", "env.scene.robot.spawn.asset_path=/models/a.urdf"])
    assert overrides_robot_model(["++env.scene.robot.spawn.asset_path=/models/a.urdf"])
    assert not overrides_robot_model(["env.scene.robot.spawn.usd_dir=/tmp/usd", "agent.seed=1", "--headless"])


def test_the_hint_names_the_commit_to_check_out_only_when_it_is_the_model():
    clean = record("732cc60dcb8f2b4fd26c3d7346b35f9b89c3cd47")
    assert robot_model._supply_hint(clean) == (
        "To load it, set ASIMOV_1_MODEL_DIR=<the sim-model folder of menloresearch/asimov-1 @ 732cc60>."
    )
    assert robot_model._supply_hint({**clean, "dirty": True}) == (
        "To load it, set ASIMOV_1_MODEL_DIR=<the sim-model folder the run was trained with>."
    )
    assert robot_model._supply_hint({**clean, "urdf_filepath": "asimov_1.urdf", "commit": None}) == (
        "To load it, set ASIMOV_1_MODEL_DIR=<the model folder the run was trained with>."
    )
