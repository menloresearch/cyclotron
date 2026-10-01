from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import tempfile
import types

import pytest

from isaac_asimov.hub import (
    HubError,
    apply_env_yaml,
    download_asimov_model,
    infer_play_task,
    load_yaml,
    overlay_cfg,
    strip_local_paths,
)

# Trimmed copy of a real shared env.yaml: Isaac Lab tags, training-machine paths and internal module names.
ENV_YAML = """\
sim:
  dt: 0.005
  render_interval: 4
  gravity: !!python/tuple
  - 0.0
  - 0.0
  - -9.81
seed: 42
decimation: 4
scene:
  num_envs: 8192
  robot:
    class_type: isaaclab.assets.articulation.articulation:Articulation
    prim_path: /World/envs/env_.*/Robot
    spawn:
      asset_path: /home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf
    init_state:
      pos: !!python/tuple
      - 0.0
      - 0.0
      - 0.639
      joint_pos:
        left_knee_joint: 0.45
        .*_hip_roll_joint: 0.0
    actuators:
      hip_pitch:
        class_type: isaaclab.actuators.actuator_pd:DelayedPDActuator
        joint_names_expr:
        - .*_hip_pitch_joint
        effort_limit: 45.0
        stiffness: 150.0
        damping: 5.0
      neck:
        stiffness: 10.0
observations:
  policy:
    enable_corruption: true
    base_ang_vel:
      func: isaaclab_tasks.manager_based.locomotion.velocity.config.asimov_1.mdp.observations:delayed_obs
      params:
        quantity: base_ang_vel
        max_lag: 1
      noise:
        func: isaaclab.utils.noise.noise_model:uniform_noise
        n_min: -0.01
      scale: 0.25
    joint_pos:
      params:
        asset_cfg:
          name: robot
          joint_names:
          - left_hip_pitch_joint
          joint_ids: !!python/object/apply:builtins.slice
          - null
          - null
          - null
    extra_term:
      scale: 1.0
  amp:
    concatenate_terms: true
actions:
  joint_pos:
    class_type: isaaclab.envs.mdp.actions.joint_actions:JointPositionAction
    joint_names:
    - left_hip_pitch_joint
    - left_knee_joint
    scale: 0.25
"""


# Stand-ins for the Isaac Lab config classes of a Play task.
@dataclasses.dataclass
class SimCfg:
    dt: float = 0.01
    render_interval: int = 2
    gravity: tuple = (0.0, 0.0, -9.81)


@dataclasses.dataclass
class InitStateCfg:
    pos: tuple = (0.0, 0.0, 0.7)
    joint_pos: dict = dataclasses.field(default_factory=lambda: {".*": 0.0})


@dataclasses.dataclass
class ActuatorCfg:
    joint_names_expr: list = dataclasses.field(default_factory=list)
    effort_limit: float | None = None
    stiffness: float = 100.0
    damping: float = 2.0
    class_type: str = "local.Actuator"


@dataclasses.dataclass
class SpawnCfg:
    asset_path: str = "/repo/asimov_1.urdf"


@dataclasses.dataclass
class RobotCfg:
    prim_path: str = "{ENV_REGEX_NS}/Robot"
    spawn: SpawnCfg = dataclasses.field(default_factory=SpawnCfg)
    init_state: InitStateCfg = dataclasses.field(default_factory=InitStateCfg)
    actuators: dict = dataclasses.field(default_factory=lambda: {"hip_pitch": ActuatorCfg()})


@dataclasses.dataclass
class SceneCfg:
    num_envs: int = 32
    robot: RobotCfg = dataclasses.field(default_factory=RobotCfg)


@dataclasses.dataclass
class SceneEntityCfg:
    name: str = "robot"
    joint_names: list | None = None
    joint_ids: slice = slice(None)


@dataclasses.dataclass
class ObsTermCfg:
    func: str = "local.func"
    params: dict = dataclasses.field(default_factory=dict)
    noise: str | None = None
    scale: float | None = None


@dataclasses.dataclass
class PolicyObsCfg:
    enable_corruption: bool = False
    base_ang_vel: ObsTermCfg = dataclasses.field(
        default_factory=lambda: ObsTermCfg(params={"quantity": "base_ang_vel", "max_lag": 0})
    )
    joint_pos: ObsTermCfg = dataclasses.field(
        default_factory=lambda: ObsTermCfg(params={"asset_cfg": SceneEntityCfg()})
    )


@dataclasses.dataclass
class ObservationsCfg:
    policy: PolicyObsCfg = dataclasses.field(default_factory=PolicyObsCfg)


@dataclasses.dataclass
class JointPositionActionCfg:
    joint_names: list = dataclasses.field(default_factory=lambda: [".*"])
    scale: float = 0.5
    class_type: str = "local.Action"


@dataclasses.dataclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = dataclasses.field(default_factory=JointPositionActionCfg)


@dataclasses.dataclass
class EnvCfg:
    sim: SimCfg = dataclasses.field(default_factory=SimCfg)
    seed: int | None = None
    decimation: int = 2
    scene: SceneCfg = dataclasses.field(default_factory=SceneCfg)
    observations: ObservationsCfg = dataclasses.field(default_factory=ObservationsCfg)
    actions: ActionsCfg = dataclasses.field(default_factory=ActionsCfg)


def env_yaml_data(text: str = ENV_YAML) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "env.yaml")
        with open(path, "w") as f:
            f.write(text)
        return load_yaml(path)


def test_load_yaml_builds_tuples_and_slices_without_running_python():
    data = env_yaml_data()
    assert data["sim"]["gravity"] == (0.0, 0.0, -9.81)
    asset_cfg = data["observations"]["policy"]["joint_pos"]["params"]["asset_cfg"]
    assert asset_cfg["joint_ids"] == slice(None)


def test_apply_env_yaml_copies_policy_settings_only():
    cfg = EnvCfg()
    changed, missing = apply_env_yaml(cfg, env_yaml_data())

    assert cfg.sim.dt == 0.005
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == 4
    assert cfg.seed == 42
    assert cfg.scene.robot.init_state.pos == (0.0, 0.0, 0.639)
    assert cfg.scene.robot.init_state.joint_pos == {"left_knee_joint": 0.45, ".*_hip_roll_joint": 0.0}
    actuator = cfg.scene.robot.actuators["hip_pitch"]
    assert (actuator.stiffness, actuator.damping, actuator.effort_limit) == (150.0, 5.0, 45.0)
    assert cfg.actions.joint_pos.joint_names == ["left_hip_pitch_joint", "left_knee_joint"]
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.observations.policy.base_ang_vel.params == {"quantity": "base_ang_vel", "max_lag": 1}
    assert cfg.observations.policy.base_ang_vel.scale == 0.25
    asset_cfg = cfg.observations.policy.joint_pos.params["asset_cfg"]
    assert isinstance(asset_cfg, SceneEntityCfg)
    assert asset_cfg.joint_names == ["left_hip_pitch_joint"]

    # The Play task's own values stay.
    assert cfg.scene.num_envs == 32
    assert cfg.observations.policy.enable_corruption is False
    assert cfg.observations.policy.base_ang_vel.noise is None
    assert cfg.observations.policy.base_ang_vel.func == "local.func"
    assert actuator.class_type == "local.Actuator"
    assert cfg.actions.joint_pos.class_type == "local.Action"
    assert cfg.scene.robot.spawn.asset_path == "/repo/asimov_1.urdf"
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"

    assert "scene.robot.actuators.hip_pitch.stiffness" in changed
    assert sorted(missing) == ["observations.policy.extra_term", "scene.robot.actuators.neck"]


def test_strip_local_paths_keeps_file_names_and_prim_paths():
    text = """\
viewer:
  cam_prim_path: /OmniverseKit_Persp
scene:
  robot:
    prim_path: /World/envs/env_.*/Robot
    spawn:
      asset_path: /home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf
      usd_dir: null
      visual_material_path: material
  contact_forces:
    filter_prim_paths_expr:
    - /World/envs/env_.*/Robot/torso_link
  terrain:
    terrain_generator:
      cache_dir: '/tmp/isaaclab/terrains'
    usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/arrow_x.usd
recorders:
  dataset_export_dir_path: C:\\runs\\logs
amp_data:
  motion_files:
  - /home/menlo/motions/walk_slow.npz
  - ~/motions/run.npz
  joint_names:
  - left_hip_pitch_joint
"""
    stripped, removed = strip_local_paths(text)
    assert removed == [
        "/home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf",
        "/tmp/isaaclab/terrains",
        "C:\\runs\\logs",
        "/home/menlo/motions/walk_slow.npz",
        "~/motions/run.npz",
    ]
    assert stripped == (
        text.replace("/home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/", "")
        .replace("/tmp/isaaclab/", "")
        .replace("C:\\runs\\", "")
        .replace("/home/menlo/motions/", "")
        .replace("~/motions/", "")
    )


def test_strip_local_paths_keeps_policy_settings():
    stripped, removed = strip_local_paths(ENV_YAML)
    assert removed == ["/home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf"]
    original_cfg, shared_cfg = EnvCfg(), EnvCfg()
    assert apply_env_yaml(shared_cfg, env_yaml_data(stripped)) == apply_env_yaml(original_cfg, env_yaml_data())
    assert shared_cfg == original_cfg


def test_overlay_cfg_can_turn_a_term_off_but_not_rebuild_one():
    cfg = PolicyObsCfg()
    overlay_cfg(cfg, {"base_ang_vel": None})
    assert cfg.base_ang_vel is None

    changed, missing = overlay_cfg(cfg, {"base_ang_vel": {"scale": 1.0}})
    assert cfg.base_ang_vel is None
    assert (changed, missing) == ([], ["base_ang_vel"])


def test_infer_play_task():
    assert infer_play_task({}, {"experiment_name": "asimov_velocity_amp"}) == "Asimov1-Velocity-AMP-Play-v0"
    assert infer_play_task({}, {"experiment_name": "asimov1_velocity"}) == "Asimov1-Velocity-Play-v0"
    assert infer_play_task({"observations": {"amp": {}}}, None) == "Asimov1-Velocity-AMP-Play-v0"
    assert infer_play_task({"observations": {"policy": {}}}, None) == "Asimov1-Velocity-Play-v0"
    with pytest.raises(HubError, match="Unknown experiment"):
        infer_play_task({}, {"experiment_name": "something_else"})


@contextlib.contextmanager
def fake_hub(library_name="asimov", files=("policy.onnx", "env.yaml", "agent.yaml", "README.md"), error=None):
    """Replace huggingface_hub with a stand-in that serves one repo."""
    calls = {}

    class HfApi:
        def model_info(self, repo_id):
            calls["model_info"] = repo_id
            if error:
                raise error
            siblings = [types.SimpleNamespace(rfilename=name) for name in files]
            return types.SimpleNamespace(library_name=library_name, siblings=siblings)

    def snapshot_download(repo_id, allow_patterns, local_dir):
        calls["snapshot_download"] = (repo_id, allow_patterns, local_dir)

    module = types.ModuleType("huggingface_hub")
    module.HfApi, module.snapshot_download = HfApi, snapshot_download
    previous = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = module
    try:
        yield calls
    finally:
        if previous is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = previous


def test_download_asimov_model():
    with fake_hub() as calls:
        model = download_asimov_model("menlo/asimov-walk")
    local_dir = os.path.abspath(os.path.join("logs", "hf", "menlo__asimov-walk"))
    assert calls["snapshot_download"] == ("menlo/asimov-walk", ["policy.onnx", "*.yaml"], local_dir)
    assert model == {
        "dir": local_dir,
        "onnx": os.path.join(local_dir, "policy.onnx"),
        "env_yaml": os.path.join(local_dir, "env.yaml"),
        "agent_yaml": os.path.join(local_dir, "agent.yaml"),
    }


def test_download_asimov_model_agent_yaml_is_optional():
    with fake_hub(files=("policy.onnx", "env.yaml")):
        assert download_asimov_model("menlo/asimov-walk")["agent_yaml"] is None


def test_download_asimov_model_rejects_bad_ids_without_calling_the_hub():
    with fake_hub() as calls:
        for repo_id in ("not-a-repo-id", "a/b/c", "policy.onnx", "logs/rsl_rl/run/model_100.pt"):
            with pytest.raises(HubError):
                download_asimov_model(repo_id)
    assert calls == {}


def test_download_asimov_model_rejects_local_paths():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(HubError, match="use --play"):
            download_asimov_model(tmp)


def test_download_asimov_model_rejects_non_asimov_models():
    with fake_hub(library_name="transformers") as calls:
        with pytest.raises(HubError, match="not an Asimov model"):
            download_asimov_model("openai-community/gpt2")
    assert "snapshot_download" not in calls


def test_download_asimov_model_lists_missing_files():
    with fake_hub(files=("agent.yaml",)):
        with pytest.raises(HubError, match="missing policy.onnx, env.yaml"):
            download_asimov_model("menlo/asimov-walk")


def test_download_asimov_model_explains_missing_repos():
    RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    with fake_hub(error=RepositoryNotFoundError("404")):
        with pytest.raises(HubError, match="huggingface-cli login"):
            download_asimov_model("menlo/missing")
