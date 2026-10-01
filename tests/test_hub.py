from __future__ import annotations

from cyclotron.hub import strip_local_paths

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
    changed = [(a, b) for a, b in zip(ENV_YAML.splitlines(), stripped.splitlines()) if a != b]
    assert changed == [
        (
            "      asset_path: /home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf",
            "      asset_path: asimov_1.urdf",
        )
    ]
