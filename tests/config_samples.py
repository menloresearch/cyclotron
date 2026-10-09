"""Trimmed env and agent configs shaped like the ones a run saves, shared by the tests that compare them."""

from __future__ import annotations

import copy

import yaml

SLOT_JOINTS = ["left_hip_pitch_joint", "right_hip_pitch_joint"]


def dump(config: dict) -> str:
    """Write a config the way Isaac Lab's dump_yaml does: Python tags, keys in their original order."""
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def term(func: str, scale=None, **params) -> dict:
    return {
        "func": func,
        "params": params,
        "modifiers": None,
        "noise": {"func": "isaaclab.utils.noise.noise_model:uniform_noise", "n_min": -0.01, "n_max": 0.01},
        "clip": None,
        "scale": scale,
        "history_length": 0,
        "flatten_history_dim": True,
    }


def make_env() -> dict:
    """A trimmed env config shaped like Isaac Lab's class_to_dict output, with tuples and slices left in."""
    asset_cfg = {"name": "robot", "joint_names": list(SLOT_JOINTS), "joint_ids": slice(None), "preserve_order": True}
    return {
        "sim": {"dt": 0.005, "gravity": (0.0, 0.0, -9.81)},
        "decimation": 4,
        "seed": 42,
        "scene": {
            "num_envs": 8192,
            "robot": {
                "init_state": {"pos": (0.0, 0.0, 0.639), "joint_pos": {".*_hip_pitch_joint": -0.15}},
                "actuators": {
                    "hip_pitch": {
                        "class_type": "isaaclab.actuators.actuator_pd:DelayedPDActuator",
                        "joint_names_expr": [".*_hip_pitch_joint"],
                        "effort_limit": 45.0,
                        "stiffness": 150.0,
                        "damping": 5.0,
                    }
                },
            },
        },
        "observations": {
            "policy": {
                "concatenate_terms": True,
                "enable_corruption": True,
                "history_length": None,
                "base_ang_vel": term(
                    "cyclotron.tasks.locomotion.mdp.observations:delayed_obs", 0.25, quantity="base_ang_vel"
                ),
                "joint_pos_slot01": term(
                    "isaaclab.envs.mdp.observations:joint_pos_rel", 1.0, asset_cfg=copy.deepcopy(asset_cfg)
                ),
                "actions": term("isaaclab.envs.mdp.observations:last_action"),
            },
            "critic": {"base_lin_vel": term("isaaclab.envs.mdp.observations:base_lin_vel")},
        },
        "actions": {
            "joint_pos": {
                "class_type": "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction",
                "debug_vis": False,
                "joint_names": list(SLOT_JOINTS),
                "scale": 0.25,
                "offset": 0.0,
                "preserve_order": True,
                "use_default_offset": True,
            }
        },
        "commands": {"twist": {"ranges": {"lin_vel_x": (-0.6, 0.8)}}},
        "events": {"push_robot": {"func": "isaaclab.envs.mdp.events:push_by_setting_velocity"}},
    }


def make_agent() -> dict:
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    return {
        "seed": 1,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "clip_actions": None,
        "class_name": "OnPolicyRunner",
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": distribution,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
    }
