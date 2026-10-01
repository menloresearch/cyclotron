"""View an Asimov policy shared on the Hugging Face Hub.

Downloads ``policy.onnx`` and the yaml files from a repo shared with ``./isaac_asimov.sh --share``, builds the Play
task with the policy settings from its ``env.yaml``, and runs the ONNX policy with onnxruntime.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

from isaac_asimov.hub import HubError, apply_env_yaml, download_asimov_model, infer_play_task, load_yaml

parser = argparse.ArgumentParser(
    description="View an Asimov policy shared on the Hugging Face Hub.",
    epilog="Example: ./isaac_asimov.sh --view <org>/<model>",
)
parser.add_argument("repo_id", type=str, help="Hugging Face model id, e.g. <org>/<model>.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Without a display, Isaac Sim fails slowly and unclearly when it tries to open its window. The flags can also be set
# through AppLauncher's HEADLESS and LIVESTREAM environment variables.
has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
headless = args_cli.headless or os.environ.get("HEADLESS", "0") == "1"
livestream = int(args_cli.livestream or 0) > 0 or os.environ.get("LIVESTREAM", "0") in ("1", "2")
if not (has_display or headless or livestream):
    sys.exit(
        "[ERROR] No display found. To watch from another machine, run with --livestream 2 (same network or VPN)"
        ' or PUBLIC_IP=<ip> ... --livestream 1 (internet). See the README section "View a shared policy".'
    )

# Resolve the model before starting Isaac Sim, so mistakes fail fast.
try:
    model = download_asimov_model(args_cli.repo_id)
    env_cfg_data = load_yaml(model["env_yaml"])
    agent_cfg_data = load_yaml(model["agent_yaml"]) if model["agent_yaml"] else None
    task = infer_play_task(env_cfg_data, agent_cfg_data)
except HubError as e:
    sys.exit(f"[ERROR] {e}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import time

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaac_asimov.tasks  # noqa: F401


def main():
    print(f"[INFO] Viewing {args_cli.repo_id} on task {task}")
    env_cfg = parse_env_cfg(task, device=args_cli.device)
    changed, missing = apply_env_yaml(env_cfg, env_cfg_data)
    print(f"[INFO] Applied {len(changed)} settings from env.yaml.")
    if missing:
        print(f"[WARN] {len(missing)} env.yaml settings have no counterpart in {task} and were left out:")
        for key in missing:
            print(f"  {key}")

    session = ort.InferenceSession(model["onnx"], providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        sys.exit("[ERROR] policy.onnx has more than one input. Recurrent policies aren't supported by --view.")
    input_name, input_shape = inputs[0].name, inputs[0].shape

    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    obs = env.get_observations()
    policy_obs = obs if isinstance(obs, torch.Tensor) else obs["policy"]
    if isinstance(input_shape[-1], int) and input_shape[-1] != policy_obs.shape[-1]:
        env.close()
        sys.exit(
            f"[ERROR] policy.onnx expects {input_shape[-1]} observations, but task {task} provides"
            f" {policy_obs.shape[-1]}."
        )
    # Exports traced with a fixed batch of 1 are run one environment at a time.
    single_batch = input_shape[0] == 1

    dt = env.unwrapped.step_dt
    while simulation_app.is_running():
        start_time = time.time()
        policy_obs = obs if isinstance(obs, torch.Tensor) else obs["policy"]
        obs_np = policy_obs.detach().cpu().numpy().astype(np.float32)
        if single_batch:
            actions_np = np.concatenate([session.run(None, {input_name: row[None]})[0] for row in obs_np])
        else:
            actions_np = session.run(None, {input_name: obs_np})[0]
        with torch.inference_mode():
            actions = torch.from_numpy(actions_np).to(env.unwrapped.device)
            obs, _, _, _ = env.step(actions)

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
