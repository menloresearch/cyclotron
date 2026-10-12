"""Available Asimov-1 environments."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List Cyclotron environments.")
parser.add_argument("--keyword", type=str, default=None, help="Optional task-name filter.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym

import cyclotron.tasks  # noqa: F401, E402


def main() -> None:
    print("Available Cyclotron environments:", flush=True)
    for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
        if task_spec.id.startswith("Asimov1-") and (
            args_cli.keyword is None or args_cli.keyword in task_spec.id
        ):
            config = task_spec.kwargs.get("env_cfg_entry_point", "")
            print(f"  {task_spec.id:<36} {config}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
