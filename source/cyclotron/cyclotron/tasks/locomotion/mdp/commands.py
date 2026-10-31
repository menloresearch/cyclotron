"""Asimov velocity commands."""

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class HeadingHoldVelocityCommand(UniformVelocityCommand):
    """Hold a moving command's initial heading across consecutive resamples."""

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # None or a slice when every environment is reset.
        env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self.is_heading_env[env_ids] = False
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previous_heading = self.is_heading_env[env_ids].clone()
        previous_standing = self.is_standing_env[env_ids].clone()
        previous_target = self.heading_target[env_ids].clone()
        super()._resample_command(env_ids)
        if self.cfg.heading_command and len(env_ids):
            continuing = previous_heading & ~previous_standing & self.is_heading_env[env_ids]
            current_heading = self.robot.data.heading_w.torch[env_ids]
            self.heading_target[env_ids] = torch.where(continuing, previous_target, current_heading)


@configclass
class HeadingHoldVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = HeadingHoldVelocityCommand
