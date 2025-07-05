# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import CartpoleCommandCfg

class CartpoleCommand(CommandTerm):

    cfg: CartpoleCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: CartpoleCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # -- metrics
        self.metrics["error_car_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_car_swing"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pole_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pole_swing"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "CartpoleCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command_b.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    """
    Properties
    """
    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        from isaaclabex.envs.mdp.statistics import joints
        from isaaclabex.envs.managers.statistics_manager import StatisticsManager

        manager: StatisticsManager = self._env.statistics_manager
        statistics: joints.StatusJPos = manager.get_term(self.cfg.statistics_name)

        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # logs data
        self.metrics["error_car_pos"] += (
            torch.abs(self.command_b[:, 0] - statistics.episode_mean_buf[:, 0]) / max_command_step
        )
        self.metrics["error_car_swing"] += (
            torch.abs(self.command_b[:, 1] - torch.sqrt(statistics.episode_variance_buf[:, 0])) / max_command_step
        )
        self.metrics["error_pole_pos"] += (
            torch.abs(statistics.episode_mean_buf[:, 1]) / max_command_step
        )
        self.metrics["error_pole_swing"] += (
            torch.abs(self.command_b[:, 2] - torch.sqrt(statistics.episode_variance_buf[:, 1])) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)

        self.command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.car_pos)

        self.command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.car_swing)

        self.command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pole_swing)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # TODO
        pass

    def _debug_vis_callback(self, event):
        # TODO
        pass
