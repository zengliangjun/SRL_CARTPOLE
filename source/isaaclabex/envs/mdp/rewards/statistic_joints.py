from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence

import torch
from isaaclabex.envs.rl_env_exts import ManagerBasedRLEnv


if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.mdp.commands import cartpole_command
    from isaaclabex.envs.mdp.statistics import joints
    from isaaclabex.envs.managers.statistics_manager import StatisticsManager

def rew_car_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    statistics_name: str,
) -> torch.Tensor:

    command = env.command_manager.get_command(command_name)

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: joints.StatusJPos = manager.get_term(statistics_name)

    episode_mean = statistics.episode_mean_buf

    if isinstance(asset_cfg.joint_ids, slice) or asset_cfg.joint_ids == None:
        asset = env.scene[asset_cfg.name]
        joint_ids = [i for i in range(len(asset.joint_names))]
    else:
        joint_ids = asset_cfg.joint_ids

    car_pos_mean = episode_mean[:, joint_ids[0]]
    r_car_pos = torch.exp(- torch.square(car_pos_mean - command[:, 0]) * 6 / command[:, 1])
    r_car_pos[statistics.zero_flag] = 0
    return r_car_pos


def rew_car_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    statistics_name: str,
) -> torch.Tensor:

    command = env.command_manager.get_command(command_name)

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: joints.StatusJPos = manager.get_term(statistics_name)

    variance_mean = statistics.episode_variance_buf

    if isinstance(asset_cfg.joint_ids, slice) or asset_cfg.joint_ids == None:
        asset = env.scene[asset_cfg.name]
        joint_ids = [i for i in range(len(asset.joint_names))]
    else:
        joint_ids = asset_cfg.joint_ids

    car_pos_swing = torch.sqrt(variance_mean[:, joint_ids[0]])
    r_car_swing = torch.exp(- torch.square(car_pos_swing - command[:, 1]) * 4 / command[:, 1])
    r_car_swing[statistics.zero_flag] = 0
    return r_car_swing


def rew_pole_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    statistics_name: str,
) -> torch.Tensor:

    command = env.command_manager.get_command(command_name)

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: joints.StatusJPos = manager.get_term(statistics_name)

    episode_mean = statistics.episode_mean_buf
    variance_mean = statistics.episode_variance_buf

    if isinstance(asset_cfg.joint_ids, slice) or asset_cfg.joint_ids == None:
        asset = env.scene[asset_cfg.name]
        joint_ids = [i for i in range(len(asset.joint_names))]
    else:
        joint_ids = asset_cfg.joint_ids

    pole_pos_mean = episode_mean[:, joint_ids[1]]
    r_pole_pos = torch.exp(- torch.square(pole_pos_mean) * 6 / command[:, 2])

    r_pole_pos[statistics.zero_flag] = 0
    return r_pole_pos


def rew_pole_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    statistics_name: str,
) -> torch.Tensor:

    command = env.command_manager.get_command(command_name)

    assert isinstance(env, ManagerBasedRLEnv)
    manager: StatisticsManager = env.statistics_manager
    statistics: joints.StatusJPos = manager.get_term(statistics_name)

    episode_mean = statistics.episode_mean_buf
    variance_mean = statistics.episode_variance_buf

    if isinstance(asset_cfg.joint_ids, slice) or asset_cfg.joint_ids == None:
        asset = env.scene[asset_cfg.name]
        joint_ids = [i for i in range(len(asset.joint_names))]
    else:
        joint_ids = asset_cfg.joint_ids

    pole_pos_swing = torch.sqrt(variance_mean[:, joint_ids[1]])

    r_pole_swing = torch.exp(- torch.square(pole_pos_swing - command[:, 2]) * 4 / command[:, 2])
    r_pole_swing[statistics.zero_flag] = 0

    return r_pole_swing
