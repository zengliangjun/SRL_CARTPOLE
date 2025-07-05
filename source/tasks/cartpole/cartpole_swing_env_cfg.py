# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

from isaaclabex.envs import rl_env_exts_cfg
from isaaclabex.envs.mdp.statistics import joints
from isaaclabex.envs.mdp.rewards import statistic_joints
from isaaclabex.envs.mdp.commands import commands_cfg

from isaaclabex.envs.managers import term_cfg

from . import cartpole_env_cfg

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    swing = commands_cfg.CartpoleCommandCfg(
        asset_name="robot",
        statistics_name="joints",
        resampling_time_range=(4.0, 8.0),

        car_pos = 0.3,
        car_swing = 0.5,
        pole_swing = 0.1,
        # debug_vis=False,

        ranges=commands_cfg.CartpoleCommandCfg.Ranges(
            car_pos=(- 0.6, 0.6), car_swing=(0.1, 0.7), pole_swing=(0.05 * math.pi , 0.3 * math.pi)
        )
    )


@configclass
class EventCfg(cartpole_env_cfg.EventCfg):
    """Configuration for events."""

    # interval
    interval_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-0, 0)}},
    )


@configclass
class RewardsCfg(cartpole_env_cfg.RewardsCfg):
    """Reward terms for the MDP."""

    rew_car_pos = RewTerm(
        func=statistic_joints.rew_car_pos,
        weight=0.8,
        params={
            "command_name": "swing",
            "statistics_name": "joints",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"], preserve_order = True)},
    )

    rew_car_swing = RewTerm(
        func=statistic_joints.rew_car_swing,
        weight=0.5,
        params={
            "command_name": "swing",
            "statistics_name": "joints",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"], preserve_order = True)},
    )
    rew_pole_pos = RewTerm(
        func=statistic_joints.rew_pole_pos,
        weight=0.25,
        params={
            "command_name": "swing",
            "statistics_name": "joints",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"], preserve_order = True)},
    )
    rew_pole_swing = RewTerm(
        func=statistic_joints.rew_pole_swing,
        weight=0.5,
        params={
            "command_name": "swing",
            "statistics_name": "joints",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart", "cart_to_pole"], preserve_order = True)},
    )
    def __post_init__(self) -> None:
        self.alive = None
        self.pole_pos = None
        self.cart_vel = None
        self.pole_vel = None


@configclass
class StatisticsCfg:
    joints = term_cfg.StatisticsTermCfg(
        func = joints.StatusJPos,
        params = {
            "asset_cfg":
            SceneEntityCfg("robot"),
        },
        export_interval = 6000000,
    )

##
# Environment configuration
##

@configclass
class CartpoleEnvCfg(rl_env_exts_cfg.ManagerBasedRLExtendsCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: cartpole_env_cfg.CartpoleSceneCfg = cartpole_env_cfg.CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: cartpole_env_cfg.ObservationsCfg = cartpole_env_cfg.ObservationsCfg()
    actions: cartpole_env_cfg.ActionsCfg = cartpole_env_cfg.ActionsCfg()
    #events: cartpole_env_cfg.EventCfg = cartpole_env_cfg.EventCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    # rewards: cartpole_env_cfg.RewardsCfg = cartpole_env_cfg.RewardsCfg()

    terminations: cartpole_env_cfg.TerminationsCfg = cartpole_env_cfg.TerminationsCfg()
    statistics: StatisticsCfg = StatisticsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 30
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class CartpoleEnvCfg_PLAY(CartpoleEnvCfg):

    def __post_init__(self) -> None:
        super(CartpoleEnvCfg_PLAY, self).__post_init__()
        self.scene.num_envs = 50
