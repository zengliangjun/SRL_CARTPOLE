# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Cartpole-ORG",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
    },
)

gym.register(
    id="Cartpole-ORG-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
    },
)

##

gym.register(
    id="Cartpole-Swing",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    #disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_swing_env_cfg:CartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SwingPPORunnerCfg",
    },
)

gym.register(
    id="Cartpole-Swing-PLAY",
    entry_point="isaaclabex.envs.rl_env_exts:ManagerBasedRLEnv_Extends",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_swing_env_cfg:CartpoleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SwingPPORunnerCfg",
    },
)
