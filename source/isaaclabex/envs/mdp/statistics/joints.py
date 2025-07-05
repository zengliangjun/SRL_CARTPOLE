
from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

class StatusBase(ManagerTermBase):
    """关节统计基类，提供公共统计方法和缓冲区"""

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        """
        初始化统计缓冲区
        Args:
            cfg: 统计项配置，包含asset_cfg参数
            env: RL环境实例
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

        # 初始化关节统计缓冲区
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        joint_count = len(self.asset.data.joint_names)

        # Episode级别统计
        self.episode_variance_buf = torch.zeros((self.num_envs, joint_count),
                              device=self.device, dtype=torch.float)
        self.episode_mean_buf = torch.zeros_like(self.episode_variance_buf)
        self.zero_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """重置指定环境的统计缓冲区
        Args:
            env_ids: 需要重置的环境ID列表
        Returns:
            空字典（保持接口统一）
        """
        if env_ids is None or len(env_ids) == 0:
            return {}

        items = {}
        if 0 == self._env.common_step_counter % self.cfg.export_interval:

            mean = self.episode_mean_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names):
                name = name.replace("_joint", "")
                items[f"em/{name}"] = torch.mean(mean[:, id])
                items[f"em2/{name}"] = torch.sqrt(torch.mean(torch.square(mean[:, id])))

            variance = self.episode_variance_buf[env_ids]
            for id, name in enumerate(self.asset.data.joint_names):
                name = name.replace("_joint", "")
                items[f"ev/{name}"] = torch.mean(variance[:, id])
                items[f"ev2/{name}"] = torch.sqrt(torch.mean(torch.square(variance[:, id])))

        # 重置所有缓冲区
        buffers = [
            self.episode_variance_buf,
            self.episode_mean_buf,
        ]

        for buf in buffers:
            buf[env_ids] = 0

        return items

    def _episode_length(self) -> torch.Tensor:
        if -1 == self.cfg.episode_truncation:
            return self._env.episode_length_buf
        else:
            return torch.clamp_max(self._env.episode_length_buf, self.cfg.episode_truncation)

    def _calculate_episode(self, diff: torch.Tensor) -> None:
        """更新episode级别的统计量（Welford算法）
        Args:
            diff: 当前step的关节状态差值
        """
        episode_length_buf = self._episode_length()
        # 计算均值
        delta0 = diff - self.episode_mean_buf
        self.episode_mean_buf += delta0 / episode_length_buf[:, None]

        # 计算方差
        delta1 = diff - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (episode_length_buf[:, None] - 2)
            + delta0 * delta1
        ) / (episode_length_buf[:, None] - 1)

        # 处理新episode
        new_episode_mask = episode_length_buf <= 1
        # self.episode_mean_buf[new_episode_mask] = 0
        self.episode_variance_buf[new_episode_mask] = 0

    def _update_flag(self):
        self.zero_flag[...] = self._env.episode_length_buf <= 1

class StatusJPos(StatusBase):
    """关节位置统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self):
        """执行统计计算"""
        diff = self._calculate_withzero()
        self._calculate_episode(diff)
        self._update_flag()

    def _calculate_withzero(self) -> torch.Tensor:
        """直接返回关节位置"""
        return self.asset.data.joint_pos

class StatusJVel(StatusBase):
    """关节速度统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self):
        """执行统计计算"""
        diff = self.asset.data.joint_vel
        self._calculate_episode(diff)
        self._calculate_step(diff)
        self._update_flag()

class StatusJAcc(StatusBase):
    """关节加速度统计"""

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self):
        """执行统计计算"""
        diff = self.asset.data.joint_acc
        self._calculate_episode(diff)
        self._calculate_step(diff)
        self._update_flag()
