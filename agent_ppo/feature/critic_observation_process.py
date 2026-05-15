# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256)] → 316 dim
critic 观测布局：[critic_proprio(60) | height_scan(256)] → 316 维

When extending to track terrain, please refer to the extension guide in
policy_observation_process.py; the critic observation must stay in sync
with the policy on the task-information convention.
扩展到 track 地形时，请参考 policy_observation_process.py 的扩展指引；
critic 观测需保持与 policy 同步的任务信息约定。
"""

import torch

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"

    def goal_position_in_robot_frame(self):
        """Compute goal position in robot frame.

        计算机器人坐标系下的目标位置。

        Returns:
            Goal observation tensor: [goal_x, goal_y, goal_yaw, goal_dist] (num_envs, 4)
            目标观测张量：[goal_x, goal_y, goal_yaw, goal_dist] (num_envs, 4)
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, 4, device=self.env.device)

        try:
            # Get robot asset from scene
            # 从场景中获取机器人资产
            robot = self.env.scene["robot"]
            robot_pos = robot.data.root_pos_w
            robot_yaw = robot.data.root_quat_w

            goal_pos = self.env.goal_positions[:, :2]
            goal_yaw = self.env.goal_yaw if hasattr(self.env, "goal_yaw") else torch.zeros(self.env.num_envs, device=self.env.device)

            # Transform goal position to robot frame
            # 将目标位置转换到机器人坐标系
            goal_vec = goal_pos - robot_pos[:, :2]
            cos_yaw = torch.cos(robot_yaw[:, 3])
            sin_yaw = torch.sin(robot_yaw[:, 3])
            goal_x = goal_vec[:, 0] * cos_yaw + goal_vec[:, 1] * sin_yaw
            goal_y = -goal_vec[:, 0] * sin_yaw + goal_vec[:, 1] * cos_yaw

            # Compute distance to goal
            # 计算到目标的距离
            goal_dist = torch.norm(goal_vec, dim=1)

            # Stack into goal observation
            # 堆叠成目标观测
            goal_obs = torch.stack([goal_x, goal_y, goal_yaw, goal_dist], dim=1)

            return goal_obs
        except Exception:
            return torch.zeros(self.env.num_envs, 4, device=self.env.device)

    def process(self):
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        return obs