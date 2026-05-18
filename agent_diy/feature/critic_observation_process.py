# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CriticObservationProcess — custom critic observation processor.
CriticObservationProcess — 自定义 critic 观测处理器。

critic obs layout: [critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2: num_goal_obs=0  → critic_obs = 316 dim
- Stage3:   num_goal_obs=4  → critic_obs = 320 dim
critic 观测布局：[critic_proprio(60) | height_scan(256) | goal(num_goal_obs)]
- Stage1/2：num_goal_obs=0  → critic_obs = 316 维
- Stage3：  num_goal_obs=4  → critic_obs = 320 维
"""

import torch

from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    """Critic observation processor with optional goal obs.

    与 Isaac Lab CriticCfg 对齐的 critic 观测处理器，可选拼接 goal obs。
    """

    target_group = "critic"

    def goal_position_in_robot_frame(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, 4, device=self.env.device)

        try:
            robot = self.env.scene["robot"]
            robot_pos = robot.data.root_pos_w
            robot_quat = robot.data.root_quat_w
            goal_pos = self.env.goal_positions[:, :2]
            goal_yaw_w = self.env.goal_yaw if hasattr(self.env, "goal_yaw") else torch.zeros(self.env.num_envs, device=self.env.device)

            qw, qx, qy, qz = robot_quat.unbind(dim=1)
            robot_yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )

            goal_vec = goal_pos - robot_pos[:, :2]
            cos_yaw = torch.cos(robot_yaw)
            sin_yaw = torch.sin(robot_yaw)
            goal_x = goal_vec[:, 0] * cos_yaw + goal_vec[:, 1] * sin_yaw
            goal_y = -goal_vec[:, 0] * sin_yaw + goal_vec[:, 1] * cos_yaw
            goal_yaw = torch.atan2(torch.sin(goal_yaw_w - robot_yaw), torch.cos(goal_yaw_w - robot_yaw))
            goal_dist = torch.norm(goal_vec, dim=1)
            return torch.stack([goal_x, goal_y, goal_yaw, goal_dist], dim=1)
        except Exception:
            return torch.zeros(self.env.num_envs, 4, device=self.env.device)

    def process(self):
        """Compute critic observation.

        计算 critic 观测。

        Stage1/2: critic_obs = 316
        Stage3:   critic_obs = 316 + goal(4) = 320
        Stage1/2：critic_obs = 316
        Stage3：  critic_obs = 316 + goal(4) = 320
        """
        obs = self.default_observation()

        if self._get_num_goal_obs() > 0:
            goal_obs = self.goal_position_in_robot_frame()
            obs = self.concatenate_terms(obs, goal_obs)

        return obs
