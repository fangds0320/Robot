# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PolicyObservationProcess — custom policy observation processor.
PolicyObservationProcess — 自定义 policy 观测处理器。

obs layout: [proprio(45) | height_scan(256)] → 301 dim
观测布局：[proprio(45) | height_scan(256)] → 301 维

Extending to track terrain (optional):
    In track terrain the environment additionally provides the following
    read-only attributes (not available in standard terrain):
      - env.goal_positions  (num_envs, 3)  — exit position in world frame
      - env.goal_yaw        (num_envs,)    — exit heading in world frame
    The environment always exposes these scene sensors (available in both
    standard and track terrains, accessed via env.scene.sensors["<name>"]):
      - "height_scanner"  — default forward ground-clearance scan
      - "nav_scanner"     — forward-looking occlusion scan (wider range,
                             suited for obstacle avoidance / turning)
    Players can construct their own obs from these inputs. After appending,
    update the Stage config (observation dim) and model input dim accordingly.

扩展到 track 地形时（可选）：
    track 地形下，环境会额外提供以下只读属性（standard 地形没有）：
      - env.goal_positions  (num_envs, 3)  — 出口在世界坐标系下的 3D 位置
      - env.goal_yaw        (num_envs,)    — 出口在世界坐标系下的朝向
    环境在两种地形下都会通过 env.scene.sensors["<name>"] 提供以下传感器：
      - "height_scanner"  — 默认前方地面高度扫描
      - "nav_scanner"     — 前瞻遮挡扫描（范围更大，适合避障 / 转向判断）
    选手可从这些属性和传感器自行构造 obs。
    拼接后需同步修改 Stage 的观测维度和 model 输入维度。
"""

import torch

from tools.base_env.observation_process import ObservationProcess


class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

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