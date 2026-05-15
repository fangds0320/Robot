# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
自定义奖励处理器。

已有内置 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase，在 TOML 中激活即可。

本文件补充自定义奖励，所有新方法均包含防御性检查：
如果底层环境属性缺失（版本差异），安全降级返回零，不会中断训练。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):

    # ------------------------------------------------------------------
    # 导航目标奖励（原始保留）
    # ------------------------------------------------------------------

    def _reward_reach_goal(self, threshold: float = 0.6):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    # ------------------------------------------------------------------
    # 运动奖励（原始保留）
    # ------------------------------------------------------------------

    def _reward_forward_velocity(self):
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]

    # ------------------------------------------------------------------
    # 步态与足部质量奖励
    # ------------------------------------------------------------------

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        try:
            sensor_cfg = self._get_foot_sensor_cfg()
            contact_sensor = self.env.scene.sensors[sensor_cfg.name]
            if not getattr(contact_sensor.cfg, "track_air_time", False):
                return torch.zeros(self.env.num_envs, device=self.env.device)

            first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
            last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
            reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

            is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
            return reward * is_moving.float()
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_feet_slide(self):
        try:
            sensor_cfg = self._get_foot_sensor_cfg()
            asset_cfg = self._get_foot_asset_cfg()
            contact_sensor = self.env.scene.sensors[sensor_cfg.name]
            asset = self.env.scene[asset_cfg.name]

            # net_forces_w_history 兼容检查
            force_data = contact_sensor.data.net_forces_w_history
            if force_data is None:
                return torch.zeros(self.env.num_envs, device=self.env.device)
            contacts = (
                force_data[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
            )
            body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
            return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_feet_stumble(self):
        try:
            sensor_cfg = self._get_foot_sensor_cfg()
            contact_sensor = self.env.scene.sensors[sensor_cfg.name]

            forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
            forces_xy = torch.linalg.norm(
                contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2
            )
            return torch.any(forces_xy > 5 * forces_z, dim=1).float()
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    # ------------------------------------------------------------------
    # 姿态与能耗优化
    # ------------------------------------------------------------------

    def _reward_joint_position_penalty(self, stand_still_scale: float = 5.0, velocity_threshold: float = 0.1):
        try:
            asset = self._get_robot_asset()
            if not hasattr(asset.data, "default_joint_pos"):
                return torch.zeros(self.env.num_envs, device=self.env.device)

            cmd = torch.linalg.norm(self.env.command_manager.get_command("base_velocity"), dim=1)
            body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
            reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
            return torch.where(
                torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
                reward,
                stand_still_scale * reward,
            )
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_energy(self):
        """Energy penalty (torque × joint velocity).

        能耗惩罚（扭矩 × 角速度），用于鼓励节能运动。
        """
        try:
            asset = self._get_robot_asset()
            # Compute mechanical power: torque * velocity
            # 计算机械功率：扭矩 × 角速度
            power = torch.sum(torch.abs(asset.data.joint_torque * asset.data.joint_vel), dim=1)
            return power
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_correct_base_height(self, target_height: float = 0.38):
        """Penalize deviation from target base height.

        惩罚基座高度偏离目标高度，用于保持机器人稳定姿态。

        Args:
            target_height: Target base height in meters.
                          目标基座高度（米）。
        """
        try:
            asset = self._get_robot_asset()
            # Get current base height (z position)
            # 获取当前基座高度（z 位置）
            current_height = asset.data.root_pos_w[:, 2]
            # Compute height deviation penalty
            # 计算高度偏差惩罚
            height_error = torch.abs(current_height - target_height)
            return height_error
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_no_fly(self):
        """Penalize when all feet are off the ground (no-fly constraint).

        惩罚四脚腾空的情况，鼓励机器人保持至少一只脚接触地面。
        """
        try:
            sensor_cfg = self._get_foot_sensor_cfg()
            contact_sensor = self.env.scene.sensors[sensor_cfg.name]

            # Check which feet are in contact (force norm > threshold)
            # 检查哪些脚在接触地面（力的范数 > 阈值）
            forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
            foot_contacts = torch.norm(forces, dim=-1) > 1.0

            # Count number of feet in contact
            # 计算接触地面的脚的数量
            num_contacts = torch.sum(foot_contacts, dim=1)

            # Penalty = 1.0 when no feet are in contact, 0.0 otherwise
            # 当没有脚接触地面时惩罚为 1.0，否则为 0.0
            return (num_contacts == 0).float()
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    # ------------------------------------------------------------------
    # 终止惩罚
    # ------------------------------------------------------------------

    def _reward_termination(self):
        try:
            if not hasattr(self.env, "termination_manager"):
                return torch.zeros(self.env.num_envs, device=self.env.device)
            term_mgr = self.env.termination_manager
            if not hasattr(term_mgr, "terminated") or not hasattr(term_mgr, "time_outs"):
                return torch.zeros(self.env.num_envs, device=self.env.device)

            failure = term_mgr.terminated & ~term_mgr.time_outs

            if hasattr(term_mgr, "active_terms") and "goal_reached" in term_mgr.active_terms:
                goal_done = term_mgr.get_term("goal_reached")
                failure = failure & ~goal_done

            return failure.float()
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    # ------------------------------------------------------------------
    # Track 导航奖励（track 模式专用）
    # ------------------------------------------------------------------

    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        turn_std: float = 0.5,
    ):
        """惩罚前方被障碍阻挡时未主动转向。
        使用 height_scan 近场窗口检测正前方高障碍物（柱子/墙壁），
        用角速度检测规避转向。
        """
        try:
            asset = self._get_robot_asset()
            sensor = self.env.scene.sensors["height_scanner"]
            scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
            grid = scan.view(self.env.num_envs, 16, 16)
            window = grid[:, body_y_start:body_y_end, :near_x_end]
            col_blocked = (window < obstacle_threshold).any(dim=-1).float()
            blocked = col_blocked.mean(dim=-1)
            yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
            not_evading = torch.exp(-yaw_rate / turn_std)
            cmd = self.env.command_manager.get_command(command_name)
            has_fwd_cmd = (cmd[:, 0] > 0.05).float()
            return blocked * not_evading * has_fwd_cmd
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_approach_goal(self):
        """接近迷宫出口奖励：距离减少→正奖励，距离增加→负奖励。"""
        try:
            if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
                return torch.zeros(self.env.num_envs, device=self.env.device)
            robot = self._get_robot_asset()
            robot_pos = robot.data.root_pos_w[:, :2]
            goal_pos = self.env.goal_positions[:, :2]
            current_dist = torch.norm(goal_pos - robot_pos, dim=1)
            if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
                self.env._previous_goal_dist = current_dist.clone()
                return torch.zeros(self.env.num_envs, device=self.env.device)
            delta_dist = current_dist - self.env._previous_goal_dist
            term_mgr = self.env.termination_manager
            reset_mask = term_mgr.terminated | term_mgr.time_outs
            delta_dist[reset_mask] = 0.0
            self.env._previous_goal_dist = current_dist.clone()
            return -delta_dist
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_navigation_time(self):
        """每步固定惩罚，鼓励快速到达出口。"""
        try:
            return torch.ones(self.env.num_envs, device=self.env.device)
        except Exception:
            return torch.zeros(self.env.num_envs, device=self.env.device)