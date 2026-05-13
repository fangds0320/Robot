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
