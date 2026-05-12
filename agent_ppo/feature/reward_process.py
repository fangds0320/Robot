# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — 增强版自定义奖励处理器

在基线奖励基础上新增 10+ 个与评分系统对齐的奖励函数：
  - 运动控制：feet_stumble / feet_slide / feet_air_time / joint_position_penalty
  - 姿态稳定：flat_orientation（增强版，exp 衰减）
  - 导航避障：obstacle_evasion / approach_goal / reach_goal
  - 终止惩罚：termination
  - 评分对齐组合：standard_score_aligned / track_score_aligned

其余通用 locomotion reward（track_lin_vel_xy / joint_acc / action_rate 等）
继承自 RewardProcessBase（见 tools/base_env/base_reward.py），
在 TOML 中激活即可，无需在此重复实现。
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):

    # ======================================================================
    # 运动控制奖励
    # ======================================================================

    def _reward_forward_velocity(self):
        """前向速度奖励：机器人本体坐标系下 x 方向速度（越大越好）"""
        robot = self._get_robot_asset()
        return robot.data.root_lin_vel_b[:, 0]

    def _reward_feet_stumble(self):
        """惩罚脚撞到垂直面（楼梯边缘、墙壁）。

        阈值 5× 对齐 legged_gym 原版：当水平力 > 5× 垂直力时判为撞击。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]

        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)

        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_feet_slide(self):
        """惩罚脚部在地面上滑动（接触时仍有速度）。

        对齐 Isaac Lab feet_slide：用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """奖励长步幅（移动时脚部滞空时间超过阈值）。

        Args:
            command_name: 命令项名称。
            threshold: 最小滞空时间阈值。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

        is_moving = torch.norm(self.env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_air_time_variance_penalty(self):
        """惩罚脚部滞空/接触时间的方差（步态对称性）。"""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")

        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        )

    def _reward_joint_position_penalty(self, stand_still_scale: float = 5.0, velocity_threshold: float = 0.1):
        """惩罚关节位置偏离默认姿态。

        Args:
            stand_still_scale: 静止时的缩放因子（静止时惩罚加重）。
            velocity_threshold: 判断是否移动的速度阈值。
        """
        asset = self._get_robot_asset()
        cmd = torch.linalg.norm(self.env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        reward = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
            reward,
            stand_still_scale * reward,
        )

    def _reward_joint_vel(self):
        """惩罚大的关节速度（鼓励平滑运动）。"""
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    # ======================================================================
    # 姿态稳定奖励
    # ======================================================================

    def _reward_flat_orientation(self):
        """惩罚基座偏离水平姿态（exp 衰减，与评分系统对齐）。

        使用 projected_gravity 的 xy 分量，值越小越接近水平。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_pose_stability(self, orientation_scale: float = 10.0):
        """姿态稳定性奖励（exp 形式），与评分系统中的 pose_score 对齐。

        Args:
            orientation_scale: 朝向误差的缩放因子。
        """
        asset = self._get_robot_asset()
        roll_pitch_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
        return torch.exp(-roll_pitch_error * orientation_scale)

    # ======================================================================
    # 导航与避障奖励
    # ======================================================================

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

        使用 height_scan 近场窗口检测正前方高障碍物，用角速度检测转向。

        Grid layout (16x16, offset 0.75m fwd, res=0.1m):
          reshaped (N, 16, 16) -> dim0=y_idx, dim1=x_idx
          y: -0.75m(idx0) .. +0.75m(idx15)
          x: 0.0m(idx0) .. 1.5m(idx15)

        Default window:
          Y [3:13] = -0.45m ~ +0.55m（通道宽度，捕捉侧壁）
          X [:10]  = 0.0m ~ 0.9m（约 1s 反应距离）

        Returns: blocked * not_evading * has_fwd_cmd
        """
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

    def _reward_approach_goal(self):
        """接近迷宫出口奖励：距离减少→正奖励，距离增加→负奖励。

        需要 TerrainExitManager 设置 env.goal_positions。
        """
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

    def _reward_reach_goal(self, threshold: float = 0.6):
        """到达目标奖励（distance < threshold 时返回 1.0）。

        Args:
            threshold: 判定到达目标的距离阈值（米），需与 _goal_reached_termination 一致。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    # ======================================================================
    # 终止惩罚
    # ======================================================================

    def _reward_termination(self):
        """惩罚真正的失败（终止且非超时截断且非到达目标）。

        对应 legged_gym 的 reset_buf * ~time_out_buf 逻辑，同时排除导航成功终止。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs

        if "goal_reached" in term_mgr.active_terms:
            goal_done = term_mgr.get_term("goal_reached")
            failure = failure & ~goal_done

        return failure.float()

    def _reward_navigation_time(self):
        """每步固定惩罚，鼓励快速到达出口（由 weight 控制大小）。"""
        return torch.ones(self.env.num_envs, device=self.env.device)

    # ======================================================================
    # 能量效率奖励
    # ======================================================================

    def _reward_energy_efficiency(self, power_scale: float = 1.0):
        """能量效率奖励，与评分系统中的 energy_score 对齐。

        计算机械功率：扭矩 × 速度，exp 衰减（功率越低奖励越高）。

        Args:
            power_scale: 功率计算缩放因子。
        """
        asset = self._get_robot_asset()
        power = torch.sum(torch.abs(asset.data.joint_torque * asset.data.joint_vel), dim=1)
        return torch.exp(-power * power_scale)

    # ======================================================================
    # 与评分系统对齐的组合奖励（可选，通过 TOML 激活）
    # ======================================================================

    def _reward_standard_score_aligned(self,
                                       forward_scale: float = 2.0,
                                       energy_scale: float = 0.1,
                                       pose_scale: float = 5.0):
        """Standard 模式评分对齐组合奖励。

        total_score = 0.4×forward + 0.2×time + 0.2×energy + 0.2×pose
        """
        asset = self._get_robot_asset()

        forward_reward = asset.data.root_lin_vel_b[:, 0] * forward_scale
        energy_reward = self._reward_energy_efficiency(power_scale=energy_scale)
        pose_reward = self._reward_pose_stability(orientation_scale=pose_scale)

        total = 0.4 * forward_reward + 0.2 * energy_reward + 0.2 * pose_reward
        return total

    def _reward_track_score_aligned(self,
                                    energy_scale: float = 0.2,
                                    pose_scale: float = 10.0):
        """Track 模式评分对齐组合奖励。

        total_score = completion_factor × (0.4×time + 0.4×pose + 0.2×energy)
        """
        pose_reward = self._reward_pose_stability(orientation_scale=pose_scale)
        energy_reward = self._reward_energy_efficiency(power_scale=energy_scale)

        base_reward = 0.4 * pose_reward + 0.2 * energy_reward

        # 基于距离的进度估计
        completion_factor = 1.0
        if hasattr(self.env, 'goal_positions') and self.env.goal_positions is not None:
            robot = self._get_robot_asset()
            robot_pos = robot.data.root_pos_w[:, :2]
            goal_pos = self.env.goal_positions[:, :2]
            distance = torch.norm(goal_pos - robot_pos, dim=1)
            completion_factor = 1.0 - torch.clamp(distance / 10.0, 0.0, 1.0)

        total = completion_factor * base_reward
        total += 0.2 * self._reward_forward_velocity()
        return total
