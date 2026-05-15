# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — custom reward processor (optimized for navigation).
RewardProcess — 自定义奖励处理器（导航优化版）。

This file contains optimized rewards for legged robot navigation competition:
本文件包含针对四足机器人导航竞赛优化的奖励项：

1. Basic locomotion rewards (inherited from base):
   基础运动奖励（继承自base）：
   - track_lin_vel_xy: velocity tracking / 速度跟踪
   - track_ang_vel_z: angular velocity tracking / 角速度跟踪
   - lin_vel_z: vertical velocity penalty / 垂直速度惩罚
   - ang_vel_xy: angular velocity penalty / 角速度惩罚
   - joint_acc: joint acceleration penalty / 关节加速度惩罚
   - joint_torques: joint torque penalty / 关节扭矩惩罚
   - action_rate: action smoothness / 动作平滑
   - undesired_contacts: contact penalty / 接触惩罚
   - flat_orientation: orientation penalty / 姿态惩罚

2. Navigation-specific rewards (custom):
   导航相关奖励（自定义）：
   - forward_velocity: forward progress / 前进进度
   - energy_efficiency: energy consumption / 能耗效率
   - base_height_tracking: base height stability / 基座高度稳定
   - approach_goal: goal approaching / 接近目标
   - reach_goal: goal reaching / 到达目标
   - obstacle_evasion: obstacle avoidance / 避障
   - termination: failure penalty / 失败惩罚
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):
    """Optimized reward processor for navigation competition.
    
    导航竞赛优化奖励处理器。
    """
    
    def _reward_forward_velocity(self):
        """Forward velocity reward: x-direction velocity in robot body frame.
        前向速度奖励：机器人本体坐标系下 x 方向速度。

        This encourages the robot to move forward, which is critical for
        both standard mode (forward distance score) and track mode (completion).
        这鼓励机器人前进，对标准模式（前进距离分数）和赛道模式（完成度）都至关重要。

        Weight emphasis: This is the most important reward (0.4 weight in scoring).
        权重强调：这是最重要的奖励（评分中权重 0.4）。
        """
        robot = self._get_robot_asset()
        forward_vel = robot.data.root_lin_vel_b[:, 0]
        # Apply soft clipping to reward: reward increases faster at low speeds
        # 对奖励应用软裁剪：低速时奖励增长更快
        clipped_vel = torch.clamp(forward_vel, 0.0, 3.0)
        return clipped_vel
    
    def _reward_energy_efficiency(self):
        """Energy efficiency reward: penalize high energy consumption.
        能效奖励：惩罚高能耗。
        
        Energy score = exp(-average_power), so we penalize power consumption.
        能耗分数 = exp(-平均功率)，所以我们惩罚功率消耗。
        
        This directly optimizes the energy score component (0.2 weight in standard, 0.2 in track).
        这直接优化能耗分数组件（标准模式权重 0.2，赛道模式权重 0.2）。
        """
        robot = self._get_robot_asset()
        # Joint power = torque * angular velocity
        # 关节功率 = 扭矩 * 角速度
        # Try different attribute names for compatibility with different Isaac Lab versions
        # 尝试不同的属性名以兼容不同版本的 Isaac Lab
        torque = getattr(robot.data, "applied_torque", None)
        if torque is None:
            torque = getattr(robot.data, "joint_torques", None)
        if torque is None:
            torque = getattr(robot.data, "torques", None)
        if torque is None:
            # Fallback: use joint velocity squared as proxy for energy
            # 回退：使用关节速度平方作为能耗代理
            energy = torch.sum(torch.square(robot.data.joint_vel), dim=1)
        else:
            joint_power = torch.abs(torque * robot.data.joint_vel)
            # Sum over all joints
            # 对所有关节求和
            energy = torch.sum(joint_power, dim=1)
        # Apply square root scaling to reduce magnitude and encourage smooth movements
        # 应用平方根缩放以减少幅度并鼓励平滑运动
        return torch.sqrt(energy + 1e-6)
    
    def _reward_base_height_tracking(self, target_height: float = 0.38):
        """Base height tracking reward: maintain optimal base height.
        基座高度跟踪奖励：维持最优基座高度。

        This helps with posture stability score (roll/pitch deviation).
        这有助于姿态稳定性分数（roll/pitch偏差）。

        Args:
            target_height: Target base height in meters. / 目标基座高度（米）。
        """
        robot = self._get_robot_asset()
        base_height = robot.data.root_pos_w[:, 2]
        height_error = torch.abs(base_height - target_height)
        return height_error

    def _reward_posture_stability(self):
        """Posture stability reward: penalize roll/pitch deviation.
        姿态稳定性奖励：惩罚 roll/pitch 偏差。

        This directly optimizes the posture score (0.2 weight in standard, 0.4 in track).
        这直接优化姿态分数（标准模式权重 0.2，赛道模式权重 0.4）。
        """
        robot = self._get_robot_asset()
        # Convert quaternion to roll/pitch angles
        # 四元数转 roll/pitch 角度（w, x, y, z 格式）
        q = robot.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Roll (x-axis rotation)
        # Roll（X 轴旋转）
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        # Pitch（Y 轴旋转）
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1, 
                           torch.sign(sinp) * torch.pi / 2.0,
                           torch.asin(sinp))
        
        roll_pitch_error = torch.abs(roll) + torch.abs(pitch)
        return roll_pitch_error
    
    def _reward_leg_lift(self):
        """Leg lift reward: encourage lifting legs for stairs climbing.
        腿部抬升奖励：鼓励抬腿以适应楼梯攀爬。

        This helps with stair climbing by encouraging the robot to lift its feet higher.
        通过鼓励机器人抬高脚来帮助爬楼梯。
        """
        robot = self._get_robot_asset()
        # Get foot positions (assuming feet are body indices 12-15)
        # 获取脚部位置（假设脚是body索引12-15）
        foot_height = robot.data.body_pos_w[:, [12, 13, 14, 15], 2]
        # Encourage lifting feet above 0.1m
        # 鼓励抬脚超过0.1米
        lift_reward = torch.clamp(foot_height - 0.1, 0, 0.5)
        return lift_reward.sum(dim=1) * 2.0
    
    def _reward_approach_goal(self):
        """Reward for approaching the goal: distance reduction.
        接近目标奖励：距离减少。
        
        This provides dense navigation signal for track mode.
        这为赛道模式提供密集的导航信号。
        
        Requires env.goal_positions to be set by the environment.
        需要环境设置env.goal_positions。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
        goal_pos = self.env.goal_positions[:, :2]  # (N, 2)
        
        current_dist = torch.norm(goal_pos - robot_pos, dim=1)  # (N,)
        
        # Initialize previous distance on first call
        # 首次调用时初始化previous距离
        if not hasattr(self.env, "_previous_goal_dist") or self.env._previous_goal_dist is None:
            self.env._previous_goal_dist = current_dist.clone()
            return torch.zeros(self.env.num_envs, device=self.env.device)
        
        # Distance change (positive = moving away, negative = approaching)
        # 距离变化（正=远离，负=接近）
        delta_dist = current_dist - self.env._previous_goal_dist
        
        # Don't compute delta for reset environments (distance jumps)
        # 对重置的环境不计算delta（距离跳变）
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        delta_dist[reset_mask] = 0.0
        
        # Update previous distance
        # 更新previous距离
        self.env._previous_goal_dist = current_dist.clone()
        
        # Return negative distance change = approaching → positive reward
        # 返回负的距离变化 = 接近 → 正奖励
        return -delta_dist
    
    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward for reaching the goal (returns 1.0 when distance < threshold).
        到达目标奖励（距离 < 阈值时返回1.0）。
        
        This provides a large one-time reward for completing the navigation task.
        这为完成导航任务提供一次性大奖。
        
        Args:
            threshold: Distance threshold in meters. / 距离阈值（米）。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        
        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()
    
    def _reward_obstacle_evasion(
        self,
        command_name: str = "base_velocity",
        obstacle_threshold: float = -0.3,
        near_x_end: int = 10,
        body_y_start: int = 3,
        body_y_end: int = 13,
        turn_std: float = 0.5,
    ):
        """Penalize forward-blocked path when robot is not actively turning.
        惩罚前方被障碍阻挡时未主动转向。
        
        This helps with obstacle avoidance in maze terrain.
        这有助于在迷宫地形中避障。
        
        Args:
            command_name: Command term name. / 命令项名称。
            obstacle_threshold: Height threshold for obstacle detection. / 障碍物检测高度阈值。
            near_x_end: Near-field x-range end index. / 近场x范围结束索引。
            body_y_start: Body-width y-range start index. / 身体宽度y范围开始索引。
            body_y_end: Body-width y-range end index. / 身体宽度y范围结束索引。
            turn_std: Standard deviation for turn detection. / 转向检测标准差。
        """
        robot = self._get_robot_asset()
        sensor = self.env.scene.sensors["height_scanner"]
        
        # raw height: base_z - hit_z (positive=ground below, negative=obstacle above)
        # 原始高度：base_z - hit_z（正值=下方为地面，负值=上方有障碍）
        scan = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[..., 2]
        grid = scan.view(self.env.num_envs, 16, 16)
        
        # near-field body-width window
        # 近场、身体宽度的窗口
        window = grid[:, body_y_start:body_y_end, :near_x_end]
        
        # column-projection: for each y-strip, any obstacle in forward range?
        # 列投影：每个y条带在前方范围内是否存在障碍物
        col_blocked = (window < obstacle_threshold).any(dim=-1).float()
        blocked = col_blocked.mean(dim=-1)
        
        # evasion signal: turning hard -> low penalty
        # 规避信号：转弯幅度大 -> 惩罚低
        yaw_rate = torch.abs(robot.data.root_ang_vel_b[:, 2])
        not_evading = torch.exp(-yaw_rate / turn_std)
        
        # gate: only when forward command exists
        # 门控：仅在存在前进指令时生效
        cmd = self.env.command_manager.get_command(command_name)
        has_fwd_cmd = (cmd[:, 0] > 0.05).float()
        
        return blocked * not_evading * has_fwd_cmd
    
    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out AND NOT goal-reached).
        惩罚真正的失败（被终止且非超时截断且非到达目标）。
        
        This discourages behaviors that lead to falling or crashing.
        这阻止导致摔倒或碰撞的行为。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        
        # Exclude goal_reached (navigation success should not be penalized)
        # 排除goal_reached（导航成功不应被惩罚）
        if "goal_reached" in term_mgr.active_terms:
            goal_done = term_mgr.get_term("goal_reached")
            failure = failure & ~goal_done
        
        return failure.float()
    
    def _reward_navigation_time(self):
        """Per-step penalty to encourage fast navigation.
        每步固定惩罚，鼓励快速到达出口。
        
        This directly optimizes the time score component.
        这直接优化时间分数组件。
        """
        return torch.ones(self.env.num_envs, device=self.env.device)
