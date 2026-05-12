#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
import time
import os
import math
import numpy as np

from agent_ppo.feature.definition import RolloutStorage


# ==============================================================================
# 奖励归一化（移动均值/标准差）
# ==============================================================================


class RunningMeanStdReward:
    """
    奖励的移动均值/标准差归一化，含裁剪防止异常值。
    """

    def __init__(self, epsilon=1e-5, clip_range=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.epsilon = epsilon
        self.clip_range = clip_range

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        M2 = (self.var * self.count
              + batch_var * batch_count
              + np.square(delta) * self.count * batch_count / total_count)
        self.var = M2 / total_count
        self.count += batch_count

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            x_numpy = x.detach().cpu().numpy()
            x_normalized = (x_numpy - self.mean) / np.sqrt(self.var + self.epsilon)
            x_normalized = np.clip(x_normalized, -self.clip_range, self.clip_range)
            return torch.from_numpy(x_normalized).to(x.device)
        else:
            x_normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
            return np.clip(x_normalized, -self.clip_range, self.clip_range)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return self.normalize(x)


# ==============================================================================
# 余弦退火 + 预热学习率调度器
# ==============================================================================


class CosineWarmupScheduler:
    """
    带预热的余弦退火学习率调度器。
    """

    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # 线性预热
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # 余弦衰减
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ==============================================================================
# 增强版 PPO 算法
# ==============================================================================


class AlgorithmPPO:
    """
    增强版 PPO 算法，包含：
    1. 余弦退火 + 预热学习率调度
    2. 奖励归一化（稳定训练）
    3. 自适应 KL 学习率微调
    4. NaN/Inf 防护
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        logger: Any = None,
        monitor: Any = None,
        # PPO 超参数
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        normalize_value_loss: bool = True,
        num_mini_batches: int = 4,
        num_learning_epochs: int = 5,
        desired_kl: float = 0.01,
        schedule: str = "adaptive",
        # 增强特性
        use_cosine_lr: bool = True,
        warmup_steps: int = 500,
        total_training_steps: int = 500000,
        min_learning_rate: float = 1e-6,
        use_reward_normalization: bool = True,
        **kwargs,
    ):
        """
        初始化增强版 PPO 算法

        Args:
            model: Actor-Critic 网络
            optimizer: 模型参数优化器
            device: 计算设备
            logger: 日志记录器
            monitor: 性能监控器
            clip_param: PPO 裁剪参数 (epsilon)
            gamma: 折扣因子
            lam: GAE lambda 参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            learning_rate: 初始学习率
            max_grad_norm: 梯度裁剪最大范数
            use_clipped_value_loss: 是否裁剪价值损失
            normalize_value_loss: 是否按回报方差归一化价值损失
            num_mini_batches: 每个 epoch 的 mini-batch 数量
            num_learning_epochs: 每次更新的 epoch 数量
            desired_kl: 自适应学习率的目标 KL 散度
            schedule: 学习率调度策略（"adaptive"/"fixed"）
            use_cosine_lr: 是否使用余弦退火 LR 调度
            warmup_steps: 预热步数
            total_training_steps: 总训练步数
            min_learning_rate: 最小学习率
            use_reward_normalization: 是否使用奖励归一化
        """
        self.device = device
        self.actor_critic = model
        self.optimizer = optimizer
        self.logger = logger
        self.monitor = monitor

        # PPO 超参数
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_value_loss = normalize_value_loss
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.desired_kl = desired_kl
        self.schedule = schedule

        # 增强特性
        self.use_cosine_lr = use_cosine_lr
        self.use_reward_normalization = use_reward_normalization

        # 标准差下限
        from agent_ppo.conf.conf import Config

        self.min_std = torch.tensor(Config.CURRENT.min_normalized_std, device=device)

        # 奖励归一化
        if use_reward_normalization:
            self.reward_norm = RunningMeanStdReward(epsilon=1e-5, clip_range=10.0)
        else:
            self.reward_norm = None

        # 余弦退火调度器
        if use_cosine_lr:
            self.lr_scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_training_steps,
                base_lr=learning_rate,
                min_lr=min_learning_rate,
            )
        else:
            self.lr_scheduler = None

        # 训练状态
        self.train_step = 0
        self.last_report_monitor_time = 0

        # 存储（待初始化）
        self.storage = None

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        device: torch.device = None,
    ):
        """初始化 rollout 存储缓冲区"""
        device = device or self.device
        self.storage = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            device=device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> tuple:
        """
        生成动作并计算值函数

        Returns:
            tuple: (actions, values, log_probs, action_mean, action_std)
        """
        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            actions = self.actor_critic.act(obs, update_norm=True)
            values = self.actor_critic.evaluate(critic_obs, update_norm=True)
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            action_mean = self.actor_critic.action_mean.detach()
            action_std = self.actor_critic.action_std.detach()

        return actions, values, log_probs, action_mean, action_std

    def compute_returns(self, last_obs: torch.Tensor):
        """
        使用 GAE 方法计算回报和优势函数
        """
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_obs)

        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self) -> tuple:
        """
        使用增强版 PPO 算法训练策略

        Returns:
            tuple: (mean_surrogate_loss, mean_value_loss, mean_entropy_loss)
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for sample_idx, sample in enumerate(generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            # 奖励归一化（更新统计量 + 归一化 returns）
            if self.use_reward_normalization and self.reward_norm is not None:
                self.reward_norm.update(returns_batch)
                returns_batch = self.reward_norm.normalize(returns_batch)

            # 前向传播
            self.actor_critic.update_distribution(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            entropy_batch = self.actor_critic.entropy
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std

            # 自适应 KL 学习率微调
            self._update_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)

            # 计算损失
            surrogate_loss = self._compute_surrogate_loss(
                actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch
            )
            value_loss = self._compute_value_loss(value_batch, returns_batch, target_values_batch)

            # 组合损失
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # NaN/Inf 防护
            if not torch.isfinite(loss):
                if self.logger:
                    self.logger.warning(
                        f"[PPO] NaN/Inf loss at step {self.train_step}, "
                        f"mini-batch {sample_idx}. Skipping."
                    )
                continue

            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度 NaN 防护
            grad_finite = True
            for p in self.actor_critic.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break

            if not grad_finite:
                if self.logger:
                    self.logger.warning(
                        f"[PPO] NaN/Inf gradient at step {self.train_step}, mini-batch {sample_idx}. Skipping."
                    )
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # 钳制动作标准差
            if hasattr(self.actor_critic, "std") and self.min_std is not None:
                max_std_t = torch.full_like(self.actor_critic.std.data, 1.0e6)
                safe_std = torch.nan_to_num(
                    self.actor_critic.std.data, nan=1.0, posinf=1.0e6, neginf=0.0,
                )
                self.actor_critic.std.data.copy_(torch.clamp(safe_std, min=self.min_std, max=max_std_t))

            # 累加损失
            sl = surrogate_loss.item()
            vl = value_loss.item()
            el = entropy_batch.mean().item()
            mean_surrogate_loss += sl if not (sl != sl) else 0.0
            mean_value_loss += vl if not (vl != vl) else 0.0
            mean_entropy_loss += el if not (el != el) else 0.0

        # 更新余弦退火调度器
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates

        # 上报指标
        self._report_training_metrics(mean_surrogate_loss, mean_value_loss, mean_entropy_loss)

        self.train_step += 1
        return mean_surrogate_loss, mean_value_loss, mean_entropy_loss

    def _update_learning_rate(
        self,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ):
        """
        基于 KL 散度自适应微调学习率（在余弦调度的基础上）。
        """
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _compute_surrogate_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        """计算带裁剪的 PPO 替代损失"""
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(
        self,
        value_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
    ) -> torch.Tensor:
        """计算价值函数损失（可选裁剪和归一化）"""
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            raw_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            raw_loss = (returns_batch - value_batch).pow(2).mean()

        if self.normalize_value_loss:
            returns_var = returns_batch.detach().var() + 1e-8
            return raw_loss / returns_var

        return raw_loss

    def _report_training_metrics(
        self,
        mean_surrogate_loss: float,
        mean_value_loss: float,
        mean_entropy_loss: float,
    ):
        """向监控系统上报训练指标"""
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            current_lr = self.lr_scheduler.get_lr() if self.lr_scheduler is not None else self.learning_rate
            monitor_data = {
                "policy_loss": mean_surrogate_loss,
                "value_loss": mean_value_loss,
                "entropy_loss": mean_entropy_loss,
                "total_loss": mean_surrogate_loss + mean_value_loss + mean_entropy_loss,
                "learning_rate": current_lr,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now
