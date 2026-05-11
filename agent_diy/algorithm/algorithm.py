#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math
import time
from typing import Any

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import RolloutStorage


class RunningMeanStdReward:
    """
    Running mean/std normalization for rewards
    奖励的移动均值和标准差归一化
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

        # Update mean
        self.mean += delta * batch_count / total_count

        # Update variance
        M2 = self.var * self.count + batch_var * batch_count + np.square(delta) * self.count * batch_count / total_count
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


class CosineWarmupScheduler:
    """
    Cosine annealing with warmup learning rate scheduler
    带预热的余弦退火学习率调度器
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
            # Linear warmup
            # 线性预热
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            # 余弦衰减
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Algorithm:
    """
    Enhanced PPO algorithm with improved features:
    1. Larger network with SELU activation and LayerNorm on every layer
    2. Residual connections for better gradient flow
    3. Running mean/std normalization for observations
    4. Reward normalization with clipping
    5. 8 reward functions aligned with scoring system
    6. Warmup + cosine learning rate scheduling

    增强PPO算法，包含改进特性：
    1. 更大的网络，SELU激活和每层LayerNorm
    2. 残差连接改善梯度流动
    3. 移动均值和标准差观测归一化
    4. 带裁剪的奖励归一化
    5. 8个奖励函数对齐评分系统
    6. Warmup + 余弦学习率调度
    """

    def __init__(self, model, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.logger = logger
        self.monitor = monitor

        # Configuration from Config
        # 从Config获取配置
        self.stage = Config.CURRENT

        # Check if enhanced features are enabled
        # 检查是否启用增强特性
        self.use_obs_normalization = getattr(self.stage, 'obs_normalization', True)
        self.use_reward_normalization = getattr(self.stage, 'reward_normalization', True)
        self.use_residual = getattr(self.stage, 'use_residual', True)

        # PPO hyperparameters - dynamically load from stage config
        # PPO超参数 - 从stage配置动态加载
        self.clip_param = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.learning_rate = getattr(self.stage, 'lr', 3e-4)  # 从stage配置获取学习率
        self.max_grad_norm = 1.0
        self.num_mini_batches = getattr(self.stage, 'num_mini_batches', 4)
        self.num_learning_epochs = getattr(self.stage, 'num_learning_epochs', 5)
        self.desired_kl = 0.01

        # Storage (set by workflow during training)
        # 存储缓冲区（由workflow在训练时设置）
        self.storage = None

        # Training state
        # 训练状态
        self.train_step = 0
        self.total_env_steps = 0
        self.last_report_time = 0

        # Reward normalization
        # 奖励归一化
        if self.use_reward_normalization:
            self.reward_norm = RunningMeanStdReward(epsilon=1e-5, clip_range=10.0)
        else:
            self.reward_norm = None

        # Learning rate scheduler
        # 学习率调度器
        self.warmup_steps = 1000
        self.total_training_steps = 1000000
        self.min_learning_rate = 1e-6

        # Create optimizer
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )

        # Create learning rate scheduler
        # 创建学习率调度器
        self.lr_scheduler = CosineWarmupScheduler(
            self.optimizer,
            self.warmup_steps,
            self.total_training_steps,
            self.learning_rate,
            self.min_learning_rate
        )

    def act(self, obs, critic_obs):
        """
        Generate actions and compute values
        生成动作并计算值函数

        Args:
            obs: Actor observations
            obs: Actor观测
            critic_obs: Critic observations
            critic_obs: Critic观测

        Returns:
            tuple: (actions, values, log_probs, action_mean, action_std, obs, critic_obs)
            返回值：(动作, 值函数, 对数概率, 动作均值, 动作标准差, 观测, Critic观测)
        """
        with torch.no_grad():
            # Sample actions
            # 采样动作
            actions = self.model.act(obs, update_norm=self.use_obs_normalization)

            # Compute values
            # 计算价值
            values = self.model.evaluate(critic_obs, update_norm=self.use_obs_normalization)

            # Get log probabilities
            # 获取对数概率
            log_probs = self.model.get_actions_log_prob(actions)

            # Get distribution parameters
            # 获取分布参数
            action_mean = self.model.action_mean
            action_std = self.model.action_std

        return actions, values, log_probs, action_mean, action_std, obs, critic_obs

    def _compute_gae(self, rewards, values, last_value, dones=None):
        """
        Compute Generalized Advantage Estimation (GAE)
        计算广义优势估计（GAE）

        Args:
            rewards: Tensor of rewards
            rewards: 奖励张量
            values: Tensor of value estimates
            values: 价值估计张量
            last_value: Value estimate for last state
            last_value: 最后状态的价值估计
            dones: Tensor of done flags
            dones: 完成标志张量

        Returns:
            tuple: (advantages, returns)
            返回值：(优势函数, 回报)
        """
        rewards = rewards.detach()
        values = values.detach()
        if dones is None:
            dones = torch.zeros_like(rewards)

        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # Compute advantages recursively
        # 递归计算优势函数
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # Compute returns
        # 计算回报
        returns = advantages + values

        # Normalize advantages
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _compute_surrogate_loss(self, new_log_probs, old_log_probs, advantages):
        """
        Compute PPO surrogate loss with clipping
        计算带裁剪的PPO替代损失
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = ratio * advantages
        surrogate_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        return -torch.min(surrogate, surrogate_clipped).mean()

    def _compute_value_loss(self, values, returns, old_values=None):
        """
        Compute value function loss with clipping
        计算带裁剪的价值函数损失
        """
        if old_values is not None:
            # Clipped value loss
            # 裁剪后的价值损失
            value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            # Unclipped value loss
            # 未裁剪的价值损失
            value_loss = (returns - values).pow(2).mean()

        return value_loss

    def _update_learning_rate(self, kl_divergence):
        """
        Adaptively update learning rate based on KL divergence
        基于KL散度自适应更新学习率
        """
        if self.desired_kl is None:
            return

        if kl_divergence > self.desired_kl * 2.0:
            # KL too high, reduce learning rate
            # KL太高，降低学习率
            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
        elif kl_divergence < self.desired_kl / 2.0 and kl_divergence > 0.0:
            # KL too low, increase learning rate
            # KL太低，增加学习率
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def learn(self):
        """
        Train the model using stored sample data with enhanced PPO
        使用存储的样本数据通过增强PPO训练模型

        Returns:
            dict: Training statistics
            返回值：训练统计信息字典
        """
        # Note: In this enhanced PPO implementation, we use the agent's storage for data
        # (RolloutStorage), which is filled by the workflow. We'll implement the training
        # loop here based on that assumption.

        # Perform PPO update steps
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0

        # Iterate through epochs (mini_batch_generator handles internal shuffling per epoch)
        # 遍历epoch（mini_batch_generator在每轮内部处理shuffle）
        for epoch in range(self.num_learning_epochs):
            # Generate mini batches from storage (one shuffle epoch at a time)
            # 从storage生成mini batch（每次一个shuffle轮次）
            for batch in self.storage.mini_batch_generator(self.num_mini_batches, num_epochs=1):
                # Unpack the batch
                (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hidden_states_batch,
                    masks_batch
                ) = batch

                # Update actor critic networks
                policy_loss, value_loss, entropy_loss = self._update_actor_critic(
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch
                )

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy_loss += entropy_loss
                total_loss += policy_loss + value_loss - entropy_loss

        # Update learning rate scheduler
        # 更新学习率调度器
        self.lr_scheduler.step()

        # Calculate average losses
        avg_policy_loss = total_policy_loss / (self.num_learning_epochs * self.num_mini_batches)
        avg_value_loss = total_value_loss / (self.num_learning_epochs * self.num_mini_batches)
        avg_entropy_loss = total_entropy_loss / (self.num_learning_epochs * self.num_mini_batches)
        avg_total_loss = total_loss / (self.num_learning_epochs * self.num_mini_batches)

        # Record training statistics
        # 记录训练统计信息
        stats = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss,
            'learning_rate': self.lr_scheduler.get_lr(),
            'train_step': self.train_step,
            'total_env_steps': self.total_env_steps
        }

        self.train_step += 1

        # Report metrics periodically
        # 定期上报指标
        current_time = time.time()
        if current_time - self.last_report_time >= 60 or self.train_step % 100 == 0:
            if self.monitor:
                self.monitor.put_data({os.getpid(): stats})

            if self.logger and self.train_step % 100 == 0:
                self.logger.info(
                    f"[EnhancedPPO] Step {self.train_step}: "
                    f"lr={stats['learning_rate']:.6f}, "
                    f"policy_loss={stats['policy_loss']:.4f}, "
                    f"value_loss={stats['value_loss']:.4f}, "
                    f"entropy_loss={stats['entropy_loss']:.4f}"
                )

            self.last_report_time = current_time

        return stats

    def _update_actor_critic(self, obs_batch, critic_obs_batch, actions_batch, values_batch,
                             advantages_batch, returns_batch, old_actions_log_prob_batch):
        """
        Update actor and critic networks
        更新演员和评论家网络
        """
        # Update the actor network (policy)
        # 更新演员网络（策略）
        self.model.update_distribution(obs_batch)
        new_actions_log_prob = self.model.get_actions_log_prob(actions_batch)
        entropy = self.model.entropy

        # Calculate policy loss with clipping
        # 计算带裁剪的策略损失
        ratio = torch.exp(new_actions_log_prob - old_actions_log_prob_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()

        # Update the critic network (value function)
        # 更新评论家网络（价值函数）
        new_values = self.model.evaluate(critic_obs_batch)
        value_loss = (returns_batch - new_values).pow(2).mean()

        # Combine losses
        # 组合损失
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        # Backpropagate
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Step optimizer
        # 优化器更新
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.mean().item()

    def save_checkpoint(self, path):
        """
        Save model and optimizer state
        保存模型和优化器状态
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'total_env_steps': self.total_env_steps,
            'learning_rate': self.learning_rate
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load model and optimizer state
        加载模型和优化器状态
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint.get('train_step', 0)
        self.total_env_steps = checkpoint.get('total_env_steps', 0)
        self.learning_rate = checkpoint.get('learning_rate', self.stage.lr)
