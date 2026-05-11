#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from __future__ import annotations

from typing import Any, Union

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from agent_diy.conf.conf import Config


class RunningMeanStd(nn.Module):
    """
    Running mean/std normalization module for observation normalization
    观测归一化的运行均值/标准差模块
    """

    def __init__(self, shape, epsilon=1e-5, momentum=0.01):
        super(RunningMeanStd, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(epsilon))
        self.epsilon = epsilon
        self.momentum = momentum

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        self.mean += delta * batch_count / total_count

        # Update variance
        M2 = self.var * self.count + batch_var * batch_count + torch.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return self.normalize(x)


class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm and activation
    带LayerNorm和激活函数的残差块
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation="selu", use_layernorm=True):
        super(ResidualBlock, self).__init__()
        self.use_residual = (input_dim == output_dim)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(output_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.activation = self._get_activation(activation)

        # Projection layer for residual connection if dimensions don't match
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def _get_activation(self, activation):
        activation_map = {
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
        }
        return activation_map.get(activation, nn.SELU())

    def forward(self, x):
        identity = x

        # First layer
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)

        # Second layer
        out = self.linear2(out)
        out = self.norm2(out)

        # Residual connection
        if self.use_residual:
            out = out + identity
        elif self.residual_proj is not None:
            out = out + self.residual_proj(identity)

        out = self.activation(out)
        return out


class EnhancedActorCritic(nn.Module):
    """
    Enhanced Actor-Critic network with:
    1. Larger MLP: [1024, 512, 256] vs [512, 256, 128]
    2. SELU activation instead of ELU
    3. LayerNorm on every layer (not just last critic layer)
    4. Residual connections
    5. RunningMeanStd observation normalization
    6. 8 reward functions (implemented elsewhere)

    增强版Actor-Critic网络，包含：
    1. 更大的MLP：[1024, 512, 256] vs [512, 256, 128]
    2. SELU激活函数代替ELU
    3. 每层都有LayerNorm（不仅仅是critic最后一层）
    4. 残差连接
    5. RunningMeanStd观测归一化
    6. 8个奖励函数（在其他地方实现）
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = (1024, 512, 256),
        critic_hidden_dims: tuple[int] | list[int] = (1024, 512, 256),
        activation: str = "selu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        use_residual: bool = True,
        use_layernorm_per_layer: bool = True,
        obs_normalization: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Initialize EnhancedActorCritic
        初始化EnhancedActorCritic

        Args:
            num_obs: Dimension of actor observation
            num_obs: Actor观测维度
            num_critic_obs: Dimension of critic observation
            num_critic_obs: Critic观测维度
            num_actions: Number of action dimensions
            num_actions: 动作维度
            actor_hidden_dims: Hidden layer sizes for actor MLP
            actor_hidden_dims: Actor MLP隐藏层大小
            critic_hidden_dims: Hidden layer sizes for critic MLP
            critic_hidden_dims: Critic MLP隐藏层大小
            activation: Activation function name
            activation: 激活函数名称
            init_noise_std: Initial noise std for exploration
            init_noise_std: 探索噪声初始标准差
            noise_std_type: "scalar" or "log"
            noise_std_type: 标准差类型，"scalar"或"log"
            use_residual: Whether to use residual connections
            use_residual: 是否使用残差连接
            use_layernorm_per_layer: Whether to use LayerNorm on every layer
            use_layernorm_per_layer: 是否每层都使用LayerNorm
            obs_normalization: Whether to use observation normalization
            obs_normalization: 是否使用观测归一化
        """
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm_per_layer = use_layernorm_per_layer
        self.obs_normalization = obs_normalization

        # Observation normalization modules
        # 观测归一化模块
        if obs_normalization:
            self.actor_obs_norm = RunningMeanStd(num_obs)
            self.critic_obs_norm = RunningMeanStd(num_critic_obs)
        else:
            self.actor_obs_norm = nn.Identity()
            self.critic_obs_norm = nn.Identity()

        # Build actor MLP with residual blocks
        # 构建带残差块的actor MLP
        self.actor = self._build_mlp(
            input_dim=num_obs,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            is_critic=False
        )

        # Build critic MLP with residual blocks
        # 构建带残差块的critic MLP
        self.critic = self._build_mlp(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            is_critic=True
        )

        # Action noise initialization
        # 动作噪声初始化
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (set by update_distribution)
        # 动作分布（由update_distribution设置）
        self.distribution = None
        # Disable args validation for speedup
        # 禁用分布验证加速
        from torch.distributions import Normal
        Normal.set_default_validate_args(False)

    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation="selu", is_critic=False):
        """
        Build MLP with optional residual connections and LayerNorm
        构建带可选残差连接和LayerNorm的MLP

        Args:
            input_dim: Input dimension
            input_dim: 输入维度
            output_dim: Output dimension
            output_dim: 输出维度
            hidden_dims: List of hidden layer dimensions
            hidden_dims: 隐藏层维度列表
            activation: Activation function name
            activation: 激活函数名称
            is_critic: Whether building critic network
            is_critic: 是否构建critic网络
        """
        layers = []

        # Input layer
        # 输入层
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if self.use_residual and i > 0 and hidden_dim == prev_dim:
                # Use residual block when dimensions match
                # 当维度匹配时使用残差块
                layers.append(ResidualBlock(
                    prev_dim, hidden_dim, hidden_dim,
                    activation=activation,
                    use_layernorm=self.use_layernorm_per_layer
                ))
            else:
                # Regular linear layer
                # 常规线性层
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if self.use_layernorm_per_layer:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(self._get_activation(activation))

            prev_dim = hidden_dim

        # Output layer
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        # Only add LayerNorm to critic output if specified
        # 仅当指定时才为critic输出添加LayerNorm
        if is_critic and self.use_layernorm_per_layer:
            layers.append(nn.LayerNorm(output_dim))

        return nn.Sequential(*layers)

    def _get_activation(self, activation):
        """
        Get activation function by name
        根据名称获取激活函数
        """
        activation_map = {
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
        }
        return activation_map.get(activation, nn.SELU())

    @staticmethod
    def init_weights(sequential, scales):
        """
        Initialize weights using orthogonal initialization
        使用正交初始化方法初始化权重
        """
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """
        Reset hidden states for terminated episodes
        重置已终止episode的隐藏状态
        """
        pass

    def forward(self):
        """
        Forward pass (not implemented, use act/evaluate instead)
        前向传播（未实现，请使用act/evaluate方法）
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """
        Get mean of action distribution
        获取动作分布的均值
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        Get standard deviation of action distribution
        获取动作分布的标准差
        """
        return self.distribution.stddev

    @property
    def entropy(self):
        """
        Get entropy of action distribution
        获取动作分布的熵
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs: torch.Tensor, update_norm=False):
        """
        Update action distribution based on observations
        基于观测更新动作分布

        Args:
            obs: [B, num_obs] flat actor observation tensor
            obs: [B, num_obs] Actor观测张量
            update_norm: Whether to update running statistics
            update_norm: 是否更新运行统计量
        """
        # Normalize observations
        # 归一化观测
        norm_obs = self.actor_obs_norm(obs, update=update_norm)

        mean = self.actor(norm_obs)
        if self.noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, update_norm=False, **kwargs) -> torch.Tensor:
        """
        Sample actions from policy distribution
        从策略分布中采样动作

        Args:
            obs: [B, num_obs]
            obs: [B, num_obs] 观测张量
            update_norm: Whether to update running statistics
            update_norm: 是否更新运行统计量

        Returns:
            actions: [B, num_actions]
            返回值：[B, num_actions] 动作张量
        """
        self.update_distribution(obs, update_norm=update_norm)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action (mean) for inference
        推理时的确定性动作（均值）

        Args:
            obs: [B, num_obs]
            obs: [B, num_obs] 观测张量

        Returns:
            actions: [B, num_actions]
            返回值：[B, num_actions] 动作张量
        """
        norm_obs = self.actor_obs_norm(obs, update=False)
        return self.actor(norm_obs)

    def evaluate(self, critic_obs: torch.Tensor, update_norm=False, **kwargs) -> torch.Tensor:
        """
        Evaluate state value using critic network
        使用critic网络评估状态价值

        Args:
            critic_obs: [B, num_critic_obs]
            critic_obs: [B, num_critic_obs] Critic观测张量
            update_norm: Whether to update running statistics
            update_norm: 是否更新运行统计量

        Returns:
            values: [B, 1]
            返回值：[B, 1] 状态价值
        """
        norm_critic_obs = self.critic_obs_norm(critic_obs, update=update_norm)
        return self.critic(norm_critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions under current distribution
        计算动作在当前分布下的对数概率

        Args:
            actions: [B, num_actions]
            actions: [B, num_actions] 动作张量

        Returns:
            log_prob: [B]
            返回值：[B] 对数概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1)


# Factory function to create enhanced model
# 创建增强模型的工厂函数
def create_enhanced_model():
    """Create an EnhancedActorCritic model based on current configuration
    基于当前配置创建EnhancedActorCritic模型
    """
    from agent_diy.conf.conf import Config

    stage = Config.CURRENT

    # Check if enhanced features are enabled
    # 检查是否启用增强特性
    use_residual = getattr(stage, 'use_residual', True)
    use_layernorm_per_layer = getattr(stage, 'use_layernorm_per_layer', True)
    obs_normalization = getattr(stage, 'obs_normalization', True)
    reward_normalization = getattr(stage, 'reward_normalization', True)

    # Determine number of observations based on task type
    # 根据任务类型确定观测维度
    if stage.task_type == "standard":
        num_obs = stage.num_proprio_obs + stage.num_scan
        num_critic_obs = stage.num_critic_observations
    elif stage.task_type == "track":
        # Track mode adds goal information
        # Track模式添加目标信息
        num_obs = stage.num_proprio_obs + stage.num_scan + 4  # goal_pos(3) + goal_yaw(1)
        num_critic_obs = stage.num_critic_observations + 4
    else:
        raise ValueError(f"Unknown task_type: {stage.task_type}")

    model = EnhancedActorCritic(
        num_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=stage.num_actions,
        actor_hidden_dims=stage.actor_hidden_dims,
        critic_hidden_dims=stage.critic_hidden_dims,
        activation=stage.activation,
        use_residual=use_residual,
        use_layernorm_per_layer=use_layernorm_per_layer,
        obs_normalization=obs_normalization,
    )

    return model


# Factory function to create appropriate model based on config
# 基于配置创建适当模型的工厂函数
def create_model():
    """Create model based on current configuration
    基于当前配置创建模型
    """
    from agent_diy.conf.conf import Config

    stage = Config.CURRENT

    # Determine model class based on configuration
    # 根据配置确定模型类别
    if getattr(stage, 'model_class', 'ActorCritic').lower() == 'enhancedactorcritic':
        return create_enhanced_model()
    else:
        # Fall back to standard model (would need to import baseline ActorCritic)
        # 回退到标准模型（需要导入基线ActorCritic）
        # For now, return enhanced model as default
        # 目前，返回增强模型作为默认值
        return create_enhanced_model()


# Alias for backward compatibility
ActorCritic = EnhancedActorCritic


# Main Model class
# 主模型类
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # User-defined network
        # 用户自定义网络
        self.network = create_model()

    def forward(self):
        """Forward pass (not implemented, use act/evaluate instead)
        前向传播（未实现，请使用act/evaluate方法）
        """
        raise NotImplementedError

    def act(self, obs: torch.Tensor, update_norm=False, **kwargs):
        """Sample actions from policy
        从策略中采样动作
        """
        return self.network.act(obs, update_norm=update_norm, **kwargs)

    def act_inference(self, obs: torch.Tensor):
        """Deterministic action for inference
        推理时的确定性动作
        """
        return self.network.act_inference(obs)

    def evaluate(self, critic_obs: torch.Tensor, update_norm=False, **kwargs):
        """Evaluate state value
        评估状态价值
        """
        return self.network.evaluate(critic_obs, update_norm=update_norm, **kwargs)

    def update_distribution(self, obs: torch.Tensor, update_norm=False):
        """Update action distribution
        更新动作分布
        """
        self.network.update_distribution(obs, update_norm=update_norm)

    def get_actions_log_prob(self, actions: torch.Tensor):
        """Compute log probability of actions
        计算动作的对数概率
        """
        return self.network.get_actions_log_prob(actions)

    @property
    def action_mean(self):
        """Get mean of action distribution
        获取动作分布的均值
        """
        return self.network.action_mean

    @property
    def action_std(self):
        """Get standard deviation of action distribution
        获取动作分布的标准差
        """
        return self.network.action_std

    @property
    def entropy(self):
        """Get entropy of action distribution
        获取动作分布的熵
        """
        return self.network.entropy

    def reset(self, dones=None):
        """Reset network state
        重置网络状态
        """
        self.network.reset(dones)