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
from torch.distributions import Normal
from typing import Any


# ==============================================================================
# 工具模块：激活函数 / 观测归一化 / 残差块
# ==============================================================================


def resolve_nn_activation(activation: str) -> nn.Module:
    """根据名称获取激活函数"""
    activation_map = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")
    return activation_map[activation]


class RunningMeanStd(nn.Module):
    """
    观测归一化的移动均值/标准差模块
    """

    def __init__(self, shape, epsilon=1e-5, momentum=0.01):
        super(RunningMeanStd, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(epsilon, dtype=torch.float32))
        self.epsilon = epsilon
        self.momentum = momentum

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count

        M2 = (self.var * self.count
              + batch_var * batch_count
              + torch.square(delta) * self.count * batch_count / total_count)
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
    带 LayerNorm 和激活函数的残差块
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation="selu", use_layernorm=True):
        super(ResidualBlock, self).__init__()
        self.use_residual = (input_dim == output_dim)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.norm1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

        self.activation = resolve_nn_activation(activation)

        if not self.use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.linear2(out)
        out = self.norm2(out)

        if self.use_residual:
            out = out + identity
        elif self.residual_proj is not None:
            out = out + self.residual_proj(identity)

        out = self.activation(out)
        return out


# ==============================================================================
# 增强版 ActorCritic 网络
# ==============================================================================


class ActorCritic(nn.Module):
    """
    增强版 Actor-Critic 网络，包含：
    1. 可配置激活函数（ELU / SELU / ReLU）
    2. 可选每层 LayerNorm（提升训练稳定性）
    3. 可选残差连接（改善深层梯度流动）
    4. RunningMeanStd 观测归一化（加速收敛）
    5. 可配置隐藏层维度
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        use_residual: bool = True,
        use_layernorm_per_layer: bool = True,
        obs_normalization: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        初始化增强版 ActorCritic

        Args:
            num_obs: Actor 观测维度
            num_critic_obs: Critic 观测维度
            num_actions: 动作维度
            actor_hidden_dims: Actor MLP 隐藏层大小
            critic_hidden_dims: Critic MLP 隐藏层大小
            activation: 激活函数名称（"elu"/"selu"/"relu"/"lrelu"/"tanh"）
            init_noise_std: 探索噪声初始标准差
            noise_std_type: "scalar" 或 "log"
            use_residual: 是否使用残差连接
            use_layernorm_per_layer: 是否每层都使用 LayerNorm
            obs_normalization: 是否使用 RunningMeanStd 观测归一化
        """
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm_per_layer = use_layernorm_per_layer
        self.obs_normalization = obs_normalization

        # 观测归一化模块
        if obs_normalization:
            self.actor_obs_norm = RunningMeanStd(num_obs)
            self.critic_obs_norm = RunningMeanStd(num_critic_obs)
        else:
            self.actor_obs_norm = nn.Identity()
            self.critic_obs_norm = nn.Identity()

        # 构建 Actor & Critic MLP
        self.actor = self._build_mlp(
            input_dim=num_obs,
            output_dim=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            is_output_layer=False,
        )
        self.critic = self._build_mlp(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            is_output_layer=True,
        )

        # 动作噪声初始化
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation, is_output_layer):
        """
        构建 MLP，支持可选残差连接和每层 LayerNorm。
        """
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if self.use_residual and i > 0 and hidden_dim == prev_dim:
                layers.append(ResidualBlock(
                    prev_dim, hidden_dim, hidden_dim,
                    activation=activation,
                    use_layernorm=self.use_layernorm_per_layer,
                ))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if self.use_layernorm_per_layer:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(resolve_nn_activation(activation))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # 初始化 / 重置
    # ------------------------------------------------------------------

    @staticmethod
    def init_weights(sequential, scales):
        """
        使用正交初始化方法初始化权重
        """
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """重置隐藏状态（前馈网络无需操作）"""
        pass

    def forward(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 属性（需要 distribution 已设置）
    # ------------------------------------------------------------------

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def update_distribution(self, obs: torch.Tensor, update_norm=False):
        """
        基于观测更新动作分布。

        Args:
            obs: [B, num_obs] Actor 观测张量
            update_norm: 是否更新 RunningMeanStd 统计量
        """
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
        从策略分布中采样动作。

        Args:
            obs: [B, num_obs] 观测张量
            update_norm: 是否更新 RunningMeanStd

        Returns:
            actions: [B, num_actions]
        """
        self.update_distribution(obs, update_norm=update_norm)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        推理时的确定性动作（均值）。

        Args:
            obs: [B, num_obs]

        Returns:
            actions: [B, num_actions]
        """
        norm_obs = self.actor_obs_norm(obs, update=False)
        return self.actor(norm_obs)

    def evaluate(self, critic_obs: torch.Tensor, update_norm=False, **kwargs) -> torch.Tensor:
        """
        使用 Critic 网络评估状态价值。

        Args:
            critic_obs: [B, num_critic_obs]
            update_norm: 是否更新 RunningMeanStd

        Returns:
            values: [B, 1]
        """
        norm_critic_obs = self.critic_obs_norm(critic_obs, update=update_norm)
        return self.critic(norm_critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作在当前分布下的对数概率。

        Args:
            actions: [B, num_actions]

        Returns:
            log_prob: [B]
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
