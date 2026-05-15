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


class _IdentityWithUpdate(nn.Module):
    def forward(self, x, update=False):
        return x


class RunningMeanStd(nn.Module):
    """
    运行均值/标准差归一化模块，用于观测归一化。
    """

    def __init__(self, shape, epsilon=1e-5, momentum=0.01):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))
        self.epsilon = epsilon
        self.momentum = momentum

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        M2 = self.var * self.count + batch_var * batch_count + torch.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def forward(self, x, update=False):
        if update:
            self.update(x)
        return self.normalize(x)


def resolve_nn_activation(activation: str) -> nn.Module:
    """
    根据名称获取激活函数
    """
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


class ResidualBlock(nn.Module):
    """
    带 LayerNorm 和可选投影的残差块。
    input_dim == output_dim 时直接跳跃连接，否则通过 Linear 投影对齐维度。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "elu", use_layernorm: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        act_fn = resolve_nn_activation(activation)
        self.activation = act_fn

        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(output_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if input_dim != output_dim:
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
        if self.residual_proj is not None:
            out = out + self.residual_proj(identity)
        else:
            out = out + identity
        out = self.activation(out)
        return out


class ActorCritic(nn.Module):
    """
    使用扁平张量接口的Actor-Critic网络
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = (1024, 512, 256),
        critic_hidden_dims: tuple[int] | list[int] = (1024, 512, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        obs_normalization: bool = True,
        use_residual: bool = True,
        use_layernorm_per_layer: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        初始化ActorCritic

        Args:
            num_obs: Actor观测维度
            num_critic_obs: Critic观测维度
            num_actions: 动作维度
            actor_hidden_dims: Actor MLP隐藏层大小
            critic_hidden_dims: Critic MLP隐藏层大小
            activation: 激活函数名称
            init_noise_std: 探索噪声初始标准差
            noise_std_type: 标准差类型，"scalar"或"log"
            obs_normalization: 是否启用观测归一化
            use_residual: 是否使用残差连接
            use_layernorm_per_layer: 是否每层都加 LayerNorm
        """
        super().__init__()

        self.use_residual = use_residual
        self.use_layernorm_per_layer = use_layernorm_per_layer

        # 观测归一化模块
        if obs_normalization:
            self.actor_obs_norm = RunningMeanStd(num_obs)
            self.critic_obs_norm = RunningMeanStd(num_critic_obs)
        else:
            self.actor_obs_norm = _IdentityWithUpdate()
            self.critic_obs_norm = _IdentityWithUpdate()

        # 构建策略网络 & 价值网络
        self.actor = self._build_mlp(num_obs, num_actions, actor_hidden_dims, activation, is_critic=False)
        self.critic = self._build_mlp(num_critic_obs, 1, critic_hidden_dims, activation, is_critic=True)

        # 动作噪声初始化
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        # 动作分布（由update_distribution设置）
        self.distribution = None
        # 禁用分布验证加速
        Normal.set_default_validate_args(False)

    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str, is_critic: bool = False) -> nn.Sequential:
        """
        构建 MLP，每层可配 LayerNorm，非首层可配残差连接。
        """
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if self.use_residual and i > 0:
                layers.append(ResidualBlock(prev_dim, hidden_dim, hidden_dim, activation, self.use_layernorm_per_layer))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if self.use_layernorm_per_layer:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(resolve_nn_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

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
        """
        重置已终止episode的隐藏状态
        """
        pass

    def forward(self):
        """
        前向传播（未实现，请使用act/evaluate方法）
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """
        获取动作分布的均值
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        获取动作分布的标准差
        """
        return self.distribution.stddev

    @property
    def entropy(self):
        """
        获取动作分布的熵
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs: torch.Tensor, update_norm: bool = False):
        """
        基于观测更新动作分布

        Args:
            obs: [B, num_obs] Actor观测张量
            update_norm: 是否更新运行统计量
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

    def act(self, obs: torch.Tensor, update_norm: bool = False, **kwargs) -> torch.Tensor:
        """
        从策略分布中采样动作

        Args:
            obs: [B, num_obs] 观测张量
            update_norm: 是否更新运行统计量

        Returns:
            actions: [B, num_actions] 动作张量
        """
        self.update_distribution(obs, update_norm=update_norm)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        推理时的确定性动作（均值）

        Args:
            obs: [B, num_obs] 观测张量

        Returns:
            actions: [B, num_actions] 动作张量
        """
        norm_obs = self.actor_obs_norm(obs, update=False)
        return self.actor(norm_obs)

    def evaluate(self, critic_obs: torch.Tensor, update_norm: bool = False, **kwargs) -> torch.Tensor:
        """
        使用critic网络评估状态价值

        Args:
            critic_obs: [B, num_critic_obs] Critic观测张量
            update_norm: 是否更新运行统计量

        Returns:
            values: [B, 1] 状态价值
        """
        norm_critic_obs = self.critic_obs_norm(critic_obs, update=update_norm)
        return self.critic(norm_critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作在当前分布下的对数概率

        Args:
            actions: [B, num_actions] 动作张量

        Returns:
            log_prob: [B] 对数概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1)