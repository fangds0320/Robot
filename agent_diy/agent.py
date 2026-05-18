#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import numpy as np
import os

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.feature.definition import ActData
from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "EnhancedActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        # Load configuration
        # 加载配置
        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

        self.stage = stage
        env_conf = usr_conf["env"]
        self.num_envs = env_conf["num_envs"]

        # Model architecture dims come from StageConfig (architecture constants,
        # not user-tunable business params). Do NOT read them from TOML.
        # 模型架构维度来自 StageConfig（架构常量，非业务可调参数），不从 TOML 读。
        self.num_actions = stage.num_actions
        self.num_critic_obs = stage.num_critic_observations

        num_proprio = stage.num_proprio_obs
        num_scan = stage.num_scan

        num_goal = getattr(stage, "num_goal_obs", 0)
        self.num_obs = num_proprio + num_scan + num_goal
        self.num_critic_obs = stage.num_critic_observations + num_goal

        # Initialize enhanced model
        # 初始化增强模型
        self._init_flat(num_proprio, num_scan, stage)

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        super().__init__(agent_type, device, logger, monitor)

    def _init_flat(self, num_proprio, num_scan, stage):
        """
        Initialize enhanced single-model architecture.
        初始化增强的单模型架构。
        """
        # Create enhanced model based on configuration
        # 基于配置创建增强模型
        self.model = Model().to(self.device)

        # Convert to channel-last memory format for better performance
        # 转换为通道最后内存格式以获得更好的性能
        # Linear MLP parameters do not benefit from channels_last and some runtimes
        # reject memory_format conversion for non-4D tensors.

        self.logger.info(f"Enhanced Actor-Critic model initialized")
        self.logger.info(f"Enhanced features: residual={getattr(stage, 'use_residual', True)}, "
                        f"layernorm_per_layer={getattr(stage, 'use_layernorm_per_layer', True)}, "
                        f"obs_norm={getattr(stage, 'obs_normalization', True)}, "
                        f"reward_norm={getattr(stage, 'reward_normalization', True)}")

        # Initialize optimizer
        # 初始化优化器
        params = [{"params": self.model.parameters(), "name": "enhanced_actor_critic"}]
        self.optimizer = optim.Adam(params, lr=stage.lr, eps=1e-5)

        # Initialize enhanced algorithm
        # 初始化增强算法
        self.algorithm = Algorithm(self.model, self.device, self.logger, self.monitor)

    def learn(self, list_sample_data):
        """
        Trigger learning process.
        使用样本数据触发学习过程。

        Note: Algorithm.learn() reads from its internal storage that was filled by workflow.
        注：Algorithm.learn() 直接读取 workflow 填充的内部存储。
        """
        return self.algorithm.learn()

    def predict(self, list_obs_data):
        """
        Generate predictions with enhanced actor-critic network.
        使用增强的 actor-critic 网络生成预测。
        """
        (obs, critic_obs) = list_obs_data

        with torch.no_grad():
            actions, values, log_probs, action_mean, action_std, obs_out, critic_obs_out = self.algorithm.act(obs, critic_obs)

        return [
            actions,
            values,
            log_probs,
            action_mean,
            action_std,
            obs_out.detach(),
            critic_obs_out.detach(),
        ]

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection
        利用已学习的策略进行动作选择
        """
        if isinstance(list_obs_data, (list, tuple)) and len(list_obs_data) == 2:
            obs, critic_obs = list_obs_data
        else:
            obs = list_obs_data[0] if isinstance(list_obs_data, (list, tuple)) else list_obs_data
            critic_obs = obs
        with torch.no_grad():
            actions = self.model.act_inference(obs.to(self.device))

        return [ActData(action=actions)]

    def save_model(self, path=None, id="1"):
        """
        Save model checkpoint.
        保存模型 checkpoint。
        """
        if path is None:
            path = "."
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        Load model checkpoint.
        加载模型 checkpoint。
        """
        if path is None:
            path = "."
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        if not os.path.exists(model_file_path):
            self.logger.warning(f"model file {model_file_path} not found, skip loading")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()

        has_mismatch = False
        for key in pretrained:
            if key in current_state and pretrained[key].shape != current_state[key].shape:
                has_mismatch = True
                break

        if not has_mismatch:
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self.cur_model_name = model_file_path

    def _load_model_partial(self, model, pretrained, model_file_path):
        """
        Partial checkpoint loading for cross-stage transfer.
        部分加载 checkpoint，用于跨阶段迁移。
        """
        current_state = model.state_dict()
        loaded_keys = []
        partial_keys = []
        skipped_keys = []

        for key in current_state:
            if key not in pretrained:
                skipped_keys.append(key)
                continue

            old_param = pretrained[key]
            new_param = current_state[key]

            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                loaded_keys.append(key)
            else:
                with torch.no_grad():
                    new_param.zero_()
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_param.shape, new_param.shape))
                    new_param[slices] = old_param[slices]
                partial_keys.append(f"{key} {list(old_param.shape)}→{list(new_param.shape)}")

        model.load_state_dict(current_state)

        self.logger.info(f"Partial load {model_file_path}: {len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped")
        if partial_keys and self.logger:
            self.logger.info(f"Partial keys: {', '.join(partial_keys[:5])}{'...' if len(partial_keys) > 5 else ''}")

    def action_process(self, act_data):
        """
        Process action data
        处理动作数据
        """
        action = act_data.action if hasattr(act_data, "action") else act_data
        if isinstance(action, torch.Tensor):
            return torch.clamp(action, -6.0, 6.0)
        return action

    def observation_process(self, obs_q):
        if isinstance(obs_q, torch.Tensor):
            return torch.nan_to_num(obs_q.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        return obs_q

    def reset(self):
        """
        Reset agent state
        重置智能体状态
        """
        if hasattr(self, "model"):
            self.model.reset()
