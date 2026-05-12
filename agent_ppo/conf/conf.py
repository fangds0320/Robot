#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os

import toml


# 有效任务类型（Isaac Lab 原生配置格式）
_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """
    训练阶段配置基类。

    继承此类并覆盖字段来定义新的训练阶段。
    """

    # --- 阶段标识 ---
    name = ""
    task_type = "standard"
    num_goal_obs = 0

    # --- 模型架构维度（Isaac Lab Unitree-Go2-Velocity 常量）---
    # 由 Isaac Lab 任务定义与网络结构决定，用户不应修改；也不应放进用户 TOML。
    num_actions = 12          # Go2 关节动作维度
    num_proprio_obs = 45      # 本体感知观测维度
    num_scan = 256            # 16x16 高度扫描维度
    num_critic_observations = 316  # proprio(45) + scan(256) + privileged(15)

    # --- 模型架构 ---
    model_class = "ActorCritic"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    use_residual = True
    use_layernorm_per_layer = True
    obs_normalization = True

    # --- 训练超参数 ---
    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    min_normalized_std = [0.05, 0.02, 0.05] * 4

    # --- 学习率调度 ---
    use_cosine_lr = True
    warmup_steps = 500
    total_training_steps = 500000
    min_learning_rate = 1e-6

    # --- 归一化 ---
    reward_normalization = True

    # --- 保存 ---
    model_save_interval = 500


class CustomConfig(StageConfig):
    # TODO: 可参考 LocomotionConfig 自行设计 track 地形导航训练阶段。
    # 需要明确：
    # 1. stage 名称；
    # 2. task_type；
    # 3. 是否采用分层训练；
    # 4. policy action 的语义和维度；
    # 5. obs 维度（是否拼接 goal 信息）；
    # 6. 训练超参。
    #
    # 新增训练阶段后，需在同目录创建对应训练配置文件。
    # 文件命名规则：train_env_conf_<task_type>_<stage.name>.toml
    # 可参考 train_env_conf_standard_locomotion.toml
    pass


class LocomotionConfig(StageConfig):
    """
    增强版 Locomotion 阶段：在混合地形上学习稳定行走。

    相比基线版本：
    - SELU 激活 + 每层 LayerNorm + 残差连接
    - RunningMeanStd 观测归一化
    - 余弦退火 + 预热学习率调度
    - 奖励归一化
    """

    name = "locomotion"
    task_type = "standard"

    # --- 增强模型架构 ---
    activation = "selu"
    use_residual = True
    use_layernorm_per_layer = True
    obs_normalization = True

    # --- 增强训练 ---
    use_cosine_lr = True
    warmup_steps = 500
    reward_normalization = True


class Config:
    """
    统一配置入口。

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。
    """

    # 通过修改 CURRENT 切换阶段
    CURRENT = LocomotionConfig

    @staticmethod
    def load_conf(logger):
        """
        根据当前阶段加载用户配置文件。

        Args:
            logger: 日志实例

        Returns:
            tuple: (usr_conf, usr_conf_file, is_eval, stage)
        """
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'. "
                f"Only {_VALID_TASKS} are supported."
            )

        # 判断是否为评估模式
        is_eval = False
        if hasattr(CONFIG, "run_mode"):
            is_eval = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval:
            usr_conf_file = f"tools/eval/conf/eval_env_conf.toml"
        else:
            usr_conf_file = f"agent_ppo/conf/train_env_conf_{task_type}_{stage.name}.toml"

        usr_conf = _load_conf(usr_conf_file, logger)

        if usr_conf is None:
            error_msg = f"usr_conf is None, please check {usr_conf_file}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, "
                     f"model: {stage.model_class}, activation: {stage.activation}")

        return usr_conf, usr_conf_file, is_eval, stage


def _deep_merge(base, override):
    """
    递归将 override 字典合并到 base 字典中（override 优先）。
    """
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_conf(conf_file, logger):
    """
    加载配置：先加载 base TOML，再用用户 TOML 覆盖合并。

    Base 文件提供模型架构维度参数，用户配置只需保留业务可调参数。
    """
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    # 根据模式选择 base 文件（eval 或 train）
    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    # 加载 base 配置（可选 — base 缺失不致命）
    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    # 加载用户配置
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    # 深度合并：base ← user（用户配置优先）
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config
