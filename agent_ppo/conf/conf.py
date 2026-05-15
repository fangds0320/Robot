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


# Valid task types (Isaac Lab native config format)
# 有效任务类型（Isaac Lab 原生配置格式）
_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """
    Base class for training stage configuration.
    训练阶段配置基类。

    Subclass this and override fields to define a new training stage.
    继承此类并覆盖字段来定义新的训练阶段。
    """

    # --- Stage identity
    # 阶段标识 ---
    name = ""
    task_type = "standard"

    # --- Model architecture dimensions (Isaac Lab Unitree-Go2-Velocity constants)
    # These are fixed by the Isaac Lab task definition and the network structure;
    # users are not expected to change them. Do NOT move them into user TOML.
    # 模型架构维度（Isaac Lab Unitree-Go2-Velocity 常量）
    # 由 Isaac Lab 任务定义与网络结构决定，用户不应修改；也不应放进用户 TOML。
    num_actions = 12  # Go2 joint action dim / Go2 关节动作维度
    num_proprio_obs = 45  # proprioceptive obs dim / 本体感知观测维度
    num_scan = 256  # 16x16 height-scan dim / 16x16 高度扫描维度
    num_critic_observations = 316  # proprio(45) + scan(256) + privileged(15)

    # --- Model architecture (optimized for navigation)
    # 模型架构（针对导航优化）---
    model_class = "ActorCritic"
    # Increased network capacity for better feature extraction
    # 增加网络容量以获得更好的特征提取能力
    actor_hidden_dims = [1024, 512, 256, 128]
    critic_hidden_dims = [1024, 512, 256, 128]
    activation = "elu"

    # --- Training hyperparameters (v3 optimized for stability + energy) ---
    # 训练超参数（v3 针对稳定性 + 能耗优化）---
    lr = 8e-5                 # Lower LR for stable gradient updates
    num_learning_epochs = 5   # Fewer epochs to reduce overfitting risk
    num_mini_batches = 4      # Fewer mini-batches for better gradient estimates
    num_steps_per_env = 64    # Longer trajectory for better GAE estimation
    min_normalized_std = [0.05, 0.025, 0.05] * 4  # Tighter std cap to limit entropy
    
    # PPO-specific hyperparameters
    # PPO 特定超参数
    entropy_coef = 0.005      # Low entropy coeff to prevent policy divergence
    lam = 0.95               # GAE lambda
    clip_param = 0.2         # PPO clip parameter

    # --- Additional PPO parameters (defaults, overridable per-stage) ---
    # --- 额外 PPO 参数（默认值，各阶段可覆盖）---
    gamma = 0.998             # High discount factor for long-term energy optimization
    max_grad_norm = 0.5       # Tight gradient clipping
    desired_kl = 0.005        # Tight KL target for adaptive LR

    # --- Saving
    # 保存 ---
    model_save_interval = 500


class CustomConfig(StageConfig):
    # TODO: you can refer to LocomotionConfig to design your own track-terrain
    # navigation training stage. The following items need to be specified:
    # 1. stage name;
    # 2. task_type;
    # 3. whether to use hierarchical training;
    # 4. semantics and dimension of the policy action;
    # 5. obs dimension (whether to concatenate goal information);
    # 6. training hyperparameters.
    #
    # After adding a new training stage, a corresponding training config file
    # must be created in the same directory.
    # Filename convention: train_env_conf_<task_type>_<stage.name>.toml
    # Refer to train_env_conf_standard_locomotion.toml as an example.
    #
    # TODO：可参考 LocomotionConfig 自行设计 track 地形导航训练阶段。
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
    # 可参考 train_env_conf_standard_locomotion.toml。
    pass


class LocomotionConfig(StageConfig):
    """
    Stage: locomotion — learn stable walking on mixed terrain.
    阶段：locomotion —— 在混合地形上学习稳定行走。

    Optimized for training convergence (v2):
    - Lower entropy_coef to prevent policy divergence
    - Lower LR + tighter clip for stable gradient updates
    - Fewer epochs per batch to avoid overfitting
    - Shorter rollout for fresher GAE estimates
    - Reward scaling to keep gradient magnitudes stable
    针对训练收敛优化（v2）：
    - 降低熵系数防止策略发散
    - 降低学习率 + 收紧 clip 实现稳定梯度更新
    - 减少每 batch 的 epoch 数避免过拟合
    - 缩短 rollout 获得更新鲜的 GAE 估计
    - 奖励缩放保持梯度幅值稳定
    """

    name = "locomotion"
    task_type = "standard"

    # --- Optimized PPO hyperparameters for stable convergence (v3) ---
    # --- 优化后的 PPO 超参数，实现稳定收敛（v3）---
    # Key fixes for 1-hour training plateau:
    # - entropy_loss still rising 16→20 → further cut entropy_coef (0.015→0.005)
    #   to stop policy std divergence and restore deterministic behavior
    # - 熵损失仍从16升到20 → 进一步降低 entropy_coef (0.015→0.005)
    #   以阻止策略 std 发散，恢复确定性行为
    # - total_loss climbing → reduce lr for more stable gradient steps
    # - total_loss 攀升 → 降低学习率以获得更稳定的梯度更新
    lr = 8e-5               # 进一步降低学习率，防止熵驱动的梯度更新过大（之前1e-4）
    entropy_coef = 0.005    # 大幅削减熵系数，阻止策略分布持续发散（之前0.015仍导致熵16→20）
    lam = 0.95              # 标准 GAE lambda
    clip_param = 0.2        # 收紧 PPO clip 范围，限制策略更新幅度
    num_learning_epochs = 5  # 减少 epoch 数，降低每 batch 过拟合风险（之前6）
    num_mini_batches = 4     # 减少 mini-batch 数，增大每 batch 样本量提高梯度估计精度
    num_steps_per_env = 64  # 缩短 rollout 长度，让 GAE 估计更及时
    gamma = 0.998            # 更高折扣因子，关注长期回报，让能耗优化有长期视野（之前0.995）
    max_grad_norm = 0.5      # 更严格的梯度裁剪，防止大 reward 导致梯度爆炸
    desired_kl = 0.005       # 收紧目标 KL 散度，自适应学习率更敏感（之前0.008）
    min_normalized_std = [0.05, 0.025, 0.05] * 4  # 收紧 min_std 限制熵增长上限（之前[0.08,0.04,0.08]）


class TrackLocomotionConfig(StageConfig):
    """
    Stage: track locomotion — learn navigation on track terrain.
    阶段：赛道运动 —— 在赛道地形上学习导航。

    Optimized for track competition (v4 — navigation-first design):
    Track scoring: Total = completion_coeff × (0.4×Time + 0.4×Posture + 0.2×Energy)
    NO forward_distance score — completion is a multiplier on everything!

    Key design decisions:
    - Higher num_steps_per_env (128) for 150s episodes with better GAE
    - Moderate entropy (0.008) — diverse terrain needs exploration, but
      maze navigation demands deterministic turning
    - gamma=0.998 — reach_goal bonus must propagate back ~1600 steps
      with ~40% retained value for effective credit assignment
    - lam=0.97 — higher bias-variance tradeoff for long trajectories
    - lr=1e-4 — slightly higher than standard, track navigation has
      steeper reward landscape (sparse reach_goal)
    - Larger network [1024,1024,512,256] — must learn obstacle-aware
      locomotion + maze navigation from height-scan alone
    - Slightly looser min_std for terrain adaptation while keeping
      entropy under control

    针对赛道竞赛优化（v4 — 导航优先设计）：
    赛道评分：总分 = 完成系数 × (0.4×时间 + 0.4×姿态 + 0.2×能耗)
    没有前进距离分数 — 完成率是总分的乘数！

    关键设计决策：
    - 提高 num_steps_per_env (128) 适配 150s episode，改善 GAE 估计
    - 中等 entropy (0.008) — 多样地形需要探索，但迷宫导航要求确定性转弯
    - gamma=0.998 — reach_goal 奖励需反向传播 ~1600 步且保留 ~40% 价值
    - lam=0.97 — 长轨迹下更高的 bias-variance 权衡
    - lr=1e-4 — 略高于标准模式，赛道导航奖励 landscape 更陡峭（稀疏 reach_goal）
    - 更大网络 [1024,1024,512,256] — 必须仅从 height-scan 学会感知障碍的运动 + 迷宫导航
    - 稍宽松 min_std 以适应地形多样性，同时控制熵
    """

    name = "track_locomotion"
    task_type = "track"

    # Expanded network for complex navigation (obstacle-aware locomotion + maze)
    # 扩展网络用于复杂导航（感障运动 + 迷宫）
    actor_hidden_dims = [1024, 1024, 512, 256]
    critic_hidden_dims = [1024, 1024, 512, 256]

    # Optimized PPO hyperparameters for track navigation (v4)
    # 优化的 PPO 超参数用于赛道导航（v4）
    lr = 1e-4               # 略高学习率用于赛道陡峭奖励 landscape（之前 8e-5）
    entropy_coef = 0.008    # 中等探索：坡/楼梯/迷宫多样地形需探索，但不能发散（之前 0.004）
    lam = 0.97              # 更高 lambda 适配长轨迹 GAE 估计（之前 0.97 不变）
    clip_param = 0.2        # 标准 PPO clip
    gamma = 0.998            # 高折扣因子让 reach_goal 奖励反向传播整条赛道
    max_grad_norm = 0.5      # 严格梯度裁剪

    num_learning_epochs = 6  # 适中 epoch 数：262K batch ÷ 4 mini-batches × 6 = 24 updates（之前 7）
    num_mini_batches = 4     # 4 mini-batches：每 batch 65K 样本，梯度估计精度高
    num_steps_per_env = 128  # 翻倍：150s episode 需要更长 rollout 改善 GAE（之前 64）
    desired_kl = 0.008       # 稍宽松 KL 目标：赛道奖励陡峭，允许略大策略更新（之前继承 0.005）
    min_normalized_std = [0.06, 0.03, 0.06] * 4  # 稍宽松适应多样地形（之前 [0.04,0.02,0.04]）


class Config:
    """
    Unified config entry point.
    统一配置入口。

    Set ``Config.CURRENT`` to a StageConfig subclass, then read
    hyperparameters via ``Config.CURRENT.lr``, ``Config.CURRENT.num_mini_batches``, etc.

    设置 ``Config.CURRENT`` 为某个 StageConfig 子类，然后通过
    ``Config.CURRENT.lr``、``Config.CURRENT.num_mini_batches`` 等读取超参数。
    """

    # Switch stage by changing CURRENT
    # 通过修改 CURRENT 切换阶段
    CURRENT = LocomotionConfig

    @staticmethod
    def load_conf(logger):
        """
        Load user configuration file based on current stage.
        根据当前阶段加载用户配置文件。

        Args:
            logger: logger instance | 日志实例

        Returns:
            tuple: (usr_conf, usr_conf_file, is_eval, stage)
        """
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'. " f"Only {_VALID_TASKS} are supported."
            )

        # Determine if it's evaluation mode
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

        logger.info(f"Stage: {stage.name}, task_type: {task_type}, model: {stage.model_class}")

        return usr_conf, usr_conf_file, is_eval, stage


def _deep_merge(base, override):
    """
    Recursively merge override dict into base dict.
    递归将 override 字典合并到 base 字典中（override 优先）。

    Args:
        base: Base config dictionary | 基础配置字典
        override: Override config dictionary | 覆盖配置字典

    Returns:
        dict: Merged config dictionary
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
    Load config: first load base TOML, then deep-merge user TOML on top.
    加载配置：先加载 base TOML，再用用户 TOML 覆盖合并。

    Base files provide model architecture dimensions (num_actions, num_proprio_obs, etc.)
    so user configs only need business-tunable parameters.
    Base 文件提供模型架构维度参数，用户配置只需保留业务可调参数。

    Args:
        conf_file: Path to the user TOML config file | 用户配置文件路径
        logger: Logger instance | 日志实例

    Returns:
        dict: Merged config dictionary, or None on failure
    """
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    # Determine base file by mode (eval or train)
    # 根据模式选择 base 文件（eval 或 train）
    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    # Load base config (optional — missing base is not fatal)
    # 加载 base 配置（可选 — base 缺失不致命）
    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    # Load user config
    # 加载用户配置
    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    # Deep merge: base ← user (user wins)
    # 深度合并：base ← user（用户配置优先）
    if base_config:
        config = _deep_merge(base_config, user_config)
    else:
        config = user_config

    return config
