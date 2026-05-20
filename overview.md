# 项目核心关键词与需求提炼

## 一、项目定位

**四足机器人自主导航运控竞赛** — 使用强化学习训练 GO2 机器狗在 Isaac Lab 仿真环境中实现自主导航与运动控制。

## 二、核心关键词（按命中率排序）

### Tier 1 — 必命中（最高频、最核心）

| 关键词 | 释义 |
|--------|------|
| **强化学习 / RL** | 整个项目的核心方法论 |
| **PPO** | 指定的基准算法，Proximal Policy Optimization |
| **四足机器人 / GO2** | 被控对象，Unitree GO2 12-DOF 机器狗 |
| **运动控制 / Locomotion** | 核心任务之一：各类地形稳定行走 |
| **自主导航 / Navigation** | 核心任务之二：路径规划与障碍避让 |
| **Actor-Critic** | 不对称网络架构，Actor/Critic 观测维度不同 |
| **Isaac Lab** | 仿真环境底层 |
| **Reward 塑形** | 得分关键，默认11项 reward 可扩展 |

### Tier 2 — 高命中率（关键技术点）

| 关键词 | 释义 |
|--------|------|
| **课程学习 / Curriculum** | 地形难度递进训练策略 |
| **域随机化 / Domain Rand** | 摩擦力、推力等物理参数随机化增强鲁棒性 |
| **不对称 Actor-Critic** | Critic 含特权信息（base_lin_vel, joint_effort），Actor 不含 |
| **height_scan** | 256维高度扫描观测，地形感知的核心输入 |
| **Track 赛道模式** | 子地形串联赛道，需导航到终点 |
| **Standard 标准模式** | 多种独立地形，以走穿为目标 |
| **TOML 配置驱动** | 环境、奖励、训练参数均由 TOML 文件配置 |
| **特征工程** | 自行构造导航特征（goal_positions, goal_yaw, nav_scanner） |

### Tier 3 — 中命中率（工程与优化细节）

| 关键词 | 释义 |
|--------|------|
| **KaiwuDRL** | 腾讯开悟分布式训练框架 |
| **PD 控制器** | 动作经 action_scale 缩放后作为 PD 目标角度 |
| **域随机化** | friction_range, push_robots, 观测噪声 |
| **姿态稳定性** | 评分维度之一，身体/关节不可接触地面 |
| **能量效率** | 评分维度之一，关节力矩/加速度惩罚 |
| **trimesh 地形** | 坡面/楼梯/迷宫三种大类 |
| **12维连续动作** | 4腿×3关节（hip/thigh/calf） |
| **2048 并行环境** | GPU 大规模并行仿真 |
| **model_save_interval** | 模型保存频率限制 |
| **预训练模型** | 支持加载已有模型继续训练 |

## 三、项目需求清单

### 3.1 功能需求

1. **运动控制策略**：在坡面(pyramid_slope)、反向坡面(pyramid_slope_inv)、楼梯(pyramid_stairs)、反向楼梯(pyramid_stairs_inv)、迷宫(maze/open_entry_maze)上稳定行走
2. **自主导航策略**：Track 模式下从起点导航至终点，利用 goal_positions / goal_yaw / nav_scanner 构造导航特征
3. **鲁棒性**：域随机化下保持稳定（摩擦0.3-1.5、随机推力、观测噪声）
4. **课程学习**：从低难度到高难度递进训练（difficulty_range [0.0, 1.0]）
5. **奖励设计**：在默认11项 reward 基础上自行扩展 reward_process.py

### 3.2 性能需求

1. **Standard 模式**：尽可能走远，同时保持低能耗和姿态稳定
2. **Track 模式**：在限定步数内完成赛道，完成率、速度、稳定、节能综合评分
3. **姿态稳定**：主体/关节不接触地面（否则 episode 失败）
4. **能量效率**：最小化关节力矩、加速度、动作变化率

### 3.3 技术约束

1. **观测维度**：Actor obs = 301维(45 proprio + 256 height_scan)，Critic obs = 316维
2. **动作维度**：12维连续动作，action_scale=0.25
3. **框架**：必须基于 KaiwuDRL 分布式框架和 TOML 配置
4. **数据格式**：torch.Tensor, GPU 传递
5. **扩展点**：observation_process, reward_process, model, conf 可自定义

### 3.4 评分维度

| 维度 | Standard | Track |
|------|----------|-------|
| 前进距离/完成率 | ✅ 核心指标 | ✅ 完成系数 |
| 通过时间 | ✅ | ✅ |
| 能量效率 | ✅ | ✅ |
| 姿态稳定性 | ✅ | ✅ |

## 四、选手可优化方向

1. **特征工程**：构造导航特征（目标点方向、距离、遮挡信息等）
2. **奖励塑形**：自定义 reward 引导策略学习
3. **网络结构**：调整 Actor/Critic MLP 层数和宽度
4. **超参调优**：学习率、clip_ratio、GAE lambda、entropy coefficient 等
5. **课程学习策略**：自定义难度递进方案
6. **域随机化配置**：调整随机化范围提升 sim-to-real 迁移能力
7. **多阶段训练**：Standard → Track 分阶段训练策略
