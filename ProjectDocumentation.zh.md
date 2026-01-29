# 项目文档：CMG 和 TWIST 集成

本文档提供了条件运动生成器（CMG）和 TWIST（远程操控全身模仿系统）项目的详细技术文档。

---

## 目录

1. [CMG（条件运动生成器）](#1-cmg条件运动生成器)
   - [1.1 概述](#11-概述)
   - [1.2 训练过程](#12-训练过程)
   - [1.3 模型架构](#13-模型架构)
   - [1.4 输入/输出规格](#14-输入输出规格)
   - [1.5 数据管道](#15-数据管道)

2. [TWIST（远程操控全身模仿系统）](#2-twist远程操控全身模仿系统)
   - [2.1 概述](#21-概述)
   - [2.2 训练过程](#22-训练过程)
   - [2.3 模型架构](#23-模型架构)
   - [2.4 输入/输出规格](#24-输入输出规格)
   - [2.5 两阶段训练管道](#25-两阶段训练管道)

3. [部署](#3-部署)

---

## 1. CMG（条件运动生成器）

### 1.1 概述

**目的**：基于速度命令为人形机器人生成参考运动，消除对动作捕捉数据的需求。

**主要特性**：
- 从速度输入（vx、vy、yaw_rate）进行基于命令的控制
- 采用专家混合（MoE）架构实现多样化运动生成
- 自回归生成平滑、连续的运动序列
- 运行频率为 50 FPS
- 可生成行走、跑步和转向行为

**位置**：`CMG_Ref/` 目录

### 1.2 训练过程

#### 数据要求
- 训练数据格式：`cmg_training_data.pt`
- 数据结构：
  ```python
  {
    "samples": 运动序列列表,
    "stats": {
      "motion_dim": 58,      # 29个关节位置 + 29个关节速度
      "command_dim": 3,      # [vx, vy, yaw_rate]
      "motion_mean": ndarray,
      "motion_std": ndarray,
      "command_min": ndarray,
      "command_max": ndarray
    }
  }
  ```
- 每个样本包含：
  - `motion`：[seq_len+1, 58] - 随时间变化的关节位置和速度
  - `command`：[seq_len, 3] - 速度命令（vx、vy、yaw_rate）

#### 训练配置
```python
# 关键超参数（来自 train.py）
BATCH_SIZE = 256
NUM_EPOCHS = 400
LEARNING_RATE = 3e-4
SAVE_INTERVAL = 10 个 epoch

# 模型架构
hidden_dim = 512
num_experts = 4
num_layers = 3
```

#### 训练算法
1. **计划采样策略**：
   - 教师强制概率从 1.0 开始
   - 每个 epoch 衰减 0.995
   - 最小教师概率：0.3
   - 在真实值引导和自回归学习之间取得平衡

2. **损失函数**：
   - 预测和目标运动状态之间的均方误差（MSE）
   - 对序列长度取平均

3. **学习率调度**：
   - ReduceLROnPlateau 调度器
   - 当损失停滞 10 个 epoch 时，学习率降低 0.5
   - 最小学习率：1e-6

4. **优化**：
   - 优化器：Adam
   - 梯度裁剪（通过 PyTorch 默认值）

#### 训练脚本
```bash
cd CMG_Ref
python train.py
```

**输出**：
- 检查点：`runs/cmg_YYYYMMDD_HHMMSS/`
- 最佳模型：`cmg_best.pt`
- 定期检查点：`cmg_ckpt_N.pt`（每 10 个 epoch）
- 最终模型：`cmg_final.pt`
- 用于跟踪训练指标的 TensorBoard 日志

### 1.3 模型架构

#### 整体结构
```
CMG 模型（条件运动生成器）
├── 门控网络
│   └── 从输入计算专家混合权重
├── MoE 层 1：(motion_dim + command_dim) → 512
├── MoE 层 2：512 → 512
└── MoE 层 3：512 → motion_dim (58)
```

#### 组件详情

**1. 门控网络**（`gating_network.py`）：
```python
输入：[batch, motion_dim + command_dim]（61 维）
架构：
  - Linear(61, 512) + ELU
  - Linear(512, num_experts=4) + Softmax
输出：[batch, 4] 专家权重
```

**2. MoE 层**（`moe_layer.py`）：
- 包含 4 个专家网络（每个都是线性层）
- 使用门控权重对专家输出进行加权组合
- 公式：`output = sum(weight[i] * expert[i](input) for i in range(4))`

**3. 前向传播**：
```python
1. 连接运动状态和命令：[batch, 61]
2. 通过门控网络计算专家权重：[batch, 4]
3. 对于每个 MoE 层：
   - 应用加权专家组合
   - 应用 ELU 激活（除了最后一层）
4. 输出：下一个运动状态 [batch, 58]
```

#### 激活函数
- 全程使用 ELU（指数线性单元）
- 最终输出层无激活（回归任务）

### 1.4 输入/输出规格

#### 训练输入/输出
**输入**：
- `motion`：[batch, seq_len+1, 58]
  - 关节位置（29 维）+ 关节速度（29 维）
  - 使用数据集统计信息归一化
- `command`：[batch, seq_len, 3]
  - 机器人局部坐标系中的 [vx, vy, yaw_rate]
  - 归一化到 [-1, 1] 范围

**输出**：
- 预测的下一个运动状态：[batch, 58]

#### 推理输入/输出
**输入**：
- `current_motion`：[1, 58] - 当前关节状态
- `command`：[1, 3] - 期望速度 [vx (m/s), vy (m/s), yaw (rad/s)]

**输出**：
- `next_motion`：[1, 58] - 预测的下一个关节状态

**典型命令范围**：
- 前向速度（vx）：0.0 - 3.0 m/s
- 横向速度（vy）：-0.5 - 0.5 m/s
- 偏航率：-1.0 - 1.0 rad/s

#### 运动生成过程
```python
# 来自 eval_cmg.py
1. 加载训练好的模型和归一化统计信息
2. 使用样本运动帧或零初始化
3. 对于每个时间步（50 Hz）：
   a. 归一化当前运动状态
   b. 归一化命令
   c. 通过 CMG 模型前向传播
   d. 反归一化预测的运动
   e. 使用预测作为下一个当前运动（自回归）
4. 将生成的序列保存为 NPZ 文件
```

### 1.5 数据管道

#### 数据准备
**运动数据过滤**（来自 `dataloader.py`）：
- 基于速度标准过滤运动序列：
  - 最小速度阈值
  - 最大速度阈值
  - 最大横向速度
  - 最大偏航率
  - 使用基于百分位数的过滤以允许一些异常值

**数据增强**：
- 镜像运动对称性以实现左右变化

#### 归一化
- **运动**：Z-score 归一化（均值=0，标准差=1）
- **命令**：最小-最大归一化到 [-1, 1]

---

## 2. TWIST（远程操控全身模仿系统）

### 2.1 概述

**目的**：训练一个低级控制策略，可以在物理人形机器人上跟踪来自动作捕捉或 CMG 的参考运动。

**主要特性**：
- 两阶段训练：教师（特权）→ 学生（可部署）
- 强化学习与行为克隆（RL+BC）
- 50 Hz 的实时运动跟踪
- 仿真到现实的迁移能力
- 支持 Unitree G1、H1、H1_2 机器人

**位置**：主仓库根目录，主要在 `legged_gym/` 和 `rsl_rl/` 中

### 2.2 训练过程

#### 阶段 1：教师策略训练

**目的**：通过访问特权信息（真实状态、地形信息等）学习鲁棒的运动跟踪。

**环境**：`g1_priv_mimic`

**训练命令**：
```bash
bash train_teacher.sh EXPERIMENT_NAME cuda:0
```

**关键配置**（来自 `g1_mimic_distill_config.py`）：
```python
# 环境
num_envs = 4096
num_actions = 23  # 机器人 DOF
episode_length_s = 10

# 观察空间
n_proprio = 3 + 2 + 3*num_actions = 71
  # 3: 投影重力
  # 2: 命令（vx, vy）
  # 3*23: 关节位置、速度、目标位置

n_priv_mimic_obs = 20 * (8 + 23 + 3*9) = 2040
  # 20 个时间步：
  #   8: 根姿态（位置 + 四元数）
  #   23: 目标关节位置
  #   27: 关键体位置（9 个体 × 3）

n_priv_info = 3 + 1 + 27 + 2 + 4 + 1 + 46 = 84
  # 3: 基座线速度
  # 1: 根高度
  # 27: 关键体位置
  # 2: 接触掩码（脚）
  # 4: 特权潜在变量
  # 1: 地形信息
  # 46: 摩擦/恢复

num_observations = 2195（n_proprio + n_priv_mimic_obs + n_priv_info）
```

**奖励函数组件**：
1. **跟踪奖励**（主要）：
   - 关节 DOF 跟踪
   - 关节速度跟踪
   - 根姿态跟踪（位置 + 方向）
   - 根速度跟踪
   - 关键体位置跟踪
   - 脚高度跟踪

2. **正则化奖励**：
   - 动作平滑度
   - DOF 加速度惩罚
   - 扭矩限制
   - 接触力
   - 方向惩罚

**算法**：PPO（近端策略优化）
```python
# 关键超参数
learning_rate = 2e-4
num_learning_epochs = 5
num_mini_batches = 4
clip_param = 0.2
entropy_coef = 0.01
gamma = 0.99
lam = 0.95
```

**训练时长**：
- 最大迭代次数：20,000
- 每次迭代的步数：24
- 总时间步：约 2M 每次迭代 × 20k = 40B+ 时间步
- 训练时间：RTX 4090 上 1-2 天

#### 阶段 2：学生策略训练

**目的**：将教师知识蒸馏到无特权信息的可部署策略中。

**环境**：`g1_stu_rl`

**训练命令**：
```bash
bash train_student.sh STUDENT_EXP TEACHER_EXP cuda:0
```

**与教师的主要区别**：
- 无特权信息（地形、精确状态）
- 历史编码：使用过去 10 个时间步的观察
- 较小的观察空间（仅学生特征）
- RL + 从教师进行行为克隆（DAGGER）

**DAGGER-PPO 算法**：
```python
# 行为克隆组件
dagger_coef = 0.1  # 教师模仿的权重
dagger_coef_anneal_steps = 30000
dagger_update_freq = 20  # 更新频率

# 损失函数
total_loss = ppo_loss + dagger_coef * bc_loss
bc_loss = KL_divergence(student_action_dist, teacher_action_dist)
```

**观察编码**：
- 运动编码器：1D CNN 编码参考运动序列
  - 输入：[batch, timesteps, n_motion_obs]
  - Conv1D 层提取时间特征
  - 输出：运动潜在向量

### 2.3 模型架构

#### 教师策略架构

**Actor 网络**（`actor_critic_mimic.py`）：
```
输入：观察 [batch, n_obs]
├── 运动编码器
│   ├── Linear: n_single_motion_obs → 60
│   ├── Conv1D: 60 → 40（kernel=8, stride=4）
│   ├── Conv1D: 40 → 20（kernel=5, stride=1）
│   ├── Conv1D: 20 → 20（kernel=5, stride=1）
│   └── Linear: 60 → motion_latent_dim (32)
│
├── 本体感知 + 运动潜在 → MLP
│   ├── Linear: (n_proprio + latent_dim) → 512
│   ├── ELU
│   ├── Linear: 512 → 256
│   ├── ELU
│   ├── Linear: 256 → 128
│   ├── ELU
│   └── Linear: 128 → num_actions (23)
│
输出：动作均值（标准差单独学习）
```

**Critic 网络**：
```
输入：特权观察 [batch, n_priv_obs]
├── 相同的运动编码器
├── MLP: (n_proprio + latent + n_priv_info) → 512 → 256 → 128 → 1
输出：价值估计
```

#### 学生策略架构

与教师类似，但是：
- Critic 中无特权信息
- 使用历史编码（10 个时间步）
- 在观察中添加本体感知历史

### 2.4 输入/输出规格

#### 教师策略

**Actor 输入**（n_obs = 2195）：
```python
# 1. 参考运动（n_priv_mimic_obs = 2040）
# 20 个时间步 × [root_pose(8) + dof_pos(23) + key_body_pos(27)]
reference_motion: [batch, 20, 58]

# 2. 本体感知（n_proprio = 71）
projected_gravity: [batch, 3]
commands: [batch, 2]  # vx, vy
dof_pos: [batch, 23]
dof_vel: [batch, 23]
target_dof_pos: [batch, 23]

# 3. 特权信息（n_priv_info = 84）
base_lin_vel: [batch, 3]
root_height: [batch, 1]
key_body_positions: [batch, 27]
contact_mask: [batch, 2]
priv_latent: [batch, 4]
terrain_info: [batch, 1]
friction_restitution: [batch, 46]
```

**Actor 输出**：
```python
action: [batch, 23]  # 目标关节位置（PD 控制）
```

**动作到扭矩的映射**：
- 动作是目标关节位置
- PD 控制器转换为扭矩：
  ```python
  torque = Kp * (action - current_pos) + Kd * (0 - current_vel)
  ```
- 不同关节类型（髋、膝、踝等）有不同的 Kp、Kd

#### 学生策略

**Actor 输入**（减少，无特权信息）：
```python
# 参考运动（与教师相同）
reference_motion: [batch, 20, 58]

# 带历史的本体感知
proprioception_history: [batch, 10, n_proprio_single]

# 命令
commands: [batch, 2]
```

**输出**：与教师相同

### 2.5 两阶段训练管道

#### 完整训练流程

1. **准备运动数据集**：
   ```bash
   # 下载 TWIST 运动数据集
   # 或使用 CMG 生成（见 CMG 部分）
   ```

2. **训练教师策略**：
   ```bash
   cd legged_gym/legged_gym/scripts
   python train.py --task g1_priv_mimic \
                   --proj_name g1_priv_mimic \
                   --exptid teacher_experiment \
                   --device cuda:0
   ```
   - 使用完整特权信息训练
   - 使用 PPO 算法
   - 将检查点保存到 `legged_gym/logs/g1_priv_mimic/teacher_experiment/`

3. **训练学生策略**：
   ```bash
   python train.py --task g1_stu_rl \
                   --proj_name g1_stu_rl \
                   --exptid student_experiment \
                   --teacher_exptid teacher_experiment \
                   --device cuda:0
   ```
   - 通过 DAGGER 蒸馏教师知识
   - 仅使用可观察信息
   - 保存到 `legged_gym/logs/g1_stu_rl/student_experiment/`

4. **导出为 JIT 模型**：
   ```bash
   bash to_jit.sh student_experiment
   ```
   - 创建可部署的 TorchScript 模型
   - 输出：`traced/student_experiment-XXXXX-jit.pt`

5. **部署**：
   - Sim2sim：`python server_low_level_g1_sim.py --policy_path MODEL.pt`
   - Sim2real：`python server_low_level_g1_real.py --policy_path MODEL.pt --net INTERFACE`

#### 训练监控

**WandB 集成**：
- 自动记录到 Weights & Biases
- 项目名称：`{robot}_mimic`（例如 `g1_mimic`）
- 跟踪：
  - 情节奖励
  - 单个奖励组件
  - 策略损失、价值损失
  - 学习率
  - 情节长度

**要监控的关键指标**：
- 总情节奖励（应增加）
- 关节跟踪误差（应减少）
- 根姿态跟踪误差
- 成功率（没有早期终止的情节）

---

## 3. 部署

### 3.1 系统架构

TWIST 使用双服务器架构：

1. **高级运动服务器**：
   - 向低级控制器发送参考运动
   - 可以是动作捕捉、CMG 生成或预先录制
   - 运行频率为 50 Hz
   - 使用 Redis 进行通信

2. **低级控制器**：
   - 学生策略网络
   - 从 Redis 读取参考运动
   - 输出关节命令
   - 在仿真中以 50 Hz 运行，在机器人上实时运行

### 3.2 部署模式

#### Sim2Sim 测试
```bash
# 启动低级控制器
cd deploy_real
python server_low_level_g1_sim.py --policy_path PATH/TO/JIT/MODEL

# 在另一个终端，发送运动
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION --vis
```

#### Sim2Real 部署
```bash
# 1. 连接到机器人（以太网电缆）
# 2. 设置笔记本电脑 IP：192.168.123.222
# 3. 测试连接：ping 192.168.123.164
# 4. 在机器人上进入开发模式（遥控器上的 L2+R2）

# 启动低级控制器
python server_low_level_g1_real.py \
    --policy_path PATH/TO/JIT/MODEL \
    --net YOUR_NETWORK_INTERFACE

# 在另一个终端，发送运动
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION --vis
```

### 3.3 控制流程

```
用户输入 / 远程操控 / CMG
    ↓
高级运动服务器（50 Hz）
    ↓（Redis）
参考运动缓冲区
    ↓
低级学生策略（50 Hz）
    ↓
关节位置命令
    ↓
PD 控制器
    ↓
机器人执行器
```

---

## 4. 关键集成点

### 4.1 运动格式兼容性

CMG 和 TWIST 都使用类似的运动表示：

**CMG 输出格式**：
```python
motion = [T, 58]  # 29 个关节位置 + 29 个关节速度
# 保存为带有额外元数据的 NPZ
```

**TWIST 运动格式**（来自运动数据集）：
```python
{
  'dof_positions': [T, num_dofs],
  'dof_velocities': [T, num_dofs],
  'body_positions': [T, num_bodies, 3],
  'body_rotations': [T, num_bodies, 4],
  'fps': 50
}
```

**兼容性说明**：CMG 生成 29-DOF 运动，可能需要重新映射到特定机器人 DOF（例如，G1 有 23 DOF）。

### 4.2 频率对齐

- **CMG**：50 Hz 生成
- **TWIST**：50 Hz 控制循环
- ✅ **已经对齐**

### 4.3 坐标系

- **CMG**：局部机器人坐标系（机器人坐标中的命令）
- **TWIST**：带局部跟踪的世界坐标系
- ⚠️ **可能需要转换**

---

## 5. 总结

### CMG 优势
- 从速度命令生成多样化运动
- 无需动作捕捉
- 灵活、按需生成运动
- 快速推理

### TWIST 优势
- 在物理机器人上进行鲁棒运动跟踪
- 仿真到现实的迁移
- 处理地形变化
- 实时性能

### 集成优势
通过结合 CMG 和 TWIST：
1. **CMG** 从高级命令生成参考运动
2. **TWIST** 在物理机器人上跟踪这些运动
3. 实现无需动作捕捉的基于命令的控制
4. 支持多样化的运动行为（行走、跑步、转向）

---

## 6. 参考资料

- **TWIST 论文**：[arXiv:2505.02833](https://arxiv.org/abs/2505.02833)
- **TWIST 网站**：https://humanoid-teleop.github.io/
- **CMG 工作空间**：https://github.com/PMY9527/cmg_workspace
- **仓库**：当前仓库

---

**文档版本**：1.0
**最后更新**：2026-01-29
**作者**：基于 TWIST_CMG 仓库的代码分析
