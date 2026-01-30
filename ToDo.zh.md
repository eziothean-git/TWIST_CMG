# 待办列表：CMG-TWIST 速度命令行走训练

本文档列出将CMG集成到TWIST训练流程，实现基于速度命令locomotion所需的**核心训练任务**。

---

## 项目概览

**目标**：训练TWIST策略网络，使G1机器人能够根据速度命令行走（CMG提供参考动作）

**训练架构**：
```
速度命令 → CMG生成器 → 参考动作(29 DOF) → TWIST环境 
                                           ↓
                                    策略学习残差修正
                                           ↓
                                    最终动作 = 参考 + 残差
```

**当前状态**：
- ✅ CMG已训练完成（29 DOF，固定frozen）
- ✅ TWIST配置已更新为29 DOF
- ✅ CMGMotionGenerator集成工具已完成
- ⏳ 训练环境集成待完成
- ❌ 训练尚未开始

---

## 任务 1：添加正式的DOF映射脚本

**状态**：✅ **已完成（无需映射）**

**解决方案**：TWIST配置已更新为29 DOF，与CMG完全对齐

### 已完成的工作（2026-01-30）

- [x] 更新TWIST为29 DOF配置
  - 文件：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
  - `num_actions = 29`（从23更新）
  - 更新观察维度、奖励权重、默认关节角度
  - 新增6个手腕DOF（左右腕各3个）

- [x] 关节顺序映射文档
  - CMG训练数据顺序 = URDF顺序 = TWIST顺序
  - 左腿(6) → 右腿(6) → 腰部(3) → 左臂(4) → 左腕(3) → 右臂(4) → 右腕(3)
  - 总计：29 DOF

- [x] 前向运动学集成
  - FK模型：`pose/pose/util_funcs/kinematics_model.py`
  - 集成工具：`CMG_Ref/utils/fk_integration.py`
  - 支持从29 DOF计算body位置和旋转

**结论**：无需DOF映射脚本，端到端29 DOF对齐完成 ✅

---

## 任务 2：添加locomotion相关的奖励

**状态**：⏳ **待实现**

**优先级**：🔴 HIGH（训练质量的关键）

### 2.1 当前奖励函数分析

当前TWIST奖励主要关注**参考动作跟踪**：
- `reward_mimic_dof_pos`：关节位置跟踪
- `reward_mimic_dof_vel`：关节速度跟踪
- `reward_mimic_body_pos`：关键body位置跟踪

**问题**：缺少locomotion特定的速度跟踪和稳定性奖励

### 2.2 需要添加的locomotion奖励

**文件位置**：
- 配置：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
- 环境：`legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`

#### a. 线速度跟踪奖励

```python
# 配置中添加
class rewards(HumanoidMimicCfg.rewards):
    class scales:
        # 现有的...
        tracking_lin_vel = 1.5      # 重要：跟踪速度命令
        tracking_ang_vel = 1.0      # 跟踪角速度命令

# 环境中实现
def _reward_tracking_lin_vel(self):
    """奖励base线速度接近命令"""
    lin_vel_error = torch.sum(torch.square(
        self.commands[:, :2] - self.base_lin_vel[:, :2]
    ), dim=1)
    return torch.exp(-lin_vel_error / 0.25)

def _reward_tracking_ang_vel(self):
    """奖励base角速度接近命令"""
    ang_vel_error = torch.square(
        self.commands[:, 2] - self.base_ang_vel[:, 2]
    )
    return torch.exp(-ang_vel_error / 0.25)
```

#### b. 基座姿态稳定性奖励

```python
class rewards(HumanoidMimicCfg.rewards):
    class scales:
        # ...
        orientation = 1.0           # 保持直立
        base_height = 0.5          # 保持合理高度

def _reward_orientation(self):
    """惩罚base倾斜"""
    # projected_gravity应该接近[0, 0, -1]
    return torch.sum(torch.square(
        self.projected_gravity[:, :2]
    ), dim=1)

def _reward_base_height(self):
    """惩罚base高度偏离目标"""
    target_height = 0.75  # G1站立高度约0.75m
    return torch.square(self.root_states[:, 2] - target_height)
```

#### c. 运动平滑性奖励

```python
class rewards(HumanoidMimicCfg.rewards):
    class scales:
        # ...
        action_rate = 0.01         # 惩罚动作变化率
        torques = 0.0001           # 惩罚大扭矩

def _reward_action_rate(self):
    """惩罚动作突变"""
    return torch.sum(torch.square(
        self.actions - self.last_actions
    ), dim=1)

def _reward_torques(self):
    """惩罚大扭矩（能量效率）"""
    return torch.sum(torch.square(self.torques), dim=1)
```

#### d. 足部接触奖励

```python
class rewards(HumanoidMimicCfg.rewards):
    class scales:
        # ...
        feet_air_time = 0.5        # 奖励合理的摆动时间
        no_fly = 0.25              # 惩罚双脚离地

def _reward_feet_air_time(self):
    """奖励合理的足部摆动时间"""
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, self.last_contacts)
    self.last_contacts = contact
    
    first_contact = (self.feet_air_time > 0.) * contact_filt
    self.feet_air_time += self.dt
    
    reward = torch.sum((self.feet_air_time - 0.5).clip(min=0.) * first_contact, dim=1)
    self.feet_air_time *= ~contact_filt
    return reward

def _reward_no_fly(self):
    """惩罚双脚同时离地"""
    contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
    single_contact = torch.sum(1. * contacts, dim=1) == 1
    return 1. * single_contact
```

### 2.3 实现步骤

- [ ] **步骤1**：在`g1_mimic_distill_config.py`中添加新奖励权重
- [ ] **步骤2**：在`g1_mimic_distill.py`中实现奖励函数
- [ ] **步骤3**：初始化所需的buffer（`last_actions`, `feet_air_time`等）
- [ ] **步骤4**：测试每个奖励函数单独工作
- [ ] **步骤5**：调整权重平衡跟踪vs locomotion

**推荐权重分配**：
```python
# 参考跟踪（保持原有）：60%
mimic_dof_pos: 1.0
mimic_dof_vel: 0.5
mimic_body_pos: 0.8

# Locomotion跟踪（新增）：30%
tracking_lin_vel: 1.5
tracking_ang_vel: 1.0
orientation: 1.0

# 平滑性和稳定性：10%
action_rate: 0.01
torques: 0.0001
feet_air_time: 0.5
```

---

## 任务 3：正式实现teacher特权观测

**状态**：⏳ **待实现**

**优先级**：🔴 HIGH（teacher-student训练架构核心）

### 3.1 当前观测结构

**当前实现**（`g1_mimic_distill.py`）：
```python
# 观测维度（priv模式）
n_proprio = 3 + 2 + 3*29 = 92
  # 3: projected_gravity
  # 2: commands (vx, vy) - 缺少yaw!
  # 87: dof_pos(29) + dof_vel(29) + target_dof_pos(29)

n_priv_mimic_obs = 20 * (8 + 29 + 27) = 1280
  # 20步未来参考 × (root_pose(8) + dof_pos(29) + key_body_pos(27))

n_priv_info = 3 + 1 + 27 + 2 + 4 + 1 + 58 = 96
  # base_lin_vel(3) + root_height(1) + key_body_pos(27)
  # + contact_mask(2) + priv_latent(4) + terrain(1)
  # + friction/restitution(58)
```

**问题**：
1. ❌ Commands只有2维(vx, vy)，缺少yaw
2. ❌ 未来参考帧来自motion library，不是CMG生成
3. ❌ 特权信息不包含地形高度图

### 3.2 需要实现的改进

#### a. 修复命令维度

```python
# g1_mimic_distill_config.py
class commands:
    num_commands = 3  # 从2改为3
    # vx_range, vy_range, yaw_range

# g1_mimic_distill.py
def _resample_commands(self, env_ids):
    self.commands[env_ids, 0] = torch_rand_float(
        self.command_ranges["lin_vel_x"][0],
        self.command_ranges["lin_vel_x"][1],
        (len(env_ids), 1), device=self.device
    ).squeeze()
    self.commands[env_ids, 1] = torch_rand_float(
        self.command_ranges["lin_vel_y"][0],
        self.command_ranges["lin_vel_y"][1],
        (len(env_ids), 1), device=self.device
    ).squeeze()
    self.commands[env_ids, 2] = torch_rand_float(  # 新增yaw
        self.command_ranges["ang_vel_yaw"][0],
        self.command_ranges["ang_vel_yaw"][1],
        (len(env_ids), 1), device=self.device
    ).squeeze()
```

#### b. 集成CMG生成未来参考帧

**当前问题**：`_reset_ref_motion()`和`_update_ref_motion()`从motion library加载

**目标**：改为从CMG实时生成

```python
# g1_mimic_distill.py

class G1MimicDistill(HumanoidMimic):
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        
        # 初始化CMG生成器
        if cfg.env.use_cmg_reference:
            from CMG_Ref.utils.cmg_motion_generator import CMGMotionGenerator
            self.cmg_generator = CMGMotionGenerator(
                model_path=cfg.cmg.model_path,
                data_path=cfg.cmg.data_path,
                num_envs=self.num_envs,
                device=self.device,
                mode='pregenerated',  # 训练初期用预生成
                preload_duration=500  # 10秒@50Hz
            )
    
    def _reset_ref_motion(self, env_ids, motion_ids=None):
        """使用CMG生成参考动作"""
        # 采样速度命令
        commands = self._sample_commands(len(env_ids))
        
        # 重置CMG生成器
        self.cmg_generator.reset(
            env_ids=env_ids,
            commands=commands
        )
        
        # 获取初始参考帧
        ref_dof_pos, ref_dof_vel = self.cmg_generator.get_motion(env_ids)
        
        # 如果需要body位置，使用FK计算
        if self.cfg.env.enable_fk:
            result = self.cmg_generator.get_motion_with_body_transforms(env_ids)
            ref_body_pos = result['body_positions']
        
        # 更新参考状态
        self._ref_dof_pos[env_ids] = ref_dof_pos
        self._ref_dof_vel[env_ids] = ref_dof_vel
        # ... 更新其他参考状态
    
    def _update_ref_motion(self):
        """每步更新参考动作"""
        ref_dof_pos, ref_dof_vel = self.cmg_generator.get_motion()
        self._ref_dof_pos[:] = ref_dof_pos
        self._ref_dof_vel[:] = ref_dof_vel
        # ...
```

#### c. 添加地形高度图（可选，后期）

```python
# 特权信息中添加地形感知
class env(HumanoidMimicCfg.env):
    terrain_heightmap_size = 20  # 20x20网格
    terrain_scan_range = 1.0     # 扫描1m范围
    
def _get_terrain_obs(self):
    """获取机器人周围的地形高度图"""
    # 基于机器人位置采样地形
    # [num_envs, heightmap_size, heightmap_size]
    pass
```

### 3.3 实现步骤

- [ ] **步骤1**：修复commands维度（2→3，添加yaw）
- [ ] **步骤2**：在config中添加CMG配置选项
  ```python
  class cmg:
      use_cmg_reference = True
      model_path = "CMG_Ref/runs/cmg_XXXXXX/cmg_final.pt"
      data_path = "CMG_Ref/dataloader/cmg_training_data.pt"
      enable_fk = False  # 如果需要body位置
  ```
- [ ] **步骤3**：在环境初始化时加载CMG生成器
- [ ] **步骤4**：修改`_reset_ref_motion()`使用CMG
- [ ] **步骤5**：修改`_update_ref_motion()`使用CMG
- [ ] **步骤6**：测试CMG生成的参考动作质量
- [ ] **步骤7**：（可选）添加地形高度图特权信息

---

## 任务 4：定义残差模型结构

**状态**：⏳ **待实现**

**优先级**：🟡 MEDIUM（初期可以直接输出动作，后期优化）

### 4.1 残差学习原理

**当前TWIST**：策略直接输出动作
```python
action = policy(observation)  # [num_envs, 29]
```

**残差学习版本**：策略输出残差修正
```python
reference_action = CMG(velocity_command)        # [num_envs, 29]
residual = policy(observation, reference_action) # [num_envs, 29]
final_action = reference_action + residual_scale * residual
```

**优势**：
- 策略只需学习小的修正量
- 更快收敛
- 更好的泛化
- 更安全（残差有界）

### 4.2 残差网络实现

#### 选项A：简单残差（推荐初期）

**不修改网络结构**，在环境中实现残差逻辑：

```python
# g1_mimic_distill.py

def compute_observations(self):
    # 获取CMG参考动作
    self.ref_actions, _ = self.cmg_generator.get_motion()
    
    # 观测包含参考动作
    self.obs_buf = torch.cat([
        self.proprio_obs,        # base状态、关节状态
        self.ref_actions,        # CMG参考动作
        self.priv_mimic_obs,     # 未来参考帧
        # ...
    ], dim=-1)

def step(self, actions):
    # actions是策略输出的残差
    residual_scale = 0.1  # 限制残差幅度
    final_actions = self.ref_actions + residual_scale * actions
    
    # Clip到合理范围
    final_actions = torch.clamp(final_actions, -1.0, 1.0)
    
    # 应用到仿真器
    self.gym.set_dof_position_target_tensor(...)
```

#### 选项B：显式残差网络（推荐后期）

创建专门的残差Actor-Critic：

```python
# rsl_rl/rsl_rl/modules/actor_critic_residual.py

class ActorCriticResidual(nn.Module):
    """
    Actor-Critic网络，学习残差修正
    """
    def __init__(self, num_obs, num_actions, num_ref_actions=None, **kwargs):
        super().__init__()
        
        num_ref_actions = num_ref_actions or num_actions
        
        # Actor输入：observation + reference_action
        actor_input_dim = num_obs + num_ref_actions
        actor_hidden = kwargs.get('actor_hidden_dims', [512, 256, 128])
        
        # Actor输出：residual
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden[0]))
        actor_layers.append(nn.ELU())
        
        for i in range(len(actor_hidden) - 1):
            actor_layers.append(nn.Linear(actor_hidden[i], actor_hidden[i+1]))
            actor_layers.append(nn.ELU())
        
        actor_layers.append(nn.Linear(actor_hidden[-1], num_actions))
        actor_layers.append(nn.Tanh())  # 残差限制在[-1, 1]
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic输入：observation（可包含特权信息）
        critic_input_dim = num_obs
        critic_hidden = kwargs.get('critic_hidden_dims', [512, 256, 128])
        
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden[0]))
        critic_layers.append(nn.ELU())
        
        for i in range(len(critic_hidden) - 1):
            critic_layers.append(nn.Linear(critic_hidden[i], critic_hidden[i+1]))
            critic_layers.append(nn.ELU())
        
        critic_layers.append(nn.Linear(critic_hidden[-1], 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs, ref_action):
        """
        Args:
            obs: [num_envs, num_obs]
            ref_action: [num_envs, num_ref_actions]
        
        Returns:
            residual: [num_envs, num_actions]
            value: [num_envs, 1]
        """
        actor_input = torch.cat([obs, ref_action], dim=-1)
        residual = self.actor(actor_input)
        value = self.critic(obs)
        return residual, value
    
    def act(self, obs, ref_action):
        """用于推理"""
        residual, _ = self.forward(obs, ref_action)
        return residual
    
    def evaluate(self, obs, ref_action):
        """用于训练（返回value）"""
        return self.forward(obs, ref_action)
```

使用残差网络：

```python
# 训练配置
class policy:
    class_name = 'ActorCriticResidual'
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    residual_scale = 0.1  # 残差缩放系数

# 环境中使用
def step(self, actions):
    # actions已经是residual
    final_actions = self.ref_actions + self.cfg.policy.residual_scale * actions
    final_actions = torch.clamp(final_actions, -1.0, 1.0)
    # ...
```

### 4.3 实现步骤

- [ ] **步骤1**：先用选项A（简单残差）开始训练
  - 在观测中添加`ref_actions`
  - 在`step()`中实现残差加法
  - 设置`residual_scale = 0.1`

- [ ] **步骤2**：训练并观察结果
  - 如果策略输出接近0，说明参考动作已经很好
  - 如果残差较大，说明需要更多修正

- [ ] **步骤3**：（可选）实现显式残差网络（选项B）
  - 创建`actor_critic_residual.py`
  - 更新训练配置使用新网络
  - 重新训练对比效果

**推荐策略**：
- 第一轮训练：用选项A，验证CMG参考质量
- 如果效果好：继续用选项A（更简单）
- 如果需要优化：实现选项B（更灵活）

---

## 任务 5：开始平地训练

**状态**：⏳ **待实现**

**优先级**：🔴 HIGH（主要任务）

### 5.1 训练前检查清单

在开始训练前，确保以下都已完成：

- [ ] ✅ DOF对齐完成（29 DOF）
- [ ] ⏳ Locomotion奖励已添加（任务2）
- [ ] ⏳ Commands维度修复为3（任务3.a）
- [ ] ⏳ CMG生成器已集成到环境（任务3.b）
- [ ] ⏳ 残差逻辑已实现（任务4，至少选项A）
- [ ] ⏳ 地形设置为平地

### 5.2 训练配置

#### 基础配置

```python
# g1_mimic_distill_config.py

class G1CMGLocoCfg(G1MimicPrivCfg):
    """CMG Locomotion训练配置"""
    
    class env(G1MimicPrivCfg.env):
        num_envs = 4096
        num_actions = 29
        episode_length_s = 10
        
        # CMG集成
        use_cmg_reference = True
        enable_fk = False  # 初期不需要FK
        
        # 命令范围
        commands_curriculum = True
        
    class cmg:
        model_path = "CMG_Ref/runs/cmg_20260130/cmg_final.pt"
        data_path = "CMG_Ref/dataloader/cmg_training_data.pt"
        mode = 'pregenerated'
        preload_duration = 500  # 10秒
    
    class commands:
        num_commands = 3
        resampling_time = 10.0  # 每10秒重新采样
        
        class ranges:
            # 初期：保守的速度范围
            lin_vel_x = [0.0, 1.0]   # 前进 0-1 m/s
            lin_vel_y = [-0.3, 0.3]  # 侧向 ±0.3 m/s
            ang_vel_yaw = [-0.5, 0.5]  # 转向 ±0.5 rad/s
    
    class terrain(G1MimicPrivCfg.terrain):
        mesh_type = 'plane'  # 平地训练
        height = [0, 0]
        horizontal_scale = 0.1
    
    class rewards(G1MimicPrivCfg.rewards):
        class scales:
            # 参考跟踪（基础）
            mimic_dof_pos = 1.0
            mimic_dof_vel = 0.5
            mimic_body_pos = 0.8
            
            # Locomotion跟踪（关键）
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            orientation = 1.0
            base_height = 0.5
            
            # 平滑性
            action_rate = 0.01
            torques = 0.0001
            
            # 终止惩罚
            termination = -10.0
    
    class normalization(G1MimicPrivCfg.normalization):
        clip_observations = 100.0
        clip_actions = 10.0
```

#### PPO训练参数

```python
class G1CMGLocoCfgPPO(G1MimicPrivCfgPPO):
    """PPO算法配置"""
    
    class algorithm(G1MimicPrivCfgPPO.algorithm):
        # PPO参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        
        # 训练步数
        num_learning_epochs = 5
        num_mini_batches = 4
        
        # 学习率
        learning_rate = 3e-4
        schedule = 'adaptive'  # 'fixed', 'linear', 'adaptive'
        
        # Gamma和Lambda
        gamma = 0.99
        lam = 0.95
        
        # 梯度裁剪
        max_grad_norm = 1.0
        
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCritic'  # 或'ActorCriticResidual'
        algorithm_class_name = 'PPO'
        
        num_steps_per_env = 24  # 采样步数
        max_iterations = 20000   # 总训练iteration
        
        # 保存和日志
        save_interval = 500
        experiment_name = 'g1_cmg_loco_flat'
        run_name = ''
        
        # 日志
        log_interval = 10
        empirical_normalization = False
```

### 5.3 启动训练

#### 创建训练脚本

```bash
# train_cmg_loco.sh

#!/bin/bash

EXPTID=${1:-"g1_cmg_loco_test"}
DEVICE=${2:-"cuda:0"}

python legged_gym/scripts/train.py \
    --task=g1_cmg_loco \
    --run_name=${EXPTID} \
    --headless \
    --device=${DEVICE} \
    --num_envs=4096 \
    --max_iterations=20000
```

#### 运行训练

```bash
cd /home/eziothean/TWIST_CMG

# 激活环境
conda activate twist

# 启动训练
bash train_cmg_loco.sh g1_cmg_flat_v1 cuda:0
```

### 5.4 监控训练

#### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=legged_gym/logs/g1_cmg_loco/ --port=6006

# 访问 http://localhost:6006
```

**关键指标**：
- `episode/rew_tracking_lin_vel`：速度跟踪质量
- `episode/rew_tracking_ang_vel`：转向跟踪质量
- `episode/rew_orientation`：姿态稳定性
- `episode/episode_length`：episode长度（越长越好）
- `train/mean_reward`：总奖励

#### 可视化测试

```python
# play_cmg_loco.sh

#!/bin/bash

EXPTID=${1:-"g1_cmg_loco_test"}
CHECKPOINT=${2:-"model_10000.pt"}

python legged_gym/scripts/play.py \
    --task=g1_cmg_loco \
    --run_name=${EXPTID} \
    --checkpoint=${CHECKPOINT}
```

```bash
# 测试训练的策略
bash play_cmg_loco.sh g1_cmg_flat_v1 model_5000.pt
```

### 5.5 训练阶段策略

#### 阶段1：基础训练（0-5k iterations）

**目标**：学习跟踪CMG参考动作

```python
# 保守的命令范围
lin_vel_x = [0.0, 0.5]  # 慢速前进
lin_vel_y = [0.0, 0.0]  # 无侧向
ang_vel_yaw = [0.0, 0.0]  # 无转向

# 高权重的参考跟踪
mimic_dof_pos = 1.5
mimic_dof_vel = 0.8
```

**期望**：
- 机器人能稳定站立
- 能跟随CMG参考前进
- Episode不早终止

#### 阶段2：速度范围扩展（5k-10k iterations）

```python
# 扩大命令范围
lin_vel_x = [0.0, 1.0]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-0.3, 0.3]

# 增加locomotion权重
tracking_lin_vel = 2.0
tracking_ang_vel = 1.5
```

**期望**：
- 能跟踪不同速度命令
- 能侧向行走和转向
- 速度跟踪误差 < 0.2 m/s

#### 阶段3：精细调优（10k-20k iterations）

```python
# 全范围命令
lin_vel_x = [-0.5, 1.5]  # 包含后退
lin_vel_y = [-0.5, 0.5]
ang_vel_yaw = [-1.0, 1.0]

# 优化平滑性
action_rate = 0.02
torques = 0.0002
```

**期望**：
- 平滑的运动
- 低能耗
- 鲁棒的命令跟踪

### 5.6 常见问题和调试

#### 问题1：机器人倒地

**可能原因**：
- CMG参考动作不适合当前命令
- 残差修正过大
- 奖励权重不平衡

**调试方法**：
```python
# 降低残差幅度
residual_scale = 0.05  # 从0.1降低

# 增加姿态稳定奖励
orientation = 2.0  # 增加权重

# 检查CMG生成质量
# 在play模式下不应用策略，只播放CMG参考
```

#### 问题2：速度跟踪不准

**可能原因**：
- Locomotion奖励权重太低
- 命令范围不合理
- 参考动作与命令不匹配

**调试方法**：
```python
# 增加速度跟踪奖励
tracking_lin_vel = 3.0
tracking_ang_vel = 2.0

# 记录实际速度vs命令
# 在环境中添加logging
print(f"Cmd: {self.commands[0]}, Actual: {self.base_lin_vel[0]}")
```

#### 问题3：训练不收敛

**可能原因**：
- 学习率太高/太低
- Batch size不合适
- 观测维度错误

**调试方法**：
```python
# 调整学习率
learning_rate = 1e-4  # 更保守

# 检查观测维度
print(f"Obs shape: {self.obs_buf.shape}")
print(f"Expected: {self.cfg.env.num_observations}")

# 检查reward scale
print(f"Mean reward: {self.rew_buf.mean()}")
```

---

## 任务 6：添加崎岖/摩擦力/扰动

**状态**：⏳ **待实现**（平地训练成功后）

**优先级**：🟢 LOW（第一版不需要）

### 6.1 实施时机

⚠️ **重要**：只在以下条件满足后再实施：

1. ✅ 平地训练完全成功
2. ✅ 策略能稳定跟踪所有速度命令
3. ✅ 运动质量满意（平滑、低能耗）
4. ✅ 仿真测试通过

### 6.2 域随机化（Domain Randomization）

#### 6.2.1 摩擦力随机化

```python
# g1_mimic_distill_config.py

class domain_rand:
    randomize_friction = True
    friction_range = [0.5, 1.25]  # 低摩擦（冰）到高摩擦（橡胶）
    
    randomize_restitution = True
    restitution_range = [0.0, 0.4]
    
    randomize_base_mass = True
    added_mass_range = [-2.0, 5.0]  # kg
    
    push_robots = True
    push_interval_s = 10
    max_push_vel_xy = 0.5  # m/s
```

```python
# 环境中实现
def _randomize_friction(self):
    """随机化地面摩擦力"""
    friction = torch_rand_float(
        self.cfg.domain_rand.friction_range[0],
        self.cfg.domain_rand.friction_range[1],
        (self.num_envs, 1), device=self.device
    )
    
    for i in range(self.num_envs):
        self.gym.set_actor_friction(
            self.envs[i],
            self.actor_handles[i],
            friction[i].item()
        )

def _push_robots(self):
    """施加随机扰动"""
    push_env_ids = (self.episode_length_buf % 
                    int(self.cfg.domain_rand.push_interval_s / self.dt) == 0)
    
    if push_env_ids.any():
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        push_vel = torch_rand_float(
            -max_vel, max_vel,
            (push_env_ids.sum(), 2), device=self.device
        )
        
        self.root_states[push_env_ids, 7:9] += push_vel
        self.gym.set_actor_root_state_tensor(...)
```

#### 6.2.2 质量和惯性随机化

```python
class domain_rand:
    randomize_base_mass = True
    added_mass_range = [-2.0, 5.0]
    
    randomize_link_mass = True
    link_mass_multiplier_range = [0.8, 1.2]
    
    randomize_com = True
    com_displacement_range = [-0.05, 0.05]  # m

def _randomize_dof_props(self):
    """随机化动力学参数"""
    for i in range(self.num_envs):
        # 随机化质量
        base_mass = self.default_base_mass + torch.rand(1) * \
                    (self.cfg.domain_rand.added_mass_range[1] - 
                     self.cfg.domain_rand.added_mass_range[0]) + \
                    self.cfg.domain_rand.added_mass_range[0]
        
        # 应用到仿真器
        props = self.gym.get_actor_rigid_body_properties(
            self.envs[i], self.actor_handles[i]
        )
        props[0].mass = base_mass.item()
        self.gym.set_actor_rigid_body_properties(
            self.envs[i], self.actor_handles[i], props
        )
```

### 6.3 地形复杂化

#### 6.3.1 地形类型

```python
class terrain:
    mesh_type = 'trimesh'  # 从'plane'改为'trimesh'
    curriculum = True
    
    # 地形参数
    terrain_types = ['flat', 'slope', 'stairs', 'rough']
    terrain_proportions = [0.3, 0.3, 0.2, 0.2]
    
    # 斜坡
    slope_threshold = 0.75  # 最大倾角
    
    # 楼梯
    stair_height_range = [0.05, 0.15]  # m
    stair_depth_range = [0.2, 0.4]    # m
    
    # 粗糙地形
    roughness = 0.05  # m
    
    # 课程学习
    curriculum_start_difficulty = 0.0
    curriculum_end_difficulty = 1.0
    curriculum_length_iters = 10000
```

#### 6.3.2 地形生成

```python
# legged_gym/legged_gym/envs/terrain_generator.py

class ProceduralTerrainGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def generate_flat_terrain(self):
        """生成平地"""
        pass
    
    def generate_slope_terrain(self, difficulty=0.5):
        """生成斜坡地形"""
        angle = difficulty * self.cfg.slope_threshold
        # 使用Perlin噪声生成自然的斜坡
        pass
    
    def generate_stairs_terrain(self, difficulty=0.5):
        """生成楼梯"""
        step_height = lerp(
            self.cfg.stair_height_range[0],
            self.cfg.stair_height_range[1],
            difficulty
        )
        # 生成规则或随机楼梯
        pass
    
    def generate_rough_terrain(self, difficulty=0.5):
        """生成粗糙地形"""
        roughness = difficulty * self.cfg.roughness
        # 使用Perlin噪声
        pass
```

### 6.4 特权信息增强

当使用复杂地形时，需要增强teacher的特权信息：

```python
class env:
    # 地形感知
    terrain_scan_points = 187  # 11x17网格
    terrain_scan_range = 1.0   # 1m范围
    
def _get_terrain_obs(self):
    """采样机器人周围的地形高度"""
    # [num_envs, scan_points]
    # 只有teacher能访问
    pass
```

### 6.5 渐进式训练策略

```python
# 阶段1: 平地 (0-20k iters)
terrain_difficulty = 0.0
domain_rand.enabled = False

# 阶段2: 轻微随机化 (20k-30k iters)
terrain_difficulty = 0.0  # 仍然平地
domain_rand.enabled = True
friction_range = [0.8, 1.2]  # 小范围

# 阶段3: 简单地形 (30k-40k iters)
terrain_difficulty = 0.3
terrain_types = ['flat', 'slope']  # 只有平地和小斜坡

# 阶段4: 混合地形 (40k+ iters)
terrain_difficulty = 0.5 → 1.0 (curriculum)
terrain_types = ['flat', 'slope', 'stairs', 'rough']
domain_rand.full_enabled = True
```

### 6.6 实施检查清单

⚠️ **只在平地训练完全成功后进行**

- [ ] 平地训练成功（>15k iterations，reward稳定）
- [ ] 策略能跟踪所有命令（误差<0.2m/s）
- [ ] 仿真中运动自然流畅
- [ ] （可选）实机测试平地行走成功

然后逐步添加：

- [ ] **第1步**：添加摩擦力随机化（简单）
- [ ] **第2步**：添加扰动推力（测试鲁棒性）
- [ ] **第3步**：添加质量随机化
- [ ] **第4步**：引入小斜坡（<10度）
- [ ] **第5步**：引入楼梯和粗糙地形
- [ ] **第6步**：实施完整地形课程

---

## 总结：完整训练流程

### 快速开始路径

```bash
# 1. 准备工作
cd /home/eziothean/TWIST_CMG
conda activate twist

# 2. 修复commands维度（任务3.a）
# 编辑 legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py
#   num_commands = 3

# 3. 添加locomotion奖励（任务2）
# 在配置中添加奖励权重
# 在环境中实现奖励函数

# 4. 集成CMG（任务3.b）
# 在配置中添加CMG选项
# 在环境初始化时加载CMGMotionGenerator
# 修改_reset_ref_motion()和_update_ref_motion()

# 5. 实现残差逻辑（任务4）
# 在step()中添加残差加法

# 6. 开始训练（任务5）
bash train_cmg_loco.sh g1_cmg_flat_v1 cuda:0

# 7. 监控训练
tensorboard --logdir=legged_gym/logs/g1_cmg_loco/ --port=6006

# 8. 测试策略
bash play_cmg_loco.sh g1_cmg_flat_v1 model_5000.pt

# 9. （可选）添加域随机化（任务6）
# 只在平地训练成功后
```

### 核心文件清单

**必须修改的文件**：
1. `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` - 配置
2. `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py` - 环境逻辑

**可能需要创建的文件**：
3. `rsl_rl/rsl_rl/modules/actor_critic_residual.py` - 残差网络（可选）
4. `legged_gym/legged_gym/envs/terrain_generator.py` - 地形生成器（后期）

### 预期时间线

- **任务1**：✅ 已完成
- **任务2-4**：1-2天（代码实现和测试）
- **任务5**：3-5天（训练迭代）
  - 基础训练：1-2天
  - 扩展训练：1-2天
  - 精调：1天
- **任务6**：1-2周（可选，后期优化）

**总计**：约1周完成平地locomotion训练

---

**文档版本**：3.0  
**最后更新**：2026-01-30  
**状态**：聚焦6个核心训练任务

