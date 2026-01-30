# DOF配置分析报告

## 结论

**实际部署的机器人确实是29 DOF，而TWIST项目训练时使用的是23 DOF。这是TWIST项目裁切了DOF，而不是URDF本来就不一样。**

---

## 详细分析

### 1. URDF文件对比

#### 29 DOF URDF (`g1_29dof_rev_1_0.urdf`)
包含29个revolute关节：
- **腿部**: 12 DOF (每条腿6个)
  - left/right_hip_pitch_joint
  - left/right_hip_roll_joint
  - left/right_hip_yaw_joint
  - left/right_knee_joint
  - left/right_ankle_pitch_joint
  - left/right_ankle_roll_joint

- **腰部**: 3 DOF
  - waist_yaw_joint
  - waist_roll_joint
  - waist_pitch_joint

- **手臂**: 8 DOF (每只手臂4个)
  - left/right_shoulder_pitch_joint
  - left/right_shoulder_roll_joint
  - left/right_shoulder_yaw_joint
  - left/right_elbow_joint

- **手腕**: 6 DOF (每只手腕3个) **← 这是关键差异**
  - left/right_wrist_roll_joint
  - left/right_wrist_pitch_joint
  - left/right_wrist_yaw_joint

**总计: 12 + 3 + 8 + 6 = 29 DOF**

---

#### 23 DOF URDF (`g1_custom_collision_with_fixed_hand.urdf`)
包含23个revolute关节：
- **腿部**: 12 DOF (每条腿6个) - 同上
- **腰部**: 3 DOF - 同上  
- **手臂**: 8 DOF (每只手臂4个) - 同上
- **手腕**: 0 DOF (被fixed joints替代)

**总计: 12 + 3 + 8 = 23 DOF**

**差异**: 29 DOF版本比23 DOF版本**多了6个手腕关节**（每只手腕3个：roll, pitch, yaw）

---

### 2. TWIST项目配置

#### 训练环境配置
文件: `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

```python
class env:
    num_actions = 23  # ← 明确指定23个DOF
    
class asset:
    file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_with_fixed_hand.urdf'
    # ↑ 使用的是23 DOF URDF
```

#### 默认关节角度配置
```python
default_joint_angles = {
    # 腿部: 12 DOF
    'left_hip_pitch_joint': -0.2,
    'left_hip_roll_joint': 0.0,
    # ... (省略其他腿部关节)
    
    # 腰部: 3 DOF
    'waist_yaw_joint': 0.0,
    'waist_roll_joint': 0.0,
    'waist_pitch_joint': 0.0,
    
    # 手臂: 8 DOF (无手腕)
    'left_shoulder_pitch_joint': 0.0,
    'left_shoulder_roll_joint': 0.4,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 1.2,
    # ... (右臂类似)
}
# 注意: 没有任何wrist关节配置
```

---

### 3. 实际部署配置

#### 配置文件
文件: `deploy_real/robot_control/configs/g1.yaml`

```yaml
num_actions: 23  # 策略输出23个DOF

# 腿部关节 (12个)
leg_joint2motor_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 手臂和腰部 (11个: 3腰部 + 8手臂)
arm_waist_joint2motor_idx: [12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25]

# 手腕关节 (6个) ← 注意这里！
wrist_joint2motor_idx: [19, 20, 21, 26, 27, 28]
```

#### 部署代码
文件: `deploy_real/server_low_level_g1_real.py`

```python
def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    total_degrees = 33  # 这里似乎是包含其他观测维度
    wrist_ids = [27, 32]  # 手腕在观测中的索引
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]  # 策略目标（不含手腕）
    wrist_dof_pos = mimic_obs[wrist_ids]  # 手腕位置单独提取
    return policy_target, wrist_dof_pos
```

从配置可以看出，**实际机器人有29个电机（包括6个手腕电机）**，但是：
- **策略网络只输出23 DOF的动作**（不包含手腕）
- **手腕关节被设置为固定目标值**（`wrist_target: [0, 0, 0, 0, 0, 0]`）

---

### 4. CMG模型输出

文件: `CMG_Ref/eval_cmg.py`

```python
def motion_to_npz(motion, output_path, fps=50):
    T = motion.shape[0]
    
    dof_positions = motion[:, :29].astype(np.float32)  # ← CMG输出29 DOF
    dof_velocities = motion[:, 29:].astype(np.float32)
```

**CMG模型确实输出29 DOF的运动数据**

---

## 问题所在

### 当前状态
1. **CMG训练数据**: 29 DOF（包含手腕）
2. **TWIST训练环境**: 23 DOF（不包含手腕）
3. **实际机器人**: 29个电机（包含6个手腕电机）

### 不对齐的原因
**TWIST项目在训练时主动裁切了手腕DOF**，原因可能是：
1. 简化控制复杂度
2. 手腕对行走任务不重要
3. 手腕关节在模拟环境中难以准确建模

### 当前的解决方案
部署代码采用了**混合控制策略**：
- 策略网络控制23个主要关节（腿、腰、手臂）
- 手腕6个关节保持固定姿态（设为0）

```python
# 策略输出23 DOF
policy_output = policy(obs)  # shape: [23]

# 拼接手腕固定姿态，形成完整的29 DOF命令
full_command = combine(policy_output, wrist_fixed_pose)  # shape: [29]
```

---

## 需要做的修改

### 如果要对齐到实际29 DOF机器人，有两个选择：

#### 选项A: 重新训练TWIST（推荐）
1. 修改训练URDF使用 `g1_29dof_rev_1_0.urdf`
2. 修改 `num_actions = 29`
3. 添加手腕关节的默认角度和PD增益
4. 重新训练Teacher和Student策略
5. 直接输出29 DOF动作到实机

**优点**: 端到端对齐，无转换损失
**缺点**: 需要重新训练，耗时较长

#### 选项B: 保持23 DOF + 映射（当前方案）
1. 继续使用23 DOF训练
2. 部署时映射到29 DOF（手腕用固定值或简单规则）
3. 可选：添加手腕运动学解算器

**优点**: 无需重新训练
**缺点**: 手腕运动受限，可能影响某些任务

---

## 相关文件清单

### URDF文件
- `assets/g1/g1_29dof_rev_1_0.urdf` - 29 DOF完整模型
- `assets/g1/g1_custom_collision_with_fixed_hand.urdf` - 23 DOF简化模型（当前使用）

### 配置文件
- `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` - TWIST训练配置（23 DOF）
- `deploy_real/robot_control/configs/g1.yaml` - 实机部署配置（29电机）

### 部署代码
- `deploy_real/server_low_level_g1_real.py` - 实机低层控制
- `deploy_real/server_low_level_g1_sim.py` - 仿真低层控制
- `deploy_real/robot_control/g1_wrapper.py` - 机器人接口封装

### CMG相关
- `CMG_Ref/eval_cmg.py` - CMG评估（输出29 DOF）
- `CMG_Ref/module/cmg.py` - CMG模型定义

---

## 建议

鉴于你提到"实际部署的机器DOF是29而不是23"，建议：

1. **短期方案**: 检查当前部署代码是否正确处理手腕关节
   - 确认 `wrist_joint2motor_idx` 映射正确
   - 验证手腕固定姿态是否合理
   - 测试实机运动时手腕是否稳定

2. **长期方案**: 考虑重新训练29 DOF版本
   - 如果任务需要灵活的手腕控制（如抓取）
   - 如果手腕固定姿态导致运动不自然
   - 如果要充分利用机器人硬件能力

3. **当前可以做的**:
   - 用MuJoCo加载29 DOF模型，验证CMG输出的手腕动作是否合理
   - 实机测试时记录手腕电机状态，确认是否被正确控制
   - 评估手腕对整体运动质量的影响
