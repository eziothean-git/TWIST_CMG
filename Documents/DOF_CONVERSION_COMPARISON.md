# DOF转换方案对比：转换器 vs 替换URDF

## 结论

**推荐方案：写转换器（方案A）**

- ✅ **更简单**：无需重新训练，只需编写映射逻辑
- ✅ **无需修改TWIST架构**：现有训练代码和模型完全不变
- ✅ **快速验证**：立即可在实机上测试
- ⚠️ **限制**：手腕固定或使用简单规则

---

## 详细对比

### 方案A：编写DOF转换器（推荐）

#### 实现复杂度：⭐⭐ (简单)

**需要修改的地方：**
1. **部署代码**（仅此一处）
   - 文件：`deploy_real/server_low_level_g1_real.py`
   - 修改：在策略输出后添加手腕DOF插值

**代码量估计：** ~20-30行

#### 具体实现

```python
# deploy_real/server_low_level_g1_real.py

def convert_23dof_to_29dof(policy_output_23, wrist_strategy='zero'):
    """
    将策略输出的23 DOF动作转换为29 DOF
    
    Args:
        policy_output_23: (23,) 策略输出
            [0:12]  - 腿部 (6+6)
            [12:15] - 腰部 (3)
            [15:23] - 手臂 (4+4)
        wrist_strategy: 手腕处理策略
            'zero': 手腕保持0（默认）
            'fixed': 使用配置的固定值
            'mirror': 从肘部镜像
    
    Returns:
        action_29: (29,) 完整动作
            [0:12]  - 腿部
            [12:15] - 腰部
            [15:19] - 左臂（肩3+肘1）
            [19:22] - 左手腕（roll+pitch+yaw）← 新增
            [22:26] - 右臂（肩3+肘1）
            [26:29] - 右手腕（roll+pitch+yaw）← 新增
    """
    action_29 = np.zeros(29, dtype=np.float32)
    
    # 1. 复制腿部和腰部 (0-15)
    action_29[0:15] = policy_output_23[0:15]
    
    # 2. 复制手臂
    action_29[15:19] = policy_output_23[15:19]  # 左臂
    action_29[22:26] = policy_output_23[19:23]  # 右臂
    
    # 3. 填充手腕 DOF
    if wrist_strategy == 'zero':
        action_29[19:22] = 0.0  # 左手腕
        action_29[26:29] = 0.0  # 右手腕
    
    elif wrist_strategy == 'fixed':
        # 从配置读取（config.wrist_target）
        action_29[19:22] = [0.0, 0.0, 0.0]  # 左手腕固定姿态
        action_29[26:29] = [0.0, 0.0, 0.0]  # 右手腕固定姿态
    
    elif wrist_strategy == 'mirror':
        # 简单策略：手腕roll跟随肘部
        left_elbow = policy_output_23[18]
        right_elbow = policy_output_23[22]
        action_29[19] = left_elbow * 0.3   # 左手腕roll
        action_29[26] = right_elbow * 0.3  # 右手腕roll
        action_29[20:22] = 0.0             # pitch, yaw = 0
        action_29[27:29] = 0.0
    
    return action_29
```

**在部署代码中调用：**
```python
# 原来：
policy_output = policy(obs)  # shape: (23,)
self.env.step(policy_output)

# 修改为：
policy_output_23 = policy(obs)  # shape: (23,)
policy_output_29 = convert_23dof_to_29dof(policy_output_23, wrist_strategy='zero')
self.env.step(policy_output_29)
```

#### 是否需要修改TWIST观测？

**❌ 不需要！** 理由：

1. **训练阶段完全不变**
   - TWIST依然使用23 DOF URDF训练
   - `self.num_dof` 自动从URDF读取 = 23
   - 观测维度自动适配：
     ```python
     # legged_gym/legged_gym/envs/base/legged_robot.py
     self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # = 23
     ```

2. **部署阶段独立处理**
   - 策略网络输入输出不变（23维）
   - 转换器在**策略输出之后、发送到机器人之前**执行
   - 机器人接收29维命令

3. **观测流程图**：
   ```
   训练时：
   机器人(23DOF) → 传感器读取(23) → 观测处理 → 策略网络(23→23)
   
   部署时：
   机器人(29DOF) → 传感器读取(29) → [丢弃手腕6DOF] → 观测(23) 
                                                      ↓
                                            策略网络(23→23)
                                                      ↓
                                       [转换器补充手腕] → 命令(29) → 机器人
   ```

#### 优点
- ✅ **无需重新训练**（节省数天训练时间）
- ✅ **代码改动最小**（仅部署代码）
- ✅ **快速验证**（立即可测试）
- ✅ **灵活调整**（可尝试不同手腕策略）
- ✅ **风险低**（训练模型不动，只改部署）

#### 缺点
- ⚠️ **手腕运动受限**（固定或简单规则）
- ⚠️ **无法学习手腕协调**（对走路影响很小）

---

### 方案B：替换URDF重新训练

#### 实现复杂度：⭐⭐⭐⭐⭐ (复杂)

**需要修改的地方：**

#### 1. 配置文件修改
```python
# legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py

class G1MimicPrivCfg(HumanoidMimicCfg):
    class env:
        num_actions = 29  # ← 改为29
        
        # 所有依赖num_actions的计算都需要更新
        n_priv_latent = 4 + 1 + 2*29  # 原来是2*23
        n_proprio = 3 + 2 + 3*29      # 原来是3*23
        # ... 更多依赖num_actions的变量
    
    class asset:
        # 替换URDF
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_29dof_rev_1_0.urdf'
        
        # 更新dof_armature（从23→29）
        dof_armature = [
            0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597,  # 左腿
            0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597,  # 右腿
            0.0103, 0.0103, 0.0103,                               # 腰
            0.003597, 0.003597, 0.003597, 0.003597,               # 左臂
            0.00425, 0.00425, 0.00425,                            # 左手腕 ← 新增
            0.003597, 0.003597, 0.003597, 0.003597,               # 右臂
            0.00425, 0.00425, 0.00425,                            # 右手腕 ← 新增
        ]
    
    class init_state:
        default_joint_angles = {
            # 原有23个关节...
            # 新增6个手腕关节
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }
    
    class control:
        stiffness = {
            # 原有...
            'wrist': 20,  # 新增手腕刚度
        }
        damping = {
            # 原有...
            'wrist': 1,   # 新增手腕阻尼
        }
```

#### 2. 环境代码检查（可能需要修改）

需要检查所有硬编码23的地方：
```bash
grep -r "23" legged_gym/legged_gym/envs/g1/ --include="*.py"
```

可能需要修改：
- 观测构建逻辑
- 奖励计算（如果有逐关节的权重）
- 任何硬编码的索引

#### 3. Motion Library适配

**关键问题：** CMG输出的29 DOF数据中，手腕关节的顺序和URDF是否一致？

需要验证：
```python
# CMG输出顺序 vs URDF关节顺序
# 可能需要重新映射
```

#### 4. 重新训练

**时间成本：**
- Teacher训练：~24小时（取决于硬件）
- Student训练：~12-24小时
- 调试迭代：额外时间

**计算资源：**
- GPU：高性能GPU（至少RTX 3090级别）
- 可能需要多次实验调整超参数

#### 是否需要修改观测？

**✅ 需要！** 因为：

1. **观测维度自动变化**
   ```python
   # 训练时从URDF自动读取
   self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # 29→29
   
   # 所有基于num_dof的观测都会变
   n_proprio = 3 + 2 + 3*self.num_dof  # 3*23 → 3*29
   ```

2. **网络架构变化**
   - 输入层变大（观测维度增加）
   - 输出层变大（动作维度29）
   - 需要重新训练整个网络

3. **不会"自适应"** 
   - 虽然代码写的是 `self.num_dof`，看起来通用
   - 但改变DOF数量后，网络权重完全不兼容
   - 必须从头训练

#### 优点
- ✅ **端到端对齐**（训练和部署一致）
- ✅ **手腕可学习**（如果需要灵活手腕）
- ✅ **理论上更优**（无转换损失）

#### 缺点
- ❌ **训练时间长**（1-3天）
- ❌ **需要调试**（可能需要多次迭代）
- ❌ **资源消耗大**（GPU、电力）
- ❌ **代码改动多**（配置、环境、可能的硬编码）
- ❌ **风险高**（可能训不好，回退成本高）

---

## 架构自适应性分析

### TWIST代码的"自适应"程度

#### ✅ 部分自适应
```python
# 这些会自动适配DOF数量
self.num_dof = self.gym.get_asset_dof_count(robot_asset)
self.dof_pos = torch.zeros(self.num_envs, self.num_dof, ...)
self.torques = torch.zeros(self.num_envs, self.num_dof, ...)
```

#### ❌ 不自适应的部分
1. **配置中的硬编码**
   ```python
   num_actions = 23  # 需要手动改为29
   dof_armature = [...]  # 23个数值，需要改为29个
   default_joint_angles = {...}  # 23个关节，需要加6个
   ```

2. **奖励权重**
   ```python
   dof_err_w = [1.0, 0.8, ..., 1.0]  # 如果有23个权重，需要改为29个
   ```

3. **网络模型**
   - 输入维度、输出维度与DOF数量耦合
   - **无法加载旧模型权重**

### 结论：需要重新训练，不能直接"自适应"

虽然代码中很多地方使用了 `self.num_dof`，但：
- 改变URDF后，`num_dof` 会自动变化
- 但配置、网络架构都需要相应修改
- **已训练的模型权重无法复用**

---

## 最终推荐

### 针对"走路"工况

**方案A：写转换器** ← 强烈推荐

**理由：**
1. **手腕对走路不重要**
   - 走路主要靠腿、腰、手臂摆动
   - 手腕固定或简单规则完全够用
   - CMG数据中的手腕运动可能也只是跟随肘部

2. **实现成本对比**
   - 方案A：1小时编码 + 立即测试
   - 方案B：2-3天训练 + 不确定效果

3. **风险对比**
   - 方案A：失败了随时回退
   - 方案B：训不好浪费大量时间

### 实施步骤（方案A）

#### Step 1: 编写转换器（10分钟）
```python
# deploy_real/dof_converter.py
def convert_23_to_29(action_23):
    action_29 = np.zeros(29)
    action_29[0:15] = action_23[0:15]     # 腿+腰
    action_29[15:19] = action_23[15:19]   # 左臂
    action_29[19:22] = 0.0                # 左手腕(固定)
    action_29[22:26] = action_23[19:23]   # 右臂
    action_29[26:29] = 0.0                # 右手腕(固定)
    return action_29
```

#### Step 2: 修改部署代码（10分钟）
```python
# deploy_real/server_low_level_g1_real.py
from dof_converter import convert_23_to_29

# 在策略输出后添加
action = convert_23_to_29(policy_output)
```

#### Step 3: 更新配置（5分钟）
```yaml
# deploy_real/robot_control/configs/g1.yaml
# 确认wrist_joint2motor_idx正确
# 确认wrist_target为固定值
```

#### Step 4: 实机测试（立即）
- 部署到服务器
- 观察手腕是否稳定
- 观察整体运动质量

#### Step 5: 优化（可选）
如果手腕抖动或不自然，尝试：
- 调整 `wrist_target` 值
- 添加手腕阻尼
- 实现简单的手腕跟随规则

---

## 何时考虑方案B？

仅在以下情况考虑重新训练29 DOF：

1. **任务需要灵活手腕**
   - 如：需要抓取、操作物体
   - 如：手腕姿态影响平衡

2. **已验证手腕很重要**
   - 实测发现固定手腕导致性能下降
   - CMG数据显示手腕有复杂运动模式

3. **有充足时间和资源**
   - 有高性能GPU集群
   - 可以接受1-3天训练时间
   - 有预算进行多次实验

**对于当前"走路"工况，强烈建议先尝试方案A！**

---

## 附录：关键代码位置

### 转换器实现位置
- **新建文件**: `deploy_real/dof_converter.py`
- **修改文件**: `deploy_real/server_low_level_g1_real.py`
- **修改行数**: ~10行

### 如果选择方案B需要修改的文件
1. `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
2. `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py` (可能)
3. 所有训练脚本的配置参数
4. Motion library加载逻辑(可能)

### 验证清单（方案A）
- [ ] 转换器正确映射23→29索引
- [ ] 手腕电机ID配置正确
- [ ] 手腕PD增益设置合理
- [ ] 实机测试手腕不抖动
- [ ] 整体运动质量不下降
