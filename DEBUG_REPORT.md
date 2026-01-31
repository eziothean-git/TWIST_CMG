# 🔍 TWIST_CMG 训练调试报告

## 📊 诊断进展

### ✅ 已验证（非问题）
1. **CMG 参考值生成** ✓
   - `_ref_dof_pos` 非零且在变化
   - `_ref_dof_vel` 值合理
   - 关节顺序与 URDF 完全匹配（29 个关节）

2. **维度配置** ✓
   - `num_dof = 29`
   - `num_actions = 29`
   - CMG `dof_dim = 29`
   - 所有维度一致

3. **物理参数** ✓
   - 刚度、阻尼配置合理
   - PD 控制器设置正确

4. **CMG 模型输出质量** ✓
   - 关节位置范围合理：[-2.047, 2.415] rad
   - 左右腿相关性正确：-0.718
   - 无 NaN/Inf

### ❓ 待调查（问题可能在这里）

#### 1. **Reward 信号驱动**
   当前调试标准：
   - `tracking_joint_dof` reward 在早期是否 > 0.5？
   - `tracking_joint_vel` reward 是否也大于 0？
   - 这两项的加权和是否足以驱动学习？

   **调试方式**：查看训练日志中的 reward 分解（已添加详细打印）

#### 2. **位置误差过大**
   从调试输出看：
   ```
   Step 1:
     _ref_dof_pos[0, :6]:  [-0.55, -0.12, -0.00, 0.49, 0.35, 0.00]
     dof_pos[0, :6]:       [ 0.20, -0.13, -0.46, 0.10, 0.26, -0.20]
     差异:                  [-0.75, +0.01, -0.46, +0.39, +0.09, -0.20]
   ```
   
   关节 0（left_hip_pitch）的误差达到 **0.75 rad ≈ 43°**
   
   这在物理上是可以达到的（腿部关节范围一般是 ±2 rad），但：
   - 这个误差导致的 reward 可能很低
   - 如果 reward 太低，模型无法学习

#### 3. **CMG 生成的参考是否真的可达？**
   检查方式：
   - 在不使用策略的情况下，让 PD 控制器追踪 CMG 参考
   - 观察是否能成功跟踪
   - 如果不能，说明参考不可达

## 🎯 下一步诊断方案

### 优先级 1：**立即查看 Reward 分解**
运行命令：
```bash
cd /home/eziothean/TWIST_CMG
# 从头训练10步并输出 reward 分解
python -m legged_gym.envs.adapt_env with model=teacher num_envs=128 max_iterations=10
```

查看输出中的：
- `tracking_joint_dof` 值（应该 > 0.3）
- `tracking_joint_vel` 值（应该 > 0.2）
- 这些是否在逐步变化（表明机器人在尝试）

### 优先级 2：**验证 PD 追踪能力**
创建简单脚本直接验证：
```python
# 让仿真器用 PD 追踪 CMG 参考，不用策略网络
ref_pos = get_cmg_reference()
for t in range(100):
    error = ref_pos[t] - current_pos
    torque = kp * error + kd * (ref_vel[t] - current_vel)
    step_simulation(torque)
    
# 检查最终误差是否 < 0.1 rad
```

### 优先级 3：**检查初始化条件**
确认：
- 机器人初始位置是否在可达范围内
- 第一帧的 `_ref_dof_pos` 是否与 `dof_pos` 相近
- `episode_length_buf` 的初始值是否正确

## 📝 已添加的调试代码

### 1. `humanoid_mimic.py` - 参考观测调试
```python
def _post_physics_step_callback(self):
    if self.common_step_counter < 5:
        # 打印前5步的参考值对比
        print(f"_ref_dof_pos[0, :6]: {self._ref_dof_pos[0, :6]}")
        print(f"dof_pos[0, :6]:      {self.dof_pos[0, :6]}")
        print(f"差异: {(self._ref_dof_pos[0, :6] - self.dof_pos[0, :6])}")
        print(f"actions[0]: {self.actions[0]}")
```

### 2. `legged_robot.py` - Reward 分解调试
```python
def compute_reward(self):
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]
        if self.common_step_counter < 5:
            # 打印每个reward分量
            print(f"{name:30s}: {rew[0]:.6f}")
    print(f"{'TOTAL':30s}: {self.rew_buf[0]:.6f}")
```

### 3. `humanoid_mimic.py` - DOF 顺序验证
```python
def _init_motion_buffers(self):
    cprint(f"[DEBUG] IsaacGym DOF 顺序 (共{self.num_dof}个):", "yellow")
    for i, name in enumerate(self.dof_names):
        cprint(f"  [{i:2d}] {name}", "cyan")
```

## 🚀 快速验证命令

```bash
# 1. 运行10步训练并观察日志
cd /home/eziothean/TWIST_CMG
timeout 30 python -m legged_gym.envs.adapt_env with model=teacher 2>&1 | grep -E "(DEBUG|tracking_joint|TOTAL)" | head -n 100

# 2. 查看 reward 权重配置是否被正确加载
grep -r "tracking_joint_dof" CMG_Ref/... legged_gym/...

# 3. 检查是否有异常的权重值
python -c "from legged_gym.envs.g1.g1_mimic_distill_config import g1_mimic_distill_cfg; cfg = g1_mimic_distill_cfg(); print(cfg.rewards)"
```

## 💡 最可能的问题

基于当前信息，**最可能的问题是**：

**CMG 生成的参考动作在第一步就与机器人的初始状态有很大差异，导致 reward 一开始就很低。**

原因分析：
- CMG 是从随机初始状态生成的轨迹
- 但机器人可能从固定的初始状态开始（如站立姿态）
- 导致第一步的误差很大（0.5+ rad）
- 这个大误差使得 `exp(-0.15 * large_error²)` 接近 0
- 模型没有清楚的学习信号，最终学到"保持初始状态" = 最小成本

**解决方案**：
让 CMG 生成与机器人初始状态一致的轨迹，或者在重置时同步机器人到 CMG 参考的起始状态。

