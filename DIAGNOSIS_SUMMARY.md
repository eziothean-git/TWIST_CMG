# 🔍 TWIST_CMG 训练诊断 - 完整总结

## 问题陈述
在 CMG 模式下训练教师网络时，机器人在约 2000 步迭代后开始"张开腿等死"（停止运动，完全不响应）。

## 诊断进展

### ✅ 第一阶段：数据流验证（已完成）
所有基础数据流都是正确的：

| 检查项 | 结果 | 说明 |
|------|------|------|
| CMG参考值生成 | ✅ | `_ref_dof_pos` 非零且在变化 |
| DOF维度匹配 | ✅ | 29 DOF 全部配置正确 |
| DOF顺序 | ✅ | URDF与IsaacGym完全一致 |
| 关节初始化 | ✅ | 打印显示所有29个关节 |
| 物理参数 | ✅ | 刚度、阻尼、力矩配置合理 |
| 权重配置 | ✅ | Tracking reward占89.9%，权重充足 |
| CMG模型输出 | ✅ | 无NaN/Inf，范围合理 |

### 🔍 第二阶段：根因诊断（进行中）
添加了详细的调试输出来追踪问题。根据初步分析，**最可能的原因**是以下之一：

#### 可能原因1：参考不可达 (概率40%)
**症状**：CMG生成的参考角度物理上超出机器人关节范围
- 导致 tracking reward 始终为0
- 模型无法学习有用信号
- 最终"放弃"跟踪

**验证**：已创建诊断脚本 `check_cmg_feasibility.py`

#### 可能原因2：时间同步问题 (概率30%)  
**症状**：Reset 时的参考与 Step 1 的参考不同步
- 机器人初始位置与第一步参考位置差异大
- 导致初期 reward 很低
- 模型学不到清晰的学习信号

**验证**：已添加 RESET 和 Step 1 时间戳对比输出

#### 可能原因3：Reward 信号衰减 (概率20%)
**症状**：训练过程中 reward 逐步下降
- 模型发现"不动"的 reward 比"跟踪"更高
- 导致策略收敛到"不动"状态

**验证**：已添加逐步的 reward 分解输出

#### 可能原因4：学习率问题 (概率10%)
**症状**：学习率过高导致梯度爆炸或策略崩溃
- 模型在第2000步后突然崩溃

**验证**：需要查看训练日志中的损失曲线

## 🚀 已实施的诊断措施

### 1. 添加了详细的调试输出
#### `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
```python
# ✅ 第 102-107 行：初始化时打印 DOF 顺序
# ✅ 第 141-147 行：Reset 时记录参考值
# ✅ 第 415-431 行：Step 1-5 时打印参考对比
```

#### `legged_gym/legged_gym/envs/base/legged_robot.py`
```python
# ✅ 第 272-296 行：逐个打印 reward 分量
# ✅ 显示每个 reward 项的实际值和权重
```

### 2. 创建了诊断脚本
- `check_cmg_feasibility.py` - 检查参考关节是否在合理范围
- `debug_reward_logic.py` - 分析 reward 计算逻辑
- `DIAGNOSIS_PLAN.md` - 详细的诊断计划

### 3. 保存了完整的配置信息
- 权重配置：Tracking 89.9%, 其他 10.1%
- 物理参数：正确配置
- DOF 顺序：29 个关节完整匹配

## 📋 下一步建议（优先级顺序）

### 优先级1：收集训练日志
**立即运行**：
```bash
cd /home/eziothean/TWIST_CMG
timeout 120 python -m legged_gym.envs.adapt_env \
  with model=teacher \
  num_envs=256 \
  max_iterations=100 \
  2>&1 | tee /tmp/training_debug.log
```

**查看输出**：
```bash
# 检查RESET时间同步
grep "DEBUG RESET" /tmp/training_debug.log | head -3

# 检查Step 1-5的参考值
grep "DEBUG Step" /tmp/training_debug.log | head -20

# 检查Reward分解（第1步）
grep "tracking_joint_dof\|tracking_joint_vel\|TOTAL" /tmp/training_debug.log | head -20
```

### 优先级2：验证参考可达性
```bash
cd /home/eziothean/TWIST_CMG
python check_cmg_feasibility.py
```

**预期输出**：
- 如果有关节超出范围 → 这是根本原因！
- 如果所有关节在范围内 → 问题不在参考范围

### 优先级3：对比Mocap模式
```python
# 临时改为使用Mocap参考（如果有pkl文件）
# 对比行为是否相同
# 如果Mocap模式正常 → 问题确实在CMG
```

## 🛠️ 可能的快速修复

### 修复1：禁用Random Reset（快速测试）
文件：`g1_mimic_distill_config.py` 第46行
```python
rand_reset = False  # 从 t=0 开始，不随机采样
```

### 修复2：限制参考范围到安全值
文件：`pose/utils/motion_lib_cmg.py` 第 209 行添加
```python
# 在生成dof_pos_pool后
self.dof_pos_pool = torch.clamp(self.dof_pos_pool, -2.5, 2.5)
```

### 修复3：增加初期Tracking Reward
文件：`g1_mimic_distill_config.py` 第 219 行
```python
tracking_joint_dof = 1.2  # 从 0.84 增加
tracking_joint_vel = 0.4   # 从 0.28 增加
```

### 修复4：同步机器人到参考初始状态
在 `reset_idx` 中确保机器人位置与参考完全同步（已正确实现，但需验证）

## 📊 关键数据点需要验证

在运行诊断后，需要确认以下数据：

```
□ Step 1 的 tracking_joint_dof reward 是否 > 0.3？
□ Step 1 的 tracking_joint_vel reward 是否 > 0.1？
□ RESET 时间与 Step 1 时间是否相同？
□ Step 1 的参考DOF与RESET时的参考DOF是否相同？
□ CMG 参考关节是否都在 [-2.5, 2.5] rad 内？
□ Step 2000 时的 tracking reward 是否显著下降？
```

## 💡 最可能的最终结论

基于当前的代码分析，我的**最佳猜测**是：

**CMG 生成的参考动作初值与机器人的重置位置有较大差异，导致第一步的 reward 就很低，模型在这个低奖励信号下，最终学到了"保持初始状态不动"这个局部最优解。**

原因链：
1. CMG 生成的第一帧可能不是"站立姿态"
2. 机器人被重置到"站立姿态"
3. 两者有 0.5+ rad 的差异
4. 导致 `exp(-0.15 * 0.5²) ≈ 0.96`（还可以）但 `exp(-0.15 * 多关节误差²)` 很小
5. 模型学不到清晰的"应该跟踪"的信号
6. 最终学到"不动" = 最小功率消耗

**解决方案**：
- 让 CMG 的第一帧与机器人的初始状态对齐
- 或者让机器人重置到 CMG 参考的初始状态
- 或者增加初期 tracking reward 权重

## 📞 需要用户提供的信息

为了进一步诊断，需要：
1. CMG 模型是否已成功生成？确保 `model.pt` 和 `data.pt` 存在
2. 训练时使用的 `max_iterations` 值？
3. 是否看到"张开腿"的具体表现？（如关节电流突然为0）
4. 第 2000 步前的 reward 曲线是什么样的？（上升后平台还是下降？）

---

## 总结

✅ 已验证基础数据流没有问题
✅ 已添加详细调试输出  
✅ 已创建诊断脚本
⏳ 需要运行训练并收集日志
❓ 最终根因需要日志分析来确认

