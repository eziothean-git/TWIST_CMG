# 🎯 核心诊断 - 为什么机器人在2000步后"张开腿等死"

## 问题现象
- 前1000步：正常训练
- 第2000步后：机器人张开腿，停止移动，无反应

## 已验证✅（不是问题）
✅ CMG 参考值正确生成并传入（`_ref_dof_pos` 非零）
✅ DOF 顺序完全匹配（29个）
✅ 物理参数正确（刚度、阻尼、力矩）
✅ 权重配置合理（tracking reward占89.9%）
✅ CMG 模型输出质量良好

## 可能的根本原因

### 假设1：参考动作不可达（最可能 🔴）
**症状**：机器人物理上无法达到CMG生成的参考角度
- CMG 采样的参考动作可能超出机器人关节范围
- 导致 tracking reward 始终为0
- 模型学不到有用信号，最终"放弃"→选择最小功率（张开腿不动）

**验证方式**：
```python
# 检查参考关节角是否在合理范围内
print(f"ref_dof_pos 范围: [{_ref_dof_pos.min()}, {_ref_dof_pos.max()}]")
# 应该在 [-π, π] 范围内，最好在 [-2.5, 2.5] 内
```

### 假设2：Random Reset 导致的初始化问题
**症状**：`rand_reset=True` 时，每次重置参考动作都从随机时间点开始
- 机器人被重置到参考轨迹的 t=random_time 位置
- 第一步时，参考时间继续从 0 开始计数
- 可能导致参考与实际机器人状态不同步

**验证方式**：
查看调试输出中的时间同步：
```
[DEBUG RESET] 重置环境参考值:
  motion_times[0]: X.XXXXs
  _motion_time_offsets[env_ids[0]]: X.XXXXs  ← 应该相同

[DEBUG Step 1] 参考动作检查:
  motion_time: X.XXXXs  ← 应该与RESET时相同
  _ref_dof_pos[0, :6]: [...]  ← 应该与RESET时完全相同
```

### 假设3：Reward 信号在训练中变弱
**症状**：模型初期学到跟踪，但后期 reward 结构鼓励其他行为
- 随着训练，其他reward项（如 stance stabilization）变弱
- 模型发现"不动"比"跟踪"的总reward更高
- 最终选择不动状态（因为不动=无关节损伤，无运动能量损耗）

**验证方式**：
查看训练日志中的 reward 变化曲线
```
Step 100: tracking_dof=0.6, tracking_vel=0.3, total=1.2
Step 500: tracking_dof=0.5, tracking_vel=0.2, total=1.0
Step 2000: tracking_dof=0.1, tracking_vel=0.05, total=0.2  ← 下降趋势！
```

## 🔍 立即检查的三个指标

### 1. **参考关节范围**
运行调试脚本查看：
```bash
python -c "
import torch
from pose.utils.motion_lib_cmg import MotionLibCMG
# 加载 CMG 动作库
motion_lib = MotionLibCMG(...)
# 检查所有关节位置的范围
print(f'ref_dof_pos 范围: [{motion_lib.dof_pos_pool.min()}, {motion_lib.dof_pos_pool.max()}]')
# 检查各关节的单独范围
for i in range(29):
    min_val = motion_lib.dof_pos_pool[..., i].min()
    max_val = motion_lib.dof_pos_pool[..., i].max()
    print(f'Joint {i}: [{min_val:.3f}, {max_val:.3f}]')
"
```

### 2. **时间同步验证**
运行训练并检查日志：
```bash
# 查看 RESET 和 Step 1 的时间同步
grep "DEBUG RESET" /tmp/training.log | head -5
grep "DEBUG Step 1" /tmp/training.log | head -5
# motion_times 应该相同
```

### 3. **Reward 变化趋势**
```bash
# 提取 reward 分解数据
grep "tracking_joint_dof" /tmp/training.log | tail -20
# 观察是否有下降趋势
```

## 🛠️ 建议的快速修复

### 修复1：禁用 Random Reset（快速测试）
```python
# 在 config 中设置
rand_reset = False  # 改为始终从 t=0 开始
```

### 修复2：确保参考可达
```python
# 在 CMG 配置中限制参考范围
dof_pos = torch.clamp(dof_pos, -2.5, 2.5)  # 限制在安全范围内
```

### 修复3：增加 Tracking Reward 权重
```python
# 早期训练增加权重
tracking_joint_dof = 1.5  # 从 0.84 增加到 1.5
```

## 📊 数据收集清单

要完全诊断问题，需要：
- [ ] 前5步的参考/实际DOF值对比
- [ ] 每10步的reward分解曲线
- [ ] CMG参考关节位置的全范围统计
- [ ] Motion Reset 时的时间同步验证
- [ ] 训练2000步后的最终 reward 值

这些都已在调试代码中输出，运行一次训练即可收集。

