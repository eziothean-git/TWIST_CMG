# CMG 桥接器模块改进记录

## 概述
CMGBridge 模块支持两种工作模式：在线模式和离线模式，用于管理 CMG 自回归模型生成的参考轨迹。

## 关键改进

### 1. 在线模式优化

#### 问题
- 原始实现在缓冲区快耗尽时才开始重新生成，导致特权观测可能不完整
- 没有手动更新参考轨迹的接口，无法处理指令动态变化的场景

#### 解决方案

**配置参数新增**
- `lookahead_s` (float): 前瞻时长，保证始终可查询未来 2s 轨迹（默认 2.0s）
- `safety_margin_s` (float): 安全缓冲，用于续生成策略的计算（默认 0.5s）

**自动续生成策略**
- 计算 `_reuse_threshold = buffer_frames - lookahead_frames`
- 当帧索引达到阈值时（而非缓冲区末尾），自动触发续生成
- 一次续生成至少生成 `lookahead_s + safety_margin_s` 的轨迹长度
- 新生成的轨迹从当前查询位置开始，覆盖缓冲区末尾部分

**手动更新接口**
```python
def update_reference(env_ids, commands):
    # 当指令改变时（如动作衔接过程），调用此方法
    # 会根据新指令重新生成参考轨迹
    # 用于支持动态变化的控制指令
```

### 2. 离线模式的采样优化

#### 问题
- 离线模式在初始化时预计算所有轨迹，采样完全随机
- 无法支持课程学习、指定特定速度范围等需求

#### 解决方案

**灵活的预计算接口**
```python
def _precompute_offline_trajectories(commands: Optional[Tensor] = None):
    # 如果 commands 为 None，使用配置范围随机采样
    # 如果 commands 不为 None，使用指定的指令生成轨迹
```

**reset() 方法扩展**
- 离线模式下，`commands` 参数现在被解释为轨迹索引数组
- 支持从轨迹池中指定选择轨迹（便于课程学习和实验）
- 传 `None` 时随机分配轨迹

```python
# 示例：课程学习
# 第一阶段只用前 512 条轨迹（低速）
traj_indices = torch.arange(512, device='cuda')
bridge.reset(env_ids, traj_indices)

# 第二阶段使用所有轨迹
bridge.reset(env_ids)  # 随机分配
```

## API 汇总

### 在线模式
```python
# 初始化生成参考轨迹
generate_trajectory(env_ids, commands=None)

# 手动更新指令并重新生成参考轨迹
update_reference(env_ids, commands)

# 逐步推进，自动触发续生成
step(env_ids=None)

# 重置到新初始状态
reset(env_ids, commands=None)

# 查询接口
get_current_frame(env_ids=None)      # 当前帧
get_future_frames(env_ids, offsets)  # 未来多帧（用于特权观测）
```

### 离线模式
```python
# 预计算轨迹池（可选择传入指定指令）
_precompute_offline_trajectories(commands=None)

# 从轨迹池随机或指定分配
reset(env_ids, commands=None)
# - commands=None: 随机分配
# - commands=tensor: 用作轨迹索引，指定分配

# 逐步推进
step(env_ids=None)

# 查询接口
get_current_frame(env_ids=None)      # 当前帧
get_future_frames(env_ids, offsets)  # 未来多帧
```

## 实现细节

### 缓冲区管理（在线模式）
- `_buffer_frames`: 缓冲区大小（2s = 100 帧）
- `_generate_frames`: 一次生成的最小帧数（max(buffer_frames, lookahead + safety_margin)）
- `_reuse_threshold`: 触发续生成的阈值（buffer_frames - lookahead_frames）

### 轨迹池管理（离线模式）
- `_pool_*`: 存储预计算的轨迹数据
- `_pool_commands`: 每条轨迹对应的指令
- `_env_traj_idx`: 每个环境当前分配的轨迹索引
- `_frame_idx`: 每个环境当前的帧索引

### 根节点状态计算
- 在线模式中，根节点位置/旋转通过速度指令连续积分计算
- 确保轨迹连续性和物理正确性

## 注意事项

1. **在线模式前瞻余量**
   - 务必保证 `buffer_frames > lookahead_frames + safety_margin_frames`
   - 否则无法维持持续的 2s 前瞻观测

2. **离线模式的轨迹索引**
   - 索引范围必须在 `[0, num_trajectories)` 内
   - 可用于实现采样策略的监督

3. **CMG 模型连续性**
   - 在线模式下，续生成与原生成使用相同的指令
   - 确保模型状态跟踪的连续性

4. **设备一致性**
   - 所有张量操作在指定的 device 上进行
   - 确保 GPU 显存使用的效率
