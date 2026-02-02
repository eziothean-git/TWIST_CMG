# CMG 桥接器模块改进记录

## 最新更新（2026-02-02）

### 奖励权重优化 + 速度指令追踪
**问题**：800轮训练，30% motion_end，30% timeout，追踪效果不佳。

**原因**：
1. tracking奖励权重过低（总和4.4 vs feet_air_time 5.0）
2. **关键遗漏**：没有速度指令追踪奖励！CMG根据速度指令生成轨迹，不追踪指令无法对齐

**解决方案**（2倍tracking提升 + 新增速度指令追踪）：
```python
# 轨迹追踪（约2倍提升）
tracking_keybody_pos = 4.0       # 2.0→4.0
tracking_root_vel = 2.0          # 1.0→2.0  
tracking_root_pose = 1.2         # 0.6→1.2
tracking_joint_dof = 1.2         # 0.6→1.2
tracking_joint_vel = 0.4         # 0.2→0.4

# 速度指令追踪（新增）
tracking_lin_vel_exp = 2.0       # xy速度指令
tracking_ang_vel = 1.0           # yaw角速度指令

# 步态优化
feet_air_time = 1.0              # 5.0→1.0（保留但降权）
```

**代码修改**：
1. `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`：添加速度指令追踪奖励项
2. `legged_gym/legged_gym/envs/base/humanoid_mimic.py`：`_update_ref_motion()`中更新commands
   ```python
   if getattr(self, '_use_cmg', False):
       self.commands[:, :3] = self._motion_lib.get_commands()
   ```

**效果预期**：
- tracking总权重：8.8（轨迹）+ 3.0（速度指令）= 11.8
- 模型同时学习轨迹追踪和速度指令对齐
- timeout应下降，追踪精度提升

---

## 概述
CMGBridge 模块支持两种工作模式：在线模式和离线模式，用于管理 CMG 自回归模型生成的参考轨迹。

## 关键改进

### 1. 指令来源的正确设计

#### 核心原则
- **初始姿态**：从数据集中采样（通过 `_get_default_init_motion()`）
- **速度指令**：必须从训练部分传回，而不是在桥接器内独立采样

#### 实现细节

**原有问题**
```python
# 错误：桥接器内独立采样指令
if commands is None:
    commands = self.sample_commands(n)  # 不应该这样
```

**改进后**
```python
# 正确：从训练部分接收指令
def generate_trajectory(self, env_ids, commands):
    # commands 必须由调用者提供，来自训练部分
    if commands is None:
        raise RuntimeError("在线模式下必须提供指令")
```

**方法调整**
- `sample_commands()` → 重命名为 `sample_commands_from_config()`
  - 标记为"仅用于离线模式预计算轨迹池"
  - 不在训练时使用
- `generate_trajectory(env_ids, commands)` → commands 现为必需参数
- `reset(env_ids, commands)` → 在线模式下 commands 不能为 None

#### 设计意义
- **分离关注点**：桥接器只负责轨迹管理，指令采样由训练环境负责
- **便于课程学习**：训练部分可以精细控制指令范围变化
- **数据一致性**：保证指令和采样状态来自同一采样源

### 2. 在线模式优化

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

### 3. 离线模式的采样优化

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

### 4. 离线模式显存释放

#### 目标
- 参考动作预计算完成后主动释放 CMG 模型与统计张量显存

#### 实现
- 预计算完成后调用 `_release_offline_resources()`
- 释放 `_cmg`、归一化统计与初始化样本，并清理 CUDA 缓存

## API 汇总

### 在线模式
```python
# 初始化生成参考轨迹（必须提供指令，来自训练部分）
generate_trajectory(env_ids, commands)

# 手动更新指令并重新生成参考轨迹
update_reference(env_ids, commands)

# 逐步推进，自动触发续生成
step(env_ids=None)

# 重置到新初始状态（必须提供指令）
reset(env_ids, commands)

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

### 指令流向
```
训练部分（环境）
    ↓
采样速度指令 (commands)
    ↓
CMGBridge.reset() / update_reference()
    ↓
轨迹生成与存储
    ↓
训练部分查询特权观测
```

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

5. **指令来源验证**
   - 确保 reset() 和 generate_trajectory() 调用时提供有效的指令
   - 指令形状必须为 (n, 3)，对应 (vx, vy, yaw_rate)
   - 指令来自训练部分的环境采样，保证数据一致性

---

## CMGMotionLib 适配层更新（新增）

### 概述
`pose/pose/utils/cmg_motion_lib.py` 现作为 CMGBridge 的适配层，提供与原 MotionLib 兼容的接口。

### 核心设计
- **职责分离**：CMGMotionLib 只负责接口转换，不进行指令采样
- **指令来源**：所有指令从训练环境传入（通过 `reset()` 和 `update_commands()`）
- **FK 计算**：内部使用 ForwardKinematics 计算关键身体部位位置

### 关键接口

```python
class CMGMotionLib:
    # 初始化
    __init__(cmg_model_path, cmg_data_path, urdf_path, device, num_envs, ...)
    
    # 重置环境
    reset(env_ids, commands)
    # - 离线模式：commands 为轨迹索引 (n,) 或 None（随机分配）
    # - 在线模式：commands 为速度指令 (n, 3)
    
    # 推进时间步
    step(env_ids=None)
    
    # 更新指令（仅在线模式）
    update_commands(env_ids, commands)
    
    # 计算运动帧（MotionLib 兼容接口）
    calc_motion_frame(motion_ids, motion_times, env_ids=None)
    # 返回：root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos
    
    # 采样接口（MotionLib 兼容）
    sample_motions(n, motion_difficulty=None)
    sample_time(motion_ids)
    get_motion_length(motion_ids)
```

### 配置更新

**g1_mimic_distill_config.py** 新增参数：
```python
class motion:
    cmg_offline_mode = True      # 是否使用离线模式
    cmg_num_trajectories = 2048  # 离线轨迹池大小
```

**路径更新**：
- `cmg_workspace` → `CMG_Ref`（统一使用项目内路径）

### 训练部分修改

**g1_mimic_distill.py**：
- `_reset_ref_motion()`: 将 motion_ids 传递给 `reset()` 作为轨迹索引
- `_update_ref_motion()`: 调用 `step()` 推进时间步

**humanoid_mimic.py**：
- `_load_motions()`: 添加 `offline_mode` 和 `num_trajectories` 参数传递

#### 实现细节

**原有问题**
```python
# 错误：桥接器内独立采样指令
if commands is None:
    commands = self.sample_commands(n)  # 不应该这样
```

**改进后**
```python
# 正确：从训练部分接收指令
def generate_trajectory(self, env_ids, commands):
    # commands 必须由调用者提供，来自训练部分
    if commands is None:
        raise RuntimeError("在线模式下必须提供指令")
```

**方法调整**
- `sample_commands()` → 重命名为 `sample_commands_from_config()`
  - 标记为"仅用于离线模式预计算轨迹池"
  - 不在训练时使用
- `generate_trajectory(env_ids, commands)` → commands 现为必需参数
- `reset(env_ids, commands)` → 在线模式下 commands 不能为 None

#### 设计意义
- **分离关注点**：桥接器只负责轨迹管理，指令采样由训练环境负责
- **便于课程学习**：训练部分可以精细控制指令范围变化
- **数据一致性**：保证指令和采样状态来自同一采样源

### 2. 在线模式优化

### 2. 在线模式优化

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

### 3. 离线模式的采样优化

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
# 初始化生成参考轨迹（必须提供指令）
generate_trajectory(env_ids, commands)

# 手动更新指令并重新生成参考轨迹
update_reference(env_ids, commands)

# 逐步推进，自动触发续生成
step(env_ids=None)

# 重置到新初始状态（必须提供指令）
reset(env_ids, commands)

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

---

## 2025-01-26 训练部分集成更新

### 创建 CMGMotionLib v2 适配器

**问题**
- 新的 CMGBridge 需要指令从训练部分传入
- 原始 `HumanoidMimic._reset_ref_motion()` 调用 `self._motion_lib.reset(env_ids)` 不传指令
- 需要一个兼容层来保持现有训练代码接口不变

**解决方案**
创建 `pose/pose/utils/cmg_motion_lib_v2.py`，提供 MotionLib 兼容接口：

1. **reset() 方法兼容性**
   - 离线模式：`commands` 参数作为轨迹索引或 None（随机分配）
   - 在线模式：`commands` 参数作为速度指令，如果为 None 则使用配置范围采样（向后兼容）

2. **calc_motion_frame() 智能查询**
   - 自动识别三种查询模式：
     - 简单查询：`batch_size == num_envs`，返回所有环境当前帧
     - 部分查询：提供 `env_ids`，返回指定环境当前帧
     - Tiled 查询：`batch_size > num_envs`，用于 `_get_mimic_obs()` 的未来帧查询

3. **step() 方法**
   - 包装 `CMGBridge.step()`，推进自回归状态

4. **其他兼容方法**
   - `num_motions()`, `get_motion_length()`, `sample_motions()` 等
   - 与原 MotionLib 接口完全兼容

### 更新训练代码

**文件：legged_gym/legged_gym/envs/base/humanoid_mimic.py**

1. **_load_motions() 方法**
   - 从 `pose.utils.cmg_motion_lib` 改为 `pose.utils.cmg_motion_lib_v2`
   - 新增参数：
     - `offline_mode`: 从配置读取，默认 True（冷启动阶段）
     - `num_trajectories`: 从配置读取，默认 2048

2. **_post_physics_step_callback() 方法**
   - 移除 `self._motion_lib._update_root_state(self.dt)` 调用
   - 根节点状态更新已整合到 `CMGBridge.step()` 内部

**文件：legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py**

新增配置参数到 `G1MimicCMGBaseCfg.motion` 类：
```python
# 冷启动阶段（True）使用离线预生成轨迹，动作衔接阶段（False）使用在线推理
cmg_offline_mode = True
# 离线模式轨迹池大小
cmg_num_trajectories = 2048
```

### 设计意义

1. **保持接口稳定**：现有训练代码无需大幅修改，只需切换导入
2. **灵活模式切换**：通过配置参数即可在冷启动和动作衔接阶段切换
3. **向后兼容**：在线模式下即使不提供指令也能运行（使用配置范围采样）
4. **清晰分层**：
   - CMGBridge：底层轨迹管理
   - CMGMotionLib v2：MotionLib 接口适配
   - HumanoidMimic：训练环境逻辑

### 使用流程

**冷启动阶段（离线模式）**
```python
# config: cmg_offline_mode = True
# 初始化时预生成 2048 条轨迹
# reset() 时从轨迹池随机分配
# 训练过程中只读取预生成轨迹，无 CMG 推理开销
```

**动作衔接阶段（在线模式）**
```python
# config: cmg_offline_mode = False
# reset() 时使用配置范围采样指令（向后兼容）
# 或后续可扩展为从环境传入动态指令
# step() 时 CMG 在线推理续生成轨迹
```

### 下一步

1. 测试冷启动阶段训练（离线模式）
2. 验证 calc_motion_frame() 的 tiled 查询逻辑
3. 准备动作衔接阶段的动态指令传递接口


## 2026-02-02 离线轨迹池配置优化

### 需求
- 增加单个 episode 的训练长度至 10 秒（不含前瞻）以获得更长的轨迹序列
- 增加离线轨迹池大小至 4096 以支持 4096+ 并行环境数量

### 变更内容

**配置文件**: `/legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
```python
# 基础配置 (G1MimicCMGBaseCfg.env)
episode_length_s = 10  # 已符合需求，保持不变

# 离线模式配置 (G1MimicCMGBaseCfg.motion)  
cmg_num_trajectories = 4096  # 2048 → 4096 (翻倍)
```

### 影响分析

#### 显存占用

轨迹池数据结构（单位：GB）：

| 组件 | 旧配置(2048) | 新配置(4096) | 增量 |
|------|------------|------------|------|
| DOF 位置 | 0.105 | 0.211 | +0.106 |
| DOF 速度 | 0.105 | 0.211 | +0.106 |
| 根节点位置 | 0.014 | 0.027 | +0.013 |
| 根节点旋转 | 0.018 | 0.037 | +0.019 |
| 根节点速度 | 0.014 | 0.027 | +0.013 |
| 根节点角速度 | 0.014 | 0.027 | +0.013 |
| 指令 | <0.001 | <0.001 | <0.001 |
| **合计** | **0.270** | **0.540** | **+0.270** |

**结论**：额外显存占用 ~270 MB，在 4090(24GB) 上完全可接受，预留空间充足

#### 预计算耗时

- 总帧数：600 帧（10s episode + 2s 前瞻）
- 旧配置：2048 轨迹 × 600 帧 = 1,228,800 次 CMG forward
- 新配置：4096 轨迹 × 600 帧 = 2,457,600 次 CMG forward
- 耗时增加：在 A100/4090 上约 +10-30 秒（初始化一次性成本）

#### 环境容纳能力

| 指标 | 旧配置 | 新配置 | 改进 |
|------|------|------|------|
| 轨迹池大小 | 2048 | 4096 | 2x |
| 最大并行环境 | 2048 | 4096+ | 支持 8192 环境(轨迹复用) |
| 轨迹覆盖度 | 100% @ 2048 env | 100% @ 4096 env | 显著提升采样多样性 |

### 配置验证

确认以下参数点已正确配置：

✅ 三个速度档位（Slow/Medium/Fast）继承基础配置
✅ Episode 长度：10 秒（约 500 帧 @50Hz）
✅ 前瞻长度：2 秒（100 帧）
✅ 时间步长：0.02 秒（与 CMG 训练一致）
✅ 输出 DOF：23（G1 关节数）

### 训练影响

**冷启动阶段（离线模式）**：
- 初始化时一次性生成 4096 条轨迹（需时 30-50 秒）
- 训练过程无额外开销（仅索引读取）
- 采样多样性提升 2 倍，助力模型泛化

**使用建议**：
```bash
# 推荐新训练命令
bash train_teacher.sh TWIST_CMG_V1 cuda:0 4096 normal "" cmg_medium

# 支持 8192 环境（轨迹池会复用）
bash train_teacher.sh TWIST_CMG_V1 cuda:0 8192 normal "" cmg_medium
```

### 帧数计算澄清

**配置参数**:
- CMG 时间步: 0.02 秒 (50Hz CMG 生成频率)
- 物理模拟: 0.002 秒 (500Hz 物理步)
- Episode 长度: 10 秒
- 前瞻长度: 2 秒

**轨迹帧数计算**:
```
总 CMG 帧数 = (episode_length_s + lookahead_s) / cmg_dt
           = (10 + 2) / 0.02
           = 600 帧
```

**对应物理时间**:
- 10 秒 episode × 50Hz CMG = 500 帧 @CMG
- 2 秒前瞻 × 50Hz CMG = 100 帧 @CMG
- 物理模拟: 600 CMG帧 × (0.02/0.002) = 6000 物理步

**注**: CMG 模型按 0.02s 时间步训练，保持此参数以确保模型性能。

