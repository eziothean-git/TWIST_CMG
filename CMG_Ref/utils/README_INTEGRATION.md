# CMG-TWIST 集成使用指南

## 概述

`CMGMotionGenerator` 提供了两种工作模式，用于不同的训练阶段：

1. **预生成模式 (Pregenerated)**: 批量生成完整轨迹序列，用于训练初期冷启动
2. **实时推理模式 (Realtime)**: 自回归生成，支持动态命令更新，用于后期复杂场景

## 快速开始

### 1. 预生成模式 (推荐用于初始训练)

```python
from CMG_Ref.utils.cmg_motion_generator import CMGMotionGenerator

# 创建生成器
generator = CMGMotionGenerator(
    model_path='runs/cmg_20260123_194851/cmg_final.pt',
    data_path='dataloader/cmg_training_data.pt',
    num_envs=4096,
    device='cuda',
    mode='pregenerated',
    preload_duration=500  # 10秒轨迹 @ 50Hz
)

# 初始化环境
commands = torch.randn(4096, 3, device='cuda') * 0.5  # 随机速度命令
generator.reset(commands=commands)

# 在训练循环中获取参考动作
for step in range(max_steps):
    # 获取当前帧的参考动作
    ref_dof_pos, ref_dof_vel = generator.get_motion()
    
    # 传递给TWIST环境
    obs = env.step(actions, ref_dof_pos, ref_dof_vel)
    
    # 当轨迹结束时重置
    if done:
        generator.reset(env_ids=done_env_ids, commands=new_commands)
```

### 2. 实时推理模式 (用于动态命令跟踪)

```python
# 创建生成器
generator = CMGMotionGenerator(
    model_path='runs/cmg_20260123_194851/cmg_final.pt',
    data_path='dataloader/cmg_training_data.pt',
    num_envs=4096,
    device='cuda',
    mode='realtime',
    buffer_size=100  # 2秒缓冲 @ 50Hz
)

# 初始化
generator.reset(commands=init_commands)

# 训练循环
for step in range(max_steps):
    # 获取参考动作
    ref_dof_pos, ref_dof_vel = generator.get_motion()
    
    # 动态更新命令 (例如每50步)
    if step % 50 == 0:
        new_commands = sample_random_commands()
        generator.update_commands(new_commands)
    
    obs = env.step(actions, ref_dof_pos, ref_dof_vel)
```

### 3. 命令平滑器 (可选)

```python
from CMG_Ref.utils.cmg_motion_generator import CommandSmoother

# 创建平滑器
smoother = CommandSmoother(
    num_envs=4096,
    interpolation_steps=25,  # 0.5秒平滑 @ 50Hz
    device='cuda'
)

# 设置新目标命令
target_cmd = torch.tensor([[1.0, 0.0, 0.2]], device='cuda').repeat(4096, 1)
smoother.set_target(target_cmd)

# 每步获取平滑后的命令
for step in range(training_steps):
    smooth_cmd = smoother.get_current()
    generator.update_commands(smooth_cmd)
    ref_dof_pos, ref_dof_vel = generator.get_motion()
```

## 集成到TWIST环境

### 方法 1: 环境内部集成

在 `g1_mimic_distill.py` 中添加：

```python
class G1MimicEnv(HumanoidMimicEnv):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 初始化CMG生成器
        self.cmg_generator = CMGMotionGenerator(
            model_path=cfg.cmg.model_path,
            data_path=cfg.cmg.data_path,
            num_envs=self.num_envs,
            device=self.device,
            mode=cfg.cmg.mode,  # 'pregenerated' or 'realtime'
            preload_duration=cfg.cmg.preload_duration,
            buffer_size=cfg.cmg.buffer_size,
        )
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        
        # 为重置的环境生成新的CMG轨迹
        commands = self.sample_commands(env_ids)
        self.cmg_generator.reset(env_ids=env_ids, commands=commands)
    
    def _compute_observations(self):
        # 获取CMG参考动作
        ref_dof_pos, ref_dof_vel = self.cmg_generator.get_motion()
        
        # 计算跟踪误差等观察
        # ... 你的观察计算代码 ...
        
        return obs
```

### 方法 2: 外部数据加载

创建预处理脚本 `CMG_Ref/utils/preprocess_trajectories.py`:

```python
"""
预处理CMG轨迹并保存为PKL格式
用于离线训练
"""

import torch
import pickle
from cmg_motion_generator import CMGMotionGenerator

def generate_training_dataset(
    model_path: str,
    data_path: str,
    num_trajectories: int = 10000,
    trajectory_length: int = 500,
    output_path: str = 'cmg_trajectories.pkl'
):
    """批量生成训练数据集"""
    
    generator = CMGMotionGenerator(
        model_path=model_path,
        data_path=data_path,
        num_envs=num_trajectories,
        device='cuda',
        mode='pregenerated',
        preload_duration=trajectory_length
    )
    
    # 采样多样化的速度命令
    commands = sample_diverse_commands(num_trajectories)
    generator.reset(commands=commands)
    
    # 收集所有轨迹
    trajectories = []
    for i in range(trajectory_length):
        ref_dof_pos, ref_dof_vel = generator.get_motion()
        trajectories.append({
            'dof_pos': ref_dof_pos.cpu(),
            'dof_vel': ref_dof_vel.cpu(),
            'commands': commands.cpu(),
        })
    
    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"Saved {num_trajectories} trajectories to {output_path}")

if __name__ == "__main__":
    generate_training_dataset(
        model_path='runs/cmg_20260123_194851/cmg_final.pt',
        data_path='dataloader/cmg_training_data.pt',
        num_trajectories=10000,
        trajectory_length=500,
        output_path='datasets/cmg_training_trajectories.pkl'
    )
```

## 配置参数说明

### CMGMotionGenerator 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | - | CMG模型权重路径 |
| `data_path` | str | - | 训练数据路径(含统计信息) |
| `num_envs` | int | 4096 | 并行环境数量 |
| `device` | str | 'cuda' | 计算设备 |
| `mode` | str | 'pregenerated' | 工作模式 ('pregenerated'/'realtime') |
| `buffer_size` | int | 100 | 实时模式缓冲区大小(帧) |
| `preload_duration` | int | 500 | 预生成模式轨迹长度(帧) |

### 推荐配置

**阶段 1: 初始训练 (0-5k iterations)**
```python
mode='pregenerated'
preload_duration=500  # 10秒轨迹
commands: 简单前进/后退/转向
```

**阶段 2: 多样化训练 (5k-15k iterations)**
```python
mode='pregenerated'
preload_duration=300  # 6秒轨迹
commands: 混合运动模式
```

**阶段 3: 动态跟踪 (15k+ iterations)**
```python
mode='realtime'
buffer_size=100
commands: 实时更新，模拟真实场景
```

## 性能优化

### 1. 批量大小选择

- 4096环境: 预生成模式 ~200ms/重置，实时模式 ~50ms/填充
- 8192环境: 可能需要梯度累积或减少`preload_duration`

### 2. 内存管理

预生成模式内存占用: `num_envs * preload_duration * 58 * 4 bytes`
- 4096 envs × 500 frames × 58 dims × 4B = ~476 MB

实时模式内存占用: `num_envs * buffer_size * 58 * 4 bytes`
- 4096 envs × 100 frames × 58 dims × 4B = ~95 MB

### 3. 推理优化

如需进一步加速，可考虑：

```python
# 方法1: TorchScript JIT编译
model_scripted = torch.jit.script(model)

# 方法2: 降低精度
model.half()  # FP16推理
```

## 故障排除

### 问题1: 内存不足
**解决方案**: 
- 减少`preload_duration`或`buffer_size`
- 使用更小的`num_envs`分批处理
- 切换到实时模式

### 问题2: 推理速度慢
**解决方案**:
- 确认GPU利用率 (`nvidia-smi`)
- 使用TorchScript或FP16
- 检查是否频繁重置环境

### 问题3: 轨迹质量差
**解决方案**:
- 检查命令范围是否在训练分布内
- 验证模型加载正确
- 检查归一化统计信息

## API参考

### CMGMotionGenerator

#### 主要方法

**`reset(env_ids, init_motion, commands)`**
- 重置指定环境的生成器状态
- `env_ids`: 可选，要重置的环境ID
- `init_motion`: 可选，初始动作 [N, 58]
- `commands`: 可选，速度命令 [N, 3]

**`get_motion(env_ids)`**
- 获取当前帧参考动作
- 返回: `(dof_pos, dof_vel)` 各为 [N, 29]

**`update_commands(commands, env_ids)`**
- 更新速度命令
- `commands`: [N, 3] 新命令
- `env_ids`: 可选，要更新的环境

**`switch_mode(new_mode)`**
- 切换工作模式
- `new_mode`: 'pregenerated' 或 'realtime'

**`get_performance_stats()`**
- 获取性能统计
- 返回: dict包含平均/最大/最小推理时间

### CommandSmoother

**`set_target(target_cmd, env_ids)`**
- 设置目标命令

**`get_current(env_ids)`**
- 获取平滑后的当前命令
- 返回: [N, 3] 平滑命令

## 测试

运行完整测试套件：

```bash
cd CMG_Ref
python test_motion_generator.py --test all
```

单独测试各模块：

```bash
# 预生成模式
python test_motion_generator.py --test pregen

# 实时模式
python test_motion_generator.py --test realtime

# 大规模并行 (4096环境)
python test_motion_generator.py --test large

# 命令平滑器
python test_motion_generator.py --test smoother

# 模式切换
python test_motion_generator.py --test switch
```

## 下一步

1. ✅ 创建 CMG-TWIST 桥接类
2. ⬜ 在TWIST环境中集成
3. ⬜ 实现命令采样策略
4. ⬜ 添加地形自适应
5. ⬜ 性能benchmark和优化

## 相关文档

- [ToDo.zh.md](../ToDo.zh.md) - 完整项目待办列表
- [ProjectDocumentation.zh.md](../ProjectDocumentation.zh.md) - 项目文档
- [eval_cmg.py](../eval_cmg.py) - CMG评估脚本示例
