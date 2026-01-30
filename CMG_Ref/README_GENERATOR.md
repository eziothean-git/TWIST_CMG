# CMG-TWIST 运动生成器

高效的双模式CMG运动生成器，专为TWIST强化学习训练设计，支持4096个并行环境。

## 快速开始

### 1. 基本使用

```python
from CMG_Ref.utils.cmg_motion_generator import CMGMotionGenerator

# 创建生成器 (预生成模式 - 推荐用于训练初期)
generator = CMGMotionGenerator(
    model_path='runs/cmg_20260123_194851/cmg_final.pt',
    data_path='dataloader/cmg_training_data.pt',
    num_envs=4096,
    mode='pregenerated',  # 或 'realtime'
    device='cuda'
)

# 初始化
import torch
commands = torch.randn(4096, 3, device='cuda') * 0.5  # [vx, vy, yaw]
generator.reset(commands=commands)

# 获取参考动作
ref_dof_pos, ref_dof_vel = generator.get_motion()  # [4096, 29]
```

### 2. 运行测试

```bash
cd CMG_Ref

# 快速测试
python integration_examples.py --example 1

# 完整测试
python test_motion_generator.py --test all

# 性能测试 (4096环境)
python test_motion_generator.py --test large
```

### 3. 两种工作模式

#### 预生成模式 (冷启动)
```python
# 一次性生成完整轨迹，训练时快速采样
mode='pregenerated'
preload_duration=500  # 10秒轨迹 @ 50Hz
```

**优点**: 
- 训练时零推理开销
- 适合初期大量重复训练

**用于**: 训练初期 (0-5k iterations)

#### 实时模式 (动态跟踪)
```python
# 自回归生成+缓冲，支持动态命令
mode='realtime'
buffer_size=100  # 2秒缓冲 @ 50Hz
```

**优点**:
- 支持动态命令更新
- 内存占用小

**用于**: 训练后期 (15k+ iterations)

### 4. 命令采样

```python
from CMG_Ref.utils.command_sampler import get_stage_sampler

# 根据训练阶段选择采样器
sampler = get_stage_sampler('stage1_basic', num_envs=4096)
commands = sampler.sample_uniform()  # 随机采样
```

训练阶段：
- `stage1_basic`: 仅前进
- `stage2_turn`: 前进+转向
- `stage3_strafe`: 全方向运动
- `stage4_mixed`: 混合运动

### 5. 命令平滑

```python
from CMG_Ref.utils.cmg_motion_generator import CommandSmoother

smoother = CommandSmoother(num_envs=4096, interpolation_steps=25)
smoother.set_target(new_commands)

# 每步获取平滑命令
smooth_cmd = smoother.get_current()
```

## 核心特性

- ✅ 支持4096并行环境
- ✅ 双模式切换 (预生成/实时)
- ✅ 批量高效推理
- ✅ 智能缓冲管理
- ✅ 命令平滑过渡
- ✅ 性能监控

## 性能

| 环境数 | 预生成初始化 | 实时填充缓冲 | 单帧获取 |
|-------|------------|------------|---------|
| 1024  | ~50ms      | ~20ms      | <1ms    |
| 4096  | ~200ms     | ~80ms      | <1ms    |

## 文件结构

```
CMG_Ref/utils/
├── cmg_motion_generator.py    # 核心生成器
├── command_sampler.py         # 命令采样
└── README_INTEGRATION.md      # 详细文档
```

## 文档

- [CMG_IMPLEMENTATION_SUMMARY.md](../CMG_IMPLEMENTATION_SUMMARY.md) - 实现总结
- [README_INTEGRATION.md](utils/README_INTEGRATION.md) - 完整集成指南
- [ToDo.zh.md](../ToDo.zh.md) - 项目待办事项

## 下一步

1. 在TWIST环境中集成 (见 `README_INTEGRATION.md`)
2. 实现运动格式转换 (NPZ → PKL)
3. 添加前向运动学支持
4. 测试仿真验证

---

**版本**: 1.0  
**日期**: 2026-01-30  
**状态**: 核心功能完成 ✅
