# CMG-TWIST 集成实现目录结构指南

本文档列出了实现 CMG-TWIST 集成所需创建/修改的所有文件及其相对位置。

---

## 📁 完整目录结构

```
TWIST_CMG/
├── IMPLEMENTATION_GUIDE.md                     # 📌 详细实现指南（本文档引用）
├── ToDo.md                                     # 英文任务列表
├── ToDo.zh.md                                  # 中文任务列表
│
└── legged_gym/
    └── legged_gym/
        ├── gym_utils/
        │   ├── __init__.py
        │   ├── helpers.py
        │   ├── logger.py
        │   ├── math.py
        │   ├── storage.py
        │   ├── terrain.py
        │   └── dof_mapping.py                  # ✨ [NEW] 关节映射脚本
        │
        ├── envs/
        │   ├── __init__.py                     # ⭐ 修改：注册新任务
        │   ├── base/
        │   │   └── ...
        │   │
        │   └── g1/
        │       ├── __init__.py
        │       ├── g1_mimic_distill.py         # 现有：学习参考
        │       ├── g1_mimic_distill_config.py  # 现有：学习参考
        │       │
        │       ├── g1_cmg_loco_env.py          # ✨ [NEW] 运动环境类
        │       ├── g1_cmg_loco_config.py       # ✨ [NEW] 运动配置
        │       │
        │       ├── g1_cmg_teacher_env.py       # ✨ [NEW] Teacher 环境
        │       ├── g1_cmg_teacher_config.py    # ✨ [NEW] Teacher 配置
        │       │
        │       ├── g1_cmg_loco_flat_config.py    # ✨ [NEW] 平地配置
        │       │
        │       └── g1_cmg_loco_rough_config.py   # ✨ [NEW] 复杂地形配置
        │
        └── scripts/
            └── train.py                        # ⭐ 修改：支持新任务
│
├── rsl_rl/
│   └── rsl_rl/
│       ├── __init__.py
│       ├── modules/
│       │   ├── __init__.py
│       │   ├── actor_critic.py                # 现有：学习参考
│       │   └── actor_critic_residual.py       # ✨ [NEW] 残差网络模块
│       │
│       ├── algorithms/
│       │   └── ...
│       │
│       └── runners/
│           └── ...
│
└── CMG_Ref/
    ├── __init__.py
    ├── utils/
    │   ├── __init__.py
    │   ├── motion_converter.py                # ⭐ 需要实现
    │   └── frame_transforms.py                # ⭐ 需要实现
    └── ...
```

---

## 📋 实现清单

### 第 1 部分：DOF 映射

| 优先级 | 文件位置 | 类型 | 说明 |
|------|---------|------|------|
| 🔴 高 | `legged_gym/legged_gym/gym_utils/dof_mapping.py` | **新建** | CMG 29→23 DOF 映射脚本 |
| 🟡 中 | `legged_gym/legged_gym/envs/g1/g1_cmg_loco_env.py` | **修改处** | 在环境初始化中调用映射 |

**关键代码段位置**：
```
dof_mapping.py
├── class CMGToG1Mapper
│   ├── __init__()
│   ├── _build_mapping_table()      # TODO: 填写实际映射索引
│   ├── map_positions()
│   ├── map_velocities()
│   └── map_trajectory()
├── get_g1_mapper()
├── map_cmg_to_g1()
└── map_cmg_to_g1_vel()
```

---

### 第 2 部分：Locomotion 奖励函数

| 优先级 | 文件位置 | 类型 | 说明 |
|------|---------|------|------|
| 🔴 高 | `legged_gym/legged_gym/envs/g1/g1_cmg_loco_env.py` | **新建** | 运动环境类，包含奖励函数 |
| 🔴 高 | `legged_gym/legged_gym/envs/g1/g1_cmg_loco_config.py` | **新建** | 运动配置，设置奖励权重 |

**关键代码段位置**：
```
g1_cmg_loco_env.py
└── class G1CMGLoco(G1MimicDistill)
    ├── compute_reward()            # 主奖励计算函数
    ├── _reward_lin_vel_error()
    ├── _reward_ang_vel_error()
    ├── _reward_orientation_error()
    ├── _reward_feet_slip()
    └── _reward_action_rate()

g1_cmg_loco_config.py
└── class G1CMGLocoConfig
    └── class rewards
        └── class scales            # 设置各项权重
```

---

### 第 3 部分：Teacher 特权观测

| 优先级 | 文件位置 | 类型 | 说明 |
|------|---------|------|------|
| 🔴 高 | `legged_gym/legged_gym/envs/g1/g1_cmg_teacher_env.py` | **新建** | Teacher 环境，提供特权观测 |
| 🔴 高 | `legged_gym/legged_gym/envs/g1/g1_cmg_teacher_config.py` | **新建** | Teacher 配置，设置观测维度 |

**关键代码段位置**：
```
g1_cmg_teacher_env.py
└── class G1CMGTeacher(G1CMGLoco)
    ├── _init_reference_buffer()
    ├── _get_mimic_obs()            # 未来参考观测
    ├── _get_future_ref_obs()       # 采样单个未来帧
    ├── get_privileged_obs()        # 完整特权观测
    ├── _get_proprio_obs()
    ├── _get_priv_info()
    ├── reset_idx()
    ├── _generate_cmg_reference()   # 生成参考轨迹
    └── _cmg_generate()             # TODO: CMG 推理

g1_cmg_teacher_config.py
└── class G1CMGTeacherConfig
    └── class env
        ├── tar_obs_steps           # 未来帧索引
        ├── n_priv_mimic_obs        # 特权观测维度
        └── n_priv_info
```

---

### 第 4 部分：残差网络模型

| 优先级 | 文件位置 | 类型 | 说明 |
|------|---------|------|------|
| 🔴 高 | `rsl_rl/rsl_rl/modules/actor_critic_residual.py` | **新建** | 残差网络模块 |
| 🟡 中 | `legged_gym/legged_gym/envs/g1/g1_cmg_student_config.py` | **新建** | 学生配置 |

**关键代码段位置**：
```
actor_critic_residual.py
├── class ActorCriticResidual(ActorCritic)
│   ├── __init__()
│   ├── forward()                   # 输出残差 + 值
│   ├── forward_actor()
│   └── forward_critic()
│
└── class ActorCriticResidualWithReference(nn.Module)
    ├── __init__()
    ├── forward(obs, reference_action)  # 参考动作作为输入
    ├── forward_actor()
    └── forward_critic()
```

---

### 第 5 部分：平地训练

| 优先级 | 文件位置 | 类型 | 说明 |
|------|---------|------|------|
| 🔴 高 | `legged_gym/legged_gym/envs/g1/g1_cmg_loco_flat_config.py` | **新建** | 平地配置 |
| 🟡 中 | `legged_gym/legged_gym/envs/__init__.py` | **修改** | 注册 `g1_cmg_loco_flat` 任务 |
| 🟡 中 | `legged_gym/scripts/train.py` | **修改** | 支持新任务加载 |

**关键代码段位置**：
```
g1_cmg_loco_flat_config.py
└── class G1CMGLocoFlatConfig(G1CMGLocoConfig)
    ├── terrain.mesh_type = 'plane'
    ├── domain_rand.randomize_friction = False
    └── domain_rand.push_robots = False

envs/__init__.py
└── task_registry.register(
        name="g1_cmg_loco_flat",
        env_class=G1CMGLoco,
        env_cfg=G1CMGLocoFlatConfig()
    )
```

---

### 第 6 部分：复杂地形训练

| 优先级 | 文件位置 | 类型 | 说明 |
|------|---------|------|------|
| 🔴 高 | `legged_gym/legged_gym/envs/g1/g1_cmg_loco_rough_config.py` | **新建** | 复杂地形配置 |
| 🟡 中 | `legged_gym/legged_gym/gym_utils/terrain.py` | **修改** | 自定义地形生成 |
| 🟡 中 | `legged_gym/legged_gym/envs/__init__.py` | **修改** | 注册 `g1_cmg_loco_rough` 任务 |

**关键代码段位置**：
```
g1_cmg_loco_rough_config.py
└── class G1CMGLocoRoughConfig
    ├── terrain.mesh_type = 'trimesh'
    ├── domain_rand.randomize_friction = True
    ├── domain_rand.push_robots = True
    └── curriculum.enabled = True

terrain.py
└── class Terrain
    ├── _create_trimesh_terrain()
    ├── _generate_height_field(difficulty)   # 难度相关地形生成
    ├── _generate_slopes()
    ├── _generate_stairs()
    └── _generate_random_terrain()

envs/__init__.py
└── task_registry.register(
        name="g1_cmg_loco_rough",
        env_class=G1CMGLoco,
        env_cfg=G1CMGLocoRoughConfig()
    )
```

---

## 🔗 文件间的依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│ dof_mapping.py                                              │
│ (CMG 29 → G1 23 DOF 映射)                                   │
└────────────────┬────────────────────────────────────────────┘
                 │ 使用
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ g1_cmg_loco_env.py                                          │
│ (基础运动环境 + 奖励函数)                                     │
└────────────┬────────────────────────────────┬───────────────┘
             │ 继承                           │ 继承
             ▼                               ▼
    ┌───────────────────┐         ┌──────────────────────┐
    │g1_cmg_teacher_env │         │ G1CMGStudentEnv      │
    │(Teacher 特权观测)  │         │ (学生残差网络)        │
    └─────────┬─────────┘         └──────────┬───────────┘
              │                              │
              │ 配置                         │ 配置
              ▼                              ▼
    ┌───────────────────┐         ┌──────────────────────┐
    │teacher_config.py  │         │student_config.py     │
    └───────────────────┘         └──────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ actor_critic_residual.py                                    │
│ (残差网络：参考动作 + 残差 = 最终动作)                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 训练配置层次                                                 │
│                                                              │
│ g1_cmg_loco_flat_config.py (基础平地)                       │
│         │ 继承                                               │
│         ▼                                                    │
│ g1_cmg_loco_rough_config.py (复杂地形 + 课程)                │
│         │ 包含                                               │
│         ▼                                                    │
│ curriculum 配置 + terrain 配置                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 修改现有文件的位置

### 1. `legged_gym/legged_gym/envs/__init__.py`

**添加位置**：在文件末尾，其他任务注册之后

```python
# 导入新的环境和配置
from legged_gym.envs.g1.g1_cmg_loco_env import G1CMGLoco
from legged_gym.envs.g1.g1_cmg_loco_config import G1CMGLocoConfig
from legged_gym.envs.g1.g1_cmg_loco_flat_config import G1CMGLocoFlatConfig
from legged_gym.envs.g1.g1_cmg_loco_rough_config import G1CMGLocoRoughConfig
from legged_gym.envs.g1.g1_cmg_teacher_env import G1CMGTeacher
from legged_gym.envs.g1.g1_cmg_teacher_config import G1CMGTeacherConfig

# 注册任务
task_registry.register(name="g1_cmg_loco_flat", env_class=G1CMGLoco, env_cfg=G1CMGLocoFlatConfig())
task_registry.register(name="g1_cmg_loco_rough", env_class=G1CMGLoco, env_cfg=G1CMGLocoRoughConfig())
task_registry.register(name="g1_cmg_teacher", env_class=G1CMGTeacher, env_cfg=G1CMGTeacherConfig())
```

### 2. `legged_gym/legged_gym/gym_utils/terrain.py`

**添加位置**：在 `Terrain` 类的现有方法之后

```python
# 在 Terrain 类中添加新方法
def _create_trimesh_terrain(self):
    """Create complex trimesh terrain."""
    # ... 实现细节见 IMPLEMENTATION_GUIDE.md

def _generate_height_field(self, difficulty):
    """Generate height field based on difficulty."""
    # ... 实现细节见 IMPLEMENTATION_GUIDE.md
```

### 3. `legged_gym/scripts/train.py`

**修改位置**：在环境和策略初始化的部分

```python
# 检查任务类型并选择合适的策略
if "teacher" in args.task:
    policy = ActorCriticMimic(...)
elif "residual" in args.task or "student" in args.task:
    policy = ActorCriticResidual(...)
else:
    policy = ActorCritic(...)
```

---

## 🚀 快速启动命令

### 查看可用任务
```bash
cd legged_gym
python scripts/train.py --help | grep task
```

### 训练平地
```bash
python scripts/train.py --task=g1_cmg_loco_flat
```

### 继续复杂地形训练
```bash
python scripts/train.py --task=g1_cmg_loco_rough --load_run=runs/g1_cmg_loco_flat/...
```

### 运行推理
```bash
python scripts/play.py --task=g1_cmg_loco_rough --load_run=runs/g1_cmg_loco_rough/...
```

---

## 📊 文件创建优先级

| 优先级 | 文件 | 工作量 | 实现天数 |
|------|------|--------|---------|
| 1️⃣ | dof_mapping.py | 中 | 1 |
| 2️⃣ | g1_cmg_loco_env.py + config | 中 | 1-2 |
| 3️⃣ | g1_cmg_loco_flat_config.py | 小 | 0.5 |
| 4️⃣ | 注册平地任务 | 小 | 0.5 |
| 5️⃣ | 平地训练 (迭代) | 大 | 3-5 |
| 6️⃣ | g1_cmg_teacher_env.py + config | 大 | 2-3 |
| 7️⃣ | actor_critic_residual.py | 中 | 1-2 |
| 8️⃣ | g1_cmg_loco_rough_config.py | 小 | 0.5 |
| 9️⃣ | terrain.py 自定义地形 | 大 | 2-3 |
| 🔟 | 复杂地形训练 (迭代) | 大 | 3-5 |

**总估计**：3-4 周（假设持续开发）

---

## 🔍 调试提示

### 1. 检查映射是否正确
```python
from legged_gym.gym_utils.dof_mapping import get_g1_mapper
mapper = get_g1_mapper()
print(f"CMG DOF: {mapper.cmg_dof}, G1 DOF: {mapper.g1_dof}")
```

### 2. 验证环境初始化
```python
from legged_gym.utils.task_registry import task_registry
env, cfg = task_registry.make_env(name="g1_cmg_loco_flat")
print(f"Num observations: {cfg.env.num_observations}")
print(f"Num actions: {cfg.env.num_actions}")
```

### 3. 监控奖励组件
```python
# 在训练脚本中记录各个奖励
wandb.log({
    "reward/lin_vel": lin_vel_reward.mean(),
    "reward/ang_vel": ang_vel_reward.mean(),
    "reward/slip": slip_reward.mean(),
})
```

---

**文档版本**：1.0  
**最后更新**：2026-01-30  
**相关文档**：IMPLEMENTATION_GUIDE.md
