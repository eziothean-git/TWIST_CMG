# 📚 CMG-TWIST 集成文档快速索引

> 快速导航文档和资源，方便查找特定信息。

---

## 📖 文档导航

### 🎯 开发者指南

| 目的 | 推荐文档 | 说明 |
|-----|---------|------|
| **了解项目目标和方案** | `ProjectDocumentation.md` / `ProjectDocumentation.zh.md` | 完整的项目设计文档，包含技术方案和架构 |
| **查看待办任务** | `ToDo.md` / `ToDo.zh.md` | 按阶段列出的所有任务，包括优先级和工作量 |
| **实现代码框架** | `IMPLEMENTATION_GUIDE.md` | 6 个核心功能的详细代码实现示例 |
| **快速找到文件位置** | `FILE_STRUCTURE.md` | 目录结构和文件修改位置速查表 |
| **了解集成进度** | `INTEGRATION_SUMMARY.md` / `zh.md` | 当前集成状态总结 |

---

## 🗂️ 按功能的文档导航

### 1️⃣ DOF 映射 (29 → 23)

**相关文档**：
- `IMPLEMENTATION_GUIDE.md` → 第 1 部分
- `FILE_STRUCTURE.md` → 第 1 部分
- `ToDo.md` → 第 1 阶段 → 1.1 节

**快速查询**：
```python
# 在 dof_mapping.py 中实现
class CMGToG1Mapper:
    def map_positions(cmg_pos)       # 映射位置
    def map_velocities(cmg_vel)      # 映射速度
    def map_trajectory(cmg_traj)     # 映射轨迹
```

**关键代码位置**：
```
legged_gym/legged_gym/gym_utils/dof_mapping.py
├── Line 1-60: 类定义和初始化
├── Line 61-90: 映射表建立 (TODO)
└── Line 91-150: 映射函数实现
```

---

### 2️⃣ Locomotion 奖励函数

**相关文档**：
- `IMPLEMENTATION_GUIDE.md` → 第 2 部分
- `FILE_STRUCTURE.md` → 第 2 部分
- `ToDo.md` → 第 1 阶段 → 4.2 节

**快速查询**：
```python
# 在 g1_cmg_loco_env.py 中实现
class G1CMGLoco(G1MimicDistill):
    def _reward_lin_vel_error()      # 线速度误差
    def _reward_ang_vel_error()      # 角速度误差
    def _reward_orientation_error()  # 姿态误差
    def _reward_feet_slip()          # 足滑惩罚
    def _reward_action_rate()        # 动作变化惩罚
```

**奖励权重配置**：
```
legged_gym/legged_gym/envs/g1/g1_cmg_loco_config.py
└── class G1CMGLocoConfig
    └── class rewards
        └── class scales  # 设置权重
            ├── lin_vel_error = 1.0
            ├── ang_vel_error = 0.5
            ├── orientation_error = 1.0
            ├── feet_slip = 0.1
            └── action_rate = 0.01
```

---

### 3️⃣ Teacher 特权观测

**相关文档**：
- `IMPLEMENTATION_GUIDE.md` → 第 3 部分
- `FILE_STRUCTURE.md` → 第 3 部分
- `ToDo.md` → 第 2 阶段 → 2.1 节

**快速查询**：
```python
# 在 g1_cmg_teacher_env.py 中实现
class G1CMGTeacher(G1CMGLoco):
    def _get_mimic_obs()             # 获取未来参考观测
    def _get_future_ref_obs(step_idx)# 采样单个帧
    def get_privileged_obs()         # 完整特权观测
    def _generate_cmg_reference()    # 生成参考轨迹
```

**特权观测维度**：
```
g1_cmg_teacher_config.py
└── class env
    ├── tar_obs_steps = [0, 1, 2, 3, 4, 5]    # 6 帧
    ├── n_priv_mimic_obs = 390  # 65 dims × 6 frames
    ├── n_priv_info = 50         # 动力学参数
    └── num_privileged_obs = 490 # 总维度
```

---

### 4️⃣ 残差网络模型

**相关文档**：
- `IMPLEMENTATION_GUIDE.md` → 第 4 部分
- `FILE_STRUCTURE.md` → 第 4 部分
- `ToDo.md` → 第 2 阶段 → 4 节

**快速查询**：
```python
# 在 actor_critic_residual.py 中实现
class ActorCriticResidual(ActorCritic):
    def forward(obs)                 # 输出残差 + 值
    def forward_actor(obs)           # 仅输出残差
    def forward_critic(obs)          # 仅输出值

# 完整版本
class ActorCriticResidualWithReference:
    def forward(obs, reference_action)
```

**模型输入输出**：
```
输入：
  - observation: [num_envs, num_obs]
  - reference_action: [num_envs, num_actions]  (来自 CMG)

输出：
  - residual: [num_envs, num_actions]
  - final_action = reference_action + residual_scale * residual
```

---

### 5️⃣ 平地训练

**相关文档**：
- `IMPLEMENTATION_GUIDE.md` → 第 5 部分
- `FILE_STRUCTURE.md` → 第 5 部分
- `ToDo.md` → 第 2 阶段

**快速启动**：
```bash
cd legged_gym
python scripts/train.py --task=g1_cmg_loco_flat
```

**配置参数**：
```
g1_cmg_loco_flat_config.py
├── terrain.mesh_type = 'plane'           # 平地
├── domain_rand.randomize_friction = False # 固定摩擦力
├── domain_rand.push_robots = False        # 无扰动
└── commands.ranges                        # 保守的命令范围
```

---

### 6️⃣ 复杂地形训练

**相关文档**：
- `IMPLEMENTATION_GUIDE.md` → 第 6 部分
- `FILE_STRUCTURE.md` → 第 6 部分
- `ToDo.md` → 第 5 阶段

**快速启动**（从平地 checkpoint 继续）：
```bash
python scripts/train.py --task=g1_cmg_loco_rough --load_run=<flat_checkpoint>
```

**地形类型**：
```python
# 根据难度自动选择
difficulty = 0.0  → flat (平地)
difficulty = 0.25 → low_slopes (低斜坡)
difficulty = 0.5  → stairs (楼梯)
difficulty = 0.75 → rough (粗糙地形)
difficulty = 1.0  → mixed (混合)
```

**课程学习配置**：
```
g1_cmg_loco_rough_config.py
└── class curriculum
    ├── enabled = True
    ├── curriculum_thresholds = [10k, 20k, 30k, 40k]
    └── difficulty_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

## 🔍 按问题类型的快速查询

### "我需要理解 X 是如何工作的"

| 问题 | 查看文档 | 行号 |
|-----|---------|-----|
| DOF 映射的原理 | IMPLEMENTATION_GUIDE.md | 第 1 部分 |
| 如何添加新的奖励项 | IMPLEMENTATION_GUIDE.md | 第 2 部分，_reward_*() 函数 |
| 特权观测是什么 | IMPLEMENTATION_GUIDE.md | 第 3 部分，_get_mimic_obs() |
| 残差网络如何工作 | IMPLEMENTATION_GUIDE.md | 第 4 部分，forward() |
| CMG 推理集成 | IMPLEMENTATION_GUIDE.md | 第 3 部分，_cmg_generate() |
| 地形课程学习 | IMPLEMENTATION_GUIDE.md | 第 6 部分，_generate_height_field() |

### "我需要修改 X 部分"

| 部分 | 文件位置 | 查看文档 |
|-----|---------|---------|
| CMG 映射表 | `legged_gym/gym_utils/dof_mapping.py` | IMPLEMENTATION_GUIDE.md → 1.1 |
| 奖励权重 | `legged_gym/envs/g1/g1_cmg_loco_config.py` | IMPLEMENTATION_GUIDE.md → 2.2 |
| 观测维度 | `legged_gym/envs/g1/g1_cmg_teacher_config.py` | IMPLEMENTATION_GUIDE.md → 3.2 |
| 网络架构 | `rsl_rl/modules/actor_critic_residual.py` | IMPLEMENTATION_GUIDE.md → 4.1 |
| 训练命令 | `legged_gym/scripts/train.py` | IMPLEMENTATION_GUIDE.md → 5.3 |
| 地形参数 | `legged_gym/envs/g1/g1_cmg_loco_rough_config.py` | IMPLEMENTATION_GUIDE.md → 6.1 |

### "我需要实现 X 功能"

| 功能 | 起点 | 关键文件 |
|-----|-----|---------|
| 关节映射 | IMPLEMENTATION_GUIDE.md 第 1 部分 | dof_mapping.py |
| 新奖励函数 | IMPLEMENTATION_GUIDE.md 第 2 部分 | g1_cmg_loco_env.py |
| CMG 推理接口 | IMPLEMENTATION_GUIDE.md 第 3 部分 | g1_cmg_teacher_env.py |
| 残差学习 | IMPLEMENTATION_GUIDE.md 第 4 部分 | actor_critic_residual.py |
| 课程学习 | IMPLEMENTATION_GUIDE.md 第 6 部分 | g1_cmg_loco_rough_config.py |

---

## 📋 任务优先级速查

### 🔴 高优先级（需立即实现）

1. **DOF 映射** (`dof_mapping.py`)
   - 文档：IMPLEMENTATION_GUIDE.md 第 1 部分
   - 工作量：中
   - 天数：1

2. **Locomotion 奖励** (`g1_cmg_loco_env.py` + config)
   - 文档：IMPLEMENTATION_GUIDE.md 第 2 部分
   - 工作量：中
   - 天数：1-2

3. **平地训练** (注册任务 + 配置)
   - 文档：IMPLEMENTATION_GUIDE.md 第 5 部分
   - 工作量：小
   - 天数：0.5-1

### 🟡 中优先级（第一周后实现）

4. **Teacher 特权观测** (`g1_cmg_teacher_env.py` + config)
   - 文档：IMPLEMENTATION_GUIDE.md 第 3 部分
   - 工作量：大
   - 天数：2-3

5. **残差网络** (`actor_critic_residual.py`)
   - 文档：IMPLEMENTATION_GUIDE.md 第 4 部分
   - 工作量：中
   - 天数：1-2

### 🟢 低优先级（可选优化）

6. **复杂地形** (`g1_cmg_loco_rough_config.py` + terrain)
   - 文档：IMPLEMENTATION_GUIDE.md 第 6 部分
   - 工作量：大
   - 天数：2-3

---

## 🎓 学习路径

### 新开发者入门
1. 阅读 `ProjectDocumentation.md` 了解整体架构 (15 min)
2. 查看 `FILE_STRUCTURE.md` 了解文件位置 (10 min)
3. 开始实现第 1 部分：DOF 映射 (IMPLEMENTATION_GUIDE.md 第 1 部分) (1 day)

### 环境和奖励
1. 阅读现有 `g1_mimic_distill.py` 代码 (30 min)
2. 实现第 2 部分：Locomotion 奖励 (IMPLEMENTATION_GUIDE.md 第 2 部分) (1-2 days)
3. 运行平地训练进行测试 (1-2 weeks)

### 高级功能
1. 实现第 3 部分：Teacher 特权观测 (2-3 days)
2. 实现第 4 部分：残差网络 (1-2 days)
3. 实现第 6 部分：复杂地形 (2-3 days)

---

## 🔗 交叉引用

### 同时涉及多个部分的任务

**"实现完整的 CMG 集成训练"**
- DOF 映射 (部分 1)
- 运动奖励 (部分 2)
- Teacher 特权 (部分 3)
- 残差网络 (部分 4)
- 平地训练 (部分 5)
- 复杂地形 (部分 6)

**相关文档**：
- IMPLEMENTATION_GUIDE.md (技术细节)
- FILE_STRUCTURE.md (文件组织)
- ToDo.md (任务清单)

**建议实现顺序**：1 → 2 → 5 → 3 → 4 → 6

---

## 📞 常见问题速查

### Q: 我从哪里开始？
**A**: 阅读 `ProjectDocumentation.md`，然后按照 `IMPLEMENTATION_GUIDE.md` 的顺序实现。

### Q: 如何找到特定功能的代码？
**A**: 查看 `FILE_STRUCTURE.md` 的目录结构部分。

### Q: 各个部分之间的依赖关系是什么？
**A**: 参考 `FILE_STRUCTURE.md` 的"文件间依赖关系"部分。

### Q: 如何快速检查我的实现是否正确？
**A**: 参考 `IMPLEMENTATION_GUIDE.md` 对应部分的"调试提示"。

### Q: 每个部分需要多长时间？
**A**: 查看 `FILE_STRUCTURE.md` 的"文件创建优先级"表。

### Q: 有哪些待办任务？
**A**: 参考 `ToDo.md` 按优先级和阶段列出的任务。

---

## 💾 备忘单

### 快速命令

```bash
# 训练平地
cd legged_gym && python scripts/train.py --task=g1_cmg_loco_flat

# 继续训练复杂地形
python scripts/train.py --task=g1_cmg_loco_rough --load_run=<checkpoint_path>

# 推理测试
python scripts/play.py --task=g1_cmg_loco_rough --load_run=<checkpoint_path>

# 查看任务列表
python scripts/train.py --help | grep task
```

### 关键概念

- **DOF**: 自由度 (Degrees of Freedom)
- **CMG**: 条件运动生成 (Conditional Motion Generator)
- **Teacher**: 拥有特权观测的策略网络
- **Student**: 仅通过真实观测学习的策略网络
- **Residual**: 学习在参考动作基础上的修正
- **Privilege Obs**: 训练时可用但部署时不可用的信息

---

**文档版本**：1.0  
**最后更新**：2026-01-30  
**相关文档**：
- IMPLEMENTATION_GUIDE.md (技术实现)
- FILE_STRUCTURE.md (文件结构)
- ToDo.md (任务列表)
- ProjectDocumentation.md (项目设计)
