# 🎯 CMG-TWIST 集成 - 快速参考卡片

> 一页纸快速参考，包含关键信息和命令。

---

## 📍 核心概念

| 术语 | 定义 | 文件 |
|-----|------|------|
| **CMG (29 DOF)** | 条件运动生成器，输出参考动作 | `CMG_Ref/` |
| **G1 (23 DOF)** | 真实机器人，有 23 个可控关节 | `legged_gym/` |
| **映射** | 29→23 DOF 的关节索引转换 | `dof_mapping.py` |
| **Locomotion** | 运动/行走控制，包含特定奖励 | `g1_cmg_loco_env.py` |
| **Teacher** | 有特权观测的主策略网络 | `g1_cmg_teacher_env.py` |
| **Student** | 无特权信息的轻量级策略 | `g1_cmg_loco_env.py` |
| **Residual** | 学习参考的修正偏差 | `actor_critic_residual.py` |

---

## 📁 必须创建的文件清单

### 第一优先级
- [ ] `legged_gym/legged_gym/gym_utils/dof_mapping.py` - DOF 映射
- [ ] `legged_gym/legged_gym/envs/g1/g1_cmg_loco_env.py` - 运动环境
- [ ] `legged_gym/legged_gym/envs/g1/g1_cmg_loco_config.py` - 运动配置
- [ ] `legged_gym/legged_gym/envs/g1/g1_cmg_loco_flat_config.py` - 平地配置

### 第二优先级
- [ ] `legged_gym/legged_gym/envs/g1/g1_cmg_teacher_env.py` - Teacher 环境
- [ ] `legged_gym/legged_gym/envs/g1/g1_cmg_teacher_config.py` - Teacher 配置
- [ ] `rsl_rl/rsl_rl/modules/actor_critic_residual.py` - 残差网络

### 第三优先级
- [ ] `legged_gym/legged_gym/envs/g1/g1_cmg_loco_rough_config.py` - 复杂地形配置

### 必须修改的文件
- [ ] `legged_gym/legged_gym/envs/__init__.py` - 注册新任务
- [ ] `legged_gym/legged_gym/gym_utils/terrain.py` - 自定义地形（可选）
- [ ] `legged_gym/scripts/train.py` - 支持新策略

---

## ⚡ 快速启动

### 环境配置
```bash
# 进入项目目录
cd /home/eziothean/TWIST_CMG

# 创建新文件时，参考 IMPLEMENTATION_GUIDE.md 中的代码模板
```

### 训练命令
```bash
# 平地训练（需先完成：dof_mapping + loco_env + loco_config + flat_config）
cd legged_gym
python scripts/train.py --task=g1_cmg_loco_flat --num_envs=4096

# 继续复杂地形（需先完成平地训练和 rough_config）
python scripts/train.py --task=g1_cmg_loco_rough --load_run=runs/g1_cmg_loco_flat/...

# 推理测试（需生成的 checkpoint）
python scripts/play.py --task=g1_cmg_loco_rough --load_run=runs/g1_cmg_loco_rough/...
```

### 调试命令
```bash
# 检查任务是否注册成功
python -c "from legged_gym.utils.task_registry import task_registry; print(task_registry.task_classes.keys())"

# 加载环境测试
python -c "
from legged_gym.utils.task_registry import task_registry
env, cfg = task_registry.make_env(name='g1_cmg_loco_flat')
print(f'Obs: {cfg.env.num_observations}, Actions: {cfg.env.num_actions}')
"
```

---

## 🔑 关键代码位置

### DOF 映射
```python
# legged_gym/legged_gym/gym_utils/dof_mapping.py
class CMGToG1Mapper:
    def _build_mapping_table(self):      # TODO: 填写索引
        mapping = np.array([...])        # 23 个索引指向 CMG 的 29 DOF
    
    def map_positions(cmg_pos):          # 用索引进行映射
        return cmg_pos[..., self.cmg_to_g1_indices]
```

### 奖励函数
```python
# legged_gym/legged_gym/envs/g1/g1_cmg_loco_env.py
class G1CMGLoco(G1MimicDistill):
    def compute_reward(self):
        r_lin_vel = self._reward_lin_vel_error()
        r_ang_vel = self._reward_ang_vel_error()
        r_orient = self._reward_orientation_error()
        r_slip = self._reward_feet_slip()
        r_rate = self._reward_action_rate()
        return w1*r_lin_vel + w2*r_ang_vel + ...  # 权重在 config 中
```

### 特权观测
```python
# legged_gym/legged_gym/envs/g1/g1_cmg_teacher_env.py
class G1CMGTeacher(G1CMGLoco):
    def get_privileged_obs(self):
        mimic_obs = self._get_mimic_obs()        # 未来 N 帧的参考
        proprio_obs = self._get_proprio_obs()    # 自身状态
        priv_info = self._get_priv_info()        # 动力学参数
        return torch.cat([mimic_obs, proprio_obs, priv_info])
```

### 残差网络
```python
# rsl_rl/rsl_rl/modules/actor_critic_residual.py
class ActorCriticResidualWithReference(nn.Module):
    def forward(self, obs, reference_action):
        residual = self.actor(torch.cat([obs, reference_action], dim=-1))
        value = self.critic(obs)
        return residual, value

# 使用方式
residual, value = model(obs, ref_action)
final_action = ref_action + residual_scale * residual
```

---

## 🎯 6 个实现目标速查

| # | 目标 | 新建文件 | 工作量 | 天数 | 起始点 |
|---|-----|---------|--------|------|--------|
| 1️⃣ | DOF 映射 | dof_mapping.py | 中 | 1 | IMPLEMENTATION_GUIDE.md 第 1 部分 |
| 2️⃣ | Loco 奖励 | g1_cmg_loco_env/config | 中 | 1-2 | IMPLEMENTATION_GUIDE.md 第 2 部分 |
| 3️⃣ | Teacher 特权 | g1_cmg_teacher_env/config | 大 | 2-3 | IMPLEMENTATION_GUIDE.md 第 3 部分 |
| 4️⃣ | 残差模型 | actor_critic_residual.py | 中 | 1-2 | IMPLEMENTATION_GUIDE.md 第 4 部分 |
| 5️⃣ | 平地训练 | g1_cmg_loco_flat_config.py | 小 | 0.5 | IMPLEMENTATION_GUIDE.md 第 5 部分 |
| 6️⃣ | 复杂地形 | g1_cmg_loco_rough_config.py | 中 | 1-2 | IMPLEMENTATION_GUIDE.md 第 6 部分 |

---

## 📊 文件依赖关系

```
1️⃣ dof_mapping.py
    ↓ 使用
2️⃣ g1_cmg_loco_env.py
    ├─ 配置 → g1_cmg_loco_config.py
    ├─ 继承 → 3️⃣ g1_cmg_teacher_env.py
    │            ├─ 配置 → g1_cmg_teacher_config.py
    │            └─ 使用 → CMG 推理
    └─ 使用 → 4️⃣ actor_critic_residual.py

5️⃣ g1_cmg_loco_flat_config.py
    继承 → g1_cmg_loco_config.py
    
6️⃣ g1_cmg_loco_rough_config.py
    继承 → g1_cmg_loco_flat_config.py
    修改 → terrain.py (自定义地形)
```

---

## 🔄 典型工作流

```
Day 1: 实现 1️⃣ DOF 映射
  ├─ 创建 dof_mapping.py
  ├─ 定义 CMGToG1Mapper 类
  └─ 测试映射是否正确

Day 2-3: 实现 2️⃣ Locomotion 环境
  ├─ 创建 g1_cmg_loco_env.py (继承 G1MimicDistill)
  ├─ 添加 5 个奖励函数
  ├─ 创建 g1_cmg_loco_config.py
  └─ 注册任务在 envs/__init__.py

Day 4: 创建 5️⃣ 平地配置
  ├─ 创建 g1_cmg_loco_flat_config.py
  ├─ 设置平地参数 (mesh_type='plane')
  └─ 禁用 domain randomization

Day 5-7: 平地训练 (迭代)
  ├─ 运行 python scripts/train.py --task=g1_cmg_loco_flat
  ├─ 监控 reward 曲线
  ├─ 调整奖励权重
  └─ 保存最好的 checkpoint

Day 8-10: 实现 3️⃣ Teacher + 4️⃣ 残差网络
  ├─ 创建 g1_cmg_teacher_env.py
  ├─ 创建 actor_critic_residual.py
  ├─ 添加 CMG 推理接口 (TODO)
  └─ Teacher 网络训练

Day 11: 创建 6️⃣ 复杂地形配置
  ├─ 创建 g1_cmg_loco_rough_config.py
  ├─ 启用 domain randomization
  ├─ 配置课程学习
  └─ 自定义地形生成

Day 12+: 复杂地形训练 (迭代)
  ├─ 从平地 checkpoint 继续训练
  ├─ 监控不同地形性能
  ├─ 调整课程参数
  └─ 最终模型评估
```

---

## 🐛 常见问题和解决方案

### Q1: 导入错误 "No module named 'legged_gym'"
**A**: 确保在 `legged_gym` 目录下运行: `cd legged_gym && python scripts/train.py ...`

### Q2: 任务未注册 "Task 'g1_cmg_loco_flat' not found"
**A**: 检查 `legged_gym/envs/__init__.py` 中是否有 `task_registry.register()`

### Q3: 观测维度不匹配错误
**A**: 检查 config 中的 `num_observations` 是否与环境的实际观测一致

### Q4: CMG 推理失败
**A**: 实现 `g1_cmg_teacher_env.py` 中的 `_cmg_generate()` 方法

### Q5: 奖励为 0
**A**: 检查 `g1_cmg_loco_config.py` 中的权重，确保至少一个权重 > 0

### Q6: 训练速度很慢
**A**: 使用 GPU (`--device=cuda`)，增加 `num_envs` (4096 推荐)

---

## 📚 文档参考

| 问题 | 查看文档 |
|------|---------|
| "我需要完整的代码示例" | IMPLEMENTATION_GUIDE.md |
| "我需要找到文件位置" | FILE_STRUCTURE.md |
| "我需要了解任务列表" | ToDo.md / ToDo.zh.md |
| "我需要快速导航" | QUICK_INDEX.md (本文档) |
| "我需要项目概览" | ProjectDocumentation.md |

---

## ✅ 验证清单

在部署到物理机器人前，检查以下各项：

### 开发阶段
- [ ] DOF 映射在小样本上通过了单元测试
- [ ] 运动环境在平地上能正常运行
- [ ] 平地训练在 1000 steps 内收敛
- [ ] Reward 曲线单调递增，无出现 NaN

### 训练阶段
- [ ] 平地训练达到预定性能目标
- [ ] Teacher 网络特权观测正确收集
- [ ] Student 网络能学习残差（loss 下降）
- [ ] 复杂地形课程逐步提高难度

### 验证阶段
- [ ] 仿真中能执行完整 episode 无崩溃
- [ ] 足接触力在合理范围内
- [ ] 关节角度未超出机械极限
- [ ] 速度命令响应正确

### 部署阶段
- [ ] 在物理 G1 上有绳行走稳定
- [ ] 关节控制延迟 < 20ms
- [ ] 电池续航 > 30 分钟
- [ ] 无意外的运动突变或抖动

---

## 🚀 成功标志

| 里程碑 | 完成标志 |
|------|---------|
| 1️⃣ 映射完成 | `map_cmg_to_g1()` 返回正确维度 |
| 2️⃣ 奖励完成 | 平地训练 return > 100 |
| 3️⃣ Teacher 完成 | priv_obs 维度与 config 一致 |
| 4️⃣ 残差完成 | Actor 输出残差，Critic 输出值 |
| 5️⃣ 平地完成 | 稳定走路，return 收敛 |
| 6️⃣ 地形完成 | 在斜坡上也能行走 |

---

**版本**: 1.0  
**更新**: 2026-01-30  
**下一步**: 打开 `IMPLEMENTATION_GUIDE.md` 开始编码！
