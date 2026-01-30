# TWIST（当前仓库）架构与输入说明

> 日期：2026-01-30
> 代码来源：`legged_gym/legged_gym/`
> 说明对象：`G1MimicDistill`（当前训练入口）

## 1. 总体架构（高层）
在本仓库里，TWIST 训练流程由三层组成：

1. **环境层（Isaac Gym）**
   - 负责仿真、观测、奖励与终止逻辑
   - 主要类：`LeggedRobot` → `HumanoidChar` → `HumanoidMimic` → `G1MimicDistill`

2. **策略与算法层（PPO）**
   - 配置在 `HumanoidMimicCfgPPO` / `G1MimicPrivCfgPPO`
   - Actor-Critic 结构，使用 `PPO` 训练
   - 可带“特权观测”分支（privileged obs）

3. **任务配置层（Config）**
   - 环境参数、观测维度、奖励权重、动作维度等
   - 当前使用 `legged_gym/envs/g1/g1_mimic_distill_config.py`

## 2. 动作与指令维度
来源：`G1MimicPrivCfg` + `humanoid_char_config.py`

- **动作维度**：`num_actions = 29`
  - 对应 G1 全 29 DOF
- **速度指令**：`num_commands = 3`
  - `[vx, vy, yaw_rate]`

## 3. 观测结构（G1 Mimic Distill）
观察由两部分组成：

- **Mimic 相关观测**（来自参考动作）
- **Proprioception（本体观测）**
- **Privileged Info（特权信息）**

### 3.1 Mimic 观测（基于 motion lib）
来自 `G1MimicDistill._get_mimic_obs()`：

- `mimic_obs`（学生/普通）：
  - `dof_pos` (29)
  - 合计 `num_actions`

- `priv_mimic_obs`（老师/特权）：
  - `root_z` (1)
  - `root_rpy` (3)
  - `root_vel` (3)
  - `root_yaw_rate` (1)
  - `dof_pos` (29)
  - `key_body_pos`: `num_key_bodies * 3`
  - 维度由 `n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*num_key_bodies)`

- `cmg_obs_seq`（特权新增）：
  - `dof_pos + dof_vel` across `tar_obs_steps`
  - 维度：`n_cmg_obs = len(tar_obs_steps) * (2 * num_actions)`

> 注：`mimic_obs` 是单步；`priv_mimic_obs` 与 `cmg_obs_seq` 会展开多个时间步（`tar_obs_steps`）用于 teacher/critic。

### 3.2 Proprioception 观测
来自 `G1MimicDistill.compute_observations()`：

- `base_ang_vel` (3)
- `imu`（roll, pitch）(2)
- `dof_pos` (29)
- `dof_vel` (29)
- `last_action` (29)

合计：`n_proprio = 3 + 2 + 3*num_actions = 92`

### 3.3 Privileged Info（特权信息）
来自 `G1MimicDistill.compute_observations()`：

- `base_lin_vel` (3)
- `root_height` (1)
- `key_body_pos` (num_key_bodies * 3)
- `foot_contact_mask` (2)
- `mass_params` (2)
- `friction_coeffs` (2)
- `motor_strength_delta` (2)

维度由：
```
n_priv_info = 3 + 1 + 3*num_key_bodies + 2 + 4 + 1 + 2*num_actions
```

## 4. 最终输入（obs_buf / privileged_obs_buf）
当前配置在 `G1MimicPrivCfg`：

- `obs_type = 'priv'` 时：
  - `obs_buf = priv_obs_buf`
- `obs_type = 'student'` 时：
  - `obs_buf = [mimic_obs + proprio] + history`

### 4.1 观测维度汇总（当前 G1）
来自 `g1_mimic_distill_config.py`：

- `num_actions = 29`
- `n_proprio = 3 + 2 + 3*num_actions = 92`
- `n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*num_key_bodies)`
- `n_cmg_obs = len(tar_obs_steps) * (2 * num_actions)`
- `n_priv_info = 3 + 1 + 3*num_key_bodies + 2 + 4 + 1 + 2*num_actions`
- `num_observations = n_priv_mimic_obs + n_cmg_obs + n_proprio + n_priv_info`

> 具体数值依赖 `key_bodies` 和 `tar_obs_steps` 的配置。

## 5. 训练策略网络结构（PPO）
来自 `HumanoidMimicCfgPPO`：

- Actor MLP: `[512, 256, 128]`
- Critic MLP: `[512, 256, 128]`
- Priv Encoder: `[64, 20]`
- Activation: `ELU`

## 6. 与 CMG 对齐的关键点
- CMG 输出 **关节位置+速度（58维）**，不包含 root pose/vel。
- 当前实现中，root 相关参考信息只保留在特权观测（`priv_mimic_obs`）。
- CMG 输出序列（`cmg_obs_seq`）已加入特权观测，学生观测仅保留 `dof_pos`。
  - 是替换 root 相关项？
  - 还是保留并额外拼接 CMG 输出？

如果你确定对齐方向（比如：直接替换 `mimic_obs` 为 CMG 的 58维），我可以下一步直接改环境逻辑。