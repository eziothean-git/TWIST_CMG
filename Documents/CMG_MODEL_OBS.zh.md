# CMG模型架构与观测维度说明（基于当前代码）

> 日期：2026-01-30
> 代码来源：`CMG_Ref/`

## 1. 模型用途与输入输出
CMG（Conditional Motion Generator）用于**基于速度指令生成参考动作**，不依赖外部 mocap。

- **输入 = 当前动作状态 + 速度指令**
- **输出 = 下一帧动作状态**
- **生成方式 = 自回归（Autoregressive）**

## 2. 观测与动作维度（真实数值）
来自 `CMG_Ref/dataloader/dataloader.py` 与 `CMG_Ref/module/cmg.py`：

### 2.1 Motion（动作状态）
- `motion_dim = 58`
- 组成：
  - 关节位置 29维（`dof_positions`）
  - 关节速度 29维（`dof_velocities`）
- 数据形状：
  - 训练样本：`motion` 形状为 `[seq_len+1, 58]`
  - 推理：单帧为 `[58]`

### 2.2 Command（速度指令）
- `command_dim = 3`
- 含义：`[vx, vy, yaw_rate]`
  - `vx`: 前向速度（m/s）
  - `vy`: 侧向速度（m/s）
  - `yaw_rate`: 角速度（rad/s）
- 数据形状：
  - 训练样本：`command` 形状为 `[seq_len, 3]`
  - 推理：单帧为 `[3]`

## 3. 归一化方式（训练与推理一致）
训练数据会在 `CMGDataset.__getitem__` 中归一化：

- Motion 归一化：
  - `(motion - motion_mean) / motion_std`
- Command 归一化：
  - `command_min/max` → 映射到 `[-1, 1]`
  - `(command - cmd_min) / (cmd_max - cmd_min) * 2 - 1`

推理时 `eval_cmg.py` 使用同样的 `stats` 做归一化和反归一化：
- 输入：先归一化
- 输出：预测结果 * `motion_std` + `motion_mean`

## 4. 模型结构（MoE MLP）
来自 `CMG_Ref/module/cmg.py`：

- **输入维度**：`motion_dim + command_dim = 61`
- **层数**：3 层 MLP
- **隐藏维度**：512
- **专家数**：4（MoE）
- **激活**：ELU（最后一层不加激活）

### 4.1 Gating Network
`CMG_Ref/module/gating_network.py`：
- 结构：`Linear(61→512) → ELU → Linear(512→4) → Softmax`
- 输出：每个 expert 的权重系数（和为1）

### 4.2 MoE Layer
`CMG_Ref/module/moe_layer.py`：
- 每层都有 4 个专家权重矩阵
- 使用 gating 的系数对专家权重进行加权融合

## 5. 自回归生成流程（推理）
`CMG_Ref/eval_cmg.py`：

1. 取初始帧 `init_motion`（58维）
2. 归一化 `init_motion` + `commands`
3. 对每个时间步：
   - 输入：`current_motion + command_t`
   - 输出：预测下一帧 motion
   - 将预测作为下一步输入（autoregressive）
4. 全序列反归一化输出

## 6. 数据与统计信息来源
训练数据保存在：
- `CMG_Ref/dataloader/cmg_training_data.pt`

其中包含：
- `samples`: 每条样本含 `motion` 与 `command`
- `stats`:
  - `motion_dim`, `command_dim`
  - `motion_mean`, `motion_std`
  - `command_min`, `command_max`

## 7. 与 TWIST 对齐时的关键信息
- CMG 只生成 **关节位置与关节速度**，不包含 root 位姿/速度。
- `command_dim = 3` 已与当前 TWIST `num_commands=3` 对齐。
- **特权观测**可考虑包含：
  - CMG 当前帧输出的 `dof_pos` / `dof_vel`（58维）
  - CMG 下一帧预测（如果需要）
  - 归一化前后要保持一致

如果你确认训练策略方向，我可以根据这个说明直接改特权观测与CMG接入逻辑。