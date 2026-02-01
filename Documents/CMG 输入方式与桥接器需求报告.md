# CMG 输入方式与桥接器需求报告

**日期**：2026-02-01

## 资料来源
- `CMG_Ref/README.md`
- `cmg_integration.md`
- `Documents/FILE_STRUCTURE.md`
- `.github/instructions/开发环境说明.instructions.md`
- `CHANGELOG_CMG_Integration.md`

## CMG 模型输入输出约定
- 输入由两部分组成：
  - 当前运动状态（关节位置 + 关节速度），维度为 58
  - 速度指令（$v_x$, $v_y$, yaw），维度为 3
- 输出为下一帧的运动状态（关节位置 + 关节速度），维度为 58
- 输入拼接顺序为 `motion` 后接 `command`，即 $[motion, command]$
- 生成方式为自回归，持续输出未来序列

## 移植项目中 CMG 接入流程要点
- 训练/运行由速度指令驱动，核心指令为 $v_x$, $v_y$, yaw
- 初始化阶段加载 CMG 模型与归一化统计，并以默认站立姿态作为参考起点
- 重置阶段采样速度指令并预生成 2 秒参考轨迹缓冲（100 帧，步长 $dt=0.02$）
- 步进阶段仅推进缓冲区索引，必要时触发重新生成
- 查询阶段支持当前帧与未来帧，用于 mimic 观测

## 教师特权观测与未来参考轨迹
- 教师模型使用未来 2 秒参考轨迹作为特权观测输入
- 未来轨迹采样由 `g1_mimic_distill.py` 中 `_get_mimic_obs()` 完成
- 采样步长来自 `g1_mimic_distill_config.py` 的 `tar_obs_steps`（对应 $dt=0.02$，覆盖约 1.9 s）
- 参考轨迹初始姿态与训练初始姿态一致，来自 `G1MimicPrivCfg.init_state.default_joint_angles`

## FK 进入训练观测的路径
- FK 计算入口：`pose/utils/cmg_motion_lib.py` 内部创建 `ForwardKinematics`
- 计算位置：`calc_motion_frame()` 返回 `local_key_body_pos`
- 训练使用位置：`g1_mimic_distill.py` 的 `_get_mimic_obs()` 将 `body_pos` 作为关键体位置拼接到 `priv_mimic_obs`
- 参考缓冲同步：`_reset_ref_motion()` 与 `_update_ref_motion()` 将关键体位置写入 `_ref_body_pos`

## 速度指令范围约定
- Slow：$v_x$ 0.5–1.5，$v_y$ -0.3–0.3，yaw -0.5–0.5
- Medium：$v_x$ 1.5–2.5，$v_y$ -0.5–0.5，yaw -0.8–0.8
- Fast：$v_x$ 2.5–3.5，$v_y$ -0.5–0.5，yaw -1.0–1.0

## DOF 处理要求
- CMG 输出为 29 DOF，G1 训练使用 23 DOF
- 需要在桥接器中支持两种输出模式：
  - 29 DOF 直出，用于对比测试
  - 23 DOF 裁剪输出，跳过双腕 6 DOF

## 桥接器模块的输入与响应建议
- 输入建议包含：
  - 速度指令（$v_x$, $v_y$, yaw）
  - 当前关节位置与速度
  - 输出模式（29 DOF 或 23 DOF）
  - 目标环境索引集合（支持 4096 并行环境）
- 输出建议包含：
  - 未来 2 秒参考轨迹（位置/速度/根节点状态）
  - 关键体位置（9 个关键体）

## 在线与离线模式的桥接职责
- 在线模式：
  - 每步接收速度指令与关节状态
  - 基于当前状态进行自回归推理，生成未来轨迹
  - 与环境保持指令一致性，保证动作对齐
- 离线模式：
  - 以默认站立姿态为初始条件预先推理参考轨迹
  - 预生成的轨迹可直接用于教师特权观测的未来 2 秒片段
  - 速度指令从轨迹池对应的指令采样
  - 面向冷启动阶段提高采样效率
