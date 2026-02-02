"""
CMG Motion Library - 使用 CMGBridge 提供 MotionLib 兼容接口

本模块作为训练部分与 CMGBridge 之间的适配层，将 MotionLib 接口转换为 CMGBridge 调用。
指令采样由训练环境负责，本模块仅负责转发。
"""

import os
import sys
import importlib.util
import torch
import numpy as np
from typing import Optional, List, Tuple

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_cmg_bridge_path = os.path.join(_project_root, 'CMG_Ref', 'utils', 'cmg_bridge.py')
if not os.path.isfile(_cmg_bridge_path):
    raise FileNotFoundError(f"未找到 CMGBridge 文件: {_cmg_bridge_path}")

_spec = importlib.util.spec_from_file_location('cmg_bridge', _cmg_bridge_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载 CMGBridge 模块: {_cmg_bridge_path}")
_cmg_bridge_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cmg_bridge_module)

CMGBridge = _cmg_bridge_module.CMGBridge
CMGBridgeConfig = _cmg_bridge_module.CMGBridgeConfig

from pose.utils.forward_kinematics import ForwardKinematics


class CMGMotionLib:
    """
    CMG 运动库 - 提供与 MotionLib 兼容的接口
    
    关键设计：
    - 使用 CMGBridge 管理轨迹生成
    - 指令由训练环境传入，不在此采样
    - 支持离线和在线两种模式
    """
    
    def __init__(
        self,
        cmg_model_path: str,
        cmg_data_path: str,
        urdf_path: str,
        device: str,
        num_envs: int,
        episode_length_s: float = 10.0,
        dt: float = 0.02,
        vx_range: Tuple[float, float] = (0.5, 1.5),
        vy_range: Tuple[float, float] = (-0.3, 0.3),
        yaw_range: Tuple[float, float] = (-0.5, 0.5),
        root_height: float = 0.75,
        offline_mode: bool = True,
        num_trajectories: int = 2048,
    ):
        """
        初始化 CMG 运动库
        
        Args:
            cmg_model_path: CMG 模型路径
            cmg_data_path: CMG 训练数据路径（用于归一化统计）
            urdf_path: 机器人 URDF 路径（用于 FK 计算）
            device: 计算设备
            num_envs: 并行环境数
            episode_length_s: episode 时长
            dt: 时间步长（应与 CMG 训练一致，通常 0.02s）
            vx_range: 前向速度范围 (m/s)
            vy_range: 侧向速度范围 (m/s)
            yaw_range: 偏航角速度范围 (rad/s)
            root_height: 默认根节点高度 (m)
            offline_mode: 是否使用离线模式
            num_trajectories: 离线轨迹池大小
        """
        self._device = device
        self._num_envs = num_envs
        self._episode_length_s = episode_length_s
        self._dt = dt
        self._root_height = root_height
        self._offline_mode = offline_mode
        
        # 创建 CMGBridge
        cfg = CMGBridgeConfig(
            cmg_model_path=cmg_model_path,
            cmg_data_path=cmg_data_path,
            num_envs=num_envs,
            dt=dt,
            buffer_frames=100,  # 2s at 50Hz
            output_dof=29,  # 使用 29DOF 进行 FK 计算，输出时再裁剪为 23
            vx_range=vx_range,
            vy_range=vy_range,
            yaw_range=yaw_range,
            root_height=root_height,
            lookahead_s=2.0,
            safety_margin_s=0.5,
            offline_mode=offline_mode,
            episode_length_s=episode_length_s,
            num_trajectories=num_trajectories,
        )
        self._bridge = CMGBridge(cfg=cfg, device=device)
        
        # FK 计算器（使用 29DOF URDF）
        self._fk = ForwardKinematics(urdf_path, device)
        
        # 29DOF → 23DOF 映射索引（跳过手腕 6 DOF）
        self._dof_29_to_23_indices = torch.tensor([
            0, 1, 2, 3, 4, 5,       # 左腿 (6)
            6, 7, 8, 9, 10, 11,     # 右腿 (6)
            12, 13, 14,             # 腰部 (3)
            15, 16, 17, 18,         # 左臂 (4)
            22, 23, 24, 25,         # 右臂 (4) - 跳过左腕 19-21
        ], device=device, dtype=torch.long)
        
        # 关键身体部位列表（与 G1 配置一致）
        self._body_link_list = [
            "left_rubber_hand", "right_rubber_hand",
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_knee_link", "right_knee_link",
            "left_elbow_link", "right_elbow_link",
            "head_mocap"
        ]
        
        # motion_ids 用于接口兼容（CMG 不需要）
        self._motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        print(f"[CMGMotionLib] 初始化: {num_envs} 环境, "
              f"{'离线' if offline_mode else '在线'}模式, "
              f"vx=[{vx_range[0]:.1f}, {vx_range[1]:.1f}]")
    
    # ==================== 采样接口（供训练环境调用） ====================
    
    def sample_commands(self, n: int) -> torch.Tensor:
        """
        采样速度指令（仅离线模式预计算时使用）
        
        注意：训练时不应调用此方法，指令应由训练环境提供
        """
        return self._bridge.sample_commands_from_config(n)
    
    # ==================== MotionLib 兼容接口 ====================
    
    def num_motions(self) -> int:
        """返回运动数量（CMG 实际上无限）"""
        if self._offline_mode:
            return self._bridge.num_trajectories
        return 10000
    
    def get_motion_length(self, motion_ids) -> torch.Tensor:
        """返回运动长度"""
        if isinstance(motion_ids, int):
            return torch.tensor(self._episode_length_s, device=self._device)
        elif isinstance(motion_ids, torch.Tensor):
            return torch.full_like(motion_ids, self._episode_length_s, dtype=torch.float)
        return torch.tensor(self._episode_length_s, device=self._device)
    
    def get_total_length(self) -> float:
        """返回总运动长度"""
        return self._episode_length_s * self.num_motions()
    
    def sample_motions(self, n: int, motion_difficulty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """采样 motion IDs（CMG 中仅用于接口兼容）"""
        if self._offline_mode:
            return torch.randint(0, self._bridge.num_trajectories, (n,), device=self._device)
        return torch.randint(0, 10000, (n,), device=self._device)
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """采样运动时间（CMG 总是从 0 开始）"""
        return torch.zeros(motion_ids.shape, device=self._device)
    
    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        """获取关键身体部位索引"""
        return [self._body_link_list.index(name) if name in self._body_link_list 
                else self._fk.get_body_idx(name) for name in key_body_names]
    
    def get_motion_names(self) -> List[str]:
        """返回运动名称"""
        return ["cmg_generated"]
    
    def get_commands(self) -> torch.Tensor:
        """获取当前所有环境的速度指令"""
        return self._bridge.commands.clone()
    
    # ==================== 核心接口 ====================
    
    def reset(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None):
        """
        重置指定环境
        
        Args:
            env_ids: 环境索引
            commands: 速度指令，来自训练环境
                     在线模式：(n, 3) 速度指令，必须提供
                     离线模式：(n,) 轨迹索引 或 None（随机分配）
        """
        self._bridge.reset(env_ids, commands)
    
    def step(self, env_ids: Optional[torch.Tensor] = None):
        """推进一步"""
        self._bridge.step(env_ids)

    def _update_root_state(self, dt: float):
        """
        更新根节点状态（兼容接口）
        离线模式下无需处理，在线模式由 CMGBridge 内部维护
        """
        self._bridge.update_root_state()
        return
    
    def update_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """
        更新指令（仅在线模式）
        
        Args:
            env_ids: 环境索引
            commands: 新的速度指令 (n, 3)
        """
        if not self._offline_mode:
            self._bridge.update_reference(env_ids, commands)
    
    def calc_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算运动帧
        
        处理三种情况：
        1. batch_size == num_envs：返回所有环境当前帧
        2. env_ids 提供且 batch_size < num_envs：返回指定环境当前帧
        3. batch_size > num_envs：用于 _get_mimic_obs 的未来帧查询
        
        Returns:
            root_pos: (batch, 3)
            root_rot: (batch, 4) xyzw 格式
            root_vel: (batch, 3)
            root_ang_vel: (batch, 3)
            dof_pos: (batch, 23)
            dof_vel: (batch, 23)
            key_body_pos: (batch, 9, 3)
        """
        batch_size = motion_ids.shape[0]
        
        if batch_size == self._num_envs:
            # 返回所有环境当前帧
            return self._calc_current_frame()
        elif env_ids is not None and batch_size == len(env_ids) and batch_size < self._num_envs:
            # 返回指定环境当前帧
            return self._calc_partial_frame(env_ids)
        else:
            # 未来帧查询（用于 _get_mimic_obs）
            return self._calc_tiled_frame(motion_times)
    
    def _calc_current_frame(self) -> Tuple:
        """计算所有环境的当前帧"""
        frame = self._bridge.get_current_frame()
        
        # 计算 FK（使用完整的 29DOF）
        key_body_pos = self._fk.compute_body_positions(
            frame.root_pos, frame.root_rot, frame.dof_pos
        )
        
        # 裁剪为 23DOF 输出
        dof_pos_23 = frame.dof_pos[:, self._dof_29_to_23_indices]
        dof_vel_23 = frame.dof_vel[:, self._dof_29_to_23_indices]
        
        return (
            frame.root_pos,
            frame.root_rot,
            frame.root_vel,
            frame.root_ang_vel,
            dof_pos_23,
            dof_vel_23,
            key_body_pos,
        )
    
    def _calc_partial_frame(self, env_ids: torch.Tensor) -> Tuple:
        """计算指定环境的当前帧"""
        frame = self._bridge.get_current_frame(env_ids)
        
        # 计算 FK（使用完整的 29DOF）
        key_body_pos = self._fk.compute_body_positions(
            frame.root_pos, frame.root_rot, frame.dof_pos
        )
        
        # 裁剪为 23DOF 输出
        dof_pos_23 = frame.dof_pos[:, self._dof_29_to_23_indices]
        dof_vel_23 = frame.dof_vel[:, self._dof_29_to_23_indices]
        
        return (
            frame.root_pos,
            frame.root_rot,
            frame.root_vel,
            frame.root_ang_vel,
            dof_pos_23,
            dof_vel_23,
            key_body_pos,
        )
            frame.root_pos,
            frame.root_rot,
            frame.root_vel,
            frame.root_ang_vel,
            frame.dof_pos,
            frame.dof_vel,
            key_body_pos,
        )
    
    def _calc_tiled_frame(self, motion_times: torch.Tensor) -> Tuple:
        """
        计算未来帧（用于 _get_mimic_obs）
        
        motion_times 的形状为 (num_envs * num_steps,)，表示每个环境的多个未来时间点
        """
        batch_size = motion_times.shape[0]
        num_steps = batch_size // self._num_envs
        
        # 构建环境索引
        env_ids = torch.arange(self._num_envs, device=self._device)
        
        # 构建帧偏移：tar_obs_steps 中的步数
        # _get_mimic_obs 中 motion_times = tar_obs_steps * dt + current_time
        # 所以 frame_offsets 应该对应 tar_obs_steps
        frame_offsets = torch.arange(num_steps, device=self._device)
        
        # 获取未来帧
        future_frames = self._bridge.get_future_frames(env_ids, frame_offsets)
        
        # 展平为 (batch_size, ...) 格式
        def flatten_frame(tensor: torch.Tensor) -> torch.Tensor:
            # (num_envs, num_steps, ...) -> (num_envs * num_steps, ...)
            shape = tensor.shape
            return tensor.view(shape[0] * shape[1], *shape[2:])
        
        dof_pos = flatten_frame(future_frames.dof_pos)
        dof_vel = flatten_frame(future_frames.dof_vel)
        root_pos = flatten_frame(future_frames.root_pos)
        root_rot = flatten_frame(future_frames.root_rot)
        root_vel = flatten_frame(future_frames.root_vel)
        root_ang_vel = flatten_frame(future_frames.root_ang_vel)
        
        # 计算 FK（使用完整的 29DOF）
        key_body_pos = self._fk.compute_body_positions(root_pos, root_rot, dof_pos)
        
        # 裁剪为 23DOF 输出
        dof_pos_23 = dof_pos[:, self._dof_29_to_23_indices]
        dof_vel_23 = dof_vel[:, self._dof_29_to_23_indices]
        
        return (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos_23,
            dof_vel_23,
            key_body_pos,
        )
    
    # ==================== 属性 ====================
    
    @property
    def offline_mode(self) -> bool:
        return self._offline_mode
    
    @property
    def bridge(self) -> CMGBridge:
        return self._bridge
    
    @property
    def dt(self) -> float:
        return self._dt
