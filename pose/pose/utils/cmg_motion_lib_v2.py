"""
CMG Motion Library - 使用 CMGBridge 管理参考轨迹

此模块为 HumanoidMimic 提供与 MotionLib 兼容的接口。
"""

import os
import torch
import numpy as np
from typing import Optional, List, Tuple

import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cmg_ref_dir = os.path.abspath(os.path.join(_current_dir, '..', '..', '..', 'CMG_Ref'))
if _cmg_ref_dir not in sys.path:
    sys.path.insert(0, _cmg_ref_dir)

from utils.cmg_bridge import CMGBridge, CMGBridgeConfig
from pose.utils.forward_kinematics import ForwardKinematics


class CMGMotionLib:
    """
    基于 CMGBridge 的运动库适配器
    
    提供与 MotionLib 兼容的接口，同时支持冷启动（离线）和动作衔接（在线）两种模式。
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
        初始化
        
        Args:
            cmg_model_path: CMG 模型路径
            cmg_data_path: CMG 训练数据路径
            urdf_path: URDF 路径用于 FK
            device: 计算设备
            num_envs: 并行环境数
            episode_length_s: episode 时长
            dt: 时间步长
            vx_range: 前向速度范围
            vy_range: 侧向速度范围
            yaw_range: 偏航角速度范围
            root_height: 默认根节点高度
            offline_mode: 是否启用离线模式（冷启动阶段 True，动作衔接阶段 False）
            num_trajectories: 离线模式轨迹池大小
        """
        self._device = device
        self._num_envs = num_envs
        self._episode_length_s = episode_length_s
        self._dt = dt
        self._root_height = root_height
        self._offline_mode = offline_mode
        
        # 创建桥接器配置
        cfg = CMGBridgeConfig(
            cmg_model_path=cmg_model_path,
            cmg_data_path=cmg_data_path,
            num_envs=num_envs,
            dt=dt,
            buffer_frames=100,
            output_dof=23,
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
        
        # 初始化桥接器
        self._bridge = CMGBridge(cfg, device)
        
        # 初始化 FK
        self._fk = ForwardKinematics(urdf_path, device)
        
        # Key body 列表
        self._body_link_list = [
            "left_rubber_hand", "right_rubber_hand",
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_knee_link", "right_knee_link",
            "left_elbow_link", "right_elbow_link",
            "head_mocap"
        ]
        
        # Motion IDs（兼容接口）
        self._motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._motion_times = torch.zeros(num_envs, device=device)
        
        mode_str = "离线" if offline_mode else "在线"
        print(f"[CMGMotionLib] 初始化完成: {mode_str}模式, {num_envs} 环境")
    
    # ==================== MotionLib 兼容接口 ====================
    
    def num_motions(self) -> int:
        """返回 motion 数量"""
        if self._offline_mode:
            return self._bridge.num_trajectories
        return 1000
    
    def get_motion_length(self, motion_ids) -> torch.Tensor:
        """返回 motion 时长"""
        if isinstance(motion_ids, int):
            return torch.tensor(self._episode_length_s, device=self._device)
        elif isinstance(motion_ids, torch.Tensor):
            return torch.full_like(motion_ids, self._episode_length_s, dtype=torch.float)
        return torch.tensor(self._episode_length_s, device=self._device)
    
    def get_total_length(self) -> float:
        """返回总时长"""
        return self._episode_length_s * self.num_motions()
    
    def sample_motions(self, n: int, motion_difficulty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """采样 motion ID"""
        if self._offline_mode:
            return torch.randint(0, self._bridge.num_trajectories, (n,), device=self._device)
        return torch.randint(0, 1000, (n,), device=self._device)
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """采样时间（CMG 总是从 0 开始）"""
        return torch.zeros(motion_ids.shape, device=self._device)
    
    def reset(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None):
        """
        重置指定环境
        
        Args:
            env_ids: 环境索引
            commands: 
                离线模式：轨迹索引或 None（随机分配）
                在线模式：速度指令 (n, 3)，必须提供
        """
        n = len(env_ids)
        if n == 0:
            return
        
        self._motion_times[env_ids] = 0.0
        
        if self._offline_mode:
            # 离线模式：从轨迹池分配
            self._bridge.reset(env_ids, commands)
        else:
            # 在线模式：必须提供指令
            if commands is None:
                # 兼容旧代码：如果没有提供指令，则使用配置范围采样
                commands = self._bridge.sample_commands_from_config(n)
            self._bridge.reset(env_ids, commands)
    
    def step(self, env_ids: Optional[torch.Tensor] = None):
        """推进一步"""
        self._bridge.step(env_ids)
        if env_ids is None:
            self._motion_times += self._dt
        else:
            self._motion_times[env_ids] += self._dt
    
    def calc_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        计算指定时间的运动帧
        
        支持三种查询模式：
        1. 简单查询（batch_size == num_envs）：返回所有环境的当前帧
        2. 部分查询（env_ids 提供）：返回指定环境的当前帧
        3. Tiled 查询（batch_size > num_envs）：用于 _get_mimic_obs 的未来帧查询
        """
        batch_size = motion_ids.shape[0]
        
        if batch_size == self._num_envs:
            return self._calc_current_frame()
        elif env_ids is not None and batch_size == len(env_ids) and batch_size < self._num_envs:
            return self._calc_partial_frame(env_ids)
        else:
            return self._calc_tiled_frame(motion_times)
    
    def _calc_current_frame(self) -> Tuple[torch.Tensor, ...]:
        """计算所有环境的当前帧"""
        frame = self._bridge.get_current_frame()
        
        # 计算 FK 得到 key body 位置
        local_key_body_pos = self._fk.compute_body_positions(
            frame.root_pos, frame.root_rot, frame.dof_pos
        )
        
        return (
            frame.root_pos,
            frame.root_rot,
            frame.root_vel,
            frame.root_ang_vel,
            frame.dof_pos,
            frame.dof_vel,
            local_key_body_pos,
        )
    
    def _calc_partial_frame(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """计算部分环境的当前帧"""
        frame = self._bridge.get_current_frame(env_ids)
        
        local_key_body_pos = self._fk.compute_body_positions(
            frame.root_pos, frame.root_rot, frame.dof_pos
        )
        
        return (
            frame.root_pos,
            frame.root_rot,
            frame.root_vel,
            frame.root_ang_vel,
            frame.dof_pos,
            frame.dof_vel,
            local_key_body_pos,
        )
    
    def _calc_tiled_frame(self, motion_times: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """计算 tiled 查询的帧（用于 _get_mimic_obs 的未来帧）"""
        batch_size = motion_times.shape[0]
        num_steps = batch_size // self._num_envs
        
        # 计算帧偏移量
        env_ids = torch.arange(self._num_envs, device=self._device)
        current_times = self._motion_times  # (num_envs,)
        
        # motion_times 形状是 (num_envs * num_steps,)，需要 reshape
        motion_times_2d = motion_times.reshape(self._num_envs, num_steps)  # (num_envs, num_steps)
        time_offsets = motion_times_2d - current_times.unsqueeze(1)  # (num_envs, num_steps)
        frame_offsets = (time_offsets / self._dt).long()  # (num_envs, num_steps)
        
        # 取第一个环境的帧偏移作为标准（所有环境的时间步数相同）
        frame_offsets_1d = frame_offsets[0]  # (num_steps,)
        
        # 获取未来帧
        future = self._bridge.get_future_frames(env_ids, frame_offsets_1d)
        
        # 展平结果
        dof_pos = future.dof_pos.reshape(batch_size, -1)
        dof_vel = future.dof_vel.reshape(batch_size, -1)
        root_pos = future.root_pos.reshape(batch_size, 3)
        root_rot = future.root_rot.reshape(batch_size, 4)
        root_vel = future.root_vel.reshape(batch_size, 3)
        root_ang_vel = future.root_ang_vel.reshape(batch_size, 3)
        
        # 计算 FK
        local_key_body_pos = self._fk.compute_body_positions(root_pos, root_rot, dof_pos)
        
        return (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            local_key_body_pos,
        )
    
    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        """获取 key body 索引"""
        key_body_idx = []
        for name in key_body_names:
            if name in self._body_link_list:
                key_body_idx.append(self._body_link_list.index(name))
            else:
                key_body_idx.append(self._fk.get_body_idx(name))
        return key_body_idx
    
    def get_motion_names(self) -> List[str]:
        """返回 motion 名称"""
        return ["cmg_generated"]
    
    def get_commands(self) -> torch.Tensor:
        """获取当前速度指令"""
        return self._bridge.commands.clone()
    
    def set_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """设置速度指令并更新参考轨迹（仅在线模式）"""
        if self._offline_mode:
            raise RuntimeError("离线模式下不应修改指令")
        self._bridge.update_reference(env_ids, commands)
    
    # ==================== 属性 ====================
    
    @property
    def offline_mode(self) -> bool:
        return self._offline_mode
    
    @property
    def bridge(self) -> CMGBridge:
        """获取底层桥接器（用于高级控制）"""
        return self._bridge
