"""
CMG 桥接器模块
用于管理 CMG 自回归模型生成的参考轨迹，支持在线与离线两种模式。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass

# 添加 CMG_Ref 到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cmg_ref_dir = os.path.abspath(os.path.join(_current_dir, '..'))
if _cmg_ref_dir not in sys.path:
    sys.path.insert(0, _cmg_ref_dir)

from module.cmg import CMG


# CMG 29 DOF → G1 23 DOF 映射（跳过双腕 6 DOF）
CMG_TO_G1_INDICES = [
    0, 1, 2, 3, 4, 5,       # 左腿 (6)
    6, 7, 8, 9, 10, 11,     # 右腿 (6)
    12, 13, 14,             # 腰部 (3)
    15, 16, 17, 18,         # 左臂 (4)
    22, 23, 24, 25,         # 右臂 (4) - 跳过左腕 19-21
]


@dataclass
class CMGBridgeConfig:
    """桥接器配置"""
    cmg_model_path: str                     # CMG 模型路径
    cmg_data_path: str                      # CMG 训练数据路径（用于归一化统计）
    num_envs: int = 4096                    # 并行环境数
    dt: float = 0.02                        # 时间步长 (50Hz)
    buffer_frames: int = 100                # 在线模式缓冲帧数 (2s)
    output_dof: int = 23                    # 输出 DOF (23 或 29)
    vx_range: Tuple[float, float] = (0.5, 1.5)
    vy_range: Tuple[float, float] = (-0.3, 0.3)
    yaw_range: Tuple[float, float] = (-0.5, 0.5)
    root_height: float = 0.75               # 默认根节点高度
    # 在线模式配置
    lookahead_s: float = 2.0                # 保证前瞻余量（秒）
    safety_margin_s: float = 0.5            # 安全缓冲（秒）
    # 离线模式配置
    offline_mode: bool = False              # 是否启用离线模式
    episode_length_s: float = 10.0          # episode 时长（秒）
    num_trajectories: int = 2048             # 离线轨迹池大小


class TrajectoryFrame(NamedTuple):
    """单帧轨迹数据"""
    dof_pos: torch.Tensor           # (batch, dof)
    dof_vel: torch.Tensor           # (batch, dof)
    root_pos: torch.Tensor          # (batch, 3)
    root_rot: torch.Tensor          # (batch, 4) xyzw
    root_vel: torch.Tensor          # (batch, 3)
    root_ang_vel: torch.Tensor      # (batch, 3)


class TrajectoryBuffer(NamedTuple):
    """轨迹缓冲区数据（用于未来帧查询）"""
    dof_pos: torch.Tensor           # (batch, frames, dof)
    dof_vel: torch.Tensor           # (batch, frames, dof)
    root_pos: torch.Tensor          # (batch, frames, 3)
    root_rot: torch.Tensor          # (batch, frames, 4) xyzw
    root_vel: torch.Tensor          # (batch, frames, 3)
    root_ang_vel: torch.Tensor      # (batch, frames, 3)


class CMGBridge:
    """
    CMG 桥接器 - 管理参考轨迹生成
    
    支持两种模式：
    - 在线模式：重置时生成短缓冲（2s），缓冲耗尽时自动续生成
    - 离线模式：初始化时预生成轨迹池（完整 episode + 2s 前瞻），训练时只索引读取
    """
    
    def __init__(self, cfg: CMGBridgeConfig, device: str = "cuda"):
        """
        初始化桥接器
        
        Args:
            cfg: 桥接器配置
            device: 计算设备
        """
        self._cfg = cfg
        self._device = device
        self._num_envs = cfg.num_envs
        self._dt = cfg.dt
        self._output_dof = cfg.output_dof
        self._offline_mode = cfg.offline_mode
        
        # 加载 CMG 模型
        self._load_cmg_model(cfg.cmg_model_path, cfg.cmg_data_path)
        
        if cfg.offline_mode:
            # 离线模式：计算完整轨迹长度
            self._total_frames = int((cfg.episode_length_s + cfg.lookahead_s) / cfg.dt)
            self._num_trajectories = cfg.num_trajectories
            self._init_offline_buffers()
            self._precompute_offline_trajectories()
        else:
            # 在线模式：保证前瞻 + 安全缓冲的最小生成长度
            min_generate_frames = int((cfg.lookahead_s + cfg.safety_margin_s) / cfg.dt)
            self._buffer_frames = cfg.buffer_frames
            self._generate_frames = max(self._buffer_frames, min_generate_frames)
            self._reuse_threshold = self._buffer_frames - int(cfg.lookahead_s / cfg.dt)
            self._init_online_buffers()
        
        mode_str = "离线" if cfg.offline_mode else "在线"
        print(f"[CMGBridge] 初始化完成: {mode_str}模式, {cfg.num_envs} 环境, {cfg.output_dof} DOF 输出")
    
    def _load_cmg_model(self, model_path: str, data_path: str):
        """加载 CMG 模型与归一化统计"""
        # 加载训练数据统计
        data = torch.load(data_path, weights_only=False, map_location=self._device)
        self._stats = data["stats"]
        self._init_samples = data["samples"]
        
        # 创建模型
        self._cmg = CMG(
            motion_dim=self._stats["motion_dim"],
            command_dim=self._stats["command_dim"],
            hidden_dim=512,
            num_experts=4,
            num_layers=3,
        )
        
        # 加载权重
        ckpt = torch.load(model_path, weights_only=False, map_location=self._device)
        self._cmg.load_state_dict(ckpt["model_state_dict"])
        self._cmg = self._cmg.to(self._device)
        self._cmg.eval()
        
        # 预计算归一化张量
        self._motion_mean = torch.from_numpy(self._stats["motion_mean"]).to(self._device)
        self._motion_std = torch.from_numpy(self._stats["motion_std"]).to(self._device)
        self._cmd_min = torch.from_numpy(self._stats["command_min"]).to(self._device)
        self._cmd_max = torch.from_numpy(self._stats["command_max"]).to(self._device)
        
        print(f"[CMGBridge] 模型加载: motion_dim={self._stats['motion_dim']}, cmd_dim={self._stats['command_dim']}")
    
    def _init_online_buffers(self):
        """初始化在线模式状态缓冲区"""
        n = self._num_envs
        f = self._buffer_frames
        dof = self._output_dof
        
        # 当前运动状态（归一化）
        self._motion_norm = torch.zeros(n, self._stats["motion_dim"], device=self._device)
        
        # 轨迹缓冲区
        self._traj_dof_pos = torch.zeros(n, f, dof, device=self._device)
        self._traj_dof_vel = torch.zeros(n, f, dof, device=self._device)
        self._traj_root_pos = torch.zeros(n, f, 3, device=self._device)
        self._traj_root_rot = torch.zeros(n, f, 4, device=self._device)
        self._traj_root_rot[:, :, 3] = 1.0  # xyzw 单位四元数
        self._traj_root_vel = torch.zeros(n, f, 3, device=self._device)
        self._traj_root_ang_vel = torch.zeros(n, f, 3, device=self._device)
        
        # 缓冲区帧索引
        self._frame_idx = torch.zeros(n, dtype=torch.long, device=self._device)
        
        # 速度指令
        self._commands = torch.zeros(n, 3, device=self._device)
        
        # 根节点状态（用于在线续生成）
        self._root_pos = torch.zeros(n, 3, device=self._device)
        self._root_pos[:, 2] = self._cfg.root_height
        self._root_yaw = torch.zeros(n, device=self._device)
        
        # 标记缓冲区是否已初始化（用于检查是否需要首次生成）
        self._initialized = torch.zeros(n, dtype=torch.bool, device=self._device)
    
    def _init_offline_buffers(self):
        """初始化离线模式缓冲区"""
        n = self._num_envs
        nt = self._num_trajectories
        f = self._total_frames
        dof = self._output_dof
        
        # 轨迹池 (num_trajectories, total_frames, ...)
        self._pool_dof_pos = torch.zeros(nt, f, dof, device=self._device)
        self._pool_dof_vel = torch.zeros(nt, f, dof, device=self._device)
        self._pool_root_pos = torch.zeros(nt, f, 3, device=self._device)
        self._pool_root_rot = torch.zeros(nt, f, 4, device=self._device)
        self._pool_root_rot[:, :, 3] = 1.0
        self._pool_root_vel = torch.zeros(nt, f, 3, device=self._device)
        self._pool_root_ang_vel = torch.zeros(nt, f, 3, device=self._device)
        
        # 轨迹池对应的速度指令
        self._pool_commands = torch.zeros(nt, 3, device=self._device)
        
        # 每个环境分配的轨迹索引与帧索引
        self._env_traj_idx = torch.zeros(n, dtype=torch.long, device=self._device)
        self._frame_idx = torch.zeros(n, dtype=torch.long, device=self._device)
        
        # 环境当前使用的指令（从轨迹池复制）
        self._commands = torch.zeros(n, 3, device=self._device)
    
    @torch.no_grad()
    def _precompute_offline_trajectories(self, commands: Optional[torch.Tensor] = None):
        """预计算离线轨迹池"""
        print(f"[CMGBridge] 预计算 {self._num_trajectories} 条离线轨迹，每条 {self._total_frames} 帧...")
        
        nt = self._num_trajectories
        f = self._total_frames
        
        # 采样每条轨迹的指令（支持外部指定）
        if commands is None:
            commands = self.sample_commands(nt)
        elif len(commands) < nt:
            raise ValueError(f"提供的指令数 {len(commands)} 少于轨迹数 {nt}")
        self._pool_commands = commands[:nt]
        cmd_norm = self._normalize_cmd(self._pool_commands)
        
        # 获取初始运动状态
        init_motion = self._get_default_init_motion(nt)
        motion_norm = self._normalize_motion(init_motion)
        
        # 提取指令分量
        vx = commands[:, 0]
        vy = commands[:, 1]
        yaw_rate = commands[:, 2]
        
        # 自回归生成完整轨迹
        for frame in range(f):
            # 解码运动状态
            motion = self._denormalize_motion(motion_norm)
            dof_pos, dof_vel = self._extract_dof(motion)
            
            # 存入轨迹池
            self._pool_dof_pos[:, frame] = dof_pos
            self._pool_dof_vel[:, frame] = dof_vel
            
            # 计算根节点状态
            t = frame * self._dt
            avg_yaw = yaw_rate * t * 0.5
            new_yaw = yaw_rate * t
            cos_yaw = torch.cos(avg_yaw)
            sin_yaw = torch.sin(avg_yaw)
            
            # 根节点位置（从原点出发）
            self._pool_root_pos[:, frame, 0] = (vx * cos_yaw - vy * sin_yaw) * t
            self._pool_root_pos[:, frame, 1] = (vx * sin_yaw + vy * cos_yaw) * t
            self._pool_root_pos[:, frame, 2] = self._cfg.root_height
            
            # 根节点旋转（xyzw 格式）
            half_yaw = new_yaw * 0.5
            self._pool_root_rot[:, frame, 0] = 0.0
            self._pool_root_rot[:, frame, 1] = 0.0
            self._pool_root_rot[:, frame, 2] = torch.sin(half_yaw)
            self._pool_root_rot[:, frame, 3] = torch.cos(half_yaw)
            
            # 根节点速度
            curr_cos = torch.cos(new_yaw)
            curr_sin = torch.sin(new_yaw)
            self._pool_root_vel[:, frame, 0] = vx * curr_cos - vy * curr_sin
            self._pool_root_vel[:, frame, 1] = vx * curr_sin + vy * curr_cos
            self._pool_root_vel[:, frame, 2] = 0.0
            
            # 根节点角速度
            self._pool_root_ang_vel[:, frame, 0] = 0.0
            self._pool_root_ang_vel[:, frame, 1] = 0.0
            self._pool_root_ang_vel[:, frame, 2] = yaw_rate
            
            # CMG 前向推理下一帧
            motion_norm = self._cmg(motion_norm, cmd_norm)
            
            if (frame + 1) % 100 == 0:
                print(f"  已生成 {frame + 1}/{f} 帧")
        
        print(f"[CMGBridge] 离线轨迹池预计算完成")
    
    # ======================== 归一化 ========================
    
    def _normalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self._motion_mean) / self._motion_std
    
    def _denormalize_motion(self, motion_norm: torch.Tensor) -> torch.Tensor:
        return motion_norm * self._motion_std + self._motion_mean
    
    def _normalize_cmd(self, cmd: torch.Tensor) -> torch.Tensor:
        return (cmd - self._cmd_min) / (self._cmd_max - self._cmd_min) * 2 - 1
    
    # ======================== DOF 映射 ========================
    
    def _map_29_to_23(self, motion_29: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CMG 29 DOF → G1 23 DOF"""
        pos_29 = motion_29[..., :29]
        vel_29 = motion_29[..., 29:]
        pos_23 = pos_29[..., CMG_TO_G1_INDICES]
        vel_23 = vel_29[..., CMG_TO_G1_INDICES]
        return pos_23, vel_23
    
    def _extract_dof(self, motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 motion 提取 DOF 位置与速度"""
        if self._output_dof == 23:
            return self._map_29_to_23(motion)
        else:
            return motion[..., :29], motion[..., 29:]
    
    # ======================== 初始姿态 ========================
    
    def _get_default_init_motion(self, n: int) -> torch.Tensor:
        """获取默认站立姿态的初始运动状态"""
        # 从训练样本随机采样初始状态
        indices = np.random.randint(0, len(self._init_samples), size=n)
        init_motions = np.stack([self._init_samples[idx]["motion"][0] for idx in indices], axis=0)
        return torch.from_numpy(init_motions).float().to(self._device)
    
    # ======================== 指令采样 ========================
    
    def sample_commands(self, n: int) -> torch.Tensor:
        """采样速度指令"""
        cfg = self._cfg
        vx = torch.rand(n, device=self._device) * (cfg.vx_range[1] - cfg.vx_range[0]) + cfg.vx_range[0]
        vy = torch.rand(n, device=self._device) * (cfg.vy_range[1] - cfg.vy_range[0]) + cfg.vy_range[0]
        yaw = torch.rand(n, device=self._device) * (cfg.yaw_range[1] - cfg.yaw_range[0]) + cfg.yaw_range[0]
        return torch.stack([vx, vy, yaw], dim=-1)
    
    # ======================== 轨迹生成（在线模式） ========================
    
    @torch.no_grad()
    def _generate_trajectory_segment(self, env_ids: torch.Tensor, num_frames: int, start_frame: int = 0):
        """
        生成指定长度的轨迹段（内部函数）
        
        Args:
            env_ids: 环境索引 (n,)
            num_frames: 要生成的帧数
            start_frame: 在缓冲区中的起始写入位置
        """
        n = len(env_ids)
        if n == 0:
            return
        
        # 归一化指令
        commands = self._commands[env_ids]
        cmd_norm = self._normalize_cmd(commands)
        
        # 获取当前运动状态
        motion_norm = self._motion_norm[env_ids]
        
        # 提取指令分量
        vx = commands[:, 0]
        vy = commands[:, 1]
        yaw_rate = commands[:, 2]
        
        # 生成轨迹段
        for frame in range(num_frames):
            buf_frame = (start_frame + frame) % self._buffer_frames
            
            # 解码运动状态
            motion = self._denormalize_motion(motion_norm)
            dof_pos, dof_vel = self._extract_dof(motion)
            
            # 存入缓冲区
            self._traj_dof_pos[env_ids, buf_frame] = dof_pos
            self._traj_dof_vel[env_ids, buf_frame] = dof_vel
            
            # 计算根节点状态
            t = (start_frame + frame) * self._dt
            avg_yaw = self._root_yaw[env_ids] + yaw_rate * t * 0.5
            new_yaw = self._root_yaw[env_ids] + yaw_rate * t
            cos_yaw = torch.cos(avg_yaw)
            sin_yaw = torch.sin(avg_yaw)
            
            # 根节点位置
            self._traj_root_pos[env_ids, buf_frame, 0] = self._root_pos[env_ids, 0] + (vx * cos_yaw - vy * sin_yaw) * t
            self._traj_root_pos[env_ids, buf_frame, 1] = self._root_pos[env_ids, 1] + (vx * sin_yaw + vy * cos_yaw) * t
            self._traj_root_pos[env_ids, buf_frame, 2] = self._cfg.root_height
            
            # 根节点旋转（xyzw 格式）
            half_yaw = new_yaw * 0.5
            self._traj_root_rot[env_ids, buf_frame, 0] = 0.0
            self._traj_root_rot[env_ids, buf_frame, 1] = 0.0
            self._traj_root_rot[env_ids, buf_frame, 2] = torch.sin(half_yaw)
            self._traj_root_rot[env_ids, buf_frame, 3] = torch.cos(half_yaw)
            
            # 根节点速度
            curr_cos = torch.cos(new_yaw)
            curr_sin = torch.sin(new_yaw)
            self._traj_root_vel[env_ids, buf_frame, 0] = vx * curr_cos - vy * curr_sin
            self._traj_root_vel[env_ids, buf_frame, 1] = vx * curr_sin + vy * curr_cos
            self._traj_root_vel[env_ids, buf_frame, 2] = 0.0
            
            # 根节点角速度
            self._traj_root_ang_vel[env_ids, buf_frame, 0] = 0.0
            self._traj_root_ang_vel[env_ids, buf_frame, 1] = 0.0
            self._traj_root_ang_vel[env_ids, buf_frame, 2] = yaw_rate
            
            # CMG 前向推理下一帧
            motion_norm = self._cmg(motion_norm, cmd_norm)
        
        # 保存最终运动状态
        self._motion_norm[env_ids] = motion_norm
    
    @torch.no_grad()
    def generate_trajectory(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None):
        """
        为指定环境生成初始参考轨迹（仅在线模式）
        
        Args:
            env_ids: 环境索引 (n,)
            commands: 速度指令 (n, 3)，为 None 时随机采样
        """
        if self._offline_mode:
            raise RuntimeError("离线模式下不应调用 generate_trajectory，请使用 reset")
        
        n = len(env_ids)
        if n == 0:
            return
        
        # 采样或设置指令
        if commands is None:
            commands = self.sample_commands(n)
        self._commands[env_ids] = commands
        
        # 获取初始状态并归一化
        init_motion = self._get_default_init_motion(n)
        motion_norm = self._normalize_motion(init_motion)
        self._motion_norm[env_ids] = motion_norm
        
        # 重置根节点状态
        self._root_pos[env_ids] = 0.0
        self._root_pos[env_ids, 2] = self._cfg.root_height
        self._root_yaw[env_ids] = 0.0
        
        # 生成初始轨迹段
        self._generate_trajectory_segment(env_ids, self._generate_frames, start_frame=0)
        
        # 重置帧索引和标记
        self._frame_idx[env_ids] = 0
        self._initialized[env_ids] = True
    
    @torch.no_grad()
    def update_reference(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """
        手动更新指定环境的参考轨迹指令并重新生成
        用于处理指令变化的场景（如动作衔接阶段）
        
        Args:
            env_ids: 环境索引 (n,)
            commands: 新的速度指令 (n, 3)
        """
        if self._offline_mode:
            raise RuntimeError("离线模式下不应手动更新指令")
        
        n = len(env_ids)
        if n == 0:
            return
        
        # 更新指令
        self._commands[env_ids] = commands
        
        # 重新生成轨迹（从当前位置开始）
        current_frame = self._frame_idx[env_ids]
        
        # 获取初始状态
        init_motion = self._get_default_init_motion(n)
        motion_norm = self._normalize_motion(init_motion)
        self._motion_norm[env_ids] = motion_norm
        
        # 重置根节点状态
        self._root_pos[env_ids] = 0.0
        self._root_pos[env_ids, 2] = self._cfg.root_height
        self._root_yaw[env_ids] = 0.0
        
        # 从当前帧重新生成
        start_pos = current_frame.clamp(min=0)
        self._generate_trajectory_segment(env_ids, self._generate_frames, start_frame=start_pos)
    
    # ======================== 查询接口 ========================
    
    def get_current_frame(self, env_ids: Optional[torch.Tensor] = None) -> TrajectoryFrame:
        """
        获取当前帧轨迹数据
        
        Args:
            env_ids: 环境索引，为 None 时返回全部环境
            
        Returns:
            TrajectoryFrame 命名元组
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        
        frame_idx = self._frame_idx[env_ids]
        
        if self._offline_mode:
            # 离线模式：从轨迹池读取
            traj_idx = self._env_traj_idx[env_ids]
            frame_idx = frame_idx.clamp(0, self._total_frames - 1)
            return TrajectoryFrame(
                dof_pos=self._pool_dof_pos[traj_idx, frame_idx],
                dof_vel=self._pool_dof_vel[traj_idx, frame_idx],
                root_pos=self._pool_root_pos[traj_idx, frame_idx],
                root_rot=self._pool_root_rot[traj_idx, frame_idx],
                root_vel=self._pool_root_vel[traj_idx, frame_idx],
                root_ang_vel=self._pool_root_ang_vel[traj_idx, frame_idx],
            )
        else:
            # 在线模式：从缓冲区读取
            frame_idx = frame_idx.clamp(0, self._buffer_frames - 1)
            return TrajectoryFrame(
                dof_pos=self._traj_dof_pos[env_ids, frame_idx],
                dof_vel=self._traj_dof_vel[env_ids, frame_idx],
                root_pos=self._traj_root_pos[env_ids, frame_idx],
                root_rot=self._traj_root_rot[env_ids, frame_idx],
                root_vel=self._traj_root_vel[env_ids, frame_idx],
                root_ang_vel=self._traj_root_ang_vel[env_ids, frame_idx],
            )
    
    def get_future_frames(
        self,
        env_ids: torch.Tensor,
        frame_offsets: torch.Tensor
    ) -> TrajectoryBuffer:
        """
        获取未来多帧轨迹数据（用于教师特权观测）
        
        Args:
            env_ids: 环境索引 (n,)
            frame_offsets: 帧偏移量 (m,)，相对当前帧的步数
            
        Returns:
            TrajectoryBuffer 命名元组，形状 (n, m, ...)
        """
        n = len(env_ids)
        m = len(frame_offsets)
        
        # 计算目标帧索引
        current_idx = self._frame_idx[env_ids].unsqueeze(1)  # (n, 1)
        offsets = frame_offsets.unsqueeze(0)                  # (1, m)
        
        if self._offline_mode:
            # 离线模式：从轨迹池读取
            traj_idx = self._env_traj_idx[env_ids].unsqueeze(1).expand(n, m)  # (n, m)
            target_idx = (current_idx + offsets).clamp(0, self._total_frames - 1)  # (n, m)
            return TrajectoryBuffer(
                dof_pos=self._pool_dof_pos[traj_idx, target_idx],
                dof_vel=self._pool_dof_vel[traj_idx, target_idx],
                root_pos=self._pool_root_pos[traj_idx, target_idx],
                root_rot=self._pool_root_rot[traj_idx, target_idx],
                root_vel=self._pool_root_vel[traj_idx, target_idx],
                root_ang_vel=self._pool_root_ang_vel[traj_idx, target_idx],
            )
        else:
            # 在线模式：从缓冲区读取
            target_idx = (current_idx + offsets).clamp(0, self._buffer_frames - 1)  # (n, m)
            env_idx = env_ids.unsqueeze(1).expand(n, m)  # (n, m)
            return TrajectoryBuffer(
                dof_pos=self._traj_dof_pos[env_idx, target_idx],
                dof_vel=self._traj_dof_vel[env_idx, target_idx],
                root_pos=self._traj_root_pos[env_idx, target_idx],
                root_rot=self._traj_root_rot[env_idx, target_idx],
                root_vel=self._traj_root_vel[env_idx, target_idx],
                root_ang_vel=self._traj_root_ang_vel[env_idx, target_idx],
            )
    
    def step(self, env_ids: Optional[torch.Tensor] = None):
        """
        推进一步
        
        Args:
            env_ids: 环境索引，为 None 时推进全部环境
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        
        self._frame_idx[env_ids] += 1
        
        if self._offline_mode:
            # 离线模式：检查是否到达轨迹末尾（不含 lookahead 部分）
            episode_frames = int(self._cfg.episode_length_s / self._dt)
            needs_reset = self._frame_idx[env_ids] >= episode_frames
            if needs_reset.any():
                reset_ids = env_ids[needs_reset]
                self.reset(reset_ids)
        else:
            # 在线模式：检查是否需要续生成以保证最少 lookahead 余量
            needs_regen = self._frame_idx[env_ids] >= self._reuse_threshold
            if needs_regen.any():
                regen_ids = env_ids[needs_regen]
                # 续生成缓冲区末尾的帧
                start_pos = self._reuse_threshold
                gen_frames = self._buffer_frames - start_pos
                motion_norm = self._motion_norm[regen_ids]
                
                # 重新初始化根节点状态用于续生成
                commands = self._commands[regen_ids]
                cmd_norm = self._normalize_cmd(commands)
                vx = commands[:, 0]
                vy = commands[:, 1]
                yaw_rate = commands[:, 2]
                
                # 计算续生成的起始时间
                start_t = start_pos * self._dt
                avg_yaw_base = self._root_yaw[regen_ids] + yaw_rate * start_t * 0.5
                
                # 生成新的轨迹段（覆盖旧数据）
                for frame in range(gen_frames):
                    buf_frame = start_pos + frame
                    
                    # 解码运动状态
                    motion = self._denormalize_motion(motion_norm)
                    dof_pos, dof_vel = self._extract_dof(motion)
                    
                    # 存入缓冲区
                    self._traj_dof_pos[regen_ids, buf_frame] = dof_pos
                    self._traj_dof_vel[regen_ids, buf_frame] = dof_vel
                    
                    # 计算根节点状态
                    t = (start_pos + frame) * self._dt
                    avg_yaw = avg_yaw_base + yaw_rate * (t - start_t) * 0.5
                    new_yaw = self._root_yaw[regen_ids] + yaw_rate * t
                    cos_yaw = torch.cos(avg_yaw)
                    sin_yaw = torch.sin(avg_yaw)
                    
                    # 根节点位置
                    self._traj_root_pos[regen_ids, buf_frame, 0] = self._root_pos[regen_ids, 0] + (vx * cos_yaw - vy * sin_yaw) * t
                    self._traj_root_pos[regen_ids, buf_frame, 1] = self._root_pos[regen_ids, 1] + (vx * sin_yaw + vy * cos_yaw) * t
                    self._traj_root_pos[regen_ids, buf_frame, 2] = self._cfg.root_height
                    
                    # 根节点旋转（xyzw 格式）
                    half_yaw = new_yaw * 0.5
                    self._traj_root_rot[regen_ids, buf_frame, 0] = 0.0
                    self._traj_root_rot[regen_ids, buf_frame, 1] = 0.0
                    self._traj_root_rot[regen_ids, buf_frame, 2] = torch.sin(half_yaw)
                    self._traj_root_rot[regen_ids, buf_frame, 3] = torch.cos(half_yaw)
                    
                    # 根节点速度
                    curr_cos = torch.cos(new_yaw)
                    curr_sin = torch.sin(new_yaw)
                    self._traj_root_vel[regen_ids, buf_frame, 0] = vx * curr_cos - vy * curr_sin
                    self._traj_root_vel[regen_ids, buf_frame, 1] = vx * curr_sin + vy * curr_cos
                    self._traj_root_vel[regen_ids, buf_frame, 2] = 0.0
                    
                    # 根节点角速度
                    self._traj_root_ang_vel[regen_ids, buf_frame, 0] = 0.0
                    self._traj_root_ang_vel[regen_ids, buf_frame, 1] = 0.0
                    self._traj_root_ang_vel[regen_ids, buf_frame, 2] = yaw_rate
                    
                    # CMG 前向推理下一帧
                    motion_norm = self._cmg(motion_norm, cmd_norm)
                
                # 更新最终运动状态
                self._motion_norm[regen_ids] = motion_norm
    
    def reset(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None):
        """
        重置指定环境
        
        Args:
            env_ids: 环境索引
            commands: 速度指令。
                     在在线模式下，为 None 时随机采样；
                     在离线模式下，为索引 (n,)，用于从轨迹池选择轨迹
        """
        n = len(env_ids)
        if n == 0:
            return
            
        if self._offline_mode:
            # 离线模式：从轨迹池分配
            if commands is None:
                # 随机分配
                traj_idx = torch.randint(0, self._num_trajectories, (n,), device=self._device)
            else:
                # 使用指定的轨迹索引（支持课程学习）
                if not isinstance(commands, torch.Tensor):
                    commands = torch.tensor(commands, dtype=torch.long, device=self._device)
                traj_idx = commands.long().to(self._device)
                if traj_idx.max() >= self._num_trajectories or traj_idx.min() < 0:
                    raise ValueError(f"轨迹索引超出范围 [0, {self._num_trajectories})")
            
            self._env_traj_idx[env_ids] = traj_idx
            self._commands[env_ids] = self._pool_commands[traj_idx]
            self._frame_idx[env_ids] = 0
        else:
            # 在线模式：生成新轨迹
            self.generate_trajectory(env_ids, commands)
    
    # ======================== 属性 ========================
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def buffer_frames(self) -> int:
        """在线模式缓冲区帧数"""
        return self._buffer_frames
    
    @property
    def total_frames(self) -> int:
        """离线模式总帧数（含 lookahead）"""
        return self._total_frames if self._offline_mode else self._buffer_frames
    
    @property
    def output_dof(self) -> int:
        return self._output_dof
    
    @property
    def commands(self) -> torch.Tensor:
        """当前速度指令 (num_envs, 3)"""
        return self._commands
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def offline_mode(self) -> bool:
        return self._offline_mode
    
    @property
    def num_trajectories(self) -> int:
        """离线模式轨迹池大小"""
        return self._num_trajectories if self._offline_mode else 0
    
    @property
    def pool_commands(self) -> Optional[torch.Tensor]:
        """离线模式轨迹池指令 (num_trajectories, 3)"""
        return self._pool_commands if self._offline_mode else None
