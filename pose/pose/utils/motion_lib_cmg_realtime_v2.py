"""
MotionLibCMGRealtime v2: 每环境独立CMG实例 + 批量同步推理

核心设计（用户需求）：
1. 每个环境维护独立的CMG推理状态（当前动作状态）
2. 每个环境维护2秒的参考动作缓冲（100帧 @ 50fps）
3. 触发条件：环境reset或command变化或缓冲耗尽
4. 推理方式：批量同步推理所有需要更新的环境
5. 初始条件：始终以当前机器人关节角度作为CMG输入
6. 同步限制：确保所有环境同步更新，避免异步问题

关键参数：
- 4090显存限制 → 最大环境数由单次批量推理确定
- 推理频率 ≠ 模拟频率 → 需要检验
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from termcolor import cprint

def _get_project_root():
    """获取TWIST_CMG根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent

PROJECT_ROOT = _get_project_root()
CMG_REF_DIR = PROJECT_ROOT / "CMG_Ref"

for path in [str(PROJECT_ROOT), str(CMG_REF_DIR), 
             str(CMG_REF_DIR / "module"), str(CMG_REF_DIR / "utils")]:
    if path not in sys.path:
        sys.path.insert(0, path)


class MotionLibCMGRealtime:
    """
    每环境独立CMG + 2s参考缓冲 + 批量同步推理
    """
    
    SETTLE_TIME = 1.0      # 落地稳定期（秒）
    BUFFER_TIME = 2.0      # 参考缓冲时长（秒）
    
    def __init__(
        self,
        cmg_model_path: str,
        cmg_data_path: str,
        num_envs: int,
        device: str,
        dof_dim: int = 29,
        fps: float = 50.0,
    ):
        """
        初始化CMG实时库（v2架构）
        
        Args:
            cmg_model_path: CMG模型路径
            cmg_data_path: CMG数据路径
            num_envs: 环境数量
            device: PyTorch设备
            dof_dim: 关节自由度
            fps: 模拟帧率
        """
        self.device = device
        self.num_envs = num_envs
        self.dof_dim = dof_dim
        self.fps = fps
        self.dt = 1.0 / fps
        
        # 推理相关
        self.settle_frames = int(self.SETTLE_TIME * fps)  # 落地稳定帧数
        self.buffer_frames = int(self.BUFFER_TIME * fps)  # 2s缓冲帧数
        
        cprint(f"[MotionLibCMGRealtime v2] 初始化", "green")
        cprint(f"  - 环境数: {num_envs}", "green")
        cprint(f"  - DOF维度: {dof_dim}", "green")
        cprint(f"  - 帧率: {fps} Hz", "green")
        cprint(f"  - 落地稳定: {self.SETTLE_TIME}s ({self.settle_frames} frames)", "green")
        cprint(f"  - 参考缓冲: {self.BUFFER_TIME}s ({self.buffer_frames} frames)", "green")
        
        # 加载CMG模型和统计信息
        self.model, self.stats = self._load_cmg_model(cmg_model_path, cmg_data_path)
        
        # 归一化参数
        self.motion_mean = torch.from_numpy(np.ascontiguousarray(self.stats["motion_mean"])).to(device)
        self.motion_std = torch.from_numpy(np.ascontiguousarray(self.stats["motion_std"])).to(device)
        self.cmd_min = torch.from_numpy(np.ascontiguousarray(self.stats["command_min"])).to(device)
        self.cmd_max = torch.from_numpy(np.ascontiguousarray(self.stats["command_max"])).to(device)
        
        # ===== 每环境维护的状态 =====
        # 当前CMG推理状态 [num_envs, 58]
        self.current_state = torch.zeros(num_envs, 58, device=device, dtype=torch.float32)
        
        # 参考动作缓冲 [num_envs, buffer_frames, 58]
        # motion_buffer[env_id, frame_idx] = [dof_pos(29), dof_vel(29)]
        self.motion_buffer = torch.zeros(num_envs, self.buffer_frames, 58, device=device, dtype=torch.float32)
        
        # 缓冲帧索引 [num_envs] - 当前读取位置
        self.buffer_read_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # ===== 推理触发标志 =====
        # 哪些环境需要重新推理（缓冲耗尽或命令变化）
        self.needs_inference = torch.ones(num_envs, dtype=torch.bool, device=device)
        
        # ===== 环境状态 =====
        # 当前speed命令 [num_envs, 3]
        self.commands = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        
        # 落地稳定计数器 [num_envs]
        self.settle_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # 环境是否已初始化
        self.is_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # ===== 监控 =====
        self.inference_count = 0
        
        cprint(f"[MotionLibCMGRealtime v2] 初始化完成", "green")
    
    def _load_cmg_model(self, model_path: str, data_path: str):
        """加载CMG模型"""
        from module.cmg import CMG
        
        data = torch.load(data_path, weights_only=False)
        stats = data["stats"]
        
        model = CMG(
            motion_dim=stats["motion_dim"],
            command_dim=stats["command_dim"],
            hidden_dim=512,
            num_experts=4,
            num_layers=3,
        )
        
        checkpoint = torch.load(model_path, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        cprint(f"[MotionLibCMGRealtime v2] CMG模型加载成功", "green")
        return model, stats
    
    def _batch_inference(self, env_ids: torch.Tensor, init_states: torch.Tensor, 
                        commands: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        批量推理指定环境的N帧动作
        
        Args:
            env_ids: 需要推理的环境ID [M]
            init_states: 初始推理状态 [M, 58]
            commands: 速度命令 [M, 3]
            num_frames: 推理帧数
        
        Returns:
            trajectory: [M, num_frames, 58]
        """
        M = len(env_ids)
        
        # 归一化输入
        state_norm = (init_states - self.motion_mean) / self.motion_std  # [M, 58]
        cmd_norm = (commands - self.cmd_min) / (self.cmd_max - self.cmd_min) * 2 - 1  # [M, 3]
        
        # 自回归推理
        trajectory = []
        current_state = state_norm.clone()
        
        with torch.no_grad():
            for _ in range(num_frames):
                # CMG推理下一帧
                next_state_norm = self.model(current_state, cmd_norm)  # [M, 58]
                trajectory.append(next_state_norm.clone())
                current_state = next_state_norm
        
        # 反归一化 [M, num_frames, 58]
        trajectory = torch.stack(trajectory, dim=1)
        trajectory = trajectory * self.motion_std + self.motion_mean
        
        return trajectory
    
    def _infer_and_buffer(self, env_ids: torch.Tensor, current_dof_pos: torch.Tensor,
                         commands: torch.Tensor):
        """
        为指定环境推理并填充缓冲
        
        Args:
            env_ids: 环境ID [M]
            current_dof_pos: 当前关节角度 [M, dof_dim]
            commands: 速度命令 [M, 3]
        """
        M = len(env_ids)
        
        # 构建初始推理状态（当前dof_pos + 零速度或缓存速度）
        init_states = torch.zeros(M, 58, device=self.device, dtype=torch.float32)
        init_states[:, :self.dof_dim] = current_dof_pos
        # dof_vel 部分保持为0（或可用缓存的最后速度）
        
        # 批量推理2s的动作序列
        trajectory = self._batch_inference(env_ids, init_states, commands, self.buffer_frames)
        
        # 填充缓冲
        self.motion_buffer[env_ids] = trajectory
        self.buffer_read_idx[env_ids] = 0
        
        # 更新当前状态为推理的最后一帧
        self.current_state[env_ids] = trajectory[:, -1, :]
        
        cprint(f"[推理] env_ids={env_ids.tolist()} 生成了{self.buffer_frames}帧", "cyan")
        self.inference_count += 1
    
    # ===== MotionLib 兼容接口 =====
    
    def num_motions(self) -> int:
        return self.num_envs
    
    def get_total_length(self) -> float:
        return 100.0 * self.num_envs
    
    def get_motion_length(self, motion_ids) -> torch.Tensor:
        if isinstance(motion_ids, int):
            return torch.tensor(100.0, device=self.device)
        return torch.full_like(motion_ids, 100.0, dtype=torch.float)
    
    def sample_motions(self, n: int, motion_difficulty=None, curriculum_level: float = None) -> torch.Tensor:
        return torch.arange(n, device=self.device, dtype=torch.int64)
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(motion_ids), device=self.device, dtype=torch.float)
    
    def calc_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        获取参考动作帧
        
        Args:
            motion_ids: [N] 环境ID
            motion_times: [N] 时间（本版本中未使用，always从缓冲读取）
        
        Returns:
            (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos)
        """
        N = len(motion_ids)
        env_ids = motion_ids
        
        # 检查缓冲是否耗尽，需要重新推理
        buffer_depleted = self.buffer_read_idx[env_ids] >= self.buffer_frames
        needs_infer = self.needs_inference[env_ids] | buffer_depleted
        
        # 【同步批量推理】所有需要更新的环境在一起推理
        if needs_infer.any():
            infer_env_ids = env_ids[needs_infer]
            
            # 这里需要从环境侧获取当前关节角度
            # 由于接口限制，暂时使用缓存的状态（dof_pos部分）
            current_dof_pos = self.current_state[infer_env_ids, :self.dof_dim]
            current_commands = self.commands[infer_env_ids]
            
            # 处理落地稳定期
            settling_mask = self.settle_counter[infer_env_ids] > 0
            settling_commands = current_commands.clone()
            settling_commands[settling_mask] = 0.0
            
            # 执行推理
            self._infer_and_buffer(infer_env_ids, current_dof_pos, settling_commands)
            
            # 清除推理标志
            self.needs_inference[infer_env_ids] = False
        
        # 更新落地稳定计数
        still_settling = self.settle_counter[env_ids] > 0
        self.settle_counter[env_ids] = torch.clamp(self.settle_counter[env_ids] - 1, min=0)
        
        # 从缓冲读取动作
        dof_pos = torch.zeros(N, self.dof_dim, device=self.device, dtype=torch.float32)
        dof_vel = torch.zeros(N, self.dof_dim, device=self.device, dtype=torch.float32)
        
        for i, env_id in enumerate(env_ids):
            frame_idx = self.buffer_read_idx[env_id].item()
            motion_data = self.motion_buffer[env_id, frame_idx]
            dof_pos[i] = motion_data[:self.dof_dim]
            dof_vel[i] = motion_data[self.dof_dim:]
            
            # 落地稳定期覆盖速度为0
            if still_settling[i]:
                dof_vel[i] = 0.0
            
            # 推进缓冲指针
            self.buffer_read_idx[env_id] += 1
        
        # 构建返回值
        root_vel = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        root_vel[:, 0] = self.commands[env_ids, 0]
        root_vel[:, 1] = self.commands[env_ids, 1]
        
        root_ang_vel = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        root_ang_vel[:, 2] = self.commands[env_ids, 2]
        
        # 落地期间返回零速度
        root_vel[still_settling] = 0.0
        root_ang_vel[still_settling] = 0.0
        
        return None, None, root_vel, root_ang_vel, dof_pos, dof_vel, None
    
    def reset_envs(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None,
                   init_dof_pos: Optional[torch.Tensor] = None):
        """
        重置指定环境
        
        Args:
            env_ids: 要重置的环境ID [N]
            commands: 新命令 [N, 3]
            init_dof_pos: 初始关节角度 [N, dof_dim]，用于推理初始状态
        """
        N = len(env_ids)
        
        # 【防御】安全地将env_ids转为numpy，避免GPU操作
        try:
            if env_ids.device.type == 'cuda':
                env_ids_np = env_ids.detach().cpu().numpy()
            else:
                env_ids_np = env_ids.numpy()
        except:
            # 如果tensor操作失败，直接作为列表处理
            env_ids_np = list(range(N))
        
        env_ids_np = env_ids_np.astype(int)  # 确保int类型
        
        # 更新命令
        if commands is not None:
            for i, env_id in enumerate(env_ids_np):
                self.commands[int(env_id)] = commands[i]
        
        # 设置初始推理状态
        if init_dof_pos is not None:
            for i, env_id in enumerate(env_ids_np):
                self.current_state[int(env_id), :self.dof_dim] = init_dof_pos[i]
        
        # 标记需要推理
        for env_id in env_ids_np:
            self.needs_inference[int(env_id)] = True
            self.settle_counter[int(env_id)] = self.settle_frames
            self.is_initialized[int(env_id)] = True
        
        cprint(f"[Reset] env_ids={env_ids_np.tolist()} 重置，落地稳定{self.SETTLE_TIME}s", "yellow")
    
    def update_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """更新命令并触发重新推理"""
        try:
            if env_ids.device.type == 'cuda':
                env_ids_np = env_ids.detach().cpu().numpy()
            else:
                env_ids_np = env_ids.numpy()
        except:
            env_ids_np = list(range(len(env_ids)))
        
        env_ids_np = env_ids_np.astype(int)
        
        for i, env_id in enumerate(env_ids_np):
            old_cmd = self.commands[int(env_id)].clone()
            self.commands[int(env_id)] = commands[i]
            
            # 如果命令显著变化，需要重新推理
            if torch.norm(commands[i] - old_cmd) > 0.01:
                self.needs_inference[int(env_id)] = True
    
    def get_motion_command(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.commands[motion_ids]
    
    def get_motion_names(self) -> List[str]:
        return [f"cmg_realtime_{i}" for i in range(self.num_envs)]
    
    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        return {
            'inference_count': self.inference_count,
            'buffer_frames': self.buffer_frames,
        }


def create_motion_lib_cmg_realtime(cfg, num_envs: int, device: str) -> MotionLibCMGRealtime:
    """工厂函数"""
    motion_cfg = cfg.motion
    
    cmg_model_path = motion_cfg.cmg_model_path
    cmg_data_path = motion_cfg.cmg_data_path
    dof_dim = getattr(motion_cfg, 'dof_dim', 29)
    fps = getattr(motion_cfg, 'fps', 50.0)
    
    if not os.path.exists(cmg_model_path):
        raise FileNotFoundError(f"CMG模型不存在: {cmg_model_path}")
    if not os.path.exists(cmg_data_path):
        raise FileNotFoundError(f"CMG数据不存在: {cmg_data_path}")
    
    return MotionLibCMGRealtime(
        cmg_model_path=cmg_model_path,
        cmg_data_path=cmg_data_path,
        num_envs=num_envs,
        device=device,
        dof_dim=dof_dim,
        fps=fps,
    )
