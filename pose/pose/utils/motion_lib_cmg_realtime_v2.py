"""
MotionLibCMGRealtime v3: 简化设计，避免复杂的张量索引

核心改进：
- 用纯Python dict/list管理环境状态（buffer_idx, needs_inference等）
- GPU张量只用于批量推理的输入输出
- 完全避免混用CPU/GPU张量的索引操作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from termcolor import cprint

def _get_project_root():
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
    每环境独立CMG + 2s参考缓冲 + 纯Python状态管理
    """
    
    SETTLE_TIME = 1.0
    BUFFER_TIME = 2.0
    
    def __init__(
        self,
        cmg_model_path: str,
        cmg_data_path: str,
        num_envs: int,
        device: str,
        dof_dim: int = 29,
        fps: float = 50.0,
    ):
        self.device = device
        self.num_envs = num_envs
        self.dof_dim = dof_dim
        self.fps = fps
        self.dt = 1.0 / fps
        
        self.settle_frames = int(self.SETTLE_TIME * fps)
        self.buffer_frames = int(self.BUFFER_TIME * fps)
        self.batch_size = 128
        
        cprint(f"[MotionLibCMGRealtime v3] 初始化", "green")
        cprint(f"  - 环境数: {num_envs}", "green")
        cprint(f"  - DOF维度: {dof_dim}", "green")
        cprint(f"  - 帧率: {fps} Hz", "green")
        cprint(f"  - 参考缓冲: {self.BUFFER_TIME}s ({self.buffer_frames} frames)", "green")
        
        # 加载模型
        self.model, self.stats = self._load_cmg_model(cmg_model_path, cmg_data_path)
        
        # 归一化参数（GPU）
        self.motion_mean = torch.from_numpy(np.ascontiguousarray(self.stats["motion_mean"])).to(device)
        self.motion_std = torch.from_numpy(np.ascontiguousarray(self.stats["motion_std"])).to(device)
        self.cmd_min = torch.from_numpy(np.ascontiguousarray(self.stats["command_min"])).to(device)
        self.cmd_max = torch.from_numpy(np.ascontiguousarray(self.stats["command_max"])).to(device)
        
        # GPU张量：仅用于推理
        self.current_state = torch.zeros(num_envs, 58, device=device, dtype=torch.float32)
        self.motion_buffer = torch.zeros(num_envs, self.buffer_frames, 58, device=device, dtype=torch.float32)
        self.commands = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        
        # 纯Python管理状态（避免任何张量索引问题）
        self.buffer_read_idx = [0] * num_envs  # 缓冲读取位置
        self.needs_inference = [True] * num_envs  # 是否需要推理
        self.settle_counter = [0] * num_envs  # 落地稳定计数
        self.is_initialized = [False] * num_envs  # 是否已初始化
        
        self.inference_count = 0
        cprint(f"[MotionLibCMGRealtime v3] 初始化完成", "green")
    
    def _load_cmg_model(self, model_path: str, data_path: str):
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
        cprint(f"[MotionLibCMGRealtime v3] CMG模型加载成功", "green")
        return model, stats
    
    def _batch_inference(self, init_states: torch.Tensor, commands: torch.Tensor, 
                        num_frames: int) -> torch.Tensor:
        """批量推理（M个环境的N帧动作）"""
        M = init_states.shape[0]
        
        # 归一化
        state_norm = (init_states - self.motion_mean) / self.motion_std
        cmd_norm = (commands - self.cmd_min) / (self.cmd_max - self.cmd_min) * 2 - 1
        
        # 自回归推理
        trajectory = []
        current_state = state_norm.clone()
        
        with torch.no_grad():
            for _ in range(num_frames):
                next_state_norm = self.model(current_state, cmd_norm)
                trajectory.append(next_state_norm.clone())
                current_state = next_state_norm
        
        # 反归一化 [M, num_frames, 58]
        trajectory = torch.stack(trajectory, dim=1)
        trajectory = trajectory * self.motion_std + self.motion_mean
        return trajectory
    
    def _infer_and_buffer_batch(self, env_indices: List[int], 
                               current_dof_pos: torch.Tensor,
                               commands: torch.Tensor):
        """为一批环境推理并填充缓冲"""
        batch_size = len(env_indices)
        
        # 构建初始状态
        init_states = torch.zeros(batch_size, 58, device=self.device, dtype=torch.float32)
        init_states[:, :self.dof_dim] = current_dof_pos
        
        # 推理
        trajectory = self._batch_inference(init_states, commands, self.buffer_frames)
        
        # 逐环境填充缓冲（纯Python操作）
        for i, env_id in enumerate(env_indices):
            self.motion_buffer[env_id] = trajectory[i]
            self.buffer_read_idx[env_id] = 0
            self.current_state[env_id] = trajectory[i, -1, :]
        
        cprint(f"[推理] {batch_size} 个环境生成参考", "cyan")
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
    
    def calc_motion_frame(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> Tuple:
        """获取参考动作帧"""
        N = motion_ids.shape[0]
        
        # 转为纯Python列表（避免张量操作）
        try:
            env_ids_list = motion_ids.cpu().tolist() if motion_ids.device.type == 'cuda' else motion_ids.tolist()
        except:
            env_ids_list = list(range(N))
        
        # 检查缓冲是否耗尽，收集需要推理的环境
        envs_need_infer = []
        for env_id in env_ids_list:
            if self.needs_inference[env_id] or self.buffer_read_idx[env_id] >= self.buffer_frames:
                envs_need_infer.append(env_id)
        
        # 分批推理
        if envs_need_infer:
            num_batches = (len(envs_need_infer) + self.batch_size - 1) // self.batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(envs_need_infer))
                batch_env_ids = envs_need_infer[start:end]
                
                # 提取该批数据
                batch_dof_pos = self.current_state[batch_env_ids, :self.dof_dim]
                batch_commands = self.commands[batch_env_ids]
                
                # 处理落地稳定期
                for i, env_id in enumerate(batch_env_ids):
                    if self.settle_counter[env_id] > 0:
                        batch_commands[i] = 0.0
                
                # 推理
                self._infer_and_buffer_batch(batch_env_ids, batch_dof_pos, batch_commands)
            
            # 清除推理标志
            for env_id in envs_need_infer:
                self.needs_inference[env_id] = False
        
        # 更新落地稳定计数
        for env_id in env_ids_list:
            if self.settle_counter[env_id] > 0:
                self.settle_counter[env_id] -= 1
        
        # 从缓冲读取动作
        dof_pos = torch.zeros(N, self.dof_dim, device=self.device, dtype=torch.float32)
        dof_vel = torch.zeros(N, self.dof_dim, device=self.device, dtype=torch.float32)
        
        for i, env_id in enumerate(env_ids_list):
            frame_idx = self.buffer_read_idx[env_id]
            motion_data = self.motion_buffer[env_id, frame_idx]
            dof_pos[i] = motion_data[:self.dof_dim]
            dof_vel[i] = motion_data[self.dof_dim:]
            
            # 落地稳定期速度为0
            if self.settle_counter[env_id] > 0:
                dof_vel[i] = 0.0
            
            # 推进缓冲指针
            self.buffer_read_idx[env_id] += 1
        
        # 构建返回值
        root_vel = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        root_ang_vel = torch.zeros(N, 3, device=self.device, dtype=torch.float32)
        
        for i, env_id in enumerate(env_ids_list):
            if self.settle_counter[env_id] > 0:
                # 落地期间零速度
                root_vel[i] = 0.0
                root_ang_vel[i] = 0.0
            else:
                root_vel[i, 0] = self.commands[env_id, 0]
                root_vel[i, 1] = self.commands[env_id, 1]
                root_ang_vel[i, 2] = self.commands[env_id, 2]
        
        return None, None, root_vel, root_ang_vel, dof_pos, dof_vel, None
    
    def reset_envs(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None,
                   init_dof_pos: Optional[torch.Tensor] = None):
        """重置指定环境"""
        N = len(env_ids)
        
        # 转为纯Python列表
        try:
            env_ids_list = env_ids.cpu().tolist() if env_ids.device.type == 'cuda' else env_ids.tolist()
        except:
            env_ids_list = list(range(N))
        
        # 更新命令和状态
        for i, env_id in enumerate(env_ids_list):
            if commands is not None:
                self.commands[env_id] = commands[i]
            
            if init_dof_pos is not None:
                self.current_state[env_id, :self.dof_dim] = init_dof_pos[i]
            
            # 标记需要推理
            self.needs_inference[env_id] = True
            self.settle_counter[env_id] = self.settle_frames
            self.is_initialized[env_id] = True
        
        cprint(f"[Reset] {N} 个环境，落地稳定 {self.SETTLE_TIME}s", "yellow")
    
    def update_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """更新命令"""
        N = len(env_ids)
        try:
            env_ids_list = env_ids.cpu().tolist() if env_ids.device.type == 'cuda' else env_ids.tolist()
        except:
            env_ids_list = list(range(N))
        
        for i, env_id in enumerate(env_ids_list):
            old_cmd = self.commands[env_id].clone()
            self.commands[env_id] = commands[i]
            if torch.norm(commands[i] - old_cmd) > 0.01:
                self.needs_inference[env_id] = True
    
    def get_motion_command(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.commands[motion_ids]
    
    def get_motion_names(self) -> List[str]:
        return [f"cmg_realtime_{i}" for i in range(self.num_envs)]
    
    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        return {'inference_count': self.inference_count}


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
