"""
CMG Motion Generator for TWIST Integration
支持两种模式:
1. 预生成模式(pregenerated): 批量生成轨迹序列，用于训练冷启动
2. 实时推理模式(realtime): 自回归生成，用于动态命令跟踪

设计目标: 支持4096个并行环境的高效推理

1.1.3 新增: 前向运动学(FK)集成 - 从关节角度计算body位置和旋转
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque
import time

try:
    from fk_integration import (
        compute_body_transforms_from_dof,
        get_default_key_bodies,
        validate_fk_implementation
    )
    FK_AVAILABLE = True
except ImportError:
    FK_AVAILABLE = False


class CMGMotionGenerator:
    """
    CMG动作生成器 - 为TWIST提供参考动作
    
    两种工作模式:
    - pregenerated: 预先生成完整轨迹序列，训练时直接采样
    - realtime: 实时自回归生成，支持动态命令更新
    
    1.1.3: 集成前向运动学 - 可选计算body位置和旋转
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        num_envs: int = 4096,
        device: str = 'cuda',
        mode: str = 'pregenerated',  # 'pregenerated' or 'realtime'
        buffer_size: int = 100,      # 实时模式缓冲区大小 (帧数，约2秒@50Hz)
        preload_duration: int = 500, # 预生成模式的轨迹长度 (帧数，10秒@50Hz)
        fk_model_path: Optional[str] = None,  # 新增: FK模型路径 (URDF/XML)
        enable_fk: bool = False,  # 新增: 是否启用FK计算
    ):
        """
        Args:
            model_path: CMG模型权重路径
            data_path: 训练数据路径 (包含统计信息)
            num_envs: 并行环境数量
            device: 计算设备
            mode: 'pregenerated' 或 'realtime'
            buffer_size: 实时模式下的帧缓冲大小
            preload_duration: 预生成模式下每个轨迹的长度
            fk_model_path: FK模型路径 (URDF或XML) - 用于计算body变换
            enable_fk: 是否启用FK计算
        """
        self.device = device
        self.num_envs = num_envs
        self.mode = mode
        self.buffer_size = buffer_size
        self.preload_duration = preload_duration
        
        # 加载模型和统计信息
        self.model, self.stats, self.samples = self._load_model(model_path, data_path)
        
        # 提取统计信息到tensor (避免重复转换)
        self.motion_mean = torch.from_numpy(self.stats["motion_mean"]).to(device)
        self.motion_std = torch.from_numpy(self.stats["motion_std"]).to(device)
        self.cmd_min = torch.from_numpy(self.stats["command_min"]).to(device)
        self.cmd_max = torch.from_numpy(self.stats["command_max"]).to(device)
        
        self.motion_dim = self.stats["motion_dim"]  # 58 (29 pos + 29 vel)
        self.command_dim = self.stats["command_dim"]  # 3 (vx, vy, yaw)
        
        # 初始化FK模型 (新增 1.1.3)
        self.fk_model = None
        self.enable_fk = enable_fk and FK_AVAILABLE
        if enable_fk and fk_model_path:
            try:
                from pose.util_funcs.kinematics_model import KinematicsModel
                self.fk_model = KinematicsModel(fk_model_path, device)
                print(f"[CMGMotionGenerator] FK model loaded from {fk_model_path}")
                print(f"  - Number of joints: {self.fk_model.num_joints}")
                # 验证FK实现
                validate_fk_implementation(self.fk_model, num_test_poses=5, device=device)
            except Exception as e:
                print(f"[CMGMotionGenerator] Warning: Failed to load FK model: {e}")
                self.fk_model = None
                self.enable_fk = False
        
        # 默认key bodies (如果使用FK)
        self.key_bodies = get_default_key_bodies() if self.enable_fk else []
        
        # 初始化状态
        self.current_motion = None  # [num_envs, 58]
        self.current_commands = None  # [num_envs, 3]
        
        # 模式特定的数据结构
        if mode == 'pregenerated':
            self.trajectories = None  # [num_envs, T, 58] 预生成的轨迹
            self.trajectory_idx = None  # [num_envs] 当前帧索引
        elif mode == 'realtime':
            self.motion_buffer = [deque(maxlen=buffer_size) for _ in range(num_envs)]
            self.need_refill = torch.ones(num_envs, dtype=torch.bool, device=device)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pregenerated' or 'realtime'")
        
        
        # 性能统计
        self.inference_times = deque(maxlen=100)
        
        print(f"[CMGMotionGenerator] Initialized in '{mode}' mode")
        print(f"  - Num Envs: {num_envs}")
        print(f"  - Motion Dim: {self.motion_dim}")
        print(f"  - Device: {device}")
        if mode == 'pregenerated':
            print(f"  - Preload Duration: {preload_duration} frames ({preload_duration/50:.1f}s @ 50Hz)")
        else:
            print(f"  - Buffer Size: {buffer_size} frames ({buffer_size/50:.1f}s @ 50Hz)")
    
    def _load_model(self, model_path: str, data_path: str):
        """加载CMG模型和数据统计信息"""
        from CMG_Ref.module.cmg import CMG
        
        # 加载数据和统计信息
        data = torch.load(data_path, weights_only=False)
        stats = data["stats"]
        samples = data["samples"]
        
        # 创建模型
        model = CMG(
            motion_dim=stats["motion_dim"],
            command_dim=stats["command_dim"],
            hidden_dim=512,
            num_experts=4,
            num_layers=3,
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model, stats, samples
    
    def reset(self, env_ids: Optional[torch.Tensor] = None, 
              init_motion: Optional[torch.Tensor] = None,
              commands: Optional[torch.Tensor] = None):
        """
        重置指定环境的状态
        
        Args:
            env_ids: 要重置的环境ID [N] 或 None (重置所有)
            init_motion: 初始动作状态 [N, 58] (未归一化) 或 None (随机采样)
            commands: 速度命令 [N, 3] (未归一化) 或 None (使用当前命令)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        num_reset = len(env_ids)
        
        # 初始化动作状态
        if init_motion is None:
            # 从训练数据中随机采样初始姿态
            sample_indices = torch.randint(0, len(self.samples), (num_reset,))
            init_motion = torch.from_numpy(
                np.stack([self.samples[i][torch.randint(0, len(self.samples[i]), (1,)).item()] 
                         for i in sample_indices])
            ).float().to(self.device)
        
        if self.current_motion is None:
            self.current_motion = torch.zeros(self.num_envs, self.motion_dim, device=self.device)
        self.current_motion[env_ids] = init_motion
        
        # 初始化命令
        if commands is not None:
            if self.current_commands is None:
                self.current_commands = torch.zeros(self.num_envs, self.command_dim, device=self.device)
            self.current_commands[env_ids] = commands
        elif self.current_commands is None:
            # 默认命令: 静止
            self.current_commands = torch.zeros(self.num_envs, self.command_dim, device=self.device)
        
        # 模式特定的重置
        if self.mode == 'pregenerated':
            self._reset_pregenerated(env_ids, commands)
        elif self.mode == 'realtime':
            self._reset_realtime(env_ids)
    
    def _reset_pregenerated(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor]):
        """预生成模式: 批量生成完整轨迹"""
        num_reset = len(env_ids)
        
        # 如果没有指定命令，使用当前命令
        if commands is None:
            commands = self.current_commands[env_ids]
        
        # 批量生成轨迹 (所有帧使用相同命令)
        with torch.no_grad():
            start_time = time.time()
            
            # 重复命令到所有时间步 [N, T, 3]
            commands_seq = commands.unsqueeze(1).expand(-1, self.preload_duration, -1)
            
            # 批量自回归生成
            trajectories = self._batch_autoregressive_generation(
                self.current_motion[env_ids],
                commands_seq
            )  # [N, T+1, 58]
            
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            # 存储轨迹
            if self.trajectories is None:
                self.trajectories = torch.zeros(
                    self.num_envs, self.preload_duration + 1, self.motion_dim,
                    device=self.device
                )
                self.trajectory_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            
            self.trajectories[env_ids] = trajectories
            self.trajectory_idx[env_ids] = 0
            
            avg_time = np.mean(self.inference_times) if len(self.inference_times) > 0 else 0
            print(f"[Pregenerated] Generated {num_reset} trajectories ({self.preload_duration} frames) "
                  f"in {inference_time:.2f}ms (avg: {avg_time:.2f}ms)")
    
    def _reset_realtime(self, env_ids: torch.Tensor):
        """实时模式: 清空缓冲区，标记需要填充"""
        for env_id in env_ids.cpu().numpy():
            self.motion_buffer[env_id].clear()
        
        self.need_refill[env_ids] = True
    
    def _batch_autoregressive_generation(
        self, 
        init_motion: torch.Tensor, 
        commands: torch.Tensor
    ) -> torch.Tensor:
        """
        批量自回归生成轨迹
        
        Args:
            init_motion: [N, 58] 初始动作 (未归一化)
            commands: [N, T, 3] 命令序列 (未归一化)
        
        Returns:
            trajectories: [N, T+1, 58] 生成的轨迹 (未归一化)
        """
        N, T, _ = commands.shape
        
        # 归一化
        current = (init_motion - self.motion_mean) / self.motion_std
        commands_norm = (commands - self.cmd_min) / (self.cmd_max - self.cmd_min) * 2 - 1
        
        # 存储结果
        trajectories = [current.clone()]
        
        with torch.no_grad():
            for t in range(T):
                cmd = commands_norm[:, t, :]  # [N, 3]
                pred = self.model(current, cmd)  # [N, 58]
                current = pred
                trajectories.append(current.clone())
        
        # 合并并反归一化
        trajectories = torch.stack(trajectories, dim=1)  # [N, T+1, 58]
        trajectories = trajectories * self.motion_std + self.motion_mean
        
        return trajectories
    
    
    def get_motion(self, env_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前帧的参考动作
        
        Args:
            env_ids: 要获取的环境ID [N] 或 None (所有环境)
        
        Returns:
            dof_pos: [N, 29] 关节位置
            dof_vel: [N, 29] 关节速度
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if self.mode == 'pregenerated':
            motion = self._get_motion_pregenerated(env_ids)
        elif self.mode == 'realtime':
            motion = self._get_motion_realtime(env_ids)
        
        # 分离位置和速度
        dof_pos = motion[:, :29]
        dof_vel = motion[:, 29:]
        
        return dof_pos, dof_vel
    
    def get_motion_with_body_transforms(
        self, 
        env_ids: Optional[torch.Tensor] = None,
        base_pos: Optional[torch.Tensor] = None,
        base_rot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        新增 1.1.3: 获取当前帧的参考动作和body变换
        
        从关节角度计算身体的全局位置和旋转
        
        Args:
            env_ids: 要获取的环境ID [N] 或 None (所有环境)
            base_pos: [N, 3] 基座位置 (默认零)
            base_rot: [N, 4] 基座旋转 (默认单位四元数)
        
        Returns:
            dict包含:
                - 'dof_positions': [N, 29]
                - 'dof_velocities': [N, 29]
                - 'body_positions': [N, num_bodies, 3] (如果FK可用)
                - 'body_rotations': [N, num_bodies, 4] (如果FK可用)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 获取基础动作
        dof_pos, dof_vel = self.get_motion(env_ids)
        
        result = {
            'dof_positions': dof_pos,
            'dof_velocities': dof_vel,
        }
        
        # 计算body变换 (如果FK可用)
        if self.fk_model is not None and self.enable_fk:
            if base_pos is None:
                base_pos = torch.zeros(len(env_ids), 3, device=self.device)
            if base_rot is None:
                base_rot = torch.zeros(len(env_ids), 4, device=self.device)
                base_rot[:, 0] = 1.0  # 单位四元数
            
            try:
                body_transforms = compute_body_transforms_from_dof(
                    dof_pos, dof_vel, self.fk_model,
                    base_pos, base_rot, self.key_bodies,
                    device=self.device
                )
                result.update(body_transforms)
            except Exception as e:
                print(f"Warning: FK computation failed: {e}")
        
        return result
    
    def _get_motion_pregenerated(self, env_ids: torch.Tensor) -> torch.Tensor:
        """预生成模式: 从轨迹中索引"""
        idx = self.trajectory_idx[env_ids]
        motion = self.trajectories[env_ids, idx]  # [N, 58]
        
        # 更新索引 (循环)
        self.trajectory_idx[env_ids] = (idx + 1) % (self.preload_duration + 1)
        
        return motion
    
    def _get_motion_realtime(self, env_ids: torch.Tensor) -> torch.Tensor:
        """实时模式: 从缓冲区弹出或生成"""
        motion = torch.zeros(len(env_ids), self.motion_dim, device=self.device)
        
        # 检查哪些环境需要填充缓冲区
        refill_mask = torch.tensor(
            [len(self.motion_buffer[env_id.item()]) < self.buffer_size // 2 
             for env_id in env_ids],
            device=self.device
        )
        
        if refill_mask.any():
            refill_env_ids = env_ids[refill_mask]
            self._refill_buffer(refill_env_ids)
        
        # 从缓冲区获取动作
        for i, env_id in enumerate(env_ids.cpu().numpy()):
            if len(self.motion_buffer[env_id]) > 0:
                motion[i] = self.motion_buffer[env_id].popleft()
            else:
                # 缓冲区空: 使用当前动作
                motion[i] = self.current_motion[env_id]
        
        return motion
    
    def _refill_buffer(self, env_ids: torch.Tensor):
        """实时模式: 填充缓冲区"""
        num_refill = len(env_ids)
        refill_steps = self.buffer_size // 2  # 填充一半缓冲区
        
        start_time = time.time()
        
        # 获取当前状态和命令
        init_motion = torch.stack([
            self.motion_buffer[env_id.item()][-1] if len(self.motion_buffer[env_id.item()]) > 0 
            else self.current_motion[env_id]
            for env_id in env_ids
        ])
        
        commands = self.current_commands[env_ids].unsqueeze(1).expand(-1, refill_steps, -1)
        
        # 批量生成
        with torch.no_grad():
            new_frames = self._batch_autoregressive_generation(init_motion, commands)  # [N, T+1, 58]
        
        # 填充缓冲区 (跳过初始帧)
        for i, env_id in enumerate(env_ids.cpu().numpy()):
            for t in range(1, refill_steps + 1):
                self.motion_buffer[env_id].append(new_frames[i, t])
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        avg_time = np.mean(self.inference_times) if len(self.inference_times) > 0 else 0
        print(f"[Realtime] Refilled {num_refill} buffers ({refill_steps} frames) "
              f"in {inference_time:.2f}ms (avg: {avg_time:.2f}ms)")
    
    def update_commands(self, commands: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        更新速度命令
        
        Args:
            commands: [N, 3] 新的速度命令 (vx, vy, yaw)
            env_ids: [N] 要更新的环境ID 或 None (所有环境)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.current_commands[env_ids] = commands
        
        # 实时模式: 清空缓冲区以重新生成
        if self.mode == 'realtime':
            for env_id in env_ids.cpu().numpy():
                self.motion_buffer[env_id].clear()
            self.need_refill[env_ids] = True
    
    def get_full_motion_data(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        获取完整的动作数据 (用于TWIST观察)
        
        Args:
            env_ids: 要获取的环境ID 或 None (所有环境)
        
        Returns:
            dict with keys:
                - dof_pos: [N, 29] 关节位置
                - dof_vel: [N, 29] 关节速度
                - motion_raw: [N, 58] 原始动作向量
        """
        dof_pos, dof_vel = self.get_motion(env_ids)
        
        if env_ids is None:
            motion_raw = torch.cat([dof_pos, dof_vel], dim=-1)
        else:
            motion_raw = torch.cat([dof_pos, dof_vel], dim=-1)
        
        return {
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'motion_raw': motion_raw
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if len(self.inference_times) == 0:
            return {
                'avg_inference_ms': 0.0,
                'max_inference_ms': 0.0,
                'min_inference_ms': 0.0
            }
        
        return {
            'avg_inference_ms': float(np.mean(self.inference_times)),
            'max_inference_ms': float(np.max(self.inference_times)),
            'min_inference_ms': float(np.min(self.inference_times)),
        }
    
    def switch_mode(self, new_mode: str):
        """
        切换工作模式
        
        Args:
            new_mode: 'pregenerated' 或 'realtime'
        """
        if new_mode == self.mode:
            return
        
        print(f"[CMGMotionGenerator] Switching mode: {self.mode} -> {new_mode}")
        
        old_mode = self.mode
        self.mode = new_mode
        
        # 清理旧模式的数据结构
        if old_mode == 'pregenerated':
            self.trajectories = None
            self.trajectory_idx = None
        elif old_mode == 'realtime':
            self.motion_buffer = None
            self.need_refill = None
        
        # 初始化新模式
        if new_mode == 'pregenerated':
            self.trajectories = None
            self.trajectory_idx = None
        elif new_mode == 'realtime':
            self.motion_buffer = [deque(maxlen=self.buffer_size) for _ in range(self.num_envs)]
            self.need_refill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 重置所有环境
        self.reset()
        
        print(f"[CMGMotionGenerator] Mode switched successfully")


# ============================================================================
# 辅助函数: 命令插值和平滑
# ============================================================================

class CommandSmoother:
    """命令平滑器 - 避免速度突变"""
    
    def __init__(self, num_envs: int, interpolation_steps: int = 25, device: str = 'cuda'):
        """
        Args:
            num_envs: 环境数量
            interpolation_steps: 插值步数 (0.5秒 @ 50Hz)
            device: 计算设备
        """
        self.num_envs = num_envs
        self.interpolation_steps = interpolation_steps
        self.device = device
        
        # 当前和目标命令
        self.current_cmd = torch.zeros(num_envs, 3, device=device)
        self.target_cmd = torch.zeros(num_envs, 3, device=device)
        
        # 插值计数器
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    def set_target(self, target_cmd: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        设置新的目标命令
        
        Args:
            target_cmd: [N, 3] 目标速度命令
            env_ids: [N] 环境ID 或 None (所有环境)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.target_cmd[env_ids] = target_cmd
        self.step_counter[env_ids] = 0
    
    def get_current(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取平滑后的当前命令
        
        Args:
            env_ids: 环境ID 或 None (所有环境)
        
        Returns:
            [N, 3] 平滑后的命令
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 计算插值权重
        alpha = torch.clamp(
            self.step_counter[env_ids].float() / self.interpolation_steps,
            0.0, 1.0
        ).unsqueeze(-1)
        
        # 线性插值
        smoothed = (1 - alpha) * self.current_cmd[env_ids] + alpha * self.target_cmd[env_ids]
        
        # 更新当前命令
        self.current_cmd[env_ids] = smoothed
        self.step_counter[env_ids] += 1
        
        return smoothed
