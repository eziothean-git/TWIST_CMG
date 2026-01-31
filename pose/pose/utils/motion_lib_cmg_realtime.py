"""
MotionLibCMGRealtime: CMG 实时在环生成参考动作

核心设计:
1. 实时自回归生成，每一步调用 CMG 模型
2. 落地稳定期（前 1 秒）给零速度命令
3. 之后根据 curriculum_level 给出实际速度命令

关键优化:
- 批量推理：一次推理 num_envs 个环境
- 缓冲机制：预生成若干帧，减少推理频率
- 落地稳定：确保机器人落地后再开始跟踪运动轨迹
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from termcolor import cprint

# 添加项目路径
def _get_project_root():
    """获取TWIST_CMG根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent

PROJECT_ROOT = _get_project_root()
CMG_REF_DIR = PROJECT_ROOT / "CMG_Ref"

# 确保路径添加到sys.path
for path in [str(PROJECT_ROOT), str(CMG_REF_DIR), 
             str(CMG_REF_DIR / "module"), str(CMG_REF_DIR / "utils")]:
    if path not in sys.path:
        sys.path.insert(0, path)


class MotionLibCMGRealtime:
    """
    CMG 实时在环动作库
    
    与预生成模式的区别:
    1. 每次 calc_motion_frame 调用时实时生成
    2. 支持动态更改命令（不需要重新采样 motion）
    3. 落地稳定期机制：前 SETTLE_TIME 秒给零速度
    
    接口与 MotionLibCMG 保持一致
    """
    
    # 落地稳定时间（秒）
    SETTLE_TIME = 1.0
    
    def __init__(
        self, 
        cmg_model_path: str,
        cmg_data_path: str,
        num_envs: int,
        device: str,
        dof_dim: int = 29,
        fps: float = 50.0,
        buffer_frames: int = 10,  # 缓冲帧数，减少推理频率
    ):
        """
        初始化 CMG 实时动作库
        
        Args:
            cmg_model_path: CMG 模型权重路径
            cmg_data_path: CMG 训练数据路径
            num_envs: 并行环境数量
            device: PyTorch 设备
            dof_dim: 关节自由度数量
            fps: 动作帧率
            buffer_frames: 缓冲帧数
        """
        self.device = device
        self.num_envs = num_envs
        self.dof_dim = dof_dim
        self.fps = fps
        self.dt = 1.0 / fps
        self.buffer_frames = buffer_frames
        self.settle_frames = int(self.SETTLE_TIME * fps)  # 落地稳定帧数
        
        cprint(f"[MotionLibCMGRealtime] 初始化实时动作库", "green")
        cprint(f"  - 环境数: {num_envs}", "green")
        cprint(f"  - DOF维度: {dof_dim}", "green")
        cprint(f"  - 帧率: {fps} Hz", "green")
        cprint(f"  - 落地稳定时间: {self.SETTLE_TIME}s ({self.settle_frames} frames)", "green")
        cprint(f"  - 缓冲帧数: {buffer_frames}", "green")
        
        # 加载 CMG 模型
        self.model, self.stats, self.samples = self._load_cmg_model(
            cmg_model_path, cmg_data_path
        )
        
        # 提取统计信息（确保数组是C连续的）
        self.motion_mean = torch.from_numpy(np.ascontiguousarray(self.stats["motion_mean"])).to(device)
        self.motion_std = torch.from_numpy(np.ascontiguousarray(self.stats["motion_std"])).to(device)
        self.cmd_min = torch.from_numpy(np.ascontiguousarray(self.stats["command_min"])).to(device)
        self.cmd_max = torch.from_numpy(np.ascontiguousarray(self.stats["command_max"])).to(device)
        
        # 【状态缓冲区】
        # current_motion: 当前 CMG 状态 [num_envs, 58]
        # motion_buffer: 预生成的帧缓冲 [num_envs, buffer_frames, 58]
        # buffer_idx: 当前缓冲区索引 [num_envs]
        self.current_motion = None  # 延迟初始化
        self.motion_buffer = None
        self.buffer_idx = None
        
        # 【命令和时间状态】
        # commands: 当前目标速度命令 [num_envs, 3]
        # settle_counter: 落地稳定计数器 [num_envs]，>0 时给零速度
        # episode_time: 每个环境的 episode 时间 [num_envs]
        self.commands = torch.zeros(num_envs, 3, device=device)
        self.settle_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_time = torch.zeros(num_envs, device=device)
        
        # 【落地稳定姿态】
        # 存储每个环境的初始参考姿态（用于落地期间）
        self.settle_dof_pos = torch.zeros(num_envs, dof_dim, device=device)
        self.settle_dof_vel = torch.zeros(num_envs, dof_dim, device=device)  # 落地时速度为0
        
        # 虚拟 motion 长度（用于兼容性，实时模式无边界）
        self._motion_lengths = torch.full(
            (num_envs,), 100.0, device=device, dtype=torch.float  # 100s，实际不会到达
        )
        
        # 性能统计
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        cprint(f"[MotionLibCMGRealtime] 初始化完成", "green")
    
    def _load_cmg_model(self, model_path: str, data_path: str):
        """加载 CMG 模型"""
        from module.cmg import CMG
        
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
        
        cprint(f"[MotionLibCMGRealtime] CMG 模型加载成功", "green")
        return model, stats, samples
    
    def _init_buffers(self):
        """延迟初始化缓冲区（首次调用时）"""
        if self.current_motion is not None:
            return
        
        # 从训练数据中采样初始姿态
        init_motions = self._sample_initial_poses(self.num_envs)
        self.current_motion = init_motions  # [num_envs, 58]
        
        # 初始化缓冲区
        self.motion_buffer = torch.zeros(
            self.num_envs, self.buffer_frames, 58, device=self.device
        )
        self.buffer_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 设置所有环境为落地稳定期
        self.settle_counter = torch.full(
            (self.num_envs,), self.settle_frames, dtype=torch.long, device=self.device
        )
        
        cprint(f"[MotionLibCMGRealtime] 缓冲区初始化完成", "cyan")
    
    def _sample_initial_poses(self, n: int) -> torch.Tensor:
        """
        获取初始站立姿态
        
        策略：使用 CMG 数据中的零速度帧作为初始姿态
        这样确保初始姿态是相对静止的站立状态
        """
        # 方法1：从所有样本中找接近静止的帧
        # 方法2：直接用第一个样本的第一帧（假设数据集是从站立开始）
        
        # 使用方法2：统一使用标准站立姿态
        standing_pose = self._get_standing_pose()  # [58]
        
        # 复制到所有环境
        init_motions = standing_pose.unsqueeze(0).expand(n, -1).clone()
        
        # 添加小随机扰动（避免完全相同）
        noise = torch.randn_like(init_motions) * 0.01
        init_motions[:, :self.dof_dim] += noise[:, :self.dof_dim]  # 只对 dof_pos 加噪声
        
        return init_motions
    
    def _get_standing_pose(self) -> torch.Tensor:
        """
        获取一个标准站立姿态
        
        从训练数据中找一个接近静止（低速度）的帧
        """
        best_pose = None
        min_vel = float('inf')
        
        # 遍历所有样本找速度最小的帧
        for sample in self.samples[:min(100, len(self.samples))]:  # 最多检查100个样本
            motion_seq = sample["motion"]  # [T, motion_dim]
            # 确保使用CPU上的numpy数组，避免GPU索引错误
            if isinstance(motion_seq, torch.Tensor):
                motion_seq = motion_seq.detach().cpu().numpy()
            # 检查前几帧（通常更接近静止）
            for t in range(min(10, len(motion_seq))):
                pose = motion_seq[t]
                vel = np.abs(pose[self.dof_dim:]).sum()  # dof_vel 的绝对值之和
                if vel < min_vel:
                    min_vel = vel
                    best_pose = pose.copy()  # 确保复制，避免引用问题
        
        if best_pose is None:
            # Fallback：使用第一个样本的第一帧
            best_pose = self.samples[0]["motion"][0].copy()
        
        # 形状防御：期望长度=2*dof_dim
        expected_dim = self.dof_dim * 2
        best_pose = np.asarray(best_pose).reshape(-1)
        if best_pose.shape[0] != expected_dim:
            cprint(f"[MotionLibCMGRealtime] WARNING: 站立姿态维度={best_pose.shape[0]} 与期望 {expected_dim} 不一致，自动截断/填充", "red")
            if best_pose.shape[0] > expected_dim:
                best_pose = best_pose[:expected_dim]
            else:
                pad = np.zeros(expected_dim - best_pose.shape[0], dtype=best_pose.dtype)
                best_pose = np.concatenate([best_pose, pad], axis=0)

        cprint(f"[MotionLibCMGRealtime] 选择站立姿态，速度和={min_vel:.4f}", "cyan")
        # 确保numpy数组是C连续的，然后转换到GPU
        best_pose = np.ascontiguousarray(best_pose)
        return torch.from_numpy(best_pose).float().to(self.device)
    
    def _generate_frames(self, env_ids: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        为指定环境生成若干帧
        
        Args:
            env_ids: 需要生成的环境 ID [N]
            num_frames: 生成帧数
        
        Returns:
            frames: [N, num_frames, 58]
        """
        import time
        start_time = time.time()
        
        N = len(env_ids)
        
        # 获取当前 CMG 状态
        current = self.current_motion[env_ids].clone()  # [N, 58]
        
        # 获取有效命令（落地稳定期给零速度）
        effective_commands = self.commands[env_ids].clone()  # [N, 3]
        settling_mask = self.settle_counter[env_ids] > 0
        effective_commands[settling_mask] = 0.0  # 落地期间零速度
        
        # 归一化
        current_norm = (current - self.motion_mean) / self.motion_std
        commands_norm = (effective_commands - self.cmd_min) / (self.cmd_max - self.cmd_min) * 2 - 1
        
        # 生成帧
        frames = []
        with torch.no_grad():
            for _ in range(num_frames):
                pred = self.model(current_norm, commands_norm)
                current_norm = pred
                
                # 反归一化并保存
                frame = current_norm * self.motion_std + self.motion_mean
                frames.append(frame.clone())
        
        # 更新当前状态
        self.current_motion[env_ids] = frames[-1]
        
        # 性能统计
        self._inference_count += 1
        self._total_inference_time += time.time() - start_time
        
        return torch.stack(frames, dim=1)  # [N, num_frames, 58]
    
    def _ensure_buffer(self, env_ids: torch.Tensor):
        """确保缓冲区有足够的帧"""
        # 找出缓冲区已空的环境
        empty_mask = self.buffer_idx[env_ids] >= self.buffer_frames
        if not empty_mask.any():
            return
        
        empty_env_ids = env_ids[empty_mask]
        
        # 生成新帧填充缓冲区
        new_frames = self._generate_frames(empty_env_ids, self.buffer_frames)
        self.motion_buffer[empty_env_ids] = new_frames
        self.buffer_idx[empty_env_ids] = 0
    
    # ==================== MotionLib 兼容接口 ====================
    
    def num_motions(self) -> int:
        """返回虚拟的动作数量（实时模式无概念）"""
        return self.num_envs  # 每个环境一个"动作"
    
    def get_total_length(self) -> float:
        """返回虚拟总时长"""
        return 100.0 * self.num_envs  # 100s per env
    
    def get_motion_length(self, motion_ids) -> torch.Tensor:
        """返回虚拟动作长度（实时模式无边界）"""
        if isinstance(motion_ids, int):
            return self._motion_lengths[motion_ids]
        return self._motion_lengths[motion_ids]
    
    def sample_motions(self, n: int, motion_difficulty=None, curriculum_level: float = None) -> torch.Tensor:
        """
        采样动作 ID（实时模式下就是环境 ID）
        
        Args:
            n: 采样数量
            motion_difficulty: 未使用
            curriculum_level: 用于采样命令范围
        
        Returns:
            motion_ids: [n] 实际上就是 0~n-1
        """
        return torch.arange(n, device=self.device)
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """返回零时间（实时模式从头开始）"""
        return torch.zeros(len(motion_ids), device=self.device, dtype=torch.float)
    
    def calc_motion_frame(
        self, 
        motion_ids: torch.Tensor, 
        motion_times: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        计算指定时刻的动作帧（实时生成）
        
        Args:
            motion_ids: [N] 环境索引
            motion_times: [N] 时间（秒），用于判断落地稳定期
        
        Returns:
            root_pos: None
            root_rot: None
            root_vel: [N, 3] 期望速度
            root_ang_vel: [N, 3] 期望角速度
            dof_pos: [N, 29] 关节位置
            dof_vel: [N, 29] 关节速度
            body_pos: None
        """
        self._init_buffers()  # 确保初始化
        
        N = motion_ids.shape[0]
        env_ids = motion_ids  # 实时模式下 motion_id = env_id
        
        # 检查落地稳定状态
        still_settling = self.settle_counter[env_ids] > 0
        num_settling = still_settling.sum().item()
        
        # 更新落地稳定计数器（每次调用减 1）
        self.settle_counter[env_ids] = torch.clamp(
            self.settle_counter[env_ids] - 1, min=0
        )
        
        # 【调试】打印落地状态
        if self._inference_count < 5 or self._inference_count % 500 == 0:
            cprint(f"[Realtime] 落地稳定中: {num_settling}/{N} 环境, counter[0]={self.settle_counter[0].item()}", "yellow")
        self._inference_count += 1
        
        # 初始化输出
        dof_pos = torch.zeros(N, self.dof_dim, device=self.device)
        dof_vel = torch.zeros(N, self.dof_dim, device=self.device)
        
        # ===== 处理落地稳定期的环境 =====
        if num_settling > 0:
            settling_ids = env_ids[still_settling]
            # 返回固定的站立姿态（零速度）
            dof_pos[still_settling] = self.settle_dof_pos[settling_ids]
            dof_vel[still_settling] = 0.0  # 落地期间参考速度为零
        
        # ===== 处理已稳定环境 =====
        if num_settling < N:
            stable_mask = ~still_settling
            stable_ids = env_ids[stable_mask]
            
            # 确保缓冲区有帧
            self._ensure_buffer(stable_ids)
            
            # 从缓冲区获取当前帧
            frame_idx = self.buffer_idx[stable_ids]
            motion_data = self.motion_buffer[stable_ids, frame_idx]  # [M, 58]
            
            # 更新缓冲区索引
            self.buffer_idx[stable_ids] += 1
            
            # 分离 dof_pos 和 dof_vel
            dof_pos[stable_mask] = motion_data[:, :self.dof_dim]
            dof_vel[stable_mask] = motion_data[:, self.dof_dim:self.dof_dim*2]
        
        # 安全检查
        dof_pos = torch.clamp(dof_pos, -3.14159, 3.14159)
        dof_vel = torch.clamp(dof_vel, -20.0, 20.0)
        
        if torch.isnan(dof_pos).any() or torch.isnan(dof_vel).any():
            cprint("[MotionLibCMGRealtime] Warning: NaN detected!", "red")
            dof_pos = torch.nan_to_num(dof_pos, nan=0.0)
            dof_vel = torch.nan_to_num(dof_vel, nan=0.0)
        
        # 获取有效命令（落地期间返回零）
        effective_commands = self.commands[env_ids].clone()
        effective_commands[still_settling] = 0.0
        
        # root_vel 和 root_ang_vel
        root_vel = torch.zeros(N, 3, device=self.device)
        root_vel[:, 0] = effective_commands[:, 0]  # vx
        root_vel[:, 1] = effective_commands[:, 1]  # vy
        
        root_ang_vel = torch.zeros(N, 3, device=self.device)
        root_ang_vel[:, 2] = effective_commands[:, 2]  # yaw_rate
        
        return None, None, root_vel, root_ang_vel, dof_pos, dof_vel, None
    
    def reset_envs(self, env_ids: torch.Tensor, commands: Optional[torch.Tensor] = None,
                   init_dof_pos: Optional[torch.Tensor] = None):
        """
        重置指定环境
        
        Args:
            env_ids: 要重置的环境 ID [N]
            commands: 新的速度命令 [N, 3]，如果为 None 则采样
            init_dof_pos: 初始关节位置 [N, dof_dim]，用于落地稳定期的参考
                         如果为 None，则使用从 CMG 数据采样的姿态
        """
        self._init_buffers()
        
        n = len(env_ids)
        
        # 重新采样初始姿态（用于 CMG 自回归的起点）
        init_motions = self._sample_initial_poses(n)
        self.current_motion[env_ids] = init_motions
        
        # 设置落地稳定期的参考姿态
        if init_dof_pos is not None:
            # 使用仿真器的初始关节角度
            self.settle_dof_pos[env_ids] = init_dof_pos
        else:
            # 使用 CMG 初始姿态的 dof_pos 部分
            self.settle_dof_pos[env_ids] = init_motions[:, :self.dof_dim]
        
        # 重置缓冲区
        self.buffer_idx[env_ids] = self.buffer_frames  # 标记为需要重新生成
        
        # 重置落地稳定计数器
        self.settle_counter[env_ids] = self.settle_frames
        
        # 更新命令
        if commands is not None:
            self.commands[env_ids] = commands
        else:
            # 采样新命令
            self.commands[env_ids] = self._sample_commands(n)
    
    def _sample_commands(self, n: int, curriculum_level: float = 0.0) -> torch.Tensor:
        """
        采样速度命令
        
        Args:
            n: 采样数量
            curriculum_level: 课程难度 (0~1)
        """
        commands = torch.zeros(n, 3, device=self.device)
        
        # 简化版本：先专注于向前走
        # vx: [0.5, 1.5] m/s（适中速度）
        commands[:, 0] = torch.rand(n, device=self.device) * 1.0 + 0.5
        
        # vy: 0（不侧移）
        commands[:, 1] = 0.0
        
        # yaw: 小范围 [-0.2, 0.2] rad/s
        commands[:, 2] = (torch.rand(n, device=self.device) - 0.5) * 0.4
        
        return commands
    
    def update_commands(self, env_ids: torch.Tensor, commands: torch.Tensor):
        """更新指定环境的命令"""
        self.commands[env_ids] = commands
        # 不重置缓冲区，让命令平滑过渡
    
    def get_motion_command(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """获取指定环境的当前命令"""
        return self.commands[motion_ids]
    
    def get_motion_names(self) -> List[str]:
        """返回动作名称列表"""
        return [f"realtime_env_{i}" for i in range(self.num_envs)]
    
    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        """CMG 不生成 body 位置"""
        return []
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if self._inference_count == 0:
            return {'avg_inference_ms': 0.0, 'total_inferences': 0}
        
        avg_ms = (self._total_inference_time / self._inference_count) * 1000
        return {
            'avg_inference_ms': avg_ms,
            'total_inferences': self._inference_count,
        }


def create_motion_lib_cmg_realtime(cfg, num_envs: int, device: str) -> MotionLibCMGRealtime:
    """
    工厂函数：创建 MotionLibCMGRealtime 实例
    """
    motion_cfg = cfg.motion
    
    cmg_model_path = motion_cfg.cmg_model_path
    cmg_data_path = motion_cfg.cmg_data_path
    
    dof_dim = getattr(motion_cfg, 'dof_dim', 29)
    fps = getattr(motion_cfg, 'fps', 50.0)
    buffer_frames = getattr(motion_cfg, 'cmg_buffer_frames', 10)
    
    if not os.path.exists(cmg_model_path):
        raise FileNotFoundError(f"CMG 模型不存在: {cmg_model_path}")
    if not os.path.exists(cmg_data_path):
        raise FileNotFoundError(f"CMG 数据不存在: {cmg_data_path}")
    
    cprint(f"[create_motion_lib_cmg_realtime] 创建 CMG 实时动作库", "green")
    cprint(f"  - 模型: {cmg_model_path}", "cyan")
    cprint(f"  - 数据: {cmg_data_path}", "cyan")
    
    return MotionLibCMGRealtime(
        cmg_model_path=cmg_model_path,
        cmg_data_path=cmg_data_path,
        num_envs=num_envs,
        device=device,
        dof_dim=dof_dim,
        fps=fps,
        buffer_frames=buffer_frames,
    )
