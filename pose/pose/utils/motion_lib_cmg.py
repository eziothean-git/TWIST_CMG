"""
MotionLibCMG: 使用CMG预生成参考动作的MotionLib实现

核心设计:
1. 初始化时一次性生成动作池（大量轨迹）
2. 训练时从池中采样，不再实时生成
3. 提供与MotionLib兼容的接口

关键设计考虑:
- 未来观测窗口: tar_obs_steps最大95步 = 1.9s，所以有效运行时间 = motion_length - 2s
- 时间边界处理: 在接近边界时触发reset，而不是clamp
- root/body信息: 从velocity command积分估计root_vel，使用FK计算body_pos
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from termcolor import cprint

# 添加CMG_Ref路径
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


class MotionLibCMG:
    """
    CMG预生成动作库
    
    初始化时生成动作池，训练时从池中采样
    提供与MotionLib兼容的接口
    
    关键设计:
    - safe_motion_length: 实际可用长度 = motion_length - future_window (2s)
    - 确保查询未来观测时不会超出边界
    - 提供root_vel估计 (从velocity command)
    """
    
    # 未来观测窗口（秒），对应tar_obs_steps最大值
    FUTURE_WINDOW_S = 2.0
    
    def __init__(
        self, 
        cmg_model_path: str,
        cmg_data_path: str,
        num_motions: int,           # 动作池大小（轨迹数量）
        motion_length: int,         # 每条轨迹的长度（帧数）
        num_envs: int,              # 并行环境数
        device: str,
        dof_dim: int = 29,          # 关节自由度
        fps: float = 50.0,          # 帧率
    ):
        """
        初始化CMG动作库
        
        Args:
            cmg_model_path: CMG模型权重路径
            cmg_data_path: CMG训练数据路径
            num_motions: 动作池大小（预生成的轨迹数量）
            motion_length: 每条轨迹的长度（帧数）
            num_envs: 并行环境数量
            device: PyTorch设备
            dof_dim: 关节自由度数量
            fps: 动作帧率
        """
        self.device = device
        self._num_motions = num_motions
        self.motion_length = motion_length
        self.num_envs = num_envs
        self.dof_dim = dof_dim
        self.fps = fps
        self.dt = 1.0 / fps
        
        # 安全运行长度：预留未来观测窗口
        self.future_window_frames = int(self.FUTURE_WINDOW_S * fps)
        self.safe_motion_length = motion_length - self.future_window_frames
        
        cprint(f"[MotionLibCMG] 初始化预生成动作库", "green")
        cprint(f"  - 动作池大小: {num_motions} 条轨迹", "green")
        cprint(f"  - 轨迹长度: {motion_length} 帧 ({motion_length / fps:.1f}s)", "green")
        cprint(f"  - 安全长度: {self.safe_motion_length} 帧 ({self.safe_motion_length / fps:.1f}s)", "green")
        cprint(f"  - 未来窗口: {self.future_window_frames} 帧 ({self.FUTURE_WINDOW_S}s)", "green")
        cprint(f"  - 环境数: {num_envs}", "green")
        cprint(f"  - 设备: {device}", "green")
        
        # 加载CMG模型
        self.model, self.stats, self.samples = self._load_cmg_model(
            cmg_model_path, cmg_data_path
        )
        
        # 提取统计信息
        self.motion_mean = torch.from_numpy(self.stats["motion_mean"]).to(device)
        self.motion_std = torch.from_numpy(self.stats["motion_std"]).to(device)
        self.cmd_min = torch.from_numpy(self.stats["command_min"]).to(device)
        self.cmd_max = torch.from_numpy(self.stats["command_max"]).to(device)
        
        # 预生成动作池
        cprint(f"[MotionLibCMG] 开始预生成动作池...", "yellow")
        self._generate_motion_pool()
        cprint(f"[MotionLibCMG] 动作池生成完成!", "green")
        
        # 释放CMG模型以节省显存（生成完成后不再需要）
        del self.model
        del self.samples
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        cprint(f"[MotionLibCMG] CMG模型已释放，显存已清理", "green")
        
        # 安全运行时长（秒，用于termination检查）
        self._motion_lengths = torch.full(
            (num_motions,), self.safe_motion_length * self.dt, 
            device=device, dtype=torch.float
        )
        
        # 存储每条轨迹对应的速度命令
        # self.motion_commands 已在 _generate_motion_pool 中设置
    
    def _load_cmg_model(self, model_path: str, data_path: str):
        """加载CMG模型"""
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
        
        cprint(f"[MotionLibCMG] CMG模型加载成功", "green")
        return model, stats, samples
    
    def _generate_motion_pool(self):
        """
        预生成动作池
        
        随机采样速度命令，批量生成所有轨迹
        """
        # 随机采样速度命令 [num_motions, 3]
        commands = self._sample_diverse_commands(self._num_motions)
        self.motion_commands = commands  # 保存用于调试
        
        # 随机采样初始姿态
        init_motions = self._sample_initial_poses(self._num_motions)
        
        # 批量生成轨迹 [num_motions, motion_length+1, 58]
        # 分批处理避免显存溢出
        batch_size = min(512, self._num_motions)
        all_trajectories = []
        
        with torch.no_grad():
            for i in range(0, self._num_motions, batch_size):
                end_idx = min(i + batch_size, self._num_motions)
                batch_commands = commands[i:end_idx]
                batch_init = init_motions[i:end_idx]
                
                # 生成该批次的轨迹
                trajectories = self._batch_generate(
                    batch_init, 
                    batch_commands,
                    self.motion_length
                )
                all_trajectories.append(trajectories)
                
                if (i // batch_size + 1) % 10 == 0:
                    cprint(f"  - 已生成 {end_idx}/{self._num_motions} 条轨迹", "cyan")
        
        # 合并所有轨迹 [num_motions, motion_length+1, 58]
        self.motion_pool = torch.cat(all_trajectories, dim=0)
        
        # 检查生成的数据是否有NaN或Inf
        if torch.isnan(self.motion_pool).any():
            cprint(f"[MotionLibCMG] Warning: NaN detected in generated motion pool!", "red")
            self.motion_pool = torch.nan_to_num(self.motion_pool, nan=0.0)
        if torch.isinf(self.motion_pool).any():
            cprint(f"[MotionLibCMG] Warning: Inf detected in generated motion pool!", "red")
            self.motion_pool = torch.clamp(self.motion_pool, -100.0, 100.0)
        
        # 分离 dof_pos 和 dof_vel
        # motion_pool: [num_motions, T, 58] -> dof_pos [num_motions, T, 29], dof_vel [num_motions, T, 29]
        self.dof_pos_pool = self.motion_pool[..., :self.dof_dim]
        self.dof_vel_pool = self.motion_pool[..., self.dof_dim:self.dof_dim*2]
        
        # Clamp关节值到合理范围
        self.dof_pos_pool = torch.clamp(self.dof_pos_pool, -3.14159, 3.14159)
        self.dof_vel_pool = torch.clamp(self.dof_vel_pool, -20.0, 20.0)
        
        cprint(f"  - 动作池形状: {self.motion_pool.shape}", "green")
        cprint(f"  - dof_pos范围: [{self.dof_pos_pool.min():.3f}, {self.dof_pos_pool.max():.3f}]", "green")
        cprint(f"  - dof_vel范围: [{self.dof_vel_pool.min():.3f}, {self.dof_vel_pool.max():.3f}]", "green")
    
    def _sample_diverse_commands(self, n: int) -> torch.Tensor:
        """
        采样多样化的速度命令
        
        简化版本：专注于向前行走，加速收敛
        - vx: [0.15, 2.5] m/s (只向前)
        - vy: 0 (不侧移)
        - yaw: [-0.26, 0.26] rad/s (最大约15 deg/s)
        """
        commands = torch.zeros(n, 3, device=self.device)
        
        # vx: 均匀采样 [0.15, 2.5] m/s
        commands[:, 0] = torch.rand(n, device=self.device) * (2.5 - 0.15) + 0.15
        
        # vy: 保持为0（不侧移）
        commands[:, 1] = 0.0
        
        # yaw: 小范围转向 [-0.26, 0.26] rad/s (约 ±15 deg/s)
        commands[:, 2] = (torch.rand(n, device=self.device) - 0.5) * 0.52  # ±0.26 rad/s
        
        # 随机打乱顺序
        perm = torch.randperm(n, device=self.device)
        commands = commands[perm]
        
        cprint(f"[MotionLibCMG] 命令采样范围:", "cyan")
        cprint(f"  - vx: [{commands[:, 0].min():.2f}, {commands[:, 0].max():.2f}] m/s", "cyan")
        cprint(f"  - vy: {commands[:, 1].mean():.2f} m/s (固定)", "cyan")
        cprint(f"  - yaw: [{commands[:, 2].min():.2f}, {commands[:, 2].max():.2f}] rad/s", "cyan")
        
        return commands
    
    def _sample_initial_poses(self, n: int) -> torch.Tensor:
        """从训练数据中随机采样初始姿态"""
        init_motions = []
        num_samples = len(self.samples)
        for _ in range(n):
            # samples 是一个列表，每个元素是字典 {"motion": [T, 58], "command": [T-1, 3]}
            sample_idx = np.random.randint(0, num_samples)
            motion_seq = self.samples[sample_idx]["motion"]  # [T, 58]
            frame_idx = np.random.randint(0, len(motion_seq))
            init_motions.append(motion_seq[frame_idx])
        
        return torch.from_numpy(np.stack(init_motions)).float().to(self.device)
    
    def _batch_generate(
        self, 
        init_motion: torch.Tensor,
        commands: torch.Tensor,
        length: int
    ) -> torch.Tensor:
        """
        批量自回归生成轨迹
        
        Args:
            init_motion: [N, 58] 初始动作（未归一化）
            commands: [N, 3] 速度命令（未归一化）
            length: 生成长度（帧数）
        
        Returns:
            trajectories: [N, length+1, 58]（未归一化）
        """
        N = init_motion.shape[0]
        
        # 归一化
        current = (init_motion - self.motion_mean) / self.motion_std
        commands_norm = (commands - self.cmd_min) / (self.cmd_max - self.cmd_min) * 2 - 1
        
        # 存储结果
        trajectories = [current.clone()]
        
        with torch.no_grad():
            for _ in range(length):
                pred = self.model(current, commands_norm)
                current = pred
                trajectories.append(current.clone())
        
        # 合并并反归一化
        trajectories = torch.stack(trajectories, dim=1)  # [N, length+1, 58]
        trajectories = trajectories * self.motion_std + self.motion_mean
        
        return trajectories
    
    # ==================== MotionLib 兼容接口 ====================
    
    def num_motions(self) -> int:
        """返回动作池中的轨迹数量"""
        return self._num_motions
    
    def get_total_length(self) -> float:
        """返回所有动作的总时长（秒），使用安全长度"""
        return self._num_motions * self.safe_motion_length * self.dt
    
    def get_motion_length(self, motion_ids) -> torch.Tensor:
        """
        返回指定动作的安全运行长度（秒）
        
        注意：返回safe_motion_length，确保查询未来窗口时不会越界
        
        Args:
            motion_ids: 动作索引
        
        Returns:
            lengths: 安全时长（秒）
        """
        if isinstance(motion_ids, int):
            return self._motion_lengths[motion_ids]
        return self._motion_lengths[motion_ids]
    
    def sample_motions(self, n: int, motion_difficulty=None, curriculum_level: float = None) -> torch.Tensor:
        """
        从动作池中采样，支持基于curriculum_level的筛选
        
        Args:
            n: 采样数量
            motion_difficulty: 未使用（兼容接口）
            curriculum_level: 课程难度 (0~1)，控制命令范围
                - 0: 只采样低速命令 (|vx|<0.5, |vy|<0.2, |yaw|<0.2)
                - 1: 全范围采样
        
        Returns:
            motion_ids: [n] 采样的动作索引
        """
        if curriculum_level is None or curriculum_level >= 0.99:
            # 全范围随机采样
            return torch.randint(0, self._num_motions, (n,), device=self.device)
        
        # 基于curriculum_level筛选合适的motion
        # 计算每个motion的"难度"（基于命令大小）
        commands = self.motion_commands  # [num_motions, 3] (vx, vy, yaw)
        
        # 定义最大命令范围
        max_vx, max_vy, max_yaw = 3.0, 0.8, 0.8
        min_vx, min_vy, min_yaw = 0.5, 0.2, 0.2
        
        # 当前允许的范围 = min + curriculum_level * (max - min)
        curr_vx = min_vx + curriculum_level * (max_vx - min_vx)
        curr_vy = min_vy + curriculum_level * (max_vy - min_vy)
        curr_yaw = min_yaw + curriculum_level * (max_yaw - min_yaw)
        
        # 筛选符合条件的motion
        valid_mask = (
            (torch.abs(commands[:, 0]) <= curr_vx) &
            (torch.abs(commands[:, 1]) <= curr_vy) &
            (torch.abs(commands[:, 2]) <= curr_yaw)
        )
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_indices) < n:
            # 如果符合条件的不够，允许一些越界
            # 按命令大小排序，取最小的前N个
            cmd_magnitude = (
                (commands[:, 0] / max_vx).abs() +
                (commands[:, 1] / max_vy).abs() +
                (commands[:, 2] / max_yaw).abs()
            )
            sorted_indices = torch.argsort(cmd_magnitude)
            valid_indices = sorted_indices[:max(n, self._num_motions // 4)]
        
        # 从有效索引中随机采样
        sample_idx = torch.randint(0, len(valid_indices), (n,), device=self.device)
        return valid_indices[sample_idx]
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """
        为指定动作采样起始时间
        
        注意：采样范围是[0, safe_motion_length)，确保运行时+未来窗口不会越界
        
        Args:
            motion_ids: [n] 动作索引
        
        Returns:
            times: [n] 起始时间（秒）
        """
        n = len(motion_ids)
        # 在安全范围内随机采样帧索引
        max_frame = self.safe_motion_length
        frame_ids = torch.randint(0, max_frame, (n,), device=self.device)
        return frame_ids.float() * self.dt
    
    def calc_motion_frame(
        self, 
        motion_ids: torch.Tensor, 
        motion_times: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        计算指定时刻的动作帧
        
        关于root信息的设计说明：
        - CMG只输出关节状态(dof_pos, dof_vel)，不直接输出root轨迹
        - root_vel/root_ang_vel: 从生成该motion时使用的velocity command获取
          这是该轨迹的"期望速度"，作为Teacher的特权信息
        - root_pos/root_rot: 返回None，因为CMG无法提供准确的世界坐标位置
          环境应该使用仿真器的实际状态
        - body_pos: 返回None，环境应该使用仿真器状态或FK计算
        
        Args:
            motion_ids: [N] 动作索引
            motion_times: [N] 时间（秒）
        
        Returns:
            root_pos: None (CMG无法提供准确的世界坐标)
            root_rot: None
            root_vel: [N, 3] 期望速度 (来自velocity command)
            root_ang_vel: [N, 3] 期望角速度 (来自velocity command)
            dof_pos: [N, 29] 关节位置
            dof_vel: [N, 29] 关节速度
            body_pos: None (需要FK或仿真器)
        """
        N = motion_ids.shape[0]
        
        # 安全检查：确保motion_ids在有效范围内
        motion_ids = torch.clamp(motion_ids, 0, self._num_motions - 1)
        
        # 时间转换为帧索引
        frame_ids = (motion_times / self.dt).long()
        max_frame = self.motion_length
        frame_ids = torch.clamp(frame_ids, 0, max_frame - 1)
        
        # 索引动作池
        dof_pos = self.dof_pos_pool[motion_ids, frame_ids]  # [N, 29]
        dof_vel = self.dof_vel_pool[motion_ids, frame_ids]  # [N, 29]
        
        # 安全检查：clamp极端值，防止仿真器崩溃
        dof_pos = torch.clamp(dof_pos, -3.14159, 3.14159)  # 关节角度范围
        dof_vel = torch.clamp(dof_vel, -20.0, 20.0)  # 关节速度范围
        
        # 检查并替换NaN
        if torch.isnan(dof_pos).any() or torch.isnan(dof_vel).any():
            cprint("[MotionLibCMG] Warning: NaN detected in dof_pos/dof_vel, replacing with zeros", "red")
            dof_pos = torch.nan_to_num(dof_pos, nan=0.0)
            dof_vel = torch.nan_to_num(dof_vel, nan=0.0)
        
        # 从velocity command获取期望速度（作为特权信息）
        # motion_commands: [num_motions, 3] = (vx, vy, yaw_rate)
        commands = self.motion_commands[motion_ids]  # [N, 3]
        
        # root_vel: 期望线速度（世界系，假设初始朝向为x轴正方向）
        root_vel = torch.zeros(N, 3, device=self.device)
        root_vel[:, 0] = commands[:, 0]  # vx
        root_vel[:, 1] = commands[:, 1]  # vy
        root_vel[:, 2] = 0.0  # vz
        
        # root_ang_vel: 期望角速度
        root_ang_vel = torch.zeros(N, 3, device=self.device)
        root_ang_vel[:, 2] = commands[:, 2]  # yaw_rate
        
        # root_pos, root_rot, body_pos: 返回None
        # 环境需要从仿真器获取实际位置，或使用默认值
        return None, None, root_vel, root_ang_vel, dof_pos, dof_vel, None
    
    def get_motion_names(self) -> List[str]:
        """返回动作名称列表"""
        return [f"cmg_motion_{i}" for i in range(self._num_motions)]
    
    def get_key_body_idx(self, key_body_names: List[str]) -> List[int]:
        """
        返回关键body索引
        
        注意: CMG不生成body位置，返回空列表
        环境应使用FK或仿真器计算body位置
        """
        cprint("[MotionLibCMG] Warning: CMG不生成body位置，返回空列表", "yellow")
        return []
    
    def get_motion_command(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """
        获取指定动作对应的速度命令
        
        Args:
            motion_ids: [N] 动作索引
        
        Returns:
            commands: [N, 3] (vx, vy, yaw_rate)
        """
        return self.motion_commands[motion_ids]


def create_motion_lib_cmg(cfg, num_envs: int, device: str) -> MotionLibCMG:
    """
    工厂函数：根据配置创建MotionLibCMG实例
    
    Args:
        cfg: 配置对象
        num_envs: 并行环境数量
        device: PyTorch设备
    
    Returns:
        MotionLibCMG实例
    """
    motion_cfg = cfg.motion
    
    # 获取路径
    cmg_model_path = motion_cfg.cmg_model_path
    cmg_data_path = motion_cfg.cmg_data_path
    
    # 获取动作池参数
    num_motions = getattr(motion_cfg, 'cmg_num_motions', 1024)  # 默认1024条轨迹
    motion_length = getattr(motion_cfg, 'cmg_motion_length', 500)  # 默认500帧 (10s)
    dof_dim = getattr(motion_cfg, 'dof_dim', 29)
    fps = getattr(motion_cfg, 'fps', 50.0)
    
    # 检查文件是否存在
    if not os.path.exists(cmg_model_path):
        raise FileNotFoundError(f"CMG模型不存在: {cmg_model_path}")
    if not os.path.exists(cmg_data_path):
        raise FileNotFoundError(f"CMG数据不存在: {cmg_data_path}")
    
    cprint(f"[create_motion_lib_cmg] 创建CMG动作库", "green")
    cprint(f"  - 模型: {cmg_model_path}", "cyan")
    cprint(f"  - 数据: {cmg_data_path}", "cyan")
    
    return MotionLibCMG(
        cmg_model_path=cmg_model_path,
        cmg_data_path=cmg_data_path,
        num_motions=num_motions,
        motion_length=motion_length,
        num_envs=num_envs,
        device=device,
        dof_dim=dof_dim,
        fps=fps,
    )
