"""
MotionLibCMG: 使用CMG预生成参考动作的MotionLib实现

核心设计:
1. 初始化时一次性生成动作池（大量轨迹）
2. 训练时从池中采样，不再实时生成
3. 提供与MotionLib兼容的接口

与原版mocap数据加载的区别:
- 不包含body位置信息（需要FK或仿真器计算）
- root信息从仿真器获取而非参考动作
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
    """
    
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
        
        cprint(f"[MotionLibCMG] 初始化预生成动作库", "green")
        cprint(f"  - 动作池大小: {num_motions} 条轨迹", "green")
        cprint(f"  - 轨迹长度: {motion_length} 帧 ({motion_length / fps:.1f}s)", "green")
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
        
        # 轨迹长度（帧数，转换为张量）
        self._motion_lengths = torch.full(
            (num_motions,), motion_length, device=device, dtype=torch.float
        )
        
        # 存储每条轨迹对应的速度命令（用于调试/分析）
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
        
        # 分离 dof_pos 和 dof_vel
        # motion_pool: [num_motions, T, 58] -> dof_pos [num_motions, T, 29], dof_vel [num_motions, T, 29]
        self.dof_pos_pool = self.motion_pool[..., :self.dof_dim]
        self.dof_vel_pool = self.motion_pool[..., self.dof_dim:self.dof_dim*2]
        
        cprint(f"  - 动作池形状: {self.motion_pool.shape}", "green")
    
    def _sample_diverse_commands(self, n: int) -> torch.Tensor:
        """
        采样多样化的速度命令
        
        包含多种运动模式：前进、后退、转向、侧移、混合
        """
        commands = torch.zeros(n, 3, device=self.device)
        
        # 将命令分为不同类别
        n_forward = n // 4           # 25% 前进
        n_turn = n // 4              # 25% 转向
        n_mixed = n // 4             # 25% 混合
        n_other = n - n_forward - n_turn - n_mixed  # 25% 其他
        
        idx = 0
        
        # 1. 前进 (vx > 0, vy ≈ 0, yaw ≈ 0)
        commands[idx:idx+n_forward, 0] = torch.rand(n_forward, device=self.device) * 2.5 + 0.5  # 0.5~3.0 m/s
        commands[idx:idx+n_forward, 1] = (torch.rand(n_forward, device=self.device) - 0.5) * 0.2  # 小侧移
        commands[idx:idx+n_forward, 2] = (torch.rand(n_forward, device=self.device) - 0.5) * 0.3  # 小转向
        idx += n_forward
        
        # 2. 转向 (vx 适中, yaw 较大)
        commands[idx:idx+n_turn, 0] = torch.rand(n_turn, device=self.device) * 1.5 + 0.3  # 0.3~1.8 m/s
        commands[idx:idx+n_turn, 1] = (torch.rand(n_turn, device=self.device) - 0.5) * 0.4
        commands[idx:idx+n_turn, 2] = (torch.rand(n_turn, device=self.device) - 0.5) * 1.6  # -0.8~0.8 rad/s
        idx += n_turn
        
        # 3. 混合运动 (全范围)
        commands[idx:idx+n_mixed, 0] = torch.rand(n_mixed, device=self.device) * 4.5 - 1.5  # -1.5~3.0 m/s
        commands[idx:idx+n_mixed, 1] = (torch.rand(n_mixed, device=self.device) - 0.5) * 1.6  # -0.8~0.8 m/s
        commands[idx:idx+n_mixed, 2] = (torch.rand(n_mixed, device=self.device) - 0.5) * 1.6  # -0.8~0.8 rad/s
        idx += n_mixed
        
        # 4. 其他 (后退、侧移、原地等)
        for i in range(n_other):
            mode = np.random.choice(['backward', 'strafe', 'stand', 'slow_walk'])
            if mode == 'backward':
                commands[idx+i, 0] = -torch.rand(1, device=self.device) * 1.0 - 0.3  # -1.3~-0.3
                commands[idx+i, 1] = (torch.rand(1, device=self.device) - 0.5) * 0.3
                commands[idx+i, 2] = (torch.rand(1, device=self.device) - 0.5) * 0.4
            elif mode == 'strafe':
                commands[idx+i, 0] = torch.rand(1, device=self.device) * 0.5  # 0~0.5
                commands[idx+i, 1] = (torch.rand(1, device=self.device) - 0.5) * 1.6  # -0.8~0.8
                commands[idx+i, 2] = (torch.rand(1, device=self.device) - 0.5) * 0.4
            elif mode == 'stand':
                commands[idx+i] = (torch.rand(3, device=self.device) - 0.5) * 0.1  # 近似静止
            else:  # slow_walk
                commands[idx+i, 0] = torch.rand(1, device=self.device) * 0.5 + 0.1  # 0.1~0.6
                commands[idx+i, 1] = (torch.rand(1, device=self.device) - 0.5) * 0.2
                commands[idx+i, 2] = (torch.rand(1, device=self.device) - 0.5) * 0.2
        
        # 随机打乱顺序
        perm = torch.randperm(n, device=self.device)
        commands = commands[perm]
        
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
        """返回所有动作的总时长（秒）"""
        return self._num_motions * self.motion_length * self.dt
    
    def get_motion_length(self, motion_ids) -> torch.Tensor:
        """
        返回指定动作的长度（秒）
        
        Args:
            motion_ids: 动作索引
        
        Returns:
            lengths: 时长（秒）
        """
        if isinstance(motion_ids, int):
            return self._motion_lengths[motion_ids] * self.dt
        return self._motion_lengths[motion_ids] * self.dt
    
    def sample_motions(self, n: int, motion_difficulty=None) -> torch.Tensor:
        """
        从动作池中随机采样
        
        Args:
            n: 采样数量
            motion_difficulty: 未使用（兼容接口）
        
        Returns:
            motion_ids: [n] 采样的动作索引
        """
        return torch.randint(0, self._num_motions, (n,), device=self.device)
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """
        为指定动作采样起始时间
        
        Args:
            motion_ids: [n] 动作索引
        
        Returns:
            times: [n] 起始时间（秒）
        """
        n = len(motion_ids)
        # 随机采样帧索引，转换为时间
        frame_ids = torch.randint(0, self.motion_length, (n,), device=self.device)
        return frame_ids.float() * self.dt
    
    def calc_motion_frame(
        self, 
        motion_ids: torch.Tensor, 
        motion_times: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        计算指定时刻的动作帧
        
        Args:
            motion_ids: [N] 动作索引
            motion_times: [N] 时间（秒）
        
        Returns:
            root_pos: None (CMG不生成，由仿真器提供)
            root_rot: None
            root_vel: None
            root_ang_vel: None
            dof_pos: [N, 29] 关节位置
            dof_vel: [N, 29] 关节速度
            body_pos: None (需要FK计算)
        """
        # 时间转换为帧索引
        frame_ids = (motion_times / self.dt).long()
        frame_ids = torch.clamp(frame_ids, 0, self.motion_length)
        
        # 索引动作池
        dof_pos = self.dof_pos_pool[motion_ids, frame_ids]  # [N, 29]
        dof_vel = self.dof_vel_pool[motion_ids, frame_ids]  # [N, 29]
        
        # CMG不生成root和body信息
        return None, None, None, None, dof_pos, dof_vel, None
    
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
