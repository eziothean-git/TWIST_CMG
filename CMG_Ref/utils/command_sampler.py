"""
速度命令采样器
为不同训练阶段生成多样化的速度命令
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class CommandSampler:
    """
    速度命令采样器
    支持多种采样策略，用于不同的训练阶段
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str = 'cuda',
        # 速度范围
        vx_range: Tuple[float, float] = (-1.5, 3.0),
        vy_range: Tuple[float, float] = (-0.8, 0.8),
        yaw_range: Tuple[float, float] = (-0.8, 0.8),
    ):
        """
        Args:
            num_envs: 环境数量
            device: 计算设备
            vx_range: 前向速度范围 [min, max] (m/s)
            vy_range: 横向速度范围 [min, max] (m/s)
            yaw_range: 旋转速度范围 [min, max] (rad/s)
        """
        self.num_envs = num_envs
        self.device = device
        
        self.vx_range = torch.tensor(vx_range, device=device)
        self.vy_range = torch.tensor(vy_range, device=device)
        self.yaw_range = torch.tensor(yaw_range, device=device)
    
    def sample_uniform(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        均匀随机采样
        
        Args:
            env_ids: 要采样的环境ID 或 None (所有环境)
        
        Returns:
            commands: [N, 3] (vx, vy, yaw)
        """
        if env_ids is None:
            num_sample = self.num_envs
        else:
            num_sample = len(env_ids)
        
        vx = torch.rand(num_sample, device=self.device) * \
             (self.vx_range[1] - self.vx_range[0]) + self.vx_range[0]
        
        vy = torch.rand(num_sample, device=self.device) * \
             (self.vy_range[1] - self.vy_range[0]) + self.vy_range[0]
        
        yaw = torch.rand(num_sample, device=self.device) * \
              (self.yaw_range[1] - self.yaw_range[0]) + self.yaw_range[0]
        
        return torch.stack([vx, vy, yaw], dim=-1)
    
    def sample_curriculum(
        self,
        difficulty: float,
        env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        课程学习采样 - 根据难度系数调整命令范围
        
        Args:
            difficulty: 难度系数 [0, 1]
                - 0.0: 简单 (仅前进)
                - 0.5: 中等 (前进+转向)
                - 1.0: 困难 (全方向运动)
            env_ids: 环境ID
        
        Returns:
            commands: [N, 3]
        """
        if env_ids is None:
            num_sample = self.num_envs
        else:
            num_sample = len(env_ids)
        
        # 根据难度缩放速度范围
        vx_scale = 0.5 + 0.5 * difficulty  # 0.5 -> 1.0
        vy_scale = difficulty  # 0.0 -> 1.0
        yaw_scale = 0.3 + 0.7 * difficulty  # 0.3 -> 1.0
        
        vx = torch.rand(num_sample, device=self.device) * \
             (self.vx_range[1] * vx_scale - self.vx_range[0] * vx_scale) + \
             self.vx_range[0] * vx_scale
        
        vy = (torch.rand(num_sample, device=self.device) - 0.5) * 2 * \
             self.vy_range[1] * vy_scale
        
        yaw = (torch.rand(num_sample, device=self.device) - 0.5) * 2 * \
              self.yaw_range[1] * yaw_scale
        
        return torch.stack([vx, vy, yaw], dim=-1)
    
    def sample_preset(
        self,
        mode: str,
        env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预设模式采样
        
        Args:
            mode: 预设模式
                - 'forward': 纯前进
                - 'backward': 纯后退
                - 'turn_left': 原地左转
                - 'turn_right': 原地右转
                - 'strafe_left': 左平移
                - 'strafe_right': 右平移
                - 'forward_turn': 前进+转向
                - 'mixed': 混合运动
            env_ids: 环境ID
        
        Returns:
            commands: [N, 3]
        """
        if env_ids is None:
            num_sample = self.num_envs
        else:
            num_sample = len(env_ids)
        
        if mode == 'forward':
            vx = torch.rand(num_sample, device=self.device) * 2.0 + 0.5  # 0.5~2.5 m/s
            vy = torch.zeros(num_sample, device=self.device)
            yaw = torch.zeros(num_sample, device=self.device)
        
        elif mode == 'backward':
            vx = torch.rand(num_sample, device=self.device) * -1.0 - 0.3  # -1.3~-0.3 m/s
            vy = torch.zeros(num_sample, device=self.device)
            yaw = torch.zeros(num_sample, device=self.device)
        
        elif mode == 'turn_left':
            vx = torch.zeros(num_sample, device=self.device)
            vy = torch.zeros(num_sample, device=self.device)
            yaw = torch.rand(num_sample, device=self.device) * 0.5 + 0.2  # 0.2~0.7 rad/s
        
        elif mode == 'turn_right':
            vx = torch.zeros(num_sample, device=self.device)
            vy = torch.zeros(num_sample, device=self.device)
            yaw = torch.rand(num_sample, device=self.device) * -0.5 - 0.2  # -0.7~-0.2 rad/s
        
        elif mode == 'strafe_left':
            vx = torch.rand(num_sample, device=self.device) * 0.5  # 0~0.5 m/s
            vy = torch.rand(num_sample, device=self.device) * 0.5 + 0.3  # 0.3~0.8 m/s
            yaw = torch.zeros(num_sample, device=self.device)
        
        elif mode == 'strafe_right':
            vx = torch.rand(num_sample, device=self.device) * 0.5
            vy = torch.rand(num_sample, device=self.device) * -0.5 - 0.3  # -0.8~-0.3 m/s
            yaw = torch.zeros(num_sample, device=self.device)
        
        elif mode == 'forward_turn':
            vx = torch.rand(num_sample, device=self.device) * 1.5 + 0.5  # 0.5~2.0 m/s
            vy = torch.zeros(num_sample, device=self.device)
            yaw = (torch.rand(num_sample, device=self.device) - 0.5) * 0.8  # -0.4~0.4 rad/s
        
        elif mode == 'mixed':
            return self.sample_uniform(env_ids)
        
        else:
            raise ValueError(f"Unknown preset mode: {mode}")
        
        return torch.stack([vx, vy, yaw], dim=-1)
    
    def sample_distribution(
        self,
        distribution: str = 'normal',
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        从分布采样
        
        Args:
            distribution: 'normal' 或 'truncated_normal'
            mean: [3] 均值 (vx, vy, yaw) 或 None (使用范围中点)
            std: [3] 标准差 或 None (使用范围宽度的1/4)
            env_ids: 环境ID
        
        Returns:
            commands: [N, 3]
        """
        if env_ids is None:
            num_sample = self.num_envs
        else:
            num_sample = len(env_ids)
        
        # 默认均值: 范围中点
        if mean is None:
            mean = torch.tensor([
                (self.vx_range[0] + self.vx_range[1]) / 2,
                (self.vy_range[0] + self.vy_range[1]) / 2,
                (self.yaw_range[0] + self.yaw_range[1]) / 2,
            ], device=self.device)
        
        # 默认标准差: 范围宽度的1/4
        if std is None:
            std = torch.tensor([
                (self.vx_range[1] - self.vx_range[0]) / 4,
                (self.vy_range[1] - self.vy_range[0]) / 4,
                (self.yaw_range[1] - self.yaw_range[0]) / 4,
            ], device=self.device)
        
        if distribution == 'normal':
            commands = torch.randn(num_sample, 3, device=self.device) * std + mean
        
        elif distribution == 'truncated_normal':
            # 截断正态分布 (在范围内)
            commands = torch.randn(num_sample, 3, device=self.device) * std + mean
            commands[:, 0] = torch.clamp(commands[:, 0], self.vx_range[0], self.vx_range[1])
            commands[:, 1] = torch.clamp(commands[:, 1], self.vy_range[0], self.vy_range[1])
            commands[:, 2] = torch.clamp(commands[:, 2], self.yaw_range[0], self.yaw_range[1])
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return commands
    
    def sample_sequence(
        self,
        duration: int,
        change_interval: int = 50,
        mode: str = 'uniform',
        env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        生成命令序列 (随时间变化)
        
        Args:
            duration: 总时长 (帧数)
            change_interval: 命令变更间隔 (帧数)
            mode: 采样模式 ('uniform', 'curriculum', 'preset')
            env_ids: 环境ID
        
        Returns:
            command_seq: [N, T, 3] 命令序列
        """
        if env_ids is None:
            num_sample = self.num_envs
        else:
            num_sample = len(env_ids)
        
        num_changes = duration // change_interval + 1
        
        # 采样关键点命令
        if mode == 'uniform':
            key_commands = [self.sample_uniform(env_ids) for _ in range(num_changes)]
        elif mode == 'curriculum':
            difficulty = np.linspace(0, 1, num_changes)
            key_commands = [self.sample_curriculum(d, env_ids) for d in difficulty]
        elif mode.startswith('preset:'):
            preset_mode = mode.split(':')[1]
            key_commands = [self.sample_preset(preset_mode, env_ids) for _ in range(num_changes)]
        else:
            raise ValueError(f"Unknown sequence mode: {mode}")
        
        # 线性插值
        command_seq = []
        for i in range(num_changes - 1):
            start_cmd = key_commands[i]
            end_cmd = key_commands[i + 1]
            
            for t in range(change_interval):
                alpha = t / change_interval
                interp_cmd = (1 - alpha) * start_cmd + alpha * end_cmd
                command_seq.append(interp_cmd)
        
        # 添加最后一段
        last_cmd = key_commands[-1]
        for _ in range(duration - len(command_seq)):
            command_seq.append(last_cmd)
        
        return torch.stack(command_seq, dim=1)  # [N, T, 3]


# ============================================================================
# 训练阶段推荐配置
# ============================================================================

def get_stage_sampler(stage: str, num_envs: int, device: str = 'cuda') -> CommandSampler:
    """
    获取适合特定训练阶段的命令采样器
    
    Args:
        stage: 训练阶段
            - 'stage1_basic': 基础前进/后退
            - 'stage2_turn': 添加转向
            - 'stage3_strafe': 添加横向移动
            - 'stage4_mixed': 全方向混合
        num_envs: 环境数量
        device: 计算设备
    
    Returns:
        sampler: 配置好的CommandSampler
    """
    if stage == 'stage1_basic':
        # 阶段1: 纯前进/后退
        sampler = CommandSampler(
            num_envs=num_envs,
            device=device,
            vx_range=(0.3, 2.0),
            vy_range=(0.0, 0.0),
            yaw_range=(0.0, 0.0)
        )
    
    elif stage == 'stage2_turn':
        # 阶段2: 前进+转向
        sampler = CommandSampler(
            num_envs=num_envs,
            device=device,
            vx_range=(0.3, 2.0),
            vy_range=(0.0, 0.0),
            yaw_range=(-0.5, 0.5)
        )
    
    elif stage == 'stage3_strafe':
        # 阶段3: 添加横向移动
        sampler = CommandSampler(
            num_envs=num_envs,
            device=device,
            vx_range=(0.0, 2.0),
            vy_range=(-0.5, 0.5),
            yaw_range=(-0.5, 0.5)
        )
    
    elif stage == 'stage4_mixed':
        # 阶段4: 全方向混合
        sampler = CommandSampler(
            num_envs=num_envs,
            device=device,
            vx_range=(-1.0, 3.0),
            vy_range=(-0.8, 0.8),
            yaw_range=(-0.8, 0.8)
        )
    
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    return sampler


if __name__ == "__main__":
    """测试命令采样器"""
    
    # 创建采样器
    sampler = CommandSampler(num_envs=8, device='cuda')
    
    print("="*80)
    print("测试 CommandSampler")
    print("="*80)
    
    # 测试均匀采样
    print("\n[1] 均匀随机采样:")
    uniform_cmds = sampler.sample_uniform()
    print(uniform_cmds)
    
    # 测试课程学习
    print("\n[2] 课程学习采样 (难度=0.3):")
    curriculum_cmds = sampler.sample_curriculum(difficulty=0.3)
    print(curriculum_cmds)
    
    # 测试预设模式
    print("\n[3] 预设模式:")
    modes = ['forward', 'turn_left', 'strafe_right', 'forward_turn']
    for mode in modes:
        preset_cmds = sampler.sample_preset(mode)
        print(f"  {mode:15s}: {preset_cmds[0].cpu().numpy()}")
    
    # 测试分布采样
    print("\n[4] 正态分布采样:")
    mean = torch.tensor([1.0, 0.0, 0.0], device='cuda')
    std = torch.tensor([0.5, 0.2, 0.2], device='cuda')
    normal_cmds = sampler.sample_distribution('truncated_normal', mean, std)
    print(normal_cmds)
    
    # 测试序列生成
    print("\n[5] 命令序列 (10帧, 每5帧变化):")
    seq = sampler.sample_sequence(duration=10, change_interval=5, mode='uniform')
    print(f"  Shape: {seq.shape}")
    print(f"  First env sequence:")
    for t in range(10):
        print(f"    Frame {t}: {seq[0, t].cpu().numpy()}")
    
    # 测试训练阶段采样器
    print("\n[6] 训练阶段采样器:")
    stages = ['stage1_basic', 'stage2_turn', 'stage3_strafe', 'stage4_mixed']
    for stage in stages:
        stage_sampler = get_stage_sampler(stage, num_envs=8, device='cuda')
        cmds = stage_sampler.sample_uniform()
        print(f"  {stage:20s}: vx={cmds[0, 0]:.2f}, vy={cmds[0, 1]:.2f}, yaw={cmds[0, 2]:.2f}")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
