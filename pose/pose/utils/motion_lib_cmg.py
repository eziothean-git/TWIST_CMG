"""
MotionLibCMG: 使用CMG生成参考动作的MotionLib实现

替代原有的从pkl文件加载mocap数据的方式，使用CMG从速度命令生成参考动作。
用于TWIST冷启动阶段的locomotion训练。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加CMG_Ref路径
# 当这个文件被导入时，通常是通过 from pose.utils.motion_lib_cmg import ...
# 所以我们需要找到TWIST_CMG根目录

def _get_project_root():
    """获取TWIST_CMG根目录"""
    # motion_lib_cmg.py 的路径: TWIST_CMG/pose/pose/utils/motion_lib_cmg.py
    current_file = Path(__file__).resolve()
    # 往上4级: utils -> pose -> pose -> TWIST_CMG
    return current_file.parent.parent.parent.parent

PROJECT_ROOT = _get_project_root()
CMG_REF_DIR = PROJECT_ROOT / "CMG_Ref"

# 确保路径添加到sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 设置CMG_Ref模块路径
CMG_MODULE_DIR = str(CMG_REF_DIR / "module")
CMG_UTILS_DIR = str(CMG_REF_DIR / "utils")

if CMG_MODULE_DIR not in sys.path:
    sys.path.insert(0, CMG_MODULE_DIR)
if CMG_UTILS_DIR not in sys.path:
    sys.path.insert(0, CMG_UTILS_DIR)
if str(CMG_REF_DIR) not in sys.path:
    sys.path.insert(0, str(CMG_REF_DIR))

# 直接导入CMGMotionGenerator - 路径已正确设置
from cmg_motion_generator import CMGMotionGenerator
from command_sampler import CommandSampler


class MotionLibCMG:
    """
    使用CMG生成参考动作，替代mocap数据加载
    
    提供与MotionLib兼容的接口，内部使用CMGMotionGenerator生成参考轨迹
    """
    
    def __init__(self, cmg_model_path, cmg_data_path, num_envs, device, 
                 mode='pregenerated', preload_duration=500, buffer_size=100):
        """
        初始化CMG Motion Library
        
        Args:
            cmg_model_path: CMG模型权重路径
            cmg_data_path: CMG训练数据路径（用于统计信息）
            num_envs: 并行环境数量
            device: PyTorch设备
            mode: 'pregenerated'（预生成完整轨迹）或 'realtime'（实时生成）
            preload_duration: 预生成模式下的轨迹长度（帧数）
            buffer_size: 实时模式下的缓冲区大小
        """
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        
        print(f"[MotionLibCMG] 初始化 CMG Motion Generator")
        print(f"  - 模式: {mode}")
        print(f"  - 环境数: {num_envs}")
        print(f"  - 设备: {device}")
        
        # 初始化CMG生成器（已通过sys.path导入）
        self.generator = CMGMotionGenerator(
            model_path=cmg_model_path,
            data_path=cmg_data_path,
            num_envs=num_envs,
            mode=mode,
            preload_duration=preload_duration,
            buffer_size=buffer_size,
            device=device
        )
        
        # 初始化命令采样器（已通过sys.path导入）
        self.command_sampler = CommandSampler(num_envs, device=device)
        
        # 状态追踪
        self.current_frame = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_lengths = torch.full((num_envs,), float(preload_duration), device=device)
        
        # 固定参数（CMG生成的动作特性）
        self.fps = 50.0
        self.dt = 1.0 / self.fps
        
        print(f"[MotionLibCMG] 初始化完成")
    
    def num_motions(self):
        """返回"动作"数量（对于CMG来说是环境数）"""
        return self.num_envs
    
    def get_total_length(self):
        """返回总动作长度"""
        return self.motion_lengths.sum().item() * self.dt
    
    def get_motion_length(self, motion_ids):
        """返回指定动作的长度"""
        if isinstance(motion_ids, int):
            return self.motion_lengths[motion_ids] * self.dt
        return self.motion_lengths[motion_ids] * self.dt
    
    def sample_motions(self, n, curriculum_level=None):
        """
        采样速度命令并生成对应的参考动作
        
        Args:
            n: 采样数量
            curriculum_level: 课程学习级别（0-1，可选）
        
        Returns:
            motion_ids: 环境索引 (相当于"动作ID")
        """
        # 根据课程学习级别选择采样策略
        if curriculum_level is not None:
            if curriculum_level < 0.25:
                strategy = 'stage1_basic'  # 前进
            elif curriculum_level < 0.5:
                strategy = 'stage2_turn'   # 前进+转向
            elif curriculum_level < 0.75:
                strategy = 'stage3_strafe' # 全方向
            else:
                strategy = 'stage4_mixed'  # 动态混合
        else:
            strategy = 'uniform_random'
        
        # 采样速度命令
        commands = self.command_sampler.sample_batch(n, strategy=strategy)
        commands = torch.from_numpy(commands).to(self.device)
        
        # 生成参考轨迹
        env_ids = torch.arange(n, device=self.device)
        self.generator.reset(env_ids=env_ids, commands=commands)
        
        # 重置帧计数
        self.current_frame[env_ids] = 0
        
        return env_ids
    
    def reset(self, env_ids, commands=None):
        """
        重置指定环境的参考轨迹
        
        Args:
            env_ids: 要重置的环境索引
            commands: 速度命令 [N, 3] (vx, vy, yaw_rate)，如果为None则随机采样
        """
        if commands is None:
            # 随机采样命令
            n = len(env_ids)
            commands = self.command_sampler.sample_batch(n, strategy='uniform_random')
            commands = torch.from_numpy(commands).to(self.device)
        
        # 生成新轨迹
        self.generator.reset(env_ids=env_ids, commands=commands)
        
        # 重置帧计数
        self.current_frame[env_ids] = 0
    
    def update_commands(self, commands):
        """
        更新速度命令（仅实时模式）
        
        Args:
            commands: 新的速度命令 [num_envs, 3]
        """
        if self.mode == 'realtime':
            self.generator.update_commands(commands)
    
    def get_motion_state(self):
        """
        获取当前帧的参考状态
        
        Returns:
            ref_dof_pos: 参考关节位置 [num_envs, 29]
            ref_dof_vel: 参考关节速度 [num_envs, 29]
        """
        ref_dof_pos, ref_dof_vel = self.generator.get_motion()
        
        # 更新帧计数
        self.current_frame += 1
        
        return ref_dof_pos, ref_dof_vel
    
    def calc_motion_frame(self, motion_ids, motion_times):
        """
        计算指定时刻的动作帧（兼容原MotionLib接口）
        
        Note: CMG使用自回归生成，不支持随机时间访问
              这里返回当前生成的帧
        
        Args:
            motion_ids: 环境索引
            motion_times: 时间偏移（秒）
        
        Returns:
            root_pos, root_rot, root_vel, root_ang_vel: 全部返回None（CMG不生成root）
            dof_pos: 关节位置 [N, 29]
            dof_vel: 关节速度 [N, 29]
            local_key_body_pos: None（CMG不生成body位置，需要FK计算）
        """
        # 获取当前帧
        ref_dof_pos, ref_dof_vel = self.get_motion_state()
        
        # 提取指定环境的数据
        if motion_ids is not None:
            ref_dof_pos = ref_dof_pos[motion_ids]
            ref_dof_vel = ref_dof_vel[motion_ids]
        
        # CMG不生成root和body位置，返回None
        # 这些信息需要环境通过FK或仿真计算
        root_pos = None
        root_rot = None
        root_vel = None
        root_ang_vel = None
        local_key_body_pos = None
        
        return root_pos, root_rot, root_vel, root_ang_vel, ref_dof_pos, ref_dof_vel, local_key_body_pos
    
    def get_motion_names(self):
        """返回"动作"名称列表（对于CMG来说是环境ID）"""
        return [f"cmg_env_{i}" for i in range(self.num_envs)]
    
    def get_key_body_idx(self, key_body_names):
        """
        返回关键body的索引
        
        Note: CMG不生成body信息，返回空列表
              环境需要使用FK计算body位置
        """
        print("[MotionLibCMG] Warning: CMG不生成body位置，需要使用FK计算")
        return []
    
    def get_stats(self):
        """返回统计信息"""
        return {
            'num_envs': self.num_envs,
            'mode': self.mode,
            'fps': self.fps,
            'dt': self.dt,
        }


def create_motion_lib_cmg(cfg, num_envs, device):
    """
    工厂函数：根据配置创建MotionLibCMG实例
    
    Args:
        cfg: 配置对象（应包含motion.cmg_*字段）
        num_envs: 并行环境数量
        device: PyTorch设备
    
    Returns:
        MotionLibCMG实例
    """
    motion_cfg = cfg.motion
    
    # 构建路径
    cmg_model_path = motion_cfg.cmg_model_path
    cmg_data_path = motion_cfg.cmg_data_path
    
    # 检查文件是否存在
    if not os.path.exists(cmg_model_path):
        raise FileNotFoundError(f"CMG模型不存在: {cmg_model_path}")
    if not os.path.exists(cmg_data_path):
        raise FileNotFoundError(f"CMG数据不存在: {cmg_data_path}")
    
    return MotionLibCMG(
        cmg_model_path=cmg_model_path,
        cmg_data_path=cmg_data_path,
        num_envs=num_envs,
        device=device,
        mode=motion_cfg.cmg_mode,
        preload_duration=motion_cfg.cmg_preload_duration,
        buffer_size=motion_cfg.cmg_buffer_size
    )
