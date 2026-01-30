"""
CMGMotionGenerator 集成示例
演示如何在TWIST训练中使用双模式生成器
"""

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.cmg_motion_generator import CMGMotionGenerator, CommandSmoother
from utils.command_sampler import CommandSampler, get_stage_sampler


def example_1_pregenerated_training():
    """
    示例1: 预生成模式用于训练冷启动
    推荐用于训练初期 (0-5k iterations)
    """
    print("\n" + "="*80)
    print("示例1: 预生成模式 - 训练冷启动")
    print("="*80)
    
    # 配置
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    NUM_ENVS = 4096
    MAX_STEPS = 500
    
    # 创建CMG生成器 (预生成模式)
    cmg_generator = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=NUM_ENVS,
        device='cuda',
        mode='pregenerated',
        preload_duration=500
    )
    
    # 创建命令采样器
    cmd_sampler = get_stage_sampler('stage1_basic', NUM_ENVS, device='cuda')
    
    # 初始化环境
    print(f"\n初始化 {NUM_ENVS} 个环境...")
    init_commands = cmd_sampler.sample_uniform()
    cmg_generator.reset(commands=init_commands)
    
    # 模拟训练循环
    print(f"\n开始训练循环 ({MAX_STEPS} 步)...")
    for step in range(MAX_STEPS):
        ref_dof_pos, ref_dof_vel = cmg_generator.get_motion()
        
        if step % 100 == 0:
            print(f"  Step {step}: ref_pos_mean={ref_dof_pos.mean():.3f}")
    
    stats = cmg_generator.get_performance_stats()
    print(f"\n性能: 平均推理 {stats['avg_inference_ms']:.2f} ms")


def example_2_realtime_dynamic():
    """示例2: 实时模式用于动态命令跟踪"""
    print("\n" + "="*80)
    print("示例2: 实时模式 - 动态命令跟踪")
    print("="*80)
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    NUM_ENVS = 4096
    
    cmg_generator = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=NUM_ENVS,
        device='cuda',
        mode='realtime',
        buffer_size=100
    )
    
    cmd_sampler = get_stage_sampler('stage4_mixed', NUM_ENVS, device='cuda')
    cmd_smoother = CommandSmoother(NUM_ENVS, interpolation_steps=25, device='cuda')
    
    init_commands = cmd_sampler.sample_uniform()
    cmg_generator.reset(commands=init_commands)
    cmd_smoother.set_target(init_commands)
    
    print(f"\n训练循环 (每50步更新命令)...")
    for step in range(500):
        smooth_cmd = cmd_smoother.get_current()
        
        if step % 50 == 0:
            new_target = cmd_sampler.sample_uniform()
            cmd_smoother.set_target(new_target)
            print(f"  Step {step}: 更新命令")
        
        ref_dof_pos, ref_dof_vel = cmg_generator.get_motion()
    
    stats = cmg_generator.get_performance_stats()
    print(f"\n性能: 平均推理 {stats['avg_inference_ms']:.2f} ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    
    if args.example == 1:
        example_1_pregenerated_training()
    else:
        example_2_realtime_dynamic()
    
    print("\n完成!")
