"""
测试 CMGMotionGenerator 的性能和正确性
"""

import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.cmg_motion_generator import CMGMotionGenerator, CommandSmoother


def test_pregenerated_mode():
    """测试预生成模式"""
    print("\n" + "="*80)
    print("测试预生成模式 (Pregenerated Mode)")
    print("="*80)
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    # 创建生成器 (小规模测试)
    num_envs = 256
    generator = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='pregenerated',
        preload_duration=100  # 2秒轨迹
    )
    
    # 测试: 重置并生成轨迹
    commands = torch.tensor([
        [1.0, 0.0, 0.0],  # 前进
        [0.0, 0.5, 0.0],  # 左移
        [0.0, 0.0, 0.5],  # 旋转
    ], device='cuda').repeat(num_envs // 3 + 1, 1)[:num_envs]
    
    print(f"\n重置环境并生成轨迹...")
    generator.reset(commands=commands)
    
    # 测试: 获取动作
    print(f"\n获取动作帧...")
    for i in range(5):
        dof_pos, dof_vel = generator.get_motion()
        print(f"  Frame {i}: dof_pos shape={dof_pos.shape}, "
              f"pos_mean={dof_pos.mean():.3f}, vel_mean={dof_vel.mean():.3f}")
    
    # 性能统计
    stats = generator.get_performance_stats()
    print(f"\n性能统计:")
    print(f"  平均推理时间: {stats['avg_inference_ms']:.2f} ms")
    print(f"  最大推理时间: {stats['max_inference_ms']:.2f} ms")
    print(f"  最小推理时间: {stats['min_inference_ms']:.2f} ms")


def test_realtime_mode():
    """测试实时模式"""
    print("\n" + "="*80)
    print("测试实时模式 (Realtime Mode)")
    print("="*80)
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    # 创建生成器
    num_envs = 256
    generator = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='realtime',
        buffer_size=100
    )
    
    # 初始化
    commands = torch.tensor([[1.0, 0.0, 0.0]], device='cuda').repeat(num_envs, 1)
    generator.reset(commands=commands)
    
    # 测试: 获取动作
    print(f"\n获取动作帧...")
    for i in range(10):
        dof_pos, dof_vel = generator.get_motion()
        print(f"  Frame {i}: dof_pos shape={dof_pos.shape}, "
              f"pos_mean={dof_pos.mean():.3f}, vel_mean={dof_vel.mean():.3f}")
    
    # 测试: 更新命令
    print(f"\n更新命令...")
    new_commands = torch.tensor([[0.0, 0.5, 0.5]], device='cuda').repeat(num_envs, 1)
    generator.update_commands(new_commands)
    
    for i in range(5):
        dof_pos, dof_vel = generator.get_motion()
        print(f"  Frame {i} (new cmd): pos_mean={dof_pos.mean():.3f}, vel_mean={dof_vel.mean():.3f}")
    
    # 性能统计
    stats = generator.get_performance_stats()
    print(f"\n性能统计:")
    print(f"  平均推理时间: {stats['avg_inference_ms']:.2f} ms")
    print(f"  最大推理时间: {stats['max_inference_ms']:.2f} ms")


def test_large_scale():
    """测试大规模并行 (4096环境)"""
    print("\n" + "="*80)
    print("测试大规模并行 (4096 Environments)")
    print("="*80)
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    num_envs = 4096
    
    # 测试预生成模式
    print(f"\n[预生成模式] {num_envs} 环境")
    generator_pregen = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='pregenerated',
        preload_duration=200  # 4秒轨迹
    )
    
    commands = torch.randn(num_envs, 3, device='cuda') * 0.5
    generator_pregen.reset(commands=commands)
    
    # 测试吞吐量
    import time
    num_frames = 100
    start = time.time()
    for _ in range(num_frames):
        dof_pos, dof_vel = generator_pregen.get_motion()
    elapsed = time.time() - start
    
    print(f"  生成 {num_frames} 帧耗时: {elapsed:.2f}s")
    print(f"  吞吐量: {num_frames * num_envs / elapsed:.0f} frames/s")
    print(f"  每帧延迟: {elapsed / num_frames * 1000:.2f} ms")
    
    stats = generator_pregen.get_performance_stats()
    print(f"  平均推理: {stats['avg_inference_ms']:.2f} ms")
    
    # 测试实时模式
    print(f"\n[实时模式] {num_envs} 环境")
    generator_rt = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='realtime',
        buffer_size=100
    )
    
    generator_rt.reset(commands=commands)
    
    start = time.time()
    for _ in range(num_frames):
        dof_pos, dof_vel = generator_rt.get_motion()
    elapsed = time.time() - start
    
    print(f"  生成 {num_frames} 帧耗时: {elapsed:.2f}s")
    print(f"  吞吐量: {num_frames * num_envs / elapsed:.0f} frames/s")
    print(f"  每帧延迟: {elapsed / num_frames * 1000:.2f} ms")
    
    stats = generator_rt.get_performance_stats()
    print(f"  平均推理: {stats['avg_inference_ms']:.2f} ms")


def test_command_smoother():
    """测试命令平滑器"""
    print("\n" + "="*80)
    print("测试命令平滑器 (Command Smoother)")
    print("="*80)
    
    num_envs = 8
    smoother = CommandSmoother(num_envs=num_envs, interpolation_steps=10, device='cuda')
    
    # 设置目标命令
    target = torch.tensor([[1.0, 0.5, 0.2]], device='cuda').repeat(num_envs, 1)
    smoother.set_target(target)
    
    print(f"\n平滑过程:")
    for i in range(15):
        current = smoother.get_current()
        print(f"  Step {i:2d}: {current[0].cpu().numpy()}")
    
    # 切换新目标
    print(f"\n切换到新目标:")
    new_target = torch.tensor([[-0.5, 1.0, -0.3]], device='cuda').repeat(num_envs, 1)
    smoother.set_target(new_target)
    
    for i in range(15):
        current = smoother.get_current()
        print(f"  Step {i:2d}: {current[0].cpu().numpy()}")


def test_mode_switching():
    """测试模式切换"""
    print("\n" + "="*80)
    print("测试模式切换 (Mode Switching)")
    print("="*80)
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    num_envs = 256
    generator = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='pregenerated',
        preload_duration=100
    )
    
    # 在预生成模式运行
    commands = torch.randn(num_envs, 3, device='cuda') * 0.5
    generator.reset(commands=commands)
    
    print(f"\n预生成模式运行 10 帧:")
    for i in range(10):
        dof_pos, dof_vel = generator.get_motion()
        print(f"  Frame {i}: pos_mean={dof_pos.mean():.3f}")
    
    # 切换到实时模式
    generator.switch_mode('realtime')
    
    print(f"\n实时模式运行 10 帧:")
    for i in range(10):
        dof_pos, dof_vel = generator.get_motion()
        print(f"  Frame {i}: pos_mean={dof_pos.mean():.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all', 
                       choices=['all', 'pregen', 'realtime', 'large', 'smoother', 'switch'])
    args = parser.parse_args()
    
    if args.test in ['all', 'pregen']:
        test_pregenerated_mode()
    
    if args.test in ['all', 'realtime']:
        test_realtime_mode()
    
    if args.test in ['all', 'large']:
        test_large_scale()
    
    if args.test in ['all', 'smoother']:
        test_command_smoother()
    
    if args.test in ['all', 'switch']:
        test_mode_switching()
    
    print("\n" + "="*80)
    print("所有测试完成！")
    print("="*80)
