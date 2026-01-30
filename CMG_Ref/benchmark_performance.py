"""
性能分析工具
用于评估CMGMotionGenerator在不同配置下的性能
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.cmg_motion_generator import CMGMotionGenerator
from utils.command_sampler import CommandSampler


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.results = []
    
    def benchmark_mode(
        self,
        mode: str,
        num_envs: int,
        duration: int = 200,
        preload_duration: int = 500,
        buffer_size: int = 100
    ) -> Dict:
        """测试单个配置"""
        print(f"\n测试: mode={mode}, num_envs={num_envs}")
        
        # 创建生成器
        generator = CMGMotionGenerator(
            model_path=self.model_path,
            data_path=self.data_path,
            num_envs=num_envs,
            device='cuda',
            mode=mode,
            preload_duration=preload_duration,
            buffer_size=buffer_size
        )
        
        sampler = CommandSampler(num_envs, device='cuda')
        commands = sampler.sample_uniform()
        
        # 测试重置
        torch.cuda.synchronize()
        start = time.time()
        generator.reset(commands=commands)
        torch.cuda.synchronize()
        reset_time = (time.time() - start) * 1000
        
        # 测试获取性能
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(duration):
            ref_pos, ref_vel = generator.get_motion()
        torch.cuda.synchronize()
        total_time = time.time() - start
        
        per_frame_time = total_time / duration * 1000
        throughput = num_envs * duration / total_time
        
        # 内存使用
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        
        result = {
            'mode': mode,
            'num_envs': num_envs,
            'reset_time_ms': reset_time,
            'per_frame_ms': per_frame_time,
            'throughput_fps': throughput,
            'memory_mb': memory_allocated,
            'duration': duration,
        }
        
        print(f"  重置时间: {reset_time:.2f} ms")
        print(f"  每帧时间: {per_frame_time:.4f} ms")
        print(f"  吞吐量: {throughput:.0f} frames/s")
        print(f"  内存: {memory_allocated:.1f} MB")
        
        self.results.append(result)
        return result
    
    def benchmark_scaling(self):
        """测试不同规模的性能"""
        print("\n" + "="*80)
        print("规模测试")
        print("="*80)
        
        env_sizes = [512, 1024, 2048, 4096]
        
        for mode in ['pregenerated', 'realtime']:
            print(f"\n{mode.upper()} 模式:")
            for num_envs in env_sizes:
                self.benchmark_mode(mode, num_envs, duration=100)
    
    def benchmark_preload_sizes(self):
        """测试不同预生成长度的影响"""
        print("\n" + "="*80)
        print("预生成长度测试")
        print("="*80)
        
        preload_sizes = [100, 200, 300, 500, 1000]
        num_envs = 2048
        
        for preload_duration in preload_sizes:
            print(f"\nPreload Duration: {preload_duration} frames ({preload_duration/50}s)")
            self.benchmark_mode(
                'pregenerated',
                num_envs,
                duration=50,
                preload_duration=preload_duration
            )
    
    def benchmark_buffer_sizes(self):
        """测试不同缓冲区大小的影响"""
        print("\n" + "="*80)
        print("缓冲区大小测试")
        print("="*80)
        
        buffer_sizes = [50, 100, 150, 200]
        num_envs = 2048
        
        for buffer_size in buffer_sizes:
            print(f"\nBuffer Size: {buffer_size} frames ({buffer_size/50}s)")
            self.benchmark_mode(
                'realtime',
                num_envs,
                duration=200,
                buffer_size=buffer_size
            )
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "="*80)
        print("性能总结")
        print("="*80)
        
        # 按模式分组
        pregen_results = [r for r in self.results if r['mode'] == 'pregenerated']
        realtime_results = [r for r in self.results if r['mode'] == 'realtime']
        
        if pregen_results:
            print("\n预生成模式:")
            print(f"{'Envs':<8} {'重置(ms)':<12} {'每帧(ms)':<12} {'吞吐量(fps)':<15} {'内存(MB)':<10}")
            print("-" * 70)
            for r in pregen_results:
                print(f"{r['num_envs']:<8} {r['reset_time_ms']:<12.2f} "
                      f"{r['per_frame_ms']:<12.4f} {r['throughput_fps']:<15.0f} "
                      f"{r['memory_mb']:<10.1f}")
        
        if realtime_results:
            print("\n实时模式:")
            print(f"{'Envs':<8} {'重置(ms)':<12} {'每帧(ms)':<12} {'吞吐量(fps)':<15} {'内存(MB)':<10}")
            print("-" * 70)
            for r in realtime_results:
                print(f"{r['num_envs']:<8} {r['reset_time_ms']:<12.2f} "
                      f"{r['per_frame_ms']:<12.4f} {r['throughput_fps']:<15.0f} "
                      f"{r['memory_mb']:<10.1f}")
    
    def save_results(self, output_path: str = 'benchmark_results.txt'):
        """保存结果"""
        with open(output_path, 'w') as f:
            f.write("CMG Motion Generator Performance Benchmark\n")
            f.write("="*80 + "\n\n")
            
            for result in self.results:
                f.write(f"Mode: {result['mode']}\n")
                f.write(f"Num Envs: {result['num_envs']}\n")
                f.write(f"Reset Time: {result['reset_time_ms']:.2f} ms\n")
                f.write(f"Per Frame: {result['per_frame_ms']:.4f} ms\n")
                f.write(f"Throughput: {result['throughput_fps']:.0f} frames/s\n")
                f.write(f"Memory: {result['memory_mb']:.1f} MB\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"\n结果已保存到: {output_path}")


def quick_performance_check():
    """快速性能检查"""
    print("\n" + "="*80)
    print("快速性能检查")
    print("="*80)
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    # 测试4096环境
    num_envs = 4096
    
    # 预生成模式
    print(f"\n测试预生成模式 ({num_envs} envs)...")
    gen_pregen = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='pregenerated',
        preload_duration=200
    )
    
    sampler = CommandSampler(num_envs, device='cuda')
    commands = sampler.sample_uniform()
    
    # 测试重置
    torch.cuda.synchronize()
    start = time.time()
    gen_pregen.reset(commands=commands)
    torch.cuda.synchronize()
    reset_time = (time.time() - start) * 1000
    print(f"  重置时间: {reset_time:.2f} ms")
    
    # 测试获取
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref_pos, ref_vel = gen_pregen.get_motion()
    torch.cuda.synchronize()
    get_time = (time.time() - start) * 10
    print(f"  获取时间: {get_time:.4f} ms/frame")
    
    # 实时模式
    print(f"\n测试实时模式 ({num_envs} envs)...")
    gen_rt = CMGMotionGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        num_envs=num_envs,
        device='cuda',
        mode='realtime',
        buffer_size=100
    )
    
    gen_rt.reset(commands=commands)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref_pos, ref_vel = gen_rt.get_motion()
    torch.cuda.synchronize()
    get_time_rt = (time.time() - start) * 10
    print(f"  获取时间: {get_time_rt:.4f} ms/frame")
    
    stats = gen_rt.get_performance_stats()
    print(f"  推理时间: {stats['avg_inference_ms']:.2f} ms (平均)")
    
    # 内存使用
    memory_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"\n总内存使用: {memory_mb:.1f} MB")
    
    # 评估
    print("\n性能评估:")
    if reset_time < 300:
        print("  ✅ 重置时间优秀 (< 300ms)")
    else:
        print("  ⚠️  重置时间较长 (> 300ms)")
    
    if get_time < 1.0:
        print("  ✅ 获取速度优秀 (< 1ms/frame)")
    else:
        print("  ⚠️  获取速度较慢 (> 1ms/frame)")
    
    if stats['avg_inference_ms'] < 100:
        print("  ✅ 推理速度优秀 (< 100ms)")
    else:
        print("  ⚠️  推理速度较慢 (> 100ms)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CMG性能测试')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'scaling', 'preload', 'buffer'])
    parser.add_argument('--save', action='store_true', help='保存结果到文件')
    args = parser.parse_args()
    
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    if args.mode == 'quick':
        quick_performance_check()
    
    else:
        benchmark = PerformanceBenchmark(MODEL_PATH, DATA_PATH)
        
        if args.mode == 'full':
            benchmark.benchmark_scaling()
            benchmark.benchmark_preload_sizes()
            benchmark.benchmark_buffer_sizes()
        elif args.mode == 'scaling':
            benchmark.benchmark_scaling()
        elif args.mode == 'preload':
            benchmark.benchmark_preload_sizes()
        elif args.mode == 'buffer':
            benchmark.benchmark_buffer_sizes()
        
        benchmark.print_summary()
        
        if args.save:
            benchmark.save_results()
    
    print("\n" + "="*80)
    print("性能测试完成！")
    print("="*80)
