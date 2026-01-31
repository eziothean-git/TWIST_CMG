#!/usr/bin/env python3
"""
本地调试脚本：验证reward计算逻辑
不需要GPU运行，纯逻辑检查
"""

import torch
import numpy as np

def verify_reward_calculation():
    """验证reward计算的基本逻辑"""
    print("="*80)
    print("Reward 计算逻辑验证")
    print("="*80)
    
    # 模拟场景：机器人在第2000步时"张开腿等死"
    num_envs = 4096
    num_dof = 29
    
    # 场景1：机器人完全不动（只是初始化的角度）
    print("\n【场景1】机器人完全不动（零速度）")
    ref_dof_pos = torch.randn(num_envs, num_dof) * 0.5
    dof_pos = torch.ones(num_envs, num_dof) * 0.2  # 固定在初始位置
    dof_vel = torch.zeros(num_envs, num_dof)  # 零速度
    ref_dof_vel = torch.randn(num_envs, num_dof) * 2.0
    
    dof_diff = ref_dof_pos - dof_pos
    dof_vel_diff = ref_dof_vel - dof_vel
    
    # tracking_joint_dof reward
    dof_err = torch.sum(dof_diff * dof_diff, dim=-1)
    tracking_dof_rew = torch.exp(-0.15 * dof_err)
    
    # tracking_joint_vel reward  
    vel_err = torch.sum(dof_vel_diff * dof_vel_diff, dim=-1)
    tracking_vel_rew = torch.exp(-0.01 * vel_err)
    
    print(f"  dof位置误差 (L2平均): {dof_err.mean():.4f}")
    print(f"  tracking_dof reward: min={tracking_dof_rew.min():.6f}, mean={tracking_dof_rew.mean():.6f}, max={tracking_dof_rew.max():.6f}")
    print(f"  dof速度误差 (L2平均): {vel_err.mean():.4f}")
    print(f"  tracking_vel reward: min={tracking_vel_rew.min():.6f}, mean={tracking_vel_rew.mean():.6f}, max={tracking_vel_rew.max():.6f}")
    
    # 假设权重配置
    tracking_dof_weight = 0.84
    tracking_vel_weight = 0.28
    
    weighted_reward_1 = tracking_dof_rew * tracking_dof_weight + tracking_vel_rew * tracking_vel_weight
    print(f"  加权reward: mean={weighted_reward_1.mean():.6f} (tracking_dof占比{tracking_dof_weight:.2f}, tracking_vel占比{tracking_vel_weight:.2f})")
    
    # 场景2：参考动作完全不可能达到（超出机器人范围）
    print("\n【场景2】参考动作物理上不可达（超出范围）")
    ref_dof_pos_extreme = torch.ones(num_envs, num_dof) * 3.0  # 极端角度 3 rad
    dof_pos_normal = torch.zeros(num_envs, num_dof)  # 机器人正常范围
    
    dof_diff_extreme = ref_dof_pos_extreme - dof_pos_normal
    dof_err_extreme = torch.sum(dof_diff_extreme * dof_diff_extreme, dim=-1)
    tracking_dof_rew_extreme = torch.exp(-0.15 * dof_err_extreme)
    
    print(f"  dof位置误差 (L2平均): {dof_err_extreme.mean():.4f}")
    print(f"  tracking_dof reward: {tracking_dof_rew_extreme.mean():.6f} (近似为0)")
    print(f"  ⚠️ 警告：如果参考动作不可达，tracking reward会近似为0，模型学不到有用的信号!")
    
    # 场景3：检查权重配置
    print("\n【场景3】权重配置分析")
    
    weights = {
        'tracking_joint_dof': 0.84,
        'tracking_joint_vel': 0.28,
        'tracking_lin_vel': 0.49,
        'tracking_ang_vel': 0.25,
        'regularization': 0.1,  # 假设
        'action_rate': 0.01,    # 假设
        'stance': 0.1,          # 假设
    }
    
    print("  权重分布:")
    total_weight = sum(weights.values())
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        percentage = w / total_weight * 100
        print(f"    {name:25s}: {w:6.3f} ({percentage:5.1f}%)")
    
    print(f"  总权重: {total_weight:.3f}")
    
    # 关键问题
    print("\n【关键问题分析】")
    tracking_total_weight = weights['tracking_joint_dof'] + weights['tracking_joint_vel'] + \
                           weights['tracking_lin_vel'] + weights['tracking_ang_vel']
    reg_total_weight = weights['regularization'] + weights['action_rate'] + weights['stance']
    
    tracking_ratio = tracking_total_weight / total_weight * 100
    reg_ratio = reg_total_weight / total_weight * 100
    
    print(f"  Tracking reward 占比: {tracking_ratio:.1f}%")
    print(f"  Regularization reward 占比: {reg_ratio:.1f}%")
    
    if tracking_ratio < 50:
        print(f"  ⚠️ 警告：Tracking reward 权重可能不足，模型可能学到的是最小化运动而不是跟踪!")
    else:
        print(f"  ✓ Tracking reward 权重充足")
    
    print("\n" + "="*80)
    print("结论:")
    print("="*80)
    print("""
如果机器人在2000步后"张开腿等死"，可能原因：

1. 【最可能】Reward 不足以驱动跟踪
   - tracking_dof/vel reward 可能太小 (exp(-大数字) ≈ 0)
   - 导致模型学到"不动"这个局部最小值

2. 【次可能】CMG生成的参考动作物理上不可达
   - 机器人无法到达这些角度
   - Reward 一直接近0，无法提供学习信号

3. 【可能】权重配置出问题
   - 某些惩罚项（如action_rate）压制了tracking reward
   - 导致模型放弃跟踪

4. 【可能】学习率过高导致梯度爆炸
   - 策略突然崩溃到一个死亡状态

建议立即查看训练日志中的 reward 分解，确认：
✓ tracking_joint_dof reward 是否 > 0.5 (在早期)
✓ 是否有其他reward项在压制tracking
✓ action_scale 和学习率是否过高
    """)

if __name__ == '__main__':
    verify_reward_calculation()
