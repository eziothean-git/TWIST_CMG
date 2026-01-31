#!/usr/bin/env python3
"""
快速诊断脚本：检查CMG参考是否可达
不需要完整训练，只检查参考的物理可行性
"""

import sys
import os
sys.path.insert(0, '/home/eziothean/TWIST_CMG')
sys.path.insert(0, '/home/eziothean/TWIST_CMG/CMG_Ref')
sys.path.insert(0, '/home/eziothean/TWIST_CMG/legged_gym')

import torch
import numpy as np
from termcolor import cprint

def check_cmg_reference_feasibility():
    """检查CMG生成的参考是否物理上可达"""
    
    cprint("\n" + "="*80, "cyan")
    cprint("CMG 参考可行性诊断", "cyan")
    cprint("="*80, "cyan")
    
    # 定义机器人关节范围（基于G1配置）
    JOINT_RANGES = {
        # 腿部关节 (12)
        0:  ('left_hip_pitch', -1.57, 1.57),
        1:  ('left_hip_roll', -0.5, 0.5),
        2:  ('left_hip_yaw', -0.5, 0.5),
        3:  ('left_knee', 0.0, 2.0),
        4:  ('left_ankle_pitch', -1.57, 1.57),
        5:  ('left_ankle_roll', -0.5, 0.5),
        
        6:  ('right_hip_pitch', -1.57, 1.57),
        7:  ('right_hip_roll', -0.5, 0.5),
        8:  ('right_hip_yaw', -0.5, 0.5),
        9:  ('right_knee', 0.0, 2.0),
        10: ('right_ankle_pitch', -1.57, 1.57),
        11: ('right_ankle_roll', -0.5, 0.5),
        
        # 腰部 (3)
        12: ('waist_yaw', -1.57, 1.57),
        13: ('waist_roll', -1.57, 1.57),
        14: ('waist_pitch', -1.57, 1.57),
        
        # 手臂 (8)
        15: ('left_shoulder_pitch', -2.0, 2.0),
        16: ('left_shoulder_roll', -2.0, 0.5),
        17: ('left_shoulder_yaw', -1.57, 1.57),
        18: ('left_elbow', 0.0, 2.0),
        
        22: ('right_shoulder_pitch', -2.0, 2.0),
        23: ('right_shoulder_roll', -0.5, 2.0),
        24: ('right_shoulder_yaw', -1.57, 1.57),
        25: ('right_elbow', 0.0, 2.0),
        
        # 手腕 (6)
        19: ('left_wrist_roll', -1.57, 1.57),
        20: ('left_wrist_pitch', -1.57, 1.57),
        21: ('left_wrist_yaw', -1.57, 1.57),
        26: ('right_wrist_roll', -1.57, 1.57),
        27: ('right_wrist_pitch', -1.57, 1.57),
        28: ('right_wrist_yaw', -1.57, 1.57),
    }
    
    print("\n关节范围定义:")
    print("-" * 80)
    for idx, (name, min_rad, max_rad) in JOINT_RANGES.items():
        print(f"  [{idx:2d}] {name:25s}: [{min_rad:6.3f}, {max_rad:6.3f}] rad")
    
    # 尝试加载CMG模型
    try:
        from pose.utils.motion_lib_cmg import MotionLibCMG
        
        # 查找CMG配置
        cmg_model_path = "/home/eziothean/TWIST_CMG/CMG_Ref/model.pt"
        cmg_data_path = "/home/eziothean/TWIST_CMG/CMG_Ref/data.pt"
        
        if not os.path.exists(cmg_model_path):
            # 尝试寻找其他路径
            possible_paths = [
                "/home/eziothean/TWIST_CMG/CMG_Ref/checkpoints/latest.pt",
                "/home/eziothean/TWIST_CMG/CMG_Ref/checkpoints/model.pt",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    cmg_model_path = path
                    break
        
        if not os.path.exists(cmg_model_path):
            cprint(f"\n❌ CMG模型文件不存在: {cmg_model_path}", "red")
            cprint("    请检查路径或确保CMG模型已生成", "yellow")
            return False
        
        cprint(f"\n✓ 发现CMG模型: {cmg_model_path}", "green")
        
        # 初始化MotionLibCMG
        print("\n初始化CMG动作库...")
        motion_lib = MotionLibCMG(
            cmg_model_path=cmg_model_path,
            cmg_data_path=cmg_data_path,
            num_motions=256,  # 减少数量以加快诊断
            motion_length=500,
            num_envs=1,
            device="cuda:0",
            dof_dim=29,
            fps=50.0
        )
        
        # 检查生成的参考位置范围
        print("\n" + "-" * 80)
        print("生成的参考位置范围分析:")
        print("-" * 80)
        
        dof_pos_pool = motion_lib.dof_pos_pool  # [num_motions, T, 29]
        
        # 全局范围
        global_min = dof_pos_pool.min().item()
        global_max = dof_pos_pool.max().item()
        print(f"\n全局范围: [{global_min:.4f}, {global_max:.4f}]")
        
        # 按关节检查
        print("\n按关节检查:")
        print("-" * 80)
        
        violations = 0  # 超出范围的关节数
        
        for idx in range(29):
            joint_min = dof_pos_pool[..., idx].min().item()
            joint_max = dof_pos_pool[..., idx].max().item()
            
            if idx in JOINT_RANGES:
                name, allowed_min, allowed_max = JOINT_RANGES[idx]
                
                # 检查是否超出范围
                exceeds_min = joint_min < allowed_min - 0.1  # 允许10度偏差
                exceeds_max = joint_max > allowed_max + 0.1
                
                status = "✓" if not (exceeds_min or exceeds_max) else "❌"
                violations += (exceeds_min or exceeds_max)
                
                print(f"  {status} [{idx:2d}] {name:25s}: [{joint_min:7.4f}, {joint_max:7.4f}]  (allowed: [{allowed_min:6.3f}, {allowed_max:6.3f}])")
            else:
                print(f"  ? [{idx:2d}] UNKNOWN                : [{joint_min:7.4f}, {joint_max:7.4f}]")
        
        print("\n" + "=" * 80)
        if violations == 0:
            cprint("✓ 所有关节都在合理范围内!", "green")
            cprint("  → CMG 生成的参考应该是物理上可达的", "green")
        else:
            cprint(f"❌ 发现 {violations} 个关节超出范围!", "red")
            cprint("  → 机器人可能无法达到这些参考角度", "yellow")
            cprint("  → 这可能导致 reward 始终为0", "yellow")
        
        print("=" * 80 + "\n")
        return violations == 0
        
    except Exception as e:
        cprint(f"\n❌ 加载CMG模型时出错: {e}", "red")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = check_cmg_reference_feasibility()
    sys.exit(0 if success else 1)
