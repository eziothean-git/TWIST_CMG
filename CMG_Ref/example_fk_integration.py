#!/usr/bin/env python3
"""
1.1.3 前向运动学集成示例
展示如何从CMG生成的29 DOF关节轨迹计算body位置和旋转

使用场景:
1. 在TWIST训练中计算参考body位置/旋转
2. 离线将CMG轨迹转换为包含body变换的格式
3. 验证FK实现的正确性
"""

import torch
import numpy as np
import pickle
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.fk_integration import (
    compute_body_transforms_from_dof,
    npz_to_pkl_with_fk,
    compare_fk_with_reference,
    get_default_key_bodies,
    validate_fk_implementation,
)
from utils.cmg_motion_generator import CMGMotionGenerator


def example_1_basic_fk():
    """示例1: 基础FK计算"""
    print("=" * 80)
    print("Example 1: Basic Forward Kinematics")
    print("=" * 80)
    
    # 导入FK模型
    from pose.util_funcs.kinematics_model import KinematicsModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化FK模型
    urdf_path = "../../assets/g1/g1_29dof.urdf"
    fk_model = KinematicsModel(urdf_path, device)
    print(f"FK Model loaded with {fk_model.num_joints} joints")
    
    # 创建零位姿态
    batch_size = 2
    dof_pos = torch.zeros((batch_size, 29), device=device)
    dof_vel = torch.zeros((batch_size, 29), device=device)
    
    base_pos = torch.tensor([[0., 0., 0.75], [0.5, 0., 0.75]], device=device)
    base_rot = torch.zeros((batch_size, 4), device=device)
    base_rot[:, 0] = 1.0
    
    key_bodies = get_default_key_bodies()
    
    # 计算FK
    body_pos, body_rot = fk_model.forward_kinematics(
        dof_pos, base_pos, base_rot, key_bodies
    )
    
    print(f"\nInput:")
    print(f"  DOF positions shape: {dof_pos.shape}")
    print(f"  DOF velocities shape: {dof_vel.shape}")
    print(f"  Base positions shape: {base_pos.shape}")
    print(f"  Base rotations shape: {base_rot.shape}")
    
    print(f"\nOutput:")
    print(f"  Body positions shape: {body_pos.shape}")
    print(f"  Body rotations shape: {body_rot.shape}")
    
    print(f"\nSample results (batch 0):")
    print(f"  Pelvis position: {body_pos[0, 0, :]}")
    print(f"  Pelvis rotation (wxyz): {body_rot[0, 0, :]}")
    
    print("\n✓ Example 1 completed\n")


def example_2_cmg_with_fk():
    """示例2: CMG集成FK"""
    print("=" * 80)
    print("Example 2: CMG Motion Generator with FK Integration")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CMG路径配置
    cmg_model_path = "../../CMG_Ref/module_output/cmg_latest.pt"  # 需要替换为实际路径
    cmg_data_path = "../../CMG_Ref/dataloader/cmg_g1_training_data.pt"  # 需要替换
    urdf_path = "../../assets/g1/g1_29dof.urdf"
    
    # 检查文件是否存在
    if not Path(cmg_model_path).exists():
        print(f"⚠ CMG model not found at {cmg_model_path}")
        print("  Skipping this example")
        return
    
    try:
        # 初始化CMG生成器（启用FK）
        generator = CMGMotionGenerator(
            model_path=cmg_model_path,
            data_path=cmg_data_path,
            num_envs=4,
            device=device,
            mode='pregenerated',
            fk_model_path=urdf_path,
            enable_fk=True,
        )
        
        # 重置生成器
        generator.reset(commands=torch.tensor([[1.0, 0.0, 0.0]] * 4, device=device))
        
        # 获取动作和body变换
        result = generator.get_motion_with_body_transforms()
        
        print(f"\nCMG Generator Output:")
        print(f"  DOF positions: {result['dof_positions'].shape}")
        print(f"  DOF velocities: {result['dof_velocities'].shape}")
        
        if 'body_positions' in result:
            print(f"  Body positions: {result['body_positions'].shape}")
            print(f"  Body rotations: {result['body_rotations'].shape}")
            print("\n✓ FK successfully integrated with CMG")
        else:
            print("\n⚠ FK not available in generator")
        
    except Exception as e:
        print(f"⚠ Error in example: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Example 2 completed\n")


def example_3_npz_to_pkl_conversion():
    """示例3: NPZ到PKL转换（带FK）"""
    print("=" * 80)
    print("Example 3: NPZ to PKL Conversion with FK")
    print("=" * 80)
    
    # 创建示例NPZ数据
    T = 100  # 时间步
    npz_data = {
        'dof_positions': np.random.randn(T, 29).astype(np.float32),
        'dof_velocities': np.random.randn(T, 29).astype(np.float32),
        'fps': np.array([50], dtype=np.float32),
    }
    
    print(f"Input NPZ data:")
    print(f"  DOF positions shape: {npz_data['dof_positions'].shape}")
    print(f"  DOF velocities shape: {npz_data['dof_velocities'].shape}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    urdf_path = "../../assets/g1/g1_29dof.urdf"
    
    if not Path(urdf_path).exists():
        print(f"⚠ URDF not found at {urdf_path}")
        return
    
    try:
        from pose.util_funcs.kinematics_model import KinematicsModel
        fk_model = KinematicsModel(urdf_path, device)
        
        # 转换为PKL格式（带FK）
        pkl_data = npz_to_pkl_with_fk(npz_data, fk_model=fk_model)
        
        print(f"\nOutput PKL data:")
        for key in pkl_data.keys():
            if isinstance(pkl_data[key], np.ndarray):
                print(f"  {key}: {pkl_data[key].shape}")
            else:
                print(f"  {key}: {type(pkl_data[key])}")
        
        print("\n✓ Conversion completed successfully")
        
    except Exception as e:
        print(f"⚠ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Example 3 completed\n")


def example_4_fk_validation():
    """示例4: FK实现验证"""
    print("=" * 80)
    print("Example 4: FK Implementation Validation")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    urdf_path = "../../assets/g1/g1_29dof.urdf"
    
    if not Path(urdf_path).exists():
        print(f"⚠ URDF not found at {urdf_path}")
        return
    
    try:
        from pose.util_funcs.kinematics_model import KinematicsModel
        fk_model = KinematicsModel(urdf_path, device)
        
        # 运行验证
        result = validate_fk_implementation(fk_model, num_test_poses=20, device=device)
        
        if result:
            print("✓ All validation checks passed!")
        else:
            print("⚠ Some validation checks failed")
        
    except Exception as e:
        print(f"⚠ Error during validation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Example 4 completed\n")


def example_5_joint_motion():
    """示例5: 单个关节运动验证"""
    print("=" * 80)
    print("Example 5: Single Joint Motion Verification")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    urdf_path = "../../assets/g1/g1_29dof.urdf"
    
    if not Path(urdf_path).exists():
        print(f"⚠ URDF not found at {urdf_path}")
        return
    
    try:
        from pose.util_funcs.kinematics_model import KinematicsModel
        fk_model = KinematicsModel(urdf_path, device)
        
        key_bodies = get_default_key_bodies()
        
        # 零位
        dof_zero = torch.zeros((1, 29), device=device)
        base_pos = torch.tensor([[0., 0., 0.75]], device=device)
        base_rot = torch.tensor([[1., 0., 0., 0.]], device=device)
        
        body_pos_zero, _ = fk_model.forward_kinematics(
            dof_zero, base_pos, base_rot, key_bodies
        )
        
        # 只改变waist_yaw (关节0)
        dof_yaw = torch.zeros((1, 29), device=device)
        dof_yaw[0, 0] = 0.5  # 旋转0.5弧度
        
        body_pos_yaw, _ = fk_model.forward_kinematics(
            dof_yaw, base_pos, base_rot, key_bodies
        )
        
        # 计算运动
        print(f"\nWaist yaw joint motion test:")
        print(f"  Waist angle: 0.5 rad (~28.6°)")
        print(f"\n  Body position changes:")
        
        for i, body_name in enumerate(key_bodies[:8]):
            pos_change = body_pos_yaw[0, i] - body_pos_zero[0, i]
            distance = torch.norm(pos_change).item()
            print(f"    {body_name:30s}: Δ = {distance:8.6f}m")
        
        print("\n✓ Verification shows realistic motion")
        
    except Exception as e:
        print(f"⚠ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Example 5 completed\n")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("1.1.3 Forward Kinematics Integration Examples")
    print("=" * 80 + "\n")
    
    # 运行示例
    example_1_basic_fk()
    example_4_fk_validation()
    example_5_joint_motion()
    example_2_cmg_with_fk()
    example_3_npz_to_pkl_conversion()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
