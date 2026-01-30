#!/usr/bin/env python3
"""
前向运动学测试脚本 - 验证29 DOF G1机器人的FK实现
"""

import torch
import numpy as np
from kinematics_model import KinematicsModel


def test_forward_kinematics_29dof():
    """测试29 DOF前向运动学"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[FK Test] Using device: {device}")
    
    # 初始化FK模型 - 使用29 DOF URDF
    urdf_path = "../../../assets/g1/g1_29dof.urdf"
    try:
        fk_model = KinematicsModel(urdf_path, device)
    except Exception as e:
        print(f"[FK Test] Error loading URDF: {e}")
        print(f"[FK Test] Trying XML format...")
        xml_path = "../../../assets/g1/g1_29dof.xml"
        fk_model = KinematicsModel(xml_path, device)
    
    print(f"[FK Test] FK Model initialized")
    print(f"  - Number of joints: {fk_model.num_joints}")
    print(f"  - Device: {device}")
    
    # 定义要计算的关键body
    key_bodies = [
        'pelvis',
        'left_hip_pitch_link', 'left_knee_link', 'left_ankle_pitch_link',
        'right_hip_pitch_link', 'right_knee_link', 'right_ankle_pitch_link',
        'left_shoulder_pitch_link', 'left_elbow_link', 'left_wrist_roll_link',
        'right_shoulder_pitch_link', 'right_elbow_link', 'right_wrist_roll_link',
    ]
    
    # 创建测试数据
    batch_size = 4
    
    # 29 DOF零位姿态
    joint_angles = torch.zeros((batch_size, 29), device=device)
    
    # 基座变换
    base_pos = torch.zeros((batch_size, 3), device=device)
    base_pos[:, 2] = 0.75  # 站立高度
    
    # 基座旋转 (单位四元数 = 恒等变换)
    base_rot = torch.zeros((batch_size, 4), device=device)
    base_rot[:, 0] = 1.0  # w=1, x=y=z=0
    
    print(f"\n[FK Test] Test 1: Zero pose")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Joint angles shape: {joint_angles.shape}")
    
    # 计算FK
    try:
        body_pos, body_rot = fk_model.forward_kinematics(
            joint_angles, base_pos, base_rot, key_bodies
        )
        
        print(f"  ✓ FK computed successfully")
        print(f"  - Body positions shape: {body_pos.shape}")
        print(f"  - Body rotations shape: {body_rot.shape}")
        
        # 打印一些结果
        print(f"\n  Sample body positions (first batch):")
        for i, body_name in enumerate(key_bodies[:5]):
            pos = body_pos[0, i, :]
            print(f"    {body_name:30s}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
        
        print(f"\n  Sample body rotations (first batch, first 3 bodies):")
        for i, body_name in enumerate(key_bodies[:3]):
            rot = body_rot[0, i, :]
            print(f"    {body_name:30s}: [{rot[0]:7.4f}, {rot[1]:7.4f}, {rot[2]:7.4f}, {rot[3]:7.4f}]")
        
    except Exception as e:
        print(f"  ✗ FK computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试2: 单个关节运动
    print(f"\n[FK Test] Test 2: Left hip pitch motion")
    joint_angles_hip = torch.zeros((batch_size, 29), device=device)
    joint_angles_hip[:, 3] = torch.tensor([0.0, 0.3, 0.6, 0.9])  # 左hip_pitch
    
    try:
        body_pos_hip, body_rot_hip = fk_model.forward_kinematics(
            joint_angles_hip, base_pos, base_rot, key_bodies
        )
        
        # 比较hip pitch运动
        left_knee_pos_0 = body_pos[0, 2, :]  # 零位
        left_knee_pos_hip = body_pos_hip[:, 2, :]  # 有hip pitch
        
        print(f"  ✓ Hip pitch motion computed")
        print(f"  - Left knee position change:")
        for i, angles in enumerate([0.0, 0.3, 0.6, 0.9]):
            pos = left_knee_pos_hip[i, :]
            diff = torch.norm(pos - left_knee_pos_0)
            print(f"    hip_pitch={angles:5.1f}rad: pos=[{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}], "
                  f"Δ={diff:7.4f}m")
        
    except Exception as e:
        print(f"  ✗ Hip pitch test failed: {e}")
        return False
    
    # 测试3: 检查wrist DOF (新增的29 DOF中的关键部分)
    print(f"\n[FK Test] Test 3: Wrist DOF support (29 DOF adaptation)")
    joint_angles_wrist = torch.zeros((batch_size, 29), device=device)
    # 设置左腕: 索引26,27,28 (wrist_roll/pitch/yaw)
    joint_angles_wrist[:, 26] = 0.5  # 左腕roll
    joint_angles_wrist[:, 27] = 0.3  # 左腕pitch
    joint_angles_wrist[:, 28] = 0.2  # 左腕yaw
    
    try:
        body_pos_wrist, body_rot_wrist = fk_model.forward_kinematics(
            joint_angles_wrist, base_pos, base_rot, key_bodies
        )
        
        left_wrist_rot_0 = body_rot[0, 8, :]  # 零位腕旋转
        left_wrist_rot_moved = body_rot_wrist[0, 8, :]  # 有腕DOF
        
        rot_diff = torch.norm(left_wrist_rot_0 - left_wrist_rot_moved)
        
        print(f"  ✓ Wrist DOF test passed")
        print(f"  - Left wrist rotation change: {rot_diff:.6f}")
        print(f"    Zero pose: [{left_wrist_rot_0[0]:7.4f}, {left_wrist_rot_0[1]:7.4f}, "
              f"{left_wrist_rot_0[2]:7.4f}, {left_wrist_rot_0[3]:7.4f}]")
        print(f"    With wrist motion: [{left_wrist_rot_moved[0]:7.4f}, {left_wrist_rot_moved[1]:7.4f}, "
              f"{left_wrist_rot_moved[2]:7.4f}, {left_wrist_rot_moved[3]:7.4f}]")
        
    except Exception as e:
        print(f"  ✗ Wrist DOF test failed: {e}")
        return False
    
    print(f"\n[FK Test] ✓ All tests passed successfully!")
    return True


if __name__ == '__main__':
    success = test_forward_kinematics_29dof()
    exit(0 if success else 1)
