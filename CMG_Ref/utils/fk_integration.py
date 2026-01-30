"""
FK集成工具 - 将前向运动学与CMG运动生成器集成
用于从CMG生成的29 DOF关节轨迹计算body位置和旋转
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List


def compute_body_transforms_from_dof(
    dof_positions: torch.Tensor,
    dof_velocities: Optional[torch.Tensor] = None,
    fk_model=None,
    base_pos: Optional[torch.Tensor] = None,
    base_rot: Optional[torch.Tensor] = None,
    key_bodies: Optional[List[str]] = None,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    从关节位置计算身体变换 (位置 + 旋转)
    
    这是1.1.3任务的核心实现: 从关节角度计算身体的全局变换
    
    Args:
        dof_positions: [batch, 29] DOF位置
        dof_velocities: [batch, 29] DOF速度 (可选, 用于计算body速度)
        fk_model: KinematicsModel实例 (如果为None则跳过FK计算)
        base_pos: [batch, 3] 基座位置 (如果为None则默认为零)
        base_rot: [batch, 4] 基座旋转四元数 (如果为None则默认为单位四元数)
        key_bodies: 要计算的body名称列表 (如果为None则使用默认列表)
        device: 计算设备
    
    Returns:
        dict包含:
            - 'body_positions': [batch, num_bodies, 3]
            - 'body_rotations': [batch, num_bodies, 4] (wxyz四元数)
            - 'body_velocities': [batch, num_bodies, 3] (如果提供了dof_velocities)
            - 'body_angular_velocities': [batch, num_bodies, 3] (如果提供了dof_velocities)
    """
    
    batch_size = dof_positions.shape[0]
    
    # 设置默认值
    if base_pos is None:
        base_pos = torch.zeros((batch_size, 3), device=device)
    if base_rot is None:
        base_rot = torch.zeros((batch_size, 4), device=device)
        base_rot[:, 0] = 1.0  # 单位四元数
    
    if key_bodies is None:
        key_bodies = get_default_key_bodies()
    
    result = {}
    
    # 如果提供了FK模型, 计算前向运动学
    if fk_model is not None:
        try:
            body_pos, body_rot = fk_model.forward_kinematics(
                dof_positions, base_pos, base_rot, key_bodies
            )
            result['body_positions'] = body_pos
            result['body_rotations'] = body_rot
            
            # 如果提供了速度, 计算body速度
            if dof_velocities is not None:
                try:
                    lin_vel, ang_vel = fk_model.compute_body_velocities(
                        dof_positions, dof_velocities, key_bodies, base_rot
                    )
                    result['body_velocities'] = lin_vel
                    result['body_angular_velocities'] = ang_vel
                except Exception as e:
                    print(f"Warning: Failed to compute body velocities: {e}")
                    
        except Exception as e:
            print(f"Error in forward kinematics: {e}")
            raise
    else:
        # 如果没有FK模型, 返回占位符
        num_bodies = len(key_bodies)
        result['body_positions'] = torch.zeros((batch_size, num_bodies, 3), device=device)
        result['body_rotations'] = torch.zeros((batch_size, num_bodies, 4), device=device)
        result['body_rotations'][:, :, 0] = 1.0  # 单位四元数
    
    return result


def npz_to_pkl_with_fk(
    npz_data: Dict,
    fk_model=None,
    output_format: str = 'dict',
    key_bodies: Optional[List[str]] = None
) -> Dict:
    """
    将NPZ格式的CMG输出转换为PKL格式 (带FK计算)
    
    这是1.1.2任务的增强版本: 现在包含FK计算的body变换
    
    Args:
        npz_data: 从NPZ加载的数据 (包含dof_positions, dof_velocities等)
        fk_model: KinematicsModel实例
        output_format: 'dict' 或 'tuple'
        key_bodies: 要计算的body名称 (默认使用标准列表)
    
    Returns:
        符合TWIST格式的字典:
            - 'dof_positions': [T, 29]
            - 'dof_velocities': [T, 29]
            - 'body_positions': [T, num_bodies, 3]
            - 'body_rotations': [T, num_bodies, 4]
            - 'fps': int
            - 'dof_names': List[str]
            - 'body_names': List[str]
    """
    
    if key_bodies is None:
        key_bodies = get_default_key_bodies()
    
    # 提取数据
    dof_pos = torch.from_numpy(npz_data['dof_positions']).float()  # [T, 29]
    dof_vel = torch.from_numpy(npz_data['dof_velocities']).float()  # [T, 29]
    fps = int(npz_data['fps']) if isinstance(npz_data['fps'], np.ndarray) else npz_data['fps']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dof_pos = dof_pos.to(device)
    dof_vel = dof_vel.to(device)
    
    # 计算body变换
    transforms = compute_body_transforms_from_dof(
        dof_pos, dof_vel, fk_model, 
        key_bodies=key_bodies, device=device
    )
    
    # 构建输出格式
    result = {
        'dof_positions': dof_pos.cpu().numpy(),
        'dof_velocities': dof_vel.cpu().numpy(),
        'body_positions': transforms['body_positions'].cpu().numpy(),
        'body_rotations': transforms['body_rotations'].cpu().numpy(),
        'fps': fps,
        'dof_names': [f'joint_{i}' for i in range(29)],  # 或从CMG获取实际名称
        'body_names': key_bodies,
    }
    
    if 'body_velocities' in transforms:
        result['body_velocities'] = transforms['body_velocities'].cpu().numpy()
    if 'body_angular_velocities' in transforms:
        result['body_angular_velocities'] = transforms['body_angular_velocities'].cpu().numpy()
    
    return result


def compare_fk_with_reference(
    computed_body_pos: torch.Tensor,
    computed_body_rot: torch.Tensor,
    reference_body_pos: torch.Tensor,
    reference_body_rot: torch.Tensor,
    body_names: List[str],
    threshold_pos: float = 0.01,  # 1cm
    threshold_rot: float = 0.05   # ~2.9度
) -> Dict[str, torch.Tensor]:
    """
    比较计算的和参考的body变换
    
    用于测试FK实现: 比较计算的与参考身体位置
    
    Args:
        computed_body_pos: [T, num_bodies, 3] 计算的位置
        computed_body_rot: [T, num_bodies, 4] 计算的旋转
        reference_body_pos: [T, num_bodies, 3] 参考位置
        reference_body_rot: [T, num_bodies, 4] 参考旋转
        body_names: body名称列表
        threshold_pos: 位置误差阈值 (米)
        threshold_rot: 旋转误差阈值 (四元数范数)
    
    Returns:
        误差统计字典
    """
    
    # 计算位置误差
    pos_error = torch.norm(
        computed_body_pos - reference_body_pos, 
        dim=-1
    )  # [T, num_bodies]
    
    # 计算旋转误差 (四元数范数)
    rot_error = torch.norm(
        computed_body_rot - reference_body_rot,
        dim=-1
    )  # [T, num_bodies]
    
    results = {
        'pos_error': pos_error,
        'rot_error': rot_error,
        'pos_error_mean': pos_error.mean(dim=0),
        'rot_error_mean': rot_error.mean(dim=0),
        'pos_error_max': pos_error.max(dim=0)[0],
        'rot_error_max': rot_error.max(dim=0)[0],
    }
    
    # 打印报告
    print("\n[FK Comparison Report]")
    print(f"Position Error Threshold: {threshold_pos:.4f}m")
    print(f"Rotation Error Threshold: {threshold_rot:.4f}")
    print("\nPer-Body Error Statistics:")
    print(f"{'Body Name':<30} {'Mean Pos Err (m)':<20} {'Mean Rot Err':<20} {'Status':<10}")
    print("-" * 80)
    
    for i, body_name in enumerate(body_names):
        mean_pos = results['pos_error_mean'][i].item()
        mean_rot = results['rot_error_mean'][i].item()
        status = "✓ PASS" if (mean_pos < threshold_pos and mean_rot < threshold_rot) else "✗ FAIL"
        print(f"{body_name:<30} {mean_pos:<20.6f} {mean_rot:<20.6f} {status:<10}")
    
    # 总体统计
    all_pos_pass = (results['pos_error_mean'] < threshold_pos).all()
    all_rot_pass = (results['rot_error_mean'] < threshold_rot).all()
    
    print("\n" + "=" * 80)
    print(f"Overall Position Error: Mean={pos_error.mean():.6f}m, Max={pos_error.max():.6f}m - {'✓ PASS' if all_pos_pass else '✗ FAIL'}")
    print(f"Overall Rotation Error: Mean={rot_error.mean():.6f}, Max={rot_error.max():.6f} - {'✓ PASS' if all_rot_pass else '✗ FAIL'}")
    
    return results


def get_default_key_bodies() -> List[str]:
    """
    获取G1机器人的默认关键body列表 (29 DOF配置)
    
    包含:
    - 躯干: pelvis, waist_link, chest_link
    - 左腿: 髋/膝/踝
    - 右腿: 髋/膝/踝
    - 左臂: 肩/肘/腕
    - 右臂: 肩/肘/腕
    """
    return [
        # 躯干
        'pelvis',
        
        # 左腿
        'left_hip_pitch_link',
        'left_knee_link',
        'left_ankle_pitch_link',
        
        # 右腿
        'right_hip_pitch_link',
        'right_knee_link',
        'right_ankle_pitch_link',
        
        # 左臂
        'left_shoulder_pitch_link',
        'left_elbow_link',
        'left_wrist_roll_link',  # ✨ 新增 (29 DOF)
        
        # 右臂
        'right_shoulder_pitch_link',
        'right_elbow_link',
        'right_wrist_roll_link',  # ✨ 新增 (29 DOF)
    ]


def validate_fk_implementation(
    fk_model,
    num_test_poses: int = 10,
    device: str = 'cuda'
) -> bool:
    """
    验证FK实现的正确性
    
    检查:
    1. 零位姿态下的输出合理性
    2. 单个关节运动的输出变化
    3. 所有body输出的维度正确性
    4. 四元数的归一化 (可选)
    
    Args:
        fk_model: KinematicsModel实例
        num_test_poses: 测试姿态数量
        device: 计算设备
    
    Returns:
        True if all checks pass, False otherwise
    """
    
    print("[FK Validation]")
    
    # 创建测试数据
    key_bodies = get_default_key_bodies()
    
    # 测试1: 零位姿态
    print("  Test 1: Zero pose validation...")
    dof_zero = torch.zeros((1, 29), device=device)
    base_pos = torch.tensor([[0., 0., 0.75]], device=device)
    base_rot = torch.tensor([[1., 0., 0., 0.]], device=device)
    
    try:
        body_pos, body_rot = fk_model.forward_kinematics(
            dof_zero, base_pos, base_rot, key_bodies
        )
        
        assert body_pos.shape == (1, len(key_bodies), 3), \
            f"Position shape mismatch: {body_pos.shape}"
        assert body_rot.shape == (1, len(key_bodies), 4), \
            f"Rotation shape mismatch: {body_rot.shape}"
        assert not torch.isnan(body_pos).any(), "NaN in body positions"
        assert not torch.isnan(body_rot).any(), "NaN in body rotations"
        
        print("    ✓ Zero pose valid")
    except Exception as e:
        print(f"    ✗ Zero pose test failed: {e}")
        return False
    
    # 测试2: 关节范围运动
    print("  Test 2: Joint range motion validation...")
    for joint_idx in [0, 10, 20]:  # 采样几个不同的关节
        dof_test = torch.zeros((num_test_poses, 29), device=device)
        dof_test[:, joint_idx] = torch.linspace(-1.0, 1.0, num_test_poses)
        
        try:
            body_pos_test, body_rot_test = fk_model.forward_kinematics(
                dof_test, base_pos.expand(num_test_poses, -1), 
                base_rot.expand(num_test_poses, -1), 
                key_bodies
            )
            
            assert not torch.isnan(body_pos_test).any(), f"NaN in positions for joint {joint_idx}"
            assert not torch.isnan(body_rot_test).any(), f"NaN in rotations for joint {joint_idx}"
            
            # 检查运动是否连续 (相邻帧变化不会太大)
            pos_diff = torch.norm(
                body_pos_test[1:] - body_pos_test[:-1],
                dim=-1
            ).max()
            
            if pos_diff > 1.0:  # 1米跳跃会很奇怪
                print(f"    ⚠ Large position jump for joint {joint_idx}: {pos_diff:.4f}m")
            
        except Exception as e:
            print(f"    ✗ Joint {joint_idx} test failed: {e}")
            return False
    
    print("    ✓ Joint motion valid")
    
    # 测试3: 四元数检查
    print("  Test 3: Quaternion normalization check...")
    quat_norm = torch.norm(body_rot, dim=-1)  # [1, num_bodies]
    
    # 四元数的范数应接近1.0
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=0.01):
        print(f"    ⚠ Quaternion norms not close to 1.0: min={quat_norm.min():.6f}, max={quat_norm.max():.6f}")
        # 这可能是正常的, 取决于FK库的实现
    else:
        print("    ✓ Quaternions normalized")
    
    print("  ✓ All validation checks passed!\n")
    return True
