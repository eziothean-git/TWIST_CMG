import torch
import numpy as np
from typing import Tuple, List

from isaacgym.torch_utils import quat_rotate, quat_apply, matrix_to_quaternion
import pytorch_kinematics as pk


class KinematicsModel:
    """
    前向运动学模型 - 从关节角度计算身体的全局位置和旋转
    
    支持29 DOF G1机器人配置:
    - 左腿: 6 DOF (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
    - 右腿: 6 DOF
    - 腰部: 3 DOF (waist_yaw/roll/pitch)
    - 左臂: 4 DOF (shoulder_pitch/roll/yaw, elbow)
    - 左手腕: 3 DOF (wrist_roll/pitch/yaw) ✨ 新增
    - 右臂: 4 DOF
    - 右手腕: 3 DOF ✨ 新增
    """
    
    def __init__(self, file_path: str, device):
        self.device = device        
        if file_path.endswith(".urdf"):
            self.chain = pk.build_chain_from_urdf(open(file_path, mode="rb").read())
        elif file_path.endswith(".xml") or file_path.endswith(".mjcf"):
            self.chain = pk.build_chain_from_mjcf(open(file_path, mode="rb").read())
            
        self.chain = self.chain.to(device=device)
        self.num_joints = len(self.chain.get_joint_parameter_names())
        
        # 关节重新索引: 从CMG输出顺序映射到URDF顺序
        # CMG: [腰部yaw/roll/pitch, 左腿6DOF, 右腿6DOF, 左臂7DOF, 右臂7DOF]
        # URDF: 不同的顺序, 需要重新索引
        if self.num_joints == 29:
            # 适配29 DOF (包含手腕)
            self.reindex = [12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24, 25, 26, 27, 28]
        else:
            # 默认23 DOF映射 (不包含手腕)
            self.reindex = [12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    def forward_kinematics(self, joint_angles: torch.Tensor, base_pos: torch.Tensor, 
                          base_rot: torch.Tensor, key_bodies: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算所有关键body的全局位置和旋转
        
        Args:
            joint_angles: [batch, num_joints] 关节角度 (归一化或原始值)
            base_pos: [batch, 3] 基座位置
            base_rot: [batch, 4] 基座旋转 (四元数 wxyz)
            key_bodies: List[str] 要计算的body名称列表
        
        Returns:
            body_pos: [batch, num_bodies, 3] 全局body位置
            body_rot: [batch, num_bodies, 4] 全局body旋转 (四元数 wxyz)
        """
        assert joint_angles.shape[1] == self.num_joints, \
            f"number of joints mismatch: {joint_angles.shape[1]} != {self.num_joints}"
        
        batch_size = joint_angles.shape[0]
        num_bodies = len(key_bodies)
        
        # 重新索引关节角度到URDF顺序
        joint_angles_reindexed = joint_angles[:, self.reindex]
        
        # 初始化输出张量
        body_positions = torch.zeros((batch_size, num_bodies, 3), device=self.device)
        body_rotations = torch.zeros((batch_size, num_bodies, 4), device=self.device)  # 四元数 wxyz
        
        # 运行链式正向运动学
        ret = self.chain.forward_kinematics(joint_angles_reindexed)
        
        # 提取每个body的位置和旋转
        for i, key_body in enumerate(key_bodies):
            if key_body in ret:
                tg = ret[key_body]
                m = tg.get_matrix()  # [batch, 4, 4] 变换矩阵
                
                # 提取位置 (第4列)
                pos_local = m[:, :3, 3]  # [batch, 3]
                
                # 提取旋转矩阵 (3x3左上)
                rot_mat = m[:, :3, :3]  # [batch, 3, 3]
                
                # 转换为四元数 (wxyz格式)
                quat_local = matrix_to_quaternion(rot_mat)  # [batch, 4] (wxyz)
                
                # 转换到全局坐标系
                body_positions[:, i, :] = quat_apply(base_rot, pos_local) + base_pos
                
                # 旋转四元数复合: global_rot = base_rot * local_rot
                body_rotations[:, i, :] = self._quat_multiply(base_rot, quat_local)
            else:
                print(f"Warning: key_body '{key_body}' not found in chain. "
                      f"Available bodies: {list(ret.keys())}")
                # 使用默认值 (基座的位置和旋转)
                body_positions[:, i, :] = base_pos
                body_rotations[:, i, :] = base_rot
        
        return body_positions, body_rotations
    
    def compute_body_velocities(self, joint_angles: torch.Tensor, joint_velocities: torch.Tensor,
                               key_bodies: list, base_rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算body的全局线速度和角速度 (使用雅可比矩阵)
        
        Args:
            joint_angles: [batch, num_joints] 关节角度
            joint_velocities: [batch, num_joints] 关节速度
            key_bodies: List[str] 要计算的body名称列表
            base_rot: [batch, 4] 基座旋转 (用于坐标变换)
        
        Returns:
            body_lin_vel: [batch, num_bodies, 3] 全局线速度
            body_ang_vel: [batch, num_bodies, 3] 全局角速度
        """
        batch_size = joint_angles.shape[0]
        num_bodies = len(key_bodies)
        
        joint_angles_reindexed = joint_angles[:, self.reindex]
        joint_velocities_reindexed = joint_velocities[:, self.reindex]
        
        body_lin_vels = torch.zeros((batch_size, num_bodies, 3), device=self.device)
        body_ang_vels = torch.zeros((batch_size, num_bodies, 3), device=self.device)
        
        # 使用chain计算雅可比矩阵并计算速度
        ret = self.chain.forward_kinematics(joint_angles_reindexed)
        
        for i, key_body in enumerate(key_bodies):
            if key_body in ret:
                # 获取雅可比矩阵
                jacob = self.chain.jacobian(key_body, joint_angles_reindexed)  # [batch, 6, num_joints]
                
                # 计算笛卡尔速度 = J * dq
                cartesian_vel = torch.bmm(jacob, joint_veloc_reindexed.unsqueeze(-1)).squeeze(-1)  # [batch, 6]
                
                # 分离线速度和角速度
                lin_vel = cartesian_vel[:, :3]  # [batch, 3]
                ang_vel = cartesian_vel[:, 3:]  # [batch, 3]
                
                # 转换到全局坐标系
                body_lin_vels[:, i, :] = quat_apply(base_rot, lin_vel)
                body_ang_vels[:, i, :] = quat_apply(base_rot, ang_vel)
        
        return body_lin_vels, body_ang_vels
    
    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        四元数乘法 (wxyz格式)
        q1 * q2 = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
        
        Args:
            q1, q2: [batch, 4] 四元数 (wxyz)
        
        Returns:
            result: [batch, 4] 四元数乘积
        """
        # 提取标量部分和向量部分
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        # 执行四元数乘法
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def get_body_names(self) -> List[str]:
        """返回所有可用的body名称"""
        return list(self.chain.get_joint_parameter_names())
        