# CMG-TWIST 集成实现指南

本文档提供了 6 个关键实现目标的详细说明，包括文件位置、代码结构和集成方法。

---

## 1. 添加正式的 DOF 映射脚本

### 目标
统一 CMG (29 DOF) 和 G1 机器人 (23 DOF) 之间的关节顺序映射。

### 文件位置
- **新建**：`legged_gym/legged_gym/gym_utils/dof_mapping.py`
- **修改**：在 G1 环境初始化处调用映射脚本

### 实现步骤

#### 步骤 1: 创建映射脚本框架

```python
# legged_gym/legged_gym/gym_utils/dof_mapping.py

import numpy as np
import torch

class CMGToG1Mapper:
    """
    Maps CMG's 29-DOF joint configuration to G1's 23-DOF configuration.
    
    CMG 29 DOF ordering (from CMG training):
    - 6 DOF per leg (12 total) + 3 DOF waist + 4 DOF per arm (8 total) + 6 extra
    
    G1 23 DOF ordering (actual robot):
    - 6 DOF per leg (12 total) + 3 DOF waist + 4 DOF per arm (8 total) + 0 extra
    """
    
    def __init__(self):
        """Initialize mapping indices and reference."""
        # TODO: Define the exact mapping from CMG 29 DOF to G1 23 DOF
        # This should be a mapping table with indices
        self.cmg_to_g1_indices = self._build_mapping_table()
        self.cmg_dof = 29
        self.g1_dof = 23
    
    def _build_mapping_table(self):
        """
        Build index mapping from CMG 29 DOF to G1 23 DOF.
        
        Returns:
            np.ndarray: Array of shape (23,) with indices into CMG's 29 DOF
        
        Example:
            [0, 1, 2, 3, 4, 5,      # Left leg (6 DOF)
             6, 7, 8, 9, 10, 11,    # Right leg (6 DOF)
             12, 13, 14,             # Waist (3 DOF)
             15, 16, 17, 18,         # Left arm (4 DOF)
             19, 20, 21, 22]         # Right arm (4 DOF)
        
        Note: Adjust indices based on actual CMG configuration.
        """
        # Placeholder - should be filled with actual indices
        mapping = np.arange(23)  # Placeholder
        return mapping
    
    def map_positions(self, cmg_pos):
        """
        Map CMG 29-DOF positions to G1 23-DOF positions.
        
        Args:
            cmg_pos: Joint positions, shape [batch_size, 29] or [29]
        
        Returns:
            g1_pos: Mapped positions, shape [batch_size, 23] or [23]
        """
        if isinstance(cmg_pos, torch.Tensor):
            return cmg_pos[..., self.cmg_to_g1_indices]
        else:  # numpy array
            return cmg_pos[..., self.cmg_to_g1_indices]
    
    def map_velocities(self, cmg_vel):
        """
        Map CMG 29-DOF velocities to G1 23-DOF velocities.
        
        Args:
            cmg_vel: Joint velocities, shape [batch_size, 29] or [29]
        
        Returns:
            g1_vel: Mapped velocities, shape [batch_size, 23] or [23]
        """
        if isinstance(cmg_vel, torch.Tensor):
            return cmg_vel[..., self.cmg_to_g1_indices]
        else:  # numpy array
            return cmg_vel[..., self.cmg_to_g1_indices]
    
    def map_trajectory(self, cmg_traj):
        """
        Map a full CMG trajectory to G1 format.
        
        Args:
            cmg_traj: Dict with keys 'dof_pos' and 'dof_vel', both shape [T, 29]
        
        Returns:
            g1_traj: Dict with same structure but shape [T, 23]
        """
        return {
            'dof_pos': self.map_positions(cmg_traj['dof_pos']),
            'dof_vel': self.map_velocities(cmg_traj['dof_vel']),
        }


# Global mapper instance
_g1_mapper = CMGToG1Mapper()


def get_g1_mapper():
    """Get the global CMG-to-G1 mapper instance."""
    return _g1_mapper


def map_cmg_to_g1(dof_pos):
    """
    Convenience function: Map CMG positions to G1.
    
    Args:
        dof_pos: Joint positions, shape [..., 29]
    
    Returns:
        Mapped positions, shape [..., 23]
    """
    return _g1_mapper.map_positions(dof_pos)


def map_cmg_to_g1_vel(dof_vel):
    """
    Convenience function: Map CMG velocities to G1.
    
    Args:
        dof_vel: Joint velocities, shape [..., 29]
    
    Returns:
        Mapped velocities, shape [..., 23]
    """
    return _g1_mapper.map_velocities(dof_vel)
```

#### 步骤 2: 在环境初始化中调用映射

```python
# Example: In legged_gym/legged_gym/envs/g1/g1_cmg_env.py or similar

from legged_gym.gym_utils.dof_mapping import map_cmg_to_g1, map_cmg_to_g1_vel

class G1CMGEnv(LeggedRobot):
    def __init__(self, cfg: DictConfig, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize mapper
        self.mapper = map_cmg_to_g1
        self.mapper_vel = map_cmg_to_g1_vel
    
    def load_cmg_reference_trajectory(self, traj_path):
        """Load and map CMG reference trajectory."""
        # Load CMG trajectory (29 DOF)
        cmg_traj = self._load_cmg_npz(traj_path)
        
        # Map to G1 (23 DOF)
        g1_traj = {
            'dof_pos': self.mapper(cmg_traj['dof_pos']),
            'dof_vel': self.mapper_vel(cmg_traj['dof_vel']),
        }
        
        return g1_traj
```

### 注意事项
- 映射表应基于 CMG 和 G1 的确切关节顺序
- 可参考 `g1_mimic_distill.py` 中现有的 `g1_body_from_38_to_52` 等函数
- 支持 PyTorch tensor 和 NumPy array 两种输入
- 在加载参考动作和输出动作时都要应用映射

---

## 2. 添加 Locomotion 相关奖励

### 目标
为走路场景添加专门的奖励函数（速度误差、角速度误差、足滑等）。

### 文件位置
- **新建环境类**：`legged_gym/legged_gym/envs/g1/g1_cmg_loco_env.py`
- **新建配置文件**：`legged_gym/legged_gym/envs/g1/g1_cmg_loco_config.py`

### 实现步骤

#### 步骤 1: 创建新环境类

```python
# legged_gym/legged_gym/envs/g1/g1_cmg_loco_env.py

import torch
from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill
from legged_gym.envs.g1.g1_cmg_loco_config import G1CMGLocoConfig

class G1CMGLoco(G1MimicDistill):
    """
    G1 locomotion environment with CMG reference tracking.
    
    Extends G1MimicDistill with additional locomotion-specific rewards:
    - Linear velocity tracking error
    - Angular velocity tracking error  
    - Base orientation error
    - Feet slip penalty
    - Action rate penalty
    """
    
    cfg_class = G1CMGLocoConfig
    
    def compute_reward(self):
        """
        Compute total reward from multiple components.
        
        Returns:
            reward: [num_envs] tensor
        """
        # Get reference velocity commands
        ref_lin_vel = self.env_cfg.commands.lin_vel_x  # or from obs
        ref_ang_vel = self.env_cfg.commands.ang_vel_z
        
        # 1. Joint position tracking error (from parent class)
        reward_dof_pos_error = self._reward_dof_pos_error()
        
        # 2. Linear velocity error
        reward_lin_vel = self._reward_lin_vel_error(ref_lin_vel)
        
        # 3. Angular velocity error
        reward_ang_vel = self._reward_ang_vel_error(ref_ang_vel)
        
        # 4. Base orientation error
        reward_orient = self._reward_orientation_error()
        
        # 5. Feet slip penalty
        reward_slip = self._reward_feet_slip()
        
        # 6. Action rate penalty
        reward_action_rate = self._reward_action_rate()
        
        # Combine all rewards
        total_reward = (
            self.reward_scales.dof_pos_error * reward_dof_pos_error +
            self.reward_scales.lin_vel_error * reward_lin_vel +
            self.reward_scales.ang_vel_error * reward_ang_vel +
            self.reward_scales.orientation_error * reward_orient +
            self.reward_scales.feet_slip * reward_slip +
            self.reward_scales.action_rate * reward_action_rate
        )
        
        return total_reward
    
    def _reward_lin_vel_error(self, ref_lin_vel):
        """
        Penalize error between base linear velocity and command.
        
        Args:
            ref_lin_vel: [num_envs] reference linear velocity
        
        Returns:
            penalty: [num_envs] scalar reward (negative)
        """
        # Get base linear velocity (x-direction in world frame)
        base_lin_vel = self.base_lin_vel[:, 0:1]  # [num_envs, 1]
        
        # Compute error
        lin_vel_error = torch.abs(base_lin_vel - ref_lin_vel.unsqueeze(-1))
        
        # Return as penalty (negative reward)
        return -lin_vel_error.squeeze(-1)
    
    def _reward_ang_vel_error(self, ref_ang_vel):
        """
        Penalize error between base angular velocity and command.
        
        Args:
            ref_ang_vel: [num_envs] reference angular velocity (yaw)
        
        Returns:
            penalty: [num_envs] scalar reward (negative)
        """
        # Get base angular velocity (z-direction in body frame)
        base_ang_vel = self.base_ang_vel[:, 2:3]  # [num_envs, 1]
        
        # Compute error
        ang_vel_error = torch.abs(base_ang_vel - ref_ang_vel.unsqueeze(-1))
        
        # Return as penalty
        return -ang_vel_error.squeeze(-1)
    
    def _reward_orientation_error(self):
        """
        Penalize deviation from upright orientation.
        
        Encourages the robot to maintain a vertical torso.
        
        Returns:
            penalty: [num_envs] scalar reward (negative)
        """
        # Get projected gravity (z-axis in body frame should point down)
        # i.e., gravity should be [0, 0, -1] in body frame
        
        forward = self.forward_vec  # [num_envs, 3]
        right = self.right_vec      # [num_envs, 3]
        up = torch.cross(right, forward, dim=-1)  # [num_envs, 3]
        
        # Gravity direction in world frame is [0, 0, -1]
        # In body frame, it should be [0, 0, -1]
        # Compute error
        gravity_in_body = torch.stack([
            torch.sum(self.gravity_vec * forward, dim=-1),
            torch.sum(self.gravity_vec * right, dim=-1),
            torch.sum(self.gravity_vec * up, dim=-1)
        ], dim=-1)  # [num_envs, 3]
        
        # Target gravity in body frame
        target_gravity = torch.tensor([0., 0., -1.], device=self.device)
        
        # Orientation error
        orient_error = torch.norm(gravity_in_body - target_gravity, dim=-1)
        
        return -orient_error
    
    def _reward_feet_slip(self):
        """
        Penalize feet slipping on the ground during stance.
        
        Returns:
            penalty: [num_envs] scalar reward (negative)
        """
        # Get foot contact forces
        contact_forces = self.contact_forces[:, self.feet_indices, :]  # [num_envs, 2, 3]
        
        # Get foot velocities (in world frame)
        foot_vel = self.rigid_body_vel[:, self.feet_indices, :3]  # [num_envs, 2, 3]
        
        # Get foot contact state (binary: in contact or not)
        foot_contact = (torch.norm(contact_forces, dim=-1) > 1.0).float()  # [num_envs, 2]
        
        # Horizontal foot velocity during stance
        foot_vel_horizontal = torch.norm(foot_vel[:, :, :2], dim=-1)  # [num_envs, 2]
        
        # Slip penalty: horizontal velocity when in contact
        slip_penalty = torch.sum(foot_vel_horizontal * foot_contact, dim=-1)
        
        return -slip_penalty
    
    def _reward_action_rate(self):
        """
        Penalize large changes in action between consecutive timesteps.
        
        Encourages smooth actions.
        
        Returns:
            penalty: [num_envs] scalar reward (negative)
        """
        if not hasattr(self, 'prev_actions'):
            self.prev_actions = torch.zeros_like(self.actions)
        
        # Compute action change
        action_diff = torch.norm(self.actions - self.prev_actions, dim=-1)
        
        # Update previous actions
        self.prev_actions = self.actions.clone()
        
        return -action_diff
```

#### 步骤 2: 创建新配置文件

```python
# legged_gym/legged_gym/envs/g1/g1_cmg_loco_config.py

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicDistillConfig
from legged_gym.utils.helpers import merge_dict

class G1CMGLocoConfig(G1MimicDistillConfig):
    """Configuration for G1 locomotion with CMG reference tracking."""
    
    class env(G1MimicDistillConfig.env):
        """Environment parameters."""
        num_envs = 4096
        num_observations = 256  # Adjust as needed
        num_actions = 23  # G1 has 23 DOF
        num_privileged_obs = 512  # Adjust as needed
        
        # Timestep
        dt = 0.02
        episode_length_s = 20
        
    class terrain(G1MimicDistillConfig.terrain):
        """Terrain parameters for locomotion."""
        mesh_type = 'plane'  # Start with flat terrain
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
    
    class commands(G1MimicDistillConfig.commands):
        """Command distribution for locomotion."""
        sample_interval = 10  # Sample new commands every 0.2s
        ranges = {
            "lin_vel_x": [-1.0, 2.0],  # Linear velocity range
            "lin_vel_y": [-0.5, 0.5],  # Lateral velocity range
            "ang_vel_z": [-1.0, 1.0],  # Angular velocity range
        }
    
    class rewards(G1MimicDistillConfig.rewards):
        """Reward scales for locomotion."""
        class scales:
            # Existing rewards from parent
            dof_pos_error = 0.5
            dof_vel_error = 0.1
            
            # New locomotion rewards
            lin_vel_error = 1.0      # Linear velocity tracking
            ang_vel_error = 0.5      # Angular velocity tracking
            orientation_error = 1.0  # Upright posture
            feet_slip = 0.1          # Feet slip penalty
            action_rate = 0.01       # Smooth actions
    
    class domain_rand(G1MimicDistillConfig.domain_rand):
        """Domain randomization for robustness."""
        randomize_friction = False  # Disabled for flat terrain
        friction_range = [0.5, 1.5]
        randomize_com_displacement = False
        com_displacement_range = [-0.05, 0.05]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        
        push_robots = False  # No external pushes on flat terrain
```

### 注意事项
- 奖励权重需要通过训练来调整
- 参考论文：Rudin et al. 2021 "Learning to Walk in Minutes"
- 可逐步添加更多复杂的奖励项
- 监控各个奖励组件的贡献度

---

## 3. 正式实现 Teacher 特权观测

### 目标
为 teacher 网络实现完整的特权观测，包括未来参考帧。

### 文件位置
- **新建环境类**：`legged_gym/legged_gym/envs/g1/g1_cmg_teacher_env.py`
- **新建配置文件**：`legged_gym/legged_gym/envs/g1/g1_cmg_teacher_config.py`

### 实现步骤

#### 步骤 1: 创建 Teacher 环境

```python
# legged_gym/legged_gym/envs/g1/g1_cmg_teacher_env.py

import torch
from legged_gym.envs.g1.g1_cmg_loco_env import G1CMGLoco
from legged_gym.envs.g1.g1_cmg_teacher_config import G1CMGTeacherConfig

class G1CMGTeacher(G1CMGLoco):
    """
    Teacher environment for G1 with CMG reference tracking.
    
    Provides privileged observations including:
    - Future reference frames (multiple steps ahead)
    - Root state history
    - Full joint states
    - Body transforms
    """
    
    cfg_class = G1CMGTeacherConfig
    
    def __init__(self, cfg: DictConfig, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize reference trajectory buffer
        self._init_reference_buffer()
    
    def _init_reference_buffer(self):
        """Initialize buffer for reference trajectories."""
        self.ref_traj_buffer = {
            'dof_pos': torch.zeros(
                self.num_envs, 
                self.env_cfg.tar_obs_steps,
                self.num_dof,
                device=self.device
            ),
            'dof_vel': torch.zeros(
                self.num_envs,
                self.env_cfg.tar_obs_steps,
                self.num_dof,
                device=self.device
            ),
            'root_pos': torch.zeros(
                self.num_envs,
                self.env_cfg.tar_obs_steps,
                3,
                device=self.device
            ),
            'root_rot': torch.zeros(
                self.num_envs,
                self.env_cfg.tar_obs_steps,
                4,
                device=self.device
            ),
            'root_vel': torch.zeros(
                self.num_envs,
                self.env_cfg.tar_obs_steps,
                6,
                device=self.device
            ),
        }
    
    def _get_mimic_obs(self):
        """
        Get future reference observations for teacher.
        
        Returns:
            priv_mimic_obs: [num_envs, n_priv_mimic_obs] tensor
        """
        # Sample future frames at predetermined timesteps
        future_frames = []
        
        for step_idx in self.env_cfg.tar_obs_steps:
            # Get reference state at this future step
            frame = self._get_future_ref_obs(step_idx)
            future_frames.append(frame)
        
        # Concatenate all future frames
        priv_mimic_obs = torch.cat(future_frames, dim=-1)
        
        return priv_mimic_obs
    
    def _get_future_ref_obs(self, step_idx):
        """
        Get reference observation at a future timestep.
        
        Args:
            step_idx: Index into the reference trajectory buffer
        
        Returns:
            frame: [num_envs, frame_dim] tensor with reference state
        """
        # Extract from buffer
        ref_dof_pos = self.ref_traj_buffer['dof_pos'][:, step_idx, :]  # [num_envs, 23]
        ref_dof_vel = self.ref_traj_buffer['dof_vel'][:, step_idx, :]  # [num_envs, 23]
        ref_root_pos = self.ref_traj_buffer['root_pos'][:, step_idx, :]  # [num_envs, 3]
        ref_root_rot = self.ref_traj_buffer['root_rot'][:, step_idx, :]  # [num_envs, 4]
        ref_root_vel = self.ref_traj_buffer['root_vel'][:, step_idx, :]  # [num_envs, 6]
        
        # Compute forward/right vectors from rotation
        rot_mat = self.quat_to_mat(ref_root_rot)
        forward = rot_mat[:, :, 0]  # [num_envs, 3]
        right = rot_mat[:, :, 1]    # [num_envs, 3]
        
        # Frame composition:
        # [ref_dof_pos, ref_dof_vel, ref_root_pos, ref_root_rot, ref_root_vel, forward, right]
        frame = torch.cat([
            ref_dof_pos,        # 23
            ref_dof_vel,        # 23
            ref_root_pos,       # 3
            ref_root_rot,       # 4
            ref_root_vel,       # 6
            forward,            # 3
            right,              # 3
        ], dim=-1)  # Total: 65
        
        return frame
    
    def get_privileged_obs(self):
        """
        Get complete privileged observations for teacher.
        
        Returns:
            priv_obs: [num_envs, n_priv_obs] tensor
        """
        # Get mimic (future reference) observations
        priv_mimic_obs = self._get_mimic_obs()
        
        # Get proprioceptive observations
        priv_proprio = self._get_proprio_obs()
        
        # Get domain randomization info (e.g., friction, masses)
        priv_info = self._get_priv_info()
        
        # Concatenate all privileged observations
        priv_obs = torch.cat([
            priv_mimic_obs,
            priv_proprio,
            priv_info,
        ], dim=-1)
        
        return priv_obs
    
    def _get_proprio_obs(self):
        """Get proprioceptive observations."""
        return torch.cat([
            self.dof_pos,
            self.dof_vel,
            self.actions,
        ], dim=-1)
    
    def _get_priv_info(self):
        """Get privileged info (e.g., dynamics parameters)."""
        # Include friction, mass variations, etc.
        return torch.zeros(self.num_envs, self.env_cfg.n_priv_info, device=self.device)
    
    def reset_idx(self, env_ids):
        """Reset environment and generate new reference trajectory."""
        super().reset_idx(env_ids)
        
        # Generate or load new CMG reference trajectory for reset environments
        self._generate_cmg_reference(env_ids)
    
    def _generate_cmg_reference(self, env_ids):
        """
        Generate or load CMG reference trajectory.
        
        Args:
            env_ids: [num_reset_envs] indices of environments to reset
        """
        # TODO: Implement CMG inference to generate reference trajectory
        # For now, use placeholder
        
        num_steps = self.env_cfg.tar_obs_steps[-1] + 1
        
        for i, env_id in enumerate(env_ids):
            # Generate trajectory using CMG
            traj = self._cmg_generate(
                initial_state=self.root_states[env_id],
                num_steps=num_steps,
                command=self.commands[env_id]
            )
            
            # Store in buffer
            self.ref_traj_buffer['dof_pos'][env_id] = traj['dof_pos'][:num_steps]
            self.ref_traj_buffer['dof_vel'][env_id] = traj['dof_vel'][:num_steps]
            self.ref_traj_buffer['root_pos'][env_id] = traj['root_pos'][:num_steps]
            self.ref_traj_buffer['root_rot'][env_id] = traj['root_rot'][:num_steps]
            self.ref_traj_buffer['root_vel'][env_id] = traj['root_vel'][:num_steps]
    
    def _cmg_generate(self, initial_state, num_steps, command):
        """
        Generate trajectory using CMG model.
        
        Args:
            initial_state: Initial root state
            num_steps: Number of steps to generate
            command: Velocity command [vx, vy, yaw]
        
        Returns:
            trajectory: Dict with 'dof_pos', 'dof_vel', etc.
        """
        # TODO: Implement CMG inference
        # Placeholder implementation
        raise NotImplementedError("CMG generation not yet implemented")
```

#### 步骤 2: 创建 Teacher 配置

```python
# legged_gym/legged_gym/envs/g1/g1_cmg_teacher_config.py

from legged_gym.envs.g1.g1_cmg_loco_config import G1CMGLocoConfig

class G1CMGTeacherConfig(G1CMGLocoConfig):
    """Configuration for teacher policy with privileged observations."""
    
    class env(G1CMGLocoConfig.env):
        """Environment parameters."""
        # Privileged observation configuration
        tar_obs_steps = [0, 1, 2, 3, 4, 5]  # Sample at these future steps
        n_priv_mimic_obs = 390  # 65 * 6 (65 dims per frame, 6 frames)
        n_priv_info = 50  # Friction, mass, etc.
        
        num_privileged_obs = 390 + 46 + 50  # mimic + proprio + info
    
    class policy(G1CMGLocoConfig.policy):
        """Policy parameters for teacher."""
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256]
        critic_hidden_dims = [512, 256]
        
        # Use teacher network that accepts privileged observations
        policy_class_name = "ActorCriticMimic"  # Or custom teacher network
```

### 注意事项
- 未来帧采样索引需要与 CMG 推理步长匹配
- 特权信息应包含动力学参数以帮助学生泛化
- 缓冲区大小取决于 episode 长度和采样间隔

---

## 4. 定义残差模型结构

### 目标
实现残差网络，学习修正参考动作的偏差。

### 文件位置
- **新建模块**：`rsl_rl/rsl_rl/modules/actor_critic_residual.py`
- **修改**：训练配置文件指向新模型

### 实现步骤

#### 步骤 1: 创建残差网络模块

```python
# rsl_rl/rsl_rl/modules/actor_critic_residual.py

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

class ActorCriticResidual(ActorCritic):
    """
    Actor-Critic network that learns residual actions on top of reference.
    
    Input:
        observation: Current state observation [num_envs, num_obs]
        reference_action: Reference action from CMG/reference policy [num_envs, num_actions]
    
    Output:
        action: Reference action + residual = final action [num_envs, num_actions]
        value: Critic value estimate [num_envs, 1]
    """
    
    def __init__(self, num_obs, num_critic_obs, num_actions, **kwargs):
        # Don't call parent __init__ yet, we need to modify it
        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        
        nn.Module.__init__(self)
        
        actor_hidden_dims = kwargs.pop('actor_hidden_dims', [256, 256])
        critic_hidden_dims = kwargs.pop('critic_hidden_dims', [256, 256])
        activation = kwargs.pop('activation', nn.Tanh)
        
        # Actor: takes [observation, reference_action] as input
        # NOTE: Assuming reference_action is concatenated with observation
        actor_input_dim = num_obs  # Observation should include reference action
        
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation())
        
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation())
        
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic: takes full observation (may include privileged obs)
        critic_input_dim = num_critic_obs
        
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation())
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation())
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights
        for layer in self.actor.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.)
        
        for layer in self.critic.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.)
    
    def forward(self, obs, privileged_obs=None):
        """
        Forward pass.
        
        Args:
            obs: [num_envs, num_obs] observation
                 Should include reference action concatenated
            privileged_obs: [num_envs, num_critic_obs] privileged observation for critic
        
        Returns:
            actions: [num_envs, num_actions] final actions (reference + residual)
            values: [num_envs, 1] critic values
        """
        # Actor outputs residual actions
        residual = self.actor(obs)
        
        # Critic uses privileged observation if available
        critic_input = privileged_obs if privileged_obs is not None else obs
        values = self.critic(critic_input)
        
        return residual, values
    
    def forward_actor(self, obs):
        """Get residual actions only."""
        return self.actor(obs)
    
    def forward_critic(self, obs):
        """Get values only."""
        return self.critic(obs)


class ActorCriticResidualWithReference(nn.Module):
    """
    Variant that explicitly includes reference action in the network.
    """
    
    def __init__(self, num_obs, num_actions, num_reference_actions=None, **kwargs):
        super().__init__()
        
        num_reference_actions = num_reference_actions or num_actions
        
        actor_hidden_dims = kwargs.pop('actor_hidden_dims', [256, 256])
        critic_hidden_dims = kwargs.pop('critic_hidden_dims', [256, 256])
        activation = kwargs.pop('activation', nn.Tanh)
        
        # Input: [obs, reference_action]
        actor_input_dim = num_obs + num_reference_actions
        
        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation())
        
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation())
        
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (takes only observation)
        critic_layers = []
        critic_layers.append(nn.Linear(num_obs, critic_hidden_dims[0]))
        critic_layers.append(activation())
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation())
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize
        for layer in self.actor.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.)
        
        for layer in self.critic.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.)
    
    def forward(self, obs, reference_action):
        """
        Args:
            obs: [num_envs, num_obs]
            reference_action: [num_envs, num_actions]
        
        Returns:
            residual: [num_envs, num_actions] residual to add to reference
            values: [num_envs, 1]
        """
        # Concatenate obs and reference action
        actor_input = torch.cat([obs, reference_action], dim=-1)
        
        # Get residual
        residual = self.actor(actor_input)
        
        # Get value
        values = self.critic(obs)
        
        return residual, values
    
    def forward_actor(self, obs, reference_action):
        """Get residual only."""
        actor_input = torch.cat([obs, reference_action], dim=-1)
        return self.actor(actor_input)
    
    def forward_critic(self, obs):
        """Get values only."""
        return self.critic(obs)
```

#### 步骤 2: 更新环境以使用残差动作

```python
# Example: In training script or environment

class G1CMGStudentEnv(G1CMGLoco):
    """Student environment that learns residual on CMG reference."""
    
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        
        # Load CMG model
        self.cmg_model = self._load_cmg_model()
        self.residual_scale = cfg.residual_scale  # e.g., 0.1
    
    def compute_actions(self, obs):
        """
        Compute final actions = reference + residual.
        
        Args:
            obs: [num_envs, num_obs] observation
        
        Returns:
            actions: [num_envs, num_actions]
        """
        # Get reference action from CMG
        with torch.no_grad():
            ref_action = self.cmg_model(self.obs_buf)
        
        # Get residual from learned policy
        residual, _ = self.policy(obs, ref_action)
        
        # Combine
        actions = ref_action + self.residual_scale * residual
        
        # Clip to valid range
        actions = torch.clamp(actions, -1, 1)
        
        return actions
```

#### 步骤 3: 更新配置文件

```python
# In g1_cmg_student_config.py

class G1CMGStudentConfig(G1CMGTeacherConfig):
    """Configuration for student policy learning residuals."""
    
    class policy(G1CMGTeacherConfig.policy):
        """Policy parameters."""
        policy_class_name = "ActorCriticResidualWithReference"
        actor_hidden_dims = [256, 256]
        critic_hidden_dims = [256, 256]
    
    class algorithm:
        """Training algorithm parameters."""
        class_name = "PPO"  # or "DAGGER"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
```

### 注意事项
- 残差应该较小（通常 <10% 的参考动作）
- 可以通过 `residual_scale` 参数控制残差幅度
- 初期可以不使用残差，即残差为 0，逐步学习

---

## 5. 开始平地训练

### 目标
在简单的平地环境上进行初始训练。

### 文件位置
- **新建配置文件**：`legged_gym/legged_gym/envs/g1/g1_cmg_loco_flat_config.py`
- **修改**：`legged_gym/envs/__init__.py` 注册新任务

### 实现步骤

#### 步骤 1: 创建平地配置

```python
# legged_gym/legged_gym/envs/g1/g1_cmg_loco_flat_config.py

from legged_gym.envs.g1.g1_cmg_loco_config import G1CMGLocoConfig

class G1CMGLocoFlatConfig(G1CMGLocoConfig):
    """Configuration for flat terrain locomotion training."""
    
    class env(G1CMGLocoConfig.env):
        """Environment parameters."""
        num_envs = 4096
        num_observations = 256
        num_actions = 23
    
    class terrain(G1CMGLocoConfig.terrain):
        """Flat terrain configuration."""
        mesh_type = 'plane'  # Simple flat plane
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
        # No height variation
        height_samples = 1
        horizontal_scale = 0.1
    
    class domain_rand(G1CMGLocoConfig.domain_rand):
        """Minimal domain randomization for flat terrain."""
        randomize_friction = False
        randomize_com_displacement = False
        randomize_motor_strength = False
        
        push_robots = False  # No external disturbances
        push_interval = 0
    
    class rewards(G1CMGLocoConfig.rewards):
        """Reward configuration for flat terrain."""
        class scales:
            dof_pos_error = 0.5
            dof_vel_error = 0.1
            lin_vel_error = 1.0
            ang_vel_error = 0.5
            orientation_error = 1.0
            feet_slip = 0.1
            action_rate = 0.01
    
    class commands(G1CMGLocoConfig.commands):
        """Command configuration."""
        sample_interval = 50  # 1 second
        ranges = {
            "lin_vel_x": [0.5, 1.5],  # Conservative range for flat
            "lin_vel_y": [-0.3, 0.3],
            "ang_vel_z": [-0.5, 0.5],
        }
```

#### 步骤 2: 注册新任务

```python
# legged_gym/legged_gym/envs/__init__.py

from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.g1.g1_cmg_loco_config import G1CMGLocoConfig
from legged_gym.envs.g1.g1_cmg_loco_env import G1CMGLoco
from legged_gym.envs.g1.g1_cmg_loco_flat_config import G1CMGLocoFlatConfig

# Register flat terrain task
task_registry.register(
    name="g1_cmg_loco_flat",
    env_class=G1CMGLoco,
    env_cfg=G1CMGLocoFlatConfig()
)

# Register teacher task
task_registry.register(
    name="g1_cmg_teacher",
    env_class=G1CMGTeacher,
    env_cfg=G1CMGTeacherConfig()
)
```

#### 步骤 3: 运行训练

```bash
cd legged_gym
python scripts/train.py --task=g1_cmg_loco_flat --load_run=<path_to_checkpoint>
```

### 训练脚本片段

```python
# legged_gym/scripts/train.py (relevant parts)

if __name__ == "__main__":
    args = get_args()
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Initialize teacher policy if needed
    if args.task == "g1_cmg_teacher":
        policy = ActorCriticMimic(
            num_obs=env_cfg.env.num_observations,
            num_critic_obs=env_cfg.env.num_privileged_obs,
            num_actions=env_cfg.env.num_actions,
            **env_cfg.policy.todict()
        )
    elif args.task == "g1_cmg_loco_flat":
        policy = ActorCriticResidual(
            num_obs=env_cfg.env.num_observations,
            num_critic_obs=env_cfg.env.num_observations,
            num_actions=env_cfg.env.num_actions,
            **env_cfg.policy.todict()
        )
    
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        device=args.device,
        **env_cfg.algorithm.todict()
    )
    
    # Train
    trainer.train()
```

### 监控训练

```python
# Monitor key metrics
wandb.log({
    "episode_reward/total": total_reward.mean(),
    "episode_reward/lin_vel": lin_vel_reward.mean(),
    "episode_reward/ang_vel": ang_vel_reward.mean(),
    "episode_reward/slip": slip_reward.mean(),
    "command/lin_vel_x": commands[:, 0].mean(),
    "command/ang_vel_z": commands[:, 2].mean(),
})
```

### 注意事项
- 从保守的命令范围开始
- 监控 reward 各个组件，调整权重
- 保存 checkpoints 用于后续训练阶段
- 在平地上达到稳定行走后再进行复杂地形

---

## 6. 添加崎岖地形、摩擦力随机化、扰动

### 目标
在复杂地形和随机化条件下进行鲁棒性训练。

### 文件位置
- **新建配置文件**：`legged_gym/legged_gym/envs/g1/g1_cmg_loco_rough_config.py`
- **修改**：`legged_gym/legged_gym/gym_utils/terrain.py`（如需自定义地形）

### 实现步骤

#### 步骤 1: 创建复杂地形配置

```python
# legged_gym/legged_gym/envs/g1/g1_cmg_loco_rough_config.py

from legged_gym.envs.g1.g1_cmg_loco_flat_config import G1CMGLocoFlatConfig

class G1CMGLocoRoughConfig(G1CMGLocoFlatConfig):
    """Configuration for rough terrain with randomization."""
    
    class terrain(G1CMGLocoFlatConfig.terrain):
        """Complex terrain configuration."""
        mesh_type = 'trimesh'  # Use mesh for complex terrain
        
        # Terrain generation
        horizontal_scale = 0.05  # 5cm resolution
        vertical_scale = 0.005   # 5mm height precision
        border_size = 25
        
        # Height field configuration
        num_rows = 100
        num_cols = 100
        max_height = 1.0  # Maximum height variation
        
        # Terrain types
        # Options: "plane", "slope", "stairs", "pyramid", "random_blocks", etc.
        curriculum = True
        
        static_friction = 0.75
        dynamic_friction = 0.75
        restitution = 0.15
        
        # Roughness parameters
        roughness = 0.0  # 0 = smooth, 1 = very rough
        measure_heights = True
        height_samples = 3
        
    class domain_rand(G1CMGLocoFlatConfig.domain_rand):
        """Enable domain randomization for robustness."""
        
        # Friction randomization
        randomize_friction = True
        friction_range = [0.4, 1.2]  # Ice to rubber
        
        # COM displacement (weight distribution)
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]  # ±5cm
        
        # Motor strength randomization (actuator strength)
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]  # ±10%
        
        # External disturbances
        push_robots = True
        push_interval = 15  # Push every 15 steps (0.3 seconds)
        max_push_force = 100  # Newtons
        
        # Lag randomization
        action_delay = 0  # Steps of delay
    
    class commands(G1CMGLocoFlatConfig.commands):
        """More aggressive command ranges."""
        sample_interval = 50
        ranges = {
            "lin_vel_x": [-1.0, 2.0],  # Full range
            "lin_vel_y": [-0.5, 0.5],
            "ang_vel_z": [-1.0, 1.0],
        }
    
    class rewards(G1CMGLocoFlatConfig.rewards):
        """Adjusted rewards for rough terrain."""
        class scales:
            dof_pos_error = 0.5
            dof_vel_error = 0.1
            lin_vel_error = 0.8  # Slightly reduced
            ang_vel_error = 0.4
            orientation_error = 1.0
            feet_slip = 0.2  # Increased for rough terrain
            action_rate = 0.01
    
    class curriculum:
        """Difficulty curriculum."""
        # Progressive increase of terrain difficulty
        enabled = True
        
        # Checkpoint for curriculum progress
        curriculum_thresholds = [
            10000,   # Iter 10k
            20000,   # Iter 20k
            30000,   # Iter 30k
            40000,   # Iter 40k
        ]
        
        # Corresponding difficulty levels (0-1)
        difficulty_levels = [
            0.0,   # Flat terrain (from 0)
            0.25,  # Low slopes
            0.5,   # Stairs and slopes
            0.75,  # Rough terrain
            1.0,   # Maximum difficulty
        ]


class G1CMGLocoTerrainCurriculumConfig(G1CMGLocoRoughConfig):
    """Advanced configuration with terrain curriculum."""
    
    class terrain(G1CMGLocoRoughConfig.terrain):
        curriculum = True
        curriculum_type = "progressive"  # "progressive" or "random"
    
    class curriculum(G1CMGLocoRoughConfig.curriculum):
        """Progressive terrain curriculum."""
        enabled = True
        type = "terrain_difficulty"
        
        def difficulty_fn(iteration):
            """Map iteration number to difficulty level (0-1)."""
            difficulty = min(iteration / 100000, 1.0)
            return difficulty
```

#### 步骤 2: 自定义地形生成（如需要）

```python
# legged_gym/legged_gym/gym_utils/terrain.py (relevant additions)

class Terrain:
    def __init__(self, cfg: DictConfig, num_robots):
        """Initialize terrain."""
        # ... existing code ...
        
        if cfg.mesh_type == 'trimesh':
            self.heightfield = self._create_trimesh_terrain()
    
    def _create_trimesh_terrain(self):
        """Create complex trimesh terrain."""
        if hasattr(self, 'curriculum') and self.curriculum:
            difficulty = self.get_curriculum_difficulty()
        else:
            difficulty = 0.5
        
        heights = self._generate_height_field(difficulty)
        vertices, triangles = self._height_to_mesh(heights)
        
        return self._create_mesh(vertices, triangles)
    
    def _generate_height_field(self, difficulty):
        """
        Generate height field based on difficulty.
        
        Args:
            difficulty: 0-1, where 0 = flat, 1 = very rough
        
        Returns:
            heights: [num_rows, num_cols] numpy array
        """
        if difficulty < 0.25:
            # Flat terrain
            heights = np.zeros((self.num_rows, self.num_cols))
        
        elif difficulty < 0.5:
            # Low slopes
            heights = self._generate_slopes(
                angle_range=[0, 5],  # degrees
                num_slopes=3
            )
        
        elif difficulty < 0.75:
            # Stairs
            heights = self._generate_stairs(
                step_height=0.1,
                step_depth=0.2,
                num_steps=5
            )
        
        else:
            # Random rough terrain
            heights = self._generate_random_terrain(
                roughness=0.3,
                frequency=5
            )
        
        return heights
    
    def _generate_slopes(self, angle_range, num_slopes):
        """Generate terrain with slopes."""
        heights = np.zeros((self.num_rows, self.num_cols))
        
        for i in range(num_slopes):
            angle = np.random.uniform(*angle_range)
            start_col = i * (self.num_cols // num_slopes)
            end_col = (i + 1) * (self.num_cols // num_slopes)
            
            slope = np.tan(np.radians(angle))
            cols = np.linspace(start_col, end_col, end_col - start_col)
            
            for row in range(self.num_rows):
                heights[row, start_col:end_col] = slope * (cols - start_col) * 0.01
        
        return heights
    
    def _generate_stairs(self, step_height, step_depth, num_steps):
        """Generate stair-like terrain."""
        heights = np.zeros((self.num_rows, self.num_cols))
        
        step_width = self.num_cols // num_steps
        for step in range(num_steps):
            col_start = step * step_width
            col_end = (step + 1) * step_width
            heights[:, col_start:col_end] = step * step_height
        
        return heights
    
    def _generate_random_terrain(self, roughness, frequency):
        """Generate random rough terrain using Perlin noise."""
        # Use Perlin noise or simplex noise for natural terrain
        from noise import pnoise2
        
        heights = np.zeros((self.num_rows, self.num_cols))
        
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                heights[i, j] = pnoise2(
                    i / frequency,
                    j / frequency,
                    octaves=2
                ) * roughness
        
        return heights
```

#### 步骤 3: 训练脚本

```bash
# Train on rough terrain with curriculum
cd legged_gym
python scripts/train.py --task=g1_cmg_loco_rough --load_run=<flat_checkpoint>
```

#### 步骤 4: 逐步课程训练

```python
# legged_gym/scripts/train.py

class CurriculumTrainer:
    """Trainer with curriculum learning."""
    
    def __init__(self, env, policy, cfg):
        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.iteration = 0
    
    def train(self, num_iterations):
        """Train with curriculum."""
        for iter in range(num_iterations):
            self.iteration = iter
            
            # Update terrain difficulty
            if self.cfg.terrain.curriculum:
                difficulty = self._get_curriculum_difficulty()
                self.env.set_terrain_difficulty(difficulty)
                
                # Log curriculum progress
                wandb.log({"curriculum/difficulty": difficulty})
            
            # Standard training step
            obs = self.env.reset()
            for step in range(self.env.num_steps_per_env):
                with torch.no_grad():
                    actions, values, log_probs = self.policy(obs)
                
                next_obs, rewards, dones, info = self.env.step(actions)
                
                # Standard PPO update
                self.update_policy(obs, actions, rewards, values, log_probs)
                
                obs = next_obs
            
            if iter % 100 == 0:
                self._save_checkpoint()
    
    def _get_curriculum_difficulty(self):
        """Get current curriculum difficulty based on iteration."""
        cfg = self.cfg.curriculum
        
        if not cfg.enabled:
            return 1.0
        
        # Linear difficulty increase
        max_iter = cfg.curriculum_thresholds[-1]
        difficulty = min(self.iteration / max_iter, 1.0)
        
        return difficulty
```

### 注意事项
- 从平地训练的 checkpoint 开始
- 逐步增加难度以提高泛化性
- 监控不同地形上的性能
- 可以使用多种地形混合训练
- 调整 domain randomization 参数以匹配实际机器人

---

## 完整训练流程

### 阶段 1: 平地训练（1-2 周）
```bash
python scripts/train.py --task=g1_cmg_loco_flat
```

### 阶段 2: 从平地继续训练复杂地形（2-3 周）
```bash
python scripts/train.py --task=g1_cmg_loco_rough --load_run=<flat_checkpoint>
```

### 阶段 3: 微调和评估（1 周）
```bash
python scripts/play.py --task=g1_cmg_loco_rough --load_run=<rough_checkpoint>
```

---

## 快速参考检查表

- [ ] 创建 `dof_mapping.py` 并验证映射
- [ ] 添加 locomotion 奖励到环境
- [ ] 实现 teacher 特权观测
- [ ] 定义残差网络模型
- [ ] 注册平地任务 `g1_cmg_loco_flat`
- [ ] 运行平地训练
- [ ] 注册复杂地形任务 `g1_cmg_loco_rough`
- [ ] 进行复杂地形训练
- [ ] 在物理机器人上验证

---

**文档版本**：1.0  
**最后更新**：2026-01-30  
**状态**：完整实现指南
