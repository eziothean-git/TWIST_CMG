# Project Documentation: CMG and TWIST Integration

This document provides detailed technical documentation for both the Conditional Motion Generator (CMG) and the TWIST (Teleoperated Whole-Body Imitation System) projects.

---

## Table of Contents

1. [CMG (Conditional Motion Generator)](#1-cmg-conditional-motion-generator)
   - [1.1 Overview](#11-overview)
   - [1.2 Training Process](#12-training-process)
   - [1.3 Model Architecture](#13-model-architecture)
   - [1.4 Input/Output Specifications](#14-inputoutput-specifications)
   - [1.5 Data Pipeline](#15-data-pipeline)

2. [TWIST (Teleoperated Whole-Body Imitation System)](#2-twist-teleoperated-whole-body-imitation-system)
   - [2.1 Overview](#21-overview)
   - [2.2 Training Process](#22-training-process)
   - [2.3 Model Architecture](#23-model-architecture)
   - [2.4 Input/Output Specifications](#24-inputoutput-specifications)
   - [2.5 Two-Stage Training Pipeline](#25-two-stage-training-pipeline)

3. [Deployment](#3-deployment)

---

## 1. CMG (Conditional Motion Generator)

### 1.1 Overview

**Purpose**: Generate reference motions for humanoid robots based on velocity commands, eliminating the need for motion capture data.

**Key Features**:
- Command-based control from velocity inputs (vx, vy, yaw_rate)
- Mixture-of-Experts (MoE) architecture for diverse motion generation
- Autoregressive generation for smooth, continuous motion sequences
- Operates at 50 FPS
- Can generate walking, running, and turning behaviors

**Location**: `CMG_Ref/` directory

### 1.2 Training Process

#### Data Requirements
- Training data format: `cmg_training_data.pt`
- Data structure:
  ```python
  {
    "samples": List of motion sequences,
    "stats": {
      "motion_dim": 58,      # 29 joint positions + 29 joint velocities
      "command_dim": 3,      # [vx, vy, yaw_rate]
      "motion_mean": ndarray,
      "motion_std": ndarray,
      "command_min": ndarray,
      "command_max": ndarray
    }
  }
  ```
- Each sample contains:
  - `motion`: [seq_len+1, 58] - joint positions and velocities over time
  - `command`: [seq_len, 3] - velocity commands (vx, vy, yaw_rate)

#### Training Configuration
```python
# Key hyperparameters (from train.py)
BATCH_SIZE = 256
NUM_EPOCHS = 400
LEARNING_RATE = 3e-4
SAVE_INTERVAL = 10 epochs

# Model architecture
hidden_dim = 512
num_experts = 4
num_layers = 3
```

#### Training Algorithm
1. **Scheduled Sampling Strategy**:
   - Teacher forcing probability starts at 1.0
   - Decays by 0.995 per epoch
   - Minimum teacher probability: 0.3
   - Balances between ground truth guidance and autoregressive learning

2. **Loss Function**:
   - Mean Squared Error (MSE) between predicted and target motion states
   - Averaged over sequence length

3. **Learning Rate Scheduling**:
   - ReduceLROnPlateau scheduler
   - Reduces LR by 0.5 when loss plateaus for 10 epochs
   - Minimum LR: 1e-6

4. **Optimization**:
   - Optimizer: Adam
   - Gradient clipping (via PyTorch defaults)

#### Training Script
```bash
cd CMG_Ref
python train.py
```

**Outputs**:
- Checkpoints: `runs/cmg_YYYYMMDD_HHMMSS/`
- Best model: `cmg_best.pt`
- Periodic checkpoints: `cmg_ckpt_N.pt` (every 10 epochs)
- Final model: `cmg_final.pt`
- TensorBoard logs for tracking training metrics

### 1.3 Model Architecture

#### Overall Structure
```
CMG Model (Conditional Motion Generator)
├── Gating Network
│   └── Computes expert mixing weights from input
├── MoE Layer 1: (motion_dim + command_dim) → 512
├── MoE Layer 2: 512 → 512
└── MoE Layer 3: 512 → motion_dim (58)
```

#### Component Details

**1. Gating Network** (`gating_network.py`):
```python
Input: [batch, motion_dim + command_dim] (61 dims)
Architecture:
  - Linear(61, 512) + ELU
  - Linear(512, num_experts=4) + Softmax
Output: [batch, 4] expert weights
```

**2. MoE Layer** (`moe_layer.py`):
- Contains 4 expert networks (each a linear layer)
- Weighted combination of expert outputs using gating weights
- Formula: `output = sum(weight[i] * expert[i](input) for i in range(4))`

**3. Forward Pass**:
```python
1. Concatenate motion state and command: [batch, 61]
2. Compute expert weights via Gating Network: [batch, 4]
3. For each MoE layer:
   - Apply weighted expert combination
   - Apply ELU activation (except last layer)
4. Output: Next motion state [batch, 58]
```

#### Activation Functions
- ELU (Exponential Linear Unit) used throughout
- No activation on final output layer (regression task)

### 1.4 Input/Output Specifications

#### Training Input/Output
**Input**:
- `motion`: [batch, seq_len+1, 58] 
  - Joint positions (29 dims) + Joint velocities (29 dims)
  - Normalized using dataset statistics
- `command`: [batch, seq_len, 3]
  - [vx, vy, yaw_rate] in local robot frame
  - Normalized to [-1, 1] range

**Output**:
- Predicted next motion state: [batch, 58]

#### Inference Input/Output
**Input**:
- `current_motion`: [1, 58] - Current joint state
- `command`: [1, 3] - Desired velocity [vx (m/s), vy (m/s), yaw (rad/s)]

**Output**:
- `next_motion`: [1, 58] - Predicted next joint state

**Typical Command Ranges**:
- Forward velocity (vx): 0.0 - 3.0 m/s
- Lateral velocity (vy): -0.5 - 0.5 m/s
- Yaw rate: -1.0 - 1.0 rad/s

#### Motion Generation Process
```python
# From eval_cmg.py
1. Load trained model and normalization stats
2. Initialize with a sample motion frame or zeros
3. For each timestep (50 Hz):
   a. Normalize current motion state
   b. Normalize command
   c. Forward pass through CMG model
   d. Denormalize predicted motion
   e. Use prediction as next current motion (autoregressive)
4. Save generated sequence as NPZ file
```

### 1.5 Data Pipeline

#### Data Preparation
**Motion Data Filtering** (from `dataloader.py`):
- Filters locomotion sequences based on velocity criteria:
  - Minimum speed threshold
  - Maximum speed threshold
  - Maximum lateral velocity
  - Maximum yaw rate
  - Uses percentile-based filtering to allow some outliers

**Data Augmentation**:
- Mirror motion symmetry for left/right variations

#### Normalization
- **Motion**: Z-score normalization (mean=0, std=1)
- **Commands**: Min-max normalization to [-1, 1]

---

## 2. TWIST (Teleoperated Whole-Body Imitation System)

### 2.1 Overview

**Purpose**: Train a low-level control policy that can track reference motions from motion capture or CMG on physical humanoid robots.

**Key Features**:
- Two-stage training: Teacher (privileged) → Student (deployable)
- Reinforcement Learning with Behavior Cloning (RL+BC)
- Real-time motion tracking at 50 Hz
- Sim-to-real transfer capability
- Supports Unitree G1, H1, H1_2 robots

**Location**: Main repository root, primarily in `legged_gym/` and `rsl_rl/`

### 2.2 Training Process

#### Stage 1: Teacher Policy Training

**Purpose**: Learn robust motion tracking with access to privileged information (ground truth states, terrain info, etc.)

**Environment**: `g1_priv_mimic`

**Training Command**:
```bash
bash train_teacher.sh EXPERIMENT_NAME cuda:0
```

**Key Configuration** (from `g1_mimic_distill_config.py`):
```python
# Environment
num_envs = 4096
num_actions = 23  # Robot DOF
episode_length_s = 10

# Observation space
n_proprio = 3 + 2 + 3*num_actions = 71
  # 3: projected gravity
  # 2: commands (vx, vy)  
  # 3*23: joint pos, vel, target pos
  
n_priv_mimic_obs = 20 * (8 + 23 + 3*9) = 2040
  # 20 time steps of:
  #   8: root pose (pos + quat)
  #   23: target joint positions
  #   27: key body positions (9 bodies × 3)

n_priv_info = 3 + 1 + 27 + 2 + 4 + 1 + 46 = 84
  # 3: base linear velocity
  # 1: root height
  # 27: key body positions
  # 2: contact mask (feet)
  # 4: privileged latent
  # 1: terrain info
  # 46: friction/restitution

num_observations = 2195 (n_proprio + n_priv_mimic_obs + n_priv_info)
```

**Reward Function Components**:
1. **Tracking Rewards** (primary):
   - Joint DOF tracking
   - Joint velocity tracking
   - Root pose tracking (position + orientation)
   - Root velocity tracking
   - Key body position tracking
   - Feet height tracking

2. **Regularization Rewards**:
   - Action smoothness
   - DOF acceleration penalty
   - Torque limits
   - Contact forces
   - Orientation penalty

**Algorithm**: PPO (Proximal Policy Optimization)
```python
# Key hyperparameters
learning_rate = 2e-4
num_learning_epochs = 5
num_mini_batches = 4
clip_param = 0.2
entropy_coef = 0.01
gamma = 0.99
lam = 0.95
```

**Training Duration**:
- Max iterations: 20,000
- Steps per iteration: 24
- Total timesteps: ~2M per iteration × 20k = 40B+ timesteps
- Training time: 1-2 days on RTX 4090

#### Stage 2: Student Policy Training

**Purpose**: Distill teacher knowledge into a deployable policy without privileged information.

**Environment**: `g1_stu_rl`

**Training Command**:
```bash
bash train_student.sh STUDENT_EXP TEACHER_EXP cuda:0
```

**Key Differences from Teacher**:
- No privileged information (terrain, exact states)
- History encoding: Uses past 10 timesteps of observations
- Smaller observation space (student-only features)
- RL + Behavior Cloning (DAGGER) from teacher

**DAGGER-PPO Algorithm**:
```python
# Behavior cloning component
dagger_coef = 0.1  # Weight for teacher imitation
dagger_coef_anneal_steps = 30000
dagger_update_freq = 20  # Update frequency

# Loss function
total_loss = ppo_loss + dagger_coef * bc_loss
bc_loss = KL_divergence(student_action_dist, teacher_action_dist)
```

**Observation Encoding**:
- Motion Encoder: 1D CNN to encode reference motion sequences
  - Input: [batch, timesteps, n_motion_obs]
  - Conv1D layers to extract temporal features
  - Output: Motion latent vector

### 2.3 Model Architecture

#### Teacher Policy Architecture

**Actor Network** (`actor_critic_mimic.py`):
```
Input: Observations [batch, n_obs]
├── Motion Encoder
│   ├── Linear: n_single_motion_obs → 60
│   ├── Conv1D: 60 → 40 (kernel=8, stride=4)
│   ├── Conv1D: 40 → 20 (kernel=5, stride=1)
│   ├── Conv1D: 20 → 20 (kernel=5, stride=1)
│   └── Linear: 60 → motion_latent_dim (32)
│
├── Proprioception + Motion Latent → MLP
│   ├── Linear: (n_proprio + latent_dim) → 512
│   ├── ELU
│   ├── Linear: 512 → 256
│   ├── ELU
│   ├── Linear: 256 → 128
│   ├── ELU
│   └── Linear: 128 → num_actions (23)
│
Output: Action mean (std is learned separately)
```

**Critic Network**:
```
Input: Privileged observations [batch, n_priv_obs]
├── Same motion encoder
├── MLP: (n_proprio + latent + n_priv_info) → 512 → 256 → 128 → 1
Output: Value estimate
```

#### Student Policy Architecture

Similar to teacher but:
- No privileged information in critic
- Uses history encoding (10 timesteps)
- Adds proprioceptive history to observations

### 2.4 Input/Output Specifications

#### Teacher Policy

**Actor Input** (n_obs = 2195):
```python
# 1. Reference motion (n_priv_mimic_obs = 2040)
# 20 timesteps × [root_pose(8) + dof_pos(23) + key_body_pos(27)]
reference_motion: [batch, 20, 58]

# 2. Proprioception (n_proprio = 71)
projected_gravity: [batch, 3]
commands: [batch, 2]  # vx, vy
dof_pos: [batch, 23]
dof_vel: [batch, 23]
target_dof_pos: [batch, 23]

# 3. Privileged info (n_priv_info = 84)
base_lin_vel: [batch, 3]
root_height: [batch, 1]
key_body_positions: [batch, 27]
contact_mask: [batch, 2]
priv_latent: [batch, 4]
terrain_info: [batch, 1]
friction_restitution: [batch, 46]
```

**Actor Output**:
```python
action: [batch, 23]  # Target joint positions (PD control)
```

**Action to Torque Mapping**:
- Actions are target joint positions
- PD controller converts to torques:
  ```python
  torque = Kp * (action - current_pos) + Kd * (0 - current_vel)
  ```
- Different Kp, Kd for different joint types (hip, knee, ankle, etc.)

#### Student Policy

**Actor Input** (reduced, no privileged info):
```python
# Reference motion (same as teacher)
reference_motion: [batch, 20, 58]

# Proprioception with history
proprioception_history: [batch, 10, n_proprio_single]

# Commands
commands: [batch, 2]
```

**Output**: Same as teacher

### 2.5 Two-Stage Training Pipeline

#### Complete Training Flow

1. **Prepare Motion Dataset**:
   ```bash
   # Download TWIST motion dataset
   # Or generate with CMG (see CMG section)
   ```

2. **Train Teacher Policy**:
   ```bash
   cd legged_gym/legged_gym/scripts
   python train.py --task g1_priv_mimic \
                   --proj_name g1_priv_mimic \
                   --exptid teacher_experiment \
                   --device cuda:0
   ```
   - Trains with full privileged information
   - Uses PPO algorithm
   - Saves checkpoints to `legged_gym/logs/g1_priv_mimic/teacher_experiment/`

3. **Train Student Policy**:
   ```bash
   python train.py --task g1_stu_rl \
                   --proj_name g1_stu_rl \
                   --exptid student_experiment \
                   --teacher_exptid teacher_experiment \
                   --device cuda:0
   ```
   - Distills teacher knowledge via DAGGER
   - Only uses observable information
   - Saves to `legged_gym/logs/g1_stu_rl/student_experiment/`

4. **Export to JIT Model**:
   ```bash
   bash to_jit.sh student_experiment
   ```
   - Creates deployable TorchScript model
   - Output: `traced/student_experiment-XXXXX-jit.pt`

5. **Deploy**:
   - Sim2sim: `python server_low_level_g1_sim.py --policy_path MODEL.pt`
   - Sim2real: `python server_low_level_g1_real.py --policy_path MODEL.pt --net INTERFACE`

#### Training Monitoring

**WandB Integration**:
- Automatic logging to Weights & Biases
- Project name: `{robot}_mimic` (e.g., `g1_mimic`)
- Tracks:
  - Episode rewards
  - Individual reward components
  - Policy loss, value loss
  - Learning rate
  - Episode length

**Key Metrics to Monitor**:
- Total episode reward (should increase)
- Joint tracking error (should decrease)
- Root pose tracking error
- Success rate (episodes without early termination)

---

## 3. Deployment

### 3.1 System Architecture

TWIST uses a two-server architecture:

1. **High-Level Motion Server**:
   - Sends reference motions to the low-level controller
   - Can be motion capture, CMG-generated, or pre-recorded
   - Runs at 50 Hz
   - Uses Redis for communication

2. **Low-Level Controller**:
   - Student policy network
   - Reads reference motion from Redis
   - Outputs joint commands
   - Runs at 50 Hz in simulation, real-time on robot

### 3.2 Deployment Modes

#### Sim2Sim Testing
```bash
# Start low-level controller
cd deploy_real
python server_low_level_g1_sim.py --policy_path PATH/TO/JIT/MODEL

# In another terminal, send motions
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION --vis
```

#### Sim2Real Deployment
```bash
# 1. Connect to robot (Ethernet cable)
# 2. Set laptop IP: 192.168.123.222
# 3. Test connection: ping 192.168.123.164
# 4. Enter dev mode on robot (L2+R2 on remote)

# Start low-level controller
python server_low_level_g1_real.py \
    --policy_path PATH/TO/JIT/MODEL \
    --net YOUR_NETWORK_INTERFACE

# In another terminal, send motions
python server_high_level_motion_lib.py --motion_file PATH/TO/MOTION --vis
```

### 3.3 Control Flow

```
User Input / Teleop / CMG
    ↓
High-Level Motion Server (50 Hz)
    ↓ (Redis)
Reference Motion Buffer
    ↓
Low-Level Student Policy (50 Hz)
    ↓
Joint Position Commands
    ↓
PD Controller
    ↓
Robot Actuators
```

---

## 4. Key Integration Points

### 4.1 Motion Format Compatibility

Both CMG and TWIST use similar motion representations:

**CMG Output Format**:
```python
motion = [T, 58]  # 29 joint pos + 29 joint vel
# Saved as NPZ with additional metadata
```

**TWIST Motion Format** (from motion dataset):
```python
{
  'dof_positions': [T, num_dofs],
  'dof_velocities': [T, num_dofs],
  'body_positions': [T, num_bodies, 3],
  'body_rotations': [T, num_bodies, 4],
  'fps': 50
}
```

**Compatibility Note**: CMG generates 29-DOF motions which may need remapping to specific robot DOF (e.g., G1 has 23 DOF).

### 4.2 Frequency Alignment

- **CMG**: 50 Hz generation
- **TWIST**: 50 Hz control loop
- ✅ **Already aligned**

### 4.3 Coordinate Frames

- **CMG**: Local robot frame (commands in robot coordinates)
- **TWIST**: World frame with local tracking
- ⚠️ **May require transformation**

---

## 5. Summary

### CMG Strengths
- Generate diverse motions from velocity commands
- No motion capture required
- Flexible, on-demand motion generation
- Fast inference

### TWIST Strengths
- Robust motion tracking on physical robots
- Sim-to-real transfer
- Handles terrain variations
- Real-time performance

### Integration Benefits
By combining CMG and TWIST:
1. **CMG** generates reference motions from high-level commands
2. **TWIST** tracks these motions on the physical robot
3. Enables command-based control without motion capture
4. Supports diverse locomotion behaviors (walk, run, turn)

---

## 6. References

- **TWIST Paper**: [arXiv:2505.02833](https://arxiv.org/abs/2505.02833)
- **TWIST Website**: https://humanoid-teleop.github.io/
- **CMG Workspace**: https://github.com/PMY9527/cmg_workspace
- **Repository**: Current repository

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-29  
**Authors**: Based on code analysis of TWIST_CMG repository
