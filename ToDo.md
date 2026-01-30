# ToDo List: CMG-TWIST Integration for Walking

This document outlines the tasks needed to integrate the Conditional Motion Generator (CMG) with TWIST to achieve command-based walking control on humanoid robots.

---

## Overview

**Goal**: Enable a humanoid robot to walk based on velocity commands by using CMG to generate reference motions and TWIST to track them.

**Current State**:
- ✅ CMG can generate motions from velocity commands
- ✅ TWIST can track reference motions on physical robots
- ❌ Integration pipeline is not yet established
- ❌ Joint mappings need to be defined
- ❌ End-to-end testing not performed

---

## Phase 1: Data Format Alignment

### 1.1 Joint Mapping
**Priority**: HIGH  
**Effort**: Medium

- [ ] **Task 1.1.1**: Document CMG's 29-DOF joint layout
  - Map each of the 29 joints to body parts (hips, knees, ankles, etc.)
  - Document joint order and naming convention
  - Create a joint index reference document

- [ ] **Task 1.1.2**: Document G1 robot's 23-DOF joint layout
  - Cross-reference with `g1_mimic_distill_config.py`
  - List joints: 6 per leg, 3 waist, 4 per arm
  - Document joint names and order

- [ ] **Task 1.1.3**: Create joint mapping function
  - File: `CMG_Ref/utils/joint_mapping.py`
  - Function: `map_cmg_to_g1(cmg_motion) -> g1_motion`
  - Handle DOF differences (29 → 23)
  - Options:
    - Drop unused joints (e.g., fingers, extra arm DOFs)
    - Use subset of joints
    - Or retrain CMG with G1's 23 DOFs

- [ ] **Task 1.1.4**: Validate joint mapping
  - Visualize mapped motions in MuJoCo
  - Check for discontinuities or invalid poses
  - Test with sample CMG-generated motions

### 1.2 Motion Format Converter
**Priority**: HIGH  
**Effort**: Medium

- [ ] **Task 1.2.1**: Create CMG-to-TWIST motion converter
  - File: `CMG_Ref/utils/motion_converter.py`
  - Function: `cmg_npz_to_twist_format(cmg_npz_path, output_pkl_path)`
  - Input: CMG's NPZ format (from `eval_cmg.py`)
  - Output: TWIST's PKL format (compatible with MotionLib)
  
  - Required fields for TWIST:
    ```python
    {
      'dof_positions': [T, 23],  # After joint mapping
      'dof_velocities': [T, 23],
      'body_positions': [T, num_bodies, 3],  # Root + key bodies
      'body_rotations': [T, num_bodies, 4],  # Quaternions
      'fps': 50,
      'dof_names': List[str],
      'body_names': List[str]
    }
    ```

- [ ] **Task 1.2.2**: Implement forward kinematics
  - Use existing FK from `pose/pose/util_funcs/kinematics_model.py`
  - Compute body positions and rotations from joint angles
  - Validate against reference motion data

- [ ] **Task 1.2.3**: Test motion converter
  - Convert sample CMG motions
  - Load in TWIST's MotionLib
  - Verify no errors during loading
  - Visualize converted motions

---

## Phase 2: CMG Training Data Alignment

### 2.1 Training Data for G1 Robot
**Priority**: MEDIUM  
**Effort**: High

- [ ] **Task 2.1.1**: Prepare G1-specific training data for CMG
  - Option A: Use TWIST's existing motion dataset
    - Extract dof_positions and dof_velocities
    - Compute commands (vx, vy, yaw) from root velocities
    - Convert to CMG training format
  
  - Option B: Collect new motion data
    - Record G1 walking/running motions in simulation
    - Label with velocity commands
    - Build training dataset

- [ ] **Task 2.1.2**: Data preprocessing script
  - File: `CMG_Ref/dataloader/prepare_g1_data.py`
  - Convert TWIST PKL → CMG PT format
  - Compute statistics (mean, std, min, max)
  - Apply data filtering (locomotion only)
  - Save as `cmg_g1_training_data.pt`

- [ ] **Task 2.1.3**: Retrain CMG on G1 data
  - Update `train.py` to use G1 dataset
  - Set `motion_dim = 46` (23 pos + 23 vel)
  - Train for 400 epochs
  - Validate generated motions

---

## Phase 3: Integration Pipeline

### 3.1 CMG-TWIST Bridge
**Priority**: HIGH  
**Effort**: Medium

- [ ] **Task 3.1.1**: Create velocity command interface
  - File: `deploy_real/cmg_motion_generator.py`
  - Class: `CMGMotionGenerator`
  - Methods:
    - `generate_motion(vx, vy, yaw, duration)`
    - `get_next_frame()` for real-time generation
  - Load CMG model on initialization
  - Handle autoregressive generation

- [ ] **Task 3.1.2**: Integrate with high-level motion server
  - Modify `server_high_level_motion_lib.py`
  - Add CMG generation mode
  - New flag: `--use_cmg` and `--cmg_model_path`
  - Generate motions on-the-fly from velocity commands
  - Send to Redis buffer for low-level controller

- [ ] **Task 3.1.3**: Command input interface
  - Option A: Keyboard control
    - Arrow keys for direction
    - +/- for speed adjustment
  
  - Option B: Gamepad/Joystick
    - Left stick: vx, vy
    - Right stick: yaw
  
  - Option C: Voice commands
    - "forward", "backward", "turn left", etc.
    - Convert to velocity commands

- [ ] **Task 3.1.4**: Smooth command transitions
  - Implement command interpolation
  - Avoid sudden velocity changes
  - Ramp up/down speeds gradually

### 3.2 Real-time Generation Pipeline
**Priority**: MEDIUM  
**Effort**: Medium

- [ ] **Task 3.2.1**: Optimize CMG inference speed
  - Profile current inference time
  - Target: < 20ms per frame (50 Hz)
  - Optimize model if needed (quantization, pruning)
  - Use GPU for inference

- [ ] **Task 3.2.2**: Implement motion buffer
  - Pre-generate short motion sequences (1-2 seconds)
  - Maintain buffer of upcoming frames
  - Regenerate when commands change

- [ ] **Task 3.2.3**: Handle command transitions
  - Smooth blending between different velocity commands
  - Interpolate between generated motion sequences
  - Avoid discontinuities

---

## Phase 4: Coordinate Frame Alignment

### 4.1 Coordinate Transformations
**Priority**: MEDIUM  
**Effort**: Low-Medium

- [ ] **Task 4.1.1**: Document coordinate frames
  - CMG: Robot-centric frame (forward = +X, left = +Y, up = +Z)
  - TWIST: World frame with robot tracking
  - Identify any discrepancies

- [ ] **Task 4.1.2**: Implement frame transformations
  - File: `CMG_Ref/utils/frame_transforms.py`
  - Convert velocity commands to appropriate frame
  - Transform generated motions if needed

- [ ] **Task 4.1.3**: Test frame transformations
  - Verify robot moves in expected direction
  - Test turning in place
  - Validate lateral motion

---

## Phase 5: Testing and Validation

### 5.1 Simulation Testing
**Priority**: HIGH  
**Effort**: Medium

- [ ] **Task 5.1.1**: Test in Isaac Gym simulation
  - Generate motion with CMG from velocity command
  - Track with TWIST student policy
  - Verify stable walking

- [ ] **Task 5.1.2**: Command response testing
  - Test various velocity commands:
    - Forward walk: vx = 0.5, 1.0, 1.5 m/s
    - Backward walk: vx = -0.5 m/s
    - Lateral walk: vy = ±0.3 m/s
    - Turning: yaw = ±0.5 rad/s
    - Combinations: forward + turn, etc.

- [ ] **Task 5.1.3**: Stress testing
  - Sudden command changes
  - Maximum velocity limits
  - Long-duration walking (> 1 minute)
  - Recovery from disturbances

- [ ] **Task 5.1.4**: Compare with motion capture baseline
  - Track same reference motion
  - Compare CMG-generated vs. MoCap-generated
  - Measure tracking error, success rate

### 5.2 Physical Robot Testing
**Priority**: HIGH  
**Effort**: High

- [ ] **Task 5.2.1**: Deploy on G1 robot (tethered)
  - Start with robot suspended
  - Test joint commands safety
  - Gradually lower to ground

- [ ] **Task 5.2.2**: Basic walking tests
  - Forward walking on flat ground
  - Verify stable gait
  - Measure maximum safe speed

- [ ] **Task 5.2.3**: Command-based control tests
  - Joystick control
  - Test all velocity command ranges
  - Verify responsiveness

- [ ] **Task 5.2.4**: Robustness testing
  - Walking on uneven terrain
  - External pushes/disturbances
  - Long-duration operation

---

## Phase 6: Optimization and Refinement

### 6.1 Performance Optimization
**Priority**: MEDIUM  
**Effort**: Medium

- [ ] **Task 6.1.1**: Optimize motion quality
  - Tune CMG generation parameters
  - Adjust TWIST tracking weights
  - Improve smoothness

- [ ] **Task 6.1.2**: Reduce latency
  - Profile end-to-end latency
  - Optimize critical paths
  - Target < 50ms total latency

- [ ] **Task 6.1.3**: Improve energy efficiency
  - Analyze joint torques
  - Optimize CMG for natural motions
  - Reduce unnecessary movements

### 6.2 Safety Features
**Priority**: HIGH  
**Effort**: Medium

- [ ] **Task 6.2.1**: Implement safety checks
  - Velocity limits (prevent over-speed)
  - Joint angle limits
  - Torque limits
  - Fall detection

- [ ] **Task 6.2.2**: Emergency stop mechanism
  - Kill switch (remote control button)
  - Automatic stop on failure detection
  - Safe shutdown procedure

- [ ] **Task 6.2.3**: Collision avoidance (optional)
  - Use sensors if available
  - Stop before obstacles
  - Adjust commands for safety

---

## Phase 7: Documentation and Deployment

### 7.1 Documentation
**Priority**: MEDIUM  
**Effort**: Low

- [ ] **Task 7.1.1**: Write integration guide
  - Step-by-step setup instructions
  - Configuration parameters
  - Troubleshooting guide

- [ ] **Task 7.1.2**: Create usage examples
  - Example scripts for common tasks
  - Video demonstrations
  - Best practices

- [ ] **Task 7.1.3**: API documentation
  - Document all new functions/classes
  - Add docstrings
  - Generate API reference

### 7.2 Deployment Package
**Priority**: MEDIUM  
**Effort**: Low

- [ ] **Task 7.2.1**: Create deployment scripts
  - One-command deployment
  - Configuration files
  - Launch scripts

- [ ] **Task 7.2.2**: Package dependencies
  - Update requirements.txt
  - Test installation on clean system
  - Create Docker container (optional)

---

## Quick Start Checklist

For rapid prototyping, focus on these essential tasks first:

1. ✅ **Essential Path**:
   - [ ] Task 1.1.3: Create joint mapping (29 → 23 DOF)
   - [ ] Task 1.2.1: Motion format converter (NPZ → PKL)
   - [ ] Task 3.1.1: CMG motion generator class
   - [ ] Task 3.1.2: Integrate with motion server
   - [ ] Task 5.1.1: Test in simulation
   - [ ] Task 5.2.1: Test on physical robot (tethered)

2. 🔧 **If issues arise**:
   - Go back to Phase 2: Retrain CMG with G1-specific data
   - Fine-tune TWIST student policy for CMG-generated motions

---

## Key Dependencies and Risks

### Dependencies
1. **CMG → TWIST**: Motion format must be compatible
2. **Joint Mapping**: Critical for correct motion transfer
3. **Coordinate Frames**: Must align for correct direction control
4. **Timing**: Both systems must run at 50 Hz

### Risks and Mitigations
1. **Risk**: CMG motions may not be trackable by TWIST
   - **Mitigation**: Retrain CMG with G1's actual DOF and motion range
   - **Mitigation**: Fine-tune TWIST on CMG-generated data

2. **Risk**: Real-time performance issues
   - **Mitigation**: Optimize inference (TorchScript, quantization)
   - **Mitigation**: Pre-generate motion buffers

3. **Risk**: Safety concerns on physical robot
   - **Mitigation**: Extensive simulation testing first
   - **Mitigation**: Start with tethered/suspended robot
   - **Mitigation**: Implement emergency stop

4. **Risk**: Poor motion quality
   - **Mitigation**: Iterate on CMG training data quality
   - **Mitigation**: Tune reward weights in TWIST

---

## Success Metrics

### Minimum Viable Product (MVP)
- [ ] Robot can walk forward at 0.5 m/s using CMG commands
- [ ] Robot can stop when commanded
- [ ] System runs without crashes for 30 seconds

### Target Performance
- [ ] Walk forward at 1.0+ m/s
- [ ] Turn in place at 0.3 rad/s
- [ ] Smooth transitions between commands
- [ ] Operate continuously for 5+ minutes
- [ ] Tracking error < 10cm over 10 meters

### Stretch Goals
- [ ] Dynamic running (> 1.5 m/s)
- [ ] Complex commands (circle, figure-8)
- [ ] Terrain adaptation
- [ ] Push recovery

---

## Timeline Estimate

**Fast Track** (assume existing CMG model works):
- Phase 1: 1 week (data alignment)
- Phase 3: 1 week (integration)
- Phase 5.1: 3 days (sim testing)
- Phase 5.2: 1 week (robot testing)
- **Total**: ~3-4 weeks

**Full Track** (with CMG retraining):
- Phase 1: 1 week
- Phase 2: 2 weeks (data prep + training)
- Phase 3: 1 week
- Phase 4: 2 days
- Phase 5: 2 weeks (testing + iteration)
- Phase 6: 1 week (optimization)
- **Total**: ~7-8 weeks

---

## Next Steps

1. **Immediate**: Start with Task 1.1.1-1.1.4 (joint mapping)
2. **Day 1-2**: Tasks 1.2.1-1.2.3 (motion converter)
3. **Day 3-4**: Task 3.1.1 (CMG integration class)
4. **Day 5**: Task 5.1.1 (simulation test)
5. **Week 2+**: Iterate based on results

---

## Implementation Recommendations and Improvements

This section outlines key improvements and design considerations for the CMG-TWIST integration project.

### 1. Data Format & Joint Mapping (数据格式与关节映射)
**Priority**: HIGH  
**Effort**: High

#### 1.1 29 → 23 DOF Joint Mapping (29 → 23 DOF 关节映射)
**Status**: Need mapping or retraining

- [ ] **Document CMG's 29-DOF and G1's 23-DOF configurations**
  - Current state: CMG uses 29 DOF, G1 robot has 23 DOF
  - **Recommendation**: Either:
    - **Option A**: Write `map_cmg_to_g1()` function to convert 29→23 DOF
    - **Option B**: Retrain CMG native on G1's 23 DOF (preferred for consistency)
  - Mapping considerations:
    - List all 29 CMG joints and corresponding G1 joints
    - Identify unused joints (fingers, extra arm DOFs)
    - Create joint index mapping table
  - Benefits of retraining:
    - Eliminates conversion errors
    - CMG model matches deployment configuration
    - Better motion quality on G1
  - Implementation:
    - File: `CMG_Ref/utils/joint_mapping.py` (if choosing mapping)
    - Or: Prepare G1 training data and retrain CMG (if choosing retraining)
    - Validate in MuJoCo before deployment

#### 1.2 Motion Format Conversion (运动格式转换)
**Status**: In-memory only, no unified format

- [ ] **Create NPZ → PKL format converter**
  - Current state: Trajectories read directly in memory from eval_cmg.py
  - **Recommendation**: Standardize to TWIST's PKL format for consistency
  - File: `CMG_Ref/utils/motion_converter.py`
  - Function: `cmg_npz_to_twist_format(cmg_npz, output_pkl)`
  - Required fields:
    ```python
    {
      'dof_positions': [T, 23],
      'dof_velocities': [T, 23],
      'body_positions': [T, num_bodies, 3],
      'body_rotations': [T, num_bodies, 4],  # Quaternions
      'fps': 50,
      'dof_names': List[str],
      'body_names': List[str]
    }
    ```

#### 1.3 Forward Kinematics Implementation (前向运动学)
**Status**: Not yet implemented

- [ ] **Compute body transforms from joint angles**
  - Use existing FK from `pose/pose/util_funcs/kinematics_model.py`
  - Input: Joint positions [23 DOF]
  - Output: Body positions + rotations for all bodies
  - Integrate into motion converter
  - Test: Compare computed vs. reference body positions

#### 1.4 G1 Training Data Preparation (训练数据对齐)
**Status**: Using original 29-DOF CMG model

- [ ] **Prepare G1-specific dataset for CMG retraining**
  - If Option B chosen: Retrain CMG with native 23-DOF support
  - Data source: Extract from TWIST's existing motion library
  - Processing:
    - Convert TWIST PKL → CMG training format
    - Compute velocity commands from root motion
    - Apply data filtering (locomotion only)
    - Compute statistics (mean, std, min, max)
  - File: `CMG_Ref/dataloader/prepare_g1_data.py`
  - Output: `cmg_g1_training_data.pt`
  - Retraining:
    - Update `train.py` with `motion_dim = 46` (23 pos + 23 vel)
    - Train for 400 epochs
    - Validate generated motions

### 2. Integration & Real-time Pipeline (集成与实时管道)
**Priority**: HIGH  
**Effort**: Medium

#### 2.1 CMG-TWIST Bridge Class (CMG–TWIST 桥接类)
**Status**: Direct environment integration, no abstraction

- [ ] **Create CMGMotionGenerator class for motion generation**
  - Current state: Direct trajectory generation in environment, no standalone interface
  - File: `deploy_real/cmg_motion_generator.py`
  - Class: `CMGMotionGenerator`
  - Methods:
    - `__init__(model_path, device='cuda')`: Load CMG model
    - `generate_motion(vx, vy, yaw, duration)`: Generate motion sequence
    - `get_next_frame()`: Retrieve next frame in real-time generation
    - `update_command(vx, vy, yaw)`: Update velocity command on-the-fly
  - Benefits:
    - Decouples CMG from environment code
    - Enables command changes during motion generation
    - Reusable across different controllers
  - Handles:
    - Autoregressive motion generation
    - Velocity command integration
    - Frame buffering

#### 2.2 High-Level Motion Server Integration (高层运动服务器集成)
**Status**: Fixed speed commands, no CMG mode

- [ ] **Integrate CMG generator into server_high_level_motion_lib.py**
  - Current state: Speed commands are fixed (e.g., [1.5, 0, 0])
  - **Recommendation**: Add CMG generation mode with command support
  - New parameters:
    - `--use_cmg`: Enable CMG motion generation
    - `--cmg_model_path`: Path to trained CMG model
    - `--use_cmg_command_input`: Enable command input (keyboard/gamepad/voice)
  - Functionality:
    - Generate motions from velocity commands
    - Send to Redis buffer for low-level controller
    - Support command switching during execution
  - Implementation:
    - Initialize CMGMotionGenerator at server startup
    - Receive commands from input interface
    - Generate motion frames at 50 Hz
    - Write to shared buffer

#### 2.3 Command Input Interface (命令输入与插值)
**Status**: Fixed velocity commands only

- [ ] **Implement smooth command input with interpolation**
  - Current state: Velocity fixed at [1.5, 0, 0], no user input
  - **Option A - Keyboard Control**:
    - W/S: Forward/backward (vx)
    - A/D: Left/right strafe (vy)
    - Q/E: Rotate (yaw)
    - +/-: Speed adjustment
  - **Option B - Gamepad/Joystick**:
    - Left stick: (vx, vy) analog control
    - Right stick: yaw rotation
    - Trigger buttons: speed ramp
  - **Option C - Voice Commands**:
    - "go forward", "turn left", "stop"
    - Convert to velocity commands
  - Smooth interpolation:
    - Avoid sudden velocity changes
    - Ramp speeds gradually over 0.2-0.5s
    - Smooth transitions between commands

#### 2.4 Real-time Generation Optimization (实时生成优化)
**Status**: Batch generation at episode reset, no real-time adaptation

- [ ] **Optimize inference speed and handle command changes**
  - Current state: Generate 2s trajectory at episode start, static thereafter
  - Issues:
    - No support for command changes mid-motion
    - Potential performance peaks
    - No real-time responsiveness
  - **Optimization targets**:
    - Inference time: < 20ms per frame (50 Hz requirement)
    - Enable command updates every 0.1-0.5s
    - Pre-generate motion buffers (1-2 seconds ahead)
  - Implementation:
    - Profile current inference speed (cuda/cpu)
    - Optimize if needed: quantization, TorchScript JIT
    - Use motion buffer: maintain queue of next N frames
    - Regenerate on command change or buffer depletion
    - Smooth blending when transitioning between sequences
  - Code structure:
    ```python
    class CMGMotionGenerator:
        def __init__(self, model, buffer_size=100):  # ~2s at 50Hz
            self.buffer = deque(maxlen=buffer_size)
            self.current_cmd = [0, 0, 0]
        
        def get_next_frame(self):
            if len(self.buffer) < buffer_size/2:
                self._refill_buffer()
            return self.buffer.popleft()
        
        def update_command(self, vx, vy, yaw):
            # Smooth transition to new command
            self._interpolate_command(vx, vy, yaw)
            # Schedule buffer regeneration
        
        def _refill_buffer(self):
            # Generate next N frames with current command
            pass
    ```

### 3. Coordinate System & Testing (坐标系与测试)
**Priority**: MEDIUM  
**Effort**: Low-Medium

#### 3.1 Coordinate System Alignment (坐标系对齐与转换)
**Status**: Not discussed, critical for deployment

- [ ] **Document and align coordinate frames**
  - CMG frame: Robot-centric (forward=+X, left=+Y, up=+Z)
  - TWIST frame: World frame with robot tracking
  - **Required transformations**:
    - Velocity command frame (input) → Robot frame (CMG input)
    - CMG output (robot frame) → World frame (TWIST input)
  - Implementation:
    - File: `CMG_Ref/utils/frame_transforms.py`
    - Functions:
      - `cmd_to_robot_frame(v_world)`: Convert command to robot frame
      - `motion_to_world_frame(motion_robot)`: Convert motion output
    - Validation:
      - Verify forward command moves robot in correct direction
      - Test turning in place
      - Validate lateral motion
  - Testing:
    - Visual inspection in MuJoCo
    - Real robot deployment (tethered)

#### 3.2 System Testing & Verification (系统测试与验证)
**Status**: Only training process documented, no validation plan

- [ ] **Comprehensive testing plan**
  - **Simulation Testing**:
    - Various velocity commands (forward, backward, strafe, turn)
    - Command response times and smoothness
    - Stress testing: sudden changes, max velocities, long duration
    - Error tracking: deviation from expected trajectory
    - Compare CMG-generated vs. MoCap reference motions
  
  - **Physical Robot Testing** (after simulation passes):
    - Tethered walking on flat ground
    - Basic motion stability and safety
    - Command responsiveness
    - Robustness to disturbances
    - Uneven terrain (if applicable)
  
  - **Safety Validation**:
    - Joint angle limit enforcement
    - Velocity/torque limits
    - Fall detection
    - Emergency stop functionality

#### 3.3 Safety & Optimization Measures (安全与优化措施)
**Status**: Basic joint tracking only, no safety/energy optimization

- [ ] **Implement safety features and motion optimization**
  - **Safety measures**:
    - Emergency stop (kill switch)
    - Joint angle hard limits with soft boundaries
    - Velocity saturation (prevent over-speed)
    - Torque/power limits
    - Fall detection and recovery
  
  - **Motion Quality Optimization**:
    - Tune CMG generation parameters
    - Adjust TWIST tracking weights (reference: [508])
    - Improve smoothness and naturalness
    - Energy efficiency analysis
  
  - **Performance Optimization**:
    - Reduce latency (target: < 50ms total)
    - Optimize inference speed (target: < 20ms)
    - Minimize jerk and discontinuities
    - Profile critical paths

### 4. Observation & Reward Improvements (奖励和观察改进)
**Priority**: MEDIUM  
**Effort**: Low-Medium

#### 4.1 Foot Contact & Joint Torque Observations (足接触与关节扭矩观察)
**Status**: Limited to position/velocity, no force feedback

- [ ] **Add foot contact sensing as proprioceptive feedback**
  - Current: Only joint pos/vel + limited privileged info
  - Add binary foot contact sensors: `[batch, 2]` (left, right foot)
  - Benefits:
    - Better ground interaction awareness
    - Improved gait phase detection
    - More robust terrain adaptation
  - Implementation:
    - Use MuJoCo contact sensor or compute from contact forces
    - Include in both teacher and student observations
    - Normalize to [0, 1] range

- [ ] **Include joint torque feedback in observations**
  - Current: Missing actual torque information
  - Add joint torques: `[batch, 23]` (all 23 DOF)
  - Benefits:
    - Force awareness for control
    - Better impedance regulation
    - Improved force sensing through policy
  - Implementation:
    - Extract from simulator: `data.qfrc_applied` or `data.qfrc_constraint`
    - Normalize by maximum torque limits
    - Include in proprioceptive observation
    - Update observation dimension: `n_proprio += 23 + 2 = +25 dims`

#### 4.2 Enhanced Reward Function (丰富奖励函数)
**Status**: Only joint tracking, no explicit velocity/posture penalties

- [ ] **Add comprehensive locomotion reward components**
  - Current reward focuses on tracking (reference [508])
  - **Recommended additions**:
  
    a. **Linear velocity tracking**:
    ```python
    r_lin_vel = -w_lin * ||v_base - v_cmd||²  # w_lin = 1.0
    ```
  
    b. **Angular velocity tracking**:
    ```python
    r_ang_vel = -w_ang * ||ω_base - ω_cmd||²  # w_ang = 0.5
    ```
  
    c. **Base orientation (keep upright)**:
    ```python
    r_orient = -w_orient * ||proj_gravity - [0,0,-1]||²  # w_orient = 1.0
    ```
  
    d. **Foot slip penalty**:
    ```python
    r_slip = -w_slip * Σ(||v_foot_xy|| * contact)  # w_slip = 0.1
    ```
  
    e. **Action rate penalty**:
    ```python
    r_action_rate = -w_rate * ||action_t - action_{t-1}||²  # w_rate = 0.01
    ```
  
  - Implementation:
    - Update `g1_mimic_distill_config.py` reward weights
    - Add these terms to compute_reward() in environment
    - Log individual components for debugging
    - Tune weights through training iterations

#### 4.3 Gait Phase Guidance (步态相位指导)
**Status**: No explicit phase signal, implicit through future frames

- [ ] **Add gait phase input to guide policy**
  - Current: TWIST infers gait implicitly from future reference frames
  - **Benefits of explicit signal**:
    - Better swing/stance synchronization
    - Improved foot placement timing
    - More natural gait transitions
  - **Implementation options**:
    - **Option A - Sinusoidal phase** (per leg):
      ```python
      phase_left = sin(2π * t * freq)   # Left leg phase
      phase_right = sin(2π * t * freq + π)  # Right leg phase (offset)
      # Add to observation: [batch, 4] (sin/cos for each leg)
      ```
    - **Option B - Discrete gait state**:
      ```python
      gait_state = [L_stance, L_swing, R_stance, R_swing]  # One-hot like
      # Shape: [batch, 4]
      ```
    - **Option C - Future contact schedule**:
      ```python
      contact_schedule = future_ref_contacts[t:t+horizon]  # [batch, horizon, 2]
      ```
  - Add to observation space (both teacher and student)
  - Update `n_proprio` or create separate gait observation group

### 5. Terrain Adaptation & Environment (地形适应与环境生成)
**Priority**: HIGH  
**Effort**: High

#### 5.1 Enhanced Teacher Terrain Information (增强特权地形信息)
**Status**: Flat terrain only, no terrain-aware training

- [ ] **Add terrain height map to teacher observations**
  - Current: No terrain information in privileged observations
  - Benefits:
    - Better terrain adaptation during training
    - More robust student policy via distillation
    - Enables complex terrain walking
  - **Option A - Height map approach**:
    ```python
    # Grid-based terrain sampling
    terrain_height_map: [batch, grid_size, grid_size]  # e.g., 20×20
    # Sample area under/ahead of robot (1m × 1m with 0.05m resolution)
    ```
  - **Option B - Ray-casting approach**:
    ```python
    # Directional terrain sensing
    terrain_rays: [batch, num_rays, 2]  # distance + height for each ray
    # 16 rays × 2 = 32 dims (more efficient)
    ```
  - Implementation:
    - Add height map sensor to privileged observations
    - Update observation space in training config
    - Include terrain features in teacher policy input
    - Ensure student can eventually access similar info

#### 5.2 Configurable Terrain Generator (可配置地形生成器)
**Status**: Flat ground only, no procedural generation

- [ ] **Implement procedural terrain with difficulty curriculum**
  - Current: All training on flat terrain, no complexity variation
  - **Terrain types to support**:
    - Flat ground (baseline)
    - Slopes (0-15° adjustable)
    - Stairs (5-15cm height, 20-40cm depth)
    - Random rough terrain (Perlin noise)
    - Stepping stones
    - Mixed terrain combinations
  
  - **Friction variation**:
    ```python
    friction_range = [0.4, 1.2]  # Low (ice ~0.1) to high (rubber ~1.0)
    # Randomize per terrain patch for domain randomization
    ```
  
  - **Difficulty curriculum**:
    ```python
    difficulty_schedule = {
        0: "flat",           # Iter 0-2k
        2000: "low_slopes",  # Iter 2k-5k
        5000: "stairs",      # Iter 5k-10k
        10000: "rough",      # Iter 10k-15k
        15000: "mixed"       # Iter 15k+
    }
    ```
  
  - Implementation:
    - File: `legged_gym/legged_gym/envs/terrain_generator.py`
    - Class: `ProceduralTerrainGenerator`
    - Methods:
      - `generate_flat_terrain()`
      - `generate_slope_terrain(angle_range, num_slopes)`
      - `generate_stairs_terrain(step_height, step_depth)`
      - `generate_rough_terrain(roughness, frequency)`
      - `generate_mixed_terrain(difficulty_level)`
    - Configuration structure:
      ```python
      terrain_config = {
          "terrain_type": "mixed",
          "terrain_size": [10.0, 10.0],  # meters
          "horizontal_scale": 0.05,      # resolution
          "vertical_scale": 0.005,       # height precision
          "slope_threshold": 0.75,       # max slope angle
          "friction_range": [0.5, 1.0],
          "restitution_range": [0.0, 0.3],
          "curriculum_enabled": True,
          "difficulty_level": 0.5  # 0.0=easy, 1.0=hard
      }
      ```
  
  - Integration with training:
    - Add terrain config to `g1_mimic_distill_config.py`
    - Generate new terrains per episode or periodically
    - Include terrain parameters in privileged information
    - Apply domain randomization (friction, restitution)

#### 5.3 Training Data Alignment & DOF Consistency (训练数据对齐与 DOF 一致性)
**Status**: CMG uses 29 DOF, suggestion to retrain on G1's 23 DOF

- [ ] **Decide between mapping vs. retraining**
  - **Option A - Mapping approach** (faster, short-term):
    - Implement 29→23 DOF conversion function
    - Handle unused joints (fingers, extra arm DOFs)
    - Pro: Reuse existing CMG model
    - Con: Conversion errors may accumulate
  
  - **Option B - Retraining** (recommended, long-term):
    - Prepare G1-specific dataset from TWIST motion library
    - Retrain CMG with native 23 DOF support
    - Pro: Better motion quality, no conversion overhead
    - Con: Requires training time (~1-2 weeks)
    - **Strongly recommended** for production deployment
  
  - **Recommendation**: Start with Option A for quick testing, then do Option B after initial validation

- [ ] **Do NOT add terrain complexity until DOF is unified**
  - Reason: Mapping errors could compound with terrain difficulty
  - Wait for Option B (retraining) before adding complex terrain
  - Testing order:
    1. DOF mapping/retraining + flat terrain validation
    2. Then introduce slope/stair/rough terrain
    3. Finally train with mixed terrain curriculum

---

## Phase 6: Documentation & Deployment (项目文档和部署)
**Priority**: MEDIUM  
**Effort**: Medium

### 6.1 Integration Documentation (集成文档完善)
**Status**: Partial documentation, missing complete integration guide

- [ ] **Write comprehensive integration and setup guide**
  - Complete installation instructions (dependencies, versions)
  - Step-by-step integration tutorial
  - Configuration parameter documentation
  - Troubleshooting guide with common issues
  - Debug techniques and performance profiling

- [ ] **Create API documentation**
  - Document CMGMotionGenerator class
  - Document motion converter functions
  - Document coordinate transformation functions
  - Add type hints and docstrings
  - Generate reference documentation

- [ ] **Develop usage examples**
  - Example: Keyboard control for motion generation
  - Example: Joystick-based command input
  - Example: Batch motion conversion
  - Example: Real-time command updating
  - Example: Performance monitoring

### 6.2 Deployment Package & Scripts (部署脚本和依赖打包)
**Status**: Deployment section not yet discussed

- [ ] **Create deployment scripts**
  - One-command setup script for development
  - Installation script for production
  - Configuration file templates
  - Launch scripts (server, client, etc.)
  - Environment variable setup

- [ ] **Package and organize dependencies**
  - Update main requirements.txt with all dependencies
  - Create requirements files by component:
    - `requirements_cmg.txt` (CMG and conversion)
    - `requirements_twist.txt` (TWIST training)
    - `requirements_deploy.txt` (deployment only)
  - Test installation on clean system
  - Document Python version compatibility (3.8+)

- [ ] **Create Docker container** (optional but recommended)
  - Dockerfile with all dependencies
  - docker-compose.yml for multi-container setup
  - Volume mounts for models and data
  - Environment variable configuration
  - Test image build and runtime

### 6.3 Coordinate System & Format Alignment (坐标与频率对齐)
**Status**: Frequency aligned (50 Hz), coordinates not yet addressed

- [ ] **Document coordinate frame alignment**
  - Reference: ProjectDocumentation.zh.md mentions 50 Hz frequency
  - Document CMG's robot-centric frame
  - Document TWIST's world frame
  - Explain any frame conversions needed
  - Create visual diagrams if possible

- [ ] **Ensure motion format compatibility**
  - Confirm 50 Hz synchronization between CMG and TWIST
  - Document PKL format fields and requirements
  - Validate NPZ to PKL conversion
  - Test round-trip conversion quality

---

## Contact and Support

For questions or issues during integration:
- CMG questions: Refer to original CMG workspace
- TWIST questions: yanjieze@stanford.edu
- Integration issues: Create issue in this repository

---

**Document Version**: 2.0  
**Last Updated**: 2026-01-30  
**Status**: Comprehensive integration plan with all phases defined
