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

## Contact and Support

For questions or issues during integration:
- CMG questions: Refer to original CMG workspace
- TWIST questions: yanjieze@stanford.edu
- Integration issues: Create issue in this repository

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-29  
**Status**: Initial integration plan
