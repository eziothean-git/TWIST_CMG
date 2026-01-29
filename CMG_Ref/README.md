# CMG Reference Motion Generator

This folder contains the Conditional Motion Generator (CMG) from [cmg_workspace](https://github.com/PMY9527/cmg_workspace).

## Overview

The CMG is a neural network model that generates reference motions based on velocity commands (linear velocity vx, vy and angular velocity yaw). Instead of using motion capture data, this model can generate smooth, continuous reference motions for humanoid robot locomotion.

## Structure

- `module/` - Core CMG model components
  - `cmg.py` - Main CMG model with Mixture-of-Experts architecture
  - `gating_network.py` - Gating network for expert selection
  - `moe_layer.py` - Mixture-of-Experts layer implementation
- `dataloader/` - Data loading and processing utilities
  - `dataloader.py` - Dataset loader
  - `dist_plot.py` - Distribution visualization
  - `cmg_training_data.pt` - Training data (if present)
- `train.py` - Training script for the CMG model
- `eval_cmg.py` - Evaluation script that generates reference motions
- `cmg_trainer.py` - Trainer class for CMG training
- `mujoco_player.py` - MuJoCo visualization player for generated motions

## Usage

### Training the CMG Model

```bash
cd CMG_Ref
python train.py
```

This will train the CMG model using the provided training data and save checkpoints to `runs/cmg_YYYYMMDD_HHMMSS/`.

### Generating Reference Motions

Edit the velocity commands in `eval_cmg.py`:

```python
VX = 3.0    # Forward velocity (m/s)
VY = 0.0    # Lateral velocity (m/s)
YAW = 0.0   # Angular velocity (rad/s)
DURATION = 1000  # Number of frames at 50 fps
```

Then run:

```bash
python eval_cmg.py
```

This generates an `autoregressive_motion.npz` file containing the reference motion.

### Visualizing Generated Motions

```bash
python mujoco_player.py autoregressive_motion.npz --no-loop
```

## Model Architecture

The CMG uses a Mixture-of-Experts (MoE) architecture with:
- **Input**: Current motion state (joint positions + velocities, 58 dims) + velocity command (3 dims)
- **Architecture**: 3-layer MLP with 512 hidden units and 4 experts
- **Output**: Next motion state (joint positions + velocities, 58 dims)
- **Generation**: Autoregressive generation for smooth motion sequences

## Integration with TWIST

This CMG model is designed to replace motion capture data in TWIST, enabling:
1. Generation of reference motions from velocity commands instead of using pre-recorded motion capture
2. Robust walking and running behaviors for humanoid robots
3. More flexible control through velocity commands rather than fixed motion sequences

## Credits

Original CMG workspace: https://github.com/PMY9527/cmg_workspace
