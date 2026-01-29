"""
Example script showing how to use the CMG (Conditional Motion Generator) 
to generate reference motions from velocity commands.

This script demonstrates:
1. Loading a trained CMG model
2. Generating reference motions from velocity commands
3. Saving the generated motion as a .npz file for use with TWIST

Requirements:
- A trained CMG model checkpoint (e.g., from runs/cmg_*/cmg_final.pt)
- Training data with statistics (e.g., dataloader/cmg_training_data.pt)
"""

import sys
import os

# Add CMG_Ref to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import numpy as np
    from module.cmg import CMG
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install:")
    print(f"  pip install torch numpy")
    print(f"\nOriginal error: {e}")
    sys.exit(1)


def load_cmg_model(model_path, data_path, device='cuda'):
    """
    Load a trained CMG model and its training statistics.
    
    Args:
        model_path: Path to the trained model checkpoint (.pt file)
        data_path: Path to the training data with statistics (.pt file)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded CMG model
        stats: Training statistics dict
        samples: Sample data (if available)
    """
    # Load training data to get statistics
    data = torch.load(data_path, weights_only=False)
    stats = data["stats"]
    samples = data.get("samples", None)
    
    # Create model with same architecture as training
    model = CMG(
        motion_dim=stats["motion_dim"],
        command_dim=stats["command_dim"],
        hidden_dim=512,
        num_experts=4,
        num_layers=3,
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, stats, samples


def generate_reference_motion(model, init_motion, commands, stats, device='cuda'):
    """
    Generate reference motion sequence using the CMG model.
    
    Args:
        model: Trained CMG model
        init_motion: Initial motion state [58] (joint positions + velocities, unnormalized)
        commands: Command sequence [T, 3] containing (vx, vy, yaw_rate) in m/s and rad/s
        stats: Statistics dict from training data
        device: Device to run inference on
    
    Returns:
        generated: Generated motion sequence [T+1, 58] (unnormalized)
    """
    # Get normalization statistics
    motion_mean = torch.from_numpy(stats["motion_mean"]).to(device)
    motion_std = torch.from_numpy(stats["motion_std"]).to(device)
    cmd_min = torch.from_numpy(stats["command_min"]).to(device)
    cmd_max = torch.from_numpy(stats["command_max"]).to(device)
    
    # Normalize initial motion
    current = (torch.from_numpy(init_motion).to(device) - motion_mean) / motion_std
    
    # Normalize commands: [min, max] → [-1, 1]
    commands = torch.from_numpy(commands).to(device)
    commands_norm = (commands - cmd_min) / (cmd_max - cmd_min) * 2 - 1
    
    # Generate motion sequence autoregressively
    generated = [current.clone()]
    
    with torch.no_grad():
        for t in range(len(commands_norm)):
            cmd = commands_norm[t:t+1]
            curr = current.unsqueeze(0)
            
            # Predict next motion state
            pred = model(curr, cmd)
            current = pred.squeeze(0)
            generated.append(current.clone())
    
    # Stack and denormalize
    generated = torch.stack(generated)
    generated = generated * motion_std + motion_mean
    
    return generated.cpu().numpy()


def save_motion_as_npz(motion, output_path, fps=50):
    """
    Save generated motion in .npz format compatible with TWIST.
    
    Args:
        motion: Motion sequence [T, 58] (joint positions + velocities)
        output_path: Path to save the .npz file
        fps: Frames per second (default: 50)
    """
    T = motion.shape[0]
    
    # Split motion into positions and velocities
    dof_positions = motion[:, :29].astype(np.float32)
    dof_velocities = motion[:, 29:].astype(np.float32)
    
    # Create placeholder body positions and rotations
    # These will be properly computed by TWIST when using the reference motion
    body_positions = np.zeros((T, 30, 3), dtype=np.float32)
    body_positions[:, 0, 2] = 0.75  # Root height
    
    body_rotations = np.zeros((T, 30, 4), dtype=np.float32)
    body_rotations[:, :, 0] = 1.0  # Identity quaternion
    
    # Save in TWIST-compatible format
    np.savez(
        output_path,
        fps=np.array([fps], dtype=np.float32),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        dof_names=np.array([f"joint_{i}" for i in range(29)]),
        body_names=np.array([f"body_{i}" for i in range(30)]),
    )
    print(f"✓ Saved motion to {output_path}")


def example_generate_walking_motion():
    """
    Example: Generate a walking motion reference.
    """
    print("="*60)
    print("Example: Generating Reference Motion with CMG")
    print("="*60)
    
    # Configuration
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    OUTPUT_PATH = 'generated_walk.npz'
    
    # Velocity commands
    VX = 1.5      # Forward velocity (m/s)
    VY = 0.0      # Lateral velocity (m/s)
    YAW = 0.0     # Angular velocity (rad/s)
    DURATION = 500  # Number of frames
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}")
        print(f"  Please train a model first using: python train.py")
        return
    
    if not os.path.exists(DATA_PATH):
        print(f"✗ Training data not found: {DATA_PATH}")
        print(f"  Please ensure training data is available")
        return
    
    print(f"\n1. Loading CMG model...")
    model, stats, samples = load_cmg_model(MODEL_PATH, DATA_PATH, device)
    print(f"   ✓ Model loaded")
    print(f"   ✓ Motion dim: {stats['motion_dim']}")
    print(f"   ✓ Command dim: {stats['command_dim']}")
    
    print(f"\n2. Setting up initial state...")
    # Use first frame from training data as initial state
    if samples and len(samples) > 0:
        init_motion = samples[0]["motion"][0]
        print(f"   ✓ Using initial state from training data")
    else:
        # Fallback: zeros (standing position approximation)
        init_motion = np.zeros(58, dtype=np.float32)
        print(f"   ✓ Using zero initial state (standing)")
    
    print(f"\n3. Generating motion with commands:")
    print(f"   • Forward velocity: {VX} m/s")
    print(f"   • Lateral velocity: {VY} m/s")
    print(f"   • Angular velocity: {YAW} rad/s")
    print(f"   • Duration: {DURATION} frames ({DURATION/50:.2f}s @ 50fps)")
    
    # Create constant velocity commands
    commands = np.tile([VX, VY, YAW], (DURATION, 1)).astype(np.float32)
    
    # Generate motion
    generated_motion = generate_reference_motion(model, init_motion, commands, stats, device)
    print(f"   ✓ Generated {generated_motion.shape[0]} frames")
    
    print(f"\n4. Saving motion...")
    save_motion_as_npz(generated_motion, OUTPUT_PATH, fps=50)
    
    print(f"\n5. Done! Generated motion can be used with TWIST:")
    print(f"   • Motion file: {OUTPUT_PATH}")
    print(f"   • Use with TWIST teacher policy for training")
    print(f"   • Or use with high-level motion sender for deployment")
    print(f"\n" + "="*60)


def example_generate_custom_trajectory():
    """
    Example: Generate motion with varying velocity commands.
    """
    print("\nExample: Custom Trajectory Generation")
    print("-" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # This is just a template - adjust paths as needed
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Model or data not found. This is just an example template.")
        return
    
    model, stats, samples = load_cmg_model(MODEL_PATH, DATA_PATH, device)
    
    # Create a trajectory with varying commands
    # E.g., accelerate, maintain speed, turn, decelerate
    commands = []
    
    # Accelerate (0-2s)
    for t in range(100):
        vx = 0.5 + (t / 100) * 2.0  # 0.5 -> 2.5 m/s
        commands.append([vx, 0.0, 0.0])
    
    # Turn left while moving (2-4s)
    for t in range(100):
        commands.append([2.5, 0.0, 0.5])  # Turn with 0.5 rad/s
    
    # Straight run (4-6s)
    for t in range(100):
        commands.append([3.0, 0.0, 0.0])
    
    # Decelerate (6-8s)
    for t in range(100):
        vx = 3.0 - (t / 100) * 2.0  # 3.0 -> 1.0 m/s
        commands.append([vx, 0.0, 0.0])
    
    commands = np.array(commands, dtype=np.float32)
    
    # Generate
    init_motion = samples[0]["motion"][0] if samples else np.zeros(58, dtype=np.float32)
    generated_motion = generate_reference_motion(model, init_motion, commands, stats, device)
    
    # Save
    save_motion_as_npz(generated_motion, 'custom_trajectory.npz', fps=50)
    print(f"✓ Custom trajectory saved to custom_trajectory.npz")


if __name__ == "__main__":
    # Check if we're in the CMG_Ref directory
    if not os.path.exists('module/cmg.py'):
        print("Error: Please run this script from the CMG_Ref directory")
        print("  cd CMG_Ref")
        print("  python example_usage.py")
        sys.exit(1)
    
    try:
        # Run the main example
        example_generate_walking_motion()
        
        # Uncomment to run the custom trajectory example
        # example_generate_custom_trajectory()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
