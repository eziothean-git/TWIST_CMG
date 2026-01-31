#!/usr/bin/env python3
"""
可视化 CMG 模型输出的关节序列
不需要 MuJoCo，使用 matplotlib 绑定关节角度变化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.cmg import CMG

# 关节名称（29 DOF）
JOINT_NAMES = [
    'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
    'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'L_shoulder_pitch', 'L_shoulder_roll', 'L_shoulder_yaw', 'L_elbow',
    'L_wrist_roll', 'L_wrist_pitch', 'L_wrist_yaw',
    'R_shoulder_pitch', 'R_shoulder_roll', 'R_shoulder_yaw', 'R_elbow',
    'R_wrist_roll', 'R_wrist_pitch', 'R_wrist_yaw',
]

def load_model_and_stats(model_path, data_path, device='cuda'):
    """加载 CMG 模型和统计信息"""
    data = torch.load(data_path, weights_only=False, map_location='cpu')
    stats = data["stats"]
    samples = data["samples"]
    
    model = CMG(
        motion_dim=stats["motion_dim"],
        command_dim=stats["command_dim"],
        hidden_dim=512,
        num_experts=4,
        num_layers=3,
    )
    
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, stats, samples


def generate_motion(model, init_motion, commands, stats, device='cuda'):
    """自回归生成动作序列"""
    motion_mean = torch.from_numpy(stats["motion_mean"]).to(device)
    motion_std = torch.from_numpy(stats["motion_std"]).to(device)
    cmd_min = torch.from_numpy(stats["command_min"]).to(device)
    cmd_max = torch.from_numpy(stats["command_max"]).to(device)
    
    # 归一化
    current = (torch.from_numpy(init_motion).to(device) - motion_mean) / motion_std
    commands_t = torch.from_numpy(commands).to(device)
    commands_norm = (commands_t - cmd_min) / (cmd_max - cmd_min) * 2 - 1
    
    generated = [current.cpu().numpy()]
    
    with torch.no_grad():
        for t in range(len(commands)):
            pred = model(current.unsqueeze(0), commands_norm[t:t+1])
            current = pred.squeeze(0)
            generated.append(current.cpu().numpy())
    
    # 反归一化
    generated = np.stack(generated)
    generated = generated * stats["motion_std"] + stats["motion_mean"]
    
    return generated


def plot_joint_trajectories(motion, title="CMG Generated Motion", save_path=None):
    """
    绘制关节角度轨迹
    
    Args:
        motion: [T, 58] 动作序列（前29维是dof_pos，后29维是dof_vel）
    """
    T = motion.shape[0]
    dof_pos = motion[:, :29]
    dof_vel = motion[:, 29:58]
    
    time = np.arange(T) / 50.0  # 50 fps
    
    # 创建图表
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=14)
    
    # 腿部关节 (0-11)
    ax = axes[0, 0]
    for i in range(6):
        ax.plot(time, dof_pos[:, i], label=JOINT_NAMES[i])
    ax.set_title("Left Leg DOF Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i in range(6, 12):
        ax.plot(time, dof_pos[:, i], label=JOINT_NAMES[i])
    ax.set_title("Right Leg DOF Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 腰部关节 (12-14)
    ax = axes[1, 0]
    for i in range(12, 15):
        ax.plot(time, dof_pos[:, i], label=JOINT_NAMES[i])
    ax.set_title("Waist DOF Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 左臂关节 (15-21)
    ax = axes[1, 1]
    for i in range(15, 22):
        ax.plot(time, dof_pos[:, i], label=JOINT_NAMES[i])
    ax.set_title("Left Arm DOF Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 右臂关节 (22-28)
    ax = axes[2, 0]
    for i in range(22, 29):
        ax.plot(time, dof_pos[:, i], label=JOINT_NAMES[i])
    ax.set_title("Right Arm DOF Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 腿部关节速度
    ax = axes[2, 1]
    for i in [0, 3, 6, 9]:  # hip_pitch 和 knee
        ax.plot(time, dof_vel[:, i], label=f"{JOINT_NAMES[i]}_vel")
    ax.set_title("Key Leg DOF Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 统计信息
    ax = axes[3, 0]
    ax.axis('off')
    stats_text = f"""
    Motion Statistics:
    ─────────────────────────────────
    Duration: {T/50:.2f} s ({T} frames @ 50fps)
    
    DOF Position Range:
      Min: {dof_pos.min():.3f} rad
      Max: {dof_pos.max():.3f} rad
      Mean: {dof_pos.mean():.3f} rad
      Std: {dof_pos.std():.3f} rad
    
    DOF Velocity Range:
      Min: {dof_vel.min():.3f} rad/s
      Max: {dof_vel.max():.3f} rad/s
      Mean: {dof_vel.mean():.3f} rad/s
      Std: {dof_vel.std():.3f} rad/s
    
    Key Joint Initial Values:
      L_hip_pitch: {dof_pos[0, 0]:.3f} rad
      L_knee: {dof_pos[0, 3]:.3f} rad
      R_hip_pitch: {dof_pos[0, 6]:.3f} rad
      R_knee: {dof_pos[0, 9]:.3f} rad
    """
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # 相位分析 - 左右腿对比
    ax = axes[3, 1]
    ax.plot(time, dof_pos[:, 0], label='L_hip_pitch', color='blue')
    ax.plot(time, dof_pos[:, 6], label='R_hip_pitch', color='red')
    ax.plot(time, dof_pos[:, 3], label='L_knee', color='blue', linestyle='--')
    ax.plot(time, dof_pos[:, 9], label='R_knee', color='red', linestyle='--')
    ax.set_title("Left vs Right Leg Phase")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def compare_with_ground_truth(model, stats, samples, sample_idx=0, device='cuda'):
    """对比 CMG 生成的轨迹与 ground truth"""
    sample = samples[sample_idx]
    gt_motion = sample["motion"]  # [T+1, 58]
    gt_command = sample["command"]  # [T, 3]
    
    # 使用 ground truth 的初始帧和命令生成轨迹
    init_motion = gt_motion[0]
    generated = generate_motion(model, init_motion, gt_command, stats, device)
    
    T = min(len(generated), len(gt_motion))
    
    # 绘制对比图
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"CMG Output vs Ground Truth (Sample {sample_idx})", fontsize=14)
    
    time = np.arange(T) / 50.0
    
    # 左髋和左膝对比
    ax = axes[0, 0]
    ax.plot(time, gt_motion[:T, 0], 'b-', label='GT L_hip_pitch', alpha=0.7)
    ax.plot(time, generated[:T, 0], 'b--', label='CMG L_hip_pitch')
    ax.plot(time, gt_motion[:T, 3], 'r-', label='GT L_knee', alpha=0.7)
    ax.plot(time, generated[:T, 3], 'r--', label='CMG L_knee')
    ax.set_title("Left Leg: GT vs CMG")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 右髋和右膝对比
    ax = axes[0, 1]
    ax.plot(time, gt_motion[:T, 6], 'b-', label='GT R_hip_pitch', alpha=0.7)
    ax.plot(time, generated[:T, 6], 'b--', label='CMG R_hip_pitch')
    ax.plot(time, gt_motion[:T, 9], 'r-', label='GT R_knee', alpha=0.7)
    ax.plot(time, generated[:T, 9], 'r--', label='CMG R_knee')
    ax.set_title("Right Leg: GT vs CMG")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 误差分析
    error = np.abs(generated[:T] - gt_motion[:T])
    dof_pos_error = error[:, :29]
    
    ax = axes[1, 0]
    ax.plot(time, dof_pos_error.mean(axis=1), 'k-', label='Mean Error')
    ax.fill_between(time, dof_pos_error.min(axis=1), dof_pos_error.max(axis=1), alpha=0.3)
    ax.set_title("DOF Position Error Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 每个关节的累积误差
    ax = axes[1, 1]
    joint_errors = dof_pos_error.mean(axis=0)
    colors = ['blue']*6 + ['red']*6 + ['green']*3 + ['orange']*7 + ['purple']*7
    ax.bar(range(29), joint_errors, color=colors, alpha=0.7)
    ax.set_title("Mean Error per Joint")
    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Mean Error (rad)")
    ax.set_xticks(range(0, 29, 2))
    ax.grid(True, alpha=0.3)
    
    # Velocity command
    ax = axes[2, 0]
    ax.plot(time[:-1], gt_command[:T-1, 0], label='vx (m/s)')
    ax.plot(time[:-1], gt_command[:T-1, 1], label='vy (m/s)')
    ax.plot(time[:-1], gt_command[:T-1, 2], label='yaw (rad/s)')
    ax.set_title("Velocity Command")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 统计摘要
    ax = axes[2, 1]
    ax.axis('off')
    summary = f"""
    Comparison Summary:
    ───────────────────────────────────
    Sample Index: {sample_idx}
    Duration: {T/50:.2f} s
    
    DOF Position Error:
      Mean: {dof_pos_error.mean():.4f} rad ({np.degrees(dof_pos_error.mean()):.2f}°)
      Max:  {dof_pos_error.max():.4f} rad ({np.degrees(dof_pos_error.max()):.2f}°)
      
    Worst Joints (by mean error):
      {JOINT_NAMES[np.argmax(joint_errors)]}: {joint_errors.max():.4f} rad
      
    Velocity Command Range:
      vx: [{gt_command[:, 0].min():.2f}, {gt_command[:, 0].max():.2f}] m/s
      vy: [{gt_command[:, 1].min():.2f}, {gt_command[:, 1].max():.2f}] m/s
      yaw: [{gt_command[:, 2].min():.2f}, {gt_command[:, 2].max():.2f}] rad/s
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"cmg_comparison_sample{sample_idx}.png", dpi=150, bbox_inches='tight')
    print(f"Saved comparison to cmg_comparison_sample{sample_idx}.png")
    plt.show()
    
    return error


def test_custom_command(model, stats, samples, vx=1.5, vy=0.0, yaw=0.0, duration=200, device='cuda'):
    """使用自定义速度命令测试 CMG"""
    # 随机选择一个初始姿态
    sample_idx = np.random.randint(0, len(samples))
    init_motion = samples[sample_idx]["motion"][0]
    
    # 创建恒定速度命令
    commands = np.tile([vx, vy, yaw], (duration, 1)).astype(np.float32)
    
    # 生成轨迹
    generated = generate_motion(model, init_motion, commands, stats, device)
    
    title = f"CMG Output: vx={vx:.1f}m/s, vy={vy:.1f}m/s, yaw={yaw:.2f}rad/s"
    plot_joint_trajectories(generated, title=title, save_path=f"cmg_custom_vx{vx}_vy{vy}_yaw{yaw}.png")
    
    return generated


if __name__ == "__main__":
    MODEL_PATH = 'runs/cmg_20260123_194851/cmg_final.pt'
    DATA_PATH = 'dataloader/cmg_training_data.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model, stats, samples = load_model_and_stats(MODEL_PATH, DATA_PATH, device)
    print(f"Loaded {len(samples)} samples")
    
    # 1. 对比 ground truth
    print("\n=== Comparing with Ground Truth ===")
    compare_with_ground_truth(model, stats, samples, sample_idx=0, device=device)
    
    # 2. 测试不同速度命令
    print("\n=== Testing Custom Commands ===")
    
    # 向前行走 1.5 m/s
    test_custom_command(model, stats, samples, vx=1.5, vy=0.0, yaw=0.0, duration=200, device=device)
    
    # 向前行走 + 转弯
    test_custom_command(model, stats, samples, vx=1.0, vy=0.0, yaw=0.3, duration=200, device=device)
    
    # 快速行走
    test_custom_command(model, stats, samples, vx=2.5, vy=0.0, yaw=0.0, duration=200, device=device)
    
    print("\n=== Done! Check generated PNG files ===")
