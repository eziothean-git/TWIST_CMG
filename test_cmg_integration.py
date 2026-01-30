#!/usr/bin/env python3
"""
测试CMG集成到TWIST环境

验证：
1. MotionLibCMG能否正确初始化
2. 能否生成参考动作
3. 与环境配置是否兼容
"""

import sys
import torch
from pathlib import Path

# 添加路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "legged_gym"))
sys.path.insert(0, str(SCRIPT_DIR / "CMG_Ref"))

from pose.pose.utils.motion_lib_cmg import MotionLibCMG, create_motion_lib_cmg
from legged_gym.legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg

def test_motion_lib_cmg():
    """测试MotionLibCMG基本功能"""
    print("=" * 60)
    print("测试 1: MotionLibCMG 初始化")
    print("=" * 60)
    
    # 配置
    cmg_model_path = "CMG_Ref/runs/cmg_20260123_194851/cmg_final.pt"
    cmg_data_path = "CMG_Ref/dataloader/cmg_training_data.pt"
    num_envs = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"设备: {device}")
    print(f"环境数: {num_envs}")
    
    try:
        motion_lib = MotionLibCMG(
            cmg_model_path=cmg_model_path,
            cmg_data_path=cmg_data_path,
            num_envs=num_envs,
            device=device,
            mode='pregenerated',
            preload_duration=100
        )
        print("✓ MotionLibCMG 初始化成功\n")
        
        # 测试采样
        print("=" * 60)
        print("测试 2: 采样速度命令并生成参考动作")
        print("=" * 60)
        
        motion_ids = motion_lib.sample_motions(num_envs)
        print(f"✓ 采样了 {len(motion_ids)} 个动作")
        
        # 测试获取动作状态
        ref_dof_pos, ref_dof_vel = motion_lib.get_motion_state()
        print(f"✓ 获取参考状态:")
        print(f"  - DOF位置: {ref_dof_pos.shape}")
        print(f"  - DOF速度: {ref_dof_vel.shape}")
        
        # 测试多帧生成
        print("\n测试连续帧生成...")
        for i in range(5):
            ref_pos, ref_vel = motion_lib.get_motion_state()
            print(f"  帧 {i+1}: pos范围=[{ref_pos.min():.3f}, {ref_pos.max():.3f}]")
        
        print("\n✓ 所有基本测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """测试配置集成"""
    print("\n" + "=" * 60)
    print("测试 3: 配置文件集成")
    print("=" * 60)
    
    try:
        cfg = G1MimicPrivCfg()
        
        # 检查CMG配置项
        required_attrs = [
            'cmg_model_path',
            'cmg_data_path',
            'cmg_mode',
            'cmg_preload_duration',
            'cmg_buffer_size'
        ]
        
        for attr in required_attrs:
            if hasattr(cfg.motion, attr):
                value = getattr(cfg.motion, attr)
                print(f"✓ {attr}: {value}")
            else:
                print(f"✗ 缺少配置: {attr}")
                return False
        
        print("\n✓ 配置文件完整")
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_factory():
    """测试工厂函数"""
    print("\n" + "=" * 60)
    print("测试 4: 工厂函数创建")
    print("=" * 60)
    
    try:
        cfg = G1MimicPrivCfg()
        num_envs = 16
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        motion_lib = create_motion_lib_cmg(cfg, num_envs, device)
        print(f"✓ 通过工厂函数创建成功")
        print(f"  - 模式: {motion_lib.mode}")
        print(f"  - 环境数: {motion_lib.num_envs}")
        print(f"  - 设备: {motion_lib.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ 工厂函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "CMG-TWIST 集成测试" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝\n")
    
    results = []
    
    # 运行测试
    results.append(("基本功能", test_motion_lib_cmg()))
    results.append(("配置集成", test_config_integration()))
    results.append(("工厂函数", test_create_factory()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ 所有测试通过！可以开始训练")
        print("\n下一步:")
        print("  bash train_teacher.sh test_cmg cuda:0")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
