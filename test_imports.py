#!/usr/bin/env python3
"""快速测试导入"""
import sys
from pathlib import Path

# 测试路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("项目根目录:", project_root)
print("")

# 测试1: 导入MotionLibCMG
print("测试1: 导入MotionLibCMG...")
try:
    from pose.utils.motion_lib_cmg import MotionLibCMG, create_motion_lib_cmg
    print("✓ 成功导入 MotionLibCMG")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 验证CMG文件
print("\n测试2: 验证CMG文件...")
cmg_files = [
    "CMG_Ref/utils/cmg_motion_generator.py",
    "CMG_Ref/utils/command_sampler.py",
]
for f in cmg_files:
    path = project_root / f
    if path.exists():
        print(f"✓ {f}")
    else:
        print(f"✗ {f} (缺失)")

# 测试3: 直接导入CMG模块
print("\n测试3: 直接导入CMG模块...")
try:
    import importlib.util
    cmg_module_path = str(project_root / "CMG_Ref" / "utils" / "cmg_motion_generator.py")
    spec = importlib.util.spec_from_file_location("cmg_motion_generator", cmg_module_path)
    cmg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmg_module)
    print("✓ 成功加载 cmg_motion_generator")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 所有测试通过！")
